"""Pure controls for mixed-source Stage-1 robustness experiments."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np


SOURCES = ("historical", "canonical")


def _stable_key(seed: int, *parts: object) -> str:
    payload = ":".join((str(seed), *(str(part) for part in parts)))
    return sha256(payload.encode("utf-8")).hexdigest()


def four_cell_epoch_indices(
    rows: Sequence[Mapping[str, Any]], *, records: int, seed: int, epoch: int
) -> np.ndarray:
    """Draw exactly equally from Historical/Canonical x C/W cells."""

    if not rows or records < 4 or records % 4 or epoch < 0:
        raise ValueError("four-cell sampling requires positive multiple-of-four records")
    cells: dict[tuple[str, bool], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        source = str(row["source_regime"])
        if source not in SOURCES:
            raise ValueError(f"unsupported source regime: {source}")
        cells[(source, bool(row["current_dense_wrong"]))].append(index)
    expected = {(source, label) for source in SOURCES for label in (False, True)}
    if set(cells) != expected or any(not cells[cell] for cell in expected):
        raise ValueError("all four source/correctness cells must be nonempty")

    rng = np.random.default_rng(int(seed) + 1_000_003 * int(epoch))
    per_cell = records // 4
    selected = []
    for cell in sorted(expected):
        values = np.asarray(cells[cell], dtype=np.int64)
        if per_cell <= len(values):
            chosen = rng.permutation(values)[:per_cell]
        else:
            chosen = rng.choice(values, size=per_cell, replace=True)
        selected.append(chosen)
    return rng.permutation(np.concatenate(selected)).astype(np.int64)


def source_stratified_validation_uids(
    rows: Sequence[Mapping[str, Any]],
    *,
    fraction: float,
    seed: int,
    run_id: str,
) -> set[str]:
    """Select group-disjoint validation records within source/dataset/label cells."""

    if not rows or not 0.0 < fraction < 0.5 or not run_id:
        raise ValueError("invalid source-stratified validation request")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["image_group_id"])].append(row)
    strata: dict[
        tuple[str, str, tuple[bool, ...]],
        list[tuple[str, list[Mapping[str, Any]]]],
    ] = defaultdict(list)
    for group_id, members in groups.items():
        source_datasets = {
            (str(row["source_regime"]), str(row["dataset"])) for row in members
        }
        if len(source_datasets) != 1:
            raise ValueError(f"image group crosses source or dataset: {group_id}")
        source, dataset = next(iter(source_datasets))
        label_signature = tuple(
            sorted({bool(row["current_dense_wrong"]) for row in members})
        )
        strata[(source, dataset, label_signature)].append((group_id, members))

    chosen_groups: set[str] = set()
    for stratum, values in sorted(strata.items()):
        ordered = sorted(
            values,
            key=lambda item: _stable_key(seed, run_id, *stratum, item[0]),
        )
        target = max(1, int(round(sum(len(members) for _, members in values) * fraction)))
        count = 0
        for group_id, members in ordered:
            if count >= target:
                break
            chosen_groups.add(group_id)
            count += len(members)
    selected = {
        str(row["uid"])
        for row in rows
        if str(row["image_group_id"]) in chosen_groups
    }
    if not selected or len(selected) == len(rows):
        raise RuntimeError("source-stratified validation selection is degenerate")
    return selected


def validate_lodo_training_rows(
    rows: Sequence[Mapping[str, Any]], *, target_dataset: str
) -> None:
    if not rows:
        raise ValueError("LODO training rows are empty")
    if any(str(row["dataset"]) == target_dataset for row in rows):
        raise ValueError("target dataset appears in LODO training")
    cells = {
        (str(row["source_regime"]), bool(row["current_dense_wrong"])) for row in rows
    }
    expected = {(source, label) for source in SOURCES for label in (False, True)}
    if cells != expected:
        raise ValueError("LODO training lacks a source/correctness cell")


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = len(labels) - positive_count
    statistic = ranks[positives].sum() - positive_count * (positive_count + 1) / 2
    return float(statistic / (positive_count * negative_count))


def bootstrap_auc_interval(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    """UID bootstrap interval for AUROC."""

    y = np.asarray(labels, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if (
        y.ndim != 1
        or values.shape != y.shape
        or not np.isin(y, (0, 1)).all()
        or set(np.unique(y)) != {0, 1}
        or not np.isfinite(values).all()
        or draws < 1
    ):
        raise ValueError("invalid AUROC bootstrap inputs")
    observed = _auroc(y, values)
    rng = np.random.default_rng(seed)
    sampled = []
    while len(sampled) < draws:
        indices = rng.integers(0, len(y), size=len(y))
        sampled_y = y[indices]
        if len(np.unique(sampled_y)) != 2:
            continue
        sampled.append(_auroc(sampled_y, values[indices]))
    results = np.asarray(sampled, dtype=np.float64)
    return {
        "auroc": observed,
        "ci_low": float(np.quantile(results, 0.025)),
        "ci_high": float(np.quantile(results, 0.975)),
        "draws": int(draws),
        "seed": int(seed),
    }


def choose_robustness_decision(
    *,
    historical_auroc: float,
    canonical_auroc: float,
    lodo_worst_source_aurocs: Sequence[float],
    historical_lodo_mean: float,
    phase49_historical_mean: float,
) -> dict[str, Any]:
    """Apply the frozen aggregate source/OOD robustness rule."""

    metrics = [
        float(historical_auroc),
        float(canonical_auroc),
        *(float(value) for value in lodo_worst_source_aurocs),
        float(historical_lodo_mean),
        float(phase49_historical_mean),
    ]
    if not all(np.isfinite(metrics)) or len(lodo_worst_source_aurocs) != 3:
        raise ValueError("robustness decision inputs are invalid")
    source_robust = min(historical_auroc, canonical_auroc) >= 0.70
    ood_broad = (
        min(lodo_worst_source_aurocs) >= 0.60
        and historical_lodo_mean >= phase49_historical_mean
    )
    if source_robust and ood_broad:
        decision = "A — Robust shared Stage-1 candidate found"
    elif source_robust:
        decision = "B — Source-robust but benchmark-OOD limited"
    else:
        decision = "C — Shared-head conflict remains"
    return {
        "decision": decision,
        "source_robust": bool(source_robust),
        "broad_ood_transfer": bool(ood_broad),
        "source_robust_minimum_auroc": 0.70,
        "ood_each_target_worst_source_minimum_auroc": 0.60,
        "historical_lodo_noninferiority_reference_mean": float(
            phase49_historical_mean
        ),
    }
