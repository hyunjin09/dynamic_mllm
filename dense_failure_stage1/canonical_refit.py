"""Pure helpers for the canonical Stage-1 refit diagnostic."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np

def _stable_key(seed: int, *parts: object) -> str:
    value = ":".join((str(seed), *(str(part) for part in parts)))
    return sha256(value.encode("utf-8")).hexdigest()


def assign_group_folds(
    rows: Sequence[Mapping[str, Any]], *, folds: int, seed: int
) -> list[dict[str, Any]]:
    """Assign homogeneous image groups to balanced dataset/label folds."""

    if folds < 2 or not rows:
        raise ValueError("group folding requires records and at least two folds")
    uids = [str(row["uid"]) for row in rows]
    if len(set(uids)) != len(uids):
        raise ValueError("fold input contains duplicate UIDs")

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["image_group_id"])].append(row)

    strata: dict[tuple[str, bool], list[tuple[str, list[Mapping[str, Any]]]]] = defaultdict(list)
    for group_id, members in groups.items():
        signatures = {
            (str(member["dataset"]), bool(member["current_dense_wrong"]))
            for member in members
        }
        if len(signatures) != 1:
            raise ValueError(f"image group crosses dataset/label strata: {group_id}")
        strata[next(iter(signatures))].append((group_id, members))

    assignment: dict[str, int] = {}
    global_counts = [0] * folds
    for stratum in sorted(strata):
        stratum_counts = [0] * folds
        ordered = sorted(
            strata[stratum],
            key=lambda item: (
                -len(item[1]),
                _stable_key(seed, stratum[0], int(stratum[1]), item[0]),
            ),
        )
        for group_id, members in ordered:
            fold = min(
                range(folds),
                key=lambda value: (stratum_counts[value], global_counts[value], value),
            )
            assignment[group_id] = fold
            stratum_counts[fold] += len(members)
            global_counts[fold] += len(members)

    output = []
    for row in sorted(rows, key=lambda item: str(item["uid"])):
        value = dict(row)
        value["fold"] = assignment[str(row["image_group_id"])]
        output.append(value)
    return output


def inner_validation_uids(
    outer_training_rows: Sequence[Mapping[str, Any]],
    *,
    fraction: float,
    seed: int,
    outer_fold: int,
) -> set[str]:
    """Select a deterministic stratified group-disjoint inner validation set."""

    if not 0.0 < fraction < 0.5 or not outer_training_rows:
        raise ValueError("invalid inner-validation request")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in outer_training_rows:
        grouped[str(row["image_group_id"])].append(row)

    strata: dict[tuple[str, bool], list[tuple[str, list[Mapping[str, Any]]]]] = defaultdict(list)
    for group_id, members in grouped.items():
        signatures = {
            (str(row["dataset"]), bool(row["current_dense_wrong"])) for row in members
        }
        if len(signatures) != 1:
            raise ValueError(f"inner-validation group is heterogeneous: {group_id}")
        strata[next(iter(signatures))].append((group_id, members))

    selected_groups: set[str] = set()
    for stratum, values in sorted(strata.items()):
        ordered = sorted(
            values,
            key=lambda item: _stable_key(
                seed, "inner", outer_fold, stratum[0], int(stratum[1]), item[0]
            ),
        )
        target_records = max(1, int(round(sum(len(item[1]) for item in values) * fraction)))
        records = 0
        for group_id, members in ordered:
            if records >= target_records:
                break
            selected_groups.add(group_id)
            records += len(members)

    selected = {
        str(row["uid"])
        for row in outer_training_rows
        if str(row["image_group_id"]) in selected_groups
    }
    if not selected or len(selected) == len(outer_training_rows):
        raise RuntimeError("inner-validation selection is degenerate")
    return selected


def balanced_epoch_indices(
    labels: Sequence[int] | np.ndarray, *, seed: int, epoch: int
) -> np.ndarray:
    """Return a no-replacement, class-balanced epoch sample."""

    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1 or y.size == 0 or set(np.unique(y)) != {0, 1}:
        raise ValueError("balanced sampling requires both binary classes")
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    target = min(len(negative), len(positive))
    rng = np.random.default_rng(int(seed) + 1_000_003 * int(epoch))
    chosen_negative = rng.permutation(negative)[:target]
    chosen_positive = rng.permutation(positive)[:target]
    return rng.permutation(np.concatenate((chosen_negative, chosen_positive))).astype(
        np.int64
    )


def paired_bootstrap_auc_difference(
    labels: Sequence[int] | np.ndarray,
    old_scores: Sequence[float] | np.ndarray,
    refit_scores: Sequence[float] | np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    """Paired UID bootstrap interval for refit-minus-old AUROC."""

    y = np.asarray(labels, dtype=np.int64)
    old = np.asarray(old_scores, dtype=np.float64)
    refit = np.asarray(refit_scores, dtype=np.float64)
    if y.ndim != 1 or old.shape != y.shape or refit.shape != y.shape:
        raise ValueError("paired bootstrap inputs must be aligned vectors")
    if draws < 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError("paired bootstrap requires draws and both classes")
    if not np.isfinite(old).all() or not np.isfinite(refit).all():
        raise ValueError("paired bootstrap scores must be finite")

    def auroc(values_y: np.ndarray, values_score: np.ndarray) -> float:
        order = np.argsort(values_score, kind="mergesort")
        ranks = np.empty(len(values_score), dtype=np.float64)
        start = 0
        while start < len(order):
            stop = start + 1
            while (
                stop < len(order)
                and values_score[order[stop]] == values_score[order[start]]
            ):
                stop += 1
            ranks[order[start:stop]] = (start + 1 + stop) / 2.0
            start = stop
        positives = values_y == 1
        positive_count = int(positives.sum())
        negative_count = len(values_y) - positive_count
        statistic = ranks[positives].sum() - positive_count * (positive_count + 1) / 2
        return float(statistic / (positive_count * negative_count))

    observed = auroc(y, refit) - auroc(y, old)
    rng = np.random.default_rng(seed)
    differences = []
    while len(differences) < draws:
        indices = rng.integers(0, len(y), size=len(y))
        sampled_y = y[indices]
        if len(np.unique(sampled_y)) != 2:
            continue
        differences.append(auroc(sampled_y, refit[indices]) - auroc(sampled_y, old[indices]))
    values = np.asarray(differences, dtype=np.float64)
    return {
        "observed_difference": observed,
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "draws": int(draws),
        "valid_draws": int(len(values)),
        "seed": int(seed),
    }


def stratum_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[tuple[str, bool]]:
    return Counter(
        (str(row["dataset"]), bool(row["current_dense_wrong"])) for row in rows
    )
