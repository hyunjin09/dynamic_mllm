"""Pure utilities for the frozen Stage-2 treatment-selectivity diagnostic."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dense_failure_stage2.v1_router import ACTION_NAMES


LABEL_KEEP = "KEEP_REQUIRED"
LABEL_INTERVENE = "INTERVENE_REQUIRED"
LABEL_MIXED = "MIXED"
LABEL_UNRESOLVED = "UNRESOLVED"


def classify_observed_actions(actions: Sequence[str]) -> str:
    """Classify an exact state using only observed successful continuations."""
    normalized = {str(action) for action in actions}
    unsupported = normalized.difference(ACTION_NAMES)
    if unsupported:
        raise ValueError(f"unsupported observed actions: {sorted(unsupported)}")
    if not normalized:
        return LABEL_UNRESOLVED
    if normalized == {"FULL"}:
        return LABEL_KEEP
    if "FULL" not in normalized:
        return LABEL_INTERVENE
    return LABEL_MIXED


def exact_router_representations(
    router: torch.nn.Module,
    text_states: torch.Tensor,
    visual_states: torch.Tensor,
    *,
    text_mask: torch.Tensor,
    visual_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expose the frozen router's exact READ/WRITE branch outputs and logits.

    This mirrors ``SharedReadWriteRouter.forward`` without changing its code or
    parameters.  Callers must validate the returned logits against a normal
    router forward before accepting extracted representations.
    """
    if text_states.ndim != 3 or visual_states.ndim != 3:
        raise ValueError("text and visual states must have [batch, tokens, hidden] shape")
    if text_states.shape[0] != visual_states.shape[0]:
        raise ValueError("text and visual batch sizes differ")
    if text_states.shape[-1] != int(router.hidden_size) or visual_states.shape[-1] != int(
        router.hidden_size
    ):
        raise ValueError("hidden-state width differs from the frozen router contract")
    if text_mask.shape != text_states.shape[:2] or visual_mask.shape != visual_states.shape[:2]:
        raise ValueError("mask shape differs from its token-state shape")
    text_mask = text_mask.bool()
    visual_mask = visual_mask.bool()
    if not bool(text_mask.any(dim=1).all()):
        raise ValueError("every sample must contain a valid text/control token")
    if not bool(visual_mask.any(dim=1).all()):
        raise ValueError("every sample must contain a valid visual mask row")

    compute_dtype = router.text_query_projection.weight.dtype
    text_states = text_states.to(dtype=compute_dtype)
    visual_states = visual_states.to(dtype=compute_dtype)
    batch = text_states.shape[0]
    last = text_mask.long().sum(dim=1) - 1
    query = text_states[torch.arange(batch, device=text_states.device), last]
    query = router.text_query_projection(query).unsqueeze(1)
    visual = router.visual_projection(visual_states)
    padding = ~visual_mask
    read, _ = router.read_attention(
        query, visual, visual, key_padding_mask=padding, need_weights=False
    )
    write_query = router.write_query.expand(batch, -1, -1)
    write, _ = router.write_attention(
        write_query, visual, visual, key_padding_mask=padding, need_weights=False
    )
    read_representation = read[:, 0]
    write_representation = write[:, 0]
    logits = router.action_head(torch.cat((read_representation, write_representation), dim=-1))
    return read_representation, write_representation, logits


def _group_key(row: Mapping[str, Any]) -> str:
    return str(row.get("image_group_id") or row["uid"])


def assign_group_folds(
    rows: Sequence[Mapping[str, Any]], *, folds: int, seed: int
) -> list[dict[str, Any]]:
    """Deterministically balance label/dataset/source strata at group level."""
    if folds < 2:
        raise ValueError("at least two folds are required")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    state_ids: set[str] = set()
    for row in rows:
        state_id = str(row["state_id"])
        if state_id in state_ids:
            raise ValueError(f"duplicate state ID: {state_id}")
        state_ids.add(state_id)
        groups[_group_key(row)].append(row)
    if len(groups) < folds:
        raise ValueError("fewer groups than folds")

    def stratum(row: Mapping[str, Any]) -> str:
        return "|".join(
            (
                str(int(row["binary_target"])),
                str(row["dataset"]),
                str(row["source_regime"]),
            )
        )

    totals = Counter(stratum(row) for row in rows)
    target = {key: value / folds for key, value in totals.items()}
    target_total = len(rows) / folds
    fold_counts = [Counter() for _ in range(folds)]
    fold_totals = [0] * folds
    fold_groups = [0] * folds
    ordered = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            sha256(f"{seed}:{item[0]}".encode()).hexdigest(),
        ),
    )
    group_fold: dict[str, int] = {}
    for group, members in ordered:
        contribution = Counter(stratum(row) for row in members)
        candidates = []
        for fold in range(folds):
            stratum_delta = sum(
                (
                    (fold_counts[fold][key] + contribution[key] - target[key]) ** 2
                    - (fold_counts[fold][key] - target[key]) ** 2
                )
                / max(target[key], 1.0)
                for key in totals
            )
            total_delta = (
                (fold_totals[fold] + len(members) - target_total) ** 2
                - (fold_totals[fold] - target_total) ** 2
            ) / max(
                target_total,
                1.0,
            )
            candidates.append(
                (
                    stratum_delta + total_delta,
                    fold_groups[fold],
                    fold_totals[fold],
                    sha256(f"{seed}:{group}:{fold}".encode()).hexdigest(),
                    fold,
                )
            )
        chosen = min(candidates)[-1]
        group_fold[group] = chosen
        fold_counts[chosen].update(contribution)
        fold_totals[chosen] += len(members)
        fold_groups[chosen] += 1
    if any(count == 0 for count in fold_groups):
        raise RuntimeError("group assignment produced an empty fold")
    return [
        {
            "state_id": str(row["state_id"]),
            "uid": str(row["uid"]),
            "image_group_id": _group_key(row),
            "fold": group_fold[_group_key(row)],
        }
        for row in rows
    ]


def training_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Equalize total UID mass, then equalize binary class mass."""
    if not rows:
        raise ValueError("training rows cannot be empty")
    uid_counts = Counter(str(row["uid"]) for row in rows)
    labels = np.asarray([int(row["binary_target"]) for row in rows], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("training fold must contain both classes")
    weights = np.asarray([1.0 / uid_counts[str(row["uid"])] for row in rows], dtype=np.float64)
    for label in (0, 1):
        mask = labels == label
        weights[mask] *= 1.0 / weights[mask].sum()
    weights *= len(weights) / weights.sum()
    return weights


def _validate_binary_metric_inputs(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("labels/scores must be finite aligned vectors")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("binary metrics require both classes")
    if weights is None:
        weights = np.ones(len(labels), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != labels.shape or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("metric weights must be finite aligned nonnegative values")
    if weights[labels == 0].sum() <= 0 or weights[labels == 1].sum() <= 0:
        raise ValueError("both classes require positive metric weight")
    return labels, scores, weights


def binary_metrics(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, float]:
    """Compute tie-aware weighted AUROC and average precision."""
    labels, scores, weights = _validate_binary_metric_inputs(labels, scores, weights)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores, sorted_labels, sorted_weights = scores[order], labels[order], weights[order]
    total_positive = float(weights[labels == 1].sum())
    total_negative = float(weights[labels == 0].sum())
    concordant = 0.0
    negative_before = 0.0
    index = 0
    while index < len(labels):
        stop = index + 1
        while stop < len(labels) and sorted_scores[stop] == sorted_scores[index]:
            stop += 1
        group_labels = sorted_labels[index:stop]
        group_weights = sorted_weights[index:stop]
        positive = float(group_weights[group_labels == 1].sum())
        negative = float(group_weights[group_labels == 0].sum())
        concordant += positive * negative_before + 0.5 * positive * negative
        negative_before += negative
        index = stop
    auroc = concordant / (total_positive * total_negative)

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_labels, sorted_weights = scores[order], labels[order], weights[order]
    true_positive = false_positive = average_precision = 0.0
    previous_recall = 0.0
    index = 0
    while index < len(labels):
        stop = index + 1
        while stop < len(labels) and sorted_scores[stop] == sorted_scores[index]:
            stop += 1
        group_labels = sorted_labels[index:stop]
        group_weights = sorted_weights[index:stop]
        true_positive += float(group_weights[group_labels == 1].sum())
        false_positive += float(group_weights[group_labels == 0].sum())
        recall = true_positive / total_positive
        precision = true_positive / (true_positive + false_positive)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        index = stop
    return {"auroc": float(auroc), "auprc": float(average_precision)}


def matched_evaluation_weights(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, int]]:
    """Equalize KEEP/INTERVENE weight within every supported nuisance cell."""
    by_cell_label = Counter(
        (str(row["match_cell"]), int(row["binary_target"])) for row in rows
    )
    supported = {
        cell
        for cell, _label in by_cell_label
        if by_cell_label[(cell, 0)] > 0 and by_cell_label[(cell, 1)] > 0
    }
    weights = np.zeros(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        cell, label = str(row["match_cell"]), int(row["binary_target"])
        if cell in supported:
            weights[index] = 1.0 / by_cell_label[(cell, label)]
    supported_states = int(np.count_nonzero(weights))
    return weights, {
        "supported_cells": len(supported),
        "supported_states": supported_states,
        "excluded_states": len(rows) - supported_states,
    }


def validate_extracted_state_rows(
    expected_state_ids: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    expected = list(map(str, expected_state_ids))
    if len(expected) != len(set(expected)):
        raise ValueError("expected state IDs are duplicated")
    actual = [str(row["state_id"]) for row in rows]
    duplicates = len(actual) - len(set(actual))
    if duplicates:
        raise RuntimeError(f"duplicate extracted states: {duplicates}")
    unexpected = sorted(set(actual).difference(expected))
    if unexpected:
        raise RuntimeError(f"unexpected extracted states: {len(unexpected)}")
    missing = len(set(expected).difference(actual))
    if missing:
        raise RuntimeError(f"missing extracted states: {missing}")
    return {"states": len(actual), "duplicates": 0, "missing": 0}


def standardize_fold(
    train: np.ndarray, heldout: np.ndarray, train_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    """Apply training-only, weight-aware normalization to one fold."""
    train = np.asarray(train, dtype=np.float32)
    heldout = np.asarray(heldout, dtype=np.float32)
    weights = np.asarray(train_weights, dtype=np.float64)
    if train.ndim != 2 or heldout.ndim != 2 or train.shape[1] != heldout.shape[1]:
        raise ValueError("train and heldout features must have a common 2-D width")
    if weights.shape != (len(train),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("normalization weights are invalid")
    if weights.sum() <= 0 or not np.isfinite(train).all() or not np.isfinite(heldout).all():
        raise ValueError("features/normalization weights must be finite and nonempty")
    normalized_weights = weights / weights.sum()
    mean = np.sum(train.astype(np.float64) * normalized_weights[:, None], axis=0)
    variance = np.sum(
        np.square(train.astype(np.float64) - mean) * normalized_weights[:, None], axis=0
    )
    std = np.sqrt(np.maximum(variance, 0.0))
    std[std < 1e-6] = 1.0
    normalized_train = ((train - mean) / std).astype(np.float32)
    normalized_heldout = ((heldout - mean) / std).astype(np.float32)
    return normalized_train, normalized_heldout, {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def _probe_model(kind: str, input_size: int, hidden_size: int | None = None) -> torch.nn.Module:
    if kind == "linear":
        return torch.nn.Linear(input_size, 1)
    if kind == "mlp":
        if hidden_size is None or hidden_size < 1:
            raise ValueError("MLP requires a positive hidden size")
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, 1),
        )
    raise ValueError(f"unsupported probe kind: {kind}")


def fit_binary_probe(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    kind: str,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: str | torch.device,
    hidden_size: int | None = None,
) -> dict[str, Any]:
    """Fit one fixed-capacity full-batch diagnostic probe."""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if features.ndim != 2 or labels.shape != (len(features),) or weights.shape != labels.shape:
        raise ValueError("probe features, labels, and weights are misaligned")
    if set(labels.astype(int).tolist()) != {0, 1}:
        raise ValueError("probe fitting requires both classes")
    if not np.isfinite(features).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("probe inputs must be finite with nonnegative weights")
    if epochs < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid probe optimization settings")
    torch.manual_seed(int(seed))
    resolved_device = torch.device(device)
    model = _probe_model(kind, features.shape[1], hidden_size).to(resolved_device)
    if kind == "linear":
        torch.nn.init.zeros_(model.weight)
        torch.nn.init.zeros_(model.bias)
    x = torch.from_numpy(features).to(resolved_device)
    y = torch.from_numpy(labels).to(resolved_device)
    w = torch.from_numpy(weights).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    first_loss = final_loss = None
    model.train()
    for _epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(-1)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, reduction="none"
        )
        loss = (losses * w).sum() / w.sum()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("probe loss is non-finite")
        if first_loss is None:
            first_loss = float(loss.detach().cpu())
        loss.backward()
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise RuntimeError("probe gradient is non-finite")
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    return {
        "kind": kind,
        "input_size": int(features.shape[1]),
        "hidden_size": hidden_size,
        "seed": int(seed),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "first_loss": first_loss,
        "final_loss": final_loss,
        "state_dict": {
            name: tensor.detach().cpu().tolist() for name, tensor in model.state_dict().items()
        },
    }


def predict_binary_probe(
    fitted: Mapping[str, Any], features: np.ndarray, *, device: str | torch.device
) -> np.ndarray:
    """Return INTERVENE probabilities for a serialized diagnostic probe."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != int(fitted["input_size"]):
        raise ValueError("prediction feature width differs from fitted probe")
    resolved_device = torch.device(device)
    model = _probe_model(
        str(fitted["kind"]),
        int(fitted["input_size"]),
        fitted.get("hidden_size"),
    ).to(resolved_device)
    template = model.state_dict()
    state = {
        name: torch.as_tensor(fitted["state_dict"][name], dtype=tensor.dtype, device=resolved_device)
        for name, tensor in template.items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        scores = torch.sigmoid(model(torch.from_numpy(features).to(resolved_device)).squeeze(-1))
    values = scores.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("probe predictions are non-finite")
    return values


def selective_operating_points(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    coverage_fractions: Sequence[float],
    precision_targets: Sequence[float],
) -> list[dict[str, Any]]:
    """Evaluate fixed high-score subsets without splitting score ties."""
    labels, scores, _weights = _validate_binary_metric_inputs(labels, scores, None)
    positives = int(labels.sum())
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_labels = scores[order], labels[order]
    prefixes = []
    selected = true_positive = 0
    index = 0
    while index < len(labels):
        stop = index + 1
        while stop < len(labels) and sorted_scores[stop] == sorted_scores[index]:
            stop += 1
        selected += stop - index
        true_positive += int(sorted_labels[index:stop].sum())
        prefixes.append(
            {
                "threshold": float(sorted_scores[index]),
                "selected": selected,
                "coverage": selected / len(labels),
                "precision": true_positive / selected,
                "recall": true_positive / positives,
            }
        )
        index = stop
    output: list[dict[str, Any]] = []
    for fraction in coverage_fractions:
        fraction = float(fraction)
        if not 0 < fraction <= 1:
            raise ValueError("coverage fractions must be in (0, 1]")
        requested = int(np.ceil(len(labels) * fraction))
        chosen = next(row for row in prefixes if int(row["selected"]) >= requested)
        output.append(
            {
                "operating_point": f"top_{int(round(100 * fraction))}pct",
                "mode": "fixed_coverage",
                "requested_coverage": fraction,
                "target_precision": None,
                **chosen,
            }
        )
    for target in precision_targets:
        target = float(target)
        if not 0 < target <= 1:
            raise ValueError("precision targets must be in (0, 1]")
        supported = [row for row in prefixes if float(row["precision"]) >= target]
        chosen = max(supported, key=lambda row: (float(row["recall"]), int(row["selected"]))) if supported else None
        output.append(
            {
                "operating_point": f"recall_at_{int(round(100 * target))}pct_precision",
                "mode": "precision_target",
                "requested_coverage": None,
                "target_precision": target,
                "threshold": None if chosen is None else chosen["threshold"],
                "selected": 0 if chosen is None else chosen["selected"],
                "coverage": 0.0 if chosen is None else chosen["coverage"],
                "precision": None if chosen is None else chosen["precision"],
                "recall": 0.0 if chosen is None else chosen["recall"],
            }
        )
    return output
