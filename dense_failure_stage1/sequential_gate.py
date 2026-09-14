"""Pure calibration and evaluation helpers for an independent sequential gate."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def validate_score_matrix(
    scores: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 28:
        raise ValueError("sequential gate requires an N x 28 score matrix")
    if y.ndim != 1 or len(y) != len(values) or len(y) == 0:
        raise ValueError("score matrix and labels must be nonempty and aligned")
    if set(np.unique(y)) != {0, 1} or not np.isfinite(values).all():
        raise ValueError("sequential gate requires finite scores and both binary classes")
    return values, y


def empirical_alpha_grid(correct_records: int, *, maximum_alpha: float = 0.10) -> list[float]:
    """Return every attainable higher-quantile breakpoint up to the cap."""

    records = int(correct_records)
    cap = float(maximum_alpha)
    if records < 2 or not 0.0 < cap < 1.0:
        raise ValueError("invalid empirical alpha-grid contract")
    denominator = records - 1
    values = [index / denominator for index in range(int(np.floor(cap * denominator)) + 1)]
    if not np.isclose(values[-1], cap):
        values.append(cap)
    return values


def layer_thresholds(
    scores: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Compute 28 conservative empirical correct-tail thresholds."""

    values, y = validate_score_matrix(scores, labels)
    tail = float(alpha)
    if not 0.0 <= tail <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    correct = values[y == 0]
    return np.quantile(correct, 1.0 - tail, axis=0, method="higher")


def first_trigger_layers(
    scores: Sequence[Sequence[float]] | np.ndarray,
    thresholds: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return the first strict threshold crossing, or -1 when none occurs."""

    values = np.asarray(scores, dtype=np.float64)
    tau = np.asarray(thresholds, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 28 or tau.shape != (28,):
        raise ValueError("sequential trigger shapes differ from the 28-layer contract")
    if not np.isfinite(values).all() or not np.isfinite(tau).all():
        raise ValueError("sequential trigger values must be finite")
    crossings = values > tau[None, :]
    any_crossing = crossings.any(axis=1)
    first = crossings.argmax(axis=1).astype(np.int64)
    first[~any_crossing] = -1
    return first


def gate_metrics(
    labels: Sequence[int] | np.ndarray,
    first_layers: Sequence[int] | np.ndarray,
) -> dict[str, float | int | None]:
    y = np.asarray(labels, dtype=np.int64)
    first = np.asarray(first_layers, dtype=np.int64)
    if y.ndim != 1 or first.shape != y.shape or len(y) == 0:
        raise ValueError("gate labels and first-trigger layers must align")
    if set(np.unique(y)) != {0, 1} or not np.isin(first, np.arange(-1, 28)).all():
        raise ValueError("invalid labels or first-trigger layer")
    triggered = first >= 0
    correct = y == 0
    wrong = y == 1
    true_positive = int(np.logical_and(triggered, wrong).sum())
    false_positive = int(np.logical_and(triggered, correct).sum())
    triggered_layers = first[triggered]
    return {
        "records": int(len(y)),
        "correct": int(correct.sum()),
        "wrong": int(wrong.sum()),
        "triggered": int(triggered.sum()),
        "no_trigger": int((~triggered).sum()),
        "correct_false_triggers": false_positive,
        "wrong_detected": true_positive,
        "correct_preservation": float(1.0 - false_positive / correct.sum()),
        "wrong_detection_recall": float(true_positive / wrong.sum()),
        "failure_precision": float(true_positive / max(true_positive + false_positive, 1)),
        "trigger_rate": float(triggered.mean()),
        "no_trigger_fraction": float((~triggered).mean()),
        "median_first_trigger_layer": (
            float(np.median(triggered_layers)) if len(triggered_layers) else None
        ),
        "mean_first_trigger_layer": (
            float(np.mean(triggered_layers)) if len(triggered_layers) else None
        ),
    }


def sweep_shared_alpha(
    scores: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    alphas: Sequence[float],
) -> list[dict[str, Any]]:
    values, y = validate_score_matrix(scores, labels)
    output = []
    previous_alpha = -1.0
    for alpha in alphas:
        tail = float(alpha)
        if tail <= previous_alpha:
            raise ValueError("alpha grid must be strictly increasing")
        thresholds = layer_thresholds(values, y, alpha=tail)
        first = first_trigger_layers(values, thresholds)
        output.append({"alpha": tail, "thresholds": thresholds, **gate_metrics(y, first)})
        previous_alpha = tail
    return output


def select_operating_point(
    sweep: Sequence[Mapping[str, Any]], *, target_preservation: float
) -> dict[str, Any] | None:
    target = float(target_preservation)
    if not 0.0 < target <= 1.0:
        raise ValueError("target preservation must lie in (0, 1]")
    qualifying = [row for row in sweep if float(row["correct_preservation"]) + 1e-12 >= target]
    if not qualifying:
        return None
    return dict(max(qualifying, key=lambda row: float(row["alpha"])))


def fixed_layer_threshold(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    target_preservation: float,
) -> float:
    """Choose a tie-safe strict-crossing threshold on validation correct rows."""

    values = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    target = float(target_preservation)
    if values.ndim != 1 or y.shape != values.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("fixed-layer threshold inputs are invalid")
    if not np.isfinite(values).all() or not 0.0 < target <= 1.0:
        raise ValueError("fixed-layer threshold contract is invalid")
    correct = np.sort(values[y == 0])[::-1]
    allowed = int(np.floor((1.0 - target) * len(correct) + 1e-12))
    if allowed == 0:
        return float(correct[0])
    if allowed >= len(correct):
        return float(np.nextafter(correct[-1], -np.inf))
    return float(correct[allowed])


def region_name(layer: int) -> str:
    value = int(layer)
    if value == -1:
        return "never"
    if 0 <= value <= 8:
        return "early"
    if 9 <= value <= 18:
        return "middle"
    if 19 <= value <= 27:
        return "late"
    raise ValueError("trigger layer lies outside -1 or 0-27")
