"""Pure helpers for Stage-1 gate winner and threshold selection."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from dense_failure_stage1.sequential_gate import (
    first_trigger_layers as independent_first_trigger_layers,
    gate_metrics,
    layer_thresholds,
    validate_score_matrix,
)
from dense_failure_stage1.shared_global_gate import first_trigger_layers as shared_first_trigger_layers


CANDIDATES = (
    "independent_sequential",
    "shared_all28",
    "shared_random4",
    "shared_fixed_l27",
)


def metrics_with_utility(
    labels: Sequence[int] | np.ndarray,
    first_layers: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    metrics = gate_metrics(labels, first_layers)
    utility = int(metrics["wrong_detected"]) - int(metrics["correct_false_triggers"])
    return {
        **metrics,
        "utility": utility,
        "utility_rate": float(utility / int(metrics["records"])),
    }


def shared_threshold_sweep(
    scores: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    layers: Sequence[int],
) -> list[dict[str, Any]]:
    values, y = validate_score_matrix(scores, labels)
    window = tuple(int(layer) for layer in layers)
    if not window:
        raise ValueError("shared threshold sweep requires a nonempty layer window")
    maxima = values[:, window].max(axis=1)
    thresholds = sorted(set(maxima.tolist()), reverse=True)
    thresholds.append(float(np.nextafter(min(thresholds), -np.inf)))
    rows = []
    for threshold in thresholds:
        first = shared_first_trigger_layers(values, threshold=threshold, window=window)
        rows.append(
            {
                "control_type": "global_raw_threshold",
                "control_value": float(threshold),
                "threshold": float(threshold),
                "layer_thresholds_json": "",
                **metrics_with_utility(y, first),
            }
        )
    return rows


def independent_alpha_sweep(
    scores: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> list[dict[str, Any]]:
    values, y = validate_score_matrix(scores, labels)
    correct_records = int((y == 0).sum())
    alphas = [index / (correct_records - 1) for index in range(correct_records)]
    rows = []
    for alpha in alphas:
        thresholds = layer_thresholds(values, y, alpha=alpha)
        first = independent_first_trigger_layers(values, thresholds)
        rows.append(
            {
                "control_type": "shared_correct_tail_alpha",
                "control_value": float(alpha),
                "threshold": float(alpha),
                "layer_thresholds_json": json.dumps(thresholds.tolist(), separators=(",", ":")),
                **metrics_with_utility(y, first),
            }
        )
    terminal = np.nextafter(values[y == 0].min(axis=0), -np.inf)
    first = independent_first_trigger_layers(values, terminal)
    rows.append(
        {
            "control_type": "terminal_below_correct_minima",
            "control_value": "terminal",
            "threshold": "terminal",
            "layer_thresholds_json": json.dumps(terminal.tolist(), separators=(",", ":")),
            **metrics_with_utility(y, first),
        }
    )
    return rows


def select_operating_point(
    rows: Sequence[Mapping[str, Any]], *, minimum_precision: float
) -> dict[str, Any]:
    floor = float(minimum_precision)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("minimum precision must lie in [0, 1]")
    eligible = [row for row in rows if float(row["failure_precision"]) + 1e-12 >= floor]
    if not eligible:
        raise RuntimeError("no threshold satisfies the precision requirement")

    def key(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
        median = row["median_first_trigger_layer"]
        earlier = -float(median) if median is not None else -math.inf
        return (
            float(row["utility_rate"]),
            float(row["correct_preservation"]),
            float(row["wrong_detection_recall"]),
            earlier,
        )

    return dict(max(eligible, key=key))


def select_winner(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not candidates:
        raise ValueError("winner selection requires candidates")

    def key(row: Mapping[str, Any]) -> tuple[float, float, float, float, int, int]:
        median = row["median_first_trigger_layer"]
        earlier = -float(median) if median is not None else -math.inf
        return (
            float(row["utility_rate"]),
            float(row["correct_preservation"]),
            float(row["wrong_detection_recall"]),
            earlier,
            -int(row["simplicity_rank"]),
            -int(row["candidate_order"]),
        )

    ranking = sorted((dict(row) for row in candidates), key=key, reverse=True)
    return ranking[0], ranking


def reference_operating_point(
    rows: Sequence[Mapping[str, Any]], *, target_preservation: float
) -> dict[str, Any]:
    target = float(target_preservation)
    if not 0.0 <= target <= 1.0 or not rows:
        raise ValueError("invalid reference preservation request")

    def key(row: Mapping[str, Any]) -> tuple[float, int, float, float]:
        preservation = float(row["correct_preservation"])
        return (
            -abs(preservation - target),
            int(preservation >= target),
            float(row["wrong_detection_recall"]),
            float(row["failure_precision"]),
        )

    return dict(max(rows, key=key))


def apply_control(
    candidate: str,
    scores: Sequence[Sequence[float]] | np.ndarray,
    control: Mapping[str, Any],
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 28 or not np.isfinite(values).all():
        raise ValueError("candidate score matrix must be finite N x 28")
    if candidate == "independent_sequential":
        thresholds = np.asarray(control["layer_thresholds"], dtype=np.float64)
        return independent_first_trigger_layers(values, thresholds)
    threshold = float(control["threshold"])
    if candidate in {"shared_all28", "shared_random4"}:
        return shared_first_trigger_layers(values, threshold=threshold, window=range(28))
    if candidate == "shared_fixed_l27":
        return shared_first_trigger_layers(values, threshold=threshold, window=[27])
    raise ValueError(f"unsupported gate candidate: {candidate}")


def paired_bootstrap(
    labels: Sequence[int] | np.ndarray,
    winner_first: Sequence[int] | np.ndarray,
    alternative_first: Sequence[int] | np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    winner = np.asarray(winner_first, dtype=np.int64) >= 0
    alternative = np.asarray(alternative_first, dtype=np.int64) >= 0
    if (
        y.ndim != 1
        or winner.shape != y.shape
        or alternative.shape != y.shape
        or set(np.unique(y)) != {0, 1}
        or replicates < 1
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("invalid paired bootstrap inputs")
    rng = np.random.default_rng(int(seed))
    utility_contribution = (2 * y - 1) * (winner.astype(int) - alternative.astype(int))
    wrong_delta = winner[y == 1].astype(float) - alternative[y == 1].astype(float)
    utility_draws = np.empty(replicates, dtype=np.float64)
    recall_draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        utility_indices = rng.integers(0, len(y), size=len(y))
        wrong_indices = rng.integers(0, len(wrong_delta), size=len(wrong_delta))
        utility_draws[index] = utility_contribution[utility_indices].mean()
        recall_draws[index] = wrong_delta[wrong_indices].mean()
    tail = (1.0 - confidence) / 2.0
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence": float(confidence),
        "winner_minus_alternative_utility_rate": {
            "observed": float(utility_contribution.mean()),
            "lower": float(np.quantile(utility_draws, tail)),
            "upper": float(np.quantile(utility_draws, 1.0 - tail)),
        },
        "winner_minus_alternative_wrong_recall": {
            "observed": float(wrong_delta.mean()),
            "lower": float(np.quantile(recall_draws, tail)),
            "upper": float(np.quantile(recall_draws, 1.0 - tail)),
        },
    }
