"""Pure helpers for robust ALL-source Stage-1 threshold calibration."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


LAYERS = tuple(range(28))
DEPTH_BINS = {
    "early": tuple(range(0, 9)),
    "middle": tuple(range(9, 19)),
    "late": tuple(range(19, 28)),
}


def validate_trajectory_matrix(
    scores: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != len(LAYERS):
        raise ValueError("scores must have shape [N, 28]")
    if y.shape != (len(values),) or not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be binary with shape [N]")
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("scores must be nonempty and finite")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("scores must be probabilities in [0, 1]")
    return values, y


def first_trigger_layers(
    scores: Sequence[Sequence[float]] | np.ndarray, threshold: float
) -> np.ndarray:
    """Return the first layer with score > threshold, or -1 when absent."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LAYERS):
        raise ValueError("scores must have shape [N, 28]")
    if not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("scores and threshold must be finite")
    crossings = values > float(threshold)
    first = np.argmax(crossings, axis=1).astype(np.int64)
    first[~crossings.any(axis=1)] = -1
    return first


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def operating_metrics(
    scores: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    threshold: float,
    *,
    include_depth: bool = True,
) -> dict[str, Any]:
    values, y = validate_trajectory_matrix(scores, labels)
    maxima = values.max(axis=1)
    triggered = maxima > float(threshold)
    correct = y == 0
    wrong = y == 1
    false_c = int(np.sum(triggered & correct))
    true_w = int(np.sum(triggered & wrong))
    total_triggered = int(triggered.sum())
    result: dict[str, Any] = {
        "records": int(len(y)),
        "correct": int(correct.sum()),
        "wrong": int(wrong.sum()),
        "triggered": total_triggered,
        "correct_false_admissions": false_c,
        "wrong_detected": true_w,
        "c_preservation": _safe_ratio(int(correct.sum()) - false_c, int(correct.sum())),
        "c_false_admission": _safe_ratio(false_c, int(correct.sum())),
        "w_recall": _safe_ratio(true_w, int(wrong.sum())),
        "w_miss_rate": _safe_ratio(int(wrong.sum()) - true_w, int(wrong.sum())),
        "trigger_precision": _safe_ratio(true_w, total_triggered),
        "trigger_rate": _safe_ratio(total_triggered, len(y)),
    }
    if not include_depth:
        return result
    first = first_trigger_layers(values, threshold)
    triggered_layers = first[first >= 0]
    result.update(
        {
            "median_first_trigger_layer": (
                float(np.median(triggered_layers)) if len(triggered_layers) else float("nan")
            ),
            "q25_first_trigger_layer": (
                float(np.quantile(triggered_layers, 0.25))
                if len(triggered_layers)
                else float("nan")
            ),
            "q75_first_trigger_layer": (
                float(np.quantile(triggered_layers, 0.75))
                if len(triggered_layers)
                else float("nan")
            ),
        }
    )
    for name, layers in DEPTH_BINS.items():
        count = int(np.isin(triggered_layers, layers).sum())
        result[f"{name}_trigger_count"] = count
        result[f"{name}_trigger_fraction"] = _safe_ratio(count, len(triggered_layers))
    return result


def useful_thresholds(*score_matrices: np.ndarray) -> list[float]:
    if not score_matrices:
        raise ValueError("at least one score matrix is required")
    maxima = []
    for matrix in score_matrices:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(LAYERS):
            raise ValueError("scores must have shape [N, 28]")
        maxima.extend(values.max(axis=1).tolist())
    unique = sorted(set(float(value) for value in maxima), reverse=True)
    if not unique:
        raise ValueError("threshold population is empty")
    unique.append(float(np.nextafter(unique[-1], -np.inf)))
    return unique


def robust_threshold_sweep(
    historical_scores: np.ndarray,
    historical_labels: np.ndarray,
    canonical_scores: np.ndarray,
    canonical_labels: np.ndarray,
    *,
    include_depth: bool,
) -> list[dict[str, Any]]:
    h_scores, h_y = validate_trajectory_matrix(historical_scores, historical_labels)
    c_scores, c_y = validate_trajectory_matrix(canonical_scores, canonical_labels)
    thresholds = useful_thresholds(h_scores, c_scores)
    pooled_scores = np.concatenate((h_scores, c_scores), axis=0)
    pooled_labels = np.concatenate((h_y, c_y), axis=0)
    rows = []
    for threshold in thresholds:
        historical = operating_metrics(
            h_scores, h_y, threshold, include_depth=include_depth
        )
        canonical = operating_metrics(
            c_scores, c_y, threshold, include_depth=include_depth
        )
        pooled = operating_metrics(
            pooled_scores, pooled_labels, threshold, include_depth=include_depth
        )
        row: dict[str, Any] = {"threshold": threshold}
        for prefix, metrics in (
            ("historical", historical),
            ("canonical", canonical),
            ("pooled", pooled),
        ):
            row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        row["worst_source_c_preservation"] = min(
            historical["c_preservation"], canonical["c_preservation"]
        )
        row["worst_source_w_recall"] = min(
            historical["w_recall"], canonical["w_recall"]
        )
        row["average_source_w_recall"] = (
            historical["w_recall"] + canonical["w_recall"]
        ) / 2.0
        rows.append(row)
    return rows


def most_permissive_at_preservation(
    rows: Sequence[Mapping[str, Any]], target: float
) -> dict[str, Any]:
    if not 0 < target <= 1:
        raise ValueError("preservation target must be in (0, 1]")
    eligible = [
        row for row in rows if float(row["worst_source_c_preservation"]) >= target
    ]
    if not eligible:
        raise RuntimeError(f"no threshold satisfies preservation target {target}")
    return dict(min(eligible, key=lambda row: float(row["threshold"])))


def select_primary_threshold(
    rows: Sequence[Mapping[str, Any]], *, minimum_preservation: float = 0.98
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if float(row["worst_source_c_preservation"]) >= minimum_preservation
    ]
    if not eligible:
        raise RuntimeError("no threshold satisfies the primary preservation rule")

    def finite_or_negative(value: Any) -> float:
        number = float(value)
        return number if np.isfinite(number) else -1.0

    return dict(
        max(
            eligible,
            key=lambda row: (
                float(row["pooled_w_recall"]),
                float(row["worst_source_c_preservation"]),
                float(row["worst_source_w_recall"]),
                finite_or_negative(row["pooled_trigger_precision"]),
                finite_or_negative(row.get("pooled_median_first_trigger_layer", -1.0)),
                float(row["threshold"]),
            ),
        )
    )


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if successes < 0 or total < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return float(center - radius), float(center + radius)


def group_bootstrap_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    draws: int,
    seed: int,
) -> list[dict[str, float]]:
    """Source-stratified image-group cluster bootstrap at one fixed threshold."""

    if draws < 1 or not rows:
        raise ValueError("bootstrap requires rows and positive draws")
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        source = str(row["source_regime"])
        if source not in {"historical", "canonical"}:
            raise ValueError(f"unexpected source: {source}")
        grouped[source][str(row["image_group_id"])].append(row)
    if set(grouped) != {"historical", "canonical"}:
        raise ValueError("bootstrap needs both sources")

    sufficient: dict[str, np.ndarray] = {}
    for source in ("historical", "canonical"):
        values = []
        for _, members in sorted(grouped[source].items()):
            c_total = c_preserved = w_total = w_triggered = total_triggered = 0
            for row in members:
                label = int(row["label"])
                triggered = float(row["score_max"]) > threshold
                if label == 0:
                    c_total += 1
                    c_preserved += int(not triggered)
                else:
                    w_total += 1
                    w_triggered += int(triggered)
                total_triggered += int(triggered)
            values.append((c_total, c_preserved, w_total, w_triggered, total_triggered))
        sufficient[source] = np.asarray(values, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    results = []
    for _ in range(draws):
        totals: dict[str, np.ndarray] = {}
        for source in ("historical", "canonical"):
            values = sufficient[source]
            indices = rng.integers(0, len(values), size=len(values))
            totals[source] = values[indices].sum(axis=0)
        h = totals["historical"]
        c = totals["canonical"]
        h_c = h[1] / h[0] if h[0] else np.nan
        c_c = c[1] / c[0] if c[0] else np.nan
        h_w = h[3] / h[2] if h[2] else np.nan
        c_w = c[3] / c[2] if c[2] else np.nan
        pooled_w = (h[3] + c[3]) / (h[2] + c[2]) if h[2] + c[2] else np.nan
        precision = (
            (h[3] + c[3]) / (h[4] + c[4]) if h[4] + c[4] else np.nan
        )
        results.append(
            {
                "historical_c_preservation": float(h_c),
                "canonical_c_preservation": float(c_c),
                "historical_w_recall": float(h_w),
                "canonical_w_recall": float(c_w),
                "pooled_w_recall": float(pooled_w),
                "trigger_precision": float(precision),
                "c_preservation_source_difference": float(h_c - c_c),
                "w_recall_source_difference": float(h_w - c_w),
            }
        )
    return results


def choose_freeze_decision(
    *,
    primary_metrics: Mapping[str, Any],
    useful_w_detection: bool,
    crossfit_stable: bool,
    catastrophic_cells: int,
) -> str:
    if float(primary_metrics["worst_source_c_preservation"]) < 0.98:
        raise ValueError("primary metrics violate the required preservation floor")
    if catastrophic_cells:
        return "C — Calibration remains source/dataset unstable"
    if not useful_w_detection:
        return "B — 98% gate too conservative"
    if not crossfit_stable:
        return "C — Calibration remains source/dataset unstable"
    return "A — Freeze robust Stage-1 gate"
