from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def outcome_label(row: Mapping[str, Any]) -> str:
    baseline = bool(row["baseline_correct"])
    routed = bool(row["router_correct"])
    if baseline and routed:
        return "preserve"
    if baseline and not routed:
        return "harm"
    if routed:
        return "rescue"
    return "unsolved"


def summarize_admission(
    rows: Sequence[Mapping[str, Any]],
    harm_scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    if len(rows) != len(harm_scores):
        raise ValueError("row and score counts differ")
    baseline = np.asarray([bool(row["baseline_correct"]) for row in rows], dtype=np.int8)
    routed = np.asarray([bool(row["router_correct"]) for row in rows], dtype=np.int8)
    route_budget = np.asarray(
        [int(row["selected_num_visual_on_layers"]) for row in rows], dtype=np.float64
    )
    admission = np.asarray(harm_scores, dtype=np.float64) <= threshold
    selected = np.where(admission, routed, baseline)
    budget = np.where(admission, route_budget, 28.0)
    delta = selected - baseline
    n = len(rows)
    delta_mean = float(delta.mean()) if n else float("nan")
    delta_std = float(delta.std(ddof=1)) if n > 1 else 0.0
    lcb = delta_mean - 1.6448536269514722 * delta_std / np.sqrt(max(n, 1))
    baseline_positive = int(baseline.sum())
    harm = int((admission & (baseline == 1) & (routed == 0)).sum())
    rescue = int((admission & (baseline == 0) & (routed == 1)).sum())
    return {
        "n": n,
        "threshold": float(threshold),
        "baseline_accuracy": float(baseline.mean()),
        "selected_accuracy": float(selected.mean()),
        "accuracy_delta": delta_mean,
        "accuracy_delta_one_sided_95_lcb": float(lcb),
        "routed_count": int(admission.sum()),
        "route_fraction": float(admission.mean()),
        "harm_count": harm,
        "rescue_count": rescue,
        "preservation": float((baseline_positive - harm) / baseline_positive)
        if baseline_positive
        else None,
        "mean_visual_on_layers": float(budget.mean()),
        "route_sensitive_layer_saving_fraction": float((28.0 - budget.mean()) / 28.0),
    }


def calibrate_threshold(
    rows: Sequence[Mapping[str, Any]],
    harm_scores: np.ndarray,
    *,
    epsilon: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores = np.asarray(harm_scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("harm scores must be finite")
    if len(rows) != len(scores):
        raise ValueError("row and score counts differ")
    if not len(rows):
        raise ValueError("cannot calibrate on an empty population")

    baseline = np.asarray([bool(row["baseline_correct"]) for row in rows], dtype=np.int8)
    routed = np.asarray([bool(row["router_correct"]) for row in rows], dtype=np.int8)
    budgets = np.asarray(
        [int(row["selected_num_visual_on_layers"]) for row in rows], dtype=np.float64
    )
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_delta = (routed - baseline)[order]
    sorted_budget = budgets[order]
    sorted_harm = ((baseline == 1) & (routed == 0))[order].astype(np.int64)
    sorted_rescue = ((baseline == 0) & (routed == 1))[order].astype(np.int64)
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )

    delta_sum = np.cumsum(sorted_delta, dtype=np.float64)
    delta_square_sum = np.cumsum(sorted_delta.astype(np.float64) ** 2)
    budget_sum = np.cumsum(sorted_budget, dtype=np.float64)
    harm_sum = np.cumsum(sorted_harm)
    rescue_sum = np.cumsum(sorted_rescue)
    n = len(rows)
    baseline_positive = int(baseline.sum())
    baseline_accuracy = float(baseline.mean())

    def build_summary(threshold: float, admitted: int) -> dict[str, Any]:
        if admitted:
            end = admitted - 1
            total_delta = float(delta_sum[end])
            total_delta_square = float(delta_square_sum[end])
            admitted_budget = float(budget_sum[end])
            harm = int(harm_sum[end])
            rescue = int(rescue_sum[end])
        else:
            total_delta = total_delta_square = admitted_budget = 0.0
            harm = rescue = 0
        delta_mean = total_delta / n
        if n > 1:
            variance = max(
                0.0,
                (total_delta_square - total_delta * total_delta / n) / (n - 1),
            )
            delta_std = float(np.sqrt(variance))
        else:
            delta_std = 0.0
        lcb = delta_mean - 1.6448536269514722 * delta_std / np.sqrt(n)
        mean_budget = (admitted_budget + (n - admitted) * 28.0) / n
        return {
            "n": n,
            "threshold": float(threshold),
            "baseline_accuracy": baseline_accuracy,
            "selected_accuracy": baseline_accuracy + delta_mean,
            "accuracy_delta": delta_mean,
            "accuracy_delta_one_sided_95_lcb": float(lcb),
            "routed_count": admitted,
            "route_fraction": admitted / n,
            "harm_count": harm,
            "rescue_count": rescue,
            "preservation": float((baseline_positive - harm) / baseline_positive)
            if baseline_positive
            else None,
            "mean_visual_on_layers": float(mean_budget),
            "route_sensitive_layer_saving_fraction": float((28.0 - mean_budget) / 28.0),
        }

    sweep = [build_summary(np.nextafter(sorted_scores[0], -np.inf), 0)]
    sweep.extend(
        build_summary(float(sorted_scores[end]), int(end + 1)) for end in group_ends
    )
    sweep.append(build_summary(np.nextafter(sorted_scores[-1], np.inf), n))
    feasible = [
        row for row in sweep if row["accuracy_delta_one_sided_95_lcb"] >= -float(epsilon)
    ]
    if not feasible:
        raise RuntimeError("no threshold satisfies the non-inferiority constraint")
    selected = max(
        feasible,
        key=lambda row: (
            row["route_sensitive_layer_saving_fraction"],
            row["selected_accuracy"],
            -row["harm_count"],
        ),
    )
    return selected, sweep
