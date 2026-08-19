"""Generation-based checkpoint selection for conservative visual routing."""

from __future__ import annotations

from typing import Any


def _mean(candidate: dict[str, Any], group: str, metric: str) -> float:
    return float(candidate["summary"][group][metric]["mean"])


def select_generation_checkpoint(
    candidates: list[dict[str, Any]],
    *,
    min_preservation: float = 0.95,
    max_baseline_drop: float = 0.0,
    accuracy_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Select a sparse checkpoint only after preservation and accuracy gates pass."""
    if not candidates:
        raise ValueError("at least one checkpoint candidate is required")
    if not 0.0 <= float(min_preservation) <= 1.0:
        raise ValueError("min_preservation must be in [0, 1]")
    if float(max_baseline_drop) < 0.0 or float(accuracy_tolerance) < 0.0:
        raise ValueError("accuracy tolerances must be non-negative")

    safe = []
    for candidate in candidates:
        preservation = _mean(candidate, "complete_correct", "online_correct_rate")
        accuracy = _mean(candidate, "all", "online_correct_rate")
        baseline = _mean(candidate, "all", "source_full_correct_rate")
        if preservation >= float(min_preservation) and accuracy >= baseline - float(max_baseline_drop):
            safe.append(candidate)

    if safe:
        best_accuracy = max(_mean(candidate, "all", "online_correct_rate") for candidate in safe)
        near_best = [
            candidate
            for candidate in safe
            if _mean(candidate, "all", "online_correct_rate") >= best_accuracy - float(accuracy_tolerance)
        ]
        chosen = min(
            near_best,
            key=lambda candidate: (
                _mean(candidate, "all", "avg_selected_layers"),
                -_mean(candidate, "all", "online_correct_rate"),
                str(candidate["checkpoint"]),
            ),
        )
        tier = "safe_noninferior"
    else:
        chosen = max(
            candidates,
            key=lambda candidate: (
                _mean(candidate, "complete_correct", "online_correct_rate"),
                _mean(candidate, "all", "online_correct_rate"),
                -_mean(candidate, "all", "avg_selected_layers"),
            ),
        )
        tier = "fallback_preservation"
    return {
        **chosen,
        "selection_tier": tier,
        "selection_policy": {
            "min_preservation": float(min_preservation),
            "max_baseline_drop": float(max_baseline_drop),
            "accuracy_tolerance": float(accuracy_tolerance),
        },
    }
