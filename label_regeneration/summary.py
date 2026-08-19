"""P5 post-generation summaries for the regenerated 8K routing cache."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable


NUM_LAYERS = 28
DATASETS = ("gqa", "textvqa", "chartqa")
STATUS_VALUES = ("correct", "wrong")


def _quantile(values: list[float], probability: float) -> float | None:
    """Return the type-7/linear sample quantile used by NumPy's default."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[int | float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": (sum(values) / len(values)) if values else None,
        "minimum": min(values) if values else None,
        "p10": _quantile(values, 0.10),
        "p25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "maximum": max(values) if values else None,
    }


def summarize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Create one outcome summary without performing route-structure analysis."""
    sample = record["sample"]
    candidates = record["candidate_executions"]
    candidate_by_id = {candidate["route_id"]: candidate for candidate in candidates}
    root = candidate_by_id[record["root_route_id"]]
    all_off = candidate_by_id[record["all_off_route_id"]]
    successful = [candidate_by_id[route_id] for route_id in record["successful_route_ids"]]
    valid_count = len(successful)
    invalid_count = len(candidates) - valid_count
    current_status = sample["current_all_on_status"]
    historical_status = sample["historical_all_on_status"]
    if historical_status == current_status:
        drift = f"stable_{current_status}"
    else:
        drift = f"historical_{historical_status}_to_current_{current_status}"
    minimum_on = min((int(route["num_visual_on_layers"]) for route in successful), default=None)
    maximum_off = None if minimum_on is None else NUM_LAYERS - minimum_on
    current_wrong = current_status == "wrong"
    return {
        "schema_version": "label_regeneration_per_sample_summary_p5_v1",
        "uid": sample["uid"],
        "dataset": sample["benchmark"],
        "sample_id": sample["sample_id"],
        "image_group_id": sample["image_group_id"],
        "historical_all_on_status": historical_status,
        "current_all_on_status": current_status,
        "contract_drift": drift,
        "current_all_on_score": float(root["score"]),
        "current_all_on_prediction": root.get("prediction"),
        "all_off_correct": bool(all_off["result_correct"]),
        "all_off_score": float(all_off["score"]),
        "evaluated_route_count": len(candidates),
        "valid_route_count": valid_count,
        "invalid_route_count": invalid_count,
        "has_valid_route": valid_count > 0,
        "has_at_least_5_valid_routes": valid_count >= 5,
        "has_at_least_10_valid_routes": valid_count >= 10,
        "has_at_least_20_valid_routes": valid_count >= 20,
        "correction_found": (valid_count > 0) if current_wrong else None,
        "correcting_route_count": valid_count if current_wrong else None,
        "minimum_visual_on_valid_route": minimum_on,
        "maximum_visual_off_valid_route": maximum_off,
        "minimum_route_visual_compute_saving_fraction": (
            maximum_off / NUM_LAYERS if maximum_off is not None else None
        ),
        "best_sparse_success_route_id": record.get("best_sparse_success_route_id"),
        "requested_simulations": int(record["mcts"]["requested_simulations"]),
        "completed_simulations": int(record["mcts"]["completed_simulations"]),
        "extension_reason": record["mcts"].get("extension_reason"),
        "actual_text_tokens": int(sample["actual_text_tokens"]),
        "actual_visual_tokens": int(sample["actual_visual_tokens"]),
        "actual_full_prompt_tokens": int(sample["actual_full_prompt_tokens"]),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_counts = [row["valid_route_count"] for row in rows]
    evaluated_counts = [row["evaluated_route_count"] for row in rows]
    current_correct = [row for row in rows if row["current_all_on_status"] == "correct"]
    current_wrong = [row for row in rows if row["current_all_on_status"] == "wrong"]
    recovered = [row for row in current_wrong if row["correction_found"]]
    preserved_minimum_on = [
        row["minimum_visual_on_valid_route"]
        for row in current_correct
        if row["minimum_visual_on_valid_route"] is not None
    ]
    preserved_savings = [
        row["maximum_visual_off_valid_route"]
        for row in current_correct
        if row["maximum_visual_off_valid_route"] is not None
    ]
    return {
        "samples": len(rows),
        "historical_all_on": dict(sorted(Counter(row["historical_all_on_status"] for row in rows).items())),
        "current_all_on": dict(sorted(Counter(row["current_all_on_status"] for row in rows).items())),
        "contract_drift": dict(sorted(Counter(row["contract_drift"] for row in rows).items())),
        "search_budgets": {
            str(key): value
            for key, value in sorted(Counter(row["requested_simulations"] for row in rows).items())
        },
        "coverage": {
            "with_at_least_1": sum(row["has_valid_route"] for row in rows),
            "with_at_least_5": sum(row["has_at_least_5_valid_routes"] for row in rows),
            "with_at_least_10": sum(row["has_at_least_10_valid_routes"] for row in rows),
            "with_at_least_20": sum(row["has_at_least_20_valid_routes"] for row in rows),
            "zero_valid": sum(not row["has_valid_route"] for row in rows),
            "valid_route_count": distribution(valid_counts),
            "evaluated_route_count": distribution(evaluated_counts),
        },
        "correction": {
            "eligible_current_wrong": len(current_wrong),
            "recovered": len(recovered),
            "recovery_fraction": len(recovered) / len(current_wrong) if current_wrong else None,
            "correcting_route_count_all_current_wrong": distribution(
                row["correcting_route_count"] for row in current_wrong
            ),
            "correcting_route_count_recovered_only": distribution(
                row["correcting_route_count"] for row in recovered
            ),
        },
        "preservation": {
            "eligible_current_correct": len(current_correct),
            "minimum_visual_on_valid_route": distribution(preserved_minimum_on),
            "maximum_visual_off_valid_route": distribution(preserved_savings),
            "mean_visual_compute_saving_fraction": (
                sum(preserved_savings) / (NUM_LAYERS * len(preserved_savings))
                if preserved_savings
                else None
            ),
        },
        "all_off": {
            "correct": sum(row["all_off_correct"] for row in rows),
            "correct_fraction": (
                sum(row["all_off_correct"] for row in rows) / len(rows) if rows else None
            ),
        },
    }


def aggregate_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate P5 outcomes overall, by dataset, and dataset/current status."""
    datasets = sorted({row["dataset"] for row in rows})
    return {
        "schema_version": "label_regeneration_quality_summary_p5_v1",
        "scope": {
            "included_datasets": list(DATASETS),
            "excluded_datasets": ["wemath2pro"],
            "stages_included": ["P5"],
            "stages_excluded": ["P6", "P7", "P8", "P9", "P10"],
        },
        "definitions": {
            "valid_route": "evaluated complete 28-bit mask with frozen benchmark score at or above threshold",
            "correction": "at least one valid evaluated route when authoritative current ALL-ON is wrong",
            "minimum_budget": "fewest visual-ON layers among evaluated valid routes; ties are irrelevant to P5",
            "quantiles": "linear/type-7 empirical quantiles",
        },
        "overall": _aggregate(rows),
        "by_dataset": {dataset: _aggregate([row for row in rows if row["dataset"] == dataset]) for dataset in datasets},
        "by_dataset_and_current_status": {
            dataset: {
                status: _aggregate(
                    [
                        row
                        for row in rows
                        if row["dataset"] == dataset and row["current_all_on_status"] == status
                    ]
                )
                for status in STATUS_VALUES
            }
            for dataset in datasets
        },
    }
