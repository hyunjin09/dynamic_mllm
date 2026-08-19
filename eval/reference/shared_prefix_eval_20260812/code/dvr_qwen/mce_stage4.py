"""MCE-4 all-ON single-layer suppression helpers."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from dvr_qwen.mce_inventory import BENCHMARKS
from dvr_qwen.mce_stage0 import source_row_for_manifest
from dvr_qwen.mce_stage3 import delta_sign, mean, route_binary, route_with_layer


DEFAULT_NUM_LAYERS = 28
SCORE_ATOL = 1e-9
FULL_FIX_SCORE = 1.0

HARMFUL_ON = "HARMFUL_ON"
CRITICAL_ON = "CRITICAL_ON"
REDUNDANT_ON = "REDUNDANT_ON"
UNRESOLVED = "UNRESOLVED"


def stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def all_on_route(num_layers: int) -> list[bool]:
    return [True] * num_layers


def num_layers_for_source_row(source_row: dict[str, Any], default: int = DEFAULT_NUM_LAYERS) -> int:
    all_visual = source_row.get("binary_all_visual_on")
    if isinstance(all_visual, dict) and isinstance(all_visual.get("visual_on_mask"), list):
        return len(all_visual["visual_on_mask"])
    oracle = source_row.get("binary_oracle")
    if isinstance(oracle, dict):
        best = oracle.get("best")
        if isinstance(best, dict) and isinstance(best.get("visual_on_mask"), list):
            return len(best["visual_on_mask"])
    return default


def build_single_layer_suppression_specs(
    manifest_rows: list[dict[str, Any]],
    source_indexes: dict[str, dict[str, dict[str, Any]]],
    *,
    num_layers: int | None = None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for sample_index, manifest_row in enumerate(manifest_rows):
        source_row = source_row_for_manifest(manifest_row, source_indexes)
        current_num_layers = int(num_layers or num_layers_for_source_row(source_row))
        parent = source_row["binary_all_visual_on"]
        route = all_on_route(current_num_layers)
        base = {
            "sample_index": sample_index,
            "sample_id": manifest_row["sample_id"],
            "dataset": manifest_row["dataset"],
            "cohort": manifest_row["cohort"],
            "source_pool": manifest_row["source_pool"],
            "source_asset_id": manifest_row.get("source_asset_id"),
            "mode": "all_on_single_layer_suppress",
            "parent_mode": "baseline_all_visual_on",
            "parent_score": float(parent["score"]),
            "parent_prediction": parent["prediction"],
            "parent_num_visual_on_layers": int(parent.get("num_visual_on_layers", current_num_layers)),
        }
        for layer_idx in range(current_num_layers):
            suppressed_route = route_with_layer(route, layer_idx, False)
            specs.append(
                {
                    **base,
                    "intervention_id": f"{manifest_row['sample_id']}:all_on_suppress:L{layer_idx + 1}",
                    "layer_zero_based": layer_idx,
                    "layer_one_based": layer_idx + 1,
                    "eligible": True,
                    "route": route_binary(suppressed_route),
                }
            )
    return specs


def specs_for_shard(
    specs: list[dict[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    return [spec for spec in specs if int(spec["sample_index"]) % num_shards == shard_index]


def counterfactual_label(delta: float | None, *, atol: float = SCORE_ATOL) -> str:
    if delta is None:
        return UNRESOLVED
    if delta > atol:
        return HARMFUL_ON
    if delta < -atol:
        return CRITICAL_ON
    return REDUNDANT_ON


def classify_single_layer_row(
    row: dict[str, Any],
    *,
    full_fix_score: float = FULL_FIX_SCORE,
    atol: float = SCORE_ATOL,
) -> dict[str, Any]:
    parent_score = float(row["parent_score"])
    score = float(row["score"])
    delta = score - parent_score
    label = counterfactual_label(delta, atol=atol)
    cohort = str(row["cohort"])
    return {
        "score_delta_vs_parent": delta,
        "delta_sign": delta_sign(delta, atol=atol),
        "counterfactual_label": label,
        "wrong_full_fix": cohort == "wrong" and parent_score < full_fix_score - atol and score >= full_fix_score - atol,
        "wrong_improved": cohort == "wrong" and delta > atol,
        "correct_regressed": cohort == "correct" and delta < -atol,
        "neutral_redundant": label == REDUNDANT_ON,
    }


def _sample_count(rows: list[dict[str, Any]], predicate_key: str) -> int:
    return len({str(row["sample_id"]) for row in rows if bool(row.get(predicate_key))})


def _sample_total(rows: list[dict[str, Any]]) -> int:
    return len({str(row["sample_id"]) for row in rows})


def _event_count(rows: list[dict[str, Any]], predicate_key: str) -> int:
    return sum(bool(row.get(predicate_key)) for row in rows)


def _rate(numer: int, denom: int) -> float:
    return float(numer / denom) if denom else 0.0


def _summary_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["score_delta_vs_parent"]) for row in rows if row.get("score_delta_vs_parent") is not None]
    label_counts = Counter(str(row.get("counterfactual_label", UNRESOLVED)) for row in rows)
    return {
        "samples": _sample_total(rows),
        "evals": len(rows),
        "score_delta_mean": mean(deltas),
        "score_delta_stddev": stddev(deltas),
        "score_delta_min": min(deltas) if deltas else None,
        "score_delta_max": max(deltas) if deltas else None,
        "label_counts": dict(label_counts),
        "harmful_on_events": label_counts.get(HARMFUL_ON, 0),
        "critical_on_events": label_counts.get(CRITICAL_ON, 0),
        "redundant_on_events": label_counts.get(REDUNDANT_ON, 0),
        "neutral_rate": _rate(label_counts.get(REDUNDANT_ON, 0), len(rows)),
        "wrong_full_fix_events": _event_count(rows, "wrong_full_fix"),
        "wrong_samples_with_any_full_fix": _sample_count(rows, "wrong_full_fix"),
        "wrong_improved_events": _event_count(rows, "wrong_improved"),
        "wrong_samples_with_any_improvement": _sample_count(rows, "wrong_improved"),
        "correct_regression_events": _event_count(rows, "correct_regressed"),
        "correct_samples_with_any_regression": _sample_count(rows, "correct_regressed"),
    }


def summarize_single_layer_specs(specs: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset = Counter(str(spec["dataset"]) for spec in specs)
    by_cohort = Counter(str(spec["cohort"]) for spec in specs)
    by_dataset_cohort = Counter((str(spec["dataset"]), str(spec["cohort"])) for spec in specs)
    by_layer = Counter(int(spec["layer_one_based"]) for spec in specs)
    return {
        "planned_count": len(specs),
        "eligible_count": sum(bool(spec.get("eligible", True)) for spec in specs),
        "sample_count": len({str(spec["sample_id"]) for spec in specs}),
        "planned_by_dataset": dict(by_dataset),
        "planned_by_cohort": dict(by_cohort),
        "planned_by_dataset_cohort": {
            f"{dataset}/{cohort}": count for (dataset, cohort), count in sorted(by_dataset_cohort.items())
        },
        "planned_by_layer_one_based": {str(layer): by_layer[layer] for layer in sorted(by_layer)},
    }


def summarize_single_layer_rows(
    rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    *,
    reused_existing_count: int = 0,
) -> dict[str, Any]:
    spec_summary = summarize_single_layer_specs(specs)
    expected_ids = {str(spec["intervention_id"]) for spec in specs if spec.get("eligible", True)}
    evaluated_ids = {str(row["intervention_id"]) for row in rows}
    missing_ids = sorted(expected_ids - evaluated_ids)
    duplicate_ids = [
        intervention_id
        for intervention_id, count in Counter(str(row["intervention_id"]) for row in rows).items()
        if count > 1
    ]

    classified_rows = [
        {**row, **classify_single_layer_row(row)}
        if row.get("score_delta_vs_parent") is None or row.get("counterfactual_label") is None
        else row
        for row in rows
    ]
    overall = _summary_for_rows(classified_rows)
    wrong_rows = [row for row in classified_rows if str(row["cohort"]) == "wrong"]
    correct_rows = [row for row in classified_rows if str(row["cohort"]) == "correct"]

    by_dataset_cohort: dict[str, dict[str, Any]] = {}
    for dataset in BENCHMARKS:
        for cohort in ("wrong", "correct"):
            subset = [
                row
                for row in classified_rows
                if str(row["dataset"]) == dataset and str(row["cohort"]) == cohort
            ]
            by_dataset_cohort[f"{dataset}/{cohort}"] = _summary_for_rows(subset)

    by_layer: dict[str, dict[str, Any]] = {}
    layers = sorted({int(spec["layer_one_based"]) for spec in specs})
    for layer in layers:
        subset = [row for row in classified_rows if int(row["layer_one_based"]) == layer]
        by_layer[str(layer)] = _summary_for_rows(subset)

    by_dataset_layer: dict[str, dict[str, Any]] = {}
    for dataset in BENCHMARKS:
        for layer in layers:
            subset = [
                row
                for row in classified_rows
                if str(row["dataset"]) == dataset and int(row["layer_one_based"]) == layer
            ]
            by_dataset_layer[f"{dataset}/L{layer}"] = _summary_for_rows(subset)

    wrong_sample_total = _sample_total(wrong_rows)
    correct_sample_total = _sample_total(correct_rows)
    wrong_eval_total = len(wrong_rows)
    correct_eval_total = len(correct_rows)
    wrong_fix_samples = _sample_count(wrong_rows, "wrong_full_fix")
    wrong_improved_samples = _sample_count(wrong_rows, "wrong_improved")
    correct_regression_samples = _sample_count(correct_rows, "correct_regressed")
    critical_samples = _sample_count(classified_rows, "correct_regressed")

    return {
        **spec_summary,
        "evaluated_count": len(rows),
        "reused_existing_count": reused_existing_count,
        "missing_eligible_count": len(missing_ids),
        "missing_eligible_ids": missing_ids,
        "duplicate_intervention_ids": sorted(duplicate_ids),
        "classified_count": len(classified_rows),
        "overall": overall,
        "wrong_cohort": {
            **_summary_for_rows(wrong_rows),
            "sample_full_fix_rate": _rate(wrong_fix_samples, wrong_sample_total),
            "sample_improvement_rate": _rate(wrong_improved_samples, wrong_sample_total),
            "event_full_fix_rate": _rate(_event_count(wrong_rows, "wrong_full_fix"), wrong_eval_total),
            "event_improvement_rate": _rate(_event_count(wrong_rows, "wrong_improved"), wrong_eval_total),
        },
        "correct_cohort": {
            **_summary_for_rows(correct_rows),
            "sample_regression_rate": _rate(correct_regression_samples, correct_sample_total),
            "event_regression_rate": _rate(_event_count(correct_rows, "correct_regressed"), correct_eval_total),
        },
        "sample_with_any_critical_layer_rate": _rate(critical_samples, _sample_total(classified_rows)),
        "by_dataset_cohort": by_dataset_cohort,
        "by_layer_one_based": by_layer,
        "by_dataset_layer": by_dataset_layer,
    }
