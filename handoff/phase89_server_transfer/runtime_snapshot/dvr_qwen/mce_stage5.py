"""MCE-5 oracle add-back planning and summary helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

from dvr_qwen.mce_inventory import BENCHMARKS
from dvr_qwen.mce_stage0 import source_row_for_manifest
from dvr_qwen.mce_stage3 import delta_sign, mean, route_binary, route_from_binary, route_with_layer
from dvr_qwen.mce_stage4 import (
    CRITICAL_ON,
    FULL_FIX_SCORE,
    HARMFUL_ON,
    REDUNDANT_ON,
    SCORE_ATOL,
    UNRESOLVED,
    stddev,
)


def _rate(numer: int, denom: int) -> float:
    return float(numer / denom) if denom else 0.0


def oracle_improved_over_all_visual(
    source_row: dict[str, Any],
    *,
    atol: float = SCORE_ATOL,
) -> bool:
    return float(source_row["binary_oracle"]["best"]["score"]) > float(source_row["binary_all_visual_on"]["score"]) + atol


def oracle_fixed_over_all_visual(
    source_row: dict[str, Any],
    *,
    full_fix_score: float = FULL_FIX_SCORE,
    atol: float = SCORE_ATOL,
) -> bool:
    all_visual_score = float(source_row["binary_all_visual_on"]["score"])
    oracle_score = float(source_row["binary_oracle"]["best"]["score"])
    return all_visual_score < full_fix_score - atol and oracle_score >= full_fix_score - atol


def build_oracle_add_back_specs(
    manifest_rows: list[dict[str, Any]],
    source_indexes: dict[str, dict[str, dict[str, Any]]],
    *,
    full_fix_score: float = FULL_FIX_SCORE,
    atol: float = SCORE_ATOL,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for sample_index, manifest_row in enumerate(manifest_rows):
        if str(manifest_row["cohort"]) != "wrong":
            continue
        source_row = source_row_for_manifest(manifest_row, source_indexes)
        if not oracle_improved_over_all_visual(source_row, atol=atol):
            continue

        all_visual = source_row["binary_all_visual_on"]
        oracle = source_row["binary_oracle"]["best"]
        oracle_route = route_from_binary(oracle["visual_on_mask"])
        oracle_fixed = oracle_fixed_over_all_visual(source_row, full_fix_score=full_fix_score, atol=atol)
        base = {
            "sample_index": sample_index,
            "sample_id": manifest_row["sample_id"],
            "dataset": manifest_row["dataset"],
            "cohort": manifest_row["cohort"],
            "source_pool": manifest_row["source_pool"],
            "source_asset_id": manifest_row.get("source_asset_id"),
            "mode": "oracle_add_back",
            "parent_mode": "baseline_oracle",
            "parent_score": float(oracle["score"]),
            "parent_prediction": oracle["prediction"],
            "parent_num_visual_on_layers": int(oracle.get("num_visual_on_layers", sum(oracle_route))),
            "parent_mask_one_based": oracle.get("mask_one_based"),
            "all_visual_score": float(all_visual["score"]),
            "all_visual_prediction": all_visual["prediction"],
            "oracle_score_delta_vs_all_visual": float(oracle["score"]) - float(all_visual["score"]),
            "oracle_fixed_over_all_visual": oracle_fixed,
            "oracle_improved_over_all_visual": True,
        }
        for layer_idx, is_on in enumerate(oracle_route):
            if is_on:
                continue
            route = route_with_layer(oracle_route, layer_idx, True)
            specs.append(
                {
                    **base,
                    "intervention_id": f"{manifest_row['sample_id']}:oracle_add_back:L{layer_idx + 1}",
                    "layer_zero_based": layer_idx,
                    "layer_one_based": layer_idx + 1,
                    "eligible": True,
                    "route": route_binary(route),
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


def add_back_counterfactual_label(delta_vs_oracle_parent: float | None, *, atol: float = SCORE_ATOL) -> str:
    if delta_vs_oracle_parent is None:
        return UNRESOLVED
    if delta_vs_oracle_parent < -atol:
        return HARMFUL_ON
    if delta_vs_oracle_parent > atol:
        return CRITICAL_ON
    return REDUNDANT_ON


def classify_oracle_add_back_row(row: dict[str, Any], *, atol: float = SCORE_ATOL) -> dict[str, Any]:
    parent_score = float(row["parent_score"])
    score = float(row["score"])
    delta = score - parent_score
    label = add_back_counterfactual_label(delta, atol=atol)
    return {
        "score_delta_vs_parent": delta,
        "delta_sign": delta_sign(delta, atol=atol),
        "counterfactual_label": label,
        "add_back_regressed": label == HARMFUL_ON,
        "add_back_improved": label == CRITICAL_ON,
        "neutral_redundant": label == REDUNDANT_ON,
    }


def _sample_total(rows: list[dict[str, Any]]) -> int:
    return len({str(row["sample_id"]) for row in rows})


def _sample_count(rows: list[dict[str, Any]], key: str) -> int:
    return len({str(row["sample_id"]) for row in rows if bool(row.get(key))})


def _event_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(bool(row.get(key)) for row in rows)


def _summary_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["score_delta_vs_parent"]) for row in rows if row.get("score_delta_vs_parent") is not None]
    label_counts = Counter(str(row.get("counterfactual_label", UNRESOLVED)) for row in rows)
    regressed = _event_count(rows, "add_back_regressed")
    improved = _event_count(rows, "add_back_improved")
    neutral = _event_count(rows, "neutral_redundant")
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
        "add_back_regression_events": regressed,
        "samples_with_any_add_back_regression": _sample_count(rows, "add_back_regressed"),
        "add_back_improvement_events": improved,
        "samples_with_any_add_back_improvement": _sample_count(rows, "add_back_improved"),
        "neutral_redundant_events": neutral,
        "event_regression_rate": _rate(regressed, len(rows)),
        "sample_regression_rate": _rate(_sample_count(rows, "add_back_regressed"), _sample_total(rows)),
        "neutral_rate": _rate(neutral, len(rows)),
    }


def summarize_oracle_add_back_specs(specs: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset = Counter(str(spec["dataset"]) for spec in specs)
    by_layer = Counter(int(spec["layer_one_based"]) for spec in specs)
    fixed_samples = {
        str(spec["sample_id"])
        for spec in specs
        if bool(spec.get("oracle_fixed_over_all_visual"))
    }
    improved_only_samples = {
        str(spec["sample_id"])
        for spec in specs
        if not bool(spec.get("oracle_fixed_over_all_visual"))
    }
    return {
        "planned_count": len(specs),
        "eligible_count": sum(bool(spec.get("eligible", True)) for spec in specs),
        "eligible_sample_count": len({str(spec["sample_id"]) for spec in specs}),
        "oracle_fixed_sample_count": len(fixed_samples),
        "oracle_improved_only_sample_count": len(improved_only_samples),
        "planned_by_dataset": dict(by_dataset),
        "planned_by_layer_one_based": {str(layer): by_layer[layer] for layer in sorted(by_layer)},
    }


def summarize_oracle_add_back_rows(
    rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    *,
    reused_existing_count: int = 0,
) -> dict[str, Any]:
    spec_summary = summarize_oracle_add_back_specs(specs)
    expected_ids = {str(spec["intervention_id"]) for spec in specs if spec.get("eligible", True)}
    evaluated_ids = {str(row["intervention_id"]) for row in rows}
    duplicate_ids = [
        intervention_id
        for intervention_id, count in Counter(str(row["intervention_id"]) for row in rows).items()
        if count > 1
    ]
    classified_rows = [
        {**row, **classify_oracle_add_back_row(row)}
        if row.get("score_delta_vs_parent") is None or row.get("counterfactual_label") is None
        else row
        for row in rows
    ]
    layers = sorted({int(spec["layer_one_based"]) for spec in specs})
    by_dataset = {
        dataset: _summary_for_rows([row for row in classified_rows if str(row["dataset"]) == dataset])
        for dataset in BENCHMARKS
    }
    by_layer = {
        str(layer): _summary_for_rows([row for row in classified_rows if int(row["layer_one_based"]) == layer])
        for layer in layers
    }
    by_dataset_layer = {
        f"{dataset}/L{layer}": _summary_for_rows(
            [
                row
                for row in classified_rows
                if str(row["dataset"]) == dataset and int(row["layer_one_based"]) == layer
            ]
        )
        for dataset in BENCHMARKS
        for layer in layers
    }
    fixed_rows = [row for row in classified_rows if bool(row.get("oracle_fixed_over_all_visual"))]
    improved_only_rows = [row for row in classified_rows if not bool(row.get("oracle_fixed_over_all_visual"))]
    return {
        **spec_summary,
        "evaluated_count": len(rows),
        "reused_existing_count": reused_existing_count,
        "missing_eligible_count": len(expected_ids - evaluated_ids),
        "missing_eligible_ids": sorted(expected_ids - evaluated_ids),
        "duplicate_intervention_ids": sorted(duplicate_ids),
        "overall": _summary_for_rows(classified_rows),
        "oracle_fixed_subset": _summary_for_rows(fixed_rows),
        "oracle_improved_only_subset": _summary_for_rows(improved_only_rows),
        "by_dataset": by_dataset,
        "by_layer_one_based": by_layer,
        "by_dataset_layer": by_dataset_layer,
    }
