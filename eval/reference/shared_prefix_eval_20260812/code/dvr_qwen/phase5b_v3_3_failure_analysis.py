"""Cache-only diagnostics for Phase 5B v3.3 calibration failures."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

import torch

from dvr_qwen.route_selector import (
    NUM_LAYERS,
    candidate_on,
    candidate_rank,
    early_mid_late_counts,
    group_rows_by_id,
    longest_false_gap,
    route_mask_from_layers,
    transition_count,
    visual_on_segments,
)


PRED_SAFE_KEY = "v3_2_safe_prob"
PRED_REGRESSION_KEY = "v3_2_regression_prob"
PRED_PRESERVE_KEY = "v3_2_preserve_prob"
PRED_DELTA_KEY = "v3_2_delta_pred"
PRED_SCORE_KEY = "v3_3_harm_score"


def _as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _as_bool(row: dict[str, Any], key: str) -> bool:
    return bool(row.get(key, False))


def candidate_bucket(row: dict[str, Any]) -> str:
    """Return the v3.3 diagnostic class for a candidate row."""

    if _as_bool(row, "safe_switch"):
        return "safe_switch"
    if _as_bool(row, "regression") or _as_bool(row, "is_regression") or _as_bool(row, "full_correct_regression"):
        return "regression"
    if _as_bool(row, "cost_only_preserve"):
        return "cost_only_preserve"
    if _as_bool(row, "preserve") or _as_bool(row, "full_correct_preserved"):
        return "preserve_other"
    return "other"


def route_pattern(row: dict[str, Any], *, num_layers: int = NUM_LAYERS) -> dict[str, Any]:
    layers = [int(layer) for layer in row.get("layers_one_based", [])]
    mask = route_mask_from_layers(layers, num_layers=num_layers)
    early, middle, late = early_mid_late_counts(mask)
    segments = visual_on_segments(mask)
    return {
        "on_count": candidate_on(row),
        "transition_count": transition_count(mask),
        "early_on_count": early,
        "middle_on_count": middle,
        "late_on_count": late,
        "first_on": int(layers[0]) if layers else 0,
        "last_on": int(layers[-1]) if layers else 0,
        "segment_count": len(segments),
        "max_segment_len": max(segments) if segments else 0,
        "longest_text_gap": longest_false_gap(mask),
    }


def annotate_prediction_fields(
    rows: list[dict[str, Any]],
    *,
    safe_weight: float = 1.0,
    regression_weight: float = 2.0,
    preserve_weight: float = 0.25,
    delta_weight: float = 1.0,
    num_layers: int = NUM_LAYERS,
) -> list[dict[str, Any]]:
    """Attach bucket, route-pattern, and v3.3 harm-score fields."""

    out: list[dict[str, Any]] = []
    for row in rows:
        safe_prob = _as_float(row, PRED_SAFE_KEY)
        regression_prob = _as_float(row, PRED_REGRESSION_KEY)
        preserve_prob = _as_float(row, PRED_PRESERVE_KEY)
        delta_pred = _as_float(row, PRED_DELTA_KEY)
        harm_score = (
            float(delta_weight) * delta_pred
            + float(safe_weight) * safe_prob
            + float(preserve_weight) * preserve_prob
            - float(regression_weight) * regression_prob
        )
        out.append(
            {
                **row,
                "v3_3_bucket": candidate_bucket(row),
                PRED_SCORE_KEY: harm_score,
                **{f"route_{key}": value for key, value in route_pattern(row, num_layers=num_layers).items()},
            }
        )
    return out


def quantiles(values: list[float], points: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9, 0.99)) -> list[float | None]:
    if not values:
        return [None for _ in points]
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return [sorted_values[0] for _ in points]
    out: list[float] = []
    for point in points:
        position = float(point) * float(len(sorted_values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(sorted_values) - 1)
        frac = position - float(lower)
        out.append(sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac)
    return out


def _numeric_summary(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for key in keys:
        values = [_as_float(row, key) for row in rows if row.get(key) is not None]
        summary[key] = {
            "mean": None if not values else mean(values),
            "quantiles_25_50_75_90_99": quantiles(values),
        }
    return summary


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        PRED_SAFE_KEY,
        PRED_REGRESSION_KEY,
        PRED_PRESERVE_KEY,
        PRED_DELTA_KEY,
        PRED_SCORE_KEY,
        "route_on_count",
        "route_transition_count",
        "route_early_on_count",
        "route_middle_on_count",
        "route_late_on_count",
    ]
    grouped_ids = {str(row["id"]) for row in rows}
    return {
        "num_rows": len(rows),
        "num_groups": len(grouped_ids),
        "numeric": _numeric_summary(rows, keys),
    }


def _increment_nested_count(target: dict[str, Any], keys: tuple[str, ...], amount: int = 1) -> None:
    current: dict[str, Any] = target
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = int(current.get(keys[-1], 0)) + int(amount)


def passes_loose_accuracy_gate(
    row: dict[str, Any],
    *,
    min_safe_prob: float = 0.5,
    max_regression_prob: float = 0.3,
    min_delta_pred: float = -0.05,
) -> bool:
    return (
        _as_float(row, PRED_SAFE_KEY) >= float(min_safe_prob)
        and _as_float(row, PRED_REGRESSION_KEY) <= float(max_regression_prob)
        and _as_float(row, PRED_DELTA_KEY) >= float(min_delta_pred)
    )


def safe_switch_false_negative_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return safe-switch candidates missed even by the loosest v3.3 switch gate."""

    return [row for row in rows if _as_bool(row, "safe_switch") and not passes_loose_accuracy_gate(row)]


def _best_by_pred_score(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            _as_float(row, PRED_SCORE_KEY),
            _as_float(row, PRED_DELTA_KEY),
            _as_float(row, PRED_SAFE_KEY),
            -_as_float(row, PRED_REGRESSION_KEY, 1.0),
            -candidate_on(row),
            -candidate_rank(row),
        ),
    )


def _compact_candidate(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "candidate_index": int(row.get("candidate_index", -1)),
        "decoder_rank": candidate_rank(row),
        "bucket": str(row.get("v3_3_bucket", candidate_bucket(row))),
        "safe_switch": _as_bool(row, "safe_switch"),
        "regression": _as_bool(row, "regression") or _as_bool(row, "is_regression"),
        "cost_only_preserve": _as_bool(row, "cost_only_preserve"),
        "candidate_score": _as_float(row, "candidate_score"),
        "full_score": _as_float(row, "full_score"),
        "delta_q": _as_float(row, "delta_q", _as_float(row, "delta_score")),
        "safe_prob": _as_float(row, PRED_SAFE_KEY),
        "regression_prob": _as_float(row, PRED_REGRESSION_KEY),
        "preserve_prob": _as_float(row, PRED_PRESERVE_KEY),
        "delta_pred": _as_float(row, PRED_DELTA_KEY),
        "harm_score": _as_float(row, PRED_SCORE_KEY),
        "on_count": candidate_on(row),
        "transition_count": int(row.get("transition_count", row.get("route_transition_count", 0))),
        "layers_one_based": list(row.get("layers_one_based", [])),
    }


def matched_group_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare best safe switch against matched non-safe candidates per group."""

    contrasts: list[dict[str, Any]] = []
    for sample_id, group in group_rows_by_id(rows).items():
        safe_rows = [row for row in group if _as_bool(row, "safe_switch")]
        if not safe_rows:
            continue
        regression_rows = [row for row in group if candidate_bucket(row) == "regression"]
        cost_only_rows = [row for row in group if candidate_bucket(row) == "cost_only_preserve"]
        non_safe_rows = [row for row in group if not _as_bool(row, "safe_switch")]
        best_safe = _best_by_pred_score(safe_rows)
        best_regression = _best_by_pred_score(regression_rows)
        best_cost_only = _best_by_pred_score(cost_only_rows)
        best_non_safe = _best_by_pred_score(non_safe_rows)
        group_by_score = sorted(
            group,
            key=lambda row: (
                _as_float(row, PRED_SCORE_KEY),
                _as_float(row, PRED_DELTA_KEY),
                _as_float(row, PRED_SAFE_KEY),
            ),
            reverse=True,
        )
        best_safe_rank = 1 + min(idx for idx, row in enumerate(group_by_score) if row is best_safe)
        best_safe_score = _as_float(best_safe or {}, PRED_SCORE_KEY)
        best_regression_score = None if best_regression is None else _as_float(best_regression, PRED_SCORE_KEY)
        best_cost_score = None if best_cost_only is None else _as_float(best_cost_only, PRED_SCORE_KEY)
        best_non_safe_score = None if best_non_safe is None else _as_float(best_non_safe, PRED_SCORE_KEY)
        contrasts.append(
            {
                "id": sample_id,
                "benchmark": group[0].get("benchmark"),
                "num_candidates": len(group),
                "safe_switch_count": len(safe_rows),
                "regression_count": len(regression_rows),
                "cost_only_preserve_count": len(cost_only_rows),
                "best_safe_rank_by_pred_score": best_safe_rank,
                "safe_passes_loose_gate_count": sum(1 for row in safe_rows if passes_loose_accuracy_gate(row)),
                "safe_outranked_by_regression": bool(
                    best_regression_score is not None and best_regression_score > best_safe_score
                ),
                "safe_outranked_by_cost_only": bool(best_cost_score is not None and best_cost_score > best_safe_score),
                "safe_outranked_by_any_non_safe": bool(
                    best_non_safe_score is not None and best_non_safe_score > best_safe_score
                ),
                "best_safe_minus_best_regression_score": None
                if best_regression_score is None
                else best_safe_score - best_regression_score,
                "best_safe_minus_best_cost_only_score": None
                if best_cost_score is None
                else best_safe_score - best_cost_score,
                "best_safe_minus_best_non_safe_score": None
                if best_non_safe_score is None
                else best_safe_score - best_non_safe_score,
                "best_safe": _compact_candidate(best_safe),
                "best_regression": _compact_candidate(best_regression),
                "best_cost_only": _compact_candidate(best_cost_only),
                "best_non_safe": _compact_candidate(best_non_safe),
            }
        )
    return contrasts


def summarize_contrasts(contrasts: list[dict[str, Any]]) -> dict[str, Any]:
    by_benchmark: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in contrasts:
        benchmark = str(row.get("benchmark", "unknown"))
        by_benchmark[benchmark]["safe_switch_groups"] += 1
        if bool(row.get("safe_outranked_by_regression", False)):
            by_benchmark[benchmark]["safe_outranked_by_regression"] += 1
        if bool(row.get("safe_outranked_by_cost_only", False)):
            by_benchmark[benchmark]["safe_outranked_by_cost_only"] += 1
        if bool(row.get("safe_outranked_by_any_non_safe", False)):
            by_benchmark[benchmark]["safe_outranked_by_any_non_safe"] += 1
        if int(row.get("safe_passes_loose_gate_count", 0)) > 0:
            by_benchmark[benchmark]["groups_with_loose_gate_safe"] += 1

    score_gap_keys = [
        "best_safe_minus_best_regression_score",
        "best_safe_minus_best_cost_only_score",
        "best_safe_minus_best_non_safe_score",
    ]
    score_gaps = {
        key: {
            "mean": None
            if not [row[key] for row in contrasts if row.get(key) is not None]
            else mean(float(row[key]) for row in contrasts if row.get(key) is not None),
            "quantiles_25_50_75_90_99": quantiles(
                [float(row[key]) for row in contrasts if row.get(key) is not None]
            ),
        }
        for key in score_gap_keys
    }
    return {
        "safe_switch_groups": len(contrasts),
        "safe_outranked_by_regression_groups": sum(
            1 for row in contrasts if bool(row.get("safe_outranked_by_regression", False))
        ),
        "safe_outranked_by_cost_only_groups": sum(
            1 for row in contrasts if bool(row.get("safe_outranked_by_cost_only", False))
        ),
        "safe_outranked_by_any_non_safe_groups": sum(
            1 for row in contrasts if bool(row.get("safe_outranked_by_any_non_safe", False))
        ),
        "groups_with_any_safe_passing_loose_gate": sum(
            1 for row in contrasts if int(row.get("safe_passes_loose_gate_count", 0)) > 0
        ),
        "score_gaps": score_gaps,
        "by_benchmark": {key: dict(value) for key, value in sorted(by_benchmark.items())},
    }


def summarize_loose_gate_by_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for row in rows:
        bucket = str(row.get("v3_3_bucket", candidate_bucket(row)))
        item = summary.setdefault(bucket, {"rows": 0, "groups": set(), "loose_gate_pass_rows": 0, "loose_gate_pass_groups": set()})
        item["rows"] += 1
        item["groups"].add(str(row["id"]))
        if passes_loose_accuracy_gate(row):
            item["loose_gate_pass_rows"] += 1
            item["loose_gate_pass_groups"].add(str(row["id"]))
    return {
        bucket: {
            "rows": int(item["rows"]),
            "groups": len(item["groups"]),
            "loose_gate_pass_rows": int(item["loose_gate_pass_rows"]),
            "loose_gate_pass_groups": len(item["loose_gate_pass_groups"]),
        }
        for bucket, item in sorted(summary.items())
    }


def summarize_top_predicted_bucket_in_safe_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_bucket: dict[str, int] = defaultdict(int)
    by_benchmark: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    safe_groups = 0
    for _, group in group_rows_by_id(rows).items():
        if not any(_as_bool(row, "safe_switch") for row in group):
            continue
        safe_groups += 1
        top = _best_by_pred_score(group)
        if top is None:
            continue
        bucket = str(top.get("v3_3_bucket", candidate_bucket(top)))
        benchmark = str(top.get("benchmark", "unknown"))
        by_bucket[bucket] += 1
        by_benchmark[benchmark]["safe_switch_groups"] += 1
        by_benchmark[benchmark][bucket] += 1
    return {
        "safe_switch_groups": safe_groups,
        "top_predicted_bucket_counts": dict(sorted(by_bucket.items())),
        "by_benchmark": {key: dict(value) for key, value in sorted(by_benchmark.items())},
    }


def analyze_v3_3_failure_rows(rows: list[dict[str, Any]], *, split_name: str) -> dict[str, Any]:
    bucket_counts: dict[str, int] = defaultdict(int)
    benchmark_bucket_counts: dict[str, Any] = {}
    bucket_summaries: dict[str, Any] = {}
    benchmark_bucket_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rows_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = str(row.get("v3_3_bucket", candidate_bucket(row)))
        benchmark = str(row.get("benchmark", "unknown"))
        bucket_counts[bucket] += 1
        _increment_nested_count(benchmark_bucket_counts, (benchmark, bucket))
        rows_by_bucket[bucket].append(row)
        benchmark_bucket_rows[(benchmark, bucket)].append(row)

    for bucket, bucket_rows in sorted(rows_by_bucket.items()):
        bucket_summaries[bucket] = summarize_rows(bucket_rows)

    per_benchmark_bucket_summary = {
        benchmark: {
            bucket: summarize_rows(group_rows)
            for (bench, bucket), group_rows in sorted(benchmark_bucket_rows.items())
            if bench == benchmark
        }
        for benchmark in sorted({benchmark for benchmark, _ in benchmark_bucket_rows})
    }

    safe_rows = rows_by_bucket.get("safe_switch", [])
    false_negative_rows = safe_switch_false_negative_rows(rows)
    contrasts = matched_group_contrasts(rows)
    return {
        "split": split_name,
        "num_rows": len(rows),
        "num_groups": len({str(row["id"]) for row in rows}),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "benchmark_bucket_counts": benchmark_bucket_counts,
        "bucket_summaries": bucket_summaries,
        "per_benchmark_bucket_summary": per_benchmark_bucket_summary,
        "safe_switch_false_negative_summary": {
            "safe_switch_candidates": len(safe_rows),
            "safe_switch_groups": len({str(row["id"]) for row in safe_rows}),
            "missed_by_selected_fallback_policy_candidates": len(safe_rows),
            "missed_by_selected_fallback_policy_groups": len({str(row["id"]) for row in safe_rows}),
            "missed_even_by_loose_gate_candidates": len(false_negative_rows),
            "missed_even_by_loose_gate_groups": len({str(row["id"]) for row in false_negative_rows}),
            "loose_gate_pass_candidates": len(safe_rows) - len(false_negative_rows),
            "loose_gate_pass_groups": len(
                {
                    str(row["id"])
                    for row in safe_rows
                    if passes_loose_accuracy_gate(row)
                }
            ),
        },
        "loose_gate_by_bucket": summarize_loose_gate_by_bucket(rows),
        "top_predicted_bucket_in_safe_groups": summarize_top_predicted_bucket_in_safe_groups(rows),
        "contrast_summary": summarize_contrasts(contrasts),
    }
