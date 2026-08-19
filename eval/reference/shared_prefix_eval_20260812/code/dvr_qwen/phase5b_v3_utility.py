"""Full-Qwen-relative utility labels for Phase 5B v3."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from dvr_qwen.route_selector import (
    NUM_LAYERS,
    base_row_for_group,
    build_route_selector_examples,
    candidate_on,
    candidate_rank,
    group_rows_by_id,
    normalized_layers,
    route_mask_from_layers,
    transition_count,
)


DEFAULT_V3_UTILITY_CONFIG: dict[str, float] = {
    "w_delta": 1.0,
    "w_improvement": 1.0,
    "w_fix": 2.0,
    "w_regression": 4.0,
    "lambda_on": 0.002,
    "lambda_transition": 0.001,
}

DEFAULT_V3_1_HARM_CONFIG: dict[str, float] = {
    "w_delta_q": 1.0,
    "w_improve": 1.0,
    "w_fix": 2.0,
    "w_regression": 4.0,
    "lambda_on_balanced": 0.002,
    "lambda_transition_balanced": 0.001,
}


def _as_float(row: dict[str, Any], key: str, *, default: float | None = None) -> float:
    value = row.get(key, default)
    if value is None:
        raise ValueError(f"row {row.get('id', '<missing-id>')} is missing required numeric field {key!r}")
    return float(value)


def _group_reference_value(rows: list[dict[str, Any]], key: str, *, default: float | None, eps: float) -> float:
    values = [_as_float(row, key, default=default) for row in rows]
    first = values[0]
    for value in values[1:]:
        if abs(value - first) > eps:
            raise ValueError(f"group {rows[0].get('id')} has inconsistent {key}: {first} vs {value}")
    return first


def _candidate_transition_count(row: dict[str, Any], *, num_layers: int) -> int:
    layers = normalized_layers([int(layer) for layer in row.get("layers_one_based", [])], num_layers=num_layers)
    return transition_count(route_mask_from_layers(layers, num_layers=num_layers))


def build_v3_1_harm_aware_rows(
    rows: list[dict[str, Any]],
    *,
    num_layers: int = NUM_LAYERS,
    utility_config: dict[str, float] | None = None,
    eps: float = 1e-8,
    improve_eps: float = 1e-8,
) -> list[dict[str, Any]]:
    """Attach v3.1 harm-aware full-Qwen-relative labels to candidate rows.

    `safe_switch` is deliberately narrower than "same answer with fewer
    layers": it only marks candidates that improve over full Qwen or fix a
    full-Qwen-wrong row, and excludes cost-only preserving routes.
    """

    if not rows:
        raise ValueError("no candidate rows were provided")
    config = dict(DEFAULT_V3_1_HARM_CONFIG if utility_config is None else utility_config)
    grouped = group_rows_by_id(rows)
    out: list[dict[str, Any]] = []

    for sample_id in sorted(grouped):
        group = grouped[sample_id]
        full_score = _group_reference_value(group, "full_score", default=None, eps=eps)
        target_score = _group_reference_value(group, "target_score", default=1.0, eps=eps)
        full_qwen_reaches_target = full_score >= target_score - eps
        full_qwen_wrong = not full_qwen_reaches_target

        for row in group:
            candidate_score = _as_float(row, "candidate_score", default=None)
            delta_q = candidate_score - full_score
            candidate_reaches_target = candidate_score >= target_score - eps
            improve = candidate_score > full_score + improve_eps
            fix = full_qwen_wrong and candidate_reaches_target
            regression = full_qwen_reaches_target and candidate_score < full_score - eps
            preserve = candidate_score >= full_score - eps
            safe_switch = bool((fix or improve) and not regression)
            cost_only_preserve = bool(preserve and not safe_switch and not regression)
            on_count = candidate_on(row)
            transitions = _candidate_transition_count(row, num_layers=num_layers)
            on_norm = float(on_count) / float(num_layers)
            transition_norm = float(transitions) / max(float(num_layers - 1), 1.0)
            accuracy_utility = (
                float(config.get("w_delta_q", 1.0)) * delta_q
                + float(config.get("w_improve", 1.0)) * float(improve)
                + float(config.get("w_fix", 2.0)) * float(fix)
                - float(config.get("w_regression", 4.0)) * float(regression)
            )
            balanced_utility = (
                accuracy_utility
                + float(config.get("lambda_on_balanced", 0.0)) * (1.0 - on_norm) * float(safe_switch)
                - float(config.get("lambda_transition_balanced", 0.0)) * transition_norm
            )
            out.append(
                {
                    **row,
                    "v3_target_score": target_score,
                    "delta_q": delta_q,
                    "full_qwen_reaches_target": full_qwen_reaches_target,
                    "full_qwen_wrong": full_qwen_wrong,
                    "candidate_reaches_target": candidate_reaches_target,
                    "improve": bool(improve),
                    "fix": bool(fix),
                    "regression": bool(regression),
                    "preserve": bool(preserve),
                    "safe_switch": safe_switch,
                    "cost_only_preserve": cost_only_preserve,
                    "live_target_positive": safe_switch,
                    "v3_1_accuracy_utility": accuracy_utility,
                    "v3_1_balanced_utility": balanced_utility,
                    "v3_1_harm_config": config,
                    "transition_count": transitions,
                }
            )
    return out


def build_v3_utility_rows(
    rows: list[dict[str, Any]],
    *,
    num_layers: int = NUM_LAYERS,
    utility_config: dict[str, float] | None = None,
    eps: float = 1e-8,
) -> list[dict[str, Any]]:
    """Attach full-Qwen-relative utility labels to candidate rows.

    The returned fields are labels/diagnostics only. They are intentionally
    disallowed from deployable selector feature schemas by `route_selector`.
    """

    if not rows:
        raise ValueError("no candidate rows were provided")
    config = dict(DEFAULT_V3_UTILITY_CONFIG if utility_config is None else utility_config)
    grouped = group_rows_by_id(rows)
    out: list[dict[str, Any]] = []

    for sample_id in sorted(grouped):
        group = grouped[sample_id]
        full_score = _group_reference_value(group, "full_score", default=None, eps=eps)
        target_score = _group_reference_value(group, "target_score", default=1.0, eps=eps)
        full_qwen_reaches_target = full_score >= target_score - eps
        full_qwen_wrong = not full_qwen_reaches_target

        for row in group:
            candidate_score = _as_float(row, "candidate_score", default=None)
            delta_score = candidate_score - full_score
            candidate_reaches_target = candidate_score >= target_score - eps
            full_wrong_improvement = full_qwen_wrong and candidate_score > full_score + eps
            full_wrong_fix = full_qwen_wrong and candidate_reaches_target
            full_correct_regression = full_qwen_reaches_target and candidate_score < full_score - eps
            full_correct_preserved = full_qwen_reaches_target and not full_correct_regression
            on_count = candidate_on(row)
            transitions = _candidate_transition_count(row, num_layers=num_layers)
            on_norm = float(on_count) / float(num_layers)
            transition_norm = float(transitions) / max(float(num_layers - 1), 1.0)
            cost_bonus_allowed = full_correct_preserved or full_wrong_improvement
            accuracy_utility = (
                delta_score
                + float(full_wrong_improvement)
                + float(full_wrong_fix)
                - float(full_correct_regression)
            )
            default_utility = (
                float(config.get("w_delta", 1.0)) * delta_score
                + float(config.get("w_improvement", 1.0)) * float(full_wrong_improvement)
                + float(config.get("w_fix", 2.0)) * float(full_wrong_fix)
                - float(config.get("w_regression", 4.0)) * float(full_correct_regression)
                + float(config.get("lambda_on", 0.0)) * (1.0 - on_norm) * float(cost_bonus_allowed)
                - float(config.get("lambda_transition", 0.0)) * transition_norm
            )
            out.append(
                {
                    **row,
                    "v3_target_score": target_score,
                    "delta_score": delta_score,
                    "full_qwen_reaches_target": full_qwen_reaches_target,
                    "full_qwen_wrong": full_qwen_wrong,
                    "candidate_reaches_target": candidate_reaches_target,
                    "full_wrong_improvement": full_wrong_improvement,
                    "full_wrong_fix": full_wrong_fix,
                    "full_correct_preserved": full_correct_preserved,
                    "full_correct_regression": full_correct_regression,
                    "v3_accuracy_utility": accuracy_utility,
                    "v3_default_utility": default_utility,
                    "v3_utility_config": config,
                    "transition_count": transitions,
                }
            )
    return out


def best_full_relative_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot select from an empty candidate group")
    return max(
        rows,
        key=lambda row: (
            float(row.get("v3_default_utility", 0.0)),
            float(row.get("candidate_score", 0.0)),
            float(row.get("delta_score", 0.0)),
            -candidate_on(row),
            -candidate_rank(row),
            -int(row.get("candidate_index", 0)),
        ),
    )


def full_qwen_fallback_row(group: list[dict[str, Any]], *, num_layers: int = NUM_LAYERS) -> dict[str, Any]:
    if not group:
        raise ValueError("cannot build a full-Qwen fallback row from an empty group")
    row = group[0]
    full_score = _as_float(row, "full_score", default=None)
    target_score = _as_float(row, "v3_target_score", default=_as_float(row, "target_score", default=1.0))
    full_qwen_wrong = bool(row.get("full_qwen_wrong", full_score < target_score))
    return {
        "id": row["id"],
        "benchmark": row.get("benchmark"),
        "policy": "full_qwen_fallback",
        "selected_full_qwen_fallback": True,
        "candidate_index": -1,
        "decoder_rank": -1,
        "candidate_score": full_score,
        "full_score": full_score,
        "target_score": target_score,
        "v3_target_score": target_score,
        "delta_score": 0.0,
        "full_qwen_reaches_target": not full_qwen_wrong,
        "full_qwen_wrong": full_qwen_wrong,
        "candidate_reaches_target": not full_qwen_wrong,
        "full_wrong_improvement": False,
        "full_wrong_fix": False,
        "full_correct_preserved": not full_qwen_wrong,
        "full_correct_regression": False,
        "v3_accuracy_utility": 0.0,
        "v3_default_utility": 0.0,
        "layers_one_based": list(range(1, int(num_layers) + 1)),
        "candidate_num_visual_on_layers": int(num_layers),
        "transition_count": 0,
        "sources": ["full_qwen_fallback"],
    }


def select_full_relative_oracle(group: list[dict[str, Any]], *, num_layers: int = NUM_LAYERS) -> dict[str, Any]:
    candidate = best_full_relative_candidate(group)
    if float(candidate.get("v3_default_utility", 0.0)) > 0.0:
        return {**candidate, "policy": "full_relative_oracle", "selected_full_qwen_fallback": False}
    return full_qwen_fallback_row(group, num_layers=num_layers)


def _with_v3_1_fallback_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "delta_q": 0.0,
        "improve": False,
        "fix": False,
        "regression": False,
        "preserve": True,
        "safe_switch": False,
        "cost_only_preserve": False,
        "v3_1_accuracy_utility": 0.0,
        "v3_1_balanced_utility": 0.0,
    }


def best_v3_1_harm_aware_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot select from an empty candidate group")
    return max(
        rows,
        key=lambda row: (
            float(row.get("v3_1_accuracy_utility", 0.0)),
            float(row.get("candidate_score", 0.0)),
            float(row.get("delta_q", row.get("delta_score", 0.0))),
            float(bool(row.get("fix", False))),
            float(bool(row.get("improve", False))),
            -candidate_on(row),
            -candidate_rank(row),
            -int(row.get("candidate_index", 0)),
        ),
    )


def select_v3_1_harm_aware_oracle(group: list[dict[str, Any]], *, num_layers: int = NUM_LAYERS) -> dict[str, Any]:
    candidate = best_v3_1_harm_aware_candidate(group)
    if float(candidate.get("v3_1_accuracy_utility", 0.0)) > 0.0:
        return {
            **candidate,
            "policy": "v3_1_harm_aware_oracle",
            "selected_full_qwen_fallback": False,
        }
    return _with_v3_1_fallback_fields(full_qwen_fallback_row(group, num_layers=num_layers))


def _empty_stats() -> dict[str, float]:
    return {
        "groups": 0.0,
        "candidate_rows": 0.0,
        "full_score_sum": 0.0,
        "target_score_sum": 0.0,
        "full_wrong_groups": 0.0,
        "full_correct_groups": 0.0,
        "groups_with_improvement_candidate": 0.0,
        "groups_with_full_wrong_fix_candidate": 0.0,
        "groups_with_full_correct_regression_candidate": 0.0,
        "decoder_score_sum": 0.0,
        "decoder_delta_sum": 0.0,
        "decoder_on_sum": 0.0,
        "decoder_full_wrong_improvements": 0.0,
        "decoder_full_wrong_fixes": 0.0,
        "decoder_full_correct_regressions": 0.0,
        "oracle_score_sum": 0.0,
        "oracle_delta_sum": 0.0,
        "oracle_on_sum": 0.0,
        "oracle_transition_sum": 0.0,
        "oracle_full_wrong_improvements": 0.0,
        "oracle_full_wrong_fixes": 0.0,
        "oracle_full_correct_regressions": 0.0,
        "oracle_full_qwen_fallbacks": 0.0,
        "positive_delta_candidates": 0.0,
        "full_wrong_fix_candidates": 0.0,
        "full_correct_regression_candidates": 0.0,
    }


def _update_stats(stats: dict[str, float], group: list[dict[str, Any]], selected: dict[str, Any]) -> None:
    default = base_row_for_group(group)
    full_score = _as_float(group[0], "full_score", default=None)
    target_score = _as_float(group[0], "v3_target_score", default=1.0)
    full_wrong = bool(group[0].get("full_qwen_wrong", False))

    stats["groups"] += 1.0
    stats["candidate_rows"] += float(len(group))
    stats["full_score_sum"] += full_score
    stats["target_score_sum"] += target_score
    stats["full_wrong_groups"] += float(full_wrong)
    stats["full_correct_groups"] += float(not full_wrong)
    stats["groups_with_improvement_candidate"] += float(any(bool(row.get("full_wrong_improvement", False)) for row in group))
    stats["groups_with_full_wrong_fix_candidate"] += float(any(bool(row.get("full_wrong_fix", False)) for row in group))
    stats["groups_with_full_correct_regression_candidate"] += float(
        any(bool(row.get("full_correct_regression", False)) for row in group)
    )

    stats["decoder_score_sum"] += _as_float(default, "candidate_score", default=0.0)
    stats["decoder_delta_sum"] += _as_float(default, "delta_score", default=0.0)
    stats["decoder_on_sum"] += float(candidate_on(default))
    stats["decoder_full_wrong_improvements"] += float(bool(default.get("full_wrong_improvement", False)))
    stats["decoder_full_wrong_fixes"] += float(bool(default.get("full_wrong_fix", False)))
    stats["decoder_full_correct_regressions"] += float(bool(default.get("full_correct_regression", False)))

    stats["oracle_score_sum"] += _as_float(selected, "candidate_score", default=0.0)
    stats["oracle_delta_sum"] += _as_float(selected, "delta_score", default=0.0)
    stats["oracle_on_sum"] += float(candidate_on(selected))
    stats["oracle_transition_sum"] += float(selected.get("transition_count", 0.0))
    stats["oracle_full_wrong_improvements"] += float(bool(selected.get("full_wrong_improvement", False)))
    stats["oracle_full_wrong_fixes"] += float(bool(selected.get("full_wrong_fix", False)))
    stats["oracle_full_correct_regressions"] += float(bool(selected.get("full_correct_regression", False)))
    stats["oracle_full_qwen_fallbacks"] += float(bool(selected.get("selected_full_qwen_fallback", False)))

    stats["positive_delta_candidates"] += float(sum(1 for row in group if float(row.get("delta_score", 0.0)) > 0.0))
    stats["full_wrong_fix_candidates"] += float(sum(1 for row in group if bool(row.get("full_wrong_fix", False))))
    stats["full_correct_regression_candidates"] += float(
        sum(1 for row in group if bool(row.get("full_correct_regression", False)))
    )


def _finalize_stats(stats: dict[str, float]) -> dict[str, Any]:
    groups = int(stats["groups"])
    candidate_rows = int(stats["candidate_rows"])
    denom = float(groups) if groups else 1.0
    return {
        "num_groups": groups,
        "num_candidate_rows": candidate_rows,
        "avg_candidates_per_group": candidate_rows / groups if groups else 0.0,
        "avg_full_qwen_score": stats["full_score_sum"] / denom,
        "avg_target_score": stats["target_score_sum"] / denom,
        "full_qwen_wrong_groups": int(stats["full_wrong_groups"]),
        "full_qwen_correct_groups": int(stats["full_correct_groups"]),
        "groups_with_improvement_candidate": int(stats["groups_with_improvement_candidate"]),
        "groups_with_full_wrong_fix_candidate": int(stats["groups_with_full_wrong_fix_candidate"]),
        "groups_with_full_correct_regression_candidate": int(
            stats["groups_with_full_correct_regression_candidate"]
        ),
        "candidate_pool": {
            "positive_delta_candidates": int(stats["positive_delta_candidates"]),
            "full_wrong_fix_candidates": int(stats["full_wrong_fix_candidates"]),
            "full_correct_regression_candidates": int(stats["full_correct_regression_candidates"]),
        },
        "decoder_top1": {
            "avg_candidate_score": stats["decoder_score_sum"] / denom,
            "avg_delta_score": stats["decoder_delta_sum"] / denom,
            "avg_on": stats["decoder_on_sum"] / denom,
            "full_wrong_improvements": int(stats["decoder_full_wrong_improvements"]),
            "full_wrong_fixes": int(stats["decoder_full_wrong_fixes"]),
            "full_correct_regressions": int(stats["decoder_full_correct_regressions"]),
        },
        "full_relative_oracle": {
            "avg_candidate_score": stats["oracle_score_sum"] / denom,
            "avg_delta_score": stats["oracle_delta_sum"] / denom,
            "avg_on": stats["oracle_on_sum"] / denom,
            "avg_transition_count": stats["oracle_transition_sum"] / denom,
            "full_wrong_improvements": int(stats["oracle_full_wrong_improvements"]),
            "full_wrong_fixes": int(stats["oracle_full_wrong_fixes"]),
            "full_correct_regressions": int(stats["oracle_full_correct_regressions"]),
            "full_qwen_fallbacks": int(stats["oracle_full_qwen_fallbacks"]),
        },
    }


def summarize_v3_utility_rows(
    rows: list[dict[str, Any]],
    *,
    num_layers: int = NUM_LAYERS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("no v3 utility rows were provided")
    grouped = group_rows_by_id(rows)
    overall = _empty_stats()
    by_benchmark_stats: dict[str, dict[str, float]] = defaultdict(_empty_stats)
    selected_rows: list[dict[str, Any]] = []

    for sample_id in sorted(grouped):
        group = grouped[sample_id]
        selected = select_full_relative_oracle(group, num_layers=num_layers)
        selected_rows.append(selected)
        _update_stats(overall, group, selected)
        _update_stats(by_benchmark_stats[str(group[0].get("benchmark", "unknown"))], group, selected)

    summary = _finalize_stats(overall)
    summary["by_benchmark"] = {
        benchmark: _finalize_stats(stats) for benchmark, stats in sorted(by_benchmark_stats.items())
    }
    return summary, selected_rows


def summarize_v3_1_harm_aware_rows(
    rows: list[dict[str, Any]],
    *,
    num_layers: int = NUM_LAYERS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("no v3.1 harm-aware rows were provided")
    grouped = group_rows_by_id(rows)
    selected_rows: list[dict[str, Any]] = []
    full_score_sum = 0.0
    target_score_sum = 0.0
    full_wrong_groups = 0
    candidate_counts = {
        "safe_switch_candidates": 0,
        "improve_candidates": 0,
        "fix_candidates": 0,
        "regression_candidates": 0,
        "preserve_candidates": 0,
        "cost_only_preserve_candidates": 0,
        "positive_accuracy_utility_candidates": 0,
    }

    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample_id in sorted(grouped):
        group = grouped[sample_id]
        selected = select_v3_1_harm_aware_oracle(group, num_layers=num_layers)
        selected_rows.append(selected)
        benchmark = str(group[0].get("benchmark", "unknown"))
        by_benchmark[benchmark].append(selected)
        full_score = _as_float(group[0], "full_score", default=None)
        target_score = _as_float(group[0], "v3_target_score", default=_as_float(group[0], "target_score", default=1.0))
        full_score_sum += full_score
        target_score_sum += target_score
        full_wrong_groups += int(bool(group[0].get("full_qwen_wrong", full_score < target_score)))
        candidate_counts["safe_switch_candidates"] += sum(1 for row in group if bool(row.get("safe_switch", False)))
        candidate_counts["improve_candidates"] += sum(1 for row in group if bool(row.get("improve", False)))
        candidate_counts["fix_candidates"] += sum(1 for row in group if bool(row.get("fix", False)))
        candidate_counts["regression_candidates"] += sum(1 for row in group if bool(row.get("regression", False)))
        candidate_counts["preserve_candidates"] += sum(1 for row in group if bool(row.get("preserve", False)))
        candidate_counts["cost_only_preserve_candidates"] += sum(
            1 for row in group if bool(row.get("cost_only_preserve", False))
        )
        candidate_counts["positive_accuracy_utility_candidates"] += sum(
            1 for row in group if float(row.get("v3_1_accuracy_utility", 0.0)) > 0.0
        )

    groups = len(grouped)
    selected_summary = evaluate_v3_selected_rows(selected_rows)
    summary = {
        "num_groups": groups,
        "num_candidate_rows": len(rows),
        "avg_candidates_per_group": len(rows) / groups if groups else 0.0,
        "avg_full_qwen_score": full_score_sum / groups if groups else 0.0,
        "avg_target_score": target_score_sum / groups if groups else 0.0,
        "full_qwen_wrong_groups": full_wrong_groups,
        "full_qwen_correct_groups": groups - full_wrong_groups,
        "candidate_pool": candidate_counts,
        "accuracy_first_oracle": selected_summary,
        "by_benchmark": {
            benchmark: evaluate_v3_selected_rows(items) for benchmark, items in sorted(by_benchmark.items())
        },
    }
    return summary, selected_rows


def build_v3_selector_examples(
    rows: list[dict[str, Any]],
    layer_scores_by_id: dict[str, torch.Tensor],
    *,
    sample_scalars_by_id: dict[str, torch.Tensor] | None = None,
    num_layers: int = NUM_LAYERS,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], list[str]]:
    """Build deployable features with v3 full-Qwen-relative utility labels."""

    features, _, metadata, feature_names = build_route_selector_examples(
        rows,
        layer_scores_by_id,
        sample_scalars_by_id=sample_scalars_by_id,
        num_layers=num_layers,
    )
    if len(metadata) != len(rows):
        raise ValueError("metadata length does not match row length")

    utilities: list[float] = []
    for row, meta in zip(rows, metadata, strict=True):
        utility = _as_float(row, "v3_default_utility", default=None)
        full_score = _as_float(row, "full_score", default=None)
        target_score = _as_float(row, "v3_target_score", default=_as_float(row, "target_score", default=1.0))
        candidate_score = _as_float(row, "candidate_score", default=None)
        full_qwen_wrong = bool(row.get("full_qwen_wrong", full_score < target_score))
        full_wrong_improvement = bool(row.get("full_wrong_improvement", False))
        full_wrong_fix = bool(row.get("full_wrong_fix", False))
        full_correct_regression = bool(row.get("full_correct_regression", False))
        live_success = utility > 0.0
        utilities.append(utility)
        meta.update(
            {
                "label": "positive" if live_success else "negative",
                "live_success": live_success,
                "base_live_success": not full_qwen_wrong,
                "is_default": False,
                "is_fix": bool(full_wrong_improvement or full_wrong_fix),
                "is_regression": full_correct_regression,
                "full_score": full_score,
                "target_score": target_score,
                "v3_target_score": target_score,
                "candidate_score": candidate_score,
                "delta_score": _as_float(row, "delta_score", default=candidate_score - full_score),
                "full_qwen_reaches_target": bool(row.get("full_qwen_reaches_target", not full_qwen_wrong)),
                "full_qwen_wrong": full_qwen_wrong,
                "candidate_reaches_target": bool(row.get("candidate_reaches_target", False)),
                "full_wrong_improvement": full_wrong_improvement,
                "full_wrong_fix": full_wrong_fix,
                "full_correct_preserved": bool(row.get("full_correct_preserved", False)),
                "full_correct_regression": full_correct_regression,
                "v3_accuracy_utility": _as_float(row, "v3_accuracy_utility", default=0.0),
                "v3_default_utility": utility,
            }
        )
    return features, torch.tensor(utilities, dtype=torch.float32), metadata, feature_names


def build_v3_1_selector_examples(
    rows: list[dict[str, Any]],
    layer_scores_by_id: dict[str, torch.Tensor],
    *,
    sample_scalars_by_id: dict[str, torch.Tensor] | None = None,
    num_layers: int = NUM_LAYERS,
    utility_field: str = "v3_1_accuracy_utility",
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], list[str]]:
    """Build deployable features with v3.1 harm-aware utility labels."""

    features, _, metadata, feature_names = build_route_selector_examples(
        rows,
        layer_scores_by_id,
        sample_scalars_by_id=sample_scalars_by_id,
        num_layers=num_layers,
    )
    if len(metadata) != len(rows):
        raise ValueError("metadata length does not match row length")

    utilities: list[float] = []
    for row, meta in zip(rows, metadata, strict=True):
        utility = _as_float(row, utility_field, default=None)
        full_score = _as_float(row, "full_score", default=None)
        target_score = _as_float(row, "v3_target_score", default=_as_float(row, "target_score", default=1.0))
        candidate_score = _as_float(row, "candidate_score", default=None)
        full_qwen_wrong = bool(row.get("full_qwen_wrong", full_score < target_score))
        safe_switch = bool(row.get("safe_switch", False))
        fix = bool(row.get("fix", False))
        improve = bool(row.get("improve", False))
        regression = bool(row.get("regression", False))
        utilities.append(utility)
        meta.update(
            {
                "label": "positive" if safe_switch else "negative",
                "live_success": safe_switch,
                "base_live_success": not full_qwen_wrong,
                "is_default": False,
                "is_fix": bool(fix or improve),
                "is_regression": regression,
                "full_score": full_score,
                "target_score": target_score,
                "v3_target_score": target_score,
                "candidate_score": candidate_score,
                "delta_q": _as_float(row, "delta_q", default=candidate_score - full_score),
                "delta_score": _as_float(
                    row,
                    "delta_score",
                    default=_as_float(row, "delta_q", default=candidate_score - full_score),
                ),
                "full_qwen_reaches_target": bool(row.get("full_qwen_reaches_target", not full_qwen_wrong)),
                "full_qwen_wrong": full_qwen_wrong,
                "candidate_reaches_target": bool(row.get("candidate_reaches_target", False)),
                "improve": improve,
                "fix": fix,
                "regression": regression,
                "preserve": bool(row.get("preserve", False)),
                "safe_switch": safe_switch,
                "cost_only_preserve": bool(row.get("cost_only_preserve", False)),
                "v3_1_accuracy_utility": _as_float(row, "v3_1_accuracy_utility", default=0.0),
                "v3_1_balanced_utility": _as_float(row, "v3_1_balanced_utility", default=0.0),
            }
        )
    return features, torch.tensor(utilities, dtype=torch.float32), metadata, feature_names


def _selected_v3_row(
    *,
    policy: str,
    selected: dict[str, Any],
    selector_logit: float | None,
    final_score: float | None,
    selected_full_qwen_fallback: bool,
) -> dict[str, Any]:
    return {
        "id": selected["id"],
        "benchmark": selected.get("benchmark"),
        "policy": policy,
        "selected_full_qwen_fallback": bool(selected_full_qwen_fallback),
        "fallback_used": bool(selected_full_qwen_fallback),
        "candidate_score": _as_float(selected, "candidate_score", default=0.0),
        "full_score": _as_float(selected, "full_score", default=0.0),
        "target_score": _as_float(selected, "target_score", default=1.0),
        "v3_target_score": _as_float(selected, "v3_target_score", default=_as_float(selected, "target_score", default=1.0)),
        "delta_score": _as_float(
            selected,
            "delta_score",
            default=_as_float(selected, "candidate_score", default=0.0) - _as_float(selected, "full_score", default=0.0),
        ),
        "full_qwen_reaches_target": bool(selected.get("full_qwen_reaches_target", False)),
        "full_qwen_wrong": bool(selected.get("full_qwen_wrong", False)),
        "candidate_reaches_target": bool(selected.get("candidate_reaches_target", False)),
        "full_wrong_improvement": bool(selected.get("full_wrong_improvement", False)),
        "full_wrong_fix": bool(selected.get("full_wrong_fix", False)),
        "full_correct_preserved": bool(selected.get("full_correct_preserved", False)),
        "full_correct_regression": bool(selected.get("full_correct_regression", False)),
        "v3_accuracy_utility": _as_float(selected, "v3_accuracy_utility", default=0.0),
        "v3_default_utility": _as_float(selected, "v3_default_utility", default=0.0),
        "delta_q": _as_float(selected, "delta_q", default=_as_float(selected, "delta_score", default=0.0)),
        "improve": bool(selected.get("improve", selected.get("full_wrong_improvement", False))),
        "fix": bool(selected.get("fix", selected.get("full_wrong_fix", False))),
        "regression": bool(selected.get("regression", selected.get("full_correct_regression", False))),
        "preserve": bool(selected.get("preserve", selected.get("full_correct_preserved", False))),
        "safe_switch": bool(selected.get("safe_switch", False)),
        "cost_only_preserve": bool(selected.get("cost_only_preserve", False)),
        "v3_1_accuracy_utility": _as_float(selected, "v3_1_accuracy_utility", default=0.0),
        "v3_1_balanced_utility": _as_float(selected, "v3_1_balanced_utility", default=0.0),
        "candidate_num_visual_on_layers": candidate_on(selected),
        "on_count": candidate_on(selected),
        "transition_count": int(selected.get("transition_count", 0)),
        "decoder_rank": int(selected.get("decoder_rank", -1)),
        "candidate_index": int(selected.get("candidate_index", -1)),
        "layers_one_based": list(selected.get("layers_one_based", [])),
        "sources": list(selected.get("sources", [])),
        "selector_logit": None if selector_logit is None else float(selector_logit),
        "final_score": None if final_score is None else float(final_score),
        "default_final_score": 0.0,
    }


def select_v3_with_policy(
    scores: torch.Tensor,
    metadata: list[dict[str, Any]],
    *,
    num_layers: int = NUM_LAYERS,
    lambda_on: float = 0.0,
    lambda_transition: float = 0.0,
    margin: float = 0.0,
    min_switch_score: float = 0.0,
    max_candidate_rank: int | None = None,
    force_full_qwen_fallback: bool = False,
    policy_name: str = "route_verifier_v3",
) -> list[dict[str, Any]]:
    """Select a sparse route only when predicted utility beats full Qwen."""

    if int(scores.numel()) != len(metadata):
        raise ValueError(f"scores length {scores.numel()} != metadata length {len(metadata)}")
    grouped: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for score, item in zip(scores.float().tolist(), metadata, strict=True):
        grouped[str(item["id"])].append((float(score), item))

    selected_rows: list[dict[str, Any]] = []
    for _, items in sorted(grouped.items()):
        candidates = [item for _, item in items]
        fallback = full_qwen_fallback_row(candidates, num_layers=num_layers)
        if force_full_qwen_fallback:
            selected_rows.append(
                _selected_v3_row(
                    policy=policy_name,
                    selected=fallback,
                    selector_logit=None,
                    final_score=0.0,
                    selected_full_qwen_fallback=True,
                )
            )
            continue
        adjusted: list[tuple[float, float, dict[str, Any]]] = []
        for score, item in items:
            if max_candidate_rank is not None and int(item.get("decoder_rank", 999999)) > int(max_candidate_rank):
                continue
            final_score = float(score) - float(lambda_on) * float(item.get("on_count", 0.0)) - float(
                lambda_transition
            ) * float(item.get("transition_count", 0.0))
            adjusted.append((final_score, float(score), item))
        if not adjusted:
            selected_rows.append(
                _selected_v3_row(
                    policy=policy_name,
                    selected=fallback,
                    selector_logit=None,
                    final_score=0.0,
                    selected_full_qwen_fallback=True,
                )
            )
            continue
        best_final, best_score, best = max(
            adjusted,
            key=lambda triple: (
                triple[0],
                triple[1],
                float(triple[2].get("delta_score", 0.0)),
                -float(triple[2].get("on_count", 0.0)),
                -int(triple[2].get("decoder_rank", 999999)),
            ),
        )
        should_switch = best_final >= float(margin) and best_score >= float(min_switch_score)
        selected = best if should_switch else fallback
        selected_rows.append(
            _selected_v3_row(
                policy=policy_name,
                selected=selected,
                selector_logit=best_score if should_switch else None,
                final_score=best_final if should_switch else 0.0,
                selected_full_qwen_fallback=not should_switch,
            )
        )
    return selected_rows


def _empty_selected_stats() -> dict[str, float]:
    return {
        "rows": 0.0,
        "score_sum": 0.0,
        "full_score_sum": 0.0,
        "delta_sum": 0.0,
        "on_sum": 0.0,
        "transition_sum": 0.0,
        "full_wrong_groups": 0.0,
        "full_correct_groups": 0.0,
        "full_wrong_improvements": 0.0,
        "full_wrong_fixes": 0.0,
        "full_correct_regressions": 0.0,
        "improvements": 0.0,
        "fixes": 0.0,
        "regressions": 0.0,
        "preserves": 0.0,
        "safe_switches": 0.0,
        "cost_only_preserves": 0.0,
        "fallbacks": 0.0,
    }


def _update_selected_stats(stats: dict[str, float], row: dict[str, Any]) -> None:
    full_wrong = bool(row.get("full_qwen_wrong", False))
    stats["rows"] += 1.0
    stats["score_sum"] += _as_float(row, "candidate_score", default=0.0)
    stats["full_score_sum"] += _as_float(row, "full_score", default=0.0)
    stats["delta_sum"] += _as_float(row, "delta_score", default=0.0)
    stats["on_sum"] += float(candidate_on(row))
    stats["transition_sum"] += float(row.get("transition_count", 0.0))
    stats["full_wrong_groups"] += float(full_wrong)
    stats["full_correct_groups"] += float(not full_wrong)
    stats["full_wrong_improvements"] += float(bool(row.get("full_wrong_improvement", False)))
    stats["full_wrong_fixes"] += float(bool(row.get("full_wrong_fix", False)))
    stats["full_correct_regressions"] += float(bool(row.get("full_correct_regression", False)))
    stats["improvements"] += float(bool(row.get("improve", False)))
    stats["fixes"] += float(bool(row.get("fix", False)))
    stats["regressions"] += float(bool(row.get("regression", False)))
    stats["preserves"] += float(bool(row.get("preserve", False)))
    stats["safe_switches"] += float(bool(row.get("safe_switch", False)))
    stats["cost_only_preserves"] += float(bool(row.get("cost_only_preserve", False)))
    stats["fallbacks"] += float(bool(row.get("selected_full_qwen_fallback", False)))


def _finalize_selected_stats(stats: dict[str, float]) -> dict[str, Any]:
    rows = int(stats["rows"])
    denom = float(rows) if rows else 1.0
    return {
        "num_rows": rows,
        "avg_candidate_score": stats["score_sum"] / denom,
        "avg_full_qwen_score": stats["full_score_sum"] / denom,
        "avg_delta_score": stats["delta_sum"] / denom,
        "avg_selected_on": stats["on_sum"] / denom,
        "avg_transition_count": stats["transition_sum"] / denom,
        "full_qwen_wrong_groups": int(stats["full_wrong_groups"]),
        "full_qwen_correct_groups": int(stats["full_correct_groups"]),
        "full_wrong_improvements": int(stats["full_wrong_improvements"]),
        "full_wrong_fixes": int(stats["full_wrong_fixes"]),
        "full_correct_regressions": int(stats["full_correct_regressions"]),
        "improvements": int(stats["improvements"]),
        "fixes": int(stats["fixes"]),
        "regressions": int(stats["regressions"]),
        "preserves": int(stats["preserves"]),
        "safe_switches": int(stats["safe_switches"]),
        "cost_only_preserves": int(stats["cost_only_preserves"]),
        "full_qwen_fallbacks": int(stats["fallbacks"]),
        "switch_rows": rows - int(stats["fallbacks"]),
    }


def evaluate_v3_selected_rows(selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = _empty_selected_stats()
    by_benchmark: dict[str, dict[str, float]] = defaultdict(_empty_selected_stats)
    for row in selected_rows:
        _update_selected_stats(overall, row)
        _update_selected_stats(by_benchmark[str(row.get("benchmark", "unknown"))], row)
    summary = _finalize_selected_stats(overall)
    summary["by_benchmark"] = {
        benchmark: _finalize_selected_stats(stats) for benchmark, stats in sorted(by_benchmark.items())
    }
    return summary
