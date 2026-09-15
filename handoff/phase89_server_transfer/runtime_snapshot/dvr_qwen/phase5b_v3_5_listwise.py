"""Fallback-inclusive listwise utilities for Phase 5B v3.5."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from dvr_qwen.phase5b_v3_utility import (
    _selected_v3_row,
    build_v3_1_selector_examples,
    full_qwen_fallback_row,
)
from dvr_qwen.route_selector import (
    NUM_LAYERS,
    candidate_on,
    candidate_rank,
    group_rows_by_id,
)


DEFAULT_V3_5_UTILITY_CONFIG: dict[str, float] = {
    "lambda_on_safe": 0.002,
    "lambda_transition_safe": 0.001,
    "regression_penalty": 4.0,
    "cost_only_preserve_penalty": 0.05,
    "wrong_preserve_penalty": 0.5,
    "other_negative_penalty": 0.25,
}


def _fallback_row(group: list[dict[str, Any]], *, num_layers: int) -> dict[str, Any]:
    row = full_qwen_fallback_row(group, num_layers=num_layers)
    return {
        **row,
        "budget_count": int(num_layers),
        "decoder_score": 0.0,
        "delta_q": 0.0,
        "improve": False,
        "fix": False,
        "regression": False,
        "preserve": True,
        "safe_switch": False,
        "cost_only_preserve": False,
        "live_target_positive": False,
        "v3_1_accuracy_utility": 0.0,
        "v3_1_balanced_utility": 0.0,
        "v3_5_listwise_utility": 0.0,
    }


def v3_5_listwise_utility(
    row: dict[str, Any],
    *,
    num_layers: int = NUM_LAYERS,
    utility_config: dict[str, float] | None = None,
) -> float:
    """Return the v3.5 group-ranking utility for one non-fallback candidate."""

    config = dict(DEFAULT_V3_5_UTILITY_CONFIG if utility_config is None else utility_config)
    if bool(row.get("safe_switch", False)):
        on_norm = float(candidate_on(row)) / float(num_layers)
        transition_norm = float(row.get("transition_count", 0.0)) / max(float(num_layers - 1), 1.0)
        return (
            float(row.get("v3_1_accuracy_utility", row.get("delta_q", 0.0)))
            + float(config.get("lambda_on_safe", 0.0)) * (1.0 - on_norm)
            - float(config.get("lambda_transition_safe", 0.0)) * transition_norm
        )
    if bool(row.get("regression", False)):
        return -float(config.get("regression_penalty", 4.0)) - abs(float(row.get("delta_q", 0.0)))
    if bool(row.get("cost_only_preserve", False)):
        return -float(config.get("cost_only_preserve_penalty", 0.05))
    if bool(row.get("preserve", False)):
        return -float(config.get("other_negative_penalty", 0.25))
    return -float(config.get("wrong_preserve_penalty", 0.5))


def build_v3_5_fallback_inclusive_rows(
    rows: list[dict[str, Any]],
    *,
    num_layers: int = NUM_LAYERS,
    utility_config: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Return candidate rows with an explicit fallback row first in each group."""

    grouped = group_rows_by_id(rows)
    inclusive: list[dict[str, Any]] = []
    for sample_id in sorted(grouped):
        group = grouped[sample_id]
        inclusive.append(_fallback_row(group, num_layers=num_layers))
        for row in group:
            inclusive.append(
                {
                    **row,
                    "selected_full_qwen_fallback": False,
                    "v3_5_listwise_utility": v3_5_listwise_utility(
                        row,
                        num_layers=num_layers,
                        utility_config=utility_config,
                    ),
                }
            )
    return inclusive


def build_v3_5_selector_examples(
    rows: list[dict[str, Any]],
    layer_scores_by_id: dict[str, torch.Tensor],
    *,
    sample_scalars_by_id: dict[str, torch.Tensor] | None = None,
    num_layers: int = NUM_LAYERS,
    utility_config: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Build deployable v3.5 features and utilities with fallback rows included."""

    inclusive_rows = build_v3_5_fallback_inclusive_rows(
        rows,
        num_layers=num_layers,
        utility_config=utility_config,
    )
    features, utilities, metadata, feature_names = build_v3_1_selector_examples(
        inclusive_rows,
        layer_scores_by_id,
        sample_scalars_by_id=sample_scalars_by_id,
        num_layers=num_layers,
        utility_field="v3_5_listwise_utility",
    )
    for row, item in zip(inclusive_rows, metadata, strict=True):
        item["selected_full_qwen_fallback"] = bool(row.get("selected_full_qwen_fallback", False))
        item["v3_5_listwise_utility"] = float(row.get("v3_5_listwise_utility", 0.0))
    return features, utilities, metadata, feature_names, inclusive_rows


def select_v3_5_with_policy(
    scores: torch.Tensor,
    metadata: list[dict[str, Any]],
    *,
    num_layers: int = NUM_LAYERS,
    lambda_on: float = 0.0,
    lambda_transition: float = 0.0,
    margin: float = 0.0,
    min_switch_score: float | None = None,
    max_candidate_rank: int | None = None,
    force_full_qwen_fallback: bool = False,
    policy_name: str = "route_verifier_v3_5",
) -> list[dict[str, Any]]:
    """Select a route by comparing candidates against the explicit fallback row."""

    if int(scores.numel()) != len(metadata):
        raise ValueError(f"scores length {scores.numel()} != metadata length {len(metadata)}")
    grouped: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for score, item in zip(scores.float().tolist(), metadata, strict=True):
        grouped[str(item["id"])].append((float(score), item))

    selected_rows: list[dict[str, Any]] = []
    for _, items in sorted(grouped.items()):
        fallback_items = [pair for pair in items if bool(pair[1].get("selected_full_qwen_fallback", False))]
        if len(fallback_items) != 1:
            raise ValueError(f"group {items[0][1].get('id')} must contain exactly one fallback row")
        fallback_score, fallback = fallback_items[0]
        if force_full_qwen_fallback:
            selected_rows.append(
                _selected_v3_row(
                    policy=policy_name,
                    selected=fallback,
                    selector_logit=fallback_score,
                    final_score=fallback_score,
                    selected_full_qwen_fallback=True,
                )
            )
            continue

        adjusted: list[tuple[float, float, dict[str, Any]]] = []
        for score, item in items:
            if bool(item.get("selected_full_qwen_fallback", False)):
                continue
            if max_candidate_rank is not None and candidate_rank(item) > int(max_candidate_rank):
                continue
            final_score = (
                float(score)
                - float(lambda_on) * float(item.get("on_count", 0.0))
                - float(lambda_transition) * float(item.get("transition_count", 0.0))
            )
            adjusted.append((final_score, float(score), item))
        if not adjusted:
            selected_rows.append(
                _selected_v3_row(
                    policy=policy_name,
                    selected=fallback,
                    selector_logit=fallback_score,
                    final_score=fallback_score,
                    selected_full_qwen_fallback=True,
                )
            )
            continue

        best_final, best_score, best = max(
            adjusted,
            key=lambda triple: (
                triple[0],
                triple[1],
                float(triple[2].get("delta_score", triple[2].get("delta_q", 0.0))),
                -float(triple[2].get("on_count", 0.0)),
                -candidate_rank(triple[2]),
            ),
        )
        should_switch = (best_final - fallback_score) >= float(margin)
        if min_switch_score is not None:
            should_switch = should_switch and best_score >= float(min_switch_score)
        selected = best if should_switch else fallback
        selected_rows.append(
            _selected_v3_row(
                policy=policy_name,
                selected=selected,
                selector_logit=best_score if should_switch else fallback_score,
                final_score=best_final if should_switch else fallback_score,
                selected_full_qwen_fallback=not should_switch,
            )
        )
    return selected_rows


def select_v3_5_oracle_rows(
    inclusive_rows: list[dict[str, Any]],
    *,
    policy_name: str = "v3_5_fallback_inclusive_oracle",
) -> list[dict[str, Any]]:
    """Select the highest v3.5 utility row from each fallback-inclusive group."""

    selected_rows: list[dict[str, Any]] = []
    for _, group in sorted(group_rows_by_id(inclusive_rows).items()):
        selected = max(
            group,
            key=lambda row: (
                float(row.get("v3_5_listwise_utility", 0.0)),
                float(row.get("candidate_score", 0.0)),
                float(row.get("delta_q", row.get("delta_score", 0.0))),
                -candidate_on(row),
                -candidate_rank(row),
                -int(row.get("candidate_index", 0)),
            ),
        )
        selected_rows.append(
            _selected_v3_row(
                policy=policy_name,
                selected=selected,
                selector_logit=None,
                final_score=float(selected.get("v3_5_listwise_utility", 0.0)),
                selected_full_qwen_fallback=bool(selected.get("selected_full_qwen_fallback", False)),
            )
        )
    return selected_rows


def _quantiles(values: list[float]) -> list[float]:
    if not values:
        return []
    tensor = torch.tensor(values, dtype=torch.float32)
    return [float(value) for value in torch.quantile(tensor, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])).tolist()]


def score_gap_diagnostics(scores: torch.Tensor, metadata: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize candidate score gaps relative to the explicit fallback."""

    if int(scores.numel()) != len(metadata):
        raise ValueError(f"scores length {scores.numel()} != metadata length {len(metadata)}")
    grouped: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for score, item in zip(scores.float().tolist(), metadata, strict=True):
        grouped[str(item["id"])].append((float(score), item))

    all_gaps: list[float] = []
    safe_gaps: list[float] = []
    regression_gaps: list[float] = []
    cost_only_gaps: list[float] = []
    groups_with_safe = 0
    safe_above = 0
    top_safe_groups = 0
    for _, items in sorted(grouped.items()):
        fallback_items = [pair for pair in items if bool(pair[1].get("selected_full_qwen_fallback", False))]
        if len(fallback_items) != 1:
            raise ValueError(f"group {items[0][1].get('id')} must contain exactly one fallback row")
        fallback_score = fallback_items[0][0]
        nonfallback = [(score, item) for score, item in items if not bool(item.get("selected_full_qwen_fallback", False))]
        if not nonfallback:
            continue
        best_score, best_item = max(nonfallback, key=lambda pair: pair[0])
        if bool(best_item.get("safe_switch", False)):
            top_safe_groups += 1
        group_has_safe = False
        for score, item in nonfallback:
            gap = float(score) - float(fallback_score)
            all_gaps.append(gap)
            if bool(item.get("safe_switch", False)):
                group_has_safe = True
                safe_gaps.append(gap)
                safe_above += int(gap > 0.0)
            if bool(item.get("regression", False)):
                regression_gaps.append(gap)
            if bool(item.get("cost_only_preserve", False)):
                cost_only_gaps.append(gap)
        groups_with_safe += int(group_has_safe)
        _ = best_score

    return {
        "groups": len(grouped),
        "groups_with_safe_switch": groups_with_safe,
        "top_nonfallback_is_safe_switch_groups": top_safe_groups,
        "candidate_gap_quantiles": _quantiles(all_gaps),
        "safe_switch_candidates": len(safe_gaps),
        "safe_switches_above_fallback": safe_above,
        "safe_switch_gap_quantiles": _quantiles(safe_gaps),
        "regression_candidates": len(regression_gaps),
        "regression_gap_quantiles": _quantiles(regression_gaps),
        "cost_only_preserve_candidates": len(cost_only_gaps),
        "cost_only_preserve_gap_quantiles": _quantiles(cost_only_gaps),
    }
