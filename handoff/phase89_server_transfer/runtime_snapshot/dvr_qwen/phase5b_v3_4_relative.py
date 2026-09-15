"""Relative route features for Phase 5B v3.4."""

from __future__ import annotations

from typing import Any

import torch

from dvr_qwen.phase5b_v3_utility import (
    build_v3_1_selector_examples,
    full_qwen_fallback_row,
)
from dvr_qwen.route_selector import (
    NUM_LAYERS,
    assert_no_feature_leakage,
    candidate_rank,
    group_rows_by_id,
)


RELATIVE_PREFIXES = (
    "candidate",
    "delta_fallback",
    "abs_delta_fallback",
    "delta_group_mean",
    "delta_group_max",
    "delta_group_min",
    "delta_decoder_top1",
)


def _v3_1_fallback_row(group: list[dict[str, Any]], *, num_layers: int) -> dict[str, Any]:
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
    }


def fallback_rows_for_groups(rows: list[dict[str, Any]], *, num_layers: int = NUM_LAYERS) -> list[dict[str, Any]]:
    """Build one all-`VISUAL_ON` full-Qwen fallback row per sample group."""

    return [
        _v3_1_fallback_row(group, num_layers=num_layers)
        for _, group in sorted(group_rows_by_id(rows).items())
    ]


def _relative_feature_names(feature_names: list[str]) -> list[str]:
    names = [f"{prefix}__{name}" for prefix in RELATIVE_PREFIXES for name in feature_names]
    assert_no_feature_leakage(names)
    return names


def build_relative_features(
    candidate_features: torch.Tensor,
    candidate_metadata: list[dict[str, Any]],
    fallback_features_by_id: dict[str, torch.Tensor],
    feature_names: list[str],
) -> tuple[torch.Tensor, list[str]]:
    """Augment deployable candidate features with same-sample relative views.

    The returned feature vector contains the absolute candidate features and
    differences to the full-Qwen fallback, same-group mean/max/min candidate
    features, and decoder-top1 sibling candidate. These references are all
    deployable because they depend only on proposal features and route masks.
    """

    if int(candidate_features.shape[0]) != len(candidate_metadata):
        raise ValueError(f"feature rows {candidate_features.shape[0]} != metadata rows {len(candidate_metadata)}")
    if int(candidate_features.shape[1]) != len(feature_names):
        raise ValueError(f"feature cols {candidate_features.shape[1]} != feature name count {len(feature_names)}")
    assert_no_feature_leakage(feature_names)

    grouped_indices: dict[str, list[int]] = {}
    for idx, item in enumerate(candidate_metadata):
        grouped_indices.setdefault(str(item["id"]), []).append(idx)

    out: list[torch.Tensor] = []
    for idx, item in enumerate(candidate_metadata):
        sample_id = str(item["id"])
        if sample_id not in fallback_features_by_id:
            raise ValueError(f"missing fallback features for {sample_id}")
        group_indices = grouped_indices[sample_id]
        group_features = candidate_features[group_indices].float()
        group_metadata = [candidate_metadata[group_idx] for group_idx in group_indices]
        top1_local_index = min(
            range(len(group_metadata)),
            key=lambda local_idx: (
                candidate_rank(group_metadata[local_idx]),
                int(group_metadata[local_idx].get("candidate_index", 0)),
            ),
        )
        candidate = candidate_features[idx].float()
        fallback = fallback_features_by_id[sample_id].float()
        group_mean = group_features.mean(dim=0)
        group_max = group_features.max(dim=0).values
        group_min = group_features.min(dim=0).values
        decoder_top1 = group_features[top1_local_index]
        out.append(
            torch.cat(
                [
                    candidate,
                    candidate - fallback,
                    (candidate - fallback).abs(),
                    candidate - group_mean,
                    candidate - group_max,
                    candidate - group_min,
                    candidate - decoder_top1,
                ]
            )
        )
    return torch.stack(out), _relative_feature_names(feature_names)


def build_v3_4_relative_selector_examples(
    rows: list[dict[str, Any]],
    layer_scores_by_id: dict[str, torch.Tensor],
    *,
    sample_scalars_by_id: dict[str, torch.Tensor] | None = None,
    num_layers: int = NUM_LAYERS,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], list[str], list[str]]:
    """Build v3.1 harm labels with v3.4 relative deployable features."""

    fallback_rows = fallback_rows_for_groups(rows, num_layers=num_layers)
    combined_rows = [*rows, *fallback_rows]
    combined_features, _, combined_metadata, base_feature_names = build_v3_1_selector_examples(
        combined_rows,
        layer_scores_by_id,
        sample_scalars_by_id=sample_scalars_by_id,
        num_layers=num_layers,
    )
    candidate_count = len(rows)
    candidate_features = combined_features[:candidate_count]
    candidate_metadata = combined_metadata[:candidate_count]
    fallback_features = combined_features[candidate_count:]
    fallback_metadata = combined_metadata[candidate_count:]
    fallback_features_by_id = {
        str(item["id"]): fallback_features[idx]
        for idx, item in enumerate(fallback_metadata)
    }
    relative_features, relative_feature_names = build_relative_features(
        candidate_features,
        candidate_metadata,
        fallback_features_by_id,
        base_feature_names,
    )
    utilities = torch.tensor(
        [float(row.get("v3_1_accuracy_utility", 0.0)) for row in rows],
        dtype=torch.float32,
    )
    return relative_features, utilities, candidate_metadata, relative_feature_names, base_feature_names
