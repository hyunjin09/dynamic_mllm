"""Multi-head harm-aware route verifier for Phase 5B v3.2."""

from __future__ import annotations

from collections import defaultdict
import random
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from dvr_qwen.phase5b_v3_utility import (
    _selected_v3_row,
    full_qwen_fallback_row,
)
from dvr_qwen.route_selector import (
    NUM_LAYERS,
    combined_utility_loss,
    group_indices_from_metadata,
    pair_weight_matrices,
)


def split_fit_calibration_groups(
    metadata: list[dict[str, Any]],
    *,
    calibration_fraction: float = 0.25,
    seed: int = 0,
) -> dict[str, Any]:
    """Split train groups into disjoint model-fit and policy-calibration groups."""

    if not metadata:
        raise ValueError("metadata must not be empty")
    fraction = float(calibration_fraction)
    if fraction <= 0.0 or fraction >= 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")

    group_benchmark: dict[str, str] = {}
    for item in metadata:
        group_id = str(item["id"])
        benchmark = str(item.get("benchmark", "unknown"))
        previous = group_benchmark.setdefault(group_id, benchmark)
        if previous != benchmark:
            raise ValueError(f"group {group_id} has inconsistent benchmark values: {previous} vs {benchmark}")

    by_benchmark_ids: dict[str, list[str]] = defaultdict(list)
    for group_id, benchmark in group_benchmark.items():
        by_benchmark_ids[benchmark].append(group_id)

    rng = random.Random(int(seed))
    calibration_ids: set[str] = set()
    by_benchmark: dict[str, dict[str, int]] = {}
    for benchmark in sorted(by_benchmark_ids):
        group_ids = sorted(by_benchmark_ids[benchmark])
        rng.shuffle(group_ids)
        if len(group_ids) <= 1:
            calibration_count = 0
        else:
            calibration_count = int(round(len(group_ids) * fraction))
            calibration_count = max(1, min(len(group_ids) - 1, calibration_count))
        benchmark_calibration_ids = set(group_ids[:calibration_count])
        calibration_ids.update(benchmark_calibration_ids)
        by_benchmark[benchmark] = {
            "fit": len(group_ids) - len(benchmark_calibration_ids),
            "calibration": len(benchmark_calibration_ids),
            "total": len(group_ids),
        }

    all_ids = set(group_benchmark)
    fit_ids = all_ids - calibration_ids
    if not fit_ids or not calibration_ids:
        raise ValueError(
            f"split produced fit={len(fit_ids)} calibration={len(calibration_ids)} groups; "
            "adjust calibration_fraction or provide more groups"
        )
    return {
        "seed": int(seed),
        "calibration_fraction": fraction,
        "fit_group_ids": sorted(fit_ids),
        "calibration_group_ids": sorted(calibration_ids),
        "fit_num_groups": len(fit_ids),
        "calibration_num_groups": len(calibration_ids),
        "total_num_groups": len(all_ids),
        "by_benchmark": by_benchmark,
    }


def subset_features_metadata_by_group_ids(
    features: torch.Tensor,
    metadata: list[dict[str, Any]],
    group_ids: set[str] | list[str] | tuple[str, ...],
) -> tuple[torch.Tensor, list[dict[str, Any]], torch.Tensor]:
    """Return rows whose metadata `id` belongs to `group_ids`, preserving row order."""

    if int(features.shape[0]) != len(metadata):
        raise ValueError(f"features rows {features.shape[0]} != metadata length {len(metadata)}")
    requested = {str(group_id) for group_id in group_ids}
    if not requested:
        raise ValueError("group_ids must not be empty")
    indices = [idx for idx, item in enumerate(metadata) if str(item["id"]) in requested]
    if not indices:
        raise ValueError("no metadata rows matched requested group_ids")
    index_tensor = torch.tensor(indices, dtype=torch.long)
    return features[index_tensor], [metadata[idx] for idx in indices], index_tensor


class HarmAwareRouteVerifier(nn.Module):
    """Shared route encoder with separate improvement and harm heads."""

    def __init__(self, input_dim: int, *, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.safe_head = nn.Linear(int(hidden_dim), 1)
        self.regression_head = nn.Linear(int(hidden_dim), 1)
        self.preserve_head = nn.Linear(int(hidden_dim), 1)
        self.delta_head = nn.Linear(int(hidden_dim), 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.net(features.float())
        return {
            "safe_logit": self.safe_head(hidden).squeeze(-1),
            "regression_logit": self.regression_head(hidden).squeeze(-1),
            "preserve_logit": self.preserve_head(hidden).squeeze(-1),
            "delta_pred": self.delta_head(hidden).squeeze(-1),
        }


def harm_targets_from_metadata(metadata: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "safe": torch.tensor([float(bool(item.get("safe_switch", False))) for item in metadata], dtype=torch.float32),
        "regression": torch.tensor(
            [float(bool(item.get("regression", item.get("is_regression", False)))) for item in metadata],
            dtype=torch.float32,
        ),
        "preserve": torch.tensor([float(bool(item.get("preserve", False))) for item in metadata], dtype=torch.float32),
        "delta_q": torch.tensor(
            [float(item.get("delta_q", item.get("delta_score", 0.0))) for item in metadata],
            dtype=torch.float32,
        ),
        "utility": torch.tensor(
            [float(item.get("v3_1_accuracy_utility", item.get("delta_q", item.get("delta_score", 0.0)))) for item in metadata],
            dtype=torch.float32,
        ),
    }


def _pos_weight(target: torch.Tensor, *, cap: float) -> torch.Tensor:
    pos = float(target.float().sum().item())
    neg = float(target.numel()) - pos
    if pos <= 0.0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(min(neg / pos, float(cap)), dtype=torch.float32)


def harm_route_score(
    outputs: dict[str, torch.Tensor],
    *,
    safe_weight: float = 1.0,
    regression_weight: float = 2.0,
    delta_weight: float = 1.0,
) -> torch.Tensor:
    safe_prob = torch.sigmoid(outputs["safe_logit"].float())
    regression_prob = torch.sigmoid(outputs["regression_logit"].float())
    return (
        float(delta_weight) * outputs["delta_pred"].float()
        + float(safe_weight) * safe_prob
        - float(regression_weight) * regression_prob
    )


def harm_aware_multitask_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    metadata: list[dict[str, Any]],
    groups: list[torch.Tensor],
    *,
    pair_weights: list[torch.Tensor] | None = None,
    safe_loss_weight: float = 2.0,
    regression_loss_weight: float = 2.0,
    preserve_loss_weight: float = 0.25,
    delta_loss_weight: float = 0.5,
    rank_loss_weight: float = 0.5,
    alpha_listwise: float = 0.5,
    safe_pos_weight_cap: float = 40.0,
    regression_pos_weight_cap: float = 10.0,
) -> torch.Tensor:
    safe_target = targets["safe"].to(outputs["safe_logit"].device)
    regression_target = targets["regression"].to(outputs["regression_logit"].device)
    preserve_target = targets["preserve"].to(outputs["preserve_logit"].device)
    delta_target = targets["delta_q"].to(outputs["delta_pred"].device)
    utility_target = targets["utility"].to(outputs["delta_pred"].device)

    safe_loss = F.binary_cross_entropy_with_logits(
        outputs["safe_logit"],
        safe_target,
        pos_weight=_pos_weight(safe_target.cpu(), cap=float(safe_pos_weight_cap)).to(outputs["safe_logit"].device),
    )
    regression_loss = F.binary_cross_entropy_with_logits(
        outputs["regression_logit"],
        regression_target,
        pos_weight=_pos_weight(regression_target.cpu(), cap=float(regression_pos_weight_cap)).to(
            outputs["regression_logit"].device
        ),
    )
    preserve_loss = F.binary_cross_entropy_with_logits(outputs["preserve_logit"], preserve_target)
    delta_loss = F.smooth_l1_loss(outputs["delta_pred"], delta_target)
    rank_score = harm_route_score(outputs)
    rank_loss = combined_utility_loss(
        rank_score,
        utility_target,
        metadata,
        groups,
        pair_weights=pair_weights,
        alpha_listwise=float(alpha_listwise),
    )
    return (
        float(safe_loss_weight) * safe_loss
        + float(regression_loss_weight) * regression_loss
        + float(preserve_loss_weight) * preserve_loss
        + float(delta_loss_weight) * delta_loss
        + float(rank_loss_weight) * rank_loss
    )


def train_harm_aware_route_verifier(
    train_features: torch.Tensor,
    train_metadata: list[dict[str, Any]],
    *,
    hidden_dim: int,
    steps: int,
    lr: float,
    weight_decay: float,
    seed: int,
    dropout: float = 0.1,
    alpha_listwise: float = 0.5,
    hard_negative_weight: float = 2.0,
    below_budget_negative_weight: float = 1.0,
    safe_loss_weight: float = 2.0,
    regression_loss_weight: float = 2.0,
    preserve_loss_weight: float = 0.25,
    delta_loss_weight: float = 0.5,
    rank_loss_weight: float = 0.5,
) -> tuple[HarmAwareRouteVerifier, dict[str, Any]]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if int(train_features.shape[0]) != len(train_metadata):
        raise ValueError(f"features rows {train_features.shape[0]} != metadata length {len(train_metadata)}")
    torch.manual_seed(int(seed))
    model = HarmAwareRouteVerifier(int(train_features.shape[1]), hidden_dim=int(hidden_dim), dropout=float(dropout))
    targets = harm_targets_from_metadata(train_metadata)
    groups = group_indices_from_metadata(train_metadata)
    pair_weights = pair_weight_matrices(
        train_metadata,
        groups,
        hard_negative_weight=float(hard_negative_weight),
        below_budget_negative_weight=float(below_budget_negative_weight),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    model.train()
    with torch.no_grad():
        initial_loss = float(
            harm_aware_multitask_loss(
                model(train_features),
                targets,
                train_metadata,
                groups,
                pair_weights=pair_weights,
                alpha_listwise=float(alpha_listwise),
                safe_loss_weight=float(safe_loss_weight),
                regression_loss_weight=float(regression_loss_weight),
                preserve_loss_weight=float(preserve_loss_weight),
                delta_loss_weight=float(delta_loss_weight),
                rank_loss_weight=float(rank_loss_weight),
            ).item()
        )
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = harm_aware_multitask_loss(
            model(train_features),
            targets,
            train_metadata,
            groups,
            pair_weights=pair_weights,
            alpha_listwise=float(alpha_listwise),
            safe_loss_weight=float(safe_loss_weight),
            regression_loss_weight=float(regression_loss_weight),
            preserve_loss_weight=float(preserve_loss_weight),
            delta_loss_weight=float(delta_loss_weight),
            rank_loss_weight=float(rank_loss_weight),
        )
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_outputs = model(train_features)
        final_loss = float(
            harm_aware_multitask_loss(
                final_outputs,
                targets,
                train_metadata,
                groups,
                pair_weights=pair_weights,
                alpha_listwise=float(alpha_listwise),
                safe_loss_weight=float(safe_loss_weight),
                regression_loss_weight=float(regression_loss_weight),
                preserve_loss_weight=float(preserve_loss_weight),
                delta_loss_weight=float(delta_loss_weight),
                rank_loss_weight=float(rank_loss_weight),
            ).item()
        )
    return model, {
        "objective": "v3_2_harm_aware_multitask",
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "num_train_groups": len(groups),
        "num_train_examples": int(train_features.shape[0]),
        "num_safe_switch_examples": int(targets["safe"].sum().item()),
        "num_regression_examples": int(targets["regression"].sum().item()),
        "num_preserve_examples": int(targets["preserve"].sum().item()),
        "safe_pos_weight": float(_pos_weight(targets["safe"], cap=40.0).item()),
        "regression_pos_weight": float(_pos_weight(targets["regression"], cap=10.0).item()),
        "hidden_dim": int(hidden_dim),
        "steps": int(steps),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "dropout": float(dropout),
        "alpha_listwise": float(alpha_listwise),
        "hard_negative_weight": float(hard_negative_weight),
        "below_budget_negative_weight": float(below_budget_negative_weight),
        "safe_loss_weight": float(safe_loss_weight),
        "regression_loss_weight": float(regression_loss_weight),
        "preserve_loss_weight": float(preserve_loss_weight),
        "delta_loss_weight": float(delta_loss_weight),
        "rank_loss_weight": float(rank_loss_weight),
    }


def _metadata_with_predictions(
    outputs: dict[str, torch.Tensor],
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    safe_prob = torch.sigmoid(outputs["safe_logit"].float()).cpu().tolist()
    regression_prob = torch.sigmoid(outputs["regression_logit"].float()).cpu().tolist()
    preserve_prob = torch.sigmoid(outputs["preserve_logit"].float()).cpu().tolist()
    delta_pred = outputs["delta_pred"].float().cpu().tolist()
    scored: list[dict[str, Any]] = []
    for idx, item in enumerate(metadata):
        scored.append(
            {
                **item,
                "v3_2_safe_prob": float(safe_prob[idx]),
                "v3_2_regression_prob": float(regression_prob[idx]),
                "v3_2_preserve_prob": float(preserve_prob[idx]),
                "v3_2_delta_pred": float(delta_pred[idx]),
            }
        )
    return scored


def select_v3_2_harm_aware_with_policy(
    outputs: dict[str, torch.Tensor],
    metadata: list[dict[str, Any]],
    *,
    num_layers: int = NUM_LAYERS,
    policy_mode: str = "accuracy_first",
    min_safe_prob: float = 0.5,
    max_regression_prob: float = 0.25,
    min_delta_pred: float = 0.0,
    min_preserve_prob: float = 0.8,
    max_preserve_loss: float = 0.02,
    delta_weight: float = 1.0,
    safe_weight: float = 1.0,
    regression_weight: float = 2.0,
    preserve_weight: float = 0.25,
    lambda_on: float = 0.0,
    lambda_transition: float = 0.0,
    margin: float = 0.0,
    max_candidate_rank: int | None = None,
    force_full_qwen_fallback: bool = False,
    policy_name: str = "route_verifier_v3_2",
) -> list[dict[str, Any]]:
    if int(outputs["safe_logit"].numel()) != len(metadata):
        raise ValueError(f"outputs length {outputs['safe_logit'].numel()} != metadata length {len(metadata)}")
    scored_metadata = _metadata_with_predictions(outputs, metadata)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scored_metadata:
        grouped[str(item["id"])].append(item)

    selected_rows: list[dict[str, Any]] = []
    for _, candidates in sorted(grouped.items()):
        fallback = full_qwen_fallback_row(candidates, num_layers=num_layers)
        if force_full_qwen_fallback:
            row = _selected_v3_row(
                policy=policy_name,
                selected=fallback,
                selector_logit=None,
                final_score=0.0,
                selected_full_qwen_fallback=True,
            )
            selected_rows.append(row)
            continue

        eligible: list[tuple[float, dict[str, Any]]] = []
        for item in candidates:
            if max_candidate_rank is not None and int(item.get("decoder_rank", 999999)) > int(max_candidate_rank):
                continue
            safe_prob = float(item["v3_2_safe_prob"])
            regression_prob = float(item["v3_2_regression_prob"])
            preserve_prob = float(item["v3_2_preserve_prob"])
            delta_pred = float(item["v3_2_delta_pred"])
            improving = safe_prob >= float(min_safe_prob) and regression_prob <= float(max_regression_prob) and delta_pred >= float(min_delta_pred)
            preserving = (
                str(policy_mode) == "balanced"
                and regression_prob <= float(max_regression_prob)
                and preserve_prob >= float(min_preserve_prob)
                and delta_pred >= -float(max_preserve_loss)
            )
            if not improving and not preserving:
                continue
            final_score = (
                float(delta_weight) * delta_pred
                + float(safe_weight) * safe_prob
                + float(preserve_weight) * preserve_prob
                - float(regression_weight) * regression_prob
                - float(lambda_on) * float(item.get("on_count", 0.0))
                - float(lambda_transition) * float(item.get("transition_count", 0.0))
            )
            eligible.append((final_score, item))

        if not eligible:
            row = _selected_v3_row(
                policy=policy_name,
                selected=fallback,
                selector_logit=None,
                final_score=0.0,
                selected_full_qwen_fallback=True,
            )
            selected_rows.append(row)
            continue
        best_final, best = max(
            eligible,
            key=lambda pair: (
                pair[0],
                float(pair[1].get("v3_2_delta_pred", 0.0)),
                float(pair[1].get("v3_2_safe_prob", 0.0)),
                -float(pair[1].get("v3_2_regression_prob", 1.0)),
                -float(pair[1].get("on_count", 0.0)),
                -int(pair[1].get("decoder_rank", 999999)),
            ),
        )
        should_switch = best_final >= float(margin)
        selected = best if should_switch else fallback
        row = _selected_v3_row(
            policy=policy_name,
            selected=selected,
            selector_logit=float(best.get("v3_2_safe_prob", 0.0)) if should_switch else None,
            final_score=best_final if should_switch else 0.0,
            selected_full_qwen_fallback=not should_switch,
        )
        if should_switch:
            row.update(
                {
                    "v3_2_safe_prob": float(best["v3_2_safe_prob"]),
                    "v3_2_regression_prob": float(best["v3_2_regression_prob"]),
                    "v3_2_preserve_prob": float(best["v3_2_preserve_prob"]),
                    "v3_2_delta_pred": float(best["v3_2_delta_pred"]),
                }
            )
        selected_rows.append(row)
    return selected_rows
