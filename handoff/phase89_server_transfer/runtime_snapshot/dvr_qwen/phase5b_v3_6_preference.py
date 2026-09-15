"""Positive-balanced fallback-vs-safe preferences for Phase 5B v3.6."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import torch
import torch.nn.functional as F

from dvr_qwen.route_selector import RouteVerifier


DEFAULT_V3_6_TYPE_WEIGHTS: dict[str, float] = {
    "safe_vs_fallback": 4.0,
    "safe_vs_negative": 1.0,
    "fallback_vs_regression": 2.0,
    "fallback_vs_cost_only": 1.0,
    "fallback_vs_other": 1.0,
}


def _is_fallback(item: dict[str, Any]) -> bool:
    return bool(item.get("selected_full_qwen_fallback", False))


def _is_safe(item: dict[str, Any]) -> bool:
    return bool(item.get("safe_switch", False))


def _is_regression(item: dict[str, Any]) -> bool:
    return bool(item.get("regression", item.get("is_regression", False)))


def _is_cost_only(item: dict[str, Any]) -> bool:
    return bool(item.get("cost_only_preserve", False))


def _add_pair(
    pairs: list[dict[str, Any]],
    *,
    group_id: str,
    pair_type: str,
    winner_idx: int,
    loser_idx: int,
    weight: float,
) -> None:
    if winner_idx == loser_idx:
        return
    pairs.append(
        {
            "id": str(group_id),
            "pair_type": str(pair_type),
            "winner_idx": int(winner_idx),
            "loser_idx": int(loser_idx),
            "weight": float(weight),
        }
    )


def build_v3_6_preference_pairs(
    metadata: list[dict[str, Any]],
    *,
    safe_vs_fallback_weight: float = 8.0,
    safe_vs_negative_weight: float = 2.0,
    fallback_vs_regression_weight: float = 4.0,
    fallback_vs_cost_only_weight: float = 1.0,
    fallback_vs_other_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Build route-level preference pairs with fallback as the explicit anchor.

    The core v3.6 pressure is direct: safe-switch candidates should outrank the
    full-Qwen/all-VISUAL_ON fallback. Regression and cost-only preserve
    candidates should remain below fallback.
    """

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, item in enumerate(metadata):
        grouped[str(item["id"])].append((idx, item))

    pairs: list[dict[str, Any]] = []
    for group_id, items in sorted(grouped.items()):
        fallback = [(idx, item) for idx, item in items if _is_fallback(item)]
        if len(fallback) != 1:
            raise ValueError(f"group {group_id} must contain exactly one fallback row, found {len(fallback)}")
        fallback_idx, _ = fallback[0]
        safe_indices = [idx for idx, item in items if (not _is_fallback(item)) and _is_safe(item)]
        regression_indices = [
            idx for idx, item in items if (not _is_fallback(item)) and (not _is_safe(item)) and _is_regression(item)
        ]
        cost_only_indices = [
            idx for idx, item in items if (not _is_fallback(item)) and (not _is_safe(item)) and _is_cost_only(item)
        ]
        other_negative_indices = [
            idx
            for idx, item in items
            if (not _is_fallback(item)) and (not _is_safe(item)) and (not _is_regression(item)) and (not _is_cost_only(item))
        ]

        for safe_idx in safe_indices:
            _add_pair(
                pairs,
                group_id=group_id,
                pair_type="safe_vs_fallback",
                winner_idx=safe_idx,
                loser_idx=fallback_idx,
                weight=safe_vs_fallback_weight,
            )
            for negative_idx in [*regression_indices, *other_negative_indices]:
                _add_pair(
                    pairs,
                    group_id=group_id,
                    pair_type="safe_vs_negative",
                    winner_idx=safe_idx,
                    loser_idx=negative_idx,
                    weight=safe_vs_negative_weight,
                )

        for regression_idx in regression_indices:
            _add_pair(
                pairs,
                group_id=group_id,
                pair_type="fallback_vs_regression",
                winner_idx=fallback_idx,
                loser_idx=regression_idx,
                weight=fallback_vs_regression_weight,
            )
        for cost_only_idx in cost_only_indices:
            _add_pair(
                pairs,
                group_id=group_id,
                pair_type="fallback_vs_cost_only",
                winner_idx=fallback_idx,
                loser_idx=cost_only_idx,
                weight=fallback_vs_cost_only_weight,
            )
        for other_idx in other_negative_indices:
            _add_pair(
                pairs,
                group_id=group_id,
                pair_type="fallback_vs_other",
                winner_idx=fallback_idx,
                loser_idx=other_idx,
                weight=fallback_vs_other_weight,
            )
    return pairs


def preference_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(str(pair["pair_type"]) for pair in pairs)
    weight_by_type: dict[str, float] = defaultdict(float)
    groups_by_type: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        pair_type = str(pair["pair_type"])
        weight_by_type[pair_type] += float(pair.get("weight", 1.0))
        groups_by_type[pair_type].add(str(pair["id"]))
    return {
        "num_pairs": len(pairs),
        "num_groups_with_pairs": len({str(pair["id"]) for pair in pairs}),
        "by_type": dict(sorted(by_type.items())),
        "weight_by_type": {key: float(weight_by_type[key]) for key in sorted(weight_by_type)},
        "groups_by_type": {key: len(groups_by_type[key]) for key in sorted(groups_by_type)},
    }


def _pair_tensors(
    pairs: list[dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    winners = torch.tensor([int(pair["winner_idx"]) for pair in pairs], dtype=torch.long, device=device)
    losers = torch.tensor([int(pair["loser_idx"]) for pair in pairs], dtype=torch.long, device=device)
    weights = torch.tensor([float(pair.get("weight", 1.0)) for pair in pairs], dtype=torch.float32, device=device)
    pair_types = [str(pair["pair_type"]) for pair in pairs]
    return winners, losers, weights, pair_types


def preference_pairwise_loss(
    scores: torch.Tensor,
    pairs: list[dict[str, Any]],
    *,
    balance_by_type: bool = False,
    type_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Return weighted pairwise logistic loss over route preferences."""

    if not pairs:
        return scores.float().sum() * 0.0
    scores = scores.float().view(-1)
    winners, losers, weights, pair_types = _pair_tensors(pairs, device=scores.device)
    if int(winners.max().item()) >= int(scores.numel()) or int(losers.max().item()) >= int(scores.numel()):
        raise ValueError("preference pair index exceeds score tensor length")
    losses = F.softplus(-(scores[winners] - scores[losers])) * weights
    if not balance_by_type:
        return losses.sum() / weights.sum().clamp_min(1e-6)

    requested_type_weights = dict(DEFAULT_V3_6_TYPE_WEIGHTS if type_weights is None else type_weights)
    type_losses: list[torch.Tensor] = []
    type_loss_weights: list[float] = []
    for pair_type in sorted(set(pair_types)):
        mask = torch.tensor([item == pair_type for item in pair_types], dtype=torch.bool, device=scores.device)
        type_weight = float(requested_type_weights.get(pair_type, 1.0))
        type_losses.append((losses[mask].sum() / weights[mask].sum().clamp_min(1e-6)) * type_weight)
        type_loss_weights.append(type_weight)
    return torch.stack(type_losses).sum() / max(sum(type_loss_weights), 1e-6)


def preference_accuracy(scores: torch.Tensor, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        return {"num_pairs": 0, "accuracy": 0.0, "by_type": {}}
    scores = scores.float().view(-1).cpu()
    stats: dict[str, dict[str, float]] = defaultdict(lambda: {"pairs": 0.0, "correct": 0.0, "margin_sum": 0.0})
    total_correct = 0
    for pair in pairs:
        margin = float(scores[int(pair["winner_idx"])].item() - scores[int(pair["loser_idx"])].item())
        correct = margin > 0.0
        pair_type = str(pair["pair_type"])
        stats[pair_type]["pairs"] += 1.0
        stats[pair_type]["correct"] += float(correct)
        stats[pair_type]["margin_sum"] += margin
        total_correct += int(correct)
    by_type = {
        pair_type: {
            "num_pairs": int(values["pairs"]),
            "accuracy": values["correct"] / values["pairs"] if values["pairs"] else 0.0,
            "avg_margin": values["margin_sum"] / values["pairs"] if values["pairs"] else 0.0,
        }
        for pair_type, values in sorted(stats.items())
    }
    return {
        "num_pairs": len(pairs),
        "accuracy": total_correct / len(pairs),
        "by_type": by_type,
    }


def train_v3_6_preference_route_verifier(
    train_features: torch.Tensor,
    train_metadata: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    *,
    hidden_dim: int,
    steps: int,
    lr: float,
    weight_decay: float,
    seed: int,
    dropout: float = 0.1,
    balance_by_type: bool = True,
    type_weights: dict[str, float] | None = None,
) -> tuple[RouteVerifier, dict[str, Any]]:
    """Train a scalar route verifier from explicit preference pairs."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if int(train_features.shape[0]) != len(train_metadata):
        raise ValueError(f"features rows {train_features.shape[0]} != metadata length {len(train_metadata)}")
    if not pairs:
        raise ValueError("preference pairs must not be empty")
    torch.manual_seed(int(seed))
    model = RouteVerifier(int(train_features.shape[1]), hidden_dim=int(hidden_dim), dropout=float(dropout))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    train_features = train_features.float()

    def eval_loss_and_accuracy() -> tuple[float, dict[str, Any]]:
        model.eval()
        with torch.no_grad():
            scores = model(train_features).cpu()
            loss = preference_pairwise_loss(
                scores,
                pairs,
                balance_by_type=bool(balance_by_type),
                type_weights=type_weights,
            )
            accuracy = preference_accuracy(scores, pairs)
        return float(loss.item()), accuracy

    initial_loss, initial_accuracy = eval_loss_and_accuracy()
    for _ in range(int(steps)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        scores = model(train_features)
        loss = preference_pairwise_loss(
            scores,
            pairs,
            balance_by_type=bool(balance_by_type),
            type_weights=type_weights,
        )
        loss.backward()
        optimizer.step()
    final_loss, final_accuracy = eval_loss_and_accuracy()
    return model, {
        "objective": "positive_balanced_fallback_vs_safe_pairwise_preferences",
        "steps": int(steps),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "balance_by_type": bool(balance_by_type),
        "type_weights": dict(DEFAULT_V3_6_TYPE_WEIGHTS if type_weights is None else type_weights),
        "preference_pairs": preference_summary(pairs),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_preference_accuracy": initial_accuracy,
        "final_preference_accuracy": final_accuracy,
    }
