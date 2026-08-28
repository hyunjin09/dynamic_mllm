"""Offline complete-route metrics for categorical four-action policies."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Sequence

import torch

from .actions import FOUR_ACTIONS
from .decode import topk_factorized_routes
from .losses import exact_valid_set_nll, polar_action_bce_per_route


def nearest_valid_hamming(
    predicted: Sequence[int], valid_routes: Sequence[Sequence[int]]
) -> int:
    if not valid_routes:
        raise ValueError("valid_routes cannot be empty")
    return min(
        sum(int(left) != int(right) for left, right in zip(predicted, valid))
        for valid in valid_routes
    )


def _route_key(route: Sequence[int]) -> str:
    return "|".join(FOUR_ACTIONS[int(action)] for action in route)


def route_diversity_metrics(route_counts: Counter[str], num_layers: int) -> dict[str, Any]:
    total = sum(route_counts.values())
    if total < 1:
        raise ValueError("route_counts cannot be empty")
    entropy = -sum(
        (count / total) * math.log(count / total) for count in route_counts.values()
    )
    action_counts = Counter()
    for key, count in route_counts.items():
        actions = key.split("|")
        if len(actions) != num_layers:
            raise ValueError("route width differs from num_layers")
        for action in actions:
            action_counts[action] += count
    return {
        "unique_top1_routes": len(route_counts),
        "top1_route_entropy_nats": entropy,
        "fraction_top1_all_full": route_counts.get("|".join(["FULL"] * num_layers), 0) / total,
        "fraction_top1_all_ignore": route_counts.get("|".join(["IGNORE"] * num_layers), 0) / total,
        "mean_action_count_per_route": {
            action: action_counts[action] / total for action in FOUR_ACTIONS
        },
        "top1_route_counts": dict(sorted(route_counts.items())),
        "top10_routes": [
            {"route": route, "count": count, "fraction": count / total}
            for route, count in sorted(route_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
        ],
    }


def batch_offline_metrics(
    logits: torch.Tensor,
    valid_routes: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    candidates = topk_factorized_routes(logits, top_k=top_k)
    top1_hits = 0
    topk_hits = 0
    hamming = 0.0
    route_counts: Counter[str] = Counter()
    for batch_index, sample_candidates in enumerate(candidates):
        targets = valid_routes[batch_index, valid_mask[batch_index]].long().cpu().tolist()
        valid_set = {tuple(route) for route in targets}
        top1 = sample_candidates[0].action_indices
        route_counts[_route_key(top1)] += 1
        top1_hits += int(top1 in valid_set)
        topk_hits += int(any(candidate.action_indices in valid_set for candidate in sample_candidates))
        hamming += nearest_valid_hamming(top1, targets)
    count = int(logits.shape[0])
    return {
        "top1_valid_route_coverage": top1_hits / count,
        "topk_valid_route_coverage": topk_hits / count,
        "nearest_valid_hamming": hamming / count,
        "top1_route_counts": route_counts,
    }


def _weighted_bce_per_sample(
    logits: torch.Tensor,
    valid_routes: torch.Tensor,
    valid_mask: torch.Tensor,
    route_weights: torch.Tensor,
) -> torch.Tensor:
    batch_size, max_routes, num_layers = valid_routes.shape
    expanded = logits.unsqueeze(1).expand(-1, max_routes, -1, -1)
    selected_logits = expanded[valid_mask]
    selected_targets = valid_routes[valid_mask]
    per_route = polar_action_bce_per_route(selected_logits, selected_targets)
    weights = route_weights[valid_mask].to(per_route.dtype)
    sample_indices = (
        torch.arange(batch_size, device=logits.device)
        .unsqueeze(1)
        .expand(-1, max_routes)[valid_mask]
    )
    weighted = torch.zeros(batch_size, dtype=per_route.dtype, device=logits.device)
    mass = torch.zeros_like(weighted)
    weighted.scatter_add_(0, sample_indices, per_route * weights)
    mass.scatter_add_(0, sample_indices, weights)
    if bool((mass <= 0).any().item()):
        raise ValueError("every sample must have positive valid-route weight")
    return weighted / mass


class FourActionMetricAccumulator:
    def __init__(self, *, top_k: int = 5) -> None:
        self.top_k = int(top_k)
        self.examples = 0
        self.nll_sum = 0.0
        self.bce_sum = 0.0
        self.metric_sums: dict[str, float] = {}
        self.route_counts: Counter[str] = Counter()
        self.num_layers: int | None = None

    def update(
        self,
        logits: torch.Tensor,
        valid_routes: torch.Tensor,
        valid_mask: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> None:
        count = int(logits.shape[0])
        nll = exact_valid_set_nll(
            logits,
            valid_routes,
            valid_mask=valid_mask,
            route_weights=route_weights,
        )
        bce = _weighted_bce_per_sample(
            logits, valid_routes, valid_mask, route_weights
        ).mean()
        metrics = batch_offline_metrics(
            logits, valid_routes, valid_mask, top_k=self.top_k
        )
        self.examples += count
        self.nll_sum += float(nll) * count
        self.bce_sum += float(bce) * count
        self.num_layers = int(logits.shape[1])
        self.route_counts.update(metrics.pop("top1_route_counts"))
        for key, value in metrics.items():
            self.metric_sums[key] = self.metric_sums.get(key, 0.0) + float(value) * count

    def finalize(self, *, objective: str) -> dict[str, Any]:
        if self.examples < 1 or self.num_layers is None:
            raise ValueError("cannot finalize an empty accumulator")
        if objective not in {"duplicated_action_bce", "exact_set_nll"}:
            raise ValueError("unknown four-action objective")
        result = {
            "examples": self.examples,
            "set_nll": self.nll_sum / self.examples,
            "duplicated_action_bce": self.bce_sum / self.examples,
            **{key: value / self.examples for key, value in self.metric_sums.items()},
            **route_diversity_metrics(self.route_counts, self.num_layers),
        }
        result["objective_loss"] = result[
            "set_nll" if objective == "exact_set_nll" else "duplicated_action_bce"
        ]
        return result


def checkpoint_key(row: dict[str, Any]) -> tuple[float, ...]:
    """Prospective objective-independent checkpoint ordering; larger is better."""
    metrics = row["validation"]["overall"]
    return (
        float(metrics["top1_valid_route_coverage"]),
        float(metrics["topk_valid_route_coverage"]),
        -float(metrics["nearest_valid_hamming"]),
        -float(metrics["objective_loss"]),
        -int(row["epoch"]),
    )
