"""Multi-route objectives for factorized categorical four-action policies."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validated_route_indices(
    logits: torch.Tensor, routes: torch.Tensor
) -> torch.LongTensor:
    if logits.ndim != 3 or logits.shape[-1] != 4:
        raise ValueError("logits must have shape [B, L, 4]")
    if routes.ndim == 2:
        routes = routes.unsqueeze(1)
    if (
        routes.ndim != 3
        or routes.shape[0] != logits.shape[0]
        or routes.shape[2] != logits.shape[1]
    ):
        raise ValueError("routes must have shape [B, V, L]")
    indices = routes.to(device=logits.device, dtype=torch.long)
    if bool(((indices < 0) | (indices >= 4)).any().item()):
        raise ValueError("route action indices must lie in [0, 3]")
    return indices


def categorical_route_log_probability(
    logits: torch.Tensor, routes: torch.Tensor
) -> torch.Tensor:
    """Return complete-route log probabilities with shape `[B, V]`."""
    indices = _validated_route_indices(logits, routes)
    log_probs = F.log_softmax(logits, dim=-1).unsqueeze(1)
    expanded = log_probs.expand(-1, indices.shape[1], -1, -1)
    return expanded.gather(-1, indices.unsqueeze(-1)).squeeze(-1).sum(dim=-1)


def exact_valid_set_nll(
    logits: torch.Tensor,
    valid_routes: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    route_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative weighted probability mass on complete valid action routes."""
    log_prob = categorical_route_log_probability(logits, valid_routes)
    batch_size, route_count = log_prob.shape
    if valid_mask is None:
        valid = torch.ones(
            batch_size, route_count, dtype=torch.bool, device=logits.device
        )
    else:
        valid = valid_mask.to(device=logits.device, dtype=torch.bool)
        if valid.shape != log_prob.shape:
            raise ValueError("valid_mask must have shape [B, V]")
    if bool((valid.sum(dim=1) == 0).any().item()):
        raise ValueError("every sample must contain at least one valid route")
    if route_weights is None:
        weights = valid.to(dtype=logits.dtype)
    else:
        weights = route_weights.to(device=logits.device, dtype=logits.dtype)
        if weights.shape != log_prob.shape:
            raise ValueError("route_weights must have shape [B, V]")
        if bool((weights[valid] <= 0).any().item()):
            raise ValueError("valid route weights must be positive")
        weights = torch.where(valid, weights, torch.zeros_like(weights))
    weights = weights / weights.sum(dim=1, keepdim=True)
    weighted = log_prob + torch.where(
        valid,
        weights.log(),
        torch.full_like(weights, -torch.inf),
    )
    return -torch.logsumexp(weighted, dim=1).mean()


def polar_action_bce_per_route(
    logits: torch.Tensor,
    target_actions: torch.Tensor,
) -> torch.Tensor:
    """Return the mean one-hot BCE for each complete duplicated route."""
    if logits.ndim != 3 or logits.shape[-1] != 4:
        raise ValueError("logits must have shape [N, L, 4]")
    targets = target_actions.to(device=logits.device, dtype=torch.long)
    if targets.shape != logits.shape[:2]:
        raise ValueError("target_actions must have shape [N, L]")
    if bool(((targets < 0) | (targets >= 4)).any().item()):
        raise ValueError("target action indices must lie in [0, 3]")
    one_hot = F.one_hot(targets, num_classes=4).to(dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(
        logits, one_hot, reduction="none"
    ).mean(dim=(1, 2))


def polar_action_bce(
    logits: torch.Tensor,
    target_actions: torch.Tensor,
    *,
    route_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """POLAR duplicated-route BCE generalized to one-hot four-action targets."""
    per_route = polar_action_bce_per_route(logits, target_actions)
    if route_weights is None:
        return per_route.mean()
    weights = route_weights.to(device=logits.device, dtype=logits.dtype)
    if weights.shape != per_route.shape or bool((weights < 0).any().item()):
        raise ValueError("route_weights must be nonnegative with shape [N]")
    if float(weights.sum().item()) <= 0:
        raise ValueError("route_weights must have positive total mass")
    return (per_route * weights).sum() / weights.sum()
