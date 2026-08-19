"""Multi-valid-route objectives for the binary POLAR adaptation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bernoulli_mask_log_probability(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Return log p(mask | input) for every enumerated mask.

    ``logits`` is ``[B,L]`` and ``masks`` may be ``[B,V,L]`` or ``[B,L]``.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [B, L]")
    if masks.ndim == 2:
        masks = masks.unsqueeze(1)
    if masks.ndim != 3 or masks.shape[0] != logits.shape[0] or masks.shape[2] != logits.shape[1]:
        raise ValueError("masks must have shape [B, V, L]")
    targets = masks.to(device=logits.device, dtype=logits.dtype)
    log_p_on = F.logsigmoid(logits).unsqueeze(1)
    log_p_off = F.logsigmoid(-logits).unsqueeze(1)
    return (targets * log_p_on + (1.0 - targets) * log_p_off).sum(dim=-1)


def multi_valid_set_nll(
    logits: torch.Tensor,
    valid_masks: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    route_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative log probability mass assigned to the observed valid-mask set.

    This is the direct-mask counterpart of POLAR's multiple-valid-program
    supervision.  Duplicate masks must be removed by the data adapter first.
    Route weights are normalized within each sample and enter as log-priors.
    """
    log_prob = bernoulli_mask_log_probability(logits, valid_masks)
    batch_size, num_routes = log_prob.shape
    if valid_mask is None:
        valid_mask = torch.ones(batch_size, num_routes, dtype=torch.bool, device=logits.device)
    else:
        valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
        if valid_mask.shape != log_prob.shape:
            raise ValueError("valid_mask must have shape [B, V]")
    if bool((valid_mask.sum(dim=1) == 0).any().item()):
        raise ValueError("every sample must contain at least one valid route")
    if route_weights is None:
        weights = valid_mask.to(logits.dtype)
    else:
        weights = route_weights.to(device=logits.device, dtype=logits.dtype)
        if weights.shape != log_prob.shape:
            raise ValueError("route_weights must have shape [B, V]")
        if bool((weights[valid_mask] <= 0).any().item()):
            raise ValueError("valid route weights must be positive")
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
    weights = weights / weights.sum(dim=1, keepdim=True)
    weighted = log_prob + torch.where(valid_mask, weights.log(), torch.full_like(weights, -torch.inf))
    return -torch.logsumexp(weighted, dim=1).mean()


def polar_path_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """POLAR-faithful duplicated-path BCE with optional sample normalization."""
    per_path = polar_path_bce_per_path(logits, targets)
    if sample_weights is None:
        return per_path.mean()
    weights = sample_weights.to(device=logits.device, dtype=logits.dtype)
    if weights.shape != per_path.shape or bool((weights < 0).any().item()):
        raise ValueError("sample_weights must be nonnegative with shape [N]")
    if float(weights.sum().item()) <= 0:
        raise ValueError("sample_weights must have positive total mass")
    return (per_path * weights).sum() / weights.sum()


def polar_path_bce_per_path(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return ordinary mean-over-bits BCE for each duplicated route row."""
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("logits and targets must have identical [N, L] shapes")
    return F.binary_cross_entropy_with_logits(
        logits, targets.to(device=logits.device, dtype=logits.dtype), reduction="none"
    ).mean(dim=1)


def expected_on_count(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits).sum(dim=-1)
