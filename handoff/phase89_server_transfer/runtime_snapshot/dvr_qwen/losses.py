"""Router losses for binary VISUAL_ON supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _positive_layer_weight_matrix(
    positive_layer_weights: torch.Tensor | None,
    *,
    targets: torch.Tensor,
) -> torch.Tensor | None:
    if positive_layer_weights is None:
        return None
    weights = torch.as_tensor(positive_layer_weights, device=targets.device, dtype=targets.dtype)
    if weights.ndim == 1:
        if int(weights.shape[0]) != int(targets.shape[1]):
            raise ValueError(
                f"positive_layer_weights must have shape [L] or [B, L], got {tuple(weights.shape)}"
            )
        weights = weights.unsqueeze(0).expand_as(targets)
    elif weights.ndim == 2:
        if weights.shape != targets.shape:
            raise ValueError(
                f"positive_layer_weights must have shape [L] or [B, L], got {tuple(weights.shape)}"
            )
    else:
        raise ValueError(f"positive_layer_weights must have shape [L] or [B, L], got {tuple(weights.shape)}")
    if bool((weights <= 0).any().item()):
        raise ValueError("positive_layer_weights must be positive")
    return 1.0 + targets * (weights - 1.0)


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha_pos: float = 3.0,
    alpha_neg: float = 1.0,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Elementwise focal BCE for binary VISUAL_ON labels."""
    if logits.shape != targets.shape:
        raise ValueError(f"logits and targets must have the same shape, got {logits.shape} and {targets.shape}")
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha = alpha_pos * targets + alpha_neg * (1.0 - targets)
    return alpha * (1.0 - pt).pow(gamma) * ce


def compute_route_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    lambda_budget: float = 0.05,
    lambda_fn: float = 0.0,
    alpha_pos: float = 3.0,
    alpha_neg: float = 1.0,
    gamma: float = 2.0,
    positive_layer_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return total route loss and its focal/Budget components."""
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, L], got {tuple(logits.shape)}")
    if logits.shape != targets.shape:
        raise ValueError(f"logits and targets must have the same shape, got {logits.shape} and {targets.shape}")
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    focal_elements = focal_bce_with_logits(
        logits,
        targets,
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
        gamma=gamma,
    )
    effective_weights = _positive_layer_weight_matrix(positive_layer_weights, targets=targets)
    focal = focal_elements.mean() if effective_weights is None else (focal_elements * effective_weights).mean()
    probs = torch.sigmoid(logits)
    budget = ((probs.sum(dim=1) - targets.sum(dim=1)) ** 2).mean()
    false_negative_elements = targets * (1.0 - probs)
    false_negative = (
        false_negative_elements.mean()
        if effective_weights is None
        else (false_negative_elements * effective_weights).mean()
    )
    total = focal + float(lambda_budget) * budget + float(lambda_fn) * false_negative
    return {
        "loss": total,
        "focal_bce": focal,
        "budget": budget,
        "false_negative": false_negative,
        "unweighted_focal_bce": focal_elements.mean(),
        "unweighted_false_negative": false_negative_elements.mean(),
    }
