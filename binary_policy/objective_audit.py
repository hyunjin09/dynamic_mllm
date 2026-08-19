"""Small loss-aligned diagnostics for the direct binary route objective."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .losses import multi_valid_set_nll


def optimize_complete_mask_logits(
    valid_masks: Sequence[Sequence[int]],
    *,
    weights: Sequence[float] | None = None,
    seed: int = 20260809,
    steps: int = 300,
    learning_rate: float = 0.1,
) -> dict:
    """Optimize free logits only to validate the complete-mask objective.

    This is intentionally not a predictor or a generalization experiment. Each
    call owns one free logit vector and demonstrates only that the implemented
    loss can prefer a coherent member of an enumerated valid set.
    """
    masks = torch.tensor(valid_masks, dtype=torch.float64)
    if masks.ndim != 2 or masks.shape[0] < 1 or masks.shape[1] < 1:
        raise ValueError("valid_masks must be a nonempty [V, L] collection")
    if bool(((masks != 0) & (masks != 1)).any().item()):
        raise ValueError("valid_masks must be binary")
    if steps < 1 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")
    route_weights = None
    if weights is not None:
        route_weights = torch.tensor(weights, dtype=torch.float64).view(1, -1)
        if route_weights.shape[1] != masks.shape[0]:
            raise ValueError("weights must have one value per valid mask")
    generator = torch.Generator().manual_seed(seed)
    logits = torch.nn.Parameter(0.01 * torch.randn(1, masks.shape[1], generator=generator, dtype=torch.float64))
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    targets = masks.unsqueeze(0)
    initial_loss = float(multi_valid_set_nll(logits, targets, route_weights=route_weights).detach())
    finite_gradients = True
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = multi_valid_set_nll(logits, targets, route_weights=route_weights)
        loss.backward()
        finite_gradients &= bool(torch.isfinite(logits.grad).all().item())
        optimizer.step()
    final_loss = float(multi_valid_set_nll(logits, targets, route_weights=route_weights).detach())
    probabilities = torch.sigmoid(logits.detach())[0]
    predicted = (probabilities >= 0.5).to(torch.int64).tolist()
    valid_rows = {tuple(int(value) for value in row) for row in valid_masks}
    return {
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "probabilities": probabilities.tolist(),
        "predicted_mask": predicted,
        "top1_is_valid": tuple(predicted) in valid_rows,
        "finite_gradients": finite_gradients,
    }
