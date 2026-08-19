"""Canonical binary action representation."""

from __future__ import annotations

from collections.abc import Sequence

import torch


NUM_QWEN_LAYERS = 28
VISUAL_OFF = 0
VISUAL_ON = 1


def normalize_visual_on_mask(
    mask: Sequence[int | bool] | torch.Tensor,
    *,
    num_layers: int = NUM_QWEN_LAYERS,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> torch.BoolTensor:
    """Return a validated ``[B, L]`` boolean route tensor."""
    route = torch.as_tensor(mask, device=device)
    if route.ndim == 1:
        route = route.unsqueeze(0)
    if route.ndim != 2 or route.shape[1] != num_layers:
        raise ValueError(f"visual_on_mask must have shape [B, {num_layers}], got {tuple(route.shape)}")
    if batch_size is not None and route.shape[0] != batch_size:
        raise ValueError(f"visual_on_mask batch is {route.shape[0]}, expected {batch_size}")
    if route.dtype != torch.bool and bool(((route != 0) & (route != 1)).any().item()):
        raise ValueError("visual_on_mask must contain only 0/1 values")
    return route.bool()


def mask_key(mask: Sequence[int | bool] | torch.Tensor) -> str:
    if torch.is_tensor(mask):
        if mask.ndim not in (1, 2) or (mask.ndim == 2 and mask.shape[0] != 1):
            raise ValueError("mask_key accepts one rank-1 route or a [1, L] tensor")
        num_layers = int(mask.shape[-1])
    else:
        num_layers = len(mask)
    route = normalize_visual_on_mask(mask, num_layers=num_layers)
    if route.shape[0] != 1:
        raise ValueError("mask_key accepts exactly one route")
    return "".join("1" if value else "0" for value in route[0].tolist())


def count_transitions(mask: Sequence[int | bool]) -> int:
    values = [int(value) for value in mask]
    return sum(left != right for left, right in zip(values, values[1:]))


def maximal_runs(mask: Sequence[int | bool]) -> list[tuple[int, int, int]]:
    """Return deterministic ``(start, stop, action)`` maximal runs."""
    values = [int(value) for value in mask]
    if not values:
        return []
    if any(value not in (0, 1) for value in values):
        raise ValueError("mask must contain only 0/1 values")
    runs: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            runs.append((start, index, values[start]))
            start = index
    return runs
