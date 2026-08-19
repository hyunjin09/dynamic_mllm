"""Canonical run-length binary representation retained as a structured baseline."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .actions import maximal_runs


def mask_to_canonical_targets(mask: Sequence[int | bool]) -> tuple[list[int], list[int]]:
    """Encode a mask as deterministic maximal-run starts and ON/OFF labels."""
    values = [int(value) for value in mask]
    boundaries = [0] * len(values)
    operations = [-100] * len(values)
    for start, _, action in maximal_runs(values):
        boundaries[start] = 1
        operations[start] = action
    if boundaries:
        boundaries[0] = 0  # layer zero is an implicit segment start, matching POLAR
    return boundaries, operations


def canonical_targets_to_mask(boundaries: Sequence[int], operations: Sequence[int]) -> list[int]:
    if len(boundaries) != len(operations):
        raise ValueError("boundaries and operations must have equal length")
    if not boundaries:
        return []
    starts = [0] + [index for index in range(1, len(boundaries)) if int(boundaries[index]) == 1]
    output = [0] * len(boundaries)
    for run_idx, start in enumerate(starts):
        stop = starts[run_idx + 1] if run_idx + 1 < len(starts) else len(output)
        action = int(operations[start])
        if action not in (0, 1):
            raise ValueError(f"segment start {start} has invalid operation {action}")
        output[start:stop] = [action] * (stop - start)
    return output


def segmented_binary_loss(
    boundary_logits: torch.Tensor,
    operation_logits: torch.Tensor,
    boundary_targets: torch.Tensor,
    operation_targets: torch.Tensor,
) -> torch.Tensor:
    if boundary_logits.shape != boundary_targets.shape:
        raise ValueError("boundary shapes do not match")
    if operation_logits.shape[:2] != operation_targets.shape or operation_logits.shape[-1] != 2:
        raise ValueError("operation logits must have shape [B, L, 2]")
    boundary_loss = nn.functional.binary_cross_entropy_with_logits(
        boundary_logits[:, 1:], boundary_targets[:, 1:].to(boundary_logits.dtype)
    )
    operation_loss = nn.functional.cross_entropy(
        operation_logits.reshape(-1, 2), operation_targets.reshape(-1), ignore_index=-100
    )
    return boundary_loss + operation_loss
