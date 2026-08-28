"""Canonical action encoding shared with the unified executor."""

from __future__ import annotations

from collections.abc import Iterable

import torch


FOUR_ACTIONS = ("IGNORE", "READ_ONLY", "WRITE_ONLY", "FULL")
ACTION_TO_INDEX = {action: index for index, action in enumerate(FOUR_ACTIONS)}
INDEX_TO_ACTION = dict(enumerate(FOUR_ACTIONS))


def normalize_action(action: str) -> str:
    value = str(action).strip().upper()
    if value not in ACTION_TO_INDEX:
        raise ValueError(f"unknown four-action value: {value!r}")
    return value


def encode_action_route(
    actions: Iterable[str], *, expected_layers: int | None = 28
) -> torch.LongTensor:
    values = tuple(str(action).strip().upper() for action in actions)
    if expected_layers is not None and len(values) != expected_layers:
        raise ValueError(
            f"four-action route must contain exactly {expected_layers} layers, got {len(values)}"
        )
    unknown = [action for action in values if action not in ACTION_TO_INDEX]
    if unknown:
        raise ValueError(f"unknown four-action value: {unknown[0]!r}")
    return torch.tensor([ACTION_TO_INDEX[action] for action in values], dtype=torch.long)


def decode_action_indices(indices: torch.Tensor | Iterable[int]) -> tuple[str, ...]:
    values = indices.detach().cpu().reshape(-1).tolist() if torch.is_tensor(indices) else list(indices)
    if any(int(value) not in INDEX_TO_ACTION for value in values):
        raise ValueError("four-action indices must lie in [0, 3]")
    return tuple(INDEX_TO_ACTION[int(value)] for value in values)
