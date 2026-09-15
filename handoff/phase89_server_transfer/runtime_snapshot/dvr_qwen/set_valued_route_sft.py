"""Utilities for set-valued route behavior cloning and replay."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable

import torch
import torch.nn.functional as F


def near_local_pair_weight(hamming_distance: int) -> float:
    """Discount attribution ambiguity as more layer actions differ."""
    if int(hamming_distance) <= 0:
        raise ValueError("hamming distance must be positive")
    return 1.0 / math.sqrt(float(hamming_distance))


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("mask lengths differ")
    return sum(a != b for a, b in zip(left, right))


def select_diverse_correct_masks(
    masks: Iterable[str],
    *,
    max_routes: int,
    exclude_all_off: bool = True,
) -> list[str]:
    """Select dense anchor, shortest route, then farthest-first route modes."""
    if max_routes <= 0:
        raise ValueError("max_routes must be positive")
    unique = sorted(set(str(mask) for mask in masks))
    if not unique:
        return []
    lengths = {len(mask) for mask in unique}
    if len(lengths) != 1 or any(set(mask) - {"0", "1"} for mask in unique):
        raise ValueError("masks must be equal-length binary strings")
    if exclude_all_off:
        unique = [mask for mask in unique if "1" in mask]
    if not unique:
        return []

    all_on = "1" * len(unique[0])
    selected: list[str] = []
    if all_on in unique:
        selected.append(all_on)

    remaining = [mask for mask in unique if mask not in selected]
    if remaining and len(selected) < max_routes:
        shortest = min(remaining, key=lambda mask: (mask.count("1"), mask))
        selected.append(shortest)
        remaining.remove(shortest)

    while remaining and len(selected) < max_routes:
        candidate = min(
            remaining,
            key=lambda mask: (
                -min(hamming_distance(mask, prior) for prior in selected),
                mask.count("1"),
                mask,
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)
    return selected


def deterministic_replay_decision(*, epoch: int, step: int, fraction: float, period: int = 100) -> bool:
    """Return a reproducible replay schedule with an exact rate per period."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    if period <= 0:
        raise ValueError("period must be positive")
    replay_slots = int(round(float(fraction) * int(period)))
    if replay_slots == 0:
        return False
    slot = ((int(step) + int(epoch) * 17) * replay_slots) % int(period)
    return slot < replay_slots


def deterministic_route_index(uid: str, *, epoch: int, occurrence: int, route_count: int) -> int:
    if route_count <= 0:
        raise ValueError("route_count must be positive")
    digest = hashlib.sha256(f"{uid}:{epoch}".encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], byteorder="big") % int(route_count)
    return (offset + int(occurrence)) % int(route_count)


def trajectory_sft_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean binary action loss on logits observed along the teacher route."""
    target = mask.to(device=logits.device, dtype=logits.dtype)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    if logits.shape != target.shape:
        raise ValueError(f"logit/target shape mismatch: {tuple(logits.shape)} != {tuple(target.shape)}")
    return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
