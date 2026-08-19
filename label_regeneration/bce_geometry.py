"""Deterministic label-geometry primitives for binary visual routes."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from functools import lru_cache


Mask = tuple[int, ...]


def as_mask(values: Sequence[int]) -> Mask:
    mask = tuple(int(value) for value in values)
    if not mask or any(value not in (0, 1) for value in mask):
        raise ValueError("route masks must be nonempty and binary")
    return mask


@lru_cache(maxsize=None)
def _mask_integer(mask: Mask) -> int:
    value = 0
    for bit in mask:
        value = (value << 1) | bit
    return value


def hamming(left: Mask, right: Mask) -> int:
    if len(left) != len(right):
        raise ValueError("Hamming distance requires equal-width masks")
    return (_mask_integer(left) ^ _mask_integer(right)).bit_count()


def transition_count(mask: Mask) -> int:
    return sum(left != right for left, right in zip(mask, mask[1:]))


def polar_route_weights(masks: Sequence[Mask]) -> list[float]:
    """Match ``polar_full_downweight_0.3`` from ``binary_policy.dataset``."""
    if not masks:
        raise ValueError("at least one valid mask is required")
    width = len(masks[0])
    if any(len(mask) != width for mask in masks):
        raise ValueError("all masks must have equal width")
    all_on = (1,) * width
    has_cheaper = all_on in masks and any(sum(mask) < width for mask in masks)
    raw = [0.3 if has_cheaper and mask == all_on else 1.0 for mask in masks]
    total = sum(raw)
    return [weight / total for weight in raw]


def layer_marginals(masks: Sequence[Mask], weights: Sequence[float] | None = None) -> list[float]:
    if not masks:
        raise ValueError("at least one valid mask is required")
    width = len(masks[0])
    if any(len(mask) != width for mask in masks):
        raise ValueError("all masks must have equal width")
    if weights is None:
        normalized = [1.0 / len(masks)] * len(masks)
    else:
        if len(weights) != len(masks) or any(weight < 0 for weight in weights):
            raise ValueError("weights must be nonnegative and match masks")
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must have positive mass")
        normalized = [weight / total for weight in weights]
    marginals = [
        sum(weight * mask[layer] for mask, weight in zip(masks, normalized))
        for layer in range(width)
    ]
    if any(value < -1e-12 or value > 1.0 + 1e-12 for value in marginals):
        raise ValueError("computed marginal lies materially outside [0, 1]")
    return [min(1.0, max(0.0, value)) for value in marginals]


def threshold_mask(marginals: Sequence[float], threshold: float = 0.5) -> Mask:
    """Match deployed ``sigmoid(logit) >= threshold``; exact ties resolve ON."""
    return tuple(int(value >= threshold) for value in marginals)


def binary_entropy(probability: float) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if probability in (0.0, 1.0):
        return 0.0
    return -probability * math.log(probability) - (1.0 - probability) * math.log(1.0 - probability)


def connected_components(masks: Sequence[Mask], radius: int) -> list[list[int]]:
    """Connected components of the graph whose edges have Hamming <= radius."""
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    parent = list(range(len(masks)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(masks)):
        for right in range(left + 1, len(masks)):
            if hamming(masks[left], masks[right]) <= radius:
                union(left, right)
    grouped: dict[int, list[int]] = {}
    for index in range(len(masks)):
        grouped.setdefault(find(index), []).append(index)
    return sorted(grouped.values(), key=lambda component: (-len(component), component))


def effective_component_count(components: Sequence[Sequence[int]]) -> float:
    total = sum(len(component) for component in components)
    if total == 0:
        return 0.0
    return 1.0 / sum((len(component) / total) ** 2 for component in components)


def pareto_efficient_indices(masks: Sequence[Mask], utilities: Sequence[float]) -> list[int]:
    """Keep routes not dominated by higher/equal utility at strictly lower ON cost."""
    if len(masks) != len(utilities):
        raise ValueError("utilities must match masks")
    efficient: list[int] = []
    for index, (mask, utility) in enumerate(zip(masks, utilities)):
        cost = sum(mask)
        dominated = any(
            other_index != index
            and other_utility >= utility
            and sum(other_mask) < cost
            for other_index, (other_mask, other_utility) in enumerate(zip(masks, utilities))
        )
        if not dominated:
            efficient.append(index)
    return efficient


def diversity_balanced_indices(masks: Sequence[Mask], limit: int) -> list[int]:
    """Start at lowest ON, then greedily maximize minimum Hamming distance."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if not masks:
        return []
    keys = ["".join(map(str, mask)) for mask in masks]
    first = min(range(len(masks)), key=lambda index: (sum(masks[index]), keys[index]))
    selected = [first]
    remaining = set(range(len(masks))) - {first}
    while remaining and len(selected) < limit:
        next_index = min(
            remaining,
            key=lambda index: (
                -min(hamming(masks[index], masks[chosen]) for chosen in selected),
                sum(masks[index]),
                keys[index],
            ),
        )
        selected.append(next_index)
        remaining.remove(next_index)
    return selected


def pairwise_distances(masks: Sequence[Mask]) -> Iterable[int]:
    for left in range(len(masks)):
        for right in range(left + 1, len(masks)):
            yield hamming(masks[left], masks[right])
