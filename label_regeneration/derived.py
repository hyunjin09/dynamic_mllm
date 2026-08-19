"""Deterministic P8 supervision views derived from validated route records."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from typing import Any, Iterable, Sequence


def _mask(candidate: dict[str, Any]) -> tuple[int, ...]:
    values = tuple(int(bit) for bit in candidate.get("visual_on_mask", ()))
    if not values or any(bit not in (0, 1) for bit in values):
        raise ValueError("candidate must contain a nonempty binary visual_on_mask")
    key = "".join(str(bit) for bit in values)
    if candidate.get("mask_key") != key:
        raise ValueError(f"candidate mask_key mismatch for {candidate.get('route_id')}")
    if int(candidate.get("num_visual_on_layers", -1)) != sum(values):
        raise ValueError(f"candidate ON-count mismatch for {candidate.get('route_id')}")
    transitions = sum(left != right for left, right in zip(values, values[1:]))
    if int(candidate.get("num_transitions", -1)) != transitions:
        raise ValueError(f"candidate transition-count mismatch for {candidate.get('route_id')}")
    return values


def _hamming(left: Sequence[int], right: Sequence[int]) -> int:
    return sum(int(a) != int(b) for a, b in zip(left, right))


def _mask_integer(mask: Sequence[int]) -> int:
    value = 0
    for bit in mask:
        value = (value << 1) | int(bit)
    return value


def single_best_valid_route(routes: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the minimum-visual-ON route with frozen lexical tie-breaking."""
    routes = list(routes)
    if not routes:
        return None
    return min(routes, key=lambda route: (sum(_mask(route)), route["mask_key"]))


def select_diverse_valid_routes(
    routes: Iterable[dict[str, Any]],
    *,
    limit: int,
    seed: int,
    uid: str,
) -> list[dict[str, Any]]:
    """Select anchors plus a deterministic compute-stratified diverse subset.

    When the valid set exceeds ``limit``, the selection first preserves the
    minimum-compute route and valid ALL-OFF/ALL-ON anchors. Remaining routes are
    chosen by balancing exact ON-count strata, then maximizing minimum Hamming
    distance, transition-count coverage, and transition distance. A seeded
    digest resolves only otherwise exact ties.
    """
    if limit < 1:
        raise ValueError("route limit must be positive")
    # Cache all geometry once. The cap is applied to hundreds of routes per
    # sample, so recomputing tuple-wise Hamming distances inside every greedy
    # step is needlessly expensive; XOR bit counts are exactly equivalent.
    by_key: dict[str, tuple[dict[str, Any], tuple[int, ...], int, int, int, int]] = {}
    widths: set[int] = set()
    for route in routes:
        mask = _mask(route)
        widths.add(len(mask))
        key = route["mask_key"]
        if key in by_key:
            raise ValueError(f"duplicate valid mask {key} for {uid}")
        tie = int.from_bytes(sha256(f"{seed}:{uid}:{key}".encode()).digest()[:8], "big")
        by_key[key] = (
            route,
            mask,
            _mask_integer(mask),
            sum(mask),
            int(route["num_transitions"]),
            tie,
        )
    if len(widths) > 1:
        raise ValueError(f"inconsistent route widths for {uid}")
    ordered = sorted(by_key.values(), key=lambda entry: (entry[3], entry[0]["mask_key"]))
    if len(ordered) <= limit:
        return [entry[0] for entry in ordered]

    width = next(iter(widths))
    selected: list[tuple[dict[str, Any], tuple[int, ...], int, int, int, int]] = []
    selected_keys: set[str] = set()

    def add(entry: tuple[dict[str, Any], tuple[int, ...], int, int, int, int] | None) -> None:
        if entry is not None and entry[0]["mask_key"] not in selected_keys and len(selected) < limit:
            selected.append(entry)
            selected_keys.add(entry[0]["mask_key"])

    add(ordered[0])
    add(by_key.get("0" * width))
    add(by_key.get("1" * width))

    while len(selected) < limit:
        selected_integers = [entry[2] for entry in selected]
        selected_on = Counter(entry[3] for entry in selected)
        selected_transitions = Counter(entry[4] for entry in selected)
        transition_values = tuple(selected_transitions)

        def score(entry: tuple[dict[str, Any], tuple[int, ...], int, int, int, int]) -> tuple[int, int, int, int, int]:
            _, _, mask_integer, on_count, transitions, tie = entry
            min_hamming = min((mask_integer ^ chosen).bit_count() for chosen in selected_integers)
            min_transition_gap = min(abs(transitions - value) for value in transition_values)
            return (
                -selected_on[on_count],
                min_hamming,
                -selected_transitions[transitions],
                min_transition_gap,
                tie,
            )

        candidates = [entry for entry in ordered if entry[0]["mask_key"] not in selected_keys]
        if not candidates:
            raise RuntimeError(f"route selection exhausted before reaching {limit} for {uid}")
        add(max(candidates, key=score))
    return [entry[0] for entry in selected]


def canonical_segment_targets(mask: Sequence[int]) -> dict[str, list[int]]:
    """Return the canonical maximal-run representation used by the POLAR baseline."""
    values = [int(bit) for bit in mask]
    if not values or any(bit not in (0, 1) for bit in values):
        raise ValueError("mask must be a nonempty binary sequence")
    starts = [0]
    for index in range(1, len(values)):
        if values[index] != values[index - 1]:
            starts.append(index)
    boundaries = [0] * len(values)
    operations = [-100] * len(values)
    for start in starts:
        if start:
            boundaries[start] = 1
        operations[start] = values[start]
    return {
        "segment_starts": starts,
        "segment_actions": [values[start] for start in starts],
        "boundary_targets": boundaries,
        "operation_targets": operations,
    }
