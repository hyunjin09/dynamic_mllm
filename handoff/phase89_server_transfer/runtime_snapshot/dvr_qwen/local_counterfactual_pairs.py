"""Utilities for exact Hamming-1 visual-route counterfactuals."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


OUTCOMES = ("off_win", "on_win", "both_correct", "both_wrong")


def hamming_one_edges(routes: Iterable[dict[str, Any]], num_layers: int) -> list[dict[str, Any]]:
    """Return each observed off/on Hamming-1 edge exactly once."""
    by_mask = {str(route["mask_key"]): route for route in routes}
    edges: list[dict[str, Any]] = []
    for off_mask, off_route in by_mask.items():
        if len(off_mask) != num_layers:
            raise ValueError(f"mask length {len(off_mask)} != {num_layers}: {off_mask}")
        for layer, bit in enumerate(off_mask):
            if bit != "0":
                continue
            on_mask = f"{off_mask[:layer]}1{off_mask[layer + 1:]}"
            on_route = by_mask.get(on_mask)
            if on_route is not None:
                edges.append(
                    {
                        "layer_index": layer,
                        "off_mask_key": off_mask,
                        "on_mask_key": on_mask,
                        "off_route": off_route,
                        "on_route": on_route,
                    }
                )
    return edges


def classify_edge(off_correct: bool, on_correct: bool) -> str:
    if off_correct and not on_correct:
        return "off_win"
    if on_correct and not off_correct:
        return "on_win"
    if off_correct and on_correct:
        return "both_correct"
    return "both_wrong"


def conservative_action(outcome: str) -> str:
    """Map an observed edge outcome to a correctness-first local action.

    Turning visual context off is supervised only when it strictly improves
    correctness. All other observed outcomes retain the dense ON fallback.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown Hamming-1 outcome: {outcome!r}")
    return "off" if outcome == "off_win" else "on"


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (math.nan, math.nan)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))


def classify_local_action(
    *,
    off_wins: int,
    on_wins: int,
    minimum_support: int = 30,
    core_threshold: float = 0.8,
    variable_floor: float = 0.2,
) -> str:
    """Classify a layer's local correctness direction without hiding ambiguity."""
    decisive = off_wins + on_wins
    if decisive < minimum_support:
        return "insufficient_support"
    off_rate = off_wins / decisive
    low, high = wilson_interval(off_wins, decisive)
    if off_rate >= core_threshold and low > 0.5:
        return "local_core_off"
    if off_rate <= 1.0 - core_threshold and high < 0.5:
        return "local_core_on"
    if variable_floor <= off_rate <= 1.0 - variable_floor:
        return "variable"
    return "intermediate"


def dominant_action(off_wins: int, on_wins: int) -> str | None:
    if off_wins > on_wins:
        return "off"
    if on_wins > off_wins:
        return "on"
    return None


def pre_action_context_key(row: dict[str, Any]) -> tuple[str, int, str]:
    """Key all information observable before the indexed layer action."""
    return (str(row["uid"]), int(row["layer_index"]), str(row["common_prefix_mask"]))


def partition_pre_action_contexts(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate identifiable rows from labels that conflict at one context.

    Opposite outcomes under an identical pre-action context depend on unobserved
    future route actions. They cannot directly supervise a causal local policy.
    """
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(pre_action_context_key(row), []).append(row)
    identifiable: list[dict[str, Any]] = []
    future_conflicts: list[dict[str, Any]] = []
    for context_rows in grouped.values():
        labels = {str(row["label_action"]) for row in context_rows}
        target = future_conflicts if len(labels) > 1 else identifiable
        target.extend(context_rows)
    return identifiable, future_conflicts
