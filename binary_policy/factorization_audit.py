"""Outcome-blind label-geometry checks for direct versus run-length heads."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .decode import topk_factorized_masks
from .segmented import mask_to_canonical_targets


def _smoothed_probability(positive: int, total: int, alpha: float = 0.5) -> float:
    return (positive + alpha) / (total + 2.0 * alpha)


def empirical_direct_topk(valid_masks: Sequence[Sequence[int]], top_k: int) -> list[tuple[int, ...]]:
    if not valid_masks:
        return []
    values = torch.tensor(valid_masks, dtype=torch.float64)
    probabilities = (values.sum(dim=0) + 0.5) / (values.shape[0] + 1.0)
    logits = torch.logit(probabilities)
    return [candidate.mask for candidate in topk_factorized_masks(logits, top_k=top_k)[0]]


def empirical_segmented_topk(valid_masks: Sequence[Sequence[int]], top_k: int) -> list[tuple[int, ...]]:
    """Top-k canonical masks under empirical independent boundary/op scores."""
    if not valid_masks:
        return []
    depth = len(valid_masks[0])
    targets = [mask_to_canonical_targets(mask) for mask in valid_masks]
    boundary_p = [0.0] * depth
    op_p = [0.5] * depth
    for layer in range(1, depth):
        boundary_p[layer] = _smoothed_probability(
            sum(boundaries[layer] for boundaries, _ in targets), len(targets)
        )
    for layer in range(depth):
        observed = [ops[layer] for _, ops in targets if ops[layer] != -100]
        if observed:
            op_p[layer] = _smoothed_probability(sum(observed), len(observed))

    # Keep top-k partial masks per last action. This is exact for this first-order
    # canonical score because future contributions depend only on that action.
    beams: dict[int, list[tuple[tuple[int, ...], float]]] = {0: [], 1: []}
    for action in (0, 1):
        probability = op_p[0] if action else 1.0 - op_p[0]
        beams[action] = [((action,), math.log(probability))]
    for layer in range(1, depth):
        next_beams: dict[int, list[tuple[tuple[int, ...], float]]] = {0: [], 1: []}
        for previous, candidates in beams.items():
            for prefix, score in candidates:
                for action in (0, 1):
                    boundary = int(action != previous)
                    p_boundary = boundary_p[layer]
                    addition = math.log(p_boundary if boundary else 1.0 - p_boundary)
                    if boundary:
                        p_action = op_p[layer]
                        addition += math.log(p_action if action else 1.0 - p_action)
                    next_beams[action].append((prefix + (action,), score + addition))
        for action in (0, 1):
            next_beams[action].sort(key=lambda item: (-item[1], item[0]))
            next_beams[action] = next_beams[action][:top_k]
        beams = next_beams
    merged = beams[0] + beams[1]
    merged.sort(key=lambda item: (-item[1], item[0]))
    return [mask for mask, _ in merged[:top_k]]


def factorization_coverage(valid_masks: Sequence[Sequence[int]], top_k_values: Sequence[int] = (1, 5, 10)) -> dict:
    valid = {tuple(int(value) for value in mask) for mask in valid_masks}
    output = {}
    for top_k in top_k_values:
        direct = empirical_direct_topk(valid_masks, top_k)
        segmented = empirical_segmented_topk(valid_masks, top_k)
        output[str(top_k)] = {
            "direct_hit": any(mask in valid for mask in direct),
            "segmented_hit": any(mask in valid for mask in segmented),
        }
    return output


def direct_representation_gate(
    geometry_summary: dict,
    *,
    macro_deficit_tolerance: float = 0.02,
    cell_deficit_tolerance: float = 0.05,
) -> dict:
    """Apply the prospective direct-versus-segmented top-5 coverage gate."""
    cell_rows = {}
    direct_rates = []
    segmented_rates = []
    for cell_name, cell in sorted(geometry_summary["cells"].items()):
        denominator = int(cell["samples_with_valid_route"])
        if denominator == 0:
            raise ValueError(f"cell {cell_name} has no records with valid routes")
        counts = cell["factorization_coverage"]["5"]
        direct = counts["direct_hits"] / denominator
        segmented = counts["segmented_hits"] / denominator
        deficit = segmented - direct
        cell_rows[cell_name] = {
            "denominator": denominator,
            "direct_top5_hit_rate": direct,
            "segmented_top5_hit_rate": segmented,
            "direct_deficit": deficit,
            "passed": deficit <= cell_deficit_tolerance,
        }
        direct_rates.append(direct)
        segmented_rates.append(segmented)
    macro_direct = sum(direct_rates) / len(direct_rates)
    macro_segmented = sum(segmented_rates) / len(segmented_rates)
    return {
        "macro_direct_top5_hit_rate": macro_direct,
        "macro_segmented_top5_hit_rate": macro_segmented,
        "macro_direct_deficit": macro_segmented - macro_direct,
        "macro_deficit_tolerance": macro_deficit_tolerance,
        "cell_deficit_tolerance": cell_deficit_tolerance,
        "cells": cell_rows,
        "passed": (
            macro_segmented - macro_direct <= macro_deficit_tolerance
            and all(row["passed"] for row in cell_rows.values())
        ),
    }
