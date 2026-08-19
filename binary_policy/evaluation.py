"""Offline policy-label metrics; these do not replace fresh route execution."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import math

import torch

from .decode import topk_factorized_masks


def mask_diversity_metrics(mask_counts: Counter[tuple[int, ...]]) -> dict:
    """Summarize deterministic top-1 mask diversity from exact counts."""
    total = sum(mask_counts.values())
    if total == 0:
        raise ValueError("mask_counts cannot be empty")
    num_layers = len(next(iter(mask_counts)))
    all_on = (1,) * num_layers
    all_off = (0,) * num_layers
    entropy = -sum((count / total) * math.log(count / total) for count in mask_counts.values())
    serialized = {"".join(map(str, mask)): count for mask, count in sorted(mask_counts.items())}
    return {
        "unique_top1_masks": len(mask_counts),
        "fraction_top1_all_on": mask_counts[all_on] / total,
        "fraction_top1_all_off": mask_counts[all_off] / total,
        "average_predicted_visual_on": sum(sum(mask) * count for mask, count in mask_counts.items()) / total,
        "top1_mask_entropy_nats": entropy,
        "top1_mask_counts": serialized,
        "top5_masks": [
            {"mask": "".join(map(str, mask)), "count": count, "fraction": count / total}
            for mask, count in sorted(mask_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        ],
    }


def nearest_valid_hamming(predicted: Sequence[int], valid_masks: Sequence[Sequence[int]]) -> int:
    if not valid_masks:
        raise ValueError("valid_masks cannot be empty")
    return min(sum(int(left) != int(right) for left, right in zip(predicted, valid)) for valid in valid_masks)


def batch_offline_metrics(
    logits: torch.Tensor,
    valid_masks: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    top_k: int = 5,
) -> dict[str, float]:
    candidates = topk_factorized_masks(logits, top_k=top_k)
    top1_hits = 0
    topk_hits = 0
    hamming = 0.0
    on_count_error = 0.0
    mask_counts: Counter[tuple[int, ...]] = Counter()
    for batch_idx, sample_candidates in enumerate(candidates):
        target_rows = valid_masks[batch_idx, valid_mask[batch_idx]].int().cpu().tolist()
        valid_set = {tuple(row) for row in target_rows}
        top1 = sample_candidates[0].mask
        mask_counts[top1] += 1
        top1_hits += int(top1 in valid_set)
        topk_hits += int(any(candidate.mask in valid_set for candidate in sample_candidates))
        hamming += nearest_valid_hamming(top1, target_rows)
        target_counts = [sum(row) for row in target_rows]
        on_count_error += min(abs(sum(top1) - count) for count in target_counts)
    count = logits.shape[0]
    return {
        "top1_valid_route_coverage": top1_hits / count,
        "topk_valid_route_coverage": topk_hits / count,
        "nearest_valid_hamming": hamming / count,
        "nearest_valid_on_count_error": on_count_error / count,
        **mask_diversity_metrics(mask_counts),
    }
