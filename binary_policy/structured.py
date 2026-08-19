"""P12 lossless maximal-run representation and exact valid-set objective."""

from __future__ import annotations

from collections.abc import Sequence
from collections import Counter
import math
from statistics import median

import torch
import torch.nn.functional as F


def _distribution(values: list[int]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    ordered = sorted(values)

    def quantile(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    return {
        "mean": sum(ordered) / len(ordered),
        "minimum": min(ordered),
        "q25": quantile(0.25),
        "median": float(median(ordered)),
        "q75": quantile(0.75),
        "q90": quantile(0.90),
        "q95": quantile(0.95),
        "maximum": max(ordered),
    }


def summarize_segment_geometry(masks: Sequence[Sequence[int | bool]]) -> dict:
    """Summarize maximal-run geometry for a nonempty complete-mask collection."""
    if not masks:
        raise ValueError("masks cannot be empty")
    segments_per_mask: list[int] = []
    transitions: list[int] = []
    on_segments_per_mask: list[int] = []
    off_segments_per_mask: list[int] = []
    segment_lengths: list[int] = []
    on_lengths: list[int] = []
    off_lengths: list[int] = []
    for mask in masks:
        boundaries, operations = mask_to_p12_targets(mask)
        starts = [index for index, value in enumerate(boundaries) if value]
        current_on = 0
        current_off = 0
        for run_index, start in enumerate(starts):
            stop = starts[run_index + 1] if run_index + 1 < len(starts) else len(mask)
            length = stop - start
            action = operations[start]
            segment_lengths.append(length)
            if action == 1:
                current_on += 1
                on_lengths.append(length)
            else:
                current_off += 1
                off_lengths.append(length)
        segments_per_mask.append(len(starts))
        transitions.append(len(starts) - 1)
        on_segments_per_mask.append(current_on)
        off_segments_per_mask.append(current_off)

    def histogram(values: list[int]) -> dict[str, int]:
        return {str(key): count for key, count in sorted(Counter(values).items())}

    return {
        "masks": len(masks),
        "segments": _distribution(segments_per_mask),
        "transitions": _distribution(transitions),
        "on_segments": _distribution(on_segments_per_mask),
        "off_segments": _distribution(off_segments_per_mask),
        "segment_lengths": _distribution(segment_lengths),
        "on_segment_lengths": _distribution(on_lengths),
        "off_segment_lengths": _distribution(off_lengths) if off_lengths else None,
        "segment_count_histogram": histogram(segments_per_mask),
        "segment_length_histogram": histogram(segment_lengths),
        "on_segment_length_histogram": histogram(on_lengths),
        "off_segment_length_histogram": histogram(off_lengths),
        "fraction_at_most_segments": {
            str(threshold): sum(value <= threshold for value in segments_per_mask) / len(masks)
            for threshold in (2, 4, 6, 8)
        },
    }


def mask_to_p12_targets(mask: Sequence[int | bool]) -> tuple[list[int], list[int]]:
    """Encode a nonempty binary mask with explicit maximal-run starts."""
    values = [int(value) for value in mask]
    if not values or any(value not in (0, 1) for value in values):
        raise ValueError("mask must be a nonempty binary sequence")
    boundaries = [0] * len(values)
    operations = [-100] * len(values)
    for index, value in enumerate(values):
        if index == 0 or value != values[index - 1]:
            boundaries[index] = 1
            operations[index] = value
    return boundaries, operations


def p12_targets_to_mask(boundaries: Sequence[int], operations: Sequence[int]) -> list[int]:
    """Decode the unique P12 canonical targets to their complete binary mask."""
    if len(boundaries) != len(operations) or not boundaries:
        raise ValueError("boundaries and operations must be nonempty and equal length")
    if int(boundaries[0]) != 1:
        raise ValueError("layer zero must be an explicit segment start")
    starts = [index for index, value in enumerate(boundaries) if int(value) == 1]
    if any(int(value) not in (0, 1) for value in boundaries):
        raise ValueError("boundaries must be binary")
    output = [0] * len(boundaries)
    previous_action = None
    for run_index, start in enumerate(starts):
        stop = starts[run_index + 1] if run_index + 1 < len(starts) else len(output)
        action = int(operations[start])
        if action not in (0, 1):
            raise ValueError(f"segment start {start} has invalid operation {action}")
        if previous_action is not None and action == previous_action:
            raise ValueError("canonical maximal runs must alternate operations")
        output[start:stop] = [action] * (stop - start)
        previous_action = action
    return output


def structured_route_log_probability(
    boundary_logits: torch.Tensor,
    operation_logits: torch.Tensor,
    boundary_targets: torch.Tensor,
    operation_targets: torch.Tensor,
) -> torch.Tensor:
    """Return log P(canonical route | input) with shape ``[B,V]``."""
    if boundary_logits.ndim != 2:
        raise ValueError("boundary_logits must have shape [B,L]")
    if operation_logits.shape != (*boundary_logits.shape, 2):
        raise ValueError("operation_logits must have shape [B,L,2]")
    if boundary_targets.ndim == 2:
        boundary_targets = boundary_targets.unsqueeze(1)
        operation_targets = operation_targets.unsqueeze(1)
    expected = (boundary_logits.shape[0], boundary_targets.shape[1], boundary_logits.shape[1])
    if boundary_targets.shape != expected or operation_targets.shape != expected:
        raise ValueError("structured targets must have shape [B,V,L]")
    targets = boundary_targets.to(device=boundary_logits.device, dtype=boundary_logits.dtype)
    if bool(((targets != 0) & (targets != 1)).any().item()):
        raise ValueError("boundary targets must be binary")
    if bool((targets[:, :, 0] != 1).any().item()):
        raise ValueError("every canonical route must start at layer zero")
    log_on = F.logsigmoid(boundary_logits).unsqueeze(1)
    log_off = F.logsigmoid(-boundary_logits).unsqueeze(1)
    boundary_log_probability = (targets * log_on + (1.0 - targets) * log_off).sum(dim=-1)

    operations = operation_targets.to(device=operation_logits.device, dtype=torch.long)
    start_mask = targets.bool()
    if bool(((operations[start_mask] < 0) | (operations[start_mask] > 1)).any().item()):
        raise ValueError("segment-start operations must be ON or OFF")
    safe_operations = torch.where(start_mask, operations, torch.zeros_like(operations))
    operation_log_probs = F.log_softmax(operation_logits, dim=-1).unsqueeze(1).expand(
        -1, targets.shape[1], -1, -1
    )
    selected = operation_log_probs.gather(-1, safe_operations.unsqueeze(-1)).squeeze(-1)
    operation_log_probability = torch.where(start_mask, selected, torch.zeros_like(selected)).sum(dim=-1)
    return boundary_log_probability + operation_log_probability


def structured_valid_set_nll(
    boundary_logits: torch.Tensor,
    operation_logits: torch.Tensor,
    boundary_targets: torch.Tensor,
    operation_targets: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    route_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact weighted one-of-valid-set NLL over canonical structured routes."""
    route_log_probability = structured_route_log_probability(
        boundary_logits, operation_logits, boundary_targets, operation_targets
    )
    batch_size, num_routes = route_log_probability.shape
    if valid_mask is None:
        valid_mask = torch.ones(
            batch_size, num_routes, dtype=torch.bool, device=boundary_logits.device
        )
    else:
        valid_mask = valid_mask.to(device=boundary_logits.device, dtype=torch.bool)
        if valid_mask.shape != route_log_probability.shape:
            raise ValueError("valid_mask must have shape [B,V]")
    if bool((valid_mask.sum(dim=1) == 0).any().item()):
        raise ValueError("every input must contain a valid route")
    if route_weights is None:
        weights = valid_mask.to(boundary_logits.dtype)
    else:
        weights = route_weights.to(device=boundary_logits.device, dtype=boundary_logits.dtype)
        if weights.shape != valid_mask.shape:
            raise ValueError("route_weights must have shape [B,V]")
        if bool((weights[valid_mask] <= 0).any().item()):
            raise ValueError("valid route weights must be positive")
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
    weights = weights / weights.sum(dim=1, keepdim=True)
    weighted = route_log_probability + torch.where(
        valid_mask, weights.log(), torch.full_like(weights, -torch.inf)
    )
    return -torch.logsumexp(weighted, dim=1).mean()


def decode_structured_top1(
    boundary_logits: torch.Tensor, operation_logits: torch.Tensor
) -> list[dict]:
    """Apply the frozen P12 threshold/argmax decoder and return complete masks."""
    if boundary_logits.ndim != 2 or operation_logits.shape != (*boundary_logits.shape, 2):
        raise ValueError("structured logits must have shapes [B,L] and [B,L,2]")
    boundaries = (boundary_logits >= 0).to(torch.int64).cpu()
    boundaries[:, 0] = 1
    operations = operation_logits.argmax(dim=-1).to(torch.int64).cpu()
    output = []
    for row_boundaries, row_operations in zip(boundaries.tolist(), operations.tolist()):
        starts = [index for index, value in enumerate(row_boundaries) if value]
        mask = [0] * len(row_boundaries)
        for run_index, start in enumerate(starts):
            stop = starts[run_index + 1] if run_index + 1 < len(starts) else len(mask)
            mask[start:stop] = [row_operations[start]] * (stop - start)
        output.append(
            {
                "mask": tuple(mask),
                "boundaries": tuple(row_boundaries),
                "operations": tuple(row_operations[index] if row_boundaries[index] else -100 for index in range(len(mask))),
                "predicted_segments": len(starts),
            }
        )
    return output


@torch.no_grad()
def structured_batch_metrics(
    boundary_logits: torch.Tensor,
    operation_logits: torch.Tensor,
    valid_masks: torch.Tensor,
    boundary_targets: torch.Tensor,
    operation_targets: torch.Tensor,
    valid_mask: torch.Tensor,
    route_weights: torch.Tensor,
) -> dict:
    """Complete-mask and weighted native diagnostics for one P12 batch."""
    from collections import Counter

    from .evaluation import mask_diversity_metrics, nearest_valid_hamming

    decoded = decode_structured_top1(boundary_logits, operation_logits)
    predicted_boundaries = (boundary_logits >= 0).to(torch.int64).cpu()
    predicted_boundaries[:, 0] = 1
    predicted_operations = operation_logits.argmax(dim=-1).to(torch.int64).cpu()
    target_boundaries = boundary_targets.to(torch.int64).cpu()
    target_operations = operation_targets.to(torch.int64).cpu()
    valid = valid_mask.bool().cpu()
    weights = route_weights.float().cpu()
    masks = valid_masks.to(torch.int64).cpu()

    mask_counts: Counter[tuple[int, ...]] = Counter()
    top1_hits = 0
    hamming = 0.0
    predicted_segments = 0.0
    boundary_correct = 0.0
    boundary_total = 0.0
    boundary_tp = 0.0
    boundary_fp = 0.0
    boundary_fn = 0.0
    operation_correct = 0.0
    operation_total = 0.0
    for sample_index, prediction in enumerate(decoded):
        predicted_mask = prediction["mask"]
        mask_counts[predicted_mask] += 1
        predicted_segments += prediction["predicted_segments"]
        route_indices = valid[sample_index].nonzero(as_tuple=False).flatten().tolist()
        target_masks = [masks[sample_index, route_index].tolist() for route_index in route_indices]
        target_set = {tuple(mask) for mask in target_masks}
        top1_hits += int(predicted_mask in target_set)
        hamming += nearest_valid_hamming(predicted_mask, target_masks)
        normalizer = float(weights[sample_index, route_indices].sum())
        for route_index in route_indices:
            weight = float(weights[sample_index, route_index]) / normalizer
            target_boundary = target_boundaries[sample_index, route_index]
            predicted_boundary = predicted_boundaries[sample_index]
            boundary_correct += weight * float((predicted_boundary == target_boundary).sum())
            boundary_total += weight * target_boundary.numel()
            boundary_tp += weight * float(((predicted_boundary == 1) & (target_boundary == 1)).sum())
            boundary_fp += weight * float(((predicted_boundary == 1) & (target_boundary == 0)).sum())
            boundary_fn += weight * float(((predicted_boundary == 0) & (target_boundary == 1)).sum())
            starts = target_boundary.bool()
            operation_correct += weight * float(
                (predicted_operations[sample_index, starts] == target_operations[sample_index, route_index, starts]).sum()
            )
            operation_total += weight * float(starts.sum())
    count = len(decoded)
    return {
        "top1_valid_route_coverage": top1_hits / count,
        # P12 freezes one deterministic decode and forbids beam/candidate search.
        "topk_valid_route_coverage": top1_hits / count,
        "topk_candidate_count": 1,
        "top5_available": False,
        "nearest_valid_hamming": hamming / count,
        "average_predicted_segments": predicted_segments / count,
        "boundary_accuracy": boundary_correct / boundary_total,
        "boundary_precision": boundary_tp / max(boundary_tp + boundary_fp, 1e-12),
        "boundary_recall": boundary_tp / max(boundary_tp + boundary_fn, 1e-12),
        "segment_operation_accuracy_at_gt_boundaries": operation_correct / operation_total,
        "_boundary_correct": boundary_correct,
        "_boundary_total": boundary_total,
        "_boundary_tp": boundary_tp,
        "_boundary_fp": boundary_fp,
        "_boundary_fn": boundary_fn,
        "_operation_correct": operation_correct,
        "_operation_total": operation_total,
        "_top1_mask_counts": {
            "".join(map(str, mask)): count for mask, count in sorted(mask_counts.items())
        },
        **mask_diversity_metrics(mask_counts),
    }
