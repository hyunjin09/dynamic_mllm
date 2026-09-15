"""Losses and metrics for strict layer-local counterfactual supervision."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def uid_normalized_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    uid_group: torch.Tensor,
) -> torch.Tensor:
    """Average weighted target loss inside each UID, then average UIDs."""
    if not (logits.shape == labels.shape == weights.shape == uid_group.shape):
        raise ValueError("logits, labels, weights, and uid_group must have identical shapes")
    if logits.numel() == 0:
        raise ValueError("cannot compute loss for an empty batch")
    if bool((weights <= 0).any().item()):
        raise ValueError("all target weights must be positive")
    groups = int(uid_group.max().item()) + 1
    per_target = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    weighted_sum = logits.new_zeros(groups).index_add_(0, uid_group.long(), per_target * weights)
    weight_sum = logits.new_zeros(groups).index_add_(0, uid_group.long(), weights)
    return (weighted_sum / weight_sum.clamp_min(1e-12)).mean()


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def _average_precision(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    positives = int(labels.sum().item())
    if positives == 0:
        return math.nan
    order = torch.argsort(probabilities, descending=True, stable=True)
    ordered = labels[order].float()
    precision = ordered.cumsum(0) / torch.arange(1, len(ordered) + 1, dtype=torch.float32)
    return float((precision * ordered).sum().item() / positives)


def _auroc(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    num_positive = int(labels.sum().item())
    num_negative = int((~labels).sum().item())
    if num_positive == 0 or num_negative == 0:
        return math.nan
    order = torch.argsort(probabilities, stable=True)
    sorted_probabilities = probabilities[order]
    sorted_labels = labels[order]
    _, counts = torch.unique_consecutive(sorted_probabilities, return_counts=True)
    rank_sum = 0.0
    offset = 0
    for count in counts.tolist():
        average_rank = (offset + 1 + offset + count) / 2.0
        rank_sum += average_rank * int(sorted_labels[offset : offset + count].sum().item())
        offset += count
    return (rank_sum - num_positive * (num_positive + 1) / 2.0) / (num_positive * num_negative)


def _ece(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = (probabilities >= low) & (
            probabilities <= high if index == bins - 1 else probabilities < high
        )
        if bool(selected.any().item()):
            confidence = float(probabilities[selected].mean().item())
            accuracy = float(labels[selected].float().mean().item())
            result += float(selected.float().mean().item()) * abs(confidence - accuracy)
    return result


def binary_metrics(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> dict[str, float | int]:
    logits = logits.detach().float().cpu().flatten()
    labels_bool = labels.detach().cpu().flatten().bool()
    if logits.shape != labels_bool.shape or logits.numel() == 0:
        raise ValueError("non-empty logits and labels must have identical shapes")
    predicted_on = logits >= float(threshold)
    tp = int((predicted_on & labels_bool).sum().item())
    tn = int((~predicted_on & ~labels_bool).sum().item())
    fp = int((predicted_on & ~labels_bool).sum().item())
    fn = int((~predicted_on & labels_bool).sum().item())
    on_recall = _safe_ratio(tp, tp + fn)
    off_recall = _safe_ratio(tn, tn + fp)
    balanced = (on_recall + off_recall) / 2.0
    probabilities = torch.sigmoid(logits)
    return {
        "n": int(logits.numel()),
        "on": int(labels_bool.sum().item()),
        "off": int((~labels_bool).sum().item()),
        "accuracy": (tp + tn) / logits.numel(),
        "balanced_accuracy": balanced,
        "on_recall": on_recall,
        "off_recall": off_recall,
        "on_precision": _safe_ratio(tp, tp + fp),
        "off_precision": _safe_ratio(tn, tn + fn),
        "auroc_on": _auroc(probabilities, labels_bool),
        "auprc_on": _average_precision(probabilities, labels_bool),
        "auprc_off": _average_precision(1.0 - probabilities, ~labels_bool),
        "ece": _ece(probabilities, labels_bool),
        "predicted_off_rate": float((~predicted_on).float().mean().item()),
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def calibrate_on_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    minimum_on_recall: float = 0.98,
) -> dict[str, Any]:
    """Maximize balanced accuracy while satisfying a preservation floor."""
    if not 0.0 <= minimum_on_recall <= 1.0:
        raise ValueError("minimum_on_recall must be in [0, 1]")
    values = logits.detach().float().cpu().flatten()
    labels_bool = labels.detach().cpu().flatten().bool()
    if values.shape != labels_bool.shape or values.numel() == 0:
        raise ValueError("non-empty logits and labels must have identical shapes")
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    sorted_labels = labels_bool[order]
    unique, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    total_on = int(sorted_labels.sum().item())
    total_off = int((~sorted_labels).sum().item())
    prefix_on = torch.cat([torch.zeros(1, dtype=torch.long), sorted_labels.long().cumsum(0)])
    prefix_off = torch.cat([torch.zeros(1, dtype=torch.long), (~sorted_labels).long().cumsum(0)])

    feasible: list[tuple[float, float, float, float]] = []
    positions = [0]
    offset = 0
    for count in counts.tolist():
        offset += count
        positions.append(offset)
    thresholds = [float(unique[0] - 1.0), *[float(value) for value in unique[1:]], float(unique[-1] + 1.0)]
    # positions[k] is the number of values strictly below thresholds[k].
    for threshold, position in zip(thresholds, positions):
        false_negative = int(prefix_on[position].item())
        true_negative = int(prefix_off[position].item())
        on_recall = _safe_ratio(total_on - false_negative, total_on)
        off_recall = _safe_ratio(true_negative, total_off)
        balanced = (on_recall + off_recall) / 2.0
        if on_recall + 1e-12 >= minimum_on_recall:
            feasible.append((balanced, off_recall, threshold, on_recall))
    if not feasible:
        raise RuntimeError("no threshold satisfies the requested ON-recall floor")
    _, _, threshold, _ = max(feasible, key=lambda item: (item[0], item[1], item[2]))
    return binary_metrics(
        logits=values,
        labels=labels_bool,
        threshold=threshold,
    )
