"""Matched-capacity probes for mandatory four-action deviation boundaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
from torch import nn


class BoundaryProbe(nn.Module):
    """Two-branch MLP used identically for upfront and online features."""

    def __init__(
        self,
        *,
        hidden_width: int,
        num_layers: int,
        branch_width: int,
        layer_embedding_width: int,
        classifier_hidden_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.text_branch = nn.Sequential(
            nn.Linear(hidden_width, branch_width), nn.GELU()
        )
        self.visual_branch = nn.Sequential(
            nn.Linear(hidden_width, branch_width), nn.GELU()
        )
        self.layer_embedding = nn.Embedding(num_layers, layer_embedding_width)
        self.classifier = nn.Sequential(
            nn.Linear(2 * branch_width + layer_embedding_width, classifier_hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_width, 1),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor,
        layer_indices: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat(
            (
                self.text_branch(text_features),
                self.visual_branch(visual_features),
                self.layer_embedding(layer_indices),
            ),
            dim=-1,
        )
        return self.classifier(combined).squeeze(-1)


def binary_auroc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Compute binary AUROC with average ranks for tied probabilities."""

    y = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or scores.shape != y.shape or y.size == 0:
        raise ValueError("labels and probabilities must be equally sized nonempty vectors")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be binary")
    positive = int(y.sum())
    negative = int(y.size - positive)
    if positive == 0 or negative == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(y.size, dtype=np.float64)
    start = 0
    while start < y.size:
        end = start + 1
        while end < y.size and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[y == 1].sum())
    return (positive_rank_sum - positive * (positive + 1) / 2.0) / (
        positive * negative
    )


def binary_classification_metrics(
    labels: Sequence[int], probabilities: Sequence[float]
) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if y.shape != scores.shape or y.ndim != 1 or y.size == 0:
        raise ValueError("labels and probabilities must be equally sized nonempty vectors")
    predictions = scores >= 0.5
    truth = y == 1
    true_positive = int(np.logical_and(predictions, truth).sum())
    false_positive = int(np.logical_and(predictions, ~truth).sum())
    false_negative = int(np.logical_and(~predictions, truth).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "auroc": float(binary_auroc(y, scores)),
        "accuracy": float((predictions == truth).mean()),
        "f1": float(0.0 if denominator == 0 else 2 * true_positive / denominator),
    }


def paired_uid_bootstrap_auc_difference(
    labels: Sequence[int],
    upfront_probabilities: Sequence[float],
    online_probabilities: Sequence[float],
    uid_groups: Sequence[str],
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    """Paired group bootstrap of online-minus-upfront validation AUROC."""

    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    y = np.asarray(labels, dtype=np.int64)
    upfront = np.asarray(upfront_probabilities, dtype=np.float64)
    online = np.asarray(online_probabilities, dtype=np.float64)
    groups = np.asarray(uid_groups, dtype=object)
    if not (y.shape == upfront.shape == online.shape == groups.shape) or y.ndim != 1:
        raise ValueError("paired bootstrap inputs must be equally sized vectors")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups.tolist()):
        grouped[str(group)].append(index)
    unique_groups = sorted(grouped)
    if not unique_groups:
        raise ValueError("paired bootstrap requires at least one UID group")

    point = binary_auroc(y, online) - binary_auroc(y, upfront)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = [index for group in sampled_groups for index in grouped[str(group)]]
        sampled_labels = y[indices]
        online_auc = binary_auroc(sampled_labels, online[indices])
        upfront_auc = binary_auroc(sampled_labels, upfront[indices])
        if np.isfinite(online_auc) and np.isfinite(upfront_auc):
            differences.append(online_auc - upfront_auc)
    if not differences:
        raise RuntimeError("all paired bootstrap draws lacked both classes")
    values = np.asarray(differences, dtype=np.float64)
    return {
        "point_estimate": float(point),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "requested_draws": int(draws),
        "valid_draws": int(values.size),
        "uid_groups": int(len(unique_groups)),
    }
