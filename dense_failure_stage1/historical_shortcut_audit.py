"""Small deterministic statistics for the historical Stage-1 shortcut audit.

The audit deliberately uses simple linear centroid probes.  They expose whether
information is linearly available without introducing a new trained Stage-1
model or depending on scikit-learn.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Mapping, Sequence

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def binary_auroc(labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    """Return tie-aware binary AUROC, or NaN when either class is absent."""

    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(s)
    y, s = y[valid], s[valid]
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _rankdata(s)
    rank_sum = float(np.sum(ranks[y == 1]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    """Return threshold-based average precision with score ties grouped."""

    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(s)
    y, s = y[valid], s[valid]
    positives = int(np.sum(y == 1))
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ordered_y, ordered_s = y[order], s[order]
    true_positives = 0
    predicted = 0
    area = 0.0
    start = 0
    while start < len(ordered_y):
        end = start + 1
        while end < len(ordered_y) and ordered_s[end] == ordered_s[start]:
            end += 1
        gained = int(np.sum(ordered_y[start:end]))
        true_positives += gained
        predicted += end - start
        area += (gained / positives) * (true_positives / predicted)
        start = end
    return float(area)


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    if len(first) < 3 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return float("nan")
    return float(np.corrcoef(_rankdata(first), _rankdata(second))[0, 1])


def deterministic_group_folds(groups: Sequence[str], *, seed: int, n_folds: int) -> np.ndarray:
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    mapping = {
        group: int(sha256(f"{seed}:{group}".encode()).hexdigest(), 16) % n_folds
        for group in set(groups)
    }
    return np.asarray([mapping[group] for group in groups], dtype=np.int64)


@dataclass(frozen=True)
class CentroidProbe:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    midpoint: np.ndarray

    def score(self, features: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(features, dtype=np.float32) - self.mean) / self.scale
        return (standardized - self.midpoint) @ self.weight


def fit_centroid_probe(features: np.ndarray, labels: Sequence[int] | np.ndarray) -> CentroidProbe:
    """Fit a train-standardized nearest-centroid linear discriminant."""

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("features and labels have incompatible shapes")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("both binary classes are required")
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, np.float32(1e-6))
    z = (x - mean) / scale
    negative = z[y == 0].mean(axis=0)
    positive = z[y == 1].mean(axis=0)
    weight = (positive - negative).astype(np.float32)
    norm = float(np.linalg.norm(weight))
    if norm > 0:
        weight /= norm
    midpoint = (0.5 * (positive + negative)).astype(np.float32)
    return CentroidProbe(mean=mean, scale=scale, weight=weight, midpoint=midpoint)


def exact_stratum_match(
    rows: Iterable[Mapping], *, label_key: str, stratum_key: str, seed: int
) -> list[dict]:
    """Select equal deterministic class counts within every usable stratum."""

    cells: dict[object, dict[int, list[Mapping]]] = defaultdict(lambda: {0: [], 1: []})
    for row in rows:
        label = int(bool(row[label_key]))
        cells[row[stratum_key]][label].append(row)
    selected: list[dict] = []
    for stratum, classes in sorted(cells.items(), key=lambda item: str(item[0])):
        count = min(len(classes[0]), len(classes[1]))
        for label in (0, 1):
            ordered = sorted(
                classes[label],
                key=lambda row: sha256(
                    f"{seed}:{stratum}:{label}:{row['uid']}".encode()
                ).hexdigest(),
            )
            selected.extend(dict(row) for row in ordered[:count])
    return sorted(selected, key=lambda row: str(row["uid"]))
