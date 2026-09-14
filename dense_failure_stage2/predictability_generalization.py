"""Label-blind semantic split and question-neighbor helpers for Step-C."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import unicodedata
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def normalize_question(value: str) -> str:
    """Apply the frozen, label-blind normalization used by the encoder."""

    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last non-padding token for either left- or right-padded batches."""

    if hidden.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("hidden/mask ranks must be 3 and 2")
    if hidden.shape[:2] != attention_mask.shape:
        raise ValueError("hidden/mask shapes differ")
    if not bool((attention_mask.sum(dim=1) > 0).all()):
        raise ValueError("cannot pool an empty token sequence")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    positions = positions.unsqueeze(0).expand_as(attention_mask)
    indices = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]


def equal_frequency_bins(values: Sequence[float], bins: int = 5) -> np.ndarray:
    """Assign deterministic equal-frequency bins Q1..Qk (ties by row index)."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < bins:
        raise ValueError("not enough one-dimensional values for requested bins")
    order = np.lexsort((np.arange(len(array)), array))
    output = np.empty(len(array), dtype=np.int64)
    for rank, index in enumerate(order):
        output[index] = min(bins - 1, rank * bins // len(array)) + 1
    return output


def cosine_nearest(
    query: np.ndarray, reference: np.ndarray, *, block_size: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Exact maximum cosine similarity for already normalized embeddings."""

    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("query/reference embedding shapes differ")
    if not len(reference):
        raise ValueError("nearest-neighbor reference is empty")
    maxima = np.full(len(query), -np.inf, dtype=np.float32)
    indices = np.full(len(query), -1, dtype=np.int64)
    for start in range(0, len(reference), block_size):
        scores = query @ reference[start : start + block_size].T
        local = scores.argmax(axis=1)
        values = scores[np.arange(len(query)), local]
        replace = values > maxima
        maxima[replace] = values[replace]
        indices[replace] = start + local[replace]
    return maxima, indices


def exact_knn_prediction(
    query: np.ndarray,
    reference: np.ndarray,
    reference_targets: Sequence[float],
    *,
    k: int = 5,
    block_size: int = 512,
) -> np.ndarray:
    """Exact cosine kNN mean target, with deterministic index tie-breaking."""

    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    targets = np.asarray(reference_targets, dtype=np.float64)
    if len(reference) != len(targets) or len(reference) < k:
        raise ValueError("invalid kNN reference population")
    output = np.empty(len(query), dtype=np.float64)
    reference_index = np.arange(len(reference), dtype=np.int64)
    for start in range(0, len(query), block_size):
        scores = query[start : start + block_size] @ reference.T
        for local, row in enumerate(scores):
            order = np.lexsort((reference_index, -row))[:k]
            output[start + local] = float(targets[order].mean())
    return output


def spherical_kmeans(
    embeddings: np.ndarray,
    *,
    clusters: int,
    seed: int,
    maximum_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Deterministic spherical k-means with farthest-first initialization."""

    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or not clusters <= len(values):
        raise ValueError("invalid embedding matrix or cluster count")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-4):
        raise ValueError("spherical k-means requires normalized embeddings")
    first = int(seed % len(values))
    selected = [first]
    best = values @ values[first]
    for _ in range(1, clusters):
        candidate = int(np.argmin(best))
        selected.append(candidate)
        best = np.maximum(best, values @ values[candidate])
    centers = values[np.asarray(selected)].copy()
    assignments = np.full(len(values), -1, dtype=np.int64)
    for iteration in range(1, maximum_iterations + 1):
        updated = (values @ centers.T).argmax(axis=1).astype(np.int64)
        if np.array_equal(updated, assignments):
            return assignments, centers, iteration - 1
        assignments = updated
        for cluster in range(clusters):
            members = values[assignments == cluster]
            if len(members):
                center = members.mean(axis=0)
            else:
                similarity = (values * centers[assignments]).sum(axis=1)
                center = values[int(np.argmin(similarity))]
            norm = float(np.linalg.norm(center))
            if norm == 0:
                raise RuntimeError("zero spherical cluster center")
            centers[cluster] = center / norm
    return assignments, centers, maximum_iterations


def assign_clusters_to_folds(
    cluster_rows: Sequence[Mapping[str, Any]], *, folds: int, seed: int
) -> dict[int, int]:
    """Greedily balance clusters using only group count, dataset, and P90 support."""

    if folds < 2:
        raise ValueError("at least two folds are required")
    totals = Counter()
    for row in cluster_rows:
        totals["groups"] += int(row["groups"])
        totals[f"dataset:{row['dataset']}"] += int(row["groups"])
        totals["p90"] += int(row["p90_groups"])
    keys = sorted(totals)
    target = {key: totals[key] / folds for key in keys}
    fold_totals = [Counter() for _ in range(folds)]
    assignment: dict[int, int] = {}
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in cluster_rows:
        grouped[int(row["cluster_id"])].append(row)
    order = sorted(
        grouped,
        key=lambda cluster: (
            -sum(int(row["groups"]) for row in grouped[cluster]),
            sha256(f"{seed}|cluster|{cluster}".encode()).hexdigest(),
        ),
    )
    for cluster in order:
        vector = Counter()
        for row in grouped[cluster]:
            count = int(row["groups"])
            vector["groups"] += count
            vector[f"dataset:{row['dataset']}"] += count
            vector["p90"] += int(row["p90_groups"])
        scores = []
        for fold in range(folds):
            score = 0.0
            for candidate_fold in range(folds):
                for key in keys:
                    proposed = fold_totals[candidate_fold][key]
                    if candidate_fold == fold:
                        proposed += vector[key]
                    score += ((proposed - target[key]) / max(target[key], 1.0)) ** 2
            scores.append((score, fold_totals[fold]["groups"], fold))
        chosen = min(scores)[2]
        assignment[cluster] = chosen
        fold_totals[chosen].update(vector)
    return assignment


def label_blind_inner_roles(
    rows: Sequence[Mapping[str, Any]],
    *,
    eligible_train_uids: set[str],
    test_uids: set[str],
    calibration_fraction: float,
    seed: int,
) -> list[dict[str, str]]:
    """Make group-disjoint fit/calibration/test roles without reading targets."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["image_group_id"])].append(row)
    train_groups = [
        group for group, members in groups.items()
        if any(str(row["uid"]) in eligible_train_uids for row in members)
    ]
    ordered = sorted(
        train_groups,
        key=lambda group: sha256(f"{seed}|calibration|{group}".encode()).hexdigest(),
    )
    calibration_count = max(1, int(round(len(ordered) * calibration_fraction)))
    calibration_groups = set(ordered[:calibration_count])
    output = []
    for row in rows:
        uid = str(row["uid"])
        group = str(row["image_group_id"])
        if uid in test_uids:
            role = "outer_test"
        elif uid in eligible_train_uids:
            role = "calibration" if group in calibration_groups else "fit"
        else:
            role = "excluded"
        output.append({"uid": uid, "image_group_id": group, "role": role})
    return output
