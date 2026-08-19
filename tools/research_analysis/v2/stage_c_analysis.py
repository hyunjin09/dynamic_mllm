from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from scoring.reference_likelihood import weighted_logsumexp


_NUMERIC = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s+[+-]?(?:\d+(?:\.\d+)?|\.\d+))*$")
_ALPHABETIC = re.compile(r"^[a-z]+(?:[ '\-][a-z]+)*$")


def trimmed_mean(values: np.ndarray, fraction: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Trimmed mean requires a nonempty one-dimensional array")
    if not 0.0 <= fraction < 0.5:
        raise ValueError("Trim fraction must lie in [0, 0.5)")
    trim = int(math.floor(fraction * values.size))
    ordered = np.sort(values)
    retained = ordered[trim : values.size - trim] if trim else ordered
    if retained.size == 0:
        raise ValueError("Trim fraction removed every observation")
    return float(retained.mean())


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("Correlation inputs must be aligned one-dimensional arrays")
    if left.std() == 0.0 or right.std() == 0.0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    return pearson_correlation(_average_ranks(left), _average_ranks(right))


def uniform_accepted_aggregate(scores: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not scores:
        raise ValueError("At least one accepted-answer component is required")
    weights = [1.0 / len(scores)] * len(scores)
    return {
        "sequence_logprob": weighted_logsumexp(
            [float(score["sequence_logprob"]) for score in scores], weights
        ),
        "mean_logprob": weighted_logsumexp(
            [float(score["mean_logprob"]) for score in scores], weights
        ),
    }


def behavior_category(
    full_score: float,
    write_only_score: float,
    full_normalized: str,
    write_only_normalized: str,
) -> str:
    full_correct = float(full_score) >= 1.0
    write_only_correct = float(write_only_score) >= 1.0
    if not full_correct and write_only_correct:
        return "full_wrong_to_write_only_correct"
    if full_correct and not write_only_correct:
        return "full_correct_to_write_only_wrong"
    if full_correct and write_only_correct:
        return "unchanged_correct"
    if full_normalized != write_only_normalized:
        return "wrong_to_different_wrong"
    return "unchanged_wrong"


def reference_format_category(answers: Sequence[str]) -> str:
    normalized = [str(answer).strip().lower() for answer in answers]
    if not normalized or any(not answer for answer in normalized):
        raise ValueError("Reference-format classification requires nonempty answers")
    if all(_NUMERIC.fullmatch(answer) for answer in normalized):
        return "numeric"
    if all(_ALPHABETIC.fullmatch(answer) for answer in normalized):
        return "alphabetic"
    return "mixed_or_symbolic"


def cluster_values(values: np.ndarray, image_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    image_ids = np.asarray(image_ids)
    if values.ndim != 1 or image_ids.shape != values.shape:
        raise ValueError("Values and image IDs must be aligned one-dimensional arrays")
    unique = np.unique(image_ids)
    return np.asarray([values[image_ids == image_id].mean() for image_id in unique])


def cluster_bootstrap_ci(
    values: np.ndarray,
    image_ids: np.ndarray,
    draws: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    clusters = cluster_values(values, image_ids)
    if clusters.size < 2 or draws < 100:
        raise ValueError("Cluster bootstrap requires at least two clusters and 100 draws")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, clusters.size, size=(draws, clusters.size))
    means = clusters[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def effect_label(value: float, epsilon: float) -> str:
    if value < -epsilon:
        return "negative"
    if value > epsilon:
        return "positive"
    return "answer_silent"


def sign_agreement_fraction(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("Sign-agreement inputs must be aligned nonempty vectors")
    return float(np.mean(np.sign(left) == np.sign(right)))


def consensus_bin(score: float) -> str:
    value = float(score)
    if value >= 1.0:
        return "strict_correct"
    if value > 0.0:
        return "partial_consensus"
    return "strict_wrong"


def classify_stage_c_outcome(
    primary_pass: bool,
    covariance_null_pass: bool,
    real_residual_null_pass: bool,
) -> str:
    if not primary_pass:
        return "Outcome A"
    if not (covariance_null_pass and real_residual_null_pass):
        return "Outcome B"
    return "Outcome C"
