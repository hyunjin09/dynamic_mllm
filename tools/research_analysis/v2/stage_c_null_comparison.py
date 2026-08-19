from __future__ import annotations

from typing import Any

import numpy as np


def _cluster_means(values: np.ndarray, image_ids: np.ndarray) -> np.ndarray:
    unique = np.unique(image_ids)
    return np.asarray([values[image_ids == image_id].mean() for image_id in unique])


def _bootstrap_mean_ci(
    cluster_values: np.ndarray, draws: int, seed: int, confidence: float = 0.95
) -> tuple[float, float]:
    if cluster_values.ndim != 1 or cluster_values.size < 2:
        raise ValueError("At least two image clusters are required")
    if draws < 100:
        raise ValueError("At least 100 bootstrap draws are required")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, cluster_values.size, size=(draws, cluster_values.size)
    )
    means = cluster_values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def _family_result(
    real: np.ndarray,
    null_draws: np.ndarray,
    image_ids: np.ndarray,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    null_mean = null_draws.mean(axis=1)
    paired = real - null_mean
    cluster_paired = _cluster_means(paired, image_ids)
    low, high = _bootstrap_mean_ci(cluster_paired, bootstrap_draws, seed)
    return {
        "real_mean": float(_cluster_means(real, image_ids).mean()),
        "null_mean": float(_cluster_means(null_mean, image_ids).mean()),
        "paired_mean": float(cluster_paired.mean()),
        "paired_ci_low": low,
        "paired_ci_high": high,
        "pass": high < 0.0,
    }


def evaluate_null_superiority(
    real: np.ndarray,
    covariance_null: np.ndarray,
    real_residual_null: np.ndarray,
    image_ids: np.ndarray,
    bootstrap_draws: int,
    covariance_seed: int,
    real_residual_seed: int,
) -> dict[str, Any]:
    real = np.asarray(real, dtype=np.float64)
    image_ids = np.asarray(image_ids)
    covariance_null = np.asarray(covariance_null, dtype=np.float64)
    real_residual_null = np.asarray(real_residual_null, dtype=np.float64)
    if real.ndim != 1 or image_ids.shape != real.shape:
        raise ValueError("Real effects and image IDs must be aligned one-dimensional arrays")
    if covariance_null.ndim != 2 or real_residual_null.ndim != 2:
        raise ValueError("Each null family must have shape [samples, draws]")
    if covariance_null.shape[0] != real.size or real_residual_null.shape[0] != real.size:
        raise ValueError("Null families must align with every real sample")
    covariance = _family_result(
        real, covariance_null, image_ids, bootstrap_draws, covariance_seed
    )
    real_residual = _family_result(
        real, real_residual_null, image_ids, bootstrap_draws, real_residual_seed
    )
    return {
        "comparison": "paired real effect minus per-sample mean null effect",
        "orientation": "negative means actual READ is more answer-misaligned than null",
        "confidence": 0.95,
        "multiplicity": "intersection-union conjunction; both family-specific tests must pass",
        "covariance": covariance,
        "real_residual": real_residual,
        "gate_pass": bool(covariance["pass"] and real_residual["pass"]),
    }


def evaluate_real_residual_sensitivity(
    real: np.ndarray,
    real_residual_null: np.ndarray,
    image_ids: np.ndarray,
    included: np.ndarray,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    real = np.asarray(real, dtype=np.float64)
    real_residual_null = np.asarray(real_residual_null, dtype=np.float64)
    image_ids = np.asarray(image_ids)
    included = np.asarray(included, dtype=bool)
    if included.shape != real.shape or image_ids.shape != real.shape:
        raise ValueError("Sensitivity mask, real effects, and image IDs must align")
    if real_residual_null.ndim != 2 or real_residual_null.shape[0] != real.size:
        raise ValueError("Real-residual null draws must align with every real effect")
    if int(included.sum()) < 2:
        raise ValueError("Sensitivity analysis requires at least two included records")
    result = _family_result(
        real[included],
        real_residual_null[included],
        image_ids[included],
        bootstrap_draws,
        seed,
    )
    return {
        "role": "secondary prespecified original-caliper-supported sensitivity",
        "n_records": int(included.sum()),
        "n_image_clusters": int(np.unique(image_ids[included]).size),
        **result,
    }
