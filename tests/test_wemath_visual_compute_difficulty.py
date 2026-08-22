from __future__ import annotations

import math

import numpy as np

from experiments.analyze_wemath_visual_compute_difficulty import (
    DEGREE,
    average_ranks,
    cluster_bootstrap_mean_ci,
    distribution_summary,
    fit_ols,
    spearman,
)


def test_difficulty_degree_is_factorial_not_total_axis_order() -> None:
    assert DEGREE == {
        "base": 0,
        "x": 1,
        "y": 1,
        "z": 1,
        "xy": 2,
        "xz": 2,
        "yz": 2,
        "xyz": 3,
    }


def test_distribution_summary_uses_sample_standard_deviation() -> None:
    result = distribution_summary([0, 2, 4])
    assert result["n"] == 3
    assert result["mean"] == 2
    assert result["median"] == 2
    assert result["std"] == 2
    assert result["q25"] == 1
    assert result["q75"] == 3


def test_cluster_bootstrap_is_deterministic_and_cluster_weighted() -> None:
    values = [0.0, 2.0, 10.0]
    clusters = ["a", "a", "b"]
    first = cluster_bootstrap_mean_ci(values, clusters, draws=500, seed=7)
    second = cluster_bootstrap_mean_ci(values, clusters, draws=500, seed=7)
    assert first == second
    assert first[0] <= np.mean(values) <= first[1]


def test_average_rank_and_spearman_handle_ties() -> None:
    assert average_ranks([1, 1, 3]).tolist() == [1.5, 1.5, 3.0]
    assert math.isclose(spearman([0, 1, 2, 3], [0, 2, 4, 6]), 1.0)
    assert math.isclose(spearman([0, 1, 2, 3], [6, 4, 2, 0]), -1.0)


def test_ols_recovers_degree_effect_with_visual_control() -> None:
    degree = np.asarray([0, 1, 2, 3, 0, 1], dtype=float)
    tokens = np.asarray([1, 1, 1, 1, 2, 2], dtype=float)
    outcome = 4 + 2 * degree + 3 * tokens
    beta, r2 = fit_ols(outcome, [degree, tokens])
    assert np.allclose(beta, [4, 2, 3])
    assert math.isclose(r2, 1.0)
