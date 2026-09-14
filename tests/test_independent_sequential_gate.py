from __future__ import annotations

import numpy as np
import pytest

from dense_failure_stage1.sequential_gate import (
    empirical_alpha_grid,
    first_trigger_layers,
    fixed_layer_threshold,
    gate_metrics,
    layer_thresholds,
    region_name,
    select_operating_point,
    sweep_shared_alpha,
)


def _scores() -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((8, 28), dtype=np.float64)
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    values[:4] = np.asarray([0.1, 0.2, 0.3, 0.4])[:, None]
    values[4:] = np.asarray([0.15, 0.35, 0.8, 0.9])[:, None]
    values[4, 5] = 0.95
    return values, labels


def test_empirical_grid_covers_attainable_higher_quantile_breakpoints() -> None:
    grid = empirical_alpha_grid(400, maximum_alpha=0.10)

    assert grid[0] == 0.0
    assert grid[1] == pytest.approx(1 / 399)
    assert grid[-1] == pytest.approx(0.10)
    assert all(left < right for left, right in zip(grid, grid[1:]))


def test_higher_quantiles_and_strict_crossing_are_conservative() -> None:
    scores, labels = _scores()
    strict = layer_thresholds(scores, labels, alpha=0.0)
    relaxed = layer_thresholds(scores, labels, alpha=1 / 3)

    assert np.all(strict == 0.4)
    assert np.all(relaxed == 0.3)
    first = first_trigger_layers(scores, strict)
    assert np.all(first[:4] == -1)
    assert first[4] == 5
    assert first[5] == -1
    assert np.all(first[6:] == 0)


def test_metrics_are_sample_level_over_the_first_trigger_union() -> None:
    labels = np.asarray([0, 0, 1, 1])
    first = np.asarray([-1, 7, 3, -1])

    metrics = gate_metrics(labels, first)

    assert metrics["correct_preservation"] == pytest.approx(0.5)
    assert metrics["wrong_detection_recall"] == pytest.approx(0.5)
    assert metrics["failure_precision"] == pytest.approx(0.5)
    assert metrics["median_first_trigger_layer"] == pytest.approx(5.0)


def test_operating_point_is_largest_alpha_that_meets_sample_preservation() -> None:
    scores, labels = _scores()
    sweep = sweep_shared_alpha(scores, labels, [0.0, 1 / 3, 2 / 3])

    selected = select_operating_point(sweep, target_preservation=0.75)

    assert selected is not None
    assert selected["alpha"] == pytest.approx(1 / 3)
    assert selected["correct_preservation"] >= 0.75


def test_fixed_layer_threshold_is_tie_safe_for_strict_gate() -> None:
    scores = np.asarray([0.4, 0.4, 0.3, 0.2, 0.9, 0.8])
    labels = np.asarray([0, 0, 0, 0, 1, 1])

    threshold = fixed_layer_threshold(scores, labels, target_preservation=0.75)
    false_triggers = np.logical_and(scores > threshold, labels == 0).sum()

    assert threshold == pytest.approx(0.4)
    assert false_triggers == 0


def test_trigger_regions_are_fixed() -> None:
    assert region_name(-1) == "never"
    assert region_name(8) == "early"
    assert region_name(9) == "middle"
    assert region_name(18) == "middle"
    assert region_name(19) == "late"
    assert region_name(27) == "late"
