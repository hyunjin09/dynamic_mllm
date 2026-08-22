from __future__ import annotations

import numpy as np

from experiments.analyze_wemath_visual_access_placement import (
    axis_added,
    cluster_bootstrap_vector_ci,
    route_metrics,
    sample_route_summary,
    select_route_set,
)


def test_route_metrics_contiguous_front_loaded() -> None:
    values = route_metrics([1, 1, 1, 1] + [0] * 24)
    assert values["first_on"] == 0
    assert values["last_on"] == 3
    assert values["centroid"] == 1.5
    assert values["early_fraction"] == 1.0
    assert values["late_fraction"] == 0.0
    assert values["on_segments"] == 1
    assert values["reentries"] == 0
    assert values["max_internal_off_gap"] == 0
    assert values["has_late_reentry"] == 0


def test_route_metrics_reentry_and_late_reentry() -> None:
    mask = [0] * 28
    for layer in (0, 1, 5, 6, 19, 20, 27):
        mask[layer] = 1
    values = route_metrics(mask)
    assert values["on_segments"] == 4
    assert values["reentries"] == 3
    assert values["max_internal_off_gap"] == 12
    assert values["has_reentry"] == 1
    assert values["has_late_reentry"] == 1
    assert values["late_on"] == 3
    assert values["very_late_access"] == 1


def test_minimum_and_near_minimum_route_sets() -> None:
    masks = [
        [1] * 2 + [0] * 26,
        [0] * 2 + [1] * 3 + [0] * 23,
        [0] * 5 + [1] * 5 + [0] * 18,
    ]
    assert len(select_route_set(masks, minimum_on=2, delta=0)) == 1
    assert len(select_route_set(masks, minimum_on=2, delta=2)) == 2
    assert len(select_route_set(masks, minimum_on=2, delta=4)) == 3


def test_sample_summary_balances_routes_within_sample() -> None:
    row = {
        "uid": "u", "difficulty": "x", "difficulty_degree": 1,
        "question_id": "family", "image_group_id": "image", "knowledge_points": ["p"],
    }
    first = [1, 0] + [0] * 26
    second = [0, 1] + [0] * 26
    summary = sample_route_summary(row=row, masks=[first, second], cohort="V+", delta=0)
    assert summary["selected_route_count"] == 2
    assert summary["layer_00"] == 0.5
    assert summary["layer_01"] == 0.5
    assert summary["centroid"] == 0.5


def test_cluster_bootstrap_vector_is_deterministic_and_joint() -> None:
    matrix = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    first = cluster_bootstrap_vector_ci(matrix, ["a", "a", "b"], draws=200, seed=9)
    second = cluster_bootstrap_vector_ci(matrix, ["a", "a", "b"], draws=200, seed=9)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[0].shape == (2,)


def test_axis_addition_matches_factorial_transitions() -> None:
    assert axis_added("base", "x") == "x"
    assert axis_added("y", "xy") == "x"
    assert axis_added("xz", "xyz") == "y"
    assert axis_added("xy", "xyz") == "z"
