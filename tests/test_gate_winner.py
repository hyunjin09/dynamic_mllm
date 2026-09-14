from __future__ import annotations

import json

import numpy as np
import pytest

from dense_failure_stage1.gate_winner import (
    apply_control,
    independent_alpha_sweep,
    metrics_with_utility,
    paired_bootstrap,
    reference_operating_point,
    select_operating_point,
    select_winner,
    shared_threshold_sweep,
)


def test_utility_is_wrong_minus_correct_triggers() -> None:
    labels = np.array([0, 0, 1, 1])
    first = np.array([-1, 2, 3, 4])
    metrics = metrics_with_utility(labels, first)
    assert metrics["utility"] == 1
    assert metrics["utility_rate"] == pytest.approx(0.25)
    assert metrics["failure_precision"] == pytest.approx(2 / 3)


def test_shared_sweep_covers_no_trigger_to_all_trigger() -> None:
    scores = np.zeros((4, 28), dtype=np.float64)
    scores[:, 0] = [0.1, 0.2, 0.8, 0.9]
    labels = np.array([0, 0, 1, 1])
    rows = shared_threshold_sweep(scores, labels, layers=range(28))
    assert rows[0]["triggered"] == 0
    assert rows[-1]["triggered"] == 4


def test_independent_sweep_covers_all_empirical_tail_breakpoints() -> None:
    scores = np.tile(np.array([0.1, 0.2, 0.8, 0.9])[:, None], (1, 28))
    labels = np.array([0, 0, 1, 1])
    rows = independent_alpha_sweep(scores, labels)
    assert len(rows) == 3
    assert rows[0]["control_value"] == 0.0
    assert rows[-1]["control_type"] == "terminal_below_correct_minima"
    assert len(json.loads(rows[0]["layer_thresholds_json"])) == 28


def test_selection_respects_precision_then_frozen_ties() -> None:
    rows = [
        {"utility_rate": 0.4, "failure_precision": 0.89, "correct_preservation": 0.9, "wrong_detection_recall": 0.9, "median_first_trigger_layer": 1},
        {"utility_rate": 0.3, "failure_precision": 0.95, "correct_preservation": 0.95, "wrong_detection_recall": 0.65, "median_first_trigger_layer": 4},
    ]
    assert select_operating_point(rows, minimum_precision=0.9) == rows[1]
    candidates = [
        {"name": "complex", "utility_rate": 0.3, "correct_preservation": 0.95, "wrong_detection_recall": 0.65, "median_first_trigger_layer": 4, "simplicity_rank": 2, "candidate_order": 0},
        {"name": "simple", "utility_rate": 0.3, "correct_preservation": 0.95, "wrong_detection_recall": 0.65, "median_first_trigger_layer": 4, "simplicity_rank": 0, "candidate_order": 1},
    ]
    winner, ranking = select_winner(candidates)
    assert winner["name"] == "simple"
    assert [row["name"] for row in ranking] == ["simple", "complex"]


def test_reference_prefers_conservative_side_of_equal_distance() -> None:
    rows = [
        {"correct_preservation": 0.98, "wrong_detection_recall": 0.5, "failure_precision": 0.9},
        {"correct_preservation": 1.0, "wrong_detection_recall": 0.4, "failure_precision": 1.0},
    ]
    assert reference_operating_point(rows, target_preservation=0.99)["correct_preservation"] == 1.0


def test_apply_control_and_paired_bootstrap_are_deterministic() -> None:
    scores = np.zeros((4, 28), dtype=np.float64)
    scores[:, 27] = [0.1, 0.7, 0.8, 0.9]
    labels = np.array([0, 0, 1, 1])
    fixed = apply_control("shared_fixed_l27", scores, {"threshold": 0.75})
    sequential = apply_control("shared_random4", scores, {"threshold": 0.65})
    assert fixed.tolist() == [-1, -1, 27, 27]
    first = paired_bootstrap(labels, fixed, sequential, replicates=100, seed=9)
    second = paired_bootstrap(labels, fixed, sequential, replicates=100, seed=9)
    assert first == second
