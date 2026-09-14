from __future__ import annotations

import numpy as np
import pytest
import torch

from dense_failure_stage1.shared_global_gate import (
    SharedFailurePredictor,
    deterministic_random_layers,
    first_trigger_layers,
    layerwise_ranking_metrics,
    score_space_rows,
    select_candidate,
    threshold_sweep,
    trajectory_operating_point,
)


@pytest.mark.parametrize(
    ("variant", "state_required"),
    [
        ("state_only", True),
        ("layer_only", False),
        ("state_layer_all28", True),
        ("state_layer_random4", True),
    ],
)
def test_shared_predictor_variants(variant: str, state_required: bool) -> None:
    model = SharedFailurePredictor(
        variant=variant,
        input_size=12,
        projection_size=5,
        layer_embedding_size=3,
        hidden_size=7,
    )
    layers = torch.tensor([0, 7, 27], dtype=torch.long)
    states = torch.randn(3, 12) if state_required else None
    assert model(states, layers).shape == (3,)
    if not state_required:
        with pytest.raises(ValueError):
            model(torch.randn(3, 12), layers)


def test_random4_is_deterministic_unique_and_epoch_dependent() -> None:
    first = deterministic_random_layers(20, epoch=2, seed=17)
    repeat = deterministic_random_layers(20, epoch=2, seed=17)
    other = deterministic_random_layers(20, epoch=3, seed=17)
    assert torch.equal(first, repeat)
    assert not torch.equal(first, other)
    assert first.shape == (20, 4)
    assert all(len(set(row.tolist())) == 4 for row in first)


def test_global_threshold_is_trajectory_level_tie_safe_and_strict() -> None:
    scores = np.zeros((8, 28), dtype=np.float64)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores[:4, 0] = [0.9, 0.8, 0.7, 0.6]
    scores[4:, 0] = [0.95, 0.85, 0.75, 0.1]
    point = trajectory_operating_point(
        scores, labels, target_preservation=0.75, window=[0, 1]
    )
    assert point["threshold"] == pytest.approx(0.8)
    assert point["correct_false_triggers"] == 1
    assert point["correct_preservation"] == pytest.approx(0.75)
    assert point["wrong_detection_recall"] == pytest.approx(0.5)


def test_first_trigger_returns_actual_window_layer() -> None:
    scores = np.zeros((3, 28), dtype=np.float64)
    scores[0, 17] = 0.8
    scores[1, 20] = 0.9
    scores[2, 2] = 1.0
    first = first_trigger_layers(scores, threshold=0.7, window=range(16, 28))
    assert first.tolist() == [17, 20, -1]


def test_sweep_metrics_and_layerwise_metrics_cover_contract() -> None:
    scores = np.tile(np.linspace(0.1, 0.9, 10)[:, None], (1, 28))
    labels = np.array([0, 1] * 5)
    sweep = threshold_sweep(scores, labels, window=range(28))
    assert sweep
    assert all("correct_preservation" in row for row in sweep)
    metrics = layerwise_ranking_metrics(scores, labels)
    assert [row["layer"] for row in metrics] == list(range(28))
    analysis = score_space_rows(scores, labels)
    assert sum(row["row_type"] == "distribution" for row in analysis) == 56
    assert sum(row["row_type"] == "neighbor_correlation" for row in analysis) == 81


def test_validation_candidate_selection_uses_frozen_tiebreak() -> None:
    rows = [
        {"name": "all_0_27", "wrong_detection_recall": 0.4, "correct_preservation": 0.99, "failure_precision": 0.9},
        {"name": "late_16_27", "wrong_detection_recall": 0.4, "correct_preservation": 0.99, "failure_precision": 0.9},
    ]
    assert select_candidate(rows, preferred="all_0_27")["name"] == "all_0_27"
