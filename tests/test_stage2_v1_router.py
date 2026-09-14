from __future__ import annotations

import random

import pytest
import torch

from dense_failure_stage2.v1_router import (
    ACTION_NAMES,
    SharedReadWriteRouter,
    build_epoch_draws,
    sample_c_layers,
    sample_w_layers,
    summarize_action_predictions,
)


def test_router_uses_full_visual_sequence_and_has_no_layer_input():
    torch.manual_seed(7)
    router = SharedReadWriteRouter(hidden_size=8, router_size=4, num_heads=2, dropout=0.0)
    text = torch.randn(2, 3, 8)
    visual = torch.randn(2, 5, 8)
    text_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    visual_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    logits = router(text, visual, text_mask=text_mask, visual_mask=visual_mask)
    assert logits.shape == (2, 4)
    assert ACTION_NAMES == ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")
    assert not any("layer" in name for name, _ in router.named_parameters())
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in router.parameters())


def test_router_promotes_frozen_bfloat16_states_to_fp32_compute():
    router = SharedReadWriteRouter(hidden_size=8, router_size=4, num_heads=2, dropout=0.0)
    logits = router(
        torch.randn(1, 2, 8, dtype=torch.bfloat16),
        torch.randn(1, 3, 8, dtype=torch.bfloat16),
        text_mask=torch.ones(1, 2, dtype=torch.bool),
        visual_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_router_rejects_empty_or_misaligned_masks():
    router = SharedReadWriteRouter(hidden_size=8, router_size=4, num_heads=2, dropout=0.0)
    text = torch.randn(1, 2, 8)
    visual = torch.randn(1, 3, 8)
    with pytest.raises(ValueError, match="visual mask"):
        router(
            text,
            visual,
            text_mask=torch.ones(1, 2, dtype=torch.bool),
            visual_mask=torch.zeros(1, 3, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="shape"):
        router(
            text,
            visual,
            text_mask=torch.ones(1, 3, dtype=torch.bool),
            visual_mask=torch.ones(1, 3, dtype=torch.bool),
        )


def test_positive_anchored_random4_includes_correction_and_timing_context():
    actions = ["FULL"] * 28
    actions[13] = "WRITE_ONLY"
    selected = sample_w_layers(actions, trigger_layer=8, rng=random.Random(123))
    assert len(selected) == 4
    assert len(set(selected)) == 4
    assert 13 in selected
    assert any(8 <= layer < 13 for layer in selected)
    assert any(layer > 13 for layer in selected)
    assert sum(actions[layer] != "FULL" for layer in selected) == 1


def test_sampling_backfills_boundaries_without_duplicates():
    actions = ["FULL"] * 28
    actions[27] = "IGNORE"
    assert sample_w_layers(actions, trigger_layer=26, rng=random.Random(1)) == [26, 27]
    c_layers = sample_c_layers(trigger_layer=25, num_layers=28, rng=random.Random(1))
    assert sorted(c_layers) == [25, 26, 27]


def test_epoch_draws_cover_every_w_once_and_freeze_one_to_two_mixture():
    routes = {
        "w0": [{"route_id": "a"}, {"route_id": "b"}],
        "w1": [{"route_id": "c"}],
        "w2": [{"route_id": "d"}],
        "w3": [{"route_id": "e"}],
    }
    draws = build_epoch_draws(
        routes,
        ["c0", "c1"],
        c_draws=2,
        seed=99,
        epoch=3,
    )
    assert len(draws) == 6
    assert sorted(draw["uid"] for draw in draws if draw["kind"] == "W") == sorted(routes)
    assert sum(draw["kind"] == "C" for draw in draws) == 2
    assert all("route" in draw for draw in draws if draw["kind"] == "W")
    assert draws == build_epoch_draws(routes, ["c0", "c1"], c_draws=2, seed=99, epoch=3)


def test_action_summary_reports_supported_recall_and_distribution():
    summary = summarize_action_predictions(
        targets=[0, 0, 1, 2, 3],
        predictions=[0, 1, 1, 0, 3],
    )
    assert summary["recall"]["FULL"] == 0.5
    assert summary["recall"]["READ_ONLY"] == 1.0
    assert summary["recall"]["WRITE_ONLY"] == 0.0
    assert summary["recall"]["IGNORE"] == 1.0
    assert summary["non_full_recall"] == pytest.approx(2 / 3)
    assert summary["predicted_distribution"]["FULL"] == pytest.approx(0.4)
