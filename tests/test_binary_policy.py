from __future__ import annotations

import torch

from binary_policy.actions import mask_key, maximal_runs, normalize_visual_on_mask
from binary_policy.decode import topk_factorized_masks
from binary_policy.factorization_audit import direct_representation_gate, factorization_coverage
from binary_policy.labels import deterministic_group_split, parse_mcts_record
from binary_policy.losses import bernoulli_mask_log_probability, multi_valid_set_nll
from binary_policy.objective_audit import optimize_complete_mask_logits
from binary_policy.predictor import BinaryPolarBackbone, SegmentedBinaryPolarBackbone
from binary_policy.segmented import canonical_targets_to_mask, mask_to_canonical_targets


def test_action_roundtrip_and_runs():
    mask = [1, 1, 0, 0, 1]
    assert mask_key(mask) == "11001"
    assert normalize_visual_on_mask(mask, num_layers=5).shape == (1, 5)
    assert maximal_runs(mask) == [(0, 2, 1), (2, 4, 0), (4, 5, 1)]
    boundaries, operations = mask_to_canonical_targets(mask)
    assert canonical_targets_to_mask(boundaries, operations) == mask


def test_topk_factorized_masks_is_deterministic():
    candidates = topk_factorized_masks(torch.tensor([5.0, -5.0, 5.0]), top_k=2)[0]
    assert candidates[0].mask == (1, 0, 1)
    assert candidates[0].log_probability >= candidates[1].log_probability


def test_set_nll_rewards_probability_mass_on_any_valid_route():
    valid = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)
    good = multi_valid_set_nll(torch.tensor([[8.0, -8.0]]), valid)
    bad = multi_valid_set_nll(torch.tensor([[-8.0, -8.0]]), valid)
    assert good < bad


def test_set_nll_matches_weighted_complete_mask_formula():
    logits = torch.tensor([[0.7, -1.2, 0.2]], dtype=torch.float64)
    valid = torch.tensor([[[1, 0, 1], [0, 1, 0]]], dtype=torch.float64)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    complete_mask_logp = bernoulli_mask_log_probability(logits, valid)
    expected = -torch.logsumexp(complete_mask_logp + weights.log(), dim=1).mean()
    actual = multi_valid_set_nll(logits, valid, route_weights=weights)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_binary_polar_backbone_accepts_frozen_image_feature_without_changing_head_shape():
    torch.manual_seed(7)
    model = BinaryPolarBackbone(
        num_layers=7,
        input_dim=16,
        image_dim=8,
        d_model=32,
        num_heads=4,
        dropout=0.0,
    ).eval()
    question = torch.randn(2, 6, 16)
    mask = torch.ones(2, 6, dtype=torch.bool)
    first = model(question, mask, torch.zeros(2, 8))
    second = model(question, mask, torch.ones(2, 8))
    assert first.shape == second.shape == (2, 7)
    assert not torch.allclose(first, second)


def test_exact_set_nll_can_select_one_coherent_contradictory_mask():
    result = optimize_complete_mask_logits(
        [[1, 1, 0, 0], [0, 0, 1, 1]],
        weights=[0.5, 0.5],
        seed=13,
        steps=300,
        learning_rate=0.1,
    )
    assert result["final_loss"] < result["initial_loss"]
    assert result["predicted_mask"] in ([1, 1, 0, 0], [0, 0, 1, 1])
    assert result["top1_is_valid"]


def test_binary_polar_backbone_shapes():
    model = BinaryPolarBackbone(num_layers=7, input_dim=16, d_model=32, num_heads=4)
    output = model(torch.randn(3, 6, 16), torch.ones(3, 6, dtype=torch.bool))
    assert output.shape == (3, 7)

    segmented = SegmentedBinaryPolarBackbone(num_layers=7, input_dim=16, d_model=32, num_heads=4)
    boundaries, operations = segmented(torch.randn(3, 6, 16), torch.ones(3, 6, dtype=torch.bool))
    assert boundaries.shape == (3, 7)
    assert operations.shape == (3, 7, 2)


def test_label_parser_deduplicates_and_normalizes_weights():
    all_on = [1] * 28
    short = [1, 0] * 14
    record = {
        "phase": "binary_visual_mask_graph_mcts_v2",
        "sample": {
            "uid": "gqa:1",
            "sample_id": "1",
            "benchmark": "gqa",
            "mcts_difficulty": "easy",
            "question": "what?",
            "local_image_path": "/data/dataset/example.jpg",
            "image_content_sha256": "abc",
            "correctness_threshold": 1,
        },
        "mcts": {
            "root_reward": 1,
            "all_off_reward": 0,
            "evaluated_masks": [
                {"reward": 1, "visual_on_mask": all_on},
                {"reward": 1, "visual_on_mask": short},
                {"reward": 1, "visual_on_mask": short},
                {"reward": 0, "visual_on_mask": [0] * 28},
            ],
        },
    }
    parsed = parse_mcts_record(record, source_file="fixture.json", max_valid_routes=None)
    assert len(parsed.valid_routes) == 2
    assert abs(sum(route.weight for route in parsed.valid_routes) - 1.0) < 1e-8
    assert parsed.split_group == "sha256:abc"
    assert deterministic_group_split(parsed.split_group) == deterministic_group_split(parsed.split_group)


def test_route_cap_retains_sparse_and_all_on_anchors():
    masks = [[1] * 28, [0] * 28] + [[int((index + offset) % 3 == 0) for index in range(28)] for offset in range(1, 6)]
    record = {
        "phase": "binary_visual_mask_graph_mcts_v2",
        "sample": {
            "uid": "gqa:cap",
            "benchmark": "gqa",
            "mcts_difficulty": "easy",
            "question": "what?",
            "correctness_threshold": 1,
        },
        "mcts": {
            "root_reward": 1,
            "all_off_reward": 1,
            "evaluated_masks": [{"reward": 1, "visual_on_mask": mask} for mask in masks],
        },
    }
    parsed = parse_mcts_record(record, source_file="fixture.json", max_valid_routes=3)
    retained = {route.mask for route in parsed.valid_routes}
    assert tuple([0] * 28) in retained
    assert tuple([1] * 28) in retained


def test_factorization_audit_reports_both_representations():
    valid = [[0, 0, 1, 1], [1, 0, 1, 0]]
    result = factorization_coverage(valid, top_k_values=(1, 2))
    assert set(result["1"]) == {"direct_hit", "segmented_hit"}
    assert result["2"]["direct_hit"]


def test_direct_representation_gate_applies_macro_and_cell_tolerances():
    geometry = {
        "cells": {
            "gqa/easy": {
                "samples_with_valid_route": 100,
                "factorization_coverage": {"5": {"direct_hits": 94, "segmented_hits": 95}},
            },
            "gqa/hard": {
                "samples_with_valid_route": 100,
                "factorization_coverage": {"5": {"direct_hits": 90, "segmented_hits": 92}},
            },
        }
    }
    assert direct_representation_gate(geometry)["passed"]
