"""Deterministic contracts for the full10 training entrypoint."""

from __future__ import annotations

import torch

from binary_policy.multimodal import deterministic_group_disjoint_modality_permutations
from experiments.train_binary_polar_full10 import (
    MetricAccumulator,
    optimizer_steps_per_epoch,
    scale_gradients_to_sample_mean,
)
from experiments.evaluate_binary_polar_full10_execution import cache_membership_fields


def test_optimizer_step_count_preserves_final_partial_effective_batch():
    assert optimizer_steps_per_epoch(6043, 32, 4) == 48
    assert optimizer_steps_per_epoch(6043, 128, 1) == 48
    assert optimizer_steps_per_epoch(874, 32, 4) == 7


def test_sample_sum_then_gradient_scaling_matches_full_batch_mean():
    torch.manual_seed(7)
    reference = torch.nn.Linear(3, 1, bias=False)
    accumulated = torch.nn.Linear(3, 1, bias=False)
    accumulated.load_state_dict(reference.state_dict())
    inputs = torch.randn(5, 3)
    targets = torch.randn(5, 1)

    full_loss = torch.nn.functional.mse_loss(reference(inputs), targets)
    full_loss.backward()

    for start, end in ((0, 2), (2, 5)):
        loss = torch.nn.functional.mse_loss(
            accumulated(inputs[start:end]), targets[start:end]
        )
        (loss * (end - start)).backward()
    scale_gradients_to_sample_mean(accumulated, sample_count=5)
    assert torch.allclose(reference.weight.grad, accumulated.weight.grad, atol=1e-7)


def test_metric_accumulator_reconstructs_complete_mask_metrics():
    accumulator = MetricAccumulator()
    logits = torch.tensor([[4.0, 4.0], [-4.0, -4.0]])
    masks = torch.tensor([[[1.0, 1.0]], [[0.0, 0.0]]])
    valid = torch.ones(2, 1, dtype=torch.bool)
    weights = torch.ones(2, 1)
    accumulator.update(logits, masks, valid, weights)
    result = accumulator.finalize()
    assert result["examples"] == 2
    assert result["top1_valid_route_coverage"] == 1.0
    assert result["nearest_valid_hamming"] == 0.0
    assert result["unique_top1_masks"] == 2
    assert result["fraction_top1_all_on"] == 0.5
    assert result["fraction_top1_all_off"] == 0.5


def test_metric_accumulator_labels_pareto_and_original_valid_hits_separately():
    accumulator = MetricAccumulator()
    logits = torch.tensor([[4.0, 4.0], [-4.0, -4.0]])
    pareto_masks = torch.tensor([[[1.0, 1.0]], [[0.0, 0.0]]])
    valid = torch.ones(2, 1, dtype=torch.bool)
    weights = torch.ones(2, 1)
    accumulator.update(
        logits,
        pareto_masks,
        valid,
        weights,
        uids=["a", "b"],
        original_valid_masks={"a": {(1, 1)}, "b": {(1, 1)}},
    )

    result = accumulator.finalize()

    assert result["pareto_valid_hit_at_1"] == 1.0
    assert result["original_valid_hit_at_1"] == 0.5


def test_full10_shuffle_excludes_same_uid_and_same_image_group():
    rows = [
        {"uid": "a1", "benchmark": "gqa", "split_group": "image-a"},
        {"uid": "a2", "benchmark": "gqa", "split_group": "image-a"},
        {"uid": "b1", "benchmark": "gqa", "split_group": "image-b"},
        {"uid": "c1", "benchmark": "gqa", "split_group": "image-c"},
    ]
    mapping = deterministic_group_disjoint_modality_permutations(rows, seed=23)
    by_uid = {row["uid"]: row for row in rows}
    assert mapping == deterministic_group_disjoint_modality_permutations(rows, seed=23)
    for target_uid, donors in mapping.items():
        for donor_uid in donors.values():
            assert donor_uid != target_uid
            assert by_uid[donor_uid]["split_group"] != by_uid[target_uid]["split_group"]


def test_full10_execution_cache_schema_supplies_shared_summary_counts():
    fields = cache_membership_fields(
        "1010",
        selected_valid={"1010", "1111"},
        raw_valid={"1010", "1111", "0000"},
    )
    assert fields == {
        "selected_valid_set_size": 2,
        "raw_cached_valid_set_size": 3,
        "predicted_mask_in_selected_valid_set": True,
        "predicted_mask_in_raw_cached_valid_set": True,
    }
