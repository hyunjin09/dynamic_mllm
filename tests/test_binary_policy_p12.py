"""Deterministic P12 canonical-segment and exact-likelihood contracts."""

from __future__ import annotations

import math

import torch
from torch import nn

from binary_policy.dataset import make_structured_set_collator
from binary_policy.predictor import SegmentedBinaryPolarBackbone
from binary_policy.training import evaluate_structured_epoch, train_structured_epoch
from binary_policy.structured import (
    decode_structured_top1,
    mask_to_p12_targets,
    p12_targets_to_mask,
    structured_route_log_probability,
    structured_batch_metrics,
    structured_valid_set_nll,
    summarize_segment_geometry,
)


class _TinyTokenizer:
    def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
        del truncation, max_length
        assert padding and return_tensors == "pt"
        width = max(len(text) for text in texts)
        ids = torch.zeros(len(texts), width, dtype=torch.long)
        attention = torch.zeros_like(ids)
        for index, text in enumerate(texts):
            ids[index, -len(text) :] = torch.arange(1, len(text) + 1)
            attention[index, -len(text) :] = 1
        return {"input_ids": ids, "attention_mask": attention}


class _FrozenTinyEncoder(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.embedding = nn.Embedding(64, width)
        self.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return self.embedding(input_ids).to(torch.bfloat16)


def test_p12_round_trip_all_on_all_off_and_alternating():
    for mask, expected_segments in (
        ([1] * 28, 1),
        ([0] * 28, 1),
        ([index % 2 for index in range(28)], 28),
        ([1, 1, 0, 0, 0, 1, 0, 0], 4),
    ):
        boundaries, operations = mask_to_p12_targets(mask)
        assert boundaries[0] == 1
        assert sum(boundaries) == expected_segments
        assert p12_targets_to_mask(boundaries, operations) == mask
        assert all(operations[index] in (0, 1) for index, value in enumerate(boundaries) if value)
        assert all(operations[index] == -100 for index, value in enumerate(boundaries) if not value)


def test_structured_route_probability_matches_manual_boundary_plus_operation_terms():
    boundary_logits = torch.tensor([[0.2, -0.7, 1.1, -0.4]], dtype=torch.float64)
    operation_logits = torch.tensor(
        [[[0.1, 0.8], [0.5, -0.1], [1.2, -0.3], [-0.2, 0.6]]], dtype=torch.float64
    )
    boundaries = torch.tensor([[[1, 0, 1, 0]]], dtype=torch.float64)
    operations = torch.tensor([[[1, -100, 0, -100]]], dtype=torch.long)

    actual = structured_route_log_probability(
        boundary_logits, operation_logits, boundaries, operations
    ).squeeze()
    boundary_manual = (
        boundaries[0, 0] * torch.nn.functional.logsigmoid(boundary_logits[0])
        + (1 - boundaries[0, 0]) * torch.nn.functional.logsigmoid(-boundary_logits[0])
    ).sum()
    operation_logp = torch.log_softmax(operation_logits[0], dim=-1)
    operation_manual = operation_logp[0, 1] + operation_logp[2, 0]
    assert torch.allclose(actual, boundary_manual + operation_manual, atol=1e-12, rtol=0)


def test_structured_singleton_set_nll_matches_negative_route_log_probability():
    boundary_logits = torch.tensor([[0.3, -1.0, 0.7, -0.2]], dtype=torch.float64)
    operation_logits = torch.randn(1, 4, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    boundaries = torch.tensor([[[1, 0, 1, 0]]], dtype=torch.float64)
    operations = torch.tensor([[[1, -100, 0, -100]]], dtype=torch.long)
    expected = -structured_route_log_probability(
        boundary_logits, operation_logits, boundaries, operations
    ).mean()
    actual = structured_valid_set_nll(
        boundary_logits, operation_logits, boundaries, operations
    )
    assert torch.allclose(actual, expected, atol=1e-12, rtol=0)


def test_structured_multiple_routes_is_finite_and_padding_has_zero_mass():
    boundary_logits = torch.zeros(1, 8, dtype=torch.float64, requires_grad=True)
    operation_logits = torch.zeros(1, 8, 2, dtype=torch.float64, requires_grad=True)
    masks = ([1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1])
    targets = [mask_to_p12_targets(mask) for mask in masks]
    boundaries = torch.tensor([[targets[0][0], targets[1][0], [1] * 8]], dtype=torch.float64)
    operations = torch.tensor([[targets[0][1], targets[1][1], [1] * 8]], dtype=torch.long)
    valid = torch.tensor([[True, True, False]])
    weights = torch.tensor([[0.5, 0.5, 99.0]], dtype=torch.float64)

    loss = structured_valid_set_nll(
        boundary_logits,
        operation_logits,
        boundaries,
        operations,
        valid_mask=valid,
        route_weights=weights,
    )
    unpadded = structured_valid_set_nll(
        boundary_logits,
        operation_logits,
        boundaries[:, :2],
        operations[:, :2],
        route_weights=weights[:, :2],
    )
    assert math.isfinite(float(loss))
    assert torch.allclose(loss, unpadded, atol=1e-12, rtol=0)
    loss.backward()
    assert torch.isfinite(boundary_logits.grad).all()
    assert torch.isfinite(operation_logits.grad).all()


def test_structured_top1_forces_first_boundary_and_fills_predicted_segments():
    boundary_logits = torch.tensor([[-9.0, -2.0, 3.0, -4.0, 2.0]])
    operation_logits = torch.tensor(
        [[[0.0, 2.0], [2.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 4.0]]]
    )
    decoded = decode_structured_top1(boundary_logits, operation_logits)
    assert decoded[0]["boundaries"] == (1, 0, 1, 0, 1)
    assert decoded[0]["mask"] == (1, 1, 0, 0, 1)
    assert decoded[0]["predicted_segments"] == 3


def test_segment_geometry_reports_runs_lengths_and_compressibility():
    summary = summarize_segment_geometry(
        [[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 1, 0]]
    )
    assert summary["masks"] == 3
    assert summary["segments"]["mean"] == 7 / 3
    assert summary["transitions"]["mean"] == 4 / 3
    assert summary["on_segments"]["mean"] == 4 / 3
    assert summary["off_segments"]["mean"] == 1.0
    assert summary["fraction_at_most_segments"]["2"] == 2 / 3
    assert summary["fraction_at_most_segments"]["4"] == 1.0
    assert summary["segment_length_histogram"] == {"1": 4, "2": 2, "4": 1}


def test_structured_collator_preserves_variable_sets_and_p11_weights():
    rows = [
        {
            "uid": "a",
            "question": "a",
            "valid_routes": [{"mask": [1, 1, 1, 1]}],
        },
        {
            "uid": "b",
            "question": "bb",
            "valid_routes": [
                {"mask": [1, 1, 1, 1]},
                {"mask": [1, 1, 0, 0]},
            ],
        },
    ]
    batch = make_structured_set_collator(
        _TinyTokenizer(), route_weighting="polar_full_downweight_0.3"
    )(rows)
    assert batch["boundary_targets"].shape == (2, 2, 4)
    assert batch["operation_targets"].shape == (2, 2, 4)
    assert batch["valid_mask"].tolist() == [[True, False], [True, True]]
    assert batch["boundary_targets"][0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert batch["operation_targets"][1, 1].tolist() == [1, -100, 0, -100]
    assert torch.allclose(batch["route_weights"][1], torch.tensor([0.3 / 1.3, 1 / 1.3]))


def test_structured_metrics_use_complete_decoded_masks_and_weighted_native_targets():
    boundary_logits = torch.tensor([[3.0, -3.0, 3.0, -3.0]])
    operation_logits = torch.tensor([[[0.0, 4.0], [0.0, 0.0], [4.0, 0.0], [0.0, 0.0]]])
    mask = [1, 1, 0, 0]
    boundaries, operations = mask_to_p12_targets(mask)
    metrics = structured_batch_metrics(
        boundary_logits,
        operation_logits,
        torch.tensor([[mask]], dtype=torch.float32),
        torch.tensor([[boundaries]], dtype=torch.float32),
        torch.tensor([[operations]], dtype=torch.long),
        torch.tensor([[True]]),
        torch.tensor([[1.0]]),
    )
    assert metrics["top1_valid_route_coverage"] == 1.0
    assert metrics["nearest_valid_hamming"] == 0.0
    assert metrics["average_predicted_visual_on"] == 2.0
    assert metrics["average_predicted_segments"] == 2.0
    assert metrics["boundary_accuracy"] == 1.0
    assert metrics["boundary_precision"] == 1.0
    assert metrics["boundary_recall"] == 1.0
    assert metrics["segment_operation_accuracy_at_gt_boundaries"] == 1.0


def test_structured_train_and_bf16_validation_keep_encoder_frozen():
    rows = [
        {
            "uid": "a",
            "question": "question",
            "valid_routes": [{"mask": [1, 1, 0, 0]}, {"mask": [0, 0, 1, 1]}],
        }
    ]
    batch = make_structured_set_collator(_TinyTokenizer())(rows)
    encoder = _FrozenTinyEncoder()
    predictor = SegmentedBinaryPolarBackbone(
        num_layers=4, input_dim=8, d_model=16, num_heads=4, num_layer_blocks=1, dropout=0.0
    )
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-2)
    train = train_structured_epoch(
        predictor,
        encoder,
        [batch],
        optimizer,
        device=torch.device("cpu"),
        amp_dtype=torch.bfloat16,
    )
    validation = evaluate_structured_epoch(
        predictor,
        encoder,
        [batch],
        device=torch.device("cpu"),
        amp_dtype=torch.bfloat16,
    )
    assert math.isfinite(train["loss"])
    assert math.isfinite(validation["set_nll"])
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in predictor.parameters()
    )
