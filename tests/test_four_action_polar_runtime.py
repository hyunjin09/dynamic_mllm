from __future__ import annotations

from collections import Counter
import json

import pytest
import torch

from experiments.train_binary_polar import file_sha256
from four_action_policy.evaluation import (
    FourActionMetricAccumulator,
    batch_offline_metrics,
    checkpoint_key,
)
from four_action_policy.feature_cache import load_verified_feature_index
from four_action_policy.training import train_epoch


def test_four_action_metrics_use_exact_top_routes_and_nearest_hamming() -> None:
    logits = torch.full((2, 2, 4), -5.0)
    logits[0, 0, 0] = 5
    logits[0, 1, 1] = 5
    logits[1, 0, 3] = 5
    logits[1, 1, 2] = 5
    logits[1, 1, 3] = 4
    routes = torch.tensor(
        [
            [[0, 1], [3, 3]],
            [[3, 3], [0, 0]],
        ]
    )
    valid = torch.tensor([[True, True], [True, False]])

    metrics = batch_offline_metrics(logits, routes, valid, top_k=5)

    assert metrics["top1_valid_route_coverage"] == 0.5
    assert metrics["topk_valid_route_coverage"] == 1.0
    assert metrics["nearest_valid_hamming"] == 0.5
    assert metrics["top1_route_counts"] == Counter({"IGNORE|READ_ONLY": 1, "FULL|WRITE_ONLY": 1})


def test_metric_accumulator_reports_both_losses_and_objective_selection() -> None:
    logits = torch.zeros(1, 2, 4)
    routes = torch.tensor([[[0, 1], [3, 3]]])
    valid = torch.tensor([[True, True]])
    weights = torch.tensor([[0.5, 0.5]])
    accumulator = FourActionMetricAccumulator(top_k=5)

    accumulator.update(logits, routes, valid, weights)
    result = accumulator.finalize(objective="duplicated_action_bce")

    assert result["examples"] == 1
    assert result["set_nll"] > 0
    assert result["duplicated_action_bce"] == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert result["objective_loss"] == result["duplicated_action_bce"]


def test_checkpoint_order_uses_hit1_hit5_hamming_loss_then_earlier_epoch() -> None:
    def row(epoch, hit1, hit5, hamming, loss):
        return {
            "epoch": epoch,
            "validation": {
                "overall": {
                    "top1_valid_route_coverage": hit1,
                    "topk_valid_route_coverage": hit5,
                    "nearest_valid_hamming": hamming,
                    "objective_loss": loss,
                }
            },
        }

    rows = [
        row(1, 0.4, 0.8, 2.0, 1.0),
        row(2, 0.4, 0.9, 3.0, 2.0),
        row(3, 0.5, 0.5, 4.0, 3.0),
    ]
    assert max(rows, key=checkpoint_key)["epoch"] == 3
    assert max(rows[:2], key=checkpoint_key)["epoch"] == 2


def test_feature_cache_fails_closed_on_coverage_checksum_and_shape(tmp_path) -> None:
    tensor = torch.randn(3, 5, dtype=torch.bfloat16)
    tensor_path = tmp_path / "visual.pt"
    torch.save(tensor, tensor_path)
    manifest_path = tmp_path / "features.jsonl"
    record = {
        "uid": "gqa:a",
        "split_group": "gqa:image-a",
        "path": str(tensor_path),
        "sha256": file_sha256(tensor_path),
        "shape": [3, 5],
        "dtype": "torch.bfloat16",
        "feature_width": 5,
    }
    manifest_path.write_text(json.dumps(record) + "\n")

    index = load_verified_feature_index(
        manifest_path,
        manifest_sha256=file_sha256(manifest_path),
        expected_uids={"gqa:a"},
        expected_feature_width=5,
    )
    assert index["gqa:a"]["shape"] == [3, 5]

    with pytest.raises(RuntimeError, match="coverage"):
        load_verified_feature_index(
            manifest_path,
            manifest_sha256=file_sha256(manifest_path),
            expected_uids={"gqa:a", "gqa:b"},
            expected_feature_width=5,
        )
    record["shape"] = [4, 5]
    manifest_path.write_text(json.dumps(record) + "\n")
    with pytest.raises(RuntimeError, match="shape"):
        load_verified_feature_index(
            manifest_path,
            manifest_sha256=file_sha256(manifest_path),
            expected_uids={"gqa:a"},
            expected_feature_width=5,
        )


class _FrozenEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, input_ids, attention_mask):
        return input_ids.float().unsqueeze(-1) * self.scale


class _TinyPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, 8)

    def forward(self, question, attention_mask, image, image_mask):
        del attention_mask, image, image_mask
        pooled = question.mean(dim=1)
        return self.projection(pooled).view(-1, 2, 4)


@pytest.mark.parametrize("objective", ["exact_set_nll", "duplicated_action_bce"])
def test_train_epoch_supports_both_four_action_objectives(objective) -> None:
    encoder = _FrozenEncoder()
    predictor = _TinyPredictor()
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    common = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
        "image_features": torch.ones(2, 1, 3),
        "image_attention_mask": torch.ones(2, 1, dtype=torch.bool),
        "unique_examples": 2,
    }
    if objective == "exact_set_nll":
        batch = {
            **common,
            "valid_routes": torch.tensor([[[0, 1]], [[3, 2]]]),
            "valid_mask": torch.ones(2, 1, dtype=torch.bool),
            "route_weights": torch.ones(2, 1),
        }
    else:
        batch = {
            **common,
            "target_actions": torch.tensor([[0, 1], [3, 2], [1, 1]]),
            "route_weights": torch.tensor([1.0, 0.5, 0.5]),
            "route_sample_index": torch.tensor([0, 1, 1]),
        }

    metrics, global_step = train_epoch(
        predictor,
        encoder,
        [batch],
        optimizer,
        scheduler,
        device=torch.device("cpu"),
        objective=objective,
        accumulation_steps=1,
        gradient_clip_norm=1.0,
        duplicated_route_microbatch_size=2,
        amp_dtype=None,
        epoch=1,
        global_step=0,
    )

    assert metrics["examples"] == 2
    assert metrics["loss"] > 0
    assert global_step == 1
    assert encoder.scale.grad is None


def test_train_epoch_emits_machine_readable_early_batch_progress(capsys) -> None:
    encoder = _FrozenEncoder()
    predictor = _TinyPredictor()
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    batch = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
        "image_features": torch.ones(2, 1, 3),
        "image_attention_mask": torch.ones(2, 1, dtype=torch.bool),
        "unique_examples": 2,
        "valid_routes": torch.tensor([[[0, 1]], [[3, 2]]]),
        "valid_mask": torch.ones(2, 1, dtype=torch.bool),
        "route_weights": torch.ones(2, 1),
    }

    train_epoch(
        predictor,
        encoder,
        [batch],
        optimizer,
        scheduler,
        device=torch.device("cpu"),
        objective="exact_set_nll",
        accumulation_steps=1,
        gradient_clip_norm=1.0,
        duplicated_route_microbatch_size=2,
        amp_dtype=None,
        epoch=2,
        global_step=7,
        progress_first_batches=3,
        progress_every_batches=10,
    )

    progress = json.loads(capsys.readouterr().out)
    assert progress["event"] == "four_action_polar_train_batch"
    assert progress["epoch"] == 2
    assert progress["objective"] == "exact_set_nll"
    assert progress["batch"] == 1
    assert progress["batches"] == 1
    assert progress["examples_seen"] == 2
    assert progress["optimizer_step"] is True
    assert progress["global_step"] == 8
    assert progress["mean_loss_so_far"] > 0
