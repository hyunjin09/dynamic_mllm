#!/usr/bin/env python3
"""Run CPU-only deterministic checks for the binary objective comparison.

This script never loads the MLLM or frozen embedding model and never trains a
route predictor. It validates only loss, collation, and gradient contracts.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from binary_policy.decode import topk_factorized_masks
from binary_policy.losses import bernoulli_mask_log_probability, multi_valid_set_nll, polar_path_bce
from binary_policy.predictor import BinaryPolarBackbone
from binary_policy.training import train_epoch


ROOT = Path(__file__).resolve().parents[1]


class _FrozenEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, 8)
        self.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return self.embedding(input_ids)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optimize_contradictory(objective: str) -> dict:
    valid = torch.tensor([[[1, 1, 0, 0], [0, 0, 1, 1]]], dtype=torch.float64)
    logits = nn.Parameter(torch.tensor([[0.01, 0.02, -0.01, -0.02]], dtype=torch.float64))
    optimizer = torch.optim.Adam([logits], lr=0.1)
    curve = []
    finite = True
    for step in range(301):
        if objective == "exact_set_nll":
            loss = multi_valid_set_nll(logits, valid)
        else:
            loss = polar_path_bce(logits.expand(2, -1), valid[0])
        if step in (0, 1, 10, 50, 100, 300):
            curve.append({"step": step, "loss": float(loss.detach())})
        if step == 300:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite &= bool(torch.isfinite(logits.grad).all().item())
        optimizer.step()
    probabilities = torch.sigmoid(logits.detach())[0]
    top1 = list(topk_factorized_masks(logits.detach(), top_k=1)[0][0].mask)
    return {
        "objective": objective,
        "curve": curve,
        "finite_gradients": finite,
        "probabilities": probabilities.tolist(),
        "mean_absolute_distance_from_half": float(probabilities.sub(0.5).abs().mean()),
        "top1_mask": top1,
        "top1_is_complete_valid_mask": top1 in ([1, 1, 0, 0], [0, 0, 1, 1]),
    }


def _frozen_gradient_check(objective: str) -> dict:
    torch.manual_seed(19)
    encoder = _FrozenEncoder()
    predictor = BinaryPolarBackbone(
        num_layers=4,
        input_dim=8,
        d_model=16,
        num_heads=4,
        num_layer_blocks=1,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-2)
    common = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "uids": ["synthetic"],
    }
    if objective == "exact_set_nll":
        batch = {
            **common,
            "valid_masks": torch.tensor([[[1, 0, 1, 0], [0, 1, 0, 1]]], dtype=torch.float32),
            "valid_mask": torch.tensor([[True, True]]),
            "route_weights": torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        }
    else:
        batch = {
            "input_ids": common["input_ids"].expand(2, -1).clone(),
            "attention_mask": common["attention_mask"].expand(2, -1).clone(),
            "uids": ["synthetic", "synthetic"],
            "targets": torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32),
            "sample_weights": torch.tensor([0.5, 0.5], dtype=torch.float32),
            "route_sample_index": torch.tensor([0, 0], dtype=torch.long),
            "unique_examples": 1,
        }
    metrics = train_epoch(
        predictor,
        encoder,
        [batch],
        optimizer,
        device=torch.device("cpu"),
        objective=objective,
        gradient_clip_norm=1.0,
        duplicated_route_microbatch_size=1,
    )
    return {
        "objective": objective,
        "loss": metrics["loss"],
        "encoder_requires_grad_count": sum(parameter.requires_grad for parameter in encoder.parameters()),
        "encoder_gradient_count": sum(parameter.grad is not None for parameter in encoder.parameters()),
        "predictor_finite_gradient_count": sum(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
            for parameter in predictor.parameters()
        ),
        "predictor_parameter_tensor_count": sum(1 for _ in predictor.parameters()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logits = torch.tensor([[0.7, -1.2, 0.2, 1.1]], dtype=torch.float64)
    one_mask = torch.tensor([[[1, 0, 1, 1]]], dtype=torch.float64)
    single_expected = float(-bernoulli_mask_log_probability(logits, one_mask).squeeze())
    single_actual = float(multi_valid_set_nll(logits, one_mask))

    padded = torch.tensor([[[1, 0, 1, 1], [0, 1, 0, 0]]], dtype=torch.float64)
    padded_actual = float(
        multi_valid_set_nll(
            logits,
            padded,
            valid_mask=torch.tensor([[True, False]]),
            route_weights=torch.tensor([[1.0, 99.0]], dtype=torch.float64),
        )
    )
    contradictory = {
        objective: _optimize_contradictory(objective)
        for objective in ("duplicated_bce", "exact_set_nll")
    }
    gradients = {
        objective: _frozen_gradient_check(objective)
        for objective in ("duplicated_bce", "exact_set_nll")
    }
    checks = {
        "single_route_exact_equality": abs(single_actual - single_expected) <= 1e-12,
        "padded_route_zero_mass": abs(padded_actual - single_actual) <= 1e-12,
        "duplicated_bce_reaches_bit_marginals": contradictory["duplicated_bce"][
            "mean_absolute_distance_from_half"
        ]
        <= 1e-5,
        "exact_set_nll_selects_complete_valid_mode": contradictory["exact_set_nll"][
            "top1_is_complete_valid_mask"
        ],
        "all_objective_gradients_finite": all(row["finite_gradients"] for row in contradictory.values()),
        "frozen_encoder_has_no_gradients": all(row["encoder_gradient_count"] == 0 for row in gradients.values()),
        "predictor_gradients_are_finite": all(
            row["predictor_finite_gradient_count"] == row["predictor_parameter_tensor_count"]
            for row in gradients.values()
        ),
    }
    source_paths = [
        ROOT / "binary_policy/losses.py",
        ROOT / "binary_policy/dataset.py",
        ROOT / "binary_policy/training.py",
        ROOT / "binary_policy/predictor.py",
        ROOT / "tests/test_binary_policy_objective_comparison.py",
    ]
    payload = {
        "protocol": "binary_polar_loss_only_comparison_sanity_v1",
        "scope": "synthetic deterministic loss/data/gradient checks only; no predictor or MLLM training",
        "torch_version": torch.__version__,
        "single_route": {
            "expected_complete_mask_nll": single_expected,
            "exact_set_nll": single_actual,
            "absolute_error": abs(single_actual - single_expected),
        },
        "padding": {"padded_absolute_error": abs(padded_actual - single_actual)},
        "contradictory_routes": contradictory,
        "gradient_isolation": gradients,
        "checks": checks,
        "source_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths},
        "passed": all(checks.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{_sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "passed": payload["passed"], "checks": checks}, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
