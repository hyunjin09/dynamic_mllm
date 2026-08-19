#!/usr/bin/env python3
"""One-batch real-encoder P10 preflight with no optimizer step or training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from transformers import AutoTokenizer

from binary_policy.dataset import make_duplicated_path_collator, make_set_collator
from binary_policy.losses import multi_valid_set_nll, polar_path_bce_per_path
from binary_policy.predictor import BinaryPolarBackbone, FrozenHFTokenEncoder
from binary_policy.training import predictor_state_sha256
from experiments.train_binary_polar import file_sha256, validate_gate


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{file_sha256(path)}  {path.name}\n", encoding="utf-8"
    )


def build_predictor(config: dict, input_dim: int, device: torch.device):
    return BinaryPolarBackbone(
        num_layers=int(config["policy"]["num_layers"]),
        input_dim=input_dim,
        d_model=int(config["predictor"]["d_model"]),
        num_heads=int(config["predictor"]["num_heads"]),
        num_layer_blocks=int(config["predictor"]["num_layer_blocks"]),
        dropout=float(config["predictor"]["dropout"]),
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gates = {name: validate_gate(name, spec) for name, spec in config["gates"].items()}
    smoke_path = Path(config["smoke"]["manifest"])
    if file_sha256(smoke_path) != config["smoke"]["manifest_sha256"]:
        raise RuntimeError("smoke-manifest checksum mismatch")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    wanted = set(smoke["train_positive_uids"])
    candidates = []
    with Path(config["data"]["manifest"]).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["uid"] in wanted:
                candidates.append(row)
    rows = [
        sorted((row for row in candidates if row["benchmark"] == benchmark), key=lambda row: row["uid"])[0]
        for benchmark in ("gqa", "textvqa", "chartqa")
    ]

    seed = int(config["training"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    encoder_path = config["predictor"]["embedding_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, padding_side="left", local_files_only=True)
    encoder = FrozenHFTokenEncoder(encoder_path, dtype=torch.bfloat16).to(device)

    set_batch = make_set_collator(tokenizer, route_weighting="equal")(rows)
    bce_batch = make_duplicated_path_collator(tokenizer, route_weighting="equal")(rows)
    ids = set_batch["input_ids"].to(device)
    attention = set_batch["attention_mask"].to(device)
    with torch.no_grad():
        features = encoder(ids, attention)
    if features.dtype != torch.bfloat16 or features.shape[:2] != ids.shape:
        raise RuntimeError("frozen encoder output contract mismatch")

    torch.manual_seed(seed)
    template = build_predictor(config, encoder.output_dim, device)
    initial_state = {name: value.detach().clone() for name, value in template.state_dict().items()}
    initialization_sha = predictor_state_sha256(template)
    results = {}
    for objective in ("duplicated_bce", "exact_set_nll"):
        predictor = build_predictor(config, encoder.output_dim, device)
        predictor.load_state_dict(initial_state)
        current_initialization_sha = predictor_state_sha256(predictor)
        predictor.train()
        torch.manual_seed(seed)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if objective == "exact_set_nll":
                logits = predictor(features, attention)
                loss = multi_valid_set_nll(
                    logits,
                    set_batch["valid_masks"].to(device),
                    valid_mask=set_batch["valid_mask"].to(device),
                    route_weights=set_batch["route_weights"].to(device),
                )
            else:
                sample_index = bce_batch["route_sample_index"].to(device)
                logits = predictor(features.index_select(0, sample_index), attention.index_select(0, sample_index))
                per_path = polar_path_bce_per_path(logits, bce_batch["targets"].to(device))
                loss = (
                    per_path * bce_batch["sample_weights"].to(device=device, dtype=per_path.dtype)
                ).sum() / int(bce_batch["unique_examples"])
        loss.backward()
        gradients = [parameter.grad for parameter in predictor.parameters()]
        results[objective] = {
            "loss": float(loss.detach()),
            "finite_loss": bool(torch.isfinite(loss).item()),
            "logits_shape": list(logits.shape),
            "predictor_finite_gradient_tensors": sum(
                gradient is not None and bool(torch.isfinite(gradient).all().item()) for gradient in gradients
            ),
            "predictor_parameter_tensors": len(gradients),
            "initialization_sha256": current_initialization_sha,
        }
        del predictor, logits, loss

    checks = {
        "real_tokenizer_and_encoder_load": True,
        "encoder_output_is_bfloat16": features.dtype == torch.bfloat16,
        "encoder_has_no_gradients": all(parameter.grad is None for parameter in encoder.parameters()),
        "set_batch_shape_is_3_by_variable_k_by_28": set_batch["valid_masks"].shape[0] == 3
        and set_batch["valid_masks"].shape[2] == 28,
        "duplicated_batch_uses_three_unique_inputs": bce_batch["unique_examples"] == 3,
        "both_losses_are_finite": all(row["finite_loss"] for row in results.values()),
        "both_predictor_gradients_are_finite": all(
            row["predictor_finite_gradient_tensors"] == row["predictor_parameter_tensors"]
            for row in results.values()
        ),
        "matched_initialization": len({row["initialization_sha256"] for row in results.values()}) == 1,
        "no_optimizer_step_or_checkpoint": True,
    }
    payload = {
        "schema_version": "binary_polar_p10_real_encoder_preflight_v1",
        "scope": "one real frozen-encoder batch and one backward pass per objective; zero optimizer steps",
        "passed": all(checks.values()),
        "checks": checks,
        "uids": [row["uid"] for row in rows],
        "encoder_output": {"shape": list(features.shape), "dtype": str(features.dtype)},
        "objectives": results,
        "gates": gates,
        "config_sha256": file_sha256(args.config),
    }
    write_json(args.output, payload)
    print(json.dumps({"passed": payload["passed"], "output": str(args.output)}))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
