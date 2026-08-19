#!/usr/bin/env python3
"""Small real-encoder BF16 preflight for the P12 structured objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from transformers import AutoTokenizer

from binary_policy.dataset import BinaryPolicyManifestDataset, make_structured_set_collator
from binary_policy.predictor import FrozenHFTokenEncoder, SegmentedBinaryPolarBackbone
from binary_policy.structured import structured_valid_set_nll
from experiments.train_binary_polar import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("P12 BF16 preflight requires a scheduled GPU")
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    device = torch.device("cuda")
    encoder_path = config["predictor"]["embedding_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, padding_side="left", local_files_only=True)
    encoder = FrozenHFTokenEncoder(encoder_path, dtype=torch.bfloat16).to(device)
    predictor = SegmentedBinaryPolarBackbone(
        num_layers=int(config["policy"]["num_layers"]),
        input_dim=encoder.output_dim,
        d_model=int(config["predictor"]["d_model"]),
        num_heads=int(config["predictor"]["num_heads"]),
        num_layer_blocks=int(config["predictor"]["num_layer_blocks"]),
        dropout=float(config["predictor"]["dropout"]),
    ).to(device)
    dataset = BinaryPolicyManifestDataset(
        config["data"]["manifest"],
        "train",
        max_valid_routes=int(config["data"]["max_valid_routes_per_sample"]),
    )
    frozen_smoke = json.loads(Path(config["smoke"]["manifest"]).read_text(encoding="utf-8"))
    by_uid = {row["uid"]: row for row in dataset.rows}
    rows = [by_uid[uid] for uid in frozen_smoke["train_positive_uids"][:2]]
    batch = make_structured_set_collator(
        tokenizer,
        max_length=int(config["data"]["max_question_tokens"]),
        route_weighting=str(config["data"]["route_weighting"]),
    )(rows)
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    with torch.no_grad():
        features = encoder(batch["input_ids"], batch["attention_mask"])
    predictor.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        boundary_logits, operation_logits = predictor(features, batch["attention_mask"])
        loss = structured_valid_set_nll(
            boundary_logits,
            operation_logits,
            batch["boundary_targets"],
            batch["operation_targets"],
            valid_mask=batch["valid_mask"],
            route_weights=batch["route_weights"],
        )
    loss.backward()
    trainable_gradients_finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in predictor.parameters()
    )
    frozen_gradients_absent = all(parameter.grad is None for parameter in encoder.parameters())
    predictor.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        first = predictor(features, batch["attention_mask"])
        second = predictor(features, batch["attention_mask"])
    repeated_logits_exact = torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    payload = {
        "schema_version": "binary_polar_p12_bf16_preflight_v1",
        "passed": bool(
            torch.isfinite(loss)
            and trainable_gradients_finite
            and frozen_gradients_absent
            and repeated_logits_exact
        ),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "uids": [row["uid"] for row in rows],
        "loss": float(loss.detach()),
        "features_dtype": str(features.dtype),
        "boundary_logits_shape": list(boundary_logits.shape),
        "operation_logits_shape": list(operation_logits.shape),
        "trainable_gradients_finite": trainable_gradients_finite,
        "frozen_encoder_gradients_absent": frozen_gradients_absent,
        "repeated_validation_logits_exact": repeated_logits_exact,
        "padded_routes_excluded_by_unit_test": True,
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P12 preflight: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    if not payload["passed"]:
        raise RuntimeError("P12 BF16 preflight failed")


if __name__ == "__main__":
    main()

