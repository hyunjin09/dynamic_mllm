#!/usr/bin/env python3
"""Real-Qwen3 BF16 and modality-isolation preflight for P13."""

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

from binary_policy.dataset import BinaryPolicyManifestDataset
from binary_policy.losses import multi_valid_set_nll
from binary_policy.multimodal import MODALITIES, make_multimodal_set_collator, resolve_modality_inputs
from binary_policy.predictor import BinaryPolarBackbone, FrozenHFTokenEncoder
from experiments.train_binary_polar import file_sha256


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("P13 BF16 preflight requires a scheduled GPU")
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda")
    encoder_path = config["predictor"]["embedding_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, padding_side="left", local_files_only=True)
    encoder = FrozenHFTokenEncoder(encoder_path, dtype=torch.bfloat16).to(device)
    architecture = dict(
        num_layers=int(config["policy"]["num_layers"]),
        input_dim=encoder.output_dim,
        d_model=int(config["predictor"]["d_model"]),
        num_heads=int(config["predictor"]["num_heads"]),
        num_layer_blocks=int(config["predictor"]["num_layer_blocks"]),
        dropout=0.0,
    )
    feature_path = Path(config["p13"]["feature_manifest"])
    feature_index = {row["uid"]: row for row in read_jsonl(feature_path)}
    dataset = BinaryPolicyManifestDataset(
        config["data"]["manifest"],
        "train",
        max_valid_routes=int(config["data"]["max_valid_routes_per_sample"]),
    )
    frozen_smoke = json.loads(Path(config["smoke"]["manifest"]).read_text(encoding="utf-8"))
    by_uid = {row["uid"]: row for row in dataset.rows}
    rows = [by_uid[uid] for uid in frozen_smoke["train_positive_uids"][:3]]
    batch = make_multimodal_set_collator(
        tokenizer,
        feature_index,
        max_length=int(config["data"]["max_question_tokens"]),
        route_weighting=str(config["data"]["route_weighting"]),
    )(rows)
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    with torch.no_grad():
        features = encoder(batch["input_ids"], batch["attention_mask"])

    seed_everything(seed)
    p11 = BinaryPolarBackbone(**architecture).to(device).eval()
    seed_everything(seed)
    p13_reference = BinaryPolarBackbone(
        **architecture, image_dim=int(config["p13"]["visual_feature_width"])
    ).to(device).eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        p11_logits = p11(features, batch["attention_mask"])
        p13_question_logits = p13_reference(
            *resolve_modality_inputs(
                "question",
                features,
                batch["attention_mask"],
                batch["image_features"],
                batch["image_attention_mask"],
            )
        )
    question_only_initial_logits_exact = torch.equal(p11_logits, p13_question_logits)
    del p11, p13_reference

    modality_checks = {}
    for modality in MODALITIES:
        seed_everything(seed)
        predictor = BinaryPolarBackbone(
            **architecture, image_dim=int(config["p13"]["visual_feature_width"])
        ).to(device)
        predictor.train()
        if modality == "image":
            current_features = features.new_zeros(features.shape[0], 1, features.shape[-1])
            current_mask = batch["attention_mask"].new_zeros(features.shape[0], 1)
        else:
            current_features = features
            current_mask = batch["attention_mask"]
        inputs = resolve_modality_inputs(
            modality,
            current_features,
            current_mask,
            batch["image_features"],
            batch["image_attention_mask"],
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = predictor(*inputs)
            loss = multi_valid_set_nll(
                logits,
                batch["valid_masks"],
                valid_mask=batch["valid_mask"],
                route_weights=batch["route_weights"],
            )
        loss.backward()
        finite_grads = {
            name: bool(parameter.grad is not None and torch.isfinite(parameter.grad).all())
            for name, parameter in predictor.named_parameters()
        }
        nonzero_grads = {
            name: bool(parameter.grad is not None and parameter.grad.abs().max() > 0)
            for name, parameter in predictor.named_parameters()
        }
        predictor.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            first = predictor(*inputs)
            second = predictor(*inputs)
        modality_checks[modality] = {
            "loss": float(loss.detach()),
            "loss_finite": bool(torch.isfinite(loss)),
            "logits_shape": list(logits.shape),
            "repeated_logits_exact": torch.equal(first, second),
            "finite_gradient_tensors": sum(finite_grads.values()),
            "nonfinite_active_gradient_tensors": sum(
                parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                for parameter in predictor.parameters()
            ),
            "image_projection_active": any(
                value for name, value in nonzero_grads.items() if "image_projection" in name
            ),
            "question_projection_active": any(
                value for name, value in nonzero_grads.items() if "input_projection" in name
            ),
        }
        del predictor
    expected_activity = (
        not modality_checks["question"]["image_projection_active"]
        and modality_checks["question"]["question_projection_active"]
        and modality_checks["image"]["image_projection_active"]
        and not modality_checks["image"]["question_projection_active"]
        and modality_checks["image_question"]["image_projection_active"]
        and modality_checks["image_question"]["question_projection_active"]
    )
    passed = bool(
        question_only_initial_logits_exact
        and expected_activity
        and all(
            row["loss_finite"]
            and row["repeated_logits_exact"]
            and row["nonfinite_active_gradient_tensors"] == 0
            for row in modality_checks.values()
        )
        and all(parameter.grad is None for parameter in encoder.parameters())
    )
    payload = {
        "schema_version": "binary_polar_p13_bf16_preflight_v1",
        "passed": passed,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "uids": [row["uid"] for row in rows],
        "question_features_dtype": str(features.dtype),
        "visual_features_dtype": str(batch["image_features"].dtype),
        "question_only_initial_logits_exact_to_p11_architecture": question_only_initial_logits_exact,
        "expected_modality_gradient_activity": expected_activity,
        "frozen_qwen3_gradients_absent": all(parameter.grad is None for parameter in encoder.parameters()),
        "no_answer_or_route_outcome_features": True,
        "modality_checks": modality_checks,
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P13 preflight: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    if not passed:
        raise RuntimeError("P13 BF16 modality preflight failed")


if __name__ == "__main__":
    main()
