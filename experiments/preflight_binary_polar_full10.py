#!/usr/bin/env python3
"""One-step full10 runtime and memory preflight without scientific training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import yaml

from binary_policy.dataset import (
    BinaryPolicyManifestDataset,
    make_duplicated_path_collator,
    make_set_collator,
)
from binary_policy.losses import multi_valid_set_nll, polar_path_bce_per_path
from binary_policy.multimodal import (
    make_multimodal_duplicated_path_collator,
    make_multimodal_set_collator,
)
from binary_policy.predictor import BinaryPolarBackbone, FrozenHFTokenEncoder
from binary_policy.training import predictor_state_sha256
from experiments.train_binary_polar import file_sha256
from experiments.train_binary_polar_full10 import scale_gradients_to_sample_mean


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    objective = str(config["training"]["objective"])
    if objective not in {"exact_set_nll", "duplicated_bce"}:
        raise RuntimeError(f"unsupported full10 preflight objective: {objective}")
    if not torch.cuda.is_available():
        raise RuntimeError("full10 preflight requires a scheduled GPU")
    device = torch.device("cuda")
    seed = int(config["training"]["seed"])
    seed_all(seed)
    encoder_path = config["predictor"]["embedding_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_path, padding_side="left", local_files_only=True
    )
    encoder = FrozenHFTokenEncoder(encoder_path, dtype=torch.bfloat16).to(device)
    architecture = dict(
        num_layers=int(config["policy"]["num_layers"]),
        input_dim=encoder.output_dim,
        d_model=int(config["predictor"]["d_model"]),
        num_heads=int(config["predictor"]["num_heads"]),
        num_layer_blocks=int(config["predictor"]["num_layer_blocks"]),
        dropout=float(config["predictor"]["dropout"]),
    )
    feature_index = {
        row["uid"]: row
        for row in read_jsonl(Path(config["visual_features"]["manifest"]))
    }
    dataset = BinaryPolicyManifestDataset(
        config["data"]["manifest"],
        "train",
        max_valid_routes=int(config["data"]["max_valid_routes_per_sample"]),
    )
    width = int(config["training"]["physical_batch_size"])
    # Use the longest cached images to stress the intended physical batch.
    rows = sorted(
        dataset.rows,
        key=lambda row: (-int(feature_index[row["uid"]]["visual_tokens"]), row["uid"]),
    )[:width]
    checks = {}
    shared_hashes = {}
    for modality in ("question", "image_question"):
        seed_all(seed)
        predictor = BinaryPolarBackbone(
            **architecture,
            **(
                {"image_dim": int(config["visual_features"]["feature_width"])}
                if modality == "image_question"
                else {}
            ),
        ).to(device)
        shared_hashes[modality] = {
            name: tensor.detach().cpu()
            for name, tensor in predictor.state_dict().items()
            if "image_projection" not in name
        }
        if objective == "exact_set_nll":
            collator = (
                make_multimodal_set_collator(
                    tokenizer,
                    feature_index,
                    max_length=int(config["data"]["max_question_tokens"]),
                    route_weighting=config["data"]["route_weighting"],
                )
                if modality == "image_question"
                else make_set_collator(
                    tokenizer,
                    max_length=int(config["data"]["max_question_tokens"]),
                    route_weighting=config["data"]["route_weighting"],
                )
            )
        else:
            collator = (
                make_multimodal_duplicated_path_collator(
                    tokenizer,
                    feature_index,
                    max_length=int(config["data"]["max_question_tokens"]),
                    route_weighting=config["data"]["route_weighting"],
                )
                if modality == "image_question"
                else make_duplicated_path_collator(
                    tokenizer,
                    max_length=int(config["data"]["max_question_tokens"]),
                    route_weighting=config["data"]["route_weighting"],
                )
            )
        batch = collator(rows)
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        optimizer = torch.optim.AdamW(
            predictor.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(config["training"]["warmup_steps"]),
            num_training_steps=480,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            question = encoder(batch["input_ids"], batch["attention_mask"])
        if objective == "exact_set_nll":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = (
                    predictor(question, batch["attention_mask"])
                    if modality == "question"
                    else predictor(
                        question,
                        batch["attention_mask"],
                        batch["image_features"],
                        batch["image_attention_mask"],
                    )
                )
                loss = multi_valid_set_nll(
                    logits,
                    batch["valid_masks"],
                    valid_mask=batch["valid_mask"],
                    route_weights=batch["route_weights"],
                )
            loss_value = float(loss.detach())
            (loss * width).backward()
            logits_shape = list(logits.shape)
        else:
            route_count = int(batch["targets"].shape[0])
            chunk_size = int(config["training"]["duplicated_route_microbatch_size"])
            loss_sum = 0.0
            for start in range(0, route_count, chunk_size):
                stop = min(start + chunk_size, route_count)
                indices = batch["route_sample_index"][start:stop]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = (
                        predictor(
                            question.index_select(0, indices),
                            batch["attention_mask"].index_select(0, indices),
                        )
                        if modality == "question"
                        else predictor(
                            question.index_select(0, indices),
                            batch["attention_mask"].index_select(0, indices),
                            batch["image_features"].index_select(0, indices),
                            batch["image_attention_mask"].index_select(0, indices),
                        )
                    )
                    per_path = polar_path_bce_per_path(logits, batch["targets"][start:stop])
                    chunk_loss = (
                        per_path
                        * batch["sample_weights"][start:stop].to(per_path.dtype)
                    ).sum()
                chunk_loss.backward()
                loss_sum += float(chunk_loss.detach())
            loss_value = loss_sum / width
            logits_shape = [route_count, int(config["policy"]["num_layers"])]
        scale_gradients_to_sample_mean(predictor, width)
        finite_gradients = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in predictor.parameters()
        )
        optimizer.step()
        scheduler.step()
        predictor.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            first = (
                predictor(question, batch["attention_mask"])
                if modality == "question"
                else predictor(
                    question,
                    batch["attention_mask"],
                    batch["image_features"],
                    batch["image_attention_mask"],
                )
            )
            second = (
                predictor(question, batch["attention_mask"])
                if modality == "question"
                else predictor(
                    question,
                    batch["attention_mask"],
                    batch["image_features"],
                    batch["image_attention_mask"],
                )
            )
        checks[modality] = {
            "loss": loss_value,
            "loss_finite": math.isfinite(loss_value),
            "finite_predictor_gradients": finite_gradients,
            "frozen_encoder_gradients_absent": all(
                parameter.grad is None for parameter in encoder.parameters()
            ),
            "post_step_repeated_logits_exact": torch.equal(first, second),
            "logits_shape": logits_shape,
            "maximum_visual_tokens": max(
                int(feature_index[row["uid"]]["visual_tokens"]) for row in rows
            ),
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
            "initialization_sha256": predictor_state_sha256(predictor),
        }
        del predictor, optimizer, scheduler, batch
        torch.cuda.empty_cache()
    shared_names = set(shared_hashes["question"])
    shared_exact = shared_names == set(shared_hashes["image_question"]) and all(
        torch.equal(shared_hashes["question"][name], shared_hashes["image_question"][name])
        for name in shared_names
    )
    passed = shared_exact and all(
        row["loss_finite"]
        and row["finite_predictor_gradients"]
        and row["frozen_encoder_gradients_absent"]
        and row["post_step_repeated_logits_exact"]
        for row in checks.values()
    )
    payload = {
        "schema_version": "binary_polar_full10_preflight_v1",
        "passed": passed,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "physical_batch_size": width,
        "objective": objective,
        "stress_uids": [row["uid"] for row in rows],
        "shared_initialization_exact": shared_exact,
        "checks": checks,
        "scientific_training_steps": 0,
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite full10 preflight: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    if not passed:
        raise RuntimeError("full10 runtime preflight failed")


if __name__ == "__main__":
    main()
