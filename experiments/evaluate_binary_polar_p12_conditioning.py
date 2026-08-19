#!/usr/bin/env python3
"""Evaluate aligned versus frozen within-dataset shuffled questions for P12."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from transformers import AutoTokenizer

from binary_policy.dataset import route_weights
from binary_policy.p11 import deterministic_within_dataset_shuffle
from binary_policy.predictor import FrozenHFTokenEncoder, SegmentedBinaryPolarBackbone
from binary_policy.structured import (
    decode_structured_top1,
    mask_to_p12_targets,
    structured_batch_metrics,
    structured_valid_set_nll,
)
from experiments.train_binary_polar import file_sha256, validate_gate, validate_readiness_bundle


BENCHMARKS = ("gqa", "textvqa", "chartqa")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def padded_targets(rows: list[dict], weighting: str):
    maximum = max(len(row["valid_routes"]) for row in rows)
    width = len(rows[0]["valid_routes"][0]["mask"])
    masks = torch.zeros(len(rows), maximum, width)
    boundaries = torch.zeros_like(masks)
    operations = torch.full((len(rows), maximum, width), -100, dtype=torch.long)
    boundaries[:, :, 0] = 1
    operations[:, :, 0] = 0
    valid = torch.zeros(len(rows), maximum, dtype=torch.bool)
    weights = torch.zeros(len(rows), maximum)
    for sample_index, row in enumerate(rows):
        selected_weights = route_weights(row["valid_routes"], weighting)
        for route_index, (route, weight) in enumerate(zip(row["valid_routes"], selected_weights)):
            mask = route["mask"]
            boundary, operation = mask_to_p12_targets(mask)
            masks[sample_index, route_index] = torch.tensor(mask)
            boundaries[sample_index, route_index] = torch.tensor(boundary)
            operations[sample_index, route_index] = torch.tensor(operation)
            valid[sample_index, route_index] = True
            weights[sample_index, route_index] = weight
    return masks, boundaries, operations, valid, weights


def metrics(rows, boundary_logits, operation_logits, weighting):
    masks, boundaries, operations, valid, weights = padded_targets(rows, weighting)
    result = structured_batch_metrics(
        boundary_logits, operation_logits, masks, boundaries, operations, valid, weights
    )
    result["set_nll"] = float(
        structured_valid_set_nll(
            boundary_logits,
            operation_logits,
            boundaries,
            operations,
            valid_mask=valid,
            route_weights=weights,
        )
    )
    result["examples"] = len(rows)
    for key in tuple(result):
        if key.startswith("_"):
            del result[key]
    return result


@torch.inference_mode()
def score_questions(config, checkpoint, questions, device):
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
    ).to(device).eval()
    predictor.load_state_dict(checkpoint["predictor"], strict=True)
    boundary_batches, operation_batches = [], []
    batch_size = int(config["training"]["batch_size"])
    for start in range(0, len(questions), batch_size):
        encoded = tokenizer(
            questions[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=int(config["data"]["max_question_tokens"]),
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        features = encoder(input_ids, attention)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            boundary, operation = predictor(features, attention)
        boundary_batches.append(boundary.float().cpu())
        operation_batches.append(operation.float().cpu())
    return torch.cat(boundary_batches), torch.cat(operation_batches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--best-checkpoint", required=True)
    parser.add_argument("--readiness-gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confirm-gates", action="store_true")
    args = parser.parse_args()
    if not args.confirm_gates:
        raise RuntimeError("P12 conditioning evaluation requires --confirm-gates")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    readiness = validate_readiness_bundle(args.readiness_gate, config_path)
    if not readiness["ready_for_bounded_smoke"]:
        raise RuntimeError("P12 readiness does not authorize evaluation")
    for name, spec in config["gates"].items():
        validate_gate(name, spec)
    manifest_path = Path(config["data"]["manifest"])
    smoke_path = Path(config["smoke"]["manifest"])
    if file_sha256(manifest_path) != config["data"]["manifest_sha256"]:
        raise RuntimeError("manifest checksum mismatch")
    if file_sha256(smoke_path) != config["smoke"]["manifest_sha256"]:
        raise RuntimeError("smoke checksum mismatch")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    by_uid = {row["uid"]: row for row in read_jsonl(manifest_path)}
    rows = [by_uid[uid] for uid in smoke["validation_positive_uids"]]
    selection = json.loads(Path(args.best_checkpoint).read_text(encoding="utf-8"))
    checkpoint = torch.load(selection["checkpoint"], map_location="cpu", weights_only=False)
    if selection["mode"] != "smoke" or selection["objective"] != "structured_exact_set_nll":
        raise RuntimeError("P12 conditioning requires its bounded structured checkpoint")
    shuffle = deterministic_within_dataset_shuffle(rows, seed=int(config["p12"]["shuffle_seed"]))
    aligned_questions = [row["question"] for row in rows]
    shuffled_questions = [by_uid[shuffle[row["uid"]]]["question"] for row in rows]
    aligned_logits = score_questions(config, checkpoint, aligned_questions, torch.device("cuda"))
    shuffled_logits = score_questions(config, checkpoint, shuffled_questions, torch.device("cuda"))
    weighting = str(config["data"]["route_weighting"])

    def complete(logits):
        boundary, operation = logits
        return {
            "overall": metrics(rows, boundary, operation, weighting),
            "by_benchmark": {
                benchmark: metrics(
                    [row for row in rows if row["benchmark"] == benchmark],
                    boundary[torch.tensor([row["benchmark"] == benchmark for row in rows])],
                    operation[torch.tensor([row["benchmark"] == benchmark for row in rows])],
                    weighting,
                )
                for benchmark in BENCHMARKS
            },
        }

    aligned = complete(aligned_logits)
    shuffled = complete(shuffled_logits)
    decoded = decode_structured_top1(*aligned_logits)
    predictions = [
        {
            "uid": row["uid"],
            "benchmark": row["benchmark"],
            "mask": list(prediction["mask"]),
            "mask_key": "".join(map(str, prediction["mask"])),
            "visual_on_layers": sum(prediction["mask"]),
            "predicted_segments": prediction["predicted_segments"],
            "shuffle_donor_uid": shuffle[row["uid"]],
        }
        for row, prediction in zip(rows, decoded)
    ]
    payload = {
        "schema_version": "binary_polar_p12_conditioning_v1",
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint": selection["checkpoint"],
        "validation_uids_sha256": __import__("hashlib").sha256(
            "\n".join(row["uid"] for row in rows).encode()
        ).hexdigest(),
        "shuffle_seed": int(config["p12"]["shuffle_seed"]),
        "shuffle_has_fixed_points": any(uid == donor for uid, donor in shuffle.items()),
        "aligned": aligned,
        "shuffled": shuffled,
        "aligned_minus_shuffled": {
            key: aligned["overall"][source] - shuffled["overall"][source]
            for key, source in {
                "hit_at_1": "top1_valid_route_coverage",
                "set_nll": "set_nll",
                "nearest_valid_hamming": "nearest_valid_hamming",
                "fraction_all_on": "fraction_top1_all_on",
                "mean_visual_on": "average_predicted_visual_on",
            }.items()
        },
        "predictions": predictions,
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P12 conditioning result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

