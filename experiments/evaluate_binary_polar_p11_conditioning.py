#!/usr/bin/env python3
"""Evaluate aligned versus within-dataset shuffled questions for P11."""

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
from binary_policy.decode import topk_factorized_masks
from binary_policy.evaluation import batch_offline_metrics
from binary_policy.losses import multi_valid_set_nll
from binary_policy.p11 import deterministic_within_dataset_shuffle
from binary_policy.predictor import BinaryPolarBackbone, FrozenHFTokenEncoder
from experiments.train_binary_polar import file_sha256, validate_gate, validate_readiness_bundle


BENCHMARKS = ("gqa", "textvqa", "chartqa")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def padded_valid_sets(rows: list[dict], route_weighting: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(len(row["valid_routes"]) for row in rows)
    width = len(rows[0]["valid_routes"][0]["mask"])
    masks = torch.zeros(len(rows), maximum, width, dtype=torch.float32)
    valid = torch.zeros(len(rows), maximum, dtype=torch.bool)
    weights = torch.zeros(len(rows), maximum, dtype=torch.float32)
    for sample_index, row in enumerate(rows):
        selected_weights = route_weights(row["valid_routes"], route_weighting)
        for route_index, (route, weight) in enumerate(zip(row["valid_routes"], selected_weights)):
            masks[sample_index, route_index] = torch.tensor(route["mask"], dtype=torch.float32)
            valid[sample_index, route_index] = True
            weights[sample_index, route_index] = weight
    return masks, valid, weights


def metrics(rows: list[dict], logits: torch.Tensor, route_weighting: str, top_k: int) -> dict:
    masks, valid, weights = padded_valid_sets(rows, route_weighting)
    result = batch_offline_metrics(logits, masks, valid, top_k=top_k)
    result["set_nll"] = float(
        multi_valid_set_nll(logits, masks, valid_mask=valid, route_weights=weights)
    )
    result["examples"] = len(rows)
    return result


@torch.inference_mode()
def score_questions(config: dict, checkpoint: dict, rows: list[dict], questions: list[str], device):
    encoder_path = config["predictor"]["embedding_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, padding_side="left", local_files_only=True)
    encoder = FrozenHFTokenEncoder(encoder_path, dtype=torch.bfloat16).to(device)
    predictor = BinaryPolarBackbone(
        num_layers=int(config["policy"]["num_layers"]),
        input_dim=encoder.output_dim,
        d_model=int(config["predictor"]["d_model"]),
        num_heads=int(config["predictor"]["num_heads"]),
        num_layer_blocks=int(config["predictor"]["num_layer_blocks"]),
        dropout=float(config["predictor"]["dropout"]),
    ).to(device).eval()
    predictor.load_state_dict(checkpoint["predictor"], strict=True)
    batches = []
    batch_size = int(config["training"]["batch_size"])
    for start in range(0, len(rows), batch_size):
        current = questions[start : start + batch_size]
        encoded = tokenizer(
            current,
            padding=True,
            truncation=True,
            max_length=int(config["data"]["max_question_tokens"]),
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        features = encoder(input_ids, attention)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            batches.append(predictor(features, attention).float().cpu())
    del predictor, encoder
    torch.cuda.empty_cache()
    return torch.cat(batches, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--best-checkpoint", required=True)
    parser.add_argument("--readiness-gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confirm-gates", action="store_true")
    args = parser.parse_args()
    if not args.confirm_gates:
        raise RuntimeError("P11 conditioning evaluation requires --confirm-gates")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    readiness = validate_readiness_bundle(args.readiness_gate, config_path)
    if not readiness["ready_for_bounded_smoke"]:
        raise RuntimeError("P11 readiness gate does not authorize bounded evaluation")
    for name, specification in config["gates"].items():
        validate_gate(name, specification)
    manifest_path = Path(config["data"]["manifest"])
    if file_sha256(manifest_path) != config["data"]["manifest_sha256"]:
        raise RuntimeError("predictor manifest checksum mismatch")
    smoke_path = Path(config["smoke"]["manifest"])
    if file_sha256(smoke_path) != config["smoke"]["manifest_sha256"]:
        raise RuntimeError("P11 smoke manifest checksum mismatch")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    by_uid = {row["uid"]: row for row in read_jsonl(manifest_path)}
    rows = [by_uid[uid] for uid in smoke["validation_positive_uids"]]

    selection = json.loads(Path(args.best_checkpoint).read_text(encoding="utf-8"))
    if selection["mode"] != "smoke":
        raise RuntimeError("P11 conditioning evaluation only accepts a smoke checkpoint")
    checkpoint = torch.load(selection["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint["config"]["predictor_initialization_sha256"] != selection["predictor_initialization_sha256"]:
        raise RuntimeError("checkpoint-selection initialization mismatch")

    shuffle = deterministic_within_dataset_shuffle(rows, seed=int(config["p11"]["shuffle_seed"]))
    aligned_questions = [row["question"] for row in rows]
    shuffled_questions = [by_uid[shuffle[row["uid"]]]["question"] for row in rows]
    device = torch.device("cuda:0")
    aligned_logits = score_questions(config, checkpoint, rows, aligned_questions, device)
    shuffled_logits = score_questions(config, checkpoint, rows, shuffled_questions, device)
    route_weighting = str(config["data"]["route_weighting"])
    top_k = int(config["evaluation"]["top_k"])

    def complete_metrics(current_logits: torch.Tensor) -> dict:
        return {
            "overall": metrics(rows, current_logits, route_weighting, top_k),
            "by_benchmark": {
                benchmark: metrics(
                    [row for row in rows if row["benchmark"] == benchmark],
                    current_logits[
                        torch.tensor([row["benchmark"] == benchmark for row in rows], dtype=torch.bool)
                    ],
                    route_weighting,
                    top_k,
                )
                for benchmark in BENCHMARKS
            },
        }

    aligned = complete_metrics(aligned_logits)
    shuffled = complete_metrics(shuffled_logits)
    top1 = topk_factorized_masks(aligned_logits, top_k=1)
    predictions = [
        {
            "uid": row["uid"],
            "benchmark": row["benchmark"],
            "mask": list(candidates[0].mask),
            "mask_key": "".join(map(str, candidates[0].mask)),
            "visual_on_layers": sum(candidates[0].mask),
            "shuffle_donor_uid": shuffle[row["uid"]],
        }
        for row, candidates in zip(rows, top1)
    ]
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P11 conditioning result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "binary_polar_p11_conditioning_v1",
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint": selection["checkpoint"],
        "objective": selection["objective"],
        "validation_uids_sha256": __import__("hashlib").sha256(
            "\n".join(row["uid"] for row in rows).encode()
        ).hexdigest(),
        "shuffle_seed": int(config["p11"]["shuffle_seed"]),
        "shuffle_has_fixed_points": any(uid == donor for uid, donor in shuffle.items()),
        "aligned": aligned,
        "shuffled": shuffled,
        "aligned_minus_shuffled": {
            "hit_at_1": aligned["overall"]["top1_valid_route_coverage"]
            - shuffled["overall"]["top1_valid_route_coverage"],
            "hit_at_5": aligned["overall"]["topk_valid_route_coverage"]
            - shuffled["overall"]["topk_valid_route_coverage"],
            "set_nll": aligned["overall"]["set_nll"] - shuffled["overall"]["set_nll"],
            "nearest_valid_hamming": aligned["overall"]["nearest_valid_hamming"]
            - shuffled["overall"]["nearest_valid_hamming"],
        },
        "predictions": predictions,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
