#!/usr/bin/env python3
"""Execute a trained direct binary predictor on the frozen internal split.

Cached valid-set membership is diagnostic. Every selected top-1 mask is run
through the frozen binary Qwen executor, including masks absent from the cache.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from transformers import AutoTokenizer

from binary_policy.predictor import BinaryPolarBackbone, FrozenHFTokenEncoder
from label_regeneration.runtime import RouteEvaluator, configure_determinism, load_frozen_model
from experiments.train_binary_polar import file_sha256, validate_gate, validate_readiness_bundle


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_rows(rows: list[dict], *, per_dataset: int, seed: int) -> list[dict]:
    selected = []
    for benchmark in ("gqa", "textvqa", "chartqa"):
        candidates = [row for row in rows if row["benchmark"] == benchmark]
        candidates.sort(key=lambda row: sha256(f"{seed}:execute:{row['uid']}".encode()).hexdigest())
        if len(candidates) < per_dataset:
            raise RuntimeError(f"insufficient {benchmark} rows for execution smoke")
        selected.extend(candidates[:per_dataset])
    return sorted(selected, key=lambda row: row["uid"])


@torch.inference_mode()
def predict_masks(config: dict, checkpoint: dict, rows: list[dict], device: torch.device) -> dict[str, list[int]]:
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
    output: dict[str, list[int]] = {}
    batch_size = int(config["training"]["batch_size"])
    for start in range(0, len(rows), batch_size):
        current = rows[start : start + batch_size]
        encoded = tokenizer(
            [row["question"] for row in current],
            padding=True,
            truncation=True,
            max_length=int(config["data"]["max_question_tokens"]),
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        features = encoder(ids, attention)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = predictor(features, attention)
        masks = (logits >= 0).to(torch.int64).cpu().tolist()
        output.update({row["uid"]: mask for row, mask in zip(current, masks)})
    del predictor, encoder
    torch.cuda.empty_cache()
    return output


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    predicted_correct = sum(row["predicted_correct"] for row in rows)
    baseline_correct = sum(row["baseline_correct"] for row in rows)
    raw_positive = [row for row in rows if row["raw_cached_valid_set_size"] > 0]
    selected_positive = [row for row in rows if row["selected_valid_set_size"] > 0]
    uncached = [row for row in rows if not row["predicted_mask_in_raw_cached_valid_set"]]
    average_on = sum(row["num_visual_on_layers"] for row in rows) / max(total, 1)
    oracle_accuracy = sum(row["mcts_has_valid_route"] for row in rows) / max(total, 1)
    predicted_accuracy = predicted_correct / max(total, 1)
    return {
        "records": total,
        "baseline_accuracy": baseline_correct / max(total, 1),
        "predicted_mask_accuracy": predicted_accuracy,
        "full_wrong_to_predicted_correct": sum(
            (not row["baseline_correct"]) and row["predicted_correct"] for row in rows
        ),
        "full_correct_to_predicted_wrong": sum(
            row["baseline_correct"] and (not row["predicted_correct"]) for row in rows
        ),
        "unchanged_correct": sum(row["baseline_correct"] and row["predicted_correct"] for row in rows),
        "unchanged_wrong": sum((not row["baseline_correct"]) and (not row["predicted_correct"]) for row in rows),
        "average_visual_on_layers": average_on,
        "visual_on_layer_reduction_from_full": 28.0 - average_on,
        "visual_on_fraction_reduction_from_full": (28.0 - average_on) / 28.0,
        "visual_on_count_histogram": dict(sorted(Counter(row["num_visual_on_layers"] for row in rows).items())),
        "raw_cached_positive_records": len(raw_positive),
        "raw_cached_valid_set_hit_at_1": (
            sum(row["predicted_mask_in_raw_cached_valid_set"] for row in raw_positive) / len(raw_positive)
            if raw_positive else None
        ),
        "selected_supervision_hit_at_1": (
            sum(row["predicted_mask_in_selected_valid_set"] for row in selected_positive) / len(selected_positive)
            if selected_positive else None
        ),
        "uncached_top1_records": len(uncached),
        "uncached_top1_accuracy": sum(row["predicted_correct"] for row in uncached) / max(len(uncached), 1),
        "mcts_oracle_accuracy": oracle_accuracy,
        "oracle_accuracy_gap": oracle_accuracy - predicted_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--best-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--readiness-gate", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--confirm-gates", action="store_true")
    args = parser.parse_args()
    if not args.confirm_gates:
        raise RuntimeError("execution evaluation requires explicit --confirm-gates")
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must lie in [0, world-size)")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    readiness_gate = validate_readiness_bundle(args.readiness_gate, config_path)
    required_readiness = (
        readiness_gate["ready_for_bounded_smoke"]
        if args.mode == "smoke"
        else readiness_gate["ready_for_full_training"]
    )
    if not required_readiness:
        raise RuntimeError(f"P10 readiness gate does not authorize {args.mode} evaluation")
    validated_gates = {name: validate_gate(name, spec) for name, spec in config["gates"].items()}
    manifest_path = Path(config["data"]["manifest"])
    if file_sha256(manifest_path) != config["data"]["manifest_sha256"]:
        raise RuntimeError("binary predictor manifest checksum mismatch")
    selection = json.loads(Path(args.best_checkpoint).read_text(encoding="utf-8"))
    if selection.get("mode") != args.mode:
        raise RuntimeError("checkpoint-selection mode does not match requested evaluation mode")
    checkpoint_path = Path(selection["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["config"]["predictor_initialization_sha256"] != selection["predictor_initialization_sha256"]:
        raise RuntimeError("checkpoint/selection initialization hash mismatch")
    if checkpoint["config"]["resolved_objective"] != selection["objective"]:
        raise RuntimeError("checkpoint/selection objective mismatch")

    rows = [row for row in read_jsonl(manifest_path) if row["split"] == "validation"]
    if args.mode == "smoke":
        rows = select_rows(
            rows,
            per_dataset=int(config["smoke"]["execution_records_per_dataset"]),
            seed=int(config["smoke"]["selection_seed"]),
        )
        smoke_path = Path(config["smoke"]["manifest"])
        if file_sha256(smoke_path) != config["smoke"]["manifest_sha256"]:
            raise RuntimeError("frozen P10 smoke-manifest checksum mismatch")
        frozen_smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        if [row["uid"] for row in rows] != frozen_smoke["execution_validation_uids"]:
            raise RuntimeError("resolved execution-smoke identities differ from the frozen manifest")
    rows = [row for index, row in enumerate(rows) if index % args.world_size == args.rank]
    source = {row["uid"]: row for row in read_jsonl(Path("outputs/label_regeneration/v1/source_manifest_v1.jsonl"))}
    p5 = {
        row["uid"]: row
        for row in read_jsonl(Path("outputs/label_regeneration/v1/post_generation/per_sample_route_summary_v1.jsonl"))
    }
    record_index = {
        row["uid"]: row
        for row in read_jsonl(Path("outputs/label_regeneration/v1/post_generation/cache_record_index_v1.jsonl"))
    }

    configure_determinism(int(config["training"]["seed"]) + args.rank)
    device = torch.device(f"cuda:{args.rank}")
    predicted = predict_masks(config, checkpoint, rows, device)
    processor, base, wrapped, device = load_frozen_model(
        config["base_model"]["model_id"], config["base_model"]["revision"], args.rank
    )
    result_rows = []
    for row in rows:
        mask = predicted[row["uid"]]
        selected_valid_keys = {route["key"] for route in row.get("valid_routes", [])}
        raw_record = json.loads(Path(record_index[row["uid"]]["record_path"]).read_text(encoding="utf-8"))
        raw_valid_keys = {
            route["mask_key"]
            for route in raw_record["candidate_executions"]
            if route["result_correct"]
        }
        mask_key = "".join(map(str, mask))
        evaluator = RouteEvaluator(
            processor=processor,
            base_model=base,
            wrapped_model=wrapped,
            sample=source[row["uid"]],
            device=device,
        )
        executed = evaluator.evaluate(tuple(mask), "binary_predictor_top1")
        baseline = p5[row["uid"]]
        result_rows.append(
            {
                "uid": row["uid"],
                "benchmark": row["benchmark"],
                "predicted_mask": mask,
                "predicted_mask_key": mask_key,
                "num_visual_on_layers": sum(mask),
                "selected_valid_set_size": len(selected_valid_keys),
                "raw_cached_valid_set_size": len(raw_valid_keys),
                "predicted_mask_in_selected_valid_set": mask_key in selected_valid_keys,
                "predicted_mask_in_raw_cached_valid_set": mask_key in raw_valid_keys,
                "generated_ids": executed["generated_ids"],
                "prediction": executed["prediction"],
                "score": executed["score"],
                "predicted_correct": executed["result_correct"],
                "baseline_prediction": baseline["current_all_on_prediction"],
                "baseline_score": baseline["current_all_on_score"],
                "baseline_correct": baseline["current_all_on_status"] == "correct",
                "mcts_has_valid_route": baseline["has_valid_route"],
            }
        )
        print(json.dumps({"uid": row["uid"], "completed": len(result_rows)}), flush=True)

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "binary_polar_internal_execution_v1",
        "mode": args.mode,
        "rank": args.rank,
        "world_size": args.world_size,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "best_checkpoint_record": selection,
        "validated_gates": validated_gates,
        "validated_readiness_gate": readiness_gate,
        "rows": result_rows,
        "summary": {
            "overall": summarize(result_rows),
            "by_benchmark": {
                benchmark: summarize([row for row in result_rows if row["benchmark"] == benchmark])
                for benchmark in ("gqa", "textvqa", "chartqa")
            },
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
