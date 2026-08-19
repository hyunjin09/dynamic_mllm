#!/usr/bin/env python3
"""Execute the P12 top-1 mask on the frozen P11 60-record subset."""

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

from binary_policy.predictor import FrozenHFTokenEncoder, SegmentedBinaryPolarBackbone
from binary_policy.structured import decode_structured_top1
from experiments.evaluate_binary_polar_internal import read_jsonl, summarize
from experiments.train_binary_polar import file_sha256, validate_gate, validate_readiness_bundle
from label_regeneration.runtime import RouteEvaluator, configure_determinism, load_frozen_model


@torch.inference_mode()
def predict_masks(config, checkpoint, rows, device):
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
    output = {}
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
        input_ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        features = encoder(input_ids, attention)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            decoded = decode_structured_top1(*predictor(features, attention))
        for row, prediction in zip(current, decoded):
            output[row["uid"]] = list(prediction["mask"])
    del predictor, encoder
    torch.cuda.empty_cache()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--readiness-gate", required=True)
    parser.add_argument("--best-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confirm-gates", action="store_true")
    args = parser.parse_args()
    if not args.confirm_gates:
        raise RuntimeError("P12 execution requires --confirm-gates")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    readiness = validate_readiness_bundle(args.readiness_gate, config_path)
    if not readiness["ready_for_bounded_smoke"]:
        raise RuntimeError("P12 readiness does not authorize bounded execution")
    for name, spec in config["gates"].items():
        validate_gate(name, spec)
    smoke_path = Path(config["smoke"]["manifest"])
    manifest_path = Path(config["data"]["manifest"])
    if file_sha256(smoke_path) != config["smoke"]["manifest_sha256"]:
        raise RuntimeError("P12 smoke checksum mismatch")
    if file_sha256(manifest_path) != config["data"]["manifest_sha256"]:
        raise RuntimeError("P12 manifest checksum mismatch")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    manifest_rows = {row["uid"]: row for row in read_jsonl(manifest_path)}
    rows = [manifest_rows[uid] for uid in smoke["execution_validation_uids"]]
    selection = json.loads(Path(args.best_checkpoint).read_text(encoding="utf-8"))
    if selection["mode"] != "smoke" or selection["objective"] != "structured_exact_set_nll":
        raise RuntimeError("P12 execution accepts only its bounded structured checkpoint")
    checkpoint = torch.load(selection["checkpoint"], map_location="cpu", weights_only=False)
    predicted = predict_masks(config, checkpoint, rows, torch.device("cuda"))

    source = {row["uid"]: row for row in read_jsonl(Path("outputs/label_regeneration/v1/source_manifest_v1.jsonl"))}
    p5 = {row["uid"]: row for row in read_jsonl(Path("outputs/label_regeneration/v1/post_generation/per_sample_route_summary_v1.jsonl"))}
    record_index = {row["uid"]: row for row in read_jsonl(Path("outputs/label_regeneration/v1/post_generation/cache_record_index_v1.jsonl"))}
    configure_determinism(int(config["training"]["seed"]))
    processor, base, wrapped, device = load_frozen_model(
        config["base_model"]["model_id"], config["base_model"]["revision"], 0
    )
    stratum = {row["uid"]: row["stratum"] for row in smoke["execution_rows"]}
    result_rows = []
    for row in rows:
        mask = predicted[row["uid"]]
        key = "".join(map(str, mask))
        selected_valid = {route["key"] for route in row["valid_routes"]}
        raw = json.loads(Path(record_index[row["uid"]]["record_path"]).read_text(encoding="utf-8"))
        raw_valid = {route["mask_key"] for route in raw["candidate_executions"] if route["result_correct"]}
        evaluator = RouteEvaluator(
            processor=processor,
            base_model=base,
            wrapped_model=wrapped,
            sample=source[row["uid"]],
            device=device,
        )
        executed = evaluator.evaluate(tuple(mask), "p12_structured_exact_set")
        baseline = p5[row["uid"]]
        result_rows.append(
            {
                "uid": row["uid"],
                "benchmark": row["benchmark"],
                "stratum": stratum[row["uid"]],
                "predicted_mask": mask,
                "predicted_mask_key": key,
                "num_visual_on_layers": sum(mask),
                "selected_valid_set_size": len(selected_valid),
                "raw_cached_valid_set_size": len(raw_valid),
                "predicted_mask_in_selected_valid_set": key in selected_valid,
                "predicted_mask_in_raw_cached_valid_set": key in raw_valid,
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
    payload = {
        "schema_version": "binary_polar_p12_execution_v1",
        "strategy": "structured_exact_set_nll",
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint_record": selection,
        "rows": result_rows,
        "summary": {
            "overall": summarize(result_rows),
            "by_benchmark": {
                benchmark: summarize([row for row in result_rows if row["benchmark"] == benchmark])
                for benchmark in ("gqa", "textvqa", "chartqa")
            },
        },
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P12 execution result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
