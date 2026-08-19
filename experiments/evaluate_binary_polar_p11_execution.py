#!/usr/bin/env python3
"""Execute one frozen P11 strategy on the prespecified 60-record subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from experiments.evaluate_binary_polar_internal import predict_masks, read_jsonl, summarize
from experiments.train_binary_polar import file_sha256, validate_gate, validate_readiness_bundle
from label_regeneration.runtime import RouteEvaluator, configure_determinism, load_frozen_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--readiness-gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strategy", choices=("checkpoint", "constant"), required=True)
    parser.add_argument("--strategy-name", required=True)
    parser.add_argument("--best-checkpoint")
    parser.add_argument("--constant-mask")
    parser.add_argument("--confirm-gates", action="store_true")
    args = parser.parse_args()
    if not args.confirm_gates:
        raise RuntimeError("P11 execution requires --confirm-gates")
    if args.strategy == "checkpoint" and not args.best_checkpoint:
        raise ValueError("checkpoint strategy requires --best-checkpoint")
    if args.strategy == "constant" and not args.constant_mask:
        raise ValueError("constant strategy requires --constant-mask")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    readiness = validate_readiness_bundle(args.readiness_gate, config_path)
    if not readiness["ready_for_bounded_smoke"]:
        raise RuntimeError("P11 readiness gate does not authorize bounded execution")
    for name, specification in config["gates"].items():
        validate_gate(name, specification)
    manifest_path = Path(config["data"]["manifest"])
    if file_sha256(manifest_path) != config["data"]["manifest_sha256"]:
        raise RuntimeError("predictor manifest checksum mismatch")
    smoke_path = Path(config["smoke"]["manifest"])
    if file_sha256(smoke_path) != config["smoke"]["manifest_sha256"]:
        raise RuntimeError("P11 smoke manifest checksum mismatch")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    manifest_rows = {row["uid"]: row for row in read_jsonl(manifest_path)}
    rows = [manifest_rows[uid] for uid in smoke["execution_validation_uids"]]

    device = torch.device("cuda:0")
    checkpoint_record = None
    if args.strategy == "checkpoint":
        checkpoint_record = json.loads(Path(args.best_checkpoint).read_text(encoding="utf-8"))
        if checkpoint_record["mode"] != "smoke":
            raise RuntimeError("P11 only permits a bounded-smoke checkpoint")
        checkpoint = torch.load(checkpoint_record["checkpoint"], map_location="cpu", weights_only=False)
        predicted = predict_masks(config, checkpoint, rows, device)
    else:
        if len(args.constant_mask) != int(config["policy"]["num_layers"]):
            raise ValueError("constant mask width does not match the frozen policy")
        constant = [int(bit) for bit in args.constant_mask]
        if any(bit not in (0, 1) for bit in constant):
            raise ValueError("constant mask must contain only 0/1")
        predicted = {row["uid"]: constant for row in rows}

    source = {row["uid"]: row for row in read_jsonl(Path("outputs/label_regeneration/v1/source_manifest_v1.jsonl"))}
    p5 = {
        row["uid"]: row
        for row in read_jsonl(Path("outputs/label_regeneration/v1/post_generation/per_sample_route_summary_v1.jsonl"))
    }
    record_index = {
        row["uid"]: row
        for row in read_jsonl(Path("outputs/label_regeneration/v1/post_generation/cache_record_index_v1.jsonl"))
    }
    configure_determinism(int(config["training"]["seed"]))
    processor, base, wrapped, device = load_frozen_model(
        config["base_model"]["model_id"], config["base_model"]["revision"], 0
    )
    result_rows = []
    stratum_by_uid = {row["uid"]: row["stratum"] for row in smoke["execution_rows"]}
    for row in rows:
        mask = predicted[row["uid"]]
        mask_key = "".join(map(str, mask))
        selected_valid_keys = {route["key"] for route in row.get("valid_routes", [])}
        raw_record = json.loads(Path(record_index[row["uid"]]["record_path"]).read_text(encoding="utf-8"))
        raw_valid_keys = {
            route["mask_key"]
            for route in raw_record["candidate_executions"]
            if route["result_correct"]
        }
        evaluator = RouteEvaluator(
            processor=processor,
            base_model=base,
            wrapped_model=wrapped,
            sample=source[row["uid"]],
            device=device,
        )
        executed = evaluator.evaluate(tuple(mask), f"p11_{args.strategy_name}")
        baseline = p5[row["uid"]]
        result_rows.append(
            {
                "uid": row["uid"],
                "benchmark": row["benchmark"],
                "stratum": stratum_by_uid[row["uid"]],
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
        print(json.dumps({"strategy": args.strategy_name, "uid": row["uid"], "completed": len(result_rows)}), flush=True)

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P11 execution result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "binary_polar_p11_execution_v1",
        "strategy": args.strategy_name,
        "strategy_type": args.strategy,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint_record": checkpoint_record,
        "constant_mask": args.constant_mask,
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
