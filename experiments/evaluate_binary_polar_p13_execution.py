#!/usr/bin/env python3
"""Execute one admitted P13 modality on the frozen 60-record subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from experiments.evaluate_binary_polar_internal import read_jsonl, summarize
from experiments.train_binary_polar import file_sha256, validate_gate, validate_readiness_bundle
from label_regeneration.runtime import RouteEvaluator, configure_determinism, load_frozen_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--readiness-gate", required=True)
    parser.add_argument("--conditioning", required=True)
    parser.add_argument("--modality", choices=("image", "image_question"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confirm-gates", action="store_true")
    args = parser.parse_args()
    if not args.confirm_gates:
        raise RuntimeError("P13 execution requires --confirm-gates")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    readiness = validate_readiness_bundle(args.readiness_gate, config_path)
    if not readiness["ready_for_bounded_smoke"]:
        raise RuntimeError("P13 readiness does not authorize bounded execution")
    for name, spec in config["gates"].items():
        validate_gate(name, spec)
    conditioning_path = Path(args.conditioning)
    conditioning = json.loads(conditioning_path.read_text(encoding="utf-8"))
    if conditioning["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("P13 conditioning/config checksum mismatch")
    if conditioning["execution_admission"]["passed"] is not True:
        raise RuntimeError("P13 prediction-level execution gate did not pass")
    predictions = {
        row["uid"]: row["mask"] for row in conditioning["execution_predictions"][args.modality]
    }
    smoke = json.loads(Path(config["smoke"]["manifest"]).read_text(encoding="utf-8"))
    expected_uids = smoke["execution_validation_uids"]
    if set(predictions) != set(expected_uids):
        raise RuntimeError("P13 execution prediction identities differ from the frozen 60")
    predictor_rows = {
        row["uid"]: row for row in read_jsonl(Path(config["data"]["manifest"]))
    }
    rows = [predictor_rows[uid] for uid in expected_uids]
    source = {
        row["uid"]: row
        for row in read_jsonl(Path("outputs/label_regeneration/v1/source_manifest_v1.jsonl"))
    }
    p5 = {
        row["uid"]: row
        for row in read_jsonl(
            Path("outputs/label_regeneration/v1/post_generation/per_sample_route_summary_v1.jsonl")
        )
    }
    record_index = {
        row["uid"]: row
        for row in read_jsonl(
            Path("outputs/label_regeneration/v1/post_generation/cache_record_index_v1.jsonl")
        )
    }
    strata = {row["uid"]: row["stratum"] for row in smoke["execution_rows"]}
    configure_determinism(int(config["training"]["seed"]))
    processor, base, wrapped, device = load_frozen_model(
        config["base_model"]["model_id"], config["base_model"]["revision"], 0
    )
    result_rows = []
    for row in rows:
        mask = predictions[row["uid"]]
        key = "".join(map(str, mask))
        selected_valid = {route["key"] for route in row["valid_routes"]}
        raw = json.loads(Path(record_index[row["uid"]]["record_path"]).read_text(encoding="utf-8"))
        raw_valid = {
            route["mask_key"]
            for route in raw["candidate_executions"]
            if route["result_correct"]
        }
        evaluator = RouteEvaluator(
            processor=processor,
            base_model=base,
            wrapped_model=wrapped,
            sample=source[row["uid"]],
            device=device,
        )
        executed = evaluator.evaluate(tuple(mask), f"p13_{args.modality}")
        baseline = p5[row["uid"]]
        result_rows.append(
            {
                "uid": row["uid"],
                "benchmark": row["benchmark"],
                "stratum": strata[row["uid"]],
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
        print(json.dumps({"modality": args.modality, "uid": row["uid"], "completed": len(result_rows)}), flush=True)
    payload = {
        "schema_version": "binary_polar_p13_execution_v1",
        "modality": args.modality,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "conditioning": str(conditioning_path),
        "conditioning_sha256": file_sha256(conditioning_path),
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
        raise FileExistsError(f"refusing to overwrite P13 execution: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
