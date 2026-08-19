#!/usr/bin/env python3
"""Execute frozen full10 top-1 masks on the unchanged 60-record subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm.auto import tqdm
import yaml

from experiments.evaluate_binary_polar_internal import read_jsonl, summarize
from experiments.train_binary_polar import file_sha256
from label_regeneration.runtime import RouteEvaluator, configure_determinism, load_frozen_model


def cache_membership_fields(
    mask_key: str, selected_valid: set[str], raw_valid: set[str]
) -> dict:
    """Return the complete cache schema required by the shared summarizer."""

    return {
        "selected_valid_set_size": len(selected_valid),
        "raw_cached_valid_set_size": len(raw_valid),
        "predicted_mask_in_selected_valid_set": mask_key in selected_valid,
        "predicted_mask_in_raw_cached_valid_set": mask_key in raw_valid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--conditioning", required=True)
    parser.add_argument("--modality", choices=("question", "image_question"), required=True)
    parser.add_argument("--selection", choices=("best_hit_at_1", "final"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    conditioning_path = Path(args.conditioning)
    conditioning = json.loads(conditioning_path.read_text(encoding="utf-8"))
    if conditioning["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("full10 execution/config checksum mismatch")
    selected = conditioning["execution_predictions"][args.modality][args.selection]
    predictions = {row["uid"]: row["mask"] for row in selected["rows"]}
    smoke = json.loads(Path(config["execution"]["manifest"]).read_text(encoding="utf-8"))
    expected_uids = smoke["execution_validation_uids"]
    if set(predictions) != set(expected_uids):
        raise RuntimeError("full10 execution identities differ from frozen 60")
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
    progress = tqdm(rows, desc=f"execute {args.modality} {args.selection}", unit="sample")
    for row in progress:
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
        executed = evaluator.evaluate(tuple(mask), f"full10_{args.modality}_{args.selection}")
        baseline = p5[row["uid"]]
        result_rows.append(
            {
                "uid": row["uid"],
                "benchmark": row["benchmark"],
                "stratum": strata[row["uid"]],
                "predicted_mask": mask,
                "predicted_mask_key": key,
                "num_visual_on_layers": sum(mask),
                **cache_membership_fields(key, selected_valid, raw_valid),
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
        progress.set_postfix(correct=sum(item["predicted_correct"] for item in result_rows))
    payload = {
        "schema_version": "binary_polar_full10_execution_v1",
        "modality": args.modality,
        "selection": args.selection,
        "epoch": selected["epoch"],
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "execution_source_sha256": file_sha256(Path(__file__)),
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
        raise FileExistsError(f"refusing to overwrite full10 execution: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
