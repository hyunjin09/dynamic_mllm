#!/usr/bin/env python3
"""Execute a validation-selected four-action POLAR route on internal labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
import yaml

from binary_policy.executor.inputs import build_binary_inputs
from experiments.evaluate_four_action_polar_external import (
    execute_actions,
    load_predictor,
    predict_actions,
)
from experiments.train_binary_polar import file_sha256
from experiments.train_four_action_online_router import write_json, write_jsonl
from four_action_online_router.data import load_source_metadata, load_verified_manifest
from four_action_online_router.metrics import summarize_execution_rows
from four_action_policy.feature_cache import load_verified_feature_index
from label_regeneration.runtime import (
    build_native_processor_inputs,
    configure_determinism,
    load_frozen_model,
)


def distributed_context(expected_world_size: int) -> tuple[int, int, int, torch.device]:
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_JOB_ID")
    if any(not os.environ.get(name) for name in required):
        raise RuntimeError("internal POLAR execution requires torchrun inside Slurm")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != expected_world_size or not torch.cuda.is_available():
        raise RuntimeError(f"internal POLAR execution requires {expected_world_size} GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    return rank, world_size, local_rank, device


def render_execution_report(summary: dict[str, Any]) -> str:
    actions = summary["mean_action_layers"]
    return "\n".join(
        [
            "## Actual unified-executor validation",
            "",
            f"- Records: {summary['records']}",
            f"- W2C correct-route execution rate: {summary['w2c_rescue_rate']:.6f}",
            f"- C2C correct-route execution rate: {summary['c2c_preservation_rate']:.6f}",
            f"- Overall routed accuracy: {summary['overall_routed_accuracy']:.6f}",
            f"- Predicted exact all-FULL fraction: {summary['all_full_rate']:.6f}",
            f"- Unique executed routes: {summary['unique_routes']}",
            f"- Mean IGNORE/READ_ONLY/WRITE_ONLY/FULL layers: "
            f"{actions['IGNORE']:.3f}/{actions['READ_ONLY']:.3f}/"
            f"{actions['WRITE_ONLY']:.3f}/{actions['FULL']:.3f}",
            "",
        ]
    )


def augment_execution_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_execution_rows(rows)
    full_key = "|".join(["FULL"] * 28)
    routes = {str(row["route_key"]) for row in rows}
    summary["all_full_rate"] = sum(row["route_key"] == full_key for row in rows) / len(rows)
    summary["unique_routes"] = len(routes)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_version") != "four_action_polar_c2c_no_allfull_v1":
        raise RuntimeError("internal evaluator requires the C2C ablation config")
    protocol = Path(config["external_evaluation"]["protocol"])
    sys.path.insert(0, str(protocol / "code"))
    rank, world_size, local_rank, device = distributed_context(args.world_size)
    try:
        configure_determinism(int(config["training"]["seed"]))
        rows = load_verified_manifest(
            config["data"]["manifest"], config["data"]["manifest_sha256"]
        )
        validation_rows = [row for row in rows if row["split"] == "validation"]
        if len(validation_rows) != int(config["validation"]["expected_records"]):
            raise RuntimeError("internal validation population differs from config")
        sources = load_source_metadata(
            config["data"]["source_manifest"],
            config["data"]["source_manifest_sha256"],
            {str(row["uid"]) for row in validation_rows},
        )
        feature_manifest = Path(config["visual_features"]["manifest"])
        feature_audit = json.loads(
            Path(config["visual_features"]["cache_audit"]).read_text(encoding="utf-8")
        )
        feature_index = load_verified_feature_index(
            feature_manifest,
            manifest_sha256=feature_audit["manifest_sha256"],
            expected_uids={str(row["uid"]) for row in rows},
            expected_feature_width=int(config["visual_features"]["feature_width"]),
            verify_tensors=False,
        )
        processor, base_model, wrapped_model, _ = load_frozen_model(
            config["base_model"]["path"], config["base_model"]["revision"], local_rank
        )
        base_model.requires_grad_(False).eval()
        tokenizer, encoder, predictor, checkpoint_path, selection = load_predictor(
            config,
            Path(args.selection),
            device,
            expected_config_sha256=file_sha256(config_path),
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_path = output_dir / f"shard_{rank:02d}.jsonl"
        local = []
        for position in range(rank, len(validation_rows), world_size):
            row = validation_rows[position]
            source = sources[row["uid"]]
            sample = {**row, **source, "local_image_path": row["image_path"]}
            inputs, _metadata = build_native_processor_inputs(processor, sample, device)
            prepared = build_binary_inputs(wrapped_model, inputs)
            visual = torch.load(
                feature_index[row["uid"]]["path"], map_location="cpu", weights_only=True
            )
            prediction = predict_actions(
                row={**row, "predictor_text": row["question"]},
                visual_rows=visual,
                tokenizer=tokenizer,
                encoder=encoder,
                predictor=predictor,
                max_question_tokens=int(config["data"]["max_question_tokens"]),
                device=device,
            )
            actions = tuple(prediction.pop("actions"))
            execution = execute_actions(
                wrapped_model=wrapped_model,
                processor=processor,
                inputs=inputs,
                prepared=prepared,
                actions=actions,
                row=sample,
                eos_token_ids=list(config["external_evaluation"]["eos_token_ids"]),
                repetition_penalty=float(
                    config["external_evaluation"]["repetition_penalty"]
                ),
            )
            local.append(
                {
                    "uid": row["uid"],
                    "dataset": row["dataset"],
                    "route_type": row["route_type"],
                    **prediction,
                    "actions": list(actions),
                    "prediction": execution["prediction"],
                    "generated_ids": execution["generated_ids"],
                    "score": execution["score"],
                    "correct": execution["correct"],
                }
            )
            if len(local) <= 3 or len(local) % 25 == 0:
                print(
                    json.dumps(
                        {
                            "event": "polar_internal_execution",
                            "rank": rank,
                            "completed": len(local),
                            "uid": row["uid"],
                            "correct": execution["correct"],
                            "route_key": prediction["route_key"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        write_jsonl(shard_path, local)
        dist.barrier()
        if rank == 0:
            combined = []
            for shard in range(world_size):
                path = output_dir / f"shard_{shard:02d}.jsonl"
                combined.extend(
                    json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            uids = [str(row["uid"]) for row in combined]
            if len(combined) != len(validation_rows) or len(uids) != len(set(uids)):
                raise RuntimeError("internal execution shards do not cover validation exactly")
            combined.sort(key=lambda row: row["uid"])
            summary = augment_execution_summary(combined)
            summary.update(
                {
                    "schema_version": "four_action_polar_internal_execution_v1",
                    "config_sha256": file_sha256(config_path),
                    "selection_sha256": file_sha256(Path(args.selection)),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "best_epoch": int(selection["best_epoch"]),
                }
            )
            write_json(output_dir / "execution_summary.json", summary)
            analysis_summary = Path(
                "analysis/4action_collapse/polar_c2c_no_allfull_execution_summary.json"
            )
            write_json(analysis_summary, summary)
            report_path = Path(config["reporting"]["report"])
            existing = report_path.read_text(encoding="utf-8")
            report_path.write_text(
                existing.rstrip() + "\n\n" + render_execution_report(summary),
                encoding="utf-8",
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
