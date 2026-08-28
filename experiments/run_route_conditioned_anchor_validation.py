#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import time
import traceback
from pathlib import Path

import torch
import transformers
import yaml

from binary_policy.executor import capture_route_baseline
from binary_policy.executor.inputs import build_binary_inputs
from experiments.run_four_action_answer_alignment import (
    append_jsonl,
    baseline_state,
    git_metadata,
    load_model,
    prepare,
    rank_and_world,
    read_jsonl,
    set_determinism,
    sha256_file,
    write_json_once,
)
from tools.research_analysis.four_action.parallelism import worker_layout
from tools.research_analysis.four_action.route_conditioned import evaluate_until_current_correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate current unified correcting anchors.")
    parser.add_argument("--config", type=Path, default=Path("configs/four_action_route_conditioned.yaml"))
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def process_record(model, processor, record, device):
    started = time.monotonic()
    prompt_text, inputs = prepare(processor, record, device)
    prepared = build_binary_inputs(model, inputs)
    if int(prepared.visual_valid_mask.sum().item()) != int(record["visual_token_count"]):
        raise RuntimeError(f"visual token count drift for {record['uid']}")
    baselines = {}

    def evaluate(candidate):
        baseline = capture_route_baseline(
            model,
            inputs,
            candidate["mask"],
            prepared_inputs=prepared,
            use_cache=True,
        )
        state, _, _ = baseline_state(
            model,
            processor,
            record,
            prompt_text,
            inputs,
            baseline,
            device,
        )
        if state["correct"]:
            baselines[str(candidate["route_id"])] = baseline
        return state

    validation = evaluate_until_current_correct(record["anchor_candidates"], evaluate)
    selected = validation["selected"]
    if selected is None:
        checks = {
            "every_cached_candidate_evaluated": len(validation["evaluations"])
            == len(record["anchor_candidates"]),
            "no_candidate_current_correct": not any(
                row["correct"] for row in validation["evaluations"]
            ),
        }
        return {
            "schema_version": "route_conditioned_anchor_validation_v1",
            "uid": record["uid"],
            "dataset": record["dataset"],
            "image_group_id": record["image_group_id"],
            "analyzable": False,
            "exclusion_reason": "no_cached_correcting_route_current_correct",
            "candidate_evaluations": validation["evaluations"],
            "checks": checks,
            "passed": all(checks.values()),
            "elapsed_seconds": time.monotonic() - started,
        }
    baseline = baselines[str(selected["route_id"])]
    expected_actions = ["FULL" if value else "IGNORE" for value in selected["mask"]]
    checks = {
        "selected_current_correct": bool(selected["current_state"]["correct"]),
        "candidate_order_prefix_complete": [row["candidate_rank"] for row in validation["evaluations"]]
        == list(range(int(selected["candidate_rank"]) + 1)),
        "preceding_candidates_current_wrong": not any(
            row["correct"] for row in validation["evaluations"][:-1]
        ),
        "selected_layer_actions_match_mask": [row.action for row in baseline.layer_stats]
        == expected_actions,
        "selected_route_has_off_layer": any(value == 0 for value in selected["mask"]),
        "selected_route_not_all_off": any(value == 1 for value in selected["mask"]),
    }
    return {
        "schema_version": "route_conditioned_anchor_validation_v1",
        "uid": record["uid"],
        "dataset": record["dataset"],
        "image_group_id": record["image_group_id"],
        "analyzable": True,
        "anchor": selected,
        "anchor_off_layers": [index for index, value in enumerate(selected["mask"]) if value == 0],
        "anchor_cache_lengths": baseline.cache.lengths() if baseline.cache else None,
        "candidate_evaluations": validation["evaluations"],
        "checks": checks,
        "passed": all(checks.values()),
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rank, world = rank_and_world(args)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("anchor validation requires one Slurm allocation exposing all 8 GPUs")
    layout = worker_layout(rank, world, gpu_count=8)
    if layout.replicas_per_gpu != 1:
        raise RuntimeError("anchor validation requires exactly one worker per GPU")
    device = torch.device(f"cuda:{layout.gpu_index}")
    torch.cuda.set_device(device)
    set_determinism(int(config["seed"]) + layout.gpu_index)
    all_rows = read_jsonl(Path(config["candidate_manifest"]))
    selected = [row for row in all_rows if int(row["shard"]) == layout.gpu_index]
    output_dir = Path(config["anchor_validation_root"]) / f"shard_{layout.gpu_index:02d}"
    result_path = output_dir / "results.jsonl"
    failure_path = output_dir / "failures.jsonl"
    completed = set()
    if result_path.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {result_path} without --resume")
        completed = {row["uid"] for row in read_jsonl(result_path)}
    runtime_path = output_dir / "runtime.json"
    if not runtime_path.exists():
        write_json_once(
            runtime_path,
            {
                "schema_version": "route_conditioned_anchor_runtime_v1",
                "rank": rank,
                "world_size": world,
                "gpu_index": layout.gpu_index,
                "selected_count": len(selected),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "config_path": str(args.config),
                "config_sha256": sha256_file(args.config),
                "candidate_manifest_sha256": sha256_file(Path(config["candidate_manifest"])),
                "git": git_metadata(),
                "all_eight_gpu_workers_required": True,
            },
        )
    model, processor = load_model(config, device)
    failures = 0
    for index, record in enumerate(selected):
        if record["uid"] in completed:
            continue
        try:
            result = process_record(model, processor, record, device)
            append_jsonl(result_path, result)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "completed": index + 1,
                        "total": len(selected),
                        "uid": record["uid"],
                        "analyzable": result["analyzable"],
                        "fallback_count": None
                        if not result["analyzable"]
                        else result["anchor"]["fallback_count"],
                        "elapsed": result["elapsed_seconds"],
                    }
                ),
                flush=True,
            )
            if not result["passed"]:
                failures += 1
                append_jsonl(failure_path, {"uid": record["uid"], "error": "anchor validation gate failed"})
                break
        except Exception as exc:
            failures += 1
            append_jsonl(
                failure_path,
                {"uid": record["uid"], "error": str(exc), "traceback": traceback.format_exc()},
            )
            print(json.dumps({"rank": rank, "uid": record["uid"], "error": str(exc)}), flush=True)
            break
        finally:
            torch.cuda.empty_cache()
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
