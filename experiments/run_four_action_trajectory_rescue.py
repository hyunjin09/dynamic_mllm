#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from binary_policy.executor import capture_full_baseline, local_four_action_forward
from binary_policy.executor.inputs import build_binary_inputs
from experiments.run_four_action_answer_alignment import (
    answer_trajectory_from_cached_states,
    append_jsonl,
    baseline_state,
    git_metadata,
    intervention_state,
    load_model,
    prepare,
    set_determinism,
    sha256_file,
    unified_full_answer_trajectory,
    write_json_once,
)
from tools.research_analysis.four_action.parallelism import artifact_names, worker_layout
from tools.research_analysis.four_action.followup import trajectory_reference_from_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run population-level culprit trajectory rescues.")
    parser.add_argument("--config", type=Path, default=Path("configs/four_action_answer_alignment.yaml"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("analysis/4action_answer_alignment/trajectory_rescue/manifest.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis/4action_answer_alignment/trajectory_rescue/results"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def trajectory_result(
    model,
    processor,
    record,
    selection,
    inputs,
    prepared,
    baseline_output,
    baseline_state_row,
    correct_targets,
    wrong_target,
    full_trajectory,
    device,
    trajectory_atol,
) -> dict[str, Any]:
    output = local_four_action_forward(
        model, baseline_output, int(selection["layer"]), selection["suppressed_action"]
    )
    state = intervention_state(
        model,
        processor,
        record,
        inputs,
        output,
        correct_targets,
        wrong_target,
        device,
    )
    assert output.prefill.cache is not None
    trajectory = answer_trajectory_from_cached_states(
        model,
        output.prefill.post_layer_text_states,
        output.prefill.inputs,
        output.prefill.cache,
        baseline_state_row,
        device,
    )
    trajectory_reference = trajectory_reference_from_state(state, trajectory)
    checks = {
        "final_margin_matches_primary": abs(
            float(state["margin"]) - float(selection["expected_final_margin"])
        )
        <= 1e-5,
        "generated_ids_match_primary": state["generated_ids"]
        == selection["expected_generated_ids"],
        "correctness_matches_primary": bool(state["correct"])
        == bool(selection["expected_correct"]),
        "trajectory_final_margin_matches_reference_target_state": abs(
            float(trajectory["final_margin"])
            - float(trajectory_reference["fixed_target_state_margin"])
        )
        <= float(trajectory_atol),
    }
    return {
        "schema_version": "four_action_trajectory_rescue_v2",
        **selection,
        "state": state,
        "suppressed_trajectory": trajectory,
        "trajectory_reference": trajectory_reference,
        "full_trajectory": full_trajectory,
        "trajectory_change": {
            "final_margin_improvement": float(trajectory["final_margin"])
            - float(full_trajectory["final_margin"]),
            "peak_to_final_erosion_reduction": float(full_trajectory["peak_to_final_erosion"])
            - float(trajectory["peak_to_final_erosion"]),
            "largest_drop_magnitude_reduction": abs(
                min(0.0, float(full_trajectory["largest_adjacent_change"]))
            )
            - abs(min(0.0, float(trajectory["largest_adjacent_change"]))),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank is not None else 0))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
    if torch.cuda.device_count() != 8:
        raise RuntimeError("trajectory rescue requires exactly 8 visible GPUs")
    layout = worker_layout(rank, world, gpu_count=8)
    if layout.replicas_per_gpu != args.workers_per_gpu:
        raise RuntimeError(
            f"torchrun created {layout.replicas_per_gpu} workers/GPU but "
            f"--workers-per-gpu={args.workers_per_gpu}"
        )
    device = torch.device(f"cuda:{layout.gpu_index}")
    torch.cuda.set_device(device)
    set_determinism(int(config["seed"]) + layout.gpu_index)
    gpu_selections = [
        row for row in read_jsonl(args.manifest)
        if int(row["shard"]) == layout.gpu_index
    ]
    gpu_uids = sorted({row["uid"] for row in gpu_selections})
    replica_uids = {
        uid
        for index, uid in enumerate(gpu_uids)
        if index % layout.replicas_per_gpu == layout.replica_index
    }
    selections = [row for row in gpu_selections if row["uid"] in replica_uids]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selections:
        grouped[row["uid"]].append(row)
    cohort = {row["uid"]: row for row in read_jsonl(Path(config["cohort_manifest"]))}
    output_dir = args.output_root / f"shard_{layout.gpu_index:02d}"
    names = artifact_names(layout.replicas_per_gpu, layout.replica_index)
    result_path = output_dir / names["results"]
    completed = set()
    existing_result_paths = sorted(output_dir.glob("results*.jsonl"))
    if existing_result_paths:
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {output_dir} without --resume")
        completed = {
            row["selection_id"]
            for path in existing_result_paths
            for row in read_jsonl(path)
        }
    runtime_path = output_dir / names["runtime"]
    if not runtime_path.exists():
        write_json_once(
            runtime_path,
            {
                "schema_version": "four_action_trajectory_runtime_v1",
                "rank": rank,
                "world_size": world,
                "gpu_index": layout.gpu_index,
                "replica_index": layout.replica_index,
                "replicas_per_gpu": layout.replicas_per_gpu,
                "selection_count": len(selections),
                "gpu": torch.cuda.get_device_name(device),
                "config_sha256": sha256_file(args.config),
                "manifest_sha256": sha256_file(args.manifest),
                "git": git_metadata(),
                "all_eight_gpu_workers_required": True,
            },
        )
    model, processor = load_model(config, device)
    failures = 0
    for uid in sorted(grouped):
        pending = [row for row in grouped[uid] if row["selection_id"] not in completed]
        if not pending:
            continue
        record = cohort[uid]
        started = time.monotonic()
        try:
            prompt_text, inputs = prepare(processor, record, device)
            prepared = build_binary_inputs(model, inputs)
            baseline_output = capture_full_baseline(
                model, inputs, prepared_inputs=prepared, use_cache=True, native_causal=False
            )
            baseline_state_row, correct_targets, wrong_target = baseline_state(
                model, processor, record, prompt_text, inputs, baseline_output, device
            )
            full_trajectory = unified_full_answer_trajectory(
                model, baseline_output, baseline_state_row, device
            )
            for selection in pending:
                result = trajectory_result(
                    model,
                    processor,
                    record,
                    selection,
                    inputs,
                    prepared,
                    baseline_output,
                    baseline_state_row,
                    correct_targets,
                    wrong_target,
                    full_trajectory,
                    device,
                    config["trajectory_final_margin_atol"],
                )
                result["sample_elapsed_seconds"] = time.monotonic() - started
                append_jsonl(result_path, result)
                if not result["passed"]:
                    raise RuntimeError(
                        f"trajectory gate failed for {selection['selection_id']}: "
                        f"{[name for name, passed in result['checks'].items() if not passed]}"
                    )
            print(json.dumps({"rank": rank, "uid": uid, "completed": len(pending)}), flush=True)
        except Exception as exc:
            failures += 1
            append_jsonl(
                output_dir / names["failures"],
                {"uid": uid, "error": str(exc), "traceback": traceback.format_exc()},
            )
            break
        finally:
            torch.cuda.empty_cache()
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
