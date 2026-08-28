#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import re
import time
import traceback
from pathlib import Path

import torch
import transformers
import yaml

from binary_policy.executor import (
    capture_route_baseline,
    route_conditioned_four_action_forward,
)
from binary_policy.executor.inputs import build_binary_inputs
from experiments.run_four_action_answer_alignment import (
    append_jsonl,
    baseline_state,
    git_metadata,
    intervention_state,
    load_model,
    max_abs,
    prepare,
    rank_and_world,
    read_jsonl,
    set_determinism,
    sha256_file,
    write_json_once,
)
from scoring.reference_likelihood import factorial_effects
from tools.research_analysis.four_action.parallelism import artifact_names, worker_layout
from tools.research_analysis.four_action.route_conditioned import (
    classify_route_conditioned_cell,
    select_execution_rows,
)


MODES = ("pilot", "full")
NEW_ACTIONS = ("READ_ONLY", "WRITE_ONLY", "FULL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run route-conditioned four-action decomposition.")
    parser.add_argument("--config", type=Path, default=Path("configs/four_action_route_conditioned.yaml"))
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _fixed_correct_target(anchor_state, correct_targets):
    selected_text = anchor_state["correct_target_scores"]["selected"]["text"]
    matches = [target for target in correct_targets if target.text == selected_text]
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve fixed correct target identity: {selected_text!r}")
    return matches[0]


def _optional_abs_difference(left, right) -> float:
    if left is None or right is None:
        return 0.0 if left is right else float("inf")
    return abs(float(left) - float(right))


def _branch_checks(output, baseline, anchor_actions, target_layer, action):
    target = output.prefill.layer_stats[target_layer]
    expected_actions = list(anchor_actions)
    expected_actions[target_layer] = action
    write_on = action in {"WRITE_ONLY", "FULL"}
    visual_change = max_abs(
        output.prefill.target_post_visual_state,
        output.prefill.target_pre_visual_state,
    )
    return {
        "shared_pre_text_exact": torch.equal(
            output.prefill.target_pre_text_state,
            baseline.pre_layer_states[target_layer][0],
        ),
        "shared_pre_visual_exact": torch.equal(
            output.prefill.target_pre_visual_state,
            baseline.pre_layer_states[target_layer][1],
        ),
        "non_target_route_context_exact": [row.action for row in output.prefill.layer_stats]
        == expected_actions,
        "target_action_exact": target.action == action,
        "target_read_semantics_exact": target.read_on is (action in {"READ_ONLY", "FULL"}),
        "target_write_semantics_exact": target.write_on is write_on,
        "unified_target_two_call_contract": target.decoder_calls == 2,
        "visual_bypass_or_update_exact": (visual_change > 0.0) if write_on else visual_change == 0.0,
    }


def process_record(model, processor, record, mode, device, config):
    started = time.monotonic()
    prompt_text, inputs = prepare(processor, record, device)
    prepared = build_binary_inputs(model, inputs)
    if int(prepared.visual_valid_mask.sum().item()) != int(record["visual_token_count"]):
        raise RuntimeError(f"visual token count drift for {record['uid']}")
    baseline = capture_route_baseline(
        model,
        inputs,
        record["anchor_route_mask"],
        prepared_inputs=prepared,
        use_cache=True,
    )
    anchor_state, correct_targets, wrong_target = baseline_state(
        model,
        processor,
        record,
        prompt_text,
        inputs,
        baseline,
        device,
    )
    fixed_correct_target = _fixed_correct_target(anchor_state, correct_targets)
    manifest_state = record["anchor_current_state"]
    anchor_actions = tuple("FULL" if value else "IGNORE" for value in record["anchor_route_mask"])
    tolerance = float(config["anchor_branch_margin_atol"])
    anchor_checks = {
        "current_anchor_correct": bool(anchor_state["correct"]),
        "manifest_generation_ids_match": anchor_state["generated_ids"] == manifest_state["generated_ids"],
        "manifest_generated_answer_match": anchor_state["generated_answer"] == manifest_state["generated_answer"],
        "manifest_evaluator_score_match": anchor_state["correctness_score"]
        == manifest_state["correctness_score"],
        "manifest_correctness_match": anchor_state["correct"] == manifest_state["correct"],
        "manifest_margin_within_tolerance": _optional_abs_difference(
            anchor_state["margin"], manifest_state["margin"]
        )
        <= tolerance,
        "anchor_schedule_exact": tuple(row.action for row in baseline.layer_stats) == anchor_actions,
        "fixed_correct_target_is_anchor_selected": fixed_correct_target.text
        == anchor_state["correct_target_scores"]["selected"]["text"],
        "wrong_target_is_original_full_answer": wrong_target is not None
        and wrong_target.text == record["full_prediction"],
    }
    cells = []
    for layer in record["anchor_off_layers"]:
        layer_started = time.monotonic()
        states = {"IGNORE": anchor_state}
        branch_checks = {}
        m00_reproduction = None
        if mode == "pilot":
            output = route_conditioned_four_action_forward(model, baseline, layer, "IGNORE")
            reproduced = intervention_state(
                model,
                processor,
                record,
                inputs,
                output,
                [fixed_correct_target],
                wrong_target,
                device,
            )
            checks = _branch_checks(output, baseline, anchor_actions, layer, "IGNORE")
            checks.update(
                {
                    "generated_ids_match_anchor": reproduced["generated_ids"]
                    == anchor_state["generated_ids"],
                    "generated_answer_match_anchor": reproduced["generated_answer"]
                    == anchor_state["generated_answer"],
                    "evaluator_score_match_anchor": reproduced["correctness_score"]
                    == anchor_state["correctness_score"],
                    "correctness_match_anchor": reproduced["correct"] == anchor_state["correct"],
                    "S_correct_within_tolerance": _optional_abs_difference(
                        reproduced["S_correct"], anchor_state["S_correct"]
                    )
                    <= tolerance,
                    "S_full_wrong_within_tolerance": _optional_abs_difference(
                        reproduced["S_full_wrong"], anchor_state["S_full_wrong"]
                    )
                    <= tolerance,
                    "margin_within_tolerance": _optional_abs_difference(
                        reproduced["margin"], anchor_state["margin"]
                    )
                    <= tolerance,
                }
            )
            m00_reproduction = {"state": reproduced, "checks": checks, "passed": all(checks.values())}
            del output
        for action in NEW_ACTIONS:
            output = route_conditioned_four_action_forward(model, baseline, layer, action)
            states[action] = intervention_state(
                model,
                processor,
                record,
                inputs,
                output,
                [fixed_correct_target],
                wrong_target,
                device,
            )
            checks = _branch_checks(output, baseline, anchor_actions, layer, action)
            checks["fixed_correct_target_identity"] = (
                states[action]["correct_target_scores"]["selected"]["text"]
                == fixed_correct_target.text
            )
            checks["fixed_wrong_target_identity"] = (
                states[action]["full_wrong_target_score"]["text"] == wrong_target.text
            )
            branch_checks[action] = {"checks": checks, "passed": all(checks.values())}
            del output
        margins = {action: float(states[action]["margin"]) for action in ("IGNORE", *NEW_ACTIONS)}
        correctness = {action: bool(states[action]["correct"]) for action in ("IGNORE", *NEW_ACTIONS)}
        cell_checks = {
            "target_layer_is_anchor_off": record["anchor_route_mask"][layer] == 0,
            "anchor_m00_correct": correctness["IGNORE"],
            "all_new_branch_gates_pass": all(row["passed"] for row in branch_checks.values()),
            "m00_reproduction_pass": m00_reproduction is None or m00_reproduction["passed"],
        }
        cells.append(
            {
                "schema_version": "route_conditioned_cell_v1",
                "target_layer": int(layer),
                "anchor_off_count": int(record["anchor_off_count"]),
                "anchor_hamming_distance_from_full": int(record["anchor_hamming_distance_from_full"]),
                "fixed_correct_target_text": fixed_correct_target.text,
                "fixed_wrong_target_text": wrong_target.text,
                "states": states,
                "effects": factorial_effects(margins),
                "taxonomy": classify_route_conditioned_cell(correctness),
                "branch_checks": branch_checks,
                "m00_reproduction": m00_reproduction,
                "checks": cell_checks,
                "passed": all(cell_checks.values()),
                "elapsed_seconds": time.monotonic() - layer_started,
            }
        )
    sample_checks = {
        **anchor_checks,
        "off_layer_count_exact": len(cells) == int(record["anchor_off_count"]),
        "off_layers_unique": len({row["target_layer"] for row in cells}) == len(cells),
        "all_cell_gates_pass": all(row["passed"] for row in cells),
    }
    return {
        "schema_version": "route_conditioned_sample_v1",
        "uid": record["uid"],
        "dataset": record["dataset"],
        "image_id": record["image_id"],
        "image_group_id": record["image_group_id"],
        "mode": mode,
        "factorial_executor": config["factorial_executor"],
        "anchor_route_id": record["anchor_route_id"],
        "anchor_route_mask": record["anchor_route_mask"],
        "anchor_off_layers": record["anchor_off_layers"],
        "anchor_off_count": record["anchor_off_count"],
        "anchor_hamming_distance_from_full": record["anchor_hamming_distance_from_full"],
        "anchor_candidate_rank": record["anchor_candidate_rank"],
        "anchor_fallback_count": record["anchor_fallback_count"],
        "work_unit_id": record["work_unit_id"],
        "fixed_correct_target_text": fixed_correct_target.text,
        "fixed_wrong_target_text": wrong_target.text,
        "anchor_state": anchor_state,
        "cells": cells,
        "sample_gate": {"checks": sample_checks, "passed": all(sample_checks.values())},
        "new_intervention_cell_count": 3 * len(cells),
        "executed_branch_count": (4 if mode == "pilot" else 3) * len(cells),
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.output_tag and not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.output_tag):
        raise ValueError("--output-tag must be a safe lowercase identifier")
    rank, world = rank_and_world(args)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("route-conditioned execution requires a Slurm allocation exposing all 8 GPUs")
    layout = worker_layout(rank, world, gpu_count=8)
    if layout.replicas_per_gpu != args.workers_per_gpu:
        raise RuntimeError("torchrun worker layout does not match --workers-per-gpu")
    device = torch.device(f"cuda:{layout.gpu_index}")
    torch.cuda.set_device(device)
    set_determinism(int(config["seed"]) + layout.gpu_index)
    manifest_path = (
        Path(config["output_root"]) / "pilot_manifest.jsonl"
        if args.mode == "pilot"
        else Path(config["anchor_manifest"])
    )
    rows = select_execution_rows(
        read_jsonl(manifest_path),
        mode=args.mode,
        gpu_index=layout.gpu_index,
        replica_index=layout.replica_index,
        replicas_per_gpu=layout.replicas_per_gpu,
    )
    root = Path(config[f"{args.mode}_root"])
    mode_directory = args.mode if not args.output_tag else f"{args.mode}__{args.output_tag}"
    output_dir = root / mode_directory / f"shard_{layout.gpu_index:02d}"
    names = artifact_names(layout.replicas_per_gpu, layout.replica_index)
    result_path = output_dir / names["results"]
    failure_path = output_dir / names["failures"]
    completed = set()
    existing = sorted(output_dir.glob("results*.jsonl"))
    if existing:
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {output_dir} without --resume")
        completed = {row["uid"] for path in existing for row in read_jsonl(path)}
    runtime_path = output_dir / names["runtime"]
    if not runtime_path.exists():
        write_json_once(
            runtime_path,
            {
                "schema_version": "route_conditioned_runtime_v1",
                "mode": args.mode,
                "output_tag": args.output_tag,
                "rank": rank,
                "world_size": world,
                "gpu_index": layout.gpu_index,
                "replica_index": layout.replica_index,
                "replicas_per_gpu": layout.replicas_per_gpu,
                "selected_count": len(rows),
                "expected_new_cells": sum(3 * int(row["anchor_off_count"]) for row in rows),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "config_path": str(args.config),
                "config_sha256": sha256_file(args.config),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "git": git_metadata(),
                "all_eight_gpu_workers_required": True,
            },
        )
    model, processor = load_model(config, device)
    failures = 0
    for index, record in enumerate(rows):
        if record["uid"] in completed:
            continue
        try:
            result = process_record(model, processor, record, args.mode, device, config)
            result["worker"] = {
                "rank": rank,
                "gpu_index": layout.gpu_index,
                "replica_index": layout.replica_index,
                "world_size": world,
            }
            append_jsonl(result_path, result)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "completed": index + 1,
                        "total": len(rows),
                        "uid": record["uid"],
                        "cells": result["new_intervention_cell_count"],
                        "elapsed": result["elapsed_seconds"],
                    }
                ),
                flush=True,
            )
            if not result["sample_gate"]["passed"]:
                failures += 1
                append_jsonl(failure_path, {"uid": record["uid"], "error": "sample gate failed"})
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
