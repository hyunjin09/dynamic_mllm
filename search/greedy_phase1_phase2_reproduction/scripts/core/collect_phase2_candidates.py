#!/usr/bin/env python3
"""Collect random, local, and recombined candidates from Phase-1 masks."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("VISUAL_INJECTION_ROOT", Path(__file__).resolve().parents[2])).resolve()
ANALYSIS_DIR = ROOT / "analysis_outputs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ANALYSIS_DIR))

os.environ.setdefault("HF_HOME", "/home/hyemin/.cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/home/hyemin/.cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/home/hyemin/.cache/huggingface/hub")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TMPDIR", str(ROOT / "state" / "tmp"))

from harmful_validation_common import HF_HUB_CACHE, MODEL_SOURCE, is_correct, mask_one_based  # noqa: E402
from run_harmful_interventions import build_processor_inputs, evaluate_route, load_model, prepare_binary_dvrc_inputs  # noqa: E402
from dvr_qwen.generate import generation_policy_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-dir", type=Path, default=ROOT / "10k_dataset_mask" / "raw" / "search")
    parser.add_argument("--budget-statistics", type=Path, default=ROOT / "10k_dataset_mask" / "phase1" / "benchmark_budget_statistics.json")
    parser.add_argument("--gate-summary", type=Path, default=ROOT / "10k_dataset_mask" / "gate_v6" / "summary.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "10k_dataset_mask" / "raw" / "expand")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--benchmarks", default="chartqa,docvqa,gqa,textvqa")
    parser.add_argument("--per-benchmark-limit", type=int, default=0)
    parser.add_argument("--random-per-budget", type=int, default=2)
    parser.add_argument("--local-per-operation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--model-source", type=Path, default=MODEL_SOURCE)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--processor-use-fast", choices=["auto", "true", "false"], default="false")
    parser.add_argument("--first-gpu-max-memory-gb", type=int, default=30)
    parser.add_argument("--other-gpu-max-memory-gb", type=int, default=46)
    parser.add_argument("--cpu-max-memory-gb", type=int, default=0)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def route_key(route: list[int]) -> str:
    return "".join(str(int(value)) for value in route)


def route_id(uid: str, route: list[int]) -> str:
    digest = hashlib.sha256(f"{uid}:{route_key(route)}".encode()).hexdigest()[:16]
    return f"{uid}:mask:{digest}"


def safe_filename(uid: str) -> str:
    return uid.replace(":", "__").replace("/", "_") + ".json"


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def choose_success_bases(payload: dict[str, Any], max_bases: int = 3) -> list[list[int]]:
    by_key = {
        route_key(next(row["visual_on_mask"] for row in payload["candidate_executions"] if row["route_id"] == final["final_route_id"])):
        next(row["visual_on_mask"] for row in payload["candidate_executions"] if row["route_id"] == final["final_route_id"])
        for final in payload["permutation_finals"]
        if final["final_correct"]
    }
    candidates = sorted(by_key.values(), key=lambda route: (sum(route), route_key(route)))
    if len(candidates) <= max_bases:
        return candidates
    selected = [candidates[0]]
    while len(selected) < max_bases:
        remaining = [route for route in candidates if route not in selected]
        route = min(
            remaining,
            key=lambda item: (
                max(sum(a != b for a, b in zip(item, chosen)) for chosen in selected) * -1,
                route_key(item),
            ),
        )
        selected.append(route)
    return selected


def add_candidate(plan: dict[str, dict[str, Any]], route: list[int], origin: dict[str, Any]) -> None:
    key = route_key(route)
    if key not in plan:
        plan[key] = {"route": route, "origins": []}
    if origin not in plan[key]["origins"]:
        plan[key]["origins"].append(origin)


def candidate_plan(payload: dict[str, Any], budget_center: int, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    sample = payload["sample"]
    uid = sample["uid"]
    num_layers = int(payload["runtime"]["num_layers"])
    rng = random.Random(f"{args.seed}:{uid}")
    plan: dict[str, dict[str, Any]] = {}

    for budget in sorted({max(0, min(num_layers, budget_center + delta)) for delta in (-2, 0, 2)}):
        for index in range(args.random_per_budget):
            chosen = set(rng.sample(range(num_layers), budget))
            route = [1 if layer in chosen else 0 for layer in range(num_layers)]
            add_candidate(plan, route, {"family": "budget_stratified_random", "budget": budget, "draw": index, "seed": args.seed})

    bases = choose_success_bases(payload)
    for base_index, base in enumerate(bases):
        on = [idx for idx, value in enumerate(base) if value]
        off = [idx for idx, value in enumerate(base) if not value]
        for index in range(min(args.local_per_operation, len(on) * len(off))):
            if not on or not off:
                break
            remove_idx = rng.choice(on)
            add_idx = rng.choice(off)
            route = list(base)
            route[remove_idx] = 0
            route[add_idx] = 1
            add_candidate(plan, route, {"family": "same_budget_swap", "base_index": base_index, "draw": index})
        for index, add_idx in enumerate(rng.sample(off, min(args.local_per_operation, len(off)))):
            route = list(base)
            route[add_idx] = 1
            add_candidate(plan, route, {"family": "add_one", "base_index": base_index, "layer_one_based": add_idx + 1, "draw": index})
        for index, remove_idx in enumerate(rng.sample(on, min(args.local_per_operation, len(on)))):
            route = list(base)
            route[remove_idx] = 0
            add_candidate(plan, route, {"family": "remove_one", "base_index": base_index, "layer_one_based": remove_idx + 1, "draw": index})

    for left in range(len(bases)):
        for right in range(left + 1, len(bases)):
            union = [int(a or b) for a, b in zip(bases[left], bases[right])]
            intersection = [int(a and b) for a, b in zip(bases[left], bases[right])]
            add_candidate(plan, union, {"family": "success_union", "base_pair": [left, right]})
            add_candidate(plan, intersection, {"family": "success_intersection", "base_pair": [left, right]})
    return plan


def main() -> None:
    args = parse_args()
    if args.self_test:
        route = [1, 0, 1, 0]
        assert route_key(route) == "1010"
        plan: dict[str, dict[str, Any]] = {}
        add_candidate(plan, route, {"family": "test"})
        add_candidate(plan, route, {"family": "test"})
        assert len(plan) == 1 and len(plan["1010"]["origins"]) == 1
        print("phase2 candidate collector self-test ok")
        return
    args.phase1_dir = args.phase1_dir.resolve()
    args.budget_statistics = args.budget_statistics.resolve()
    args.gate_summary = args.gate_summary.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard settings")
    gate = json.loads(args.gate_summary.read_text(encoding="utf-8"))
    require_source_anchor_match = bool(gate.get("require_source_anchor_match", True))
    require_saved_generated_id_match = bool(gate.get("require_saved_generated_id_match", True))
    if (
        not gate.get("pass_current_hf_binary_ids")
        or gate.get("current_hf_binary_prediction_matches") != gate.get("samples_completed")
        or gate.get("current_hf_binary_score_matches") != gate.get("samples_completed")
        or (require_source_anchor_match and not gate.get("pass_source_score"))
        or (require_saved_generated_id_match and not gate.get("pass_available_saved_ids"))
    ):
        raise RuntimeError(f"generation anchor gate did not pass: {gate}")
    benchmarks = {value.strip().lower() for value in args.benchmarks.split(",") if value.strip()}
    phase1_paths = [
        path
        for path in sorted(args.phase1_dir.glob("shard_*_of_*/samples/*.json"))
        if path.name.split("__", 1)[0].lower() in benchmarks
    ]
    if args.per_benchmark_limit > 0:
        selected_paths: list[Path] = []
        benchmark_counts: dict[str, int] = {}
        for path in phase1_paths:
            benchmark = path.name.split("__", 1)[0].lower()
            if benchmark_counts.get(benchmark, 0) >= args.per_benchmark_limit:
                continue
            benchmark_counts[benchmark] = benchmark_counts.get(benchmark, 0) + 1
            selected_paths.append(path)
        phase1_paths = selected_paths
    phase1_paths = [path for index, path in enumerate(phase1_paths) if index % args.num_shards == args.shard_index]
    if args.max_samples > 0:
        phase1_paths = phase1_paths[: args.max_samples]
    budgets = json.loads(args.budget_statistics.read_text(encoding="utf-8"))

    import torch

    model, processor, device = load_model(args)
    if args.processor_use_fast != "auto":
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            str(args.model_source),
            cache_dir=str(HF_HUB_CACHE),
            local_files_only=True,
            use_fast=args.processor_use_fast == "true",
        )
    shard_dir = args.output_dir / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    sample_dir = shard_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    errors_path = shard_dir / "errors.jsonl"
    completed = skipped = errors = 0
    started = time.time()

    with errors_path.open("a", encoding="utf-8") as error_handle:
        for phase1_path in phase1_paths:
            phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
            sample = phase1["sample"]
            output_path = sample_dir / safe_filename(sample["uid"])
            if output_path.exists():
                skipped += 1
                continue
            try:
                budget_row = budgets.get(f"{sample['data_split']}/{sample['benchmark']}") or budgets[f"all/{sample['benchmark']}"]
                plan = candidate_plan(phase1, int(budget_row["rounded_budget_center"]), args)
                phase1_keys = {route_key(row["visual_on_mask"]): row["route_id"] for row in phase1["candidate_executions"]}
                processor_inputs = build_processor_inputs(processor, sample)
                prepared = prepare_binary_dvrc_inputs(model, processor_inputs)
                new_executions = []
                requests = []
                for key, item in sorted(plan.items()):
                    if key in phase1_keys:
                        requests.append({"route_id": phase1_keys[key], "already_in_phase1": True, "origins": item["origins"]})
                        continue
                    result = evaluate_route(
                        model=model,
                        processor=processor,
                        processor_inputs=processor_inputs,
                        prepared_binary_inputs=prepared,
                        sample=sample,
                        route=item["route"],
                        embed_device=device,
                    )
                    record = {
                        "route_id": route_id(sample["uid"], item["route"]),
                        "visual_on_mask": item["route"],
                        "mask_one_based": mask_one_based(item["route"]),
                        "num_visual_on_layers": sum(item["route"]),
                        "prediction": result["prediction"],
                        "generated_ids": result["generated_ids"],
                        "score": float(result["score"]),
                        "result_correct": is_correct(float(result["score"]), float(sample["correctness_threshold"])),
                        "origins": item["origins"],
                    }
                    new_executions.append(record)
                    requests.append({"route_id": record["route_id"], "already_in_phase1": False, "origins": item["origins"]})
                atomic_write(
                    output_path,
                    {
                        "phase": "random_local_recombination",
                        "sample": sample,
                        "phase1_sample_file": relative_or_absolute(phase1_path),
                        "benchmark_budget_center": int(budget_row["rounded_budget_center"]),
                        "candidate_generation": {
                            "random_per_budget": args.random_per_budget,
                            "local_per_operation": args.local_per_operation,
                            "seed": args.seed,
                        },
                        "route_requests": requests,
                        "new_candidate_executions": new_executions,
                        "runtime": {
                            "generation_policy": generation_policy_record(model),
                            "attn_implementation": args.attn_implementation,
                            "model_source": str(args.model_source),
                            "processor_use_fast": args.processor_use_fast,
                            "gate_summary": str(args.gate_summary),
                        },
                    },
                )
                completed += 1
                print(
                    json.dumps(
                        {
                            "uid": sample["uid"],
                            "requested_routes": len(requests),
                            "new_executions": len(new_executions),
                            "elapsed_seconds": round(time.time() - started, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                del processor_inputs, prepared
            except Exception as exc:
                errors += 1
                error_handle.write(json.dumps({"uid": sample["uid"], "error": repr(exc)}, sort_keys=True) + "\n")
                error_handle.flush()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    atomic_write(
        shard_dir / "summary.json",
        {
            "selected": len(phase1_paths),
            "completed": completed,
            "skipped": skipped,
            "errors": errors,
            "elapsed_seconds": time.time() - started,
        },
    )


if __name__ == "__main__":
    main()
