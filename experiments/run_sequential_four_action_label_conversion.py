#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import transformers
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import yaml

from binary_policy.executor import BinaryQwen25VL
from label_regeneration.runtime import configure_determinism
from tools.research_analysis.four_action.label_runtime import FourActionSampleRuntime
from tools.research_analysis.four_action.parallelism import worker_layout
from tools.research_analysis.four_action.sequential_label_conversion import (
    ExactRouteEvaluator,
    binary_to_four_action,
    convert_replay_valid_source_route,
    deduplicate_sequential_routes,
)
from tools.research_analysis.four_action.sequential_label_jobs import (
    SequentialAtomicSampleQueue,
    build_sequential_execution_contract,
    file_sha256,
    mode_topology,
    safe_filename,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact sequential four-action conversion of positive binary routes."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sequential_four_action_label_conversion.yaml"),
    )
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite completed record {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def load_model(config: dict[str, Any], device: torch.device):
    model_config = config["model"]
    snapshot = str(Path(model_config["snapshot_path"]).resolve())
    processor = AutoProcessor.from_pretrained(
        snapshot,
        revision=model_config["revision"],
        local_files_only=True,
        use_fast=False,
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        snapshot,
        revision=model_config["revision"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_config["attention_implementation"],
        low_cpu_mem_usage=True,
        device_map={"": str(device)},
    ).eval()
    base.requires_grad_(False)
    return processor, BinaryQwen25VL(base)


def rank_and_world(args: argparse.Namespace) -> tuple[int, int]:
    rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank is not None else 0))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
    return rank, world


def _source_diagnostic(record: dict[str, Any], current_full: dict[str, Any]) -> dict[str, Any]:
    source_status = record.get("source_current_all_on_status")
    source_prediction = record.get("source_current_all_on_prediction")
    return {
        "source_status": source_status,
        "current_unified_status": "correct" if current_full["correct"] else "wrong",
        "correctness_match": (
            None
            if source_status not in {"correct", "wrong"}
            else (source_status == "correct") == bool(current_full["correct"])
        ),
        "source_prediction": source_prediction,
        "current_unified_prediction": current_full["generated_answer"],
        "generated_answer_match": (
            None
            if source_prediction is None
            else str(source_prediction).strip() == str(current_full["generated_answer"]).strip()
        ),
    }


def _conversion_payload(conversion) -> dict[str, Any]:
    final_branches = []
    for branch in conversion.final_branches:
        payload = asdict(branch)
        payload["route"] = list(branch.route)
        payload["decisions"] = [asdict(decision) for decision in branch.decisions]
        final_branches.append(payload)
    return {
        "label_semantics": conversion.label_semantics,
        "final_branches": final_branches,
        "steps": [asdict(step) for step in conversion.steps],
        "maximum_branch_count": conversion.maximum_branch_count,
        "new_route_evaluations": conversion.new_route_evaluations,
    }


def process_sample(
    *,
    processor,
    model,
    record: dict[str, Any],
    device: torch.device,
    config: dict[str, Any],
    rank: int,
    mode: str,
    execution_contract: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    runtime = FourActionSampleRuntime(
        processor=processor,
        model=model,
        sample=record,
        device=device,
        scoring_timeout_seconds=float(config["scoring_timeout_seconds"]),
    )
    full = runtime.initialize_full().evaluation
    evaluate = ExactRouteEvaluator(runtime.evaluate)
    evaluate(("FULL",) * int(config["layer_count"]))
    all_off = evaluate(("IGNORE",) * int(config["layer_count"]))
    route_type = "C2C" if bool(full["correct"]) else "W2C"
    semantics = "preserving_c2c" if route_type == "C2C" else "corrective_w2c"
    conversions = []
    binary_checks = []

    for source_index, source in enumerate(record["source_positive_routes"]):
        source_route = binary_to_four_action(source["mask"])
        source_evaluation = evaluate(source_route)
        base = {
            "schema_version": "exact_sequential_raw_conversion_v1",
            "source_binary_route_id": source["source_binary_route_id"],
            "source_route_id": source["route_id"],
            "source_binary_route": source["mask"],
            "source_off_count": source["source_off_count"],
            "all_off_seed": source["source_all_off"],
            "source_route_evaluation": source_evaluation,
        }
        if mode == "smoke" and source_index == 0:
            old = runtime.evaluate_old_binary(tuple(int(value) for value in source["mask"]))
            binary_checks.append(
                {
                    "source_binary_route_id": source["source_binary_route_id"],
                    "generated_ids_match": old["generated_ids"]
                    == source_evaluation["generated_ids"],
                    "generated_answer_match": old["generated_answer"]
                    == source_evaluation["generated_answer"],
                    "correctness_match": old["correct"] == source_evaluation["correct"],
                    "old_binary": old,
                }
            )
        if not bool(source_evaluation["correct"]):
            conversions.append(
                {
                    **base,
                    "status": "source_route_replay_failure",
                    "failure_reason": (
                        "current unified executor did not reproduce evaluator correctness"
                    ),
                }
            )
            continue
        converted = convert_replay_valid_source_route(
            source["mask"],
            full_correct=bool(full["correct"]),
            evaluate=evaluate,
        )
        conversions.append({**base, "status": "converted", **_conversion_payload(converted)})

    unique = deduplicate_sequential_routes(conversions)
    converted = [row for row in conversions if row["status"] == "converted"]
    branch_occurrences = [branch for row in converted for branch in row["final_branches"]]
    elapsed = time.monotonic() - started
    return {
        "schema_version": "exact_sequential_four_action_sample_v1",
        "passed": True,
        "uid": record["uid"],
        "dataset": record["dataset"],
        "sample_id": record["sample_id"],
        "image_id": record.get("image_id"),
        "image_group_id": record.get("image_group_id"),
        "source_split": record["source_split"],
        "route_type": route_type,
        "label_semantics": semantics,
        "all_off_seed": any(bool(row.get("all_off_seed")) for row in conversions),
        "execution_contract": execution_contract,
        "current_unified_full": full,
        "current_unified_all_off": all_off,
        "source_full_diagnostic": _source_diagnostic(record, full),
        "source_positive_route_count": int(record["source_positive_route_count"]),
        "source_route_replay_valid_count": len(converted),
        "source_route_replay_failure_count": len(conversions) - len(converted),
        "raw_conversions": conversions,
        "unique_valid_four_action_routes": unique,
        "later_training_view": unique,
        "source_provenance_mapping": [
            {
                "route_key": row["route_key"],
                "source_binary_route_ids": row["source_binary_route_ids"],
            }
            for row in unique
        ],
        "pilot_old_binary_semantic_checks": binary_checks,
        "branching_summary": {
            "final_branch_occurrences": len(branch_occurrences),
            "unique_final_routes": len(unique),
            "maximum_active_branch_count": max(
                (int(row["maximum_branch_count"]) for row in converted), default=0
            ),
            "source_routes_with_branching": sum(
                int(row["maximum_branch_count"]) > 1 for row in converted
            ),
        },
        "route_evaluation_cache": {
            "unique_complete_routes_evaluated": len(evaluate.cache),
            "cache_hits": evaluate.cache_hits,
            "cache_misses": evaluate.cache_misses,
            "source_routes": int(record["source_positive_route_count"]),
        },
        "input_metadata": runtime.input_metadata,
        "geometry": runtime.geometry,
        "runtime": {
            "rank": rank,
            "gpu_index": device.index,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rank, world = rank_and_world(args)
    topology = mode_topology(args.mode)
    if world != topology["worker_count"]:
        raise RuntimeError(f"expected {topology['worker_count']} workers, got {world}")
    layout = worker_layout(rank, world, gpu_count=topology["gpu_count"])
    if layout.replicas_per_gpu != topology["workers_per_gpu"]:
        raise RuntimeError("worker/GPU replica layout differs from the frozen mode topology")
    device = torch.device(f"cuda:{layout.gpu_index}")
    torch.cuda.set_device(device)
    configure_determinism(int(config["seed"]) + rank)

    run_root = Path(config["output_root"]) / args.mode
    manifest = (
        run_root / "smoke_manifest_v1.jsonl"
        if args.mode == "smoke"
        else Path(config["source_manifest"])
    )
    rows = read_jsonl(manifest)
    if args.mode == "smoke" and len(rows) != 8:
        raise RuntimeError(f"smoke manifest must contain exactly 8 samples, got {len(rows)}")
    project_root = Path(__file__).resolve().parents[1]
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    execution_contract = build_sequential_execution_contract(
        project_root=project_root,
        config_path=args.config,
        manifest_path=manifest,
        config=config,
        mode=args.mode,
        git_commit=git_commit,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
    )
    records_root = run_root / "records"
    progress_path = run_root / "progress" / f"rank_{rank:02d}.jsonl"
    failures_path = run_root / "failures" / f"rank_{rank:02d}.jsonl"
    completed_uids = set()
    for row in rows:
        target = records_root / safe_filename(str(row["uid"]))
        if not target.exists():
            continue
        if not args.resume:
            raise FileExistsError(f"record exists without --resume: {target}")
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("uid") != row["uid"] or not existing.get("passed"):
            raise RuntimeError(f"invalid completed record: {target}")
        if existing.get("execution_contract") != execution_contract:
            raise RuntimeError(f"completed record execution contract differs: {target}")
        completed_uids.add(str(row["uid"]))

    if rank == 0:
        contract_path = run_root / "execution_contract_v1.json"
        if contract_path.exists():
            existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
            if existing_contract != execution_contract:
                raise RuntimeError("existing execution contract differs from this launch")
        else:
            write_atomic(contract_path, execution_contract)

    if args.mode == "smoke":
        assigned = [row for index, row in enumerate(rows) if index % world == rank]
        pending = [row for row in assigned if str(row["uid"]) not in completed_uids]
        iterator = iter(pending)
        next_record = lambda: next(iterator, None)
        launch_token = os.environ.get("SLURM_JOB_ID")
        work_assignment = "one_sample_per_worker"
    else:
        launch_token = os.environ.get("SLURM_JOB_ID") or os.environ.get(
            "SEQUENTIAL_FOUR_ACTION_LAUNCH_TOKEN"
        )
        if not launch_token:
            raise RuntimeError(
                "full dynamic queue requires SLURM_JOB_ID or "
                "SEQUENTIAL_FOUR_ACTION_LAUNCH_TOKEN"
            )
        restart = os.environ.get("SLURM_RESTART_COUNT", "0")
        queue = SequentialAtomicSampleQueue(
            rows,
            claim_root=run_root / "claims" / f"{launch_token}_{restart}",
            completed_uids=completed_uids,
            claimant=f"rank-{rank:02d}",
        )
        assigned = rows
        pending = None
        next_record = queue.claim_next
        work_assignment = "atomic_dynamic"

    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    append_jsonl(
        progress_path,
        {
            "event": "worker_start",
            "attempt": attempt,
            "rank": rank,
            "world": world,
            "gpu_index": layout.gpu_index,
            "replica_index": layout.replica_index,
            "assigned_samples": len(assigned),
            "pending_samples": len(pending) if pending is not None else len(rows) - len(completed_uids),
            "work_assignment": work_assignment,
            "launch_token": launch_token,
            "manifest": str(manifest.resolve()),
            "manifest_sha256": file_sha256(manifest),
            "execution_contract": execution_contract,
        },
    )
    record = next_record()
    if record is None:
        append_jsonl(
            progress_path,
            {"event": "worker_complete", "attempt": attempt, "rank": rank, "completed": 0},
        )
        return 0

    processor, model = load_model(config, device)
    failures = 0
    completed = 0
    sample_index = 0
    while record is not None:
        sample_started = time.monotonic()
        target = records_root / safe_filename(str(record["uid"]))
        append_jsonl(
            progress_path,
            {
                "event": "sample_start",
                "attempt": attempt,
                "rank": rank,
                "uid": record["uid"],
                "sample_index": sample_index,
            },
        )
        try:
            result = process_sample(
                processor=processor,
                model=model,
                record=record,
                device=device,
                config=config,
                rank=rank,
                mode=args.mode,
                execution_contract=execution_contract,
            )
            write_atomic(target, result)
            append_jsonl(
                progress_path,
                {
                    "event": "sample_complete",
                    "attempt": attempt,
                    "rank": rank,
                    "uid": record["uid"],
                    "elapsed_seconds": time.monotonic() - sample_started,
                    "unique_routes_evaluated": result["route_evaluation_cache"][
                        "unique_complete_routes_evaluated"
                    ],
                    "final_branch_occurrences": result["branching_summary"][
                        "final_branch_occurrences"
                    ],
                    "maximum_active_branch_count": result["branching_summary"][
                        "maximum_active_branch_count"
                    ],
                    "replay_failures": result["source_route_replay_failure_count"],
                },
            )
            completed += 1
        except Exception as exc:
            failures += 1
            append_jsonl(
                failures_path,
                {
                    "event": "sample_failure",
                    "attempt": attempt,
                    "rank": rank,
                    "uid": record["uid"],
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            append_jsonl(
                progress_path,
                {
                    "event": "sample_failed",
                    "attempt": attempt,
                    "rank": rank,
                    "uid": record["uid"],
                    "elapsed_seconds": time.monotonic() - sample_started,
                },
            )
        finally:
            torch.cuda.empty_cache()
        sample_index += 1
        record = next_record()

    append_jsonl(
        progress_path,
        {
            "event": "worker_complete",
            "attempt": attempt,
            "rank": rank,
            "failures": failures,
            "completed": completed,
        },
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
