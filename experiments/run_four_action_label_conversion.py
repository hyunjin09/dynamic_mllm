#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
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
import yaml
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from binary_policy.executor import BinaryQwen25VL
from label_regeneration.runtime import configure_determinism
from tools.research_analysis.four_action.label_conversion import (
    CachedRouteEvaluator,
    binary_to_four_action,
    canonical_route,
    convert_valid_source_route,
    deduplicate_final_routes,
)
from tools.research_analysis.four_action.label_jobs import (
    AtomicSampleQueue,
    balanced_worker_rows,
    build_conversion_execution_contract,
)
from tools.research_analysis.four_action.label_runtime import FourActionSampleRuntime
from tools.research_analysis.four_action.parallelism import worker_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert binary route labels into four actions.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/four_action_label_conversion.yaml")
    )
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_filename(uid: str) -> str:
    readable = uid.replace(":", "__").replace("/", "_")
    return f"{readable}_{hashlib.sha256(uid.encode()).hexdigest()[:10]}.json"


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
    digest = sha256_file(path)
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
            None if source_status not in {"correct", "wrong"}
            else (source_status == "correct") == bool(current_full["correct"])
        ),
        "source_prediction": source_prediction,
        "current_unified_prediction": current_full["generated_answer"],
        "generated_answer_match": (
            None if source_prediction is None
            else str(source_prediction).strip() == str(current_full["generated_answer"]).strip()
        ),
    }


def _conversion_payload(conversion) -> dict[str, Any]:
    return {
        "final_route": list(conversion.route),
        "final_evaluation": conversion.evaluation,
        "label_semantics": conversion.label_semantics,
        "purification": None if conversion.purification is None else asdict(conversion.purification),
        "refinement": None if conversion.refinement is None else asdict(conversion.refinement),
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
    cached = CachedRouteEvaluator(runtime.evaluate)
    cached(("FULL",) * int(config["layer_count"]))
    all_off = cached(("IGNORE",) * int(config["layer_count"]))
    conversions = []
    pilot_binary_checks = []
    for source_index, source in enumerate(record["source_positive_routes"]):
        source_route = binary_to_four_action(source["mask"])
        source_evaluation = cached(source_route)
        base = {
            "schema_version": "four_action_raw_conversion_v1",
            "source_binary_route_id": source["source_binary_route_id"],
            "source_route_id": source["route_id"],
            "source_binary_route": source["mask"],
            "source_off_count": source["source_off_count"],
            "all_off_seed": source["source_all_off"],
            "source_route_evaluation": source_evaluation,
        }
        if mode == "pilot" and source_index == 0:
            old = runtime.evaluate_old_binary(tuple(int(value) for value in source["mask"]))
            pilot_binary_checks.append(
                {
                    "source_binary_route_id": source["source_binary_route_id"],
                    "generated_ids_match": old["generated_ids"] == source_evaluation["generated_ids"],
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
                    "failure_reason": "current unified executor did not reproduce evaluator correctness",
                }
            )
            continue
        converted = convert_valid_source_route(
            source["mask"],
            full_correct=bool(full["correct"]),
            evaluate=cached,
            beam_width=int(config["beam_width"]),
        )
        conversions.append({**base, "status": "converted", **_conversion_payload(converted)})

    unique = deduplicate_final_routes(conversions)
    semantics = "preserving_c2c" if full["correct"] else "corrective_w2c"
    canonical = None if not unique else canonical_route(unique, label_semantics=semantics)
    elapsed = time.monotonic() - started
    return {
        "schema_version": "four_action_label_conversion_sample_v1",
        "passed": True,
        "uid": record["uid"],
        "dataset": record["dataset"],
        "sample_id": record["sample_id"],
        "image_id": record.get("image_id"),
        "image_group_id": record.get("image_group_id"),
        "source_split": record["source_split"],
        "label_semantics": semantics,
        "execution_contract": execution_contract,
        "current_unified_full": full,
        "current_unified_all_off": all_off,
        "source_full_diagnostic": _source_diagnostic(record, full),
        "source_positive_route_count": int(record["source_positive_route_count"]),
        "source_route_replay_valid_count": sum(row["status"] == "converted" for row in conversions),
        "source_route_replay_failure_count": sum(
            row["status"] == "source_route_replay_failure" for row in conversions
        ),
        "raw_conversions": conversions,
        "unique_valid_four_action_routes": unique,
        "canonical_4action_route": canonical,
        "pilot_old_binary_semantic_checks": pilot_binary_checks,
        "route_evaluation_cache": {
            "unique_complete_routes_evaluated": len(cached.cache),
            "source_routes": int(record["source_positive_route_count"]),
        },
        "input_metadata": runtime.input_metadata,
        "geometry": runtime.geometry,
        "runtime": {
            "rank": rank,
            "gpu_index": device.index,
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    rank, world = rank_and_world(args)
    if world != int(config["worker_count"]):
        raise RuntimeError(f"expected {config['worker_count']} workers, got {world}")
    layout = worker_layout(rank, world, gpu_count=int(config["gpu_count"]))
    if layout.replicas_per_gpu != int(config["workers_per_gpu"]):
        raise RuntimeError("worker/GPU replica layout differs from the frozen configuration")
    device = torch.device(f"cuda:{layout.gpu_index}")
    torch.cuda.set_device(device)
    configure_determinism(int(config["seed"]) + rank)

    if args.mode == "pilot":
        manifest = Path(config["output_root"]) / "pilot" / "pilot_manifest_v1.jsonl"
    else:
        manifest = Path(config["source_manifest"])
    rows = read_jsonl(manifest)
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    execution_contract = build_conversion_execution_contract(
        project_root=Path(__file__).resolve().parents[1],
        config_path=args.config,
        manifest_path=manifest,
        config=config,
        git_commit=git_commit,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
    )
    run_root = Path(config["output_root"]) / args.mode
    if args.mode == "full" and rank == 0:
        contract_path = run_root / "execution_contract_v1.json"
        if contract_path.exists():
            existing_contract = json.loads(contract_path.read_text())
            if existing_contract != execution_contract:
                raise RuntimeError(
                    "full-run execution contract differs from the existing resumable run"
                )
        else:
            write_atomic(contract_path, execution_contract)
    records_root = run_root / "records"
    progress_path = run_root / "progress" / f"rank_{rank:02d}.jsonl"
    failures_path = run_root / "failures" / f"rank_{rank:02d}.jsonl"
    completed_uids = set()
    rows_to_validate = rows if args.mode == "full" else balanced_worker_rows(rows, rank, world)
    for row in rows_to_validate:
        target = records_root / safe_filename(str(row["uid"]))
        if target.exists():
            if not args.resume:
                raise FileExistsError(f"record exists without --resume: {target}")
            existing = json.loads(target.read_text())
            if existing.get("uid") != row["uid"] or not existing.get("passed"):
                raise RuntimeError(f"invalid completed record: {target}")
            completed_uids.add(str(row["uid"]))

    if args.mode == "pilot":
        assigned = rows_to_validate
        pending = [row for row in assigned if str(row["uid"]) not in completed_uids]
        queue = None
        next_record = iter(pending).__next__
        launch_token = None
    else:
        launch_token = os.environ.get("SLURM_JOB_ID") or os.environ.get(
            "FOUR_ACTION_LAUNCH_TOKEN"
        )
        if not launch_token:
            raise RuntimeError(
                "full dynamic queue requires SLURM_JOB_ID or FOUR_ACTION_LAUNCH_TOKEN"
            )
        restart = os.environ.get("SLURM_RESTART_COUNT", "0")
        queue = AtomicSampleQueue(
            rows,
            claim_root=run_root / "claims" / f"{launch_token}_{restart}",
            completed_uids=completed_uids,
            claimant=f"rank-{rank:02d}",
        )
        assigned = rows
        pending = None
        next_record = queue.claim_next

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
            "pending_samples": (
                len(pending) if pending is not None else len(rows) - len(completed_uids)
            ),
            "assigned_estimated_cost": sum(
                int(row["estimated_conversion_cost"]) for row in assigned
            ),
            "work_assignment": "static_lpt" if args.mode == "pilot" else "atomic_dynamic",
            "launch_token": launch_token,
            "manifest": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "execution_contract": execution_contract,
        },
    )
    try:
        record = next_record()
    except StopIteration:
        record = None
    if record is None:
        append_jsonl(progress_path, {"event": "worker_complete", "attempt": attempt, "rank": rank})
        return 0

    processor, model = load_model(config, device)
    failures = 0
    completed = 0
    sample_index = 0
    while record is not None:
        started = time.monotonic()
        target = records_root / safe_filename(str(record["uid"]))
        append_jsonl(
            progress_path,
            {
                "event": "sample_start",
                "attempt": attempt,
                "rank": rank,
                "uid": record["uid"],
                "sample_index": sample_index,
                "pending_samples": None if pending is None else len(pending),
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
                    "elapsed_seconds": time.monotonic() - started,
                    "unique_routes_evaluated": result["route_evaluation_cache"][
                        "unique_complete_routes_evaluated"
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
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
        finally:
            torch.cuda.empty_cache()
        sample_index += 1
        try:
            record = next_record()
        except StopIteration:
            record = None
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
