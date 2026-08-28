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
import yaml
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from binary_policy.executor import BinaryQwen25VL
from label_regeneration.runtime import configure_determinism
from tools.research_analysis.four_action.label_jobs import (
    AtomicSampleQueue,
    balanced_worker_rows,
    safe_filename,
)
from tools.research_analysis.four_action.label_runtime import FourActionSampleRuntime
from tools.research_analysis.four_action.parallelism import worker_layout
from tools.research_analysis.four_action.three_action_jobs import (
    build_three_action_execution_contract,
    file_sha256,
)
from tools.research_analysis.four_action.three_action_labels import (
    CachedThreeActionEvaluator,
    binary_to_three_action,
    deduplicate_positive_routes,
    decompose_screened_positions,
    evaluate_independent_composition,
    refine_three_action_route,
    screen_binary_off_positions,
    select_canonical_c2c_route,
    select_canonical_w2c_route,
    three_action_to_executor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert binary routes into answer-aligned three-suppression labels."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/three_action_label_conversion.yaml")
    )
    parser.add_argument("--mode", choices=("calibrate", "pilot", "full"), required=True)
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
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{file_sha256(path)}  {path.name}\n", encoding="utf-8"
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


def _runtime_metadata(device, rank: int, started: float) -> dict[str, Any]:
    return {
        "rank": rank,
        "gpu_index": device.index,
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


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


def _route_row(row) -> dict[str, Any]:
    return asdict(row)


def _canonical_for_source(
    positive_rows: list[dict[str, Any]],
    *,
    route_type: str,
    source_evaluation: dict[str, Any],
    epsilon: float,
) -> dict[str, Any] | None:
    if not positive_rows:
        return None
    if route_type == "W2C":
        return select_canonical_w2c_route(
            positive_rows,
            best_seed_margin=float(source_evaluation["answer_alignment_margin"]),
            epsilon=epsilon,
        )
    return select_canonical_c2c_route(positive_rows)


def _beam_stability(
    narrow,
    wide,
    *,
    route_type: str,
    source_evaluation: dict[str, Any],
    epsilon: float,
) -> dict[str, Any]:
    narrow_rows = [_route_row(row) for row in narrow.positive_routes]
    wide_rows = [_route_row(row) for row in wide.positive_routes]
    narrow_canonical = _canonical_for_source(
        narrow_rows,
        route_type=route_type,
        source_evaluation=source_evaluation,
        epsilon=epsilon,
    )
    wide_canonical = _canonical_for_source(
        wide_rows,
        route_type=route_type,
        source_evaluation=source_evaluation,
        epsilon=epsilon,
    )
    narrow_keys = {"|".join(row["route"]) for row in narrow_rows}
    wide_keys = {"|".join(row["route"]) for row in wide_rows}
    union = narrow_keys | wide_keys
    return {
        "narrow_width": narrow.beam_width,
        "wide_width": wide.beam_width,
        "narrow_positive_count": len(narrow_rows),
        "wide_positive_count": len(wide_rows),
        "positive_route_jaccard": len(narrow_keys & wide_keys) / len(union) if union else 1.0,
        "canonical_route_match": (
            None if narrow_canonical is None or wide_canonical is None
            else narrow_canonical["route"] == wide_canonical["route"]
        ),
        "both_have_no_positive_route": narrow_canonical is None and wide_canonical is None,
        "narrow_canonical": narrow_canonical,
        "wide_canonical": wide_canonical,
    }


def calibrate_sample(
    *,
    processor,
    model,
    record: dict[str, Any],
    device: torch.device,
    config: dict[str, Any],
    rank: int,
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
    route_type = "C2C" if bool(full["correct"]) else "W2C"
    first_source = record["source_positive_routes"][0]
    semantic_source = binary_to_three_action(first_source["mask"])
    controls = []
    repetitions = int(config["noise_calibration"]["repetitions"])
    for name, executor_route in (
        ("unified_full", ("FULL",) * int(config["layer_count"])),
        ("source_both_off_route", three_action_to_executor(semantic_source)),
    ):
        evaluations = [runtime.evaluate_uncached(executor_route) for _ in range(repetitions)]
        quantity = "answer_alignment_margin" if route_type == "W2C" else "S_correct"
        reference = float(evaluations[0][quantity])
        controls.append(
            {
                "control": name,
                "executor_route": list(executor_route),
                "score_quantity": quantity,
                "evaluations": evaluations,
                "signed_differences_from_first": [
                    float(row[quantity]) - reference for row in evaluations[1:]
                ],
                "generated_ids_identical": len({tuple(row["generated_ids"]) for row in evaluations}) == 1,
                "correctness_identical": len({bool(row["correct"]) for row in evaluations}) == 1,
            }
        )
    return {
        "schema_version": "three_action_repeatability_sample_v1",
        "passed": all(
            row["generated_ids_identical"] and row["correctness_identical"] for row in controls
        ),
        "uid": record["uid"],
        "dataset": record["dataset"],
        "sample_id": record["sample_id"],
        "image_id": record.get("image_id"),
        "image_group_id": record.get("image_group_id"),
        "source_split": record["source_split"],
        "route_type": route_type,
        "execution_contract": execution_contract,
        "current_unified_full": full,
        "source_binary_route_id": first_source["source_binary_route_id"],
        "repeatability_controls": controls,
        "input_metadata": runtime.input_metadata,
        "geometry": runtime.geometry,
        "runtime": _runtime_metadata(device, rank, started),
    }


def process_sample(
    *,
    processor,
    model,
    record: dict[str, Any],
    device: torch.device,
    config: dict[str, Any],
    epsilon: float,
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
    route_type = "C2C" if bool(full["correct"]) else "W2C"
    cached = CachedThreeActionEvaluator(runtime.evaluate)
    semantic_full = ("FULL",) * int(config["layer_count"])
    cached(semantic_full)
    all_off = cached(("BOTH_OFF",) * int(config["layer_count"]))
    conversions = []
    binary_checks = []
    for source_index, source in enumerate(record["source_positive_routes"]):
        source_route = binary_to_three_action(source["mask"])
        source_evaluation = cached(source_route)
        base = {
            "schema_version": "three_action_raw_conversion_v1",
            "source_binary_route_id": source["source_binary_route_id"],
            "source_route_id": source["route_id"],
            "source_binary_route": source["mask"],
            "source_three_action_seed": list(source_route),
            "source_off_count": source["source_off_count"],
            "all_off_seed": source["source_all_off"],
            "source_route_evaluation": source_evaluation,
            "route_type": route_type,
        }
        if mode == "pilot" and source_index == 0:
            old = runtime.evaluate_old_binary(tuple(int(value) for value in source["mask"]))
            binary_checks.append(
                {
                    "source_binary_route_id": source["source_binary_route_id"],
                    "generated_ids_match": old["generated_ids"] == source_evaluation["generated_ids"],
                    "generated_answer_match": old["generated_answer"] == source_evaluation["generated_answer"],
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
        before = cached.cache_misses
        screening = screen_binary_off_positions(
            source_route,
            route_type=route_type,
            evaluate=cached,
            epsilon=epsilon,
        )
        before_decomposition = cached.cache_misses
        decomposition = decompose_screened_positions(
            screening,
            evaluate=cached,
            epsilon=epsilon,
        )
        decomposition_misses = cached.cache_misses - before_decomposition
        independent_composition = evaluate_independent_composition(
            screening,
            decomposition,
            evaluate=cached,
            epsilon=epsilon,
            unified_full_evaluation=full,
        )
        refinement = refine_three_action_route(
            screening.route,
            candidate_layers=screening.candidate_layers,
            route_type=route_type,
            evaluate=cached,
            epsilon=epsilon,
            beam_width=int(config["beam_width"]),
            unified_full_evaluation=full,
        )
        positive_rows = [_route_row(row) for row in refinement.positive_routes]
        label_semantics = (
            "C2C_COMPENSATED_ALIGNMENT"
            if route_type == "C2C"
            else (
                "W2C_HARD_CORRECTIVE"
                if any(row.classification == "HARD_NECESSARY" for row in screening.positions)
                else "W2C_SOFT_ALIGNMENT"
            )
        )
        wide = None
        stability = None
        if mode == "pilot":
            wide = refine_three_action_route(
                screening.route,
                candidate_layers=screening.candidate_layers,
                route_type=route_type,
                evaluate=cached,
                epsilon=epsilon,
                beam_width=int(config["beam_validation_width"]),
                unified_full_evaluation=full,
            )
            stability = _beam_stability(
                refinement,
                wide,
                route_type=route_type,
                source_evaluation=source_evaluation,
                epsilon=epsilon,
            )
        conversions.append(
            {
                **base,
                "status": "converted",
                "label_semantics": label_semantics,
                "screening": asdict(screening),
                "decomposition": [asdict(row) for row in decomposition],
                "independent_composition": independent_composition,
                "refinement": asdict(refinement),
                "positive_routes": positive_rows,
                "pareto_routes": [_route_row(row) for row in refinement.pareto_routes],
                "max_margin_route": (
                    None if refinement.max_margin_route is None
                    else _route_row(refinement.max_margin_route)
                ),
                "corrective_partial_candidates": [
                    _route_row(row) for row in refinement.corrective_partial_candidates
                ],
                "pilot_beam_stability": stability,
                "execution_efficiency": {
                    "candidate_positions": len(screening.candidate_layers),
                    "theoretical_four_state_evaluations": 4 * len(screening.candidate_layers),
                    "decomposition_new_cache_misses": decomposition_misses,
                    "theoretical_four_state_evaluations_avoided": (
                        4 * len(screening.candidate_layers) - decomposition_misses
                    ),
                    "new_cache_misses_for_conversion": cached.cache_misses - before,
                },
            }
        )

    unique = deduplicate_positive_routes(conversions)
    canonical = None
    if unique:
        if route_type == "W2C":
            best_seed_margin = max(
                float(row["source_route_evaluation"]["answer_alignment_margin"])
                for row in conversions if row["status"] == "converted"
            )
            canonical = select_canonical_w2c_route(
                unique, best_seed_margin=best_seed_margin, epsilon=epsilon
            )
        else:
            canonical = select_canonical_c2c_route(unique)
    return {
        "schema_version": "three_action_answer_aligned_sample_v1",
        "passed": True,
        "uid": record["uid"],
        "dataset": record["dataset"],
        "sample_id": record["sample_id"],
        "image_id": record.get("image_id"),
        "image_group_id": record.get("image_group_id"),
        "source_split": record["source_split"],
        "route_type": route_type,
        "epsilon": epsilon,
        "execution_contract": execution_contract,
        "current_unified_full": full,
        "current_unified_all_off": all_off,
        "source_full_diagnostic": _source_diagnostic(record, full),
        "source_positive_route_count": int(record["source_positive_route_count"]),
        "source_route_replay_valid_count": sum(row["status"] == "converted" for row in conversions),
        "source_route_replay_failure_count": sum(row["status"] == "source_route_replay_failure" for row in conversions),
        "raw_conversions": conversions,
        "unique_valid_three_action_routes": unique,
        "canonical_three_action_route": canonical,
        "pilot_old_binary_semantic_checks": binary_checks,
        "route_evaluation_cache": {
            "unique_complete_routes_evaluated": cached.cache_misses,
            "cache_hits": cached.cache_hits,
            "cache_misses": cached.cache_misses,
            "source_routes": int(record["source_positive_route_count"]),
        },
        "input_metadata": runtime.input_metadata,
        "geometry": runtime.geometry,
        "runtime": _runtime_metadata(device, rank, started),
    }


def _manifest_for_mode(config: dict[str, Any], mode: str) -> Path:
    if mode in {"calibrate", "pilot"}:
        return Path(config["output_root"]) / "pilot" / "pilot_manifest_v1.jsonl"
    return Path(config["source_manifest"])


def _run_root(config: dict[str, Any], mode: str) -> Path:
    return Path(config["output_root"]) / ("calibration" if mode == "calibrate" else mode)


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rank, world = rank_and_world(args)
    if world != int(config["worker_count"]):
        raise RuntimeError(f"expected {config['worker_count']} workers, got {world}")
    layout = worker_layout(rank, world, gpu_count=int(config["gpu_count"]))
    if layout.replicas_per_gpu != int(config["workers_per_gpu"]):
        raise RuntimeError("worker/GPU replica layout differs from the frozen configuration")
    device = torch.device(f"cuda:{layout.gpu_index}")
    torch.cuda.set_device(device)
    configure_determinism(int(config["seed"]) + rank)

    manifest = _manifest_for_mode(config, args.mode)
    rows = read_jsonl(manifest)
    epsilon_path = None if args.mode == "calibrate" else Path(config["noise_calibration"]["artifact_path"])
    epsilon = None
    if epsilon_path is not None:
        epsilon = float(json.loads(epsilon_path.read_text(encoding="utf-8"))["epsilon"])
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    contract = build_three_action_execution_contract(
        project_root=Path(__file__).resolve().parents[1],
        config_path=args.config,
        manifest_path=manifest,
        epsilon_path=epsilon_path,
        config=config,
        git_commit=git_commit,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        mode=args.mode,
    )
    run_root = _run_root(config, args.mode)
    if rank == 0:
        contract_path = run_root / "execution_contract_v1.json"
        if contract_path.exists():
            if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
                raise RuntimeError("execution contract differs from the existing resumable run")
        else:
            write_atomic(contract_path, contract)

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
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing.get("uid") != row["uid"] or not existing.get("passed"):
                raise RuntimeError(f"invalid completed record: {target}")
            completed_uids.add(str(row["uid"]))

    if args.mode == "full":
        launch_token = os.environ.get("SLURM_JOB_ID") or os.environ.get("THREE_ACTION_LAUNCH_TOKEN")
        if not launch_token:
            raise RuntimeError("full dynamic queue requires a launch token")
        queue = AtomicSampleQueue(
            rows,
            claim_root=run_root / "claims" / f"{launch_token}_{os.environ.get('SLURM_RESTART_COUNT', '0')}",
            completed_uids=completed_uids,
            claimant=f"rank-{rank:02d}",
        )
        next_record = queue.claim_next
        assigned = rows
        pending_count = len(rows) - len(completed_uids)
        assignment = "atomic_dynamic"
    else:
        assigned = rows_to_validate
        pending = [row for row in assigned if str(row["uid"]) not in completed_uids]
        iterator = iter(pending)
        next_record = lambda: next(iterator, None)
        pending_count = len(pending)
        assignment = "static_lpt"
        launch_token = os.environ.get("SLURM_JOB_ID")

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
            "pending_samples": pending_count,
            "work_assignment": assignment,
            "launch_token": launch_token,
            "manifest": str(manifest.resolve()),
            "manifest_sha256": file_sha256(manifest),
            "execution_contract": contract,
        },
    )
    record = next_record()
    if record is None:
        append_jsonl(progress_path, {"event": "worker_complete", "attempt": attempt, "rank": rank})
        return 0
    processor, model = load_model(config, device)
    failures = 0
    completed = 0
    sample_index = 0
    while record is not None:
        sample_started = time.monotonic()
        target = records_root / safe_filename(str(record["uid"]))
        append_jsonl(progress_path, {"event": "sample_start", "attempt": attempt, "rank": rank, "uid": record["uid"], "sample_index": sample_index})
        try:
            if args.mode == "calibrate":
                result = calibrate_sample(
                    processor=processor,
                    model=model,
                    record=record,
                    device=device,
                    config=config,
                    rank=rank,
                    execution_contract=contract,
                )
            else:
                assert epsilon is not None
                result = process_sample(
                    processor=processor,
                    model=model,
                    record=record,
                    device=device,
                    config=config,
                    epsilon=epsilon,
                    rank=rank,
                    mode=args.mode,
                    execution_contract=contract,
                )
            if not bool(result.get("passed")):
                raise RuntimeError("sample-level semantic gate failed")
            write_atomic(target, result)
            append_jsonl(progress_path, {"event": "sample_complete", "attempt": attempt, "rank": rank, "uid": record["uid"], "elapsed_seconds": time.monotonic() - sample_started})
            completed += 1
        except Exception as exc:
            failures += 1
            append_jsonl(failures_path, {"event": "sample_failure", "attempt": attempt, "rank": rank, "uid": record["uid"], "exception_type": type(exc).__name__, "exception": str(exc), "traceback": traceback.format_exc()})
            append_jsonl(progress_path, {"event": "sample_failed", "attempt": attempt, "rank": rank, "uid": record["uid"], "elapsed_seconds": time.monotonic() - sample_started})
        finally:
            torch.cuda.empty_cache()
        sample_index += 1
        record = next_record()
    append_jsonl(progress_path, {"event": "worker_complete", "attempt": attempt, "rank": rank, "failures": failures, "completed": completed})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
