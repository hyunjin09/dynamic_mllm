#!/usr/bin/env python3
"""Run the frozen smoke or one sharded full MCTS label extraction worker."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
import traceback

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from label_regeneration.data import safe_sample_filename
from label_regeneration.mcts import GraphMCTS, MCTSConfig
from label_regeneration.runtime import RouteEvaluator, configure_determinism, load_frozen_model


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def runtime_record(args, *, rank: int, world_size: int, max_new_tokens) -> dict:
    return {
        "model_source": args.model_path,
        "model_revision": args.revision,
        "contract_sha256": args.contract_sha256,
        "processor_use_fast": False,
        "native_image_processing": True,
        "custom_max_image_tokens": None,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "generation_policy": {"do_sample": False, "max_new_tokens": max_new_tokens},
        "max_simulations_per_sample": args.max_simulations_per_sample,
        "scoring_timeout_seconds": args.scoring_timeout_seconds,
        "resume_compatible_contract_sha256": sorted(
            set(args.resume_compatible_contract_sha256)
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(rank),
        "rank": rank,
        "world_size": world_size,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }


def run_smoke(args, rows, *, rank: int) -> None:
    configure_determinism(args.seed)
    processor, base, wrapped, device = load_frozen_model(args.model_path, args.revision, rank)
    results = []
    all_parity = True
    all_mixed = True
    for index, sample in enumerate(rows):
        evaluator = RouteEvaluator(
            processor=processor,
            base_model=base,
            wrapped_model=wrapped,
            sample=sample,
            device=device,
            scoring_timeout_seconds=args.scoring_timeout_seconds,
        )
        native = evaluator.native_all_on()
        binary = evaluator.evaluate((1,) * 28, "smoke_all_on")
        parity = native.generated_ids == binary["generated_ids"]
        all_parity &= parity
        mixed_results = []
        for mask_values in sample.get("mixed_masks", []):
            mask = tuple(int(value) for value in mask_values)
            first = evaluator.evaluate(mask, "smoke_mixed_first")
            second = evaluator.evaluate(mask, "smoke_mixed_repeat")
            deterministic = (
                first["generated_ids"] == second["generated_ids"]
                and first["score"] == second["score"]
            )
            all_mixed &= deterministic
            mixed_results.append(
                {
                    "mask": list(mask),
                    "first_generated_ids": first["generated_ids"],
                    "second_generated_ids": second["generated_ids"],
                    "first_score": first["score"],
                    "second_score": second["score"],
                    "passed": deterministic,
                }
            )
        results.append(
            {
                "uid": sample["uid"],
                "benchmark": sample["benchmark"],
                "geometry": evaluator.geometry,
                "input_metadata": evaluator.input_metadata,
                "native_generated_ids": native.generated_ids,
                "binary_generated_ids": binary["generated_ids"],
                "native_score": native.score,
                "binary_score": binary["score"],
                "native_scoring_timed_out": native.scoring_timed_out,
                "binary_scoring_timed_out": binary["scoring_timed_out"],
                "all_on_token_parity": parity,
                "mixed_routes": mixed_results,
            }
        )
        print(json.dumps({"smoke_completed": index + 1, "uid": sample["uid"], "parity": parity}), flush=True)
    report = {
        "schema_version": "label_regeneration_smoke_v1",
        "passed": bool(
            all_parity
            and all_mixed
            and len(results) == args.required_smoke_count
        ),
        "all_on_parity_count": sum(row["all_on_token_parity"] for row in results),
        "all_on_required": args.required_smoke_count,
        "mixed_determinism_passed": all_mixed,
        "scientific_label_extraction_started": False,
        "contract_sha256": args.contract_sha256,
        "runtime": runtime_record(
            args,
            rank=rank,
            world_size=1,
            max_new_tokens=sorted({int(row["max_new_tokens"]) for row in rows}),
        ),
        "records": results,
    }
    output = Path(args.output_root) / "smoke_report_v1.json"
    atomic_json(output, report)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    if not report["passed"]:
        raise RuntimeError("frozen label-regeneration smoke failed; full extraction blocked")


def record_complete(
    path: Path,
    *,
    uid: str,
    contract_hash: str,
    compatible_contract_hashes: tuple[str, ...] = (),
    max_simulations_per_sample: int | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    accepted_contracts = {contract_hash, *compatible_contract_hashes}
    requested = record.get("mcts", {}).get("requested_simulations")
    within_cap = (
        max_simulations_per_sample is None
        or isinstance(requested, int)
        and requested <= max_simulations_per_sample
    )
    return (
        record.get("sample", {}).get("uid") == uid
        and record.get("runtime", {}).get("contract_sha256") in accepted_contracts
        and isinstance(record.get("candidate_executions"), list)
        and record.get("mcts", {}).get("completed_simulations") == requested
        and within_cap
    )


def index_existing_records(output_root: Path) -> dict[str, tuple[Path, ...]]:
    """Index records from every prior shard layout without loading their payloads."""
    records: dict[str, list[Path]] = {}
    raw_root = output_root / "raw_route_cache"
    for shard_root in sorted(raw_root.glob("shard_*_of_*")):
        sample_root = shard_root / "samples"
        if not sample_root.is_dir():
            continue
        for path in sample_root.glob("*.json"):
            records.setdefault(path.name, []).append(path)
    return {name: tuple(sorted(paths)) for name, paths in records.items()}


def find_completed_record(
    existing_records: dict[str, tuple[Path, ...]],
    *,
    filename: str,
    uid: str,
    contract_hash: str,
    compatible_contract_hashes: tuple[str, ...] = (),
    max_simulations_per_sample: int | None = None,
) -> Path | None:
    """Return a contract-valid record even when it came from another shard count."""
    for path in existing_records.get(filename, ()):
        if record_complete(
            path,
            uid=uid,
            contract_hash=contract_hash,
            compatible_contract_hashes=compatible_contract_hashes,
            max_simulations_per_sample=max_simulations_per_sample,
        ):
            return path
    return None


def run_mcts(args, rows, *, rank: int, world_size: int) -> None:
    configure_determinism(args.seed + rank)
    cpu_threads = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "32")) // world_size)
    torch.set_num_threads(cpu_threads)
    output_root = Path(args.output_root)
    existing_records = index_existing_records(output_root)
    processor, base, wrapped, device = load_frozen_model(args.model_path, args.revision, rank)
    shard_rows = [row for index, row in enumerate(rows) if index % world_size == rank]
    shard_root = output_root / "raw_route_cache" / f"shard_{rank:03d}_of_{world_size:03d}"
    sample_root = shard_root / "samples"
    error_root = shard_root / "errors"
    completed = skipped = errors = 0
    started = time.time()
    for local_index, sample in enumerate(shard_rows):
        filename = safe_sample_filename(sample["uid"])
        path = sample_root / filename
        if find_completed_record(
            existing_records,
            filename=filename,
            uid=sample["uid"],
            contract_hash=args.contract_sha256,
            compatible_contract_hashes=tuple(args.resume_compatible_contract_sha256),
            max_simulations_per_sample=args.max_simulations_per_sample,
        ) is not None:
            skipped += 1
            continue
        try:
            evaluator = RouteEvaluator(
                processor=processor,
                base_model=base,
                wrapped_model=wrapped,
                sample=sample,
                device=device,
                scoring_timeout_seconds=args.scoring_timeout_seconds,
            )
            sample_seed = args.seed + int(sha256(sample["uid"].encode("utf-8")).hexdigest()[:8], 16)
            config = MCTSConfig(seed=sample_seed)
            search = GraphMCTS(evaluator.evaluate, config)
            root, all_off = search.evaluate_anchors()
            root_correct = bool(root["result_correct"])
            requested = 200 if root_correct else 400
            if requested > args.max_simulations_per_sample:
                raise RuntimeError(
                    f"base search budget {requested} exceeds frozen per-sample cap "
                    f"{args.max_simulations_per_sample}"
                )
            search.run(requested)
            extension_reason = None
            if (
                requested < args.max_simulations_per_sample
                and not root_correct
                and not any(row["result_correct"] for row in search.evaluations.values())
            ):
                extension_reason = "no_correcting_route_after_400"
                requested = min(600, args.max_simulations_per_sample)
                search.run(requested)
            mcts = search.result(requested_simulations=requested, extension_reason=extension_reason)
            route_by_mask = {
                tuple(row["visual_on_mask"]): row["route_id"] for row in evaluator.results
            }
            successful_ids = [route_by_mask[tuple(mask)] for mask in mcts["successful_masks"]]
            best_id = None if mcts["best_mask"] is None else route_by_mask[tuple(mcts["best_mask"])]
            record = {
                "phase": "binary_visual_mask_graph_mcts_regenerated_v1",
                "dataset_version": "8k_native_qwen_unrestricted_mask_regeneration_v1",
                "root_policy": "all_visual_on_recomputed_current_executor",
                "reward_policy": {
                    "name": "thresholded_task_correctness",
                    "binary": True,
                    "raw_score_used_by_ucb": False,
                    "metric_name": sample["metric_name"],
                    "correctness_threshold": sample["correctness_threshold"],
                },
                "mcts_config": {
                    "num_simulations": requested,
                    "base_num_simulations": 200 if root_correct else 400,
                    "max_simulations_per_sample": args.max_simulations_per_sample,
                    "exploration_constant": config.exploration_constant,
                    "length_penalty": config.length_penalty,
                    "random_prob": config.random_probability,
                    "rollout_off_probability": config.rollout_off_probability,
                    "seed": sample_seed,
                    "fixed_layer_permutation": False,
                    "transposition_table": True,
                    "stop_on_first_success": False,
                    "expansion_policy": "choose_layer_and_visual_on_off_from_all_undecided_layers",
                },
                "runtime": runtime_record(
                    args,
                    rank=rank,
                    world_size=world_size,
                    max_new_tokens=int(sample["max_new_tokens"]),
                ),
                "sample": {
                    **sample,
                    "current_all_on_prediction": root["prediction"],
                    "current_all_on_score": root["score"],
                    "current_all_on_status": "correct" if root_correct else "wrong",
                    "actual_text_tokens": root["text_tokens"],
                    "actual_visual_tokens": root["visual_tokens"],
                    "actual_full_prompt_tokens": root["full_prompt_tokens"],
                    "input_metadata": evaluator.input_metadata,
                    "scoring_timeout_count": evaluator.scoring_timeout_count,
                },
                "root_route_id": root["route_id"],
                "all_off_route_id": all_off["route_id"],
                "successful_route_ids": successful_ids,
                "best_sparse_success_route_id": best_id,
                "mcts": mcts,
                "candidate_executions": evaluator.results,
            }
            atomic_json(path, record)
            completed += 1
            error_path = error_root / safe_sample_filename(sample["uid"])
            if error_path.exists():
                error_path.unlink()
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            errors += 1
            atomic_json(
                error_root / safe_sample_filename(sample["uid"]),
                {
                    "uid": sample["uid"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "contract_sha256": args.contract_sha256,
                },
            )
        summary = {
            "schema_version": "label_regeneration_shard_summary_v1",
            "rank": rank,
            "world_size": world_size,
            "selected_samples": len(shard_rows),
            "completed_this_run": completed,
            "skipped_existing": skipped,
            "errors_this_run": errors,
            "elapsed_seconds": time.time() - started,
            "last_uid": sample["uid"],
            "contract_sha256": args.contract_sha256,
        }
        atomic_json(shard_root / "summary.json", summary)
        print(
            json.dumps(
                {
                    "rank": rank,
                    "completed": completed,
                    "skipped": skipped,
                    "errors": errors,
                    "shard_total": len(shard_rows),
                    "uid": sample["uid"],
                }
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "mcts"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument(
        "--resume-compatible-contract-sha256",
        action="append",
        default=[],
        help="Prior contract whose complete records may be retained during an audited repair.",
    )
    parser.add_argument("--scoring-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--max-simulations-per-sample", type=int, default=600)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--required-smoke-count", type=int, default=15)
    args = parser.parse_args()
    if args.max_simulations_per_sample not in {400, 600}:
        raise ValueError("max_simulations_per_sample must be 400 or 600")
    if not torch.cuda.is_available():
        raise RuntimeError("label regeneration must run in a scheduled GPU allocation")
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    rows = read_jsonl(Path(args.manifest))
    if args.mode == "smoke":
        if world_size != 1:
            raise ValueError("smoke must use exactly one worker")
        run_smoke(args, rows, rank=rank)
    else:
        run_mcts(args, rows, rank=rank, world_size=world_size)


if __name__ == "__main__":
    main()
