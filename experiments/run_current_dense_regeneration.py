#!/usr/bin/env python3
"""Fail-closed native-dense smoke and resumable four-GPU regeneration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from dense_failure_stage1.contract import (
    assert_contract_integrity,
    collect_contract_integrity,
    file_sha256,
    validate_smoke_gate,
)
from dense_failure_stage1.runtime import (
    configure_dense_determinism,
    generate_dense,
    load_dense_runtime,
)
from tools.research_analysis.dense_failure_stage1 import (
    canonical_payload_sha256,
    validate_candidate_execution_contract,
)


EXPECTED_FULL_WORLD_SIZE = 4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_json_atomic(path: Path, value) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    write_text_atomic(
        path,
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
    )


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    actual = canonical_payload_sha256(contract)
    if actual != contract.get("contract_sha256"):
        raise ValueError(
            f"contract SHA-256 is invalid: recorded={contract.get('contract_sha256')} actual={actual}"
        )
    return contract


def validate_feature_provenance(payload: Mapping, expected: Mapping) -> None:
    actual = payload.get("provenance")
    if actual != dict(expected):
        raise ValueError(
            f"feature provenance differs: actual={actual!r} expected={dict(expected)!r}"
        )


def validate_resume_marker(
    marker: Mapping,
    *,
    contract_id: str,
    run_id: str,
    rank: int,
    world_size: int,
    feature_shard_size: int,
    expected_uids: Sequence[str],
    require_features: bool,
) -> None:
    expected_state = (
        "completed_with_features" if require_features else "completed_without_features"
    )
    fixed = {
        "schema_version": "current_dense_batch_marker_v2",
        "contract_id": contract_id,
        "run_id": run_id,
        "rank": rank,
        "world_size": world_size,
        "feature_shard_size": feature_shard_size,
        "completion_state": expected_state,
        "extract_features": require_features,
        "uids": list(expected_uids),
    }
    mismatches = {
        key: (marker.get(key), value)
        for key, value in fixed.items()
        if marker.get(key) != value
    }
    if "completion_state" in mismatches or "extract_features" in mismatches:
        raise ValueError(f"resume feature completion mode differs: {mismatches}")
    if mismatches:
        raise ValueError(f"resume marker contract differs: {mismatches}")


def feature_provenance(
    contract: Mapping,
    *,
    run_id: str,
    generation_sha256: str,
) -> dict:
    return {
        "contract_id": contract["contract_sha256"],
        "model_revision": contract["integrity"]["model"]["revision"],
        "git_commit": contract["integrity"]["git"]["commit"],
        "feature_schema_sha256": contract["integrity"]["feature_schema_sha256"],
        "source_manifest_sha256": contract["integrity"]["candidate_manifest_sha256"],
        "run_id": run_id,
        "generation_sha256": generation_sha256,
    }


def output_row(
    sample: dict,
    result,
    contract: dict,
    *,
    worker_rank: int,
    elapsed: float,
) -> dict:
    return {
        "schema_version": "current_dense_output_v1",
        "uid": sample["uid"],
        "sample_id": sample["sample_id"],
        "dataset": sample["dataset"],
        "image_group_id": sample["image_group_id"],
        "image_content_sha256": sample["image_content_sha256"],
        "consumed_image_sha256": result.input_metadata["consumed_image_sha256"],
        "historical_bucket": sample["historical_bucket"],
        "dense_generated_token_ids": result.generated_ids,
        "dense_generated_text": result.generated_text,
        "normalized_prediction": result.normalized_prediction,
        "gt_answer": sample["answer"],
        "all_answer_norms": sample.get("all_answer_norms"),
        "evaluator": sample["metric_name"],
        "evaluator_score": result.score,
        "correctness_threshold": sample["correctness_threshold"],
        "current_dense_correct": result.correct,
        "current_dense_wrong": not result.correct,
        "generation_length": len(result.generated_ids),
        "prompt_token_count": result.prompt_token_count,
        "visual_token_count": result.visual_token_count,
        "user_text_token_count": result.user_text_token_count,
        "literal_prompt_sha256": result.input_metadata["literal_prompt_sha256"],
        "original_image_dimensions": result.input_metadata["original_image_dimensions"],
        "contract_id": contract["contract_sha256"],
        "model_revision": contract["integrity"]["model"]["revision"],
        "source_manifest_sha256": contract["integrity"]["candidate_manifest_sha256"],
        "worker_rank": worker_rank,
        "elapsed_seconds": elapsed,
    }


def run_samples(
    processor,
    model,
    device,
    samples,
    contract,
    *,
    extract_features: bool,
    rank: int,
):
    rows = []
    features = []
    for sample in samples:
        started = time.monotonic()
        result = generate_dense(
            processor,
            model,
            device,
            sample,
            extract_features=extract_features,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        rows.append(
            output_row(sample, result, contract, worker_rank=rank, elapsed=elapsed)
        )
        if result.features is not None:
            features.append(result.features)
        print(
            json.dumps(
                {
                    "uid": sample["uid"],
                    "correct": result.correct,
                    "tokens": len(result.generated_ids),
                    "seconds": round(elapsed, 3),
                    "features": result.features is not None,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return rows, features


def save_feature_shard(
    path: Path,
    samples: list[dict],
    features: list[dict],
    *,
    provenance: Mapping,
) -> None:
    if len(samples) != len(features):
        raise ValueError("feature count differs from sample count")
    payload = {
        "schema_version": "current_dense_features_v1",
        "provenance": dict(provenance),
        "uids": [sample["uid"] for sample in samples],
        "layer_ids": list(range(28)),
        "text_final": torch.stack([row["text_final"] for row in features]).to(
            torch.bfloat16
        ),
        "text_mean": torch.stack([row["text_mean"] for row in features]).to(
            torch.bfloat16
        ),
        "visual_mean": torch.stack([row["visual_mean"] for row in features]).to(
            torch.bfloat16
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def smoke_records(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    records = value["records"] if isinstance(value, dict) else value
    if len(records) != 24:
        raise ValueError(
            f"dense smoke requires exactly 24 records, found {len(records)}"
        )
    if len({row["uid"] for row in records}) != 24:
        raise ValueError("dense smoke manifest contains duplicate UIDs")
    return records


def run_smoke_suite(args, contract, processor, model, device) -> None:
    samples = smoke_records(Path(args.smoke_manifest))
    rank = int(os.environ.get("LOCAL_RANK", args.rank))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    if world_size != EXPECTED_FULL_WORLD_SIZE or not 0 <= rank < world_size:
        raise ValueError("smoke suite requires exactly ranks 0,1,2,3")
    samples = [sample for index, sample in enumerate(samples) if index % world_size == rank]
    smoke_root = Path(args.output_root) / "smoke" / "shards"
    stem = f"rank{rank:03d}"
    run1, _ = run_samples(
        processor, model, device, samples, contract, extract_features=False, rank=rank
    )
    run1_path = smoke_root / f"{stem}_run1.jsonl"
    write_jsonl_atomic(run1_path, run1)
    run2, _ = run_samples(
        processor, model, device, samples, contract, extract_features=False, rank=rank
    )
    run2_path = smoke_root / f"{stem}_run2.jsonl"
    write_jsonl_atomic(run2_path, run2)
    hooked, features = run_samples(
        processor, model, device, samples, contract, extract_features=True, rank=rank
    )
    hook_path = smoke_root / f"{stem}_hook.jsonl"
    write_jsonl_atomic(hook_path, hooked)
    generation_hash = file_sha256(hook_path)
    provenance = feature_provenance(
        contract, run_id=args.run_id, generation_sha256=generation_hash
    )
    feature_path = smoke_root / f"{stem}_features.pt"
    save_feature_shard(feature_path, samples, features, provenance=provenance)
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    validate_feature_provenance(payload, provenance)

    resume_marker = {
        "schema_version": "current_dense_batch_marker_v2",
        "contract_id": contract["contract_sha256"],
        "run_id": args.run_id,
        "rank": rank,
        "world_size": world_size,
        "feature_shard_size": len(samples),
        "completion_state": "completed_with_features",
        "extract_features": True,
        "uids": [row["uid"] for row in hooked],
    }
    validate_resume_marker(
        resume_marker,
        contract_id=contract["contract_sha256"],
        run_id=args.run_id,
        rank=rank,
        world_size=world_size,
        feature_shard_size=len(samples),
        expected_uids=[row["uid"] for row in hooked],
        require_features=True,
    )
    mismatch_rejected = False
    try:
        validate_resume_marker(
            resume_marker,
            contract_id=contract["contract_sha256"],
            run_id=args.run_id,
            rank=rank,
            world_size=world_size,
            feature_shard_size=len(samples),
            expected_uids=[row["uid"] for row in hooked],
            require_features=False,
        )
    except ValueError:
        mismatch_rejected = True
    if not mismatch_rejected:
        raise RuntimeError("smoke resume probe accepted an incompatible feature mode")
    write_json_atomic(
        smoke_root / f"{stem}.complete.json",
        {
            "schema_version": "current_dense_smoke_rank_marker_v1",
            "contract_id": contract["contract_sha256"],
            "run_id": args.run_id,
            "rank": rank,
            "world_size": world_size,
            "records": len(samples),
            "uids": [sample["uid"] for sample in samples],
            "run1_path": str(run1_path.resolve()),
            "run1_sha256": file_sha256(run1_path),
            "run2_path": str(run2_path.resolve()),
            "run2_sha256": file_sha256(run2_path),
            "hook_path": str(hook_path.resolve()),
            "hook_sha256": file_sha256(hook_path),
            "feature_path": str(feature_path.resolve()),
            "feature_sha256": file_sha256(feature_path),
            "feature_provenance": provenance,
            "exact_feature_resume_accepted": True,
            "incompatible_mode_rejected": True,
            "completion_state": "completed_with_features",
        },
    )
    print(
        json.dumps(
            {
                "rank": rank,
                "local_completed_not_global_success": len(samples),
                "run1": str(run1_path),
                "run2": str(run2_path),
                "hook_run": str(hook_path),
                "feature_shard": str(feature_path),
            },
            sort_keys=True,
        )
    )


def completed_batch(
    marker_path: Path,
    *,
    contract: Mapping,
    run_id: str,
    rank: int,
    world_size: int,
    feature_shard_size: int,
    expected_uids: Sequence[str],
    require_features: bool,
) -> list[dict] | None:
    if not marker_path.is_file():
        return None
    state = json.loads(marker_path.read_text(encoding="utf-8"))
    validate_resume_marker(
        state,
        contract_id=contract["contract_sha256"],
        run_id=run_id,
        rank=rank,
        world_size=world_size,
        feature_shard_size=feature_shard_size,
        expected_uids=expected_uids,
        require_features=require_features,
    )
    generation_path = marker_path.parent / state["generation_file"]
    if (
        not generation_path.is_file()
        or file_sha256(generation_path) != state["generation_sha256"]
    ):
        raise ValueError(
            f"completed generation batch failed hash validation: {marker_path}"
        )
    rows = read_jsonl(generation_path)
    if [row["uid"] for row in rows] != list(expected_uids):
        raise ValueError(
            f"completed generation batch UID sequence differs: {marker_path}"
        )
    if require_features:
        feature_path = Path(state["feature_path"])
        if (
            not feature_path.is_file()
            or file_sha256(feature_path) != state["feature_sha256"]
        ):
            raise ValueError(
                f"completed feature batch failed hash validation: {marker_path}"
            )
        payload = torch.load(feature_path, map_location="cpu", weights_only=False)
        validate_feature_provenance(
            payload,
            feature_provenance(
                contract,
                run_id=run_id,
                generation_sha256=state["generation_sha256"],
            ),
        )
        if payload.get("uids") != list(expected_uids):
            raise ValueError(
                f"completed feature batch UID sequence differs: {marker_path}"
            )
    return rows


def run_full(args, contract, processor, model, device) -> None:
    rank = int(os.environ.get("LOCAL_RANK", args.rank))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    if world_size != EXPECTED_FULL_WORLD_SIZE:
        raise ValueError(
            f"full dense regeneration requires world_size={EXPECTED_FULL_WORLD_SIZE}, got {world_size}"
        )
    if not 0 <= rank < world_size:
        raise ValueError(
            f"worker rank is outside the frozen world: rank={rank} world={world_size}"
        )
    all_samples = read_jsonl(Path(args.candidate_manifest))
    assigned = [
        sample for index, sample in enumerate(all_samples) if index % world_size == rank
    ]
    output_root = Path(args.output_root)
    batch_root = output_root / "generation" / "batches"
    feature_root = output_root / "features"
    history_path = output_root / "generation" / f"history_rank{rank:03d}.jsonl"
    history = {
        "event": "worker_start",
        "timestamp": utc_now(),
        "rank": rank,
        "world_size": world_size,
        "assigned": len(assigned),
        "extract_features": args.extract_features,
        "contract_id": contract["contract_sha256"],
        "run_id": args.run_id,
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(history, sort_keys=True) + "\n")

    completed = 0
    for batch_index, start in enumerate(
        range(0, len(assigned), args.feature_shard_size)
    ):
        samples = assigned[start : start + args.feature_shard_size]
        expected_uids = [sample["uid"] for sample in samples]
        stem = f"rank{rank:03d}_batch{batch_index:05d}"
        marker_path = batch_root / f"{stem}.complete.json"
        existing = completed_batch(
            marker_path,
            contract=contract,
            run_id=args.run_id,
            rank=rank,
            world_size=world_size,
            feature_shard_size=args.feature_shard_size,
            expected_uids=expected_uids,
            require_features=args.extract_features,
        )
        if existing is not None:
            completed += len(existing)
            print(
                json.dumps({"resume_batch": stem, "records": len(existing)}),
                flush=True,
            )
            continue

        rows, features = run_samples(
            processor,
            model,
            device,
            samples,
            contract,
            extract_features=args.extract_features,
            rank=rank,
        )
        generation_path = batch_root / f"{stem}.jsonl"
        write_jsonl_atomic(generation_path, rows)
        generation_hash = file_sha256(generation_path)
        state = {
            "schema_version": "current_dense_batch_marker_v2",
            "contract_id": contract["contract_sha256"],
            "run_id": args.run_id,
            "rank": rank,
            "world_size": world_size,
            "feature_shard_size": args.feature_shard_size,
            "batch_index": batch_index,
            "records": len(rows),
            "uids": [row["uid"] for row in rows],
            "generation_file": generation_path.name,
            "generation_sha256": generation_hash,
            "extract_features": args.extract_features,
            "completion_state": (
                "completed_with_features"
                if args.extract_features
                else "completed_without_features"
            ),
            "completed_at": utc_now(),
        }
        if args.extract_features:
            feature_path = feature_root / f"shard_{stem}.pt"
            save_feature_shard(
                feature_path,
                samples,
                features,
                provenance=feature_provenance(
                    contract,
                    run_id=args.run_id,
                    generation_sha256=generation_hash,
                ),
            )
            state.update(
                {
                    "feature_path": str(feature_path.resolve()),
                    "feature_sha256": file_sha256(feature_path),
                }
            )
        write_json_atomic(marker_path, state)
        completed += len(rows)
        print(
            json.dumps(
                {
                    "completed_batch": stem,
                    "records": len(rows),
                    "worker_completed": completed,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "worker_local_complete_not_global_success",
                    "timestamp": utc_now(),
                    "rank": rank,
                    "completed": completed,
                    "contract_id": contract["contract_sha256"],
                    "run_id": args.run_id,
                },
                sort_keys=True,
            )
            + "\n"
        )
    if completed != len(assigned):
        raise RuntimeError(f"worker completion differs: {completed} != {len(assigned)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke-suite", "full"), required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--smoke-manifest", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--smoke-gate")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--extract-features", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--feature-shard-size", type=int, default=64)
    args = parser.parse_args()
    if args.mode == "smoke-suite" and args.extract_features:
        parser.error("smoke-suite controls its own hook-free/hooked comparisons")
    if args.mode == "full" and not args.smoke_gate:
        parser.error("--smoke-gate is required for full regeneration")
    return args


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parents[1]
    contract = load_contract(Path(args.contract))
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.revision != config["model_revision"]:
        raise ValueError("CLI revision differs from frozen configuration")
    configure_dense_determinism(
        int(config["execution_seed"]), dict(config["backend_settings"])
    )
    actual_integrity = collect_contract_integrity(
        project=project,
        model_path=Path(args.model_path),
        revision=args.revision,
        processor_revision=config["processor_revision"],
        candidate_manifest=Path(args.candidate_manifest),
        smoke_manifest=Path(args.smoke_manifest),
        config_path=config_path,
        config=config,
    )
    assert_contract_integrity(contract, actual_integrity)

    if args.mode == "smoke-suite":
        execution_rows = smoke_records(Path(args.smoke_manifest))
    else:
        execution_rows = read_jsonl(Path(args.candidate_manifest))
        gate = json.loads(Path(args.smoke_gate).read_text(encoding="utf-8"))
        validate_smoke_gate(
            gate,
            contract_id=contract["contract_sha256"],
            candidate_manifest_sha256=contract["integrity"][
                "candidate_manifest_sha256"
            ],
            require_features=args.extract_features,
        )
    validate_candidate_execution_contract(
        execution_rows,
        evaluator_specs=config["evaluators"],
        max_new_tokens=int(config["generation"]["max_new_tokens"]),
    )

    rank = int(os.environ.get("LOCAL_RANK", args.rank))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    if world_size != EXPECTED_FULL_WORLD_SIZE or not 0 <= rank < world_size:
        raise ValueError(
            f"dense execution requires four ranks 0..3: rank={rank} world_size={world_size}"
        )

    # No CUDA model is loaded until every provenance and policy gate has passed.
    device_index = int(os.environ.get("LOCAL_RANK", args.device_index))
    processor, model, device = load_dense_runtime(
        args.model_path, args.revision, device_index
    )
    if args.mode == "smoke-suite":
        run_smoke_suite(args, contract, processor, model, device)
    else:
        run_full(args, contract, processor, model, device)


if __name__ == "__main__":
    main()
