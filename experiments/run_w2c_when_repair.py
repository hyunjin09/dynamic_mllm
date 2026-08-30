#!/usr/bin/env python3
"""Run deterministic smoke or full W2C WHEN repair on four direct GPUs."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
import yaml

from experiments.evaluate_four_action_polar_external import execute_actions
from experiments.train_binary_polar import file_sha256
from experiments.train_four_action_online_router import prepare_sample
from four_action_online_router.data import load_source_metadata, load_verified_manifest
from four_action_policy.when_repair import repair_w2c_sample
from label_regeneration.runtime import configure_determinism, load_frozen_model


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def record_filename(uid: str) -> str:
    return f"{sha256(str(uid).encode()).hexdigest()}.json"


def distributed_context(expected_world_size: int) -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if (
        expected_world_size != 4
        or world_size != expected_world_size
        or rank < 0
        or local_rank < 0
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < expected_world_size
        or local_rank >= torch.cuda.device_count()
    ):
        raise RuntimeError("W2C WHEN repair requires direct four-GPU torchrun")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    return rank, world_size, local_rank, device


def validate_config(config: dict[str, Any], config_path: Path) -> str:
    if config.get("protocol_version") != "w2c_when_repair_v1":
        raise RuntimeError("incompatible W2C WHEN repair protocol")
    if file_sha256(Path(config["source_plan"])) != config["source_plan_sha256"]:
        raise RuntimeError("source-plan checksum mismatch")
    parent = Path(config["parent_online_config"]["path"])
    if file_sha256(parent) != config["parent_online_config"]["sha256"]:
        raise RuntimeError("parent online-config checksum mismatch")
    if (
        int(config["search"]["per_state_variant_budget"]) != 96
        or config["search"]["strategy"]
        != "known_suffix_then_stratified_one_edit_suffix_variants"
    ):
        raise RuntimeError("repair search contract is not the frozen reviewed strategy")
    for relative, expected in config["executor_contract"]["code_sha256"].items():
        if file_sha256(Path(relative)) != expected:
            raise RuntimeError(f"executor code checksum mismatch: {relative}")
    for name in ("source_manifest", "boundary_manifest", "source_metadata_manifest"):
        path = Path(config["data"][name])
        if file_sha256(path) != config["data"][f"{name}_sha256"]:
            raise RuntimeError(f"data checksum mismatch: {name}")
    return file_sha256(config_path)


def validate_record(
    path: Path, *, config_sha256: str, uid: str, mode: str | None = None
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "w2c_when_repair_sample_v1"
        or payload.get("config_sha256") != config_sha256
        or str(payload.get("uid")) != str(uid)
        or (mode is not None and payload.get("mode") != mode)
        or payload.get("status") not in {"completed", "quarantined"}
    ):
        raise RuntimeError(f"incompatible repair record: {path}")
    return payload


def _route_evaluation(
    *,
    wrapped_model,
    processor,
    inputs,
    prepared,
    sample,
    input_metadata: dict[str, Any],
    actions: tuple[str, ...],
    config: dict[str, Any],
) -> dict[str, Any]:
    execution = execute_actions(
        wrapped_model=wrapped_model,
        processor=processor,
        inputs=inputs,
        prepared=prepared,
        actions=actions,
        row=sample,
        eos_token_ids=list(config["executor_contract"]["eos_token_ids"]),
        repetition_penalty=float(config["executor_contract"]["repetition_penalty"]),
    )
    return {
        "prediction": execution["prediction"],
        "score": float(execution["score"]),
        "correct": bool(execution["correct"]),
        "generated_ids": execution["generated_ids"],
        "execution_source": execution["execution_source"],
        "prompt_sha256": input_metadata["prompt_sha256"],
    }


def process_sample(
    *,
    processor,
    wrapped_model,
    row: dict[str, Any],
    source: dict[str, Any],
    device: torch.device,
    config: dict[str, Any],
    config_sha256: str,
    mode: str,
    rank: int,
) -> dict[str, Any]:
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    sample, inputs, input_metadata, prepared = prepare_sample(
        processor, wrapped_model, row, source, device
    )

    def evaluate(actions: tuple[str, ...]) -> dict[str, Any]:
        return _route_evaluation(
            wrapped_model=wrapped_model,
            processor=processor,
            inputs=inputs,
            prepared=prepared,
            sample=sample,
            input_metadata=input_metadata,
            actions=actions,
            config=config,
        )

    old_route_replays = []
    if mode == "smoke":
        for route_index, route in enumerate(row["valid_routes"]):
            old_route_replays.append(
                {
                    "route_index": route_index,
                    "route_key": route["route_key"],
                    **evaluate(tuple(str(action) for action in route["actions"])),
                }
            )
    repair = repair_w2c_sample(
        row,
        evaluate,
        search_budget=int(config["search"]["per_state_variant_budget"]),
        seed=int(config["search"]["seed"]),
    )
    status = "completed" if repair["status"] == "FULL_UNRESCUED_UNDER_BUDGET" else "quarantined"
    return {
        "schema_version": "w2c_when_repair_sample_v1",
        "config_sha256": config_sha256,
        "mode": mode,
        "uid": row["uid"],
        "split": row["split"],
        "dataset": row["dataset"],
        "status": status,
        "old_route_replays": old_route_replays,
        "repair": repair,
        "input_metadata": input_metadata,
        "runtime": {
            "rank": rank,
            "gpu_index": device.index,
            "elapsed_seconds": time.time() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    }


def _quarantine(
    *,
    row: dict[str, Any],
    config_sha256: str,
    mode: str,
    rank: int,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "schema_version": "w2c_when_repair_sample_v1",
        "config_sha256": config_sha256,
        "mode": mode,
        "uid": row["uid"],
        "split": row["split"],
        "dataset": row["dataset"],
        "status": "quarantined",
        "failure": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        "runtime": {"rank": rank},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="analysis/w2c_when_repair/repair_config.yaml"
    )
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_sha = validate_config(config, config_path)
    manifest_key = "smoke_manifest" if args.mode == "smoke" else "repair_manifest"
    assignment_path = Path(config["data"][manifest_key])
    if file_sha256(assignment_path) != config["data"][f"{manifest_key}_sha256"]:
        raise RuntimeError(f"{manifest_key} checksum mismatch")
    assignments = json.loads(assignment_path.read_text(encoding="utf-8"))

    source_rows = load_verified_manifest(
        config["data"]["source_manifest"], config["data"]["source_manifest_sha256"]
    )
    source_by_uid = {str(row["uid"]): row for row in source_rows}
    required_uids = {str(row["uid"]) for row in assignments}
    if len(assignments) != len(required_uids) or not required_uids.issubset(source_by_uid):
        raise RuntimeError("repair assignments are not uniquely covered by the source manifest")
    if any(source_by_uid[uid].get("route_type") != "W2C" for uid in required_uids):
        raise RuntimeError("repair assignments contain a non-W2C sample")

    parent_config = yaml.safe_load(
        Path(config["parent_online_config"]["path"]).read_text(encoding="utf-8")
    )
    sys.path.insert(
        0, str(Path(parent_config["external_evaluation"]["protocol"]) / "code")
    )
    rank, world_size, local_rank, device = distributed_context(args.world_size)
    try:
        assigned = [row for row in assignments if int(row["rank"]) == rank]
        raw_root = Path(config["execution"]["raw_output_root"])
        record_root = raw_root / args.mode / "records"
        if rank == 0:
            record_root.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        pending = []
        reused = 0
        for assignment in assigned:
            uid = str(assignment["uid"])
            output_path = record_root / record_filename(uid)
            if output_path.is_file():
                validate_record(
                    output_path,
                    config_sha256=config_sha,
                    uid=uid,
                    mode=args.mode,
                )
                reused += 1
                continue
            if args.mode == "full":
                smoke_path = raw_root / "smoke" / "records" / record_filename(uid)
                if smoke_path.is_file():
                    smoke = validate_record(
                        smoke_path,
                        config_sha256=config_sha,
                        uid=uid,
                        mode="smoke",
                    )
                    reused_record = {
                        **smoke,
                        "mode": "full",
                        "reused_from_smoke": str(smoke_path),
                    }
                    atomic_json(output_path, reused_record)
                    reused += 1
                    continue
            pending.append(assignment)

        print(
            json.dumps(
                {
                    "event": "repair_rank_ready",
                    "mode": args.mode,
                    "rank": rank,
                    "assigned": len(assigned),
                    "reused": reused,
                    "pending": len(pending),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if pending:
            configure_determinism(int(config["execution"]["seed"]))
            pending_uids = {str(row["uid"]) for row in pending}
            sources = load_source_metadata(
                config["data"]["source_metadata_manifest"],
                config["data"]["source_metadata_manifest_sha256"],
                pending_uids,
            )
            processor, base_model, wrapped_model, _ = load_frozen_model(
                parent_config["base_model"]["path"],
                parent_config["base_model"]["revision"],
                local_rank,
            )
            base_model.requires_grad_(False).eval()
            for index, assignment in enumerate(pending, start=1):
                uid = str(assignment["uid"])
                row = source_by_uid[uid]
                try:
                    payload = process_sample(
                        processor=processor,
                        wrapped_model=wrapped_model,
                        row=row,
                        source=sources[uid],
                        device=device,
                        config=config,
                        config_sha256=config_sha,
                        mode=args.mode,
                        rank=rank,
                    )
                except Exception as error:
                    payload = _quarantine(
                        row=row,
                        config_sha256=config_sha,
                        mode=args.mode,
                        rank=rank,
                        error=error,
                    )
                atomic_json(record_root / record_filename(uid), payload)
                print(
                    json.dumps(
                        {
                            "event": "repair_sample_complete",
                            "mode": args.mode,
                            "rank": rank,
                            "sample": index,
                            "pending_samples": len(pending),
                            "uid": uid,
                            "status": payload["status"],
                            "new_boundary": (payload.get("repair") or {}).get(
                                "new_boundary"
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                torch.cuda.empty_cache()

        dist.barrier()
        if rank == 0:
            records = [
                validate_record(
                    record_root / record_filename(str(assignment["uid"])),
                    config_sha256=config_sha,
                    uid=str(assignment["uid"]),
                    mode=args.mode,
                )
                for assignment in assignments
            ]
            summary = {
                "schema_version": "w2c_when_repair_run_summary_v1",
                "config_sha256": config_sha,
                "mode": args.mode,
                "records": len(records),
                "completed": sum(row["status"] == "completed" for row in records),
                "quarantined": sum(row["status"] == "quarantined" for row in records),
                "route_executions": sum(
                    len((row.get("repair") or {}).get("route_execution_cache", []))
                    + len(row.get("old_route_replays", []))
                    for row in records
                ),
            }
            atomic_json(raw_root / args.mode / "summary.json", summary)
            print(
                json.dumps(
                    {"event": "w2c_when_repair_mode_complete", **summary},
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
