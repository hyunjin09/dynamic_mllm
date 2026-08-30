#!/usr/bin/env python3
"""Execute the frozen Phase-1 FULL-insertion audit on four direct GPUs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
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
from four_action_online_router.data import (
    load_jsonl,
    load_source_metadata,
    load_verified_manifest,
)
from four_action_policy.actions import FOUR_ACTIONS
from four_action_policy.selective_continue_deviate import (
    build_full_insertion_subset,
    summarize_full_insertion_audit,
)
from label_regeneration.runtime import configure_determinism, load_frozen_model


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
        raise RuntimeError("Phase-1 audit requires direct four-GPU torchrun")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    return rank, world_size, local_rank, device


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_subset(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 128 or len({str(row.get("uid")) for row in rows}) != 128:
        raise RuntimeError("audit subset must contain 128 unique UIDs")
    if len({str(row.get("state_id")) for row in rows}) != 128:
        raise RuntimeError("audit subset must contain 128 unique state IDs")
    candidates = 0
    for row in rows:
        layer = int(row["target_layer"])
        routes = row.get("candidate_routes")
        if (
            row.get("split") != "validation"
            or row.get("route_type") != "W2C"
            or not row.get("suffix_set_complete")
            or not isinstance(routes, list)
            or not routes
        ):
            raise RuntimeError("audit subset contains an ineligible state")
        if [int(route["candidate_index"]) for route in routes] != list(
            range(len(routes))
        ):
            raise RuntimeError("candidate indices are not contiguous")
        for route in routes:
            actions = [str(action) for action in route["actions"]]
            if (
                len(actions) != 28
                or actions[:layer] != ["FULL"] * layer
                or actions[layer] != "FULL"
                or any(action not in FOUR_ACTIONS for action in actions)
                or not route.get("source_route_indices")
            ):
                raise RuntimeError("candidate violates the FULL-insertion contract")
        candidates += len(routes)
    if candidates != 252:
        raise RuntimeError("audit subset must contain exactly 252 candidate routes")


def validate_shard(
    path: Path,
    *,
    config_sha256: str,
    subset_sha256: str,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version")
        != "selective_continue_deviate_full_insertion_shard_v1"
        or payload.get("config_sha256") != config_sha256
        or payload.get("subset_sha256") != subset_sha256
        or int(payload.get("rank", -1)) != rank
        or int(payload.get("world_size", -1)) != world_size
    ):
        raise RuntimeError(f"incompatible Phase-1 audit shard: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="analysis/selective_continue_deviate/audit_config.yaml"
    )
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()

    config_path = Path(args.config)
    config_sha = file_sha256(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_version") != "selective_continue_deviate_phase1_v1":
        raise RuntimeError("audit config has an incompatible protocol version")
    if file_sha256(Path(config["source_plan"])) != config["source_plan_sha256"]:
        raise RuntimeError("source-plan checksum mismatch")
    parent_path = Path(config["parent_online_config"]["path"])
    if file_sha256(parent_path) != config["parent_online_config"]["sha256"]:
        raise RuntimeError("parent online-config checksum mismatch")
    online_config = yaml.safe_load(parent_path.read_text(encoding="utf-8"))

    subset_path = Path(config["data"]["audit_subset"])
    subset_sha = file_sha256(subset_path)
    if subset_sha != config["data"]["audit_subset_sha256"]:
        raise RuntimeError("audit-subset checksum mismatch")
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    validate_subset(subset)

    source_rows = load_verified_manifest(
        config["data"]["source_manifest"],
        config["data"]["source_manifest_sha256"],
    )
    boundaries = load_jsonl(config["data"]["boundary_manifest"])
    if file_sha256(Path(config["data"]["boundary_manifest"])) != config["data"][
        "boundary_manifest_sha256"
    ]:
        raise RuntimeError("boundary-manifest checksum mismatch")
    rebuilt, _ = build_full_insertion_subset(source_rows, boundaries, split="validation")
    if rebuilt != subset:
        raise RuntimeError("frozen audit subset is not reproducible from source inputs")

    sys.path.insert(
        0, str(Path(online_config["external_evaluation"]["protocol"]) / "code")
    )
    rank, world_size, local_rank, device = distributed_context(args.world_size)
    try:
        configure_determinism(int(config["execution"]["seed"]))
        required_uids = {str(row["uid"]) for row in subset}
        source_by_uid = {
            str(row["uid"]): row
            for row in source_rows
            if str(row["uid"]) in required_uids
        }
        if set(source_by_uid) != required_uids:
            raise RuntimeError("source manifest does not cover the frozen audit UIDs")
        sources = load_source_metadata(
            config["data"]["source_metadata_manifest"],
            config["data"]["source_metadata_manifest_sha256"],
            required_uids,
        )

        processor, base_model, wrapped_model, _ = load_frozen_model(
            online_config["base_model"]["path"],
            online_config["base_model"]["revision"],
            local_rank,
        )
        base_model.requires_grad_(False).eval()

        shard_root = Path(config["execution"]["raw_output_root"]) / "full_insertion_audit"
        if rank == 0:
            shard_root.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        shard_path = shard_root / f"shard_{rank:02d}.json"
        if shard_path.is_file():
            payload = validate_shard(
                shard_path,
                config_sha256=config_sha,
                subset_sha256=subset_sha,
                rank=rank,
                world_size=world_size,
            )
            print(
                json.dumps(
                    {
                        "event": "full_insertion_shard_reused",
                        "rank": rank,
                        "candidate_executions": len(payload["records"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            assigned = [row for index, row in enumerate(subset) if index % world_size == rank]
            records = []
            started = time.time()
            for state_index, state in enumerate(assigned, start=1):
                row = source_by_uid[state["uid"]]
                sample, inputs, input_metadata, prepared = prepare_sample(
                    processor,
                    wrapped_model,
                    row,
                    sources[state["uid"]],
                    device,
                )
                for candidate in state["candidate_routes"]:
                    execution = execute_actions(
                        wrapped_model=wrapped_model,
                        processor=processor,
                        inputs=inputs,
                        prepared=prepared,
                        actions=tuple(candidate["actions"]),
                        row=sample,
                        eos_token_ids=list(config["execution"]["eos_token_ids"]),
                        repetition_penalty=float(
                            config["execution"]["repetition_penalty"]
                        ),
                    )
                    records.append(
                        {
                            "state_id": state["state_id"],
                            "uid": state["uid"],
                            "dataset": state["dataset"],
                            "target_layer": int(state["target_layer"]),
                            "depth_bin": state["depth_bin"],
                            "known_mechanism": state["known_mechanism"],
                            "candidate_index": int(candidate["candidate_index"]),
                            "source_route_indices": candidate["source_route_indices"],
                            "source_route_keys": candidate["source_route_keys"],
                            "actions": candidate["actions"],
                            "prediction": execution["prediction"],
                            "score": float(execution["score"]),
                            "correct": bool(execution["correct"]),
                            "generated_ids": execution["generated_ids"],
                            "execution_source": execution["execution_source"],
                            "prompt_sha256": input_metadata["prompt_sha256"],
                        }
                    )
                print(
                    json.dumps(
                        {
                            "event": "full_insertion_state_complete",
                            "rank": rank,
                            "state": state_index,
                            "assigned_states": len(assigned),
                            "uid": state["uid"],
                            "candidates": len(state["candidate_routes"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            payload = {
                "schema_version": "selective_continue_deviate_full_insertion_shard_v1",
                "config_sha256": config_sha,
                "subset_sha256": subset_sha,
                "rank": rank,
                "world_size": world_size,
                "elapsed_seconds": time.time() - started,
                "records": records,
            }
            atomic_json(shard_path, payload)

        dist.barrier()
        if rank == 0:
            executions = []
            for shard_rank in range(world_size):
                shard = validate_shard(
                    shard_root / f"shard_{shard_rank:02d}.json",
                    config_sha256=config_sha,
                    subset_sha256=subset_sha,
                    rank=shard_rank,
                    world_size=world_size,
                )
                executions.extend(shard["records"])
            executions.sort(key=lambda row: (row["state_id"], row["candidate_index"]))
            summary = summarize_full_insertion_audit(
                subset,
                executions,
                bootstrap_draws=int(config["bootstrap"]["draws"]),
                bootstrap_seed=int(config["bootstrap"]["seed"]),
            )
            summary.update(
                {
                    "config_sha256": config_sha,
                    "subset_sha256": subset_sha,
                    "world_size": world_size,
                    "base_model_revision": online_config["base_model"]["revision"],
                    "executor": config["execution"]["executor"],
                    "bounded_continuation": "all_known_compatible_correcting_route_suffixes",
                }
            )
            atomic_jsonl(Path(config["reporting"]["execution_records"]), executions)
            atomic_json(Path(config["reporting"]["results"]), summary)
            print(
                json.dumps(
                    {
                        "event": "full_insertion_audit_complete",
                        "states": summary["states"],
                        "candidate_executions": summary["candidate_executions"],
                        "rescued_states": summary["overall"]["rescued"],
                        "unresolved_states": summary["status_counts"].get(
                            "unresolved", 0
                        ),
                        "output": config["reporting"]["results"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
