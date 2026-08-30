#!/usr/bin/env python3
"""Run the frozen bounded label-incompleteness route audit on four GPUs."""

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
from four_action_policy.generalization_diagnostics import (
    summarize_label_incompleteness_audit,
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
        or local_rank >= torch.cuda.device_count()
    ):
        raise RuntimeError("label-incompleteness audit requires direct four-GPU torchrun")
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


def validate_subset(rows: list[dict[str, Any]], *, max_suffixes: int) -> None:
    if not rows:
        raise RuntimeError("frozen label-incompleteness subset is empty")
    state_ids = [str(row.get("state_id") or "") for row in rows]
    audit_units = [
        (str(row.get("architecture") or ""), state_id)
        for row, state_id in zip(rows, state_ids)
    ]
    if (
        any(not state_id or not architecture for architecture, state_id in audit_units)
        or len(audit_units) != len(set(audit_units))
    ):
        raise RuntimeError("frozen label audit has empty or duplicate architecture/state units")
    for row in rows:
        if (
            row.get("split") != "validation"
            or row.get("state_kind") != "mandatory_deviation"
            or row.get("predicted_action") not in FOUR_ACTIONS
            or row["predicted_action"] == "FULL"
            or row["predicted_action"] in row["valid_actions"]
        ):
            raise RuntimeError("frozen label audit contains an ineligible state")
        layer = int(row["target_layer"])
        candidates = row.get("candidate_routes")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= max_suffixes:
            raise RuntimeError("frozen label audit violates its route cap")
        indices = [int(candidate["candidate_index"]) for candidate in candidates]
        if indices != list(range(len(candidates))):
            raise RuntimeError("frozen label audit candidate indices are not contiguous")
        for candidate in candidates:
            actions = [str(action) for action in candidate["actions"]]
            if (
                len(actions) != 28
                or actions[:layer] != row["prefix_actions"]
                or actions[layer] != row["predicted_action"]
                or any(action not in FOUR_ACTIONS for action in actions)
            ):
                raise RuntimeError("frozen label audit candidate changes the audited state")


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
        != "four_action_generalization_label_audit_shard_v1"
        or payload.get("config_sha256") != config_sha256
        or payload.get("subset_sha256") != subset_sha256
        or int(payload.get("rank", -1)) != rank
        or int(payload.get("world_size", -1)) != world_size
    ):
        raise RuntimeError(f"incompatible label-audit shard: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="analysis/4action_generalization_diagnostics/diagnostic_config.yaml",
    )
    parser.add_argument(
        "--subset",
        default="analysis/4action_generalization_diagnostics/label_incompleteness_subset.jsonl",
    )
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()

    config_path = Path(args.config)
    config_sha = file_sha256(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_version") != "four_action_generalization_diagnostics_v1":
        raise RuntimeError("label audit requires the frozen diagnostic config")
    if file_sha256(Path(config["source_plan"])) != config["source_plan_sha256"]:
        raise RuntimeError("diagnostic source-plan checksum mismatch")
    for name, parent in config["parent_configs"].items():
        if file_sha256(Path(parent["path"])) != parent["sha256"]:
            raise RuntimeError(f"{name} parent config checksum mismatch")

    subset_path = Path(args.subset)
    subset_sha = file_sha256(subset_path)
    subset = load_jsonl(subset_path)
    validate_subset(
        subset,
        max_suffixes=int(config["label_incompleteness"]["max_known_suffixes"]),
    )
    expected_candidates = sum(len(row["candidate_routes"]) for row in subset)

    online_config = yaml.safe_load(
        Path(config["parent_configs"]["online"]["path"]).read_text(encoding="utf-8")
    )
    sys.path.insert(0, str(Path(online_config["external_evaluation"]["protocol"]) / "code"))
    rank, world_size, local_rank, device = distributed_context(args.world_size)
    try:
        configure_determinism(int(config["label_incompleteness"]["seed"]))
        required_uids = {str(row["uid"]) for row in subset}
        source_rows = load_verified_manifest(
            config["data"]["source_manifest"],
            config["data"]["source_manifest_sha256"],
        )
        source_by_uid = {
            str(row["uid"]): row for row in source_rows if str(row["uid"]) in required_uids
        }
        if set(source_by_uid) != required_uids:
            raise RuntimeError("label audit UIDs are not covered by the frozen manifest")
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

        shard_root = (
            Path(config["extraction"]["raw_output_root"])
            / "label_incompleteness_audit"
        )
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
                        "event": "label_audit_shard_reused",
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
                        eos_token_ids=list(
                            online_config["external_evaluation"]["eos_token_ids"]
                        ),
                        repetition_penalty=float(
                            online_config["external_evaluation"]["repetition_penalty"]
                        ),
                    )
                    records.append(
                        {
                            "state_id": state["state_id"],
                            "uid": state["uid"],
                            "dataset": state["dataset"],
                            "architecture": state["architecture"],
                            "predicted_action": state["predicted_action"],
                            "target_layer": int(state["target_layer"]),
                            "valid_actions": state["valid_actions"],
                            "candidate_index": int(candidate["candidate_index"]),
                            "source_route_index": int(candidate["source_route_index"]),
                            "source_route_key": candidate["source_route_key"],
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
                            "event": "label_audit_state_complete",
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
                "schema_version": "four_action_generalization_label_audit_shard_v1",
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
            if len(executions) != expected_candidates:
                raise RuntimeError("label-audit shards do not cover every candidate")
            summary = summarize_label_incompleteness_audit(subset, executions)
            summary.update(
                {
                    "schema_version": "four_action_generalization_label_audit_v1",
                    "config_sha256": config_sha,
                    "subset_sha256": subset_sha,
                    "world_size": world_size,
                    "base_model_revision": online_config["base_model"]["revision"],
                    "executor": "live_unified_four_action_known_suffix_replay",
                    "bounded_continuation": "known_compatible_correcting_route_suffixes_only",
                }
            )
            output_path = Path(config["reporting"]["analysis_dir"]) / "label_incompleteness_results.json"
            atomic_json(output_path, summary)
            print(
                json.dumps(
                    {
                        "event": "label_incompleteness_audit_complete",
                        "states": summary["states"],
                        "candidate_executions": summary["candidate_executions"],
                        "rescued_states": summary[
                            "cached_invalid_but_execution_correct_states"
                        ],
                        "output": str(output_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
