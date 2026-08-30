#!/usr/bin/env python3
"""Repeat one failed original-route replay per GPU under the frozen runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.distributed as dist
import yaml

from experiments.run_w2c_when_repair import (
    _route_evaluation,
    atomic_json,
    distributed_context,
    record_filename,
    validate_config,
)
from experiments.train_four_action_online_router import prepare_sample
from four_action_online_router.data import load_source_metadata, load_verified_manifest
from label_regeneration.runtime import configure_determinism, load_frozen_model


def write_frozen_json(path: Path, payload: dict) -> None:
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != payload:
            raise RuntimeError(f"replay diagnostic changed on restart: {path}")
        return
    atomic_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="analysis/w2c_when_repair/repair_config.yaml"
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_sha = validate_config(config, config_path)
    rank, world_size, local_rank, device = distributed_context(4)
    try:
        assignments = json.loads(
            Path(config["data"]["smoke_manifest"]).read_text(encoding="utf-8")
        )
        source_rows = load_verified_manifest(
            config["data"]["source_manifest"],
            config["data"]["source_manifest_sha256"],
        )
        source_by_uid = {str(row["uid"]): row for row in source_rows}
        raw_root = Path(config["execution"]["raw_output_root"])
        smoke_root = raw_root / "smoke" / "records"
        candidates = []
        for assignment in assignments:
            if int(assignment["rank"]) != rank:
                continue
            uid = str(assignment["uid"])
            smoke = json.loads(
                (smoke_root / record_filename(uid)).read_text(encoding="utf-8")
            )
            for replay in smoke["old_route_replays"]:
                if replay["correct"] is not True:
                    candidates.append((uid, int(replay["route_index"]), assignment, replay))
        if not candidates:
            raise RuntimeError(f"rank {rank} has no failed old-route replay to diagnose")
        uid, route_index, assignment, first_replay = min(
            candidates, key=lambda row: (row[0], row[1])
        )

        configure_determinism(int(config["execution"]["seed"]))
        sources = load_source_metadata(
            config["data"]["source_metadata_manifest"],
            config["data"]["source_metadata_manifest_sha256"],
            {uid},
        )
        parent = yaml.safe_load(
            Path(config["parent_online_config"]["path"]).read_text(encoding="utf-8")
        )
        sys.path.insert(0, str(Path(parent["external_evaluation"]["protocol"]) / "code"))
        processor, base_model, wrapped_model, _ = load_frozen_model(
            parent["base_model"]["path"], parent["base_model"]["revision"], local_rank
        )
        base_model.requires_grad_(False).eval()
        row = source_by_uid[uid]
        sample, inputs, input_metadata, prepared = prepare_sample(
            processor, wrapped_model, row, sources[uid], device
        )
        actions = tuple(str(value) for value in row["valid_routes"][route_index]["actions"])
        repeated = [
            _route_evaluation(
                wrapped_model=wrapped_model,
                processor=processor,
                inputs=inputs,
                prepared=prepared,
                sample=sample,
                input_metadata=input_metadata,
                actions=actions,
                config=config,
            )
            for _ in range(2)
        ]

        source_record = json.loads(
            Path(assignment["source_record"]).read_text(encoding="utf-8")
        )
        original_by_key = {
            str(item["route_key"]): item["evaluation"]
            for item in source_record["unique_valid_four_action_routes"]
        }
        route_key = str(row["valid_routes"][route_index]["route_key"])
        payload = {
            "schema_version": "w2c_when_repair_replay_diagnostic_v1",
            "config_sha256": config_sha,
            "rank": rank,
            "uid": uid,
            "dataset": row["dataset"],
            "route_index": route_index,
            "route_key": route_key,
            "original_cached_evaluation": original_by_key[route_key],
            "first_smoke_replay": first_replay,
            "repeat_1": repeated[0],
            "repeat_2": repeated[1],
            "current_repeats_exact": repeated[0] == repeated[1],
            "current_matches_original_generated_ids": (
                repeated[0]["generated_ids"]
                == original_by_key[route_key]["generated_ids"]
            ),
        }
        output_root = raw_root / "smoke_replay_diagnostic"
        write_frozen_json(output_root / "records" / f"rank_{rank}.json", payload)
        print(json.dumps({"event": "replay_diagnostic_rank_complete", **payload}, sort_keys=True))
        dist.barrier()
        if rank == 0:
            records = [
                json.loads(
                    (output_root / "records" / f"rank_{value}.json").read_text(
                        encoding="utf-8"
                    )
                )
                for value in range(world_size)
            ]
            summary = {
                "schema_version": "w2c_when_repair_replay_diagnostic_summary_v1",
                "config_sha256": config_sha,
                "records": len(records),
                "current_repeat_exact": sum(
                    row["current_repeats_exact"] for row in records
                ),
                "current_matches_original": sum(
                    row["current_matches_original_generated_ids"] for row in records
                ),
                "uids": [row["uid"] for row in records],
            }
            write_frozen_json(output_root / "summary.json", summary)
            print(json.dumps({"event": "replay_diagnostic_complete", **summary}, sort_keys=True))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
