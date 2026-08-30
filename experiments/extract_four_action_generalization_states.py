#!/usr/bin/env python3
"""Extract selected-checkpoint logits and frozen representations on four GPUs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
import yaml

from binary_policy.predictor import FrozenHFTokenEncoder
from experiments.evaluate_persistent_corrective_polar import load_checkpoint_predictor
from experiments.train_binary_polar import file_sha256
from experiments.train_four_action_online_router import make_router, prepare_sample
from four_action_online_router.data import (
    load_jsonl,
    load_source_metadata,
    load_verified_manifest,
    manifest_route_tensor,
    manifest_trie,
)
from four_action_online_router.runtime import replay_teacher_forced_states
from four_action_policy.actions import FOUR_ACTIONS
from four_action_policy.feature_cache import load_verified_feature_index
from label_regeneration.runtime import configure_determinism, load_frozen_model


def distributed_context(expected_world_size: int) -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if (
        world_size != expected_world_size
        or world_size != 4
        or rank < 0
        or local_rank < 0
        or not torch.cuda.is_available()
        or local_rank >= torch.cuda.device_count()
    ):
        raise RuntimeError("diagnostic state extraction requires direct four-GPU torchrun")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    return rank, world_size, local_rank, device


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _online_logits_from_features(router, features) -> torch.Tensor:
    read_score = router.read_head(features.read_state).squeeze(-1)
    write_score = router.write_head(features.write_state).squeeze(-1)
    interaction = router.interaction_head(
        torch.cat([features.read_state, features.write_state], dim=-1)
    )
    return router.structured_logits(
        read_score,
        write_score,
        interaction,
        interaction_scale=router.interaction_scale,
    )


@torch.inference_mode()
def polar_outputs_for_uid(
    row: dict[str, Any],
    *,
    feature_index: dict[str, dict[str, Any]],
    tokenizer,
    encoder,
    predictor,
    max_question_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = tokenizer(
        [row["question"]],
        padding=True,
        truncation=True,
        max_length=max_question_tokens,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)
    attention = tokens["attention_mask"].to(device)
    question = encoder(input_ids, attention)
    visual = torch.load(
        feature_index[row["uid"]]["path"], map_location="cpu", weights_only=True
    ).to(device=device, dtype=torch.bfloat16)
    visual = visual.unsqueeze(0)
    visual_mask = torch.ones(visual.shape[:2], dtype=torch.bool, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        representation = predictor.encoder(question, attention, visual, visual_mask)
        logits = predictor.route_head(representation)
    return representation[0].float().cpu(), logits[0].float().cpu()


@torch.inference_mode()
def capture_online_state(
    state: dict[str, Any],
    row: dict[str, Any],
    source: dict[str, Any],
    *,
    processor,
    wrapped_model,
    router,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    _sample, _inputs, _metadata, prepared = prepare_sample(
        processor, wrapped_model, row, source, device
    )
    routes = manifest_route_tensor(row, num_layers=router.num_layers)
    route_index = int(state["teacher_route_index"])
    if not 0 <= route_index < len(routes):
        raise RuntimeError("diagnostic teacher route index is invalid")
    route = routes[route_index]
    prefix = [FOUR_ACTIONS[int(value)] for value in route[: int(state["target_layer"])] ]
    if prefix != state["prefix_actions"]:
        raise RuntimeError("diagnostic teacher route does not reach the frozen prefix")
    trajectory = replay_teacher_forced_states(
        wrapped_model,
        prepared,
        route,
        manifest_trie(row, num_layers=router.num_layers),
    )
    layer = int(state["target_layer"])
    observed_mask = trajectory.valid_action_mask[layer].detach().cpu().tolist()
    if observed_mask != state["valid_action_mask"]:
        raise RuntimeError("replayed trie actions differ from the state manifest")
    return {
        "text": trajectory.text_queries[layer].detach().to(device="cpu", dtype=torch.bfloat16),
        "visual": trajectory.visual_states[layer].detach().to(device="cpu", dtype=torch.bfloat16),
        "visual_mask": trajectory.visual_valid_mask[layer].detach().cpu().bool(),
    }


@torch.inference_mode()
def online_outputs(router, raw: dict[str, torch.Tensor], *, layer: int, device: torch.device) -> dict[str, Any]:
    text = raw["text"].to(device).unsqueeze(0)
    visual = raw["visual"].to(device).unsqueeze(0)
    mask = raw["visual_mask"].to(device).unsqueeze(0)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        features = router.forward_features(text, visual, mask, layer)
        logits = _online_logits_from_features(router, features)
    probabilities = torch.softmax(logits.float(), dim=-1)[0]
    return {
        "logits": logits.float()[0].cpu(),
        "probabilities": probabilities.cpu(),
        "z_read": features.read_state.float()[0].cpu(),
        "z_write": features.write_state.float()[0].cpu(),
        "read_visual": features.read_visual.float()[0].cpu(),
        "write_visual": features.write_visual.float()[0].cpu(),
    }


def _assigned_cells(states: list[dict[str, Any]], world_size: int) -> list[list[tuple[str, str, int]]]:
    counts = defaultdict(int)
    for state in states:
        counts[(state["split"], state["dataset"], int(state["target_layer"]))] += 1
    assignments: list[list[tuple[str, str, int]]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for cell, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        assignments[rank].append(cell)
        loads[rank] += count
    return assignments


def validate_shard(path: Path, *, config_sha: str, rank: int, world_size: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != "four_action_generalization_state_features_v1"
        or payload.get("config_sha256") != config_sha
        or int(payload.get("rank", -1)) != rank
        or int(payload.get("world_size", -1)) != world_size
    ):
        raise RuntimeError(f"incompatible diagnostic state shard: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="analysis/4action_generalization_diagnostics/diagnostic_config.yaml",
    )
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    config_path = Path(args.config)
    config_sha = file_sha256(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_version") != "four_action_generalization_diagnostics_v1":
        raise RuntimeError("state extraction requires the frozen diagnostic config")
    for name, parent in config["parent_configs"].items():
        if file_sha256(Path(parent["path"])) != parent["sha256"]:
            raise RuntimeError(f"{name} parent config checksum mismatch")
    if file_sha256(Path(config["data"]["state_manifest"])) != config["data"]["state_manifest_sha256"]:
        raise RuntimeError("diagnostic state-manifest checksum mismatch")

    rank, world_size, local_rank, device = distributed_context(args.world_size)
    try:
        configure_determinism(int(config["extraction"]["seed"]))
        state_rows = load_jsonl(config["data"]["state_manifest"])
        source_rows = load_verified_manifest(
            config["data"]["source_manifest"],
            config["data"]["source_manifest_sha256"],
        )
        source_by_uid = {str(row["uid"]): row for row in source_rows}
        required_uids = {str(row["uid"]) for row in state_rows}
        sources = load_source_metadata(
            config["data"]["source_metadata_manifest"],
            config["data"]["source_metadata_manifest_sha256"],
            required_uids,
        )
        polar_config = yaml.safe_load(
            Path(config["parent_configs"]["polar"]["path"]).read_text(encoding="utf-8")
        )
        online_config = yaml.safe_load(
            Path(config["parent_configs"]["online"]["path"]).read_text(encoding="utf-8")
        )
        feature_index = load_verified_feature_index(
            config["data"]["visual_feature_manifest"],
            manifest_sha256=config["data"]["visual_feature_manifest_sha256"],
            expected_uids={str(row["uid"]) for row in source_rows},
            expected_feature_width=int(polar_config["visual_features"]["feature_width"]),
            verify_tensors=False,
        )

        processor, base_model, wrapped_model, _ = load_frozen_model(
            online_config["base_model"]["path"],
            online_config["base_model"]["revision"],
            local_rank,
        )
        base_model.requires_grad_(False).eval()
        online_router = make_router(online_config, device).eval()
        online_payload = torch.load(
            config["checkpoints"]["online"]["path"], map_location="cpu", weights_only=False
        )
        if online_payload.get("config_sha256") != config["parent_configs"]["online"]["sha256"]:
            raise RuntimeError("online checkpoint config checksum mismatch")
        online_router.load_state_dict(online_payload["router"], strict=True)

        encoder_path = polar_config["predictor"]["embedding_model_path"]
        tokenizer = AutoTokenizer.from_pretrained(
            encoder_path, padding_side="left", local_files_only=True
        )
        encoder = FrozenHFTokenEncoder(encoder_path, dtype=torch.bfloat16).to(device).eval()
        polar_predictor = load_checkpoint_predictor(
            polar_config,
            Path(config["checkpoints"]["polar"]["path"]),
            config["parent_configs"]["polar"]["sha256"],
            encoder.output_dim,
            device,
        )

        shard_root = Path(config["extraction"]["feature_shard_root"])
        shard_path = shard_root / f"state_features_rank_{rank:02d}.pt"
        if rank == 0:
            shard_root.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        if shard_path.is_file():
            validate_shard(
                shard_path, config_sha=config_sha, rank=rank, world_size=world_size
            )
            print(json.dumps({"event": "diagnostic_state_shard_reused", "rank": rank}), flush=True)
        else:
            by_cell: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
            for state in state_rows:
                by_cell[(state["split"], state["dataset"], int(state["target_layer"]))].append(state)
            cells = _assigned_cells(state_rows, world_size)[rank]
            polar_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            output = []
            started = time.time()
            completed = 0
            for cell in cells:
                cell_states = sorted(by_cell[cell], key=lambda row: row["state_id"])
                raw_by_state: dict[str, dict[str, torch.Tensor]] = {}
                for state in cell_states:
                    row = source_by_uid[state["uid"]]
                    if state["uid"] not in polar_cache:
                        polar_cache[state["uid"]] = polar_outputs_for_uid(
                            row,
                            feature_index=feature_index,
                            tokenizer=tokenizer,
                            encoder=encoder,
                            predictor=polar_predictor,
                            max_question_tokens=int(polar_config["data"]["max_question_tokens"]),
                            device=device,
                        )
                    raw_by_state[state["state_id"]] = capture_online_state(
                        state,
                        row,
                        sources[state["uid"]],
                        processor=processor,
                        wrapped_model=wrapped_model,
                        router=online_router,
                        device=device,
                    )
                for state in cell_states:
                    layer = int(state["target_layer"])
                    original = online_outputs(
                        online_router, raw_by_state[state["state_id"]], layer=layer, device=device
                    )
                    shuffled = online_outputs(
                        online_router,
                        raw_by_state[state["shuffle_partner_state_id"]],
                        layer=layer,
                        device=device,
                    )
                    polar_representation, polar_logits = polar_cache[state["uid"]]
                    polar_probabilities = torch.softmax(polar_logits[layer], dim=-1)
                    original_probabilities = original["probabilities"]
                    shuffled_probabilities = shuffled["probabilities"]
                    kl = torch.sum(
                        original_probabilities
                        * (
                            torch.log(original_probabilities.clamp_min(1e-12))
                            - torch.log(shuffled_probabilities.clamp_min(1e-12))
                        )
                    )
                    output.append(
                        {
                            "state_id": state["state_id"],
                            "uid": state["uid"],
                            "polar_logits": polar_logits[layer].to(torch.float32),
                            "polar_probabilities": polar_probabilities.to(torch.float32),
                            "polar_feature": polar_representation[layer].to(torch.float32),
                            "online_logits": original["logits"].to(torch.float32),
                            "online_probabilities": original_probabilities.to(torch.float32),
                            "online_z_read": original["z_read"].to(torch.float32),
                            "online_z_write": original["z_write"].to(torch.float32),
                            "online_read_visual": original["read_visual"].to(torch.float32),
                            "online_write_visual": original["write_visual"].to(torch.float32),
                            "online_shuffled_logits": shuffled["logits"].to(torch.float32),
                            "online_shuffled_probabilities": shuffled_probabilities.to(torch.float32),
                            "online_shuffle_kl": float(kl.item()),
                        }
                    )
                    completed += 1
                    if completed <= 3 or completed % 20 == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "diagnostic_state_extracted",
                                    "rank": rank,
                                    "completed": completed,
                                    "assigned": sum(len(by_cell[value]) for value in cells),
                                    "cell": list(cell),
                                    "elapsed_seconds": time.time() - started,
                                    "state_id": state["state_id"],
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                del raw_by_state
                torch.cuda.empty_cache()
            atomic_torch_save(
                shard_path,
                {
                    "schema_version": "four_action_generalization_state_features_v1",
                    "config_sha256": config_sha,
                    "rank": rank,
                    "world_size": world_size,
                    "cells": [list(value) for value in cells],
                    "records": output,
                },
            )
        dist.barrier()
        if rank == 0:
            combined = []
            for shard_rank in range(world_size):
                payload = validate_shard(
                    shard_root / f"state_features_rank_{shard_rank:02d}.pt",
                    config_sha=config_sha,
                    rank=shard_rank,
                    world_size=world_size,
                )
                combined.extend(payload["records"])
            combined.sort(key=lambda row: row["state_id"])
            expected_ids = {row["state_id"] for row in state_rows}
            observed_ids = [row["state_id"] for row in combined]
            if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != expected_ids:
                raise RuntimeError("diagnostic state shards do not exactly cover the manifest")
            output_path = Path(config["reporting"]["state_outputs"])
            atomic_torch_save(
                output_path,
                {
                    "schema_version": "four_action_generalization_state_outputs_v1",
                    "config_sha256": config_sha,
                    "state_manifest_sha256": config["data"]["state_manifest_sha256"],
                    "records": combined,
                },
            )
            summary = {
                "schema_version": "four_action_generalization_state_output_summary_v1",
                "config_sha256": config_sha,
                "state_manifest_sha256": config["data"]["state_manifest_sha256"],
                "records": len(combined),
                "state_outputs": str(output_path),
                "state_outputs_sha256": file_sha256(output_path),
                "world_size": world_size,
                "finite": all(
                    bool(torch.isfinite(row[key]).all())
                    for row in combined
                    for key in (
                        "polar_logits",
                        "polar_feature",
                        "online_logits",
                        "online_z_read",
                        "online_z_write",
                        "online_shuffled_logits",
                    )
                ),
            }
            if not summary["finite"]:
                raise RuntimeError("diagnostic state outputs contain nonfinite values")
            atomic_json(output_path.with_suffix(".summary.json"), summary)
            print(json.dumps({"event": "diagnostic_state_outputs_complete", **summary}, sort_keys=True), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
