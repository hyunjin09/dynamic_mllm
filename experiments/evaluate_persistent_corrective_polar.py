#!/usr/bin/env python3
"""Execute every persistent-POLAR checkpoint on the frozen internal subset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
import yaml

from binary_policy.executor.inputs import build_binary_inputs
from binary_policy.predictor import FrozenHFTokenEncoder
from experiments.evaluate_four_action_polar_external import execute_actions, predict_actions
from experiments.evaluate_four_action_polar_internal import augment_execution_summary
from experiments.train_binary_polar import file_sha256
from experiments.train_four_action_online_router import write_json, write_jsonl
from four_action_online_router.data import load_jsonl, load_source_metadata, load_verified_manifest
from four_action_online_router.metrics import mandatory_boundary_metrics
from four_action_policy.feature_cache import load_verified_feature_index
from four_action_policy.persistent import select_behavioral_checkpoint
from four_action_policy.predictor import FourActionPolarBackbone
from label_regeneration.runtime import (
    build_native_processor_inputs,
    configure_determinism,
    load_frozen_model,
)


def distributed_context(expected_world_size: int) -> tuple[int, int, int, torch.device]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if (
        world_size != expected_world_size
        or world_size != 4
        or not torch.cuda.is_available()
        or local_rank >= torch.cuda.device_count()
    ):
        raise RuntimeError("persistent POLAR execution requires four direct torchrun GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    return rank, world_size, local_rank, device


def load_checkpoint_predictor(
    config: dict[str, Any],
    checkpoint_path: Path,
    config_sha256: str,
    input_dim: int,
    device: torch.device,
) -> FourActionPolarBackbone:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("config_sha256") != config_sha256:
        raise RuntimeError("POLAR checkpoint belongs to another frozen config")
    predictor = FourActionPolarBackbone(
        num_layers=int(config["policy"]["num_layers"]),
        input_dim=input_dim,
        image_dim=int(config["visual_features"]["feature_width"]),
        d_model=int(config["predictor"]["d_model"]),
        num_heads=int(config["predictor"]["num_heads"]),
        num_layer_blocks=int(config["predictor"]["num_layer_blocks"]),
        dropout=float(config["predictor"]["dropout"]),
    ).to(device).eval()
    predictor.load_state_dict(payload["predictor"], strict=True)
    return predictor


def render_report(
    *, selection: dict[str, Any], history: list[dict[str, Any]], config_path: Path
) -> str:
    selected_epoch = selection["selected_epoch"]
    lines = [
        "# POLAR Persistent Corrective Supervision",
        "",
        f"- Config: `{config_path}`",
        f"- Config SHA-256: `{file_sha256(config_path)}`",
        f"- Executed checkpoints: {len(history)}",
        f"- C2C preservation threshold: {selection['c2c_preservation_threshold']:.2f}",
        f"- Selected epoch: {selected_epoch}",
        f"- Eligible epochs: {selection['eligible_epochs']}",
        f"- Pareto frontier epochs: {selection['pareto_frontier_epochs']}",
        "",
        "## Per-checkpoint behavior",
        "",
        "| Epoch | W2C rescue | C2C preservation | Net change | Boundary Valid@1 | Boundary non-FULL |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in history:
        execution = row["execution"]
        boundary = row["boundary"]
        lines.append(
            f"| {row['epoch']} | {execution['w2c_rescue_rate']:.6f} | "
            f"{execution['c2c_preservation_rate']:.6f} | "
            f"{execution['w2c_rescues'] - execution['c2c_regressions']} | "
            f"{boundary['valid_action_at_1']:.6f} | {boundary['nonfull_recall']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Checkpoint selection used only free-running routed execution. No external",
            "evaluation was run.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()
    config_path = Path(args.config)
    config_sha = file_sha256(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_version") != "four_action_persistent_polar_v1":
        raise RuntimeError("evaluator requires the persistent POLAR config")
    sys.path.insert(0, str(Path(config["external_evaluation"]["protocol"]) / "code"))
    rank, world_size, local_rank, device = distributed_context(args.world_size)
    try:
        configure_determinism(int(config["training"]["seed"]))
        rows = load_verified_manifest(config["data"]["manifest"], config["data"]["manifest_sha256"])
        validation_rows = [row for row in rows if row["split"] == "validation"]
        if len(validation_rows) != int(config["validation"]["expected_records"]):
            raise RuntimeError("POLAR execution validation population mismatch")
        boundary_path = Path(config["data"]["boundary_manifest"])
        if file_sha256(boundary_path) != config["data"]["boundary_manifest_sha256"]:
            raise RuntimeError("POLAR execution boundary checksum mismatch")
        boundaries = {
            row["uid"]: row
            for row in load_jsonl(boundary_path)
            if row["uid"] in {item["uid"] for item in validation_rows}
        }
        if set(boundaries) != {
            row["uid"] for row in validation_rows if row["route_type"] == "W2C"
        }:
            raise RuntimeError("POLAR execution boundaries do not cover validation W2C")
        sources = load_source_metadata(
            config["data"]["source_manifest"],
            config["data"]["source_manifest_sha256"],
            {str(row["uid"]) for row in validation_rows},
        )
        feature_index = load_verified_feature_index(
            config["visual_features"]["manifest"],
            manifest_sha256=config["visual_features"]["manifest_sha256"],
            expected_uids={str(row["uid"]) for row in rows},
            expected_feature_width=int(config["visual_features"]["feature_width"]),
            verify_tensors=False,
        )
        processor, base_model, wrapped_model, _ = load_frozen_model(
            config["base_model"]["path"], config["base_model"]["revision"], local_rank
        )
        base_model.requires_grad_(False).eval()
        encoder_path = config["predictor"]["embedding_model_path"]
        tokenizer = AutoTokenizer.from_pretrained(
            encoder_path, padding_side="left", local_files_only=True
        )
        encoder = FrozenHFTokenEncoder(encoder_path, dtype=torch.bfloat16).to(device).eval()

        training_dir = Path(args.training_dir)
        metadata_paths = sorted(training_dir.glob("epoch_[0-9][0-9]/metadata.json"))
        if len(metadata_paths) != int(config["training"]["epochs"]):
            raise RuntimeError("POLAR execution requires every frozen checkpoint")
        output_dir = Path(args.output_dir)
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        history = []
        all_outputs = []
        checkpoint_metadata = {}
        for expected_epoch, metadata_path in enumerate(metadata_paths, start=1):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if int(metadata["epoch"]) != expected_epoch:
                raise RuntimeError("POLAR checkpoint epochs are non-contiguous")
            checkpoint = Path(metadata["checkpoint"])
            if file_sha256(checkpoint) != metadata["checkpoint_sha256"]:
                raise RuntimeError(f"POLAR checkpoint checksum mismatch at epoch {expected_epoch}")
            predictor = load_checkpoint_predictor(
                config, checkpoint, config_sha, encoder.output_dim, device
            )
            local_rows = []
            for position in range(rank, len(validation_rows), world_size):
                row = validation_rows[position]
                source = sources[row["uid"]]
                sample = {**row, **source, "local_image_path": row["image_path"]}
                inputs, _ = build_native_processor_inputs(processor, sample, device)
                prepared = build_binary_inputs(wrapped_model, inputs)
                visual = torch.load(
                    feature_index[row["uid"]]["path"], map_location="cpu", weights_only=True
                )
                prediction = predict_actions(
                    row={**row, "predictor_text": row["question"]},
                    visual_rows=visual,
                    tokenizer=tokenizer,
                    encoder=encoder,
                    predictor=predictor,
                    max_question_tokens=int(config["data"]["max_question_tokens"]),
                    device=device,
                )
                actions = tuple(prediction.pop("actions"))
                execution = execute_actions(
                    wrapped_model=wrapped_model,
                    processor=processor,
                    inputs=inputs,
                    prepared=prepared,
                    actions=actions,
                    row=sample,
                    eos_token_ids=list(config["external_evaluation"]["eos_token_ids"]),
                    repetition_penalty=float(config["external_evaluation"]["repetition_penalty"]),
                )
                output = {
                    "epoch": expected_epoch,
                    "uid": row["uid"],
                    "dataset": row["dataset"],
                    "route_type": row["route_type"],
                    **prediction,
                    "actions": list(actions),
                    **execution,
                }
                if row["route_type"] == "W2C":
                    boundary = boundaries[row["uid"]]
                    layer = int(boundary["boundary_layer"])
                    output.update(
                        {
                            "boundary_layer": layer,
                            "valid_nonfull_actions": boundary["valid_nonfull_actions"],
                            "singleton": boundary["singleton"],
                            "predicted_boundary_action": actions[layer],
                        }
                    )
                local_rows.append(output)
            gathered: list[Any] = [None] * world_size
            dist.all_gather_object(gathered, local_rows)
            if rank == 0:
                combined = [row for part in gathered for row in part]
                combined.sort(key=lambda row: row["uid"])
                if len(combined) != len(validation_rows) or len(
                    {row["uid"] for row in combined}
                ) != len(validation_rows):
                    raise RuntimeError("POLAR execution did not cover validation exactly")
                epoch_dir = output_dir / f"epoch_{expected_epoch:02d}"
                epoch_dir.mkdir(parents=False, exist_ok=False)
                write_jsonl(epoch_dir / "validation_outputs.jsonl", combined)
                execution_summary = augment_execution_summary(combined)
                boundary_summary = mandatory_boundary_metrics(
                    [row for row in combined if row["route_type"] == "W2C"],
                    num_layers=int(config["policy"]["num_layers"]),
                )
                history_row = {
                    "epoch": expected_epoch,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
                    "execution": execution_summary,
                    "boundary": boundary_summary,
                }
                history.append(history_row)
                all_outputs.extend(combined)
                checkpoint_metadata[expected_epoch] = metadata
                write_json(epoch_dir / "summary.json", history_row)
                print(json.dumps({"event": "persistent_polar_epoch_execution", **history_row}, sort_keys=True), flush=True)
            del predictor
            torch.cuda.empty_cache()
            dist.barrier()
        if rank == 0:
            selection = select_behavioral_checkpoint(
                history,
                c2c_threshold=float(config["validation"]["c2c_preservation_threshold"]),
            )
            selected_epoch = selection["selected_epoch"]
            selected_metadata = (
                checkpoint_metadata[selected_epoch] if selected_epoch is not None else None
            )
            selection_payload = {
                "schema_version": "persistent_corrective_polar_selection_v1",
                "selected_before_external_evaluation": True,
                "config": str(config_path),
                "config_sha256": config_sha,
                **selection,
                "best_epoch": selected_epoch,
                "best_checkpoint": (
                    selected_metadata["checkpoint"] if selected_metadata else None
                ),
                "best_checkpoint_sha256": (
                    selected_metadata["checkpoint_sha256"] if selected_metadata else None
                ),
                "external_evaluation_started": False,
            }
            write_json(training_dir / "best_checkpoint.json", selection_payload)
            write_json(output_dir / "execution_history.json", history)
            write_jsonl(Path(config["reporting"]["execution"]), all_outputs)
            Path(config["reporting"]["report"]).write_text(
                render_report(selection=selection, history=history, config_path=config_path),
                encoding="utf-8",
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
