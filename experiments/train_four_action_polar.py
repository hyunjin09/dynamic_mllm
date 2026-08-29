#!/usr/bin/env python3
"""Train one frozen Image+Question four-action POLAR objective for ten epochs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import yaml

from binary_policy.predictor import FrozenHFTokenEncoder
from binary_policy.training import predictor_state_sha256
from experiments.train_binary_polar import file_sha256, seed_worker
from four_action_policy.dataset import FourActionManifestDataset
from four_action_policy.evaluation import checkpoint_key
from four_action_policy.feature_cache import load_verified_feature_index
from four_action_policy.multimodal import (
    make_multimodal_duplicated_action_collator,
    make_multimodal_set_collator,
)
from four_action_policy.predictor import FourActionPolarBackbone
from four_action_policy.training import save_epoch_checkpoint, train_epoch, validate_epoch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def optimizer_steps_per_epoch(samples: int, batch: int, accumulation: int) -> int:
    return math.ceil(math.ceil(samples / batch) / accumulation)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_c2c_ablation_contract(
    config: dict[str, Any], config_path: Path, output_dir: Path
) -> None:
    if config.get("protocol_version") != "four_action_polar_c2c_no_allfull_v1":
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("POLAR C2C ablation GPU training requires Slurm")
    plan_path = Path(config["source_plan"])
    parent_path = Path(config["matched_parent_config"])
    if file_sha256(plan_path) != config["source_plan_sha256"]:
        raise RuntimeError("collapse plan checksum mismatch")
    if file_sha256(parent_path) != config["matched_parent_config_sha256"]:
        raise RuntimeError("matched POLAR parent config checksum mismatch")
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    for section in (
        "modality", "base_model", "policy", "predictor", "training",
        "validation", "external_evaluation",
    ):
        if config[section] != parent[section]:
            raise RuntimeError(f"POLAR matched parent section differs: {section}")
    for key in (
        "source_records", "zero_valid_route_exclusions", "route_cap",
        "route_weighting", "max_question_tokens",
    ):
        if config["data"][key] != parent["data"][key]:
            raise RuntimeError(f"POLAR matched parent data field differs: {key}")
    for key in ("root", "source", "feature_width", "dtype", "unpooled"):
        if config["visual_features"][key] != parent["visual_features"][key]:
            raise RuntimeError(f"POLAR matched visual field differs: {key}")
    if config["training"]["objective"] != "exact_set_nll":
        raise RuntimeError("POLAR C2C ablation must use exact-set NLL")
    if int(config["data"]["c2c_exact_allfull_route_empty_exclusions"]) != 35:
        raise RuntimeError("POLAR C2C route-empty exclusion count differs from plan")
    if output_dir.resolve() != Path(config["reporting"]["output_dir"]).resolve():
        raise RuntimeError("POLAR C2C ablation must use its canonical output directory")
    if not config_path.is_file():
        raise RuntimeError("POLAR C2C ablation config is missing")


def render_c2c_ablation_report(
    *, config_path: Path, output_dir: Path, best: dict[str, Any]
) -> str:
    validation = best["validation"]
    overall = validation["overall"]
    by_type = validation.get("by_route_type", {})
    lines = [
        "# POLAR C2C Exact-All-FULL Removal Ablation",
        "",
        f"- Config: `{config_path}`",
        f"- Config SHA-256: `{file_sha256(config_path)}`",
        f"- Output: `{output_dir}`",
        f"- Selected epoch: {best['epoch']}",
        "- Objective: exact-set NLL",
        "- Removed exact all-FULL routes from training C2C only: 3,501",
        "- Excluded route-empty training C2C samples: 35",
        "- Validation labels changed: 0/866",
        "",
        "## Route prediction",
        "",
        f"- Overall top-1 valid-route coverage: {overall['top1_valid_route_coverage']:.6f}",
        f"- Overall top-5 valid-route coverage: {overall['topk_valid_route_coverage']:.6f}",
        f"- Nearest-valid Hamming distance: {overall['nearest_valid_hamming']:.6f}",
        f"- Predicted exact all-FULL fraction: {overall['fraction_top1_all_full']:.6f}",
        f"- Unique predicted routes: {overall['unique_top1_routes']}",
    ]
    for route_type in ("W2C", "C2C"):
        values = by_type.get(route_type)
        if values:
            lines.append(
                f"- {route_type} top-1 valid-route coverage: "
                f"{values['top1_valid_route_coverage']:.6f}"
            )
    lines.extend(
        [
            "",
            "Actual unified-executor routed accuracy is reported separately after",
            "executing the validation-selected checkpoint.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device-ids",
        default="0",
        help="comma-separated CUDA indices allocated to this run",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-training", action="store_true")
    args = parser.parse_args()
    if not args.confirm_training:
        raise RuntimeError("training requires explicit --confirm-training acknowledgement")
    if not torch.cuda.is_available():
        raise RuntimeError("four-action POLAR training requires a GPU")
    device_ids = [int(value) for value in args.device_ids.split(",") if value.strip()]
    if not device_ids or len(device_ids) != len(set(device_ids)):
        raise RuntimeError("device IDs must be a nonempty list of distinct GPUs")
    if max(device_ids) >= torch.cuda.device_count():
        raise RuntimeError(
            f"requested device IDs {device_ids}, but only {torch.cuda.device_count()} are visible"
        )

    config_path = Path(args.config)
    config_sha = file_sha256(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("modality") != "image_question":
        raise RuntimeError("only Image+Question training is authorized")
    preflight_path = Path(args.preflight)
    output_dir = Path(args.output_dir)
    verify_c2c_ablation_contract(config, config_path, output_dir)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("passed") is not True
        or preflight.get("ready_for_training") is not True
        or preflight.get("config_sha256") != config_sha
    ):
        raise RuntimeError("training requires a passed, config-bound full preflight")

    manifest_path = Path(config["data"]["manifest"])
    train_dataset = FourActionManifestDataset(manifest_path, "train")
    validation_dataset = FourActionManifestDataset(manifest_path, "validation")
    if len(train_dataset) != int(config["data"]["train_records"]) or len(validation_dataset) != int(
        config["data"]["validation_records"]
    ):
        raise RuntimeError("training population differs from the frozen config")
    feature_manifest = Path(config["visual_features"]["manifest"])
    feature_sha = preflight["visual_cache"]["manifest_sha256"]
    feature_index = load_verified_feature_index(
        feature_manifest,
        manifest_sha256=feature_sha,
        expected_uids={
            str(row["uid"]) for row in train_dataset.rows + validation_dataset.rows
        },
        expected_feature_width=int(config["visual_features"]["feature_width"]),
    )

    seed = int(config["training"]["seed"])
    seed_everything(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f"cuda:{device_ids[0]}")
    torch.cuda.set_device(device)
    encoder_path = config["predictor"]["embedding_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_path, padding_side="left", local_files_only=True
    )
    encoder_core = FrozenHFTokenEncoder(encoder_path, dtype=torch.bfloat16).to(device).eval()
    encoder_output_dim = encoder_core.output_dim
    seed_everything(seed)
    predictor_core = FourActionPolarBackbone(
        num_layers=int(config["policy"]["num_layers"]),
        input_dim=encoder_output_dim,
        image_dim=int(config["visual_features"]["feature_width"]),
        d_model=int(config["predictor"]["d_model"]),
        num_heads=int(config["predictor"]["num_heads"]),
        num_layer_blocks=int(config["predictor"]["num_layer_blocks"]),
        dropout=float(config["predictor"]["dropout"]),
    ).to(device)
    initialization_sha = predictor_state_sha256(predictor_core)
    encoder = (
        torch.nn.DataParallel(encoder_core, device_ids=device_ids).eval()
        if len(device_ids) > 1
        else encoder_core
    )
    predictor = (
        torch.nn.DataParallel(predictor_core, device_ids=device_ids)
        if len(device_ids) > 1
        else predictor_core
    )
    route_weighting = str(config["data"]["route_weighting"])
    common_collator = {
        "max_length": int(config["data"]["max_question_tokens"]),
        "route_weighting": route_weighting,
    }
    validation_collator = make_multimodal_set_collator(
        tokenizer, feature_index, **common_collator
    )
    objective = str(config["training"]["objective"])
    train_collator = (
        validation_collator
        if objective == "exact_set_nll"
        else make_multimodal_duplicated_action_collator(
            tokenizer, feature_index, **common_collator
        )
    )
    physical_batch = int(config["training"]["physical_batch_size"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    if physical_batch * accumulation != int(config["training"]["effective_batch_size"]):
        raise RuntimeError("effective batch size mismatch")
    common_loader = {
        "batch_size": physical_batch,
        "num_workers": int(config["training"]["num_workers"]),
        "worker_init_fn": seed_worker,
        "pin_memory": True,
        "persistent_workers": int(config["training"]["num_workers"]) > 0,
        "prefetch_factor": 1,
    }
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        collate_fn=validation_collator,
        **common_loader,
    )
    optimizer = torch.optim.AdamW(
        predictor_core.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    steps_per_epoch = optimizer_steps_per_epoch(
        len(train_dataset), physical_batch, accumulation
    )
    total_steps = steps_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config["training"]["warmup_steps"]),
        num_training_steps=total_steps,
    )
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite training output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_assets = {
        "config": str(config_path),
        "config_sha256": config_sha,
        "preflight": str(preflight_path),
        "preflight_sha256": file_sha256(preflight_path),
        "training_manifest": str(manifest_path),
        "training_manifest_sha256": file_sha256(manifest_path),
        "visual_feature_manifest": str(feature_manifest),
        "visual_feature_manifest_sha256": feature_sha,
        "visual_cache_audit_sha256": preflight["visual_cache"]["audit_sha256"],
        "embedding_model_path": encoder_path,
    }
    initialization = {
        "schema_version": "four_action_polar_initialization_v1",
        "objective": objective,
        "modality": "image_question",
        "seed": seed,
        "predictor_initialization_sha256": initialization_sha,
        "data_parallel_device_ids": device_ids,
        "train_records": len(train_dataset),
        "validation_records": len(validation_dataset),
        "physical_batch_size": physical_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": physical_batch * accumulation,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "total_scheduler_steps": total_steps,
        "resolved_assets": resolved_assets,
    }
    initialization_path = output_dir / "initialization.json"
    history_path = output_dir / "history.json"
    history = []
    checkpoints = []
    global_step = 0
    start_epoch = 1
    if args.resume:
        if (output_dir / "training_summary.json").exists():
            raise FileExistsError(f"training output is already complete: {output_dir}")
        if not initialization_path.is_file() or not history_path.is_file():
            raise RuntimeError("resume requires initialization.json and history.json")
        frozen_initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
        if (
            frozen_initialization.get("predictor_initialization_sha256") != initialization_sha
            or frozen_initialization.get("resolved_assets") != resolved_assets
            or frozen_initialization.get("data_parallel_device_ids") != device_ids
        ):
            raise RuntimeError("resume initialization or asset contract mismatch")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not history or [int(row["epoch"]) for row in history] != list(
            range(1, len(history) + 1)
        ):
            raise RuntimeError("resume history is empty or non-contiguous")
        for epoch in range(1, len(history) + 1):
            metadata_path = output_dir / f"epoch_{epoch:02d}/metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            checkpoint_path = Path(metadata["checkpoint"])
            if file_sha256(checkpoint_path) != metadata["checkpoint_sha256"]:
                raise RuntimeError(f"resume checkpoint checksum mismatch at epoch {epoch}")
            checkpoints.append(metadata)
        payload = torch.load(
            Path(checkpoints[-1]["checkpoint"]), map_location="cpu", weights_only=False
        )
        if (
            payload.get("config_sha256") != config_sha
            or payload.get("resolved_assets") != resolved_assets
            or int(payload.get("epoch", -1)) != len(history)
        ):
            raise RuntimeError("resume checkpoint contract mismatch")
        predictor_core.load_state_dict(payload["predictor"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        global_step = int(payload["global_step"])
        start_epoch = len(history) + 1
    else:
        initialization_path.write_text(
            json.dumps(initialization, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    for epoch in range(start_epoch, epochs + 1):
        seed_everything(seed + epoch)
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
            collate_fn=train_collator,
            **common_loader,
        )
        train_metrics, global_step = train_epoch(
            predictor,
            encoder,
            train_loader,
            optimizer,
            scheduler,
            device=device,
            objective=objective,
            accumulation_steps=accumulation,
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            duplicated_route_microbatch_size=int(
                config["training"]["duplicated_route_microbatch_size"]
            ),
            amp_dtype=torch.bfloat16,
            epoch=epoch,
            global_step=global_step,
            progress_first_batches=3,
            progress_every_batches=10,
        )
        validation = validate_epoch(
            predictor,
            encoder,
            validation_loader,
            device=device,
            objective=objective,
            top_k=int(config["validation"]["top_k"]),
            amp_dtype=torch.bfloat16,
        )
        if validation["overall"]["examples"] != len(validation_dataset):
            raise RuntimeError("epoch validation was not complete")
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation,
        }
        history.append(row)
        checkpoints.append(
            save_epoch_checkpoint(
                output_dir,
                predictor_core,
                optimizer,
                scheduler,
                epoch=epoch,
                global_step=global_step,
                config=config,
                config_sha256=config_sha,
                resolved_assets=resolved_assets,
                metrics=row,
            )
        )
        history_path.write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if config.get("reporting", {}).get("history"):
            write_jsonl(Path(config["reporting"]["history"]), history)
        print(json.dumps(row, sort_keys=True), flush=True)
    if global_step != total_steps:
        raise RuntimeError(f"optimizer-step mismatch: {global_step} != {total_steps}")
    best = max(history, key=checkpoint_key)
    best_checkpoint = checkpoints[int(best["epoch"]) - 1]
    selection = {
        "schema_version": "four_action_polar_checkpoint_selection_v1",
        "selected_before_external_evaluation": True,
        "config": str(config_path),
        "config_sha256": config_sha,
        "objective": objective,
        "ordering": config["validation"]["checkpoint_order"],
        "best_epoch": int(best["epoch"]),
        "best_checkpoint": best_checkpoint["checkpoint"],
        "best_checkpoint_sha256": best_checkpoint["checkpoint_sha256"],
        "validation": best["validation"],
    }
    (output_dir / "best_checkpoint.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": "four_action_polar_training_v1",
        "passed": True,
        "objective": objective,
        "modality": "image_question",
        "epochs_completed": len(history),
        "global_steps": global_step,
        "best_epoch": int(best["epoch"]),
        "checkpoints": [
            {key: value for key, value in checkpoint.items() if key != "metrics"}
            for checkpoint in checkpoints
        ],
        "external_evaluation_started": False,
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_path.with_suffix(".json.sha256").write_text(
        f"{file_sha256(summary_path)}  {summary_path.name}\n", encoding="utf-8"
    )
    if config.get("reporting", {}).get("report"):
        report_path = Path(config["reporting"]["report"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            render_c2c_ablation_report(
                config_path=config_path, output_dir=output_dir, best=best
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
