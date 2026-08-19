#!/usr/bin/env python3
"""Run one checksum-bound P13 modality smoke with the direct exact-set head."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
import yaml
from transformers import AutoTokenizer

from binary_policy.dataset import BinaryPolicyManifestDataset
from binary_policy.multimodal import MODALITIES, make_multimodal_set_collator
from binary_policy.predictor import BinaryPolarBackbone, FrozenHFTokenEncoder
from binary_policy.training import (
    evaluate_multimodal_epoch,
    predictor_state_sha256,
    save_checkpoint,
    train_multimodal_epoch,
)
from experiments.train_binary_polar import (
    checkpoint_key,
    file_sha256,
    seed_worker,
    select_smoke_rows,
    validate_gate,
    validate_readiness_bundle,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--modality", choices=MODALITIES, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--readiness-gate", required=True)
    parser.add_argument("--confirm-gates", action="store_true")
    args = parser.parse_args()
    if not args.confirm_gates:
        raise RuntimeError("P13 smoke requires --confirm-gates")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite P13 output: {output_dir}")
    readiness = validate_readiness_bundle(args.readiness_gate, config_path)
    if not readiness["ready_for_bounded_smoke"] or readiness["ready_for_full_training"]:
        raise RuntimeError("P13 readiness must authorize only its bounded smoke")
    validated_gates = {name: validate_gate(name, spec) for name, spec in config["gates"].items()}
    if not torch.cuda.is_available():
        raise RuntimeError("P13 training requires a scheduled GPU")
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    encoder_path = config["predictor"]["embedding_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, padding_side="left", local_files_only=True)
    frozen_encoder = FrozenHFTokenEncoder(encoder_path, dtype=torch.bfloat16).to(device)
    architecture = dict(
        num_layers=int(config["policy"]["num_layers"]),
        input_dim=frozen_encoder.output_dim,
        d_model=int(config["predictor"]["d_model"]),
        num_heads=int(config["predictor"]["num_heads"]),
        num_layer_blocks=int(config["predictor"]["num_layer_blocks"]),
        dropout=float(config["predictor"]["dropout"]),
    )
    seed_everything(seed)
    p11_reference = BinaryPolarBackbone(**architecture)
    p11_shared_state = p11_reference.state_dict()
    seed_everything(seed)
    predictor = BinaryPolarBackbone(
        **architecture, image_dim=int(config["p13"]["visual_feature_width"])
    ).to(device)
    p13_state = predictor.state_dict()
    mismatches = [
        name for name, tensor in p11_shared_state.items() if not torch.equal(tensor, p13_state[name].cpu())
    ]
    if mismatches:
        raise RuntimeError(f"P13 common initialization differs from P11: {mismatches[:3]}")
    p11_shared_sha256 = predictor_state_sha256(p11_reference)
    del p11_reference

    feature_manifest_path = Path(config["p13"]["feature_manifest"])
    if file_sha256(feature_manifest_path) != config["p13"]["feature_manifest_sha256"]:
        raise RuntimeError("P13 feature manifest checksum mismatch")
    feature_index = {row["uid"]: row for row in read_jsonl(feature_manifest_path)}
    manifest_path = Path(config["data"]["manifest"])
    if file_sha256(manifest_path) != config["data"]["manifest_sha256"]:
        raise RuntimeError("P13 predictor manifest checksum mismatch")
    route_cap = int(config["data"]["max_valid_routes_per_sample"])
    train_dataset = BinaryPolicyManifestDataset(manifest_path, "train", max_valid_routes=route_cap)
    validation_dataset = BinaryPolicyManifestDataset(manifest_path, "validation", max_valid_routes=route_cap)
    smoke = config["smoke"]
    train_dataset.rows = select_smoke_rows(
        train_dataset,
        per_dataset=int(smoke["train_positive_per_dataset"]),
        seed=int(smoke["selection_seed"]),
    )
    validation_dataset.rows = select_smoke_rows(
        validation_dataset,
        per_dataset=int(smoke["validation_positive_per_dataset"]),
        seed=int(smoke["selection_seed"]),
    )
    frozen_smoke = json.loads(Path(smoke["manifest"]).read_text(encoding="utf-8"))
    if [row["uid"] for row in train_dataset.rows] != frozen_smoke["train_positive_uids"]:
        raise RuntimeError("P13 train identities differ from P11")
    if [row["uid"] for row in validation_dataset.rows] != frozen_smoke["validation_positive_uids"]:
        raise RuntimeError("P13 validation identities differ from P11")
    if set(row["uid"] for row in train_dataset.rows + validation_dataset.rows) - feature_index.keys():
        raise RuntimeError("P13 feature cache is incomplete for train/validation")

    collator = make_multimodal_set_collator(
        tokenizer,
        feature_index,
        max_length=int(config["data"]["max_question_tokens"]),
        route_weighting=str(config["data"]["route_weighting"]),
    )
    loader_args = dict(
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collator,
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **loader_args,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_args)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    output_dir.mkdir(parents=True)
    initialization_sha256 = predictor_state_sha256(predictor)
    resolved_config = {
        **config,
        "resolved_modality": args.modality,
        "resolved_mode": "smoke",
        "resolved_epochs": int(smoke["epochs"]),
        "resolved_train_positive_samples": len(train_dataset),
        "resolved_validation_positive_samples": len(validation_dataset),
        "validated_gates": validated_gates,
        "validated_readiness_gate": readiness,
        "predictor_initialization_sha256": initialization_sha256,
        "p11_shared_initialization_sha256": p11_shared_sha256,
        "p11_shared_initialization_matches": True,
    }
    (output_dir / "initialization.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "modality": args.modality,
                "predictor_initialization_sha256": initialization_sha256,
                "p11_shared_initialization_sha256": p11_shared_sha256,
                "p11_shared_initialization_matches": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    history = []
    for epoch in range(1, int(smoke["epochs"]) + 1):
        train_metrics = train_multimodal_epoch(
            predictor,
            frozen_encoder,
            train_loader,
            optimizer,
            modality=args.modality,
            device=device,
            amp_dtype=torch.bfloat16,
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
        )
        validation_metrics = evaluate_multimodal_epoch(
            predictor,
            frozen_encoder,
            validation_loader,
            modality=args.modality,
            device=device,
            top_k=int(config["evaluation"]["top_k"]),
            amp_dtype=torch.bfloat16,
        )
        row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(row)
        save_checkpoint(
            output_dir / f"checkpoint_epoch_{epoch:02d}.pt",
            predictor,
            optimizer,
            epoch=epoch,
            config=resolved_config,
            metrics=row,
        )
        print(json.dumps(row, sort_keys=True), flush=True)
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    best = max(history, key=checkpoint_key)
    (output_dir / "best_checkpoint.json").write_text(
        json.dumps(
            {
                "selection_rule": (
                    "max validation Hit@1; then Hit@5; then minimum nearest-valid Hamming; "
                    "then minimum exact-set NLL; then earliest epoch"
                ),
                "epoch": int(best["epoch"]),
                "checkpoint": str(output_dir / f"checkpoint_epoch_{int(best['epoch']):02d}.pt"),
                "validation": best["validation"],
                "objective": "exact_set_nll",
                "modality": args.modality,
                "mode": "smoke",
                "predictor_initialization_sha256": initialization_sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
