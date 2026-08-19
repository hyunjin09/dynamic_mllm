#!/usr/bin/env python3
"""Run only the checksum-bound two-epoch P12 structured-head smoke."""

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

from binary_policy.dataset import BinaryPolicyManifestDataset, make_structured_set_collator
from binary_policy.predictor import (
    BinaryPolarBackbone,
    FrozenHFTokenEncoder,
    SegmentedBinaryPolarBackbone,
)
from binary_policy.training import (
    evaluate_structured_epoch,
    predictor_state_sha256,
    save_checkpoint,
    train_structured_epoch,
)
from experiments.train_binary_polar import (
    checkpoint_key,
    file_sha256,
    seed_worker,
    select_smoke_rows,
    validate_gate,
    validate_readiness_bundle,
)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--readiness-gate", required=True)
    parser.add_argument("--confirm-gates", action="store_true")
    args = parser.parse_args()
    if not args.confirm_gates:
        raise RuntimeError("P12 smoke requires --confirm-gates")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite P12 output: {output_dir}")
    readiness = validate_readiness_bundle(args.readiness_gate, config_path)
    if not readiness["ready_for_bounded_smoke"] or readiness["ready_for_full_training"]:
        raise RuntimeError("readiness gate must authorize only the bounded P12 smoke")
    validated_gates = {name: validate_gate(name, spec) for name, spec in config["gates"].items()}
    if not torch.cuda.is_available():
        raise RuntimeError("P12 training requires a scheduled GPU")

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
    # Reset before each construction: the shared layer encoder must start from
    # exactly the same tensors as the frozen P11 direct-head architecture.
    seed_everything(seed)
    direct_reference = BinaryPolarBackbone(**architecture)
    direct_encoder_sha256 = predictor_state_sha256(direct_reference.encoder)
    del direct_reference
    seed_everything(seed)
    predictor = SegmentedBinaryPolarBackbone(**architecture).to(device)
    structured_encoder_sha256 = predictor_state_sha256(predictor.encoder)
    if direct_encoder_sha256 != structured_encoder_sha256:
        raise RuntimeError("P12 shared encoder initialization differs from P11 architecture")

    manifest_path = Path(config["data"]["manifest"])
    if file_sha256(manifest_path) != config["data"]["manifest_sha256"]:
        raise RuntimeError("P12 predictor manifest checksum mismatch")
    route_cap = int(config["data"]["max_valid_routes_per_sample"])
    train_dataset = BinaryPolicyManifestDataset(manifest_path, "train", max_valid_routes=route_cap)
    validation_dataset = BinaryPolicyManifestDataset(
        manifest_path, "validation", max_valid_routes=route_cap
    )
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
    smoke_path = Path(smoke["manifest"])
    if file_sha256(smoke_path) != smoke["manifest_sha256"]:
        raise RuntimeError("P12 frozen smoke-manifest checksum mismatch")
    frozen_smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if [row["uid"] for row in train_dataset.rows] != frozen_smoke["train_positive_uids"]:
        raise RuntimeError("P12 training identities differ from P11")
    if [row["uid"] for row in validation_dataset.rows] != frozen_smoke["validation_positive_uids"]:
        raise RuntimeError("P12 validation identities differ from P11")

    collator = make_structured_set_collator(
        tokenizer,
        max_length=int(config["data"]["max_question_tokens"]),
        route_weighting=str(config["data"]["route_weighting"]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader_arguments = dict(
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collator,
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_arguments
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_arguments)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    initialization_sha256 = predictor_state_sha256(predictor)
    resolved_config = {
        **config,
        "resolved_objective": "structured_exact_set_nll",
        "resolved_mode": "smoke",
        "resolved_epochs": int(smoke["epochs"]),
        "resolved_train_positive_samples": len(train_dataset),
        "resolved_validation_positive_samples": len(validation_dataset),
        "validated_gates": validated_gates,
        "validated_readiness_gate": readiness,
        "predictor_initialization_sha256": initialization_sha256,
        "shared_encoder_initialization_sha256": structured_encoder_sha256,
        "p11_reference_encoder_initialization_sha256": direct_encoder_sha256,
    }
    (output_dir / "initialization.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "predictor_initialization_sha256": initialization_sha256,
                "shared_encoder_initialization_sha256": structured_encoder_sha256,
                "p11_reference_encoder_initialization_sha256": direct_encoder_sha256,
                "shared_encoder_initialization_matches": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    history = []
    for epoch in range(1, int(smoke["epochs"]) + 1):
        train_metrics = train_structured_epoch(
            predictor,
            frozen_encoder,
            train_loader,
            optimizer,
            device=device,
            amp_dtype=torch.bfloat16,
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
        )
        validation_metrics = evaluate_structured_epoch(
            predictor,
            frozen_encoder,
            validation_loader,
            device=device,
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
    best_record = {
        "selection_rule": (
            "max validation Hit@1; then identical single-candidate hit; then minimum "
            "nearest-valid Hamming; then minimum structured set-NLL; then earliest epoch"
        ),
        "epoch": int(best["epoch"]),
        "checkpoint": str(output_dir / f"checkpoint_epoch_{int(best['epoch']):02d}.pt"),
        "validation": best["validation"],
        "objective": "structured_exact_set_nll",
        "mode": "smoke",
        "predictor_initialization_sha256": initialization_sha256,
        "top5_available": False,
    }
    (output_dir / "best_checkpoint.json").write_text(
        json.dumps(best_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
