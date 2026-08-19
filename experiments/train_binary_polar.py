#!/usr/bin/env python3
"""Train only the lightweight binary policy predictor.

This entrypoint is implemented but must be run through the Slurm scheduler and
is not launched by the planning/pre-training task that introduced it.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
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

from binary_policy.dataset import (
    BinaryPolicyManifestDataset,
    make_duplicated_path_collator,
    make_set_collator,
)
from binary_policy.predictor import BinaryPolarBackbone, FrozenHFTokenEncoder
from binary_policy.training import evaluate_epoch, predictor_state_sha256, save_checkpoint, train_epoch


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gate(name: str, specification) -> dict:
    if isinstance(specification, str):
        path = Path(specification)
        expected_sha256 = None
    elif isinstance(specification, dict):
        path = Path(specification["path"])
        expected_sha256 = specification.get("sha256")
    else:
        raise TypeError(f"invalid {name} gate specification")
    if not path.is_file():
        raise FileNotFoundError(f"required {name} gate is missing: {path}")
    observed_sha256 = file_sha256(path)
    if expected_sha256 and observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"required {name} gate checksum mismatch: expected {expected_sha256}, got {observed_sha256}"
        )
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("passed") is not True:
        raise RuntimeError(f"required {name} gate did not pass: {path}")
    return {"path": str(path), "sha256": observed_sha256}


def validate_readiness_bundle(path_value: str | Path, config_path: Path) -> dict:
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"P10 readiness gate is missing: {path}")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("passed") is not True:
        raise RuntimeError("P10 readiness gate did not pass")
    if bundle.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError("P10 readiness gate is not bound to the selected config")
    for name, artifact in bundle.get("artifacts", {}).items():
        artifact_path = Path(artifact["path"])
        if file_sha256(artifact_path) != artifact["sha256"]:
            raise RuntimeError(f"P10 readiness artifact checksum mismatch: {name}")
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if payload.get("passed") is not True and name != "smoke_manifest":
            raise RuntimeError(f"P10 readiness artifact did not pass: {name}")
    for source_path_value, expected_sha256 in bundle.get("source_sha256", {}).items():
        source_path = Path(source_path_value)
        if file_sha256(source_path) != expected_sha256:
            raise RuntimeError(f"P10 readiness source checksum mismatch: {source_path}")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "ready_for_bounded_smoke": bundle.get("ready_for_bounded_smoke") is True,
        "ready_for_full_training": bundle.get("ready_for_full_training") is True,
    }


def select_smoke_rows(dataset, *, per_dataset: int, seed: int):
    if per_dataset < 1:
        raise ValueError("smoke per-dataset count must be positive")
    selected = []
    for benchmark in ("gqa", "textvqa", "chartqa"):
        candidates = [row for row in dataset.rows if row["benchmark"] == benchmark]
        candidates.sort(key=lambda row: sha256(f"{seed}:{row['uid']}".encode()).hexdigest())
        if len(candidates) < per_dataset:
            raise RuntimeError(f"insufficient {benchmark} positive rows for the frozen smoke")
        selected.extend(candidates[:per_dataset])
    selected.sort(key=lambda row: row["uid"])
    return selected


def checkpoint_key(row: dict) -> tuple:
    """Common objective-independent validation ordering; larger is better."""
    metrics = row["validation"]
    return (
        float(metrics["top1_valid_route_coverage"]),
        float(metrics["topk_valid_route_coverage"]),
        -float(metrics["nearest_valid_hamming"]),
        -float(metrics["set_nll"]),
        -int(row["epoch"]),
    )


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objective", required=True, choices=("duplicated_bce", "exact_set_nll"))
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--readiness-gate", required=True)
    parser.add_argument(
        "--confirm-gates",
        action="store_true",
        help="required acknowledgement after all frozen pre-training gate files pass",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite training output directory: {output_dir}")
    if not args.confirm_gates:
        raise RuntimeError("training is disabled until --confirm-gates is supplied after explicit approval")
    readiness_gate = validate_readiness_bundle(args.readiness_gate, config_path)
    if args.mode == "smoke" and not readiness_gate["ready_for_bounded_smoke"]:
        raise RuntimeError("P10 readiness gate does not authorize the bounded smoke")
    if args.mode == "full" and not readiness_gate["ready_for_full_training"]:
        raise RuntimeError("full training remains blocked until a post-smoke readiness gate passes")
    validated_gates = {
        gate_name: validate_gate(gate_name, gate_spec)
        for gate_name, gate_spec in config["gates"].items()
    }
    if not torch.cuda.is_available():
        raise RuntimeError("training requires a scheduled GPU allocation")
    device = torch.device("cuda")
    torch.manual_seed(int(config["training"]["seed"]))
    torch.cuda.manual_seed_all(int(config["training"]["seed"]))
    random.seed(int(config["training"]["seed"]))
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    encoder_path = config["predictor"]["embedding_model_path"]
    manifest_path = Path(config["data"]["manifest"])
    expected_manifest_sha256 = config["data"].get("manifest_sha256")
    if expected_manifest_sha256 and file_sha256(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("binary predictor manifest checksum mismatch")
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, padding_side="left", local_files_only=True)
    frozen_encoder = FrozenHFTokenEncoder(encoder_path, dtype=torch.bfloat16).to(device)
    predictor = BinaryPolarBackbone(
        num_layers=int(config["policy"]["num_layers"]),
        input_dim=frozen_encoder.output_dim,
        d_model=int(config["predictor"]["d_model"]),
        num_heads=int(config["predictor"]["num_heads"]),
        num_layer_blocks=int(config["predictor"]["num_layer_blocks"]),
        dropout=float(config["predictor"]["dropout"]),
    ).to(device)
    manifest = config["data"]["manifest"]
    route_weighting = str(config["data"].get("route_weighting", "equal"))
    collator_factory = make_set_collator if args.objective == "exact_set_nll" else make_duplicated_path_collator
    collator = collator_factory(
        tokenizer,
        max_length=int(config["data"]["max_question_tokens"]),
        route_weighting=route_weighting,
    )
    validation_collator = make_set_collator(
        tokenizer,
        max_length=int(config["data"]["max_question_tokens"]),
        route_weighting=route_weighting,
    )
    route_cap = int(config["data"]["max_valid_routes_per_sample"])
    train_dataset = BinaryPolicyManifestDataset(manifest, "train", max_valid_routes=route_cap)
    validation_dataset = BinaryPolicyManifestDataset(manifest, "validation", max_valid_routes=route_cap)
    epochs = int(config["training"]["epochs"])
    if args.mode == "smoke":
        smoke = config.get("smoke")
        if not isinstance(smoke, dict):
            raise RuntimeError("smoke mode requires a frozen smoke section in the config")
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
        smoke_manifest_path = Path(smoke["manifest"])
        if file_sha256(smoke_manifest_path) != smoke["manifest_sha256"]:
            raise RuntimeError("frozen P10 smoke-manifest checksum mismatch")
        frozen_smoke = json.loads(smoke_manifest_path.read_text(encoding="utf-8"))
        if [row["uid"] for row in train_dataset.rows] != frozen_smoke["train_positive_uids"]:
            raise RuntimeError("resolved training-smoke identities differ from the frozen manifest")
        if [row["uid"] for row in validation_dataset.rows] != frozen_smoke["validation_positive_uids"]:
            raise RuntimeError("resolved validation-smoke identities differ from the frozen manifest")
        epochs = int(smoke["epochs"])
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(config["training"]["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collator,
        generator=loader_generator,
        worker_init_fn=seed_worker,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=validation_collator,
        worker_init_fn=seed_worker,
    )
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    initialization_sha256 = predictor_state_sha256(predictor)
    history = []
    config = {
        **config,
        "resolved_objective": args.objective,
        "resolved_mode": args.mode,
        "resolved_epochs": epochs,
        "validated_gates": validated_gates,
        "validated_readiness_gate": readiness_gate,
        "resolved_train_positive_samples": len(train_dataset),
        "resolved_validation_positive_samples": len(validation_dataset),
        "predictor_initialization_sha256": initialization_sha256,
    }
    (output_dir / "initialization.json").write_text(
        json.dumps(
            {
                "seed": int(config["training"]["seed"]),
                "predictor_initialization_sha256": initialization_sha256,
                "objective": args.objective,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            predictor,
            frozen_encoder,
            train_loader,
            optimizer,
            device=device,
            objective=args.objective,
            amp_dtype=torch.bfloat16,
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            duplicated_route_microbatch_size=int(
                config["training"].get("duplicated_route_microbatch_size", config["training"]["batch_size"])
            ),
        )
        validation_metrics = evaluate_epoch(
            predictor,
            frozen_encoder,
            validation_loader,
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
            config=config,
            metrics=row,
        )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    best = max(history, key=checkpoint_key)
    best_record = {
        "selection_rule": (
            "max validation Valid-Set Hit@1; then Hit@5; then minimum nearest-valid Hamming; "
            "then minimum common exact-set NLL; then earliest epoch"
        ),
        "epoch": int(best["epoch"]),
        "checkpoint": str(output_dir / f"checkpoint_epoch_{int(best['epoch']):02d}.pt"),
        "validation": best["validation"],
        "objective": args.objective,
        "mode": args.mode,
        "predictor_initialization_sha256": initialization_sha256,
    }
    (output_dir / "best_checkpoint.json").write_text(
        json.dumps(best_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
