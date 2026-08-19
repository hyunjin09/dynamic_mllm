#!/usr/bin/env python3
"""Train P11 global or dataset-conditioned bias-only route baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from binary_policy.dataset import route_weights
from binary_policy.evaluation import batch_offline_metrics
from binary_policy.losses import multi_valid_set_nll, polar_path_bce_per_path
from binary_policy.predictor import BiasOnlyBinaryPredictor
from binary_policy.training import predictor_state_sha256
from experiments.train_binary_polar import checkpoint_key, file_sha256, validate_gate


DATASET_IDS = {"gqa": 0, "textvqa": 1, "chartqa": 2}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_rows(manifest_path: Path, p11_smoke_path: Path) -> tuple[list[dict], list[dict]]:
    smoke = json.loads(p11_smoke_path.read_text(encoding="utf-8"))
    by_uid = {row["uid"]: row for row in read_jsonl(manifest_path)}
    train = [by_uid[uid] for uid in smoke["train_positive_uids"]]
    validation = [by_uid[uid] for uid in smoke["validation_positive_uids"]]
    if any(not row.get("valid_routes") for row in train + validation):
        raise RuntimeError("bias-only smoke cannot include an empty valid set")
    return train, validation


def make_set_batch(rows: list[dict], *, route_weighting: str, conditioning: str) -> dict:
    maximum = max(len(row["valid_routes"]) for row in rows)
    width = len(rows[0]["valid_routes"][0]["mask"])
    masks = torch.zeros(len(rows), maximum, width, dtype=torch.float32)
    valid = torch.zeros(len(rows), maximum, dtype=torch.bool)
    weights = torch.zeros(len(rows), maximum, dtype=torch.float32)
    ids = []
    for sample_index, row in enumerate(rows):
        ids.append(0 if conditioning == "global" else DATASET_IDS[row["benchmark"]])
        selected_weights = route_weights(row["valid_routes"], route_weighting)
        for route_index, (route, weight) in enumerate(zip(row["valid_routes"], selected_weights)):
            masks[sample_index, route_index] = torch.tensor(route["mask"], dtype=torch.float32)
            valid[sample_index, route_index] = True
            weights[sample_index, route_index] = weight
    return {
        "dataset_ids": torch.tensor(ids, dtype=torch.long),
        "valid_masks": masks,
        "valid_mask": valid,
        "route_weights": weights,
    }


def train_batch(model, rows: list[dict], *, objective: str, route_weighting: str, conditioning: str):
    batch = make_set_batch(rows, route_weighting=route_weighting, conditioning=conditioning)
    if objective == "exact_set_nll":
        logits = model(batch["dataset_ids"])
        return multi_valid_set_nll(
            logits,
            batch["valid_masks"],
            valid_mask=batch["valid_mask"],
            route_weights=batch["route_weights"],
        )
    logits_rows = []
    targets = []
    weights = []
    for sample_index, row in enumerate(rows):
        selected_weights = route_weights(row["valid_routes"], route_weighting)
        dataset_id = 0 if conditioning == "global" else DATASET_IDS[row["benchmark"]]
        for route, weight in zip(row["valid_routes"], selected_weights):
            logits_rows.append(dataset_id)
            targets.append(route["mask"])
            weights.append(weight)
    logits = model(torch.tensor(logits_rows, dtype=torch.long))
    per_path = polar_path_bce_per_path(logits, torch.tensor(targets, dtype=torch.float32))
    return (per_path * torch.tensor(weights, dtype=per_path.dtype)).sum() / len(rows)


@torch.no_grad()
def evaluate(model, rows: list[dict], *, route_weighting: str, conditioning: str, top_k: int) -> dict:
    batch = make_set_batch(rows, route_weighting=route_weighting, conditioning=conditioning)
    logits = model(batch["dataset_ids"])
    metrics = batch_offline_metrics(
        logits, batch["valid_masks"], batch["valid_mask"], top_k=top_k
    )
    metrics["set_nll"] = float(
        multi_valid_set_nll(
            logits,
            batch["valid_masks"],
            valid_mask=batch["valid_mask"],
            route_weights=batch["route_weights"],
        )
    )
    metrics["examples"] = len(rows)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--p11-smoke-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objective", choices=("duplicated_bce", "exact_set_nll"), required=True)
    parser.add_argument("--conditioning", choices=("global", "dataset"), required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for name, specification in config["gates"].items():
        validate_gate(name, specification)
    manifest_path = Path(config["data"]["manifest"])
    if file_sha256(manifest_path) != config["data"]["manifest_sha256"]:
        raise RuntimeError("predictor manifest checksum mismatch")
    smoke_path = Path(args.p11_smoke_manifest)
    if file_sha256(smoke_path) != config["smoke"]["manifest_sha256"]:
        raise RuntimeError("P11 smoke manifest checksum mismatch")
    train_rows, validation_rows = selected_rows(manifest_path, smoke_path)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite bias baseline: {output_dir}")
    output_dir.mkdir(parents=True)

    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    num_datasets = 1 if args.conditioning == "global" else len(DATASET_IDS)
    model = BiasOnlyBinaryPredictor(
        num_layers=int(config["policy"]["num_layers"]), num_datasets=num_datasets
    )
    initialization_sha256 = predictor_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(config["training"]["batch_size"])
    epochs = int(config["smoke"]["epochs"])
    route_weighting = str(config["data"]["route_weighting"])
    history = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train_rows), generator=generator).tolist()
        loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            current = [train_rows[index] for index in order[start : start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss = train_batch(
                model,
                current,
                objective=args.objective,
                route_weighting=route_weighting,
                conditioning=args.conditioning,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
            optimizer.step()
            loss_sum += float(loss.detach()) * len(current)
        validation = evaluate(
            model,
            validation_rows,
            route_weighting=route_weighting,
            conditioning=args.conditioning,
            top_k=int(config["evaluation"]["top_k"]),
        )
        row = {
            "epoch": epoch,
            "train": {"loss": loss_sum / len(train_rows), "objective": args.objective},
            "validation": validation,
        }
        history.append(row)
        torch.save(
            {
                "predictor": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": config,
                "conditioning": args.conditioning,
                "objective": args.objective,
                "predictor_initialization_sha256": initialization_sha256,
            },
            output_dir / f"checkpoint_epoch_{epoch:02d}.pt",
        )

    best = max(history, key=checkpoint_key)
    best_epoch = int(best["epoch"])
    best_record = {
        "schema_version": "binary_polar_p11_bias_baseline_v1",
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "conditioning": args.conditioning,
        "objective": args.objective,
        "initialization": "zero logits",
        "predictor_initialization_sha256": initialization_sha256,
        "train_records": len(train_rows),
        "validation_records": len(validation_rows),
        "epoch": best_epoch,
        "checkpoint": str(output_dir / f"checkpoint_epoch_{best_epoch:02d}.pt"),
        "validation": best["validation"],
        "by_benchmark": {
            benchmark: evaluate(
                model,
                [row for row in validation_rows if row["benchmark"] == benchmark],
                route_weighting=route_weighting,
                conditioning=args.conditioning,
                top_k=int(config["evaluation"]["top_k"]),
            )
            for benchmark in DATASET_IDS
        },
        "selection_rule": (
            "max validation Valid-Set Hit@1; then Hit@5; then minimum nearest-valid Hamming; "
            "then minimum common exact-set NLL; then earliest epoch"
        ),
    }
    # Restore the selected state before writing benchmark metrics if an earlier epoch won.
    checkpoint = torch.load(best_record["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["predictor"])
    best_record["by_benchmark"] = {
        benchmark: evaluate(
            model,
            [row for row in validation_rows if row["benchmark"] == benchmark],
            route_weighting=route_weighting,
            conditioning=args.conditioning,
            top_k=int(config["evaluation"]["top_k"]),
        )
        for benchmark in DATASET_IDS
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (output_dir / "best_checkpoint.json").write_text(
        json.dumps(best_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
