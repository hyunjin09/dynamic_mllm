#!/usr/bin/env python3
"""Execute every CAP-NLL epoch on the frozen internal validation split."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import yaml

from binary_policy.decode import topk_factorized_masks
from binary_policy.predictor import FrozenHFTokenEncoder
from experiments.evaluate_binary_polar_full10_conditioning import read_jsonl, score_rows
from experiments.train_binary_polar import file_sha256
from label_regeneration.runtime import RouteEvaluator, configure_determinism, load_frozen_model


BENCHMARKS = ("gqa", "textvqa", "chartqa")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row["predicted_correct"]) for row in rows)
    baseline_correct = sum(bool(row["baseline_correct"]) for row in rows)
    masks = Counter(str(row["predicted_mask_key"]) for row in rows)
    mean_on = sum(int(row["num_visual_on_layers"]) for row in rows) / total
    return {
        "records": total,
        "predicted_mask_accuracy": correct / total,
        "baseline_accuracy": baseline_correct / total,
        "full_wrong_to_predicted_correct": sum(
            (not row["baseline_correct"]) and row["predicted_correct"] for row in rows
        ),
        "full_correct_to_predicted_wrong": sum(
            row["baseline_correct"] and (not row["predicted_correct"]) for row in rows
        ),
        "unchanged_correct": sum(
            row["baseline_correct"] and row["predicted_correct"] for row in rows
        ),
        "unchanged_wrong": sum(
            (not row["baseline_correct"]) and (not row["predicted_correct"]) for row in rows
        ),
        "average_visual_on_layers": mean_on,
        "visual_on_layer_reduction_from_full": 28.0 - mean_on,
        "all_on_fraction": masks["1" * 28] / total,
        "all_off_fraction": masks["0" * 28] / total,
        "unique_predicted_masks": len(masks),
        "selected_valid_set_hit_at_1": sum(
            bool(row["predicted_mask_in_selected_valid_set"]) for row in rows
        ) / total,
    }


def select_epoch(epoch_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen accuracy/compute/NLL/epoch ordering."""

    if not epoch_rows:
        raise ValueError("epoch_rows must be nonempty")
    return max(
        epoch_rows,
        key=lambda row: (
            float(row["summary"]["overall"]["predicted_mask_accuracy"]),
            -float(row["summary"]["overall"]["average_visual_on_layers"]),
            -float(row["validation_objective_loss"]),
            -int(row["epoch"]),
        ),
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["training"]["objective"] != "exact_set_nll":
        raise RuntimeError("executed-validation selector requires exact_set_nll")
    epochs = int(config["training"]["epochs"])
    if epochs != 5:
        raise RuntimeError(f"frozen executed-validation study requires five epochs, got {epochs}")
    manifest = Path(config["data"]["manifest"])
    if file_sha256(manifest) != config["data"]["manifest_sha256"]:
        raise RuntimeError("validation manifest checksum mismatch")
    rows = [
        row for row in read_jsonl(manifest)
        if row["split"] == "validation" and row.get("valid_routes")
    ]
    if len(rows) != int(config["data"]["validation_positive_records"]):
        raise RuntimeError("validation population differs from frozen config")
    if len({row["split_group"] for row in rows}) > len(rows):
        raise RuntimeError("invalid validation image-group accounting")

    history = json.loads((args.training_root / "history.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (args.training_root / "training_summary.json").read_text(encoding="utf-8")
    )
    if summary.get("passed") is not True or summary.get("objective") != "exact_set_nll":
        raise RuntimeError("training summary is incomplete or has the wrong objective")
    if len(history) != epochs or summary["epochs_completed"] != epochs:
        raise RuntimeError("five-epoch training is incomplete")
    checkpoints = {}
    expected_hashes = {
        int(item["epoch"]): str(item["checkpoint_sha256"])
        for item in summary["checkpoints"]
    }
    for epoch in range(1, epochs + 1):
        checkpoint = args.training_root / f"epoch_{epoch:02d}" / "checkpoint.pt"
        if file_sha256(checkpoint) != expected_hashes[epoch]:
            raise RuntimeError(f"epoch {epoch} checkpoint checksum mismatch")
        checkpoints[epoch] = checkpoint

    device = torch.device("cuda")
    configure_determinism(int(config["training"]["seed"]))
    encoder_path = config["predictor"]["embedding_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_path, padding_side="left", local_files_only=True
    )
    encoder = FrozenHFTokenEncoder(encoder_path, dtype=torch.bfloat16).to(device).eval()
    feature_manifest = Path(config["visual_features"]["manifest"])
    if file_sha256(feature_manifest) != config["visual_features"]["manifest_sha256"]:
        raise RuntimeError("visual-feature manifest checksum mismatch")
    feature_index = {row["uid"]: row for row in read_jsonl(feature_manifest)}
    predictions: dict[int, dict[str, list[int]]] = {}
    for epoch, checkpoint in checkpoints.items():
        logits = score_rows(
            config,
            encoder,
            tokenizer,
            checkpoint,
            rows,
            feature_index,
            "image_question",
            device,
            f"decode validation epoch {epoch}",
        )
        decoded = topk_factorized_masks(logits, top_k=1)
        predictions[epoch] = {
            row["uid"]: list(candidates[0].mask)
            for row, candidates in zip(rows, decoded)
        }
    del encoder, tokenizer
    torch.cuda.empty_cache()

    source = {
        row["uid"]: row
        for row in read_jsonl(Path("outputs/label_regeneration/v1/source_manifest_v1.jsonl"))
    }
    baseline = {
        row["uid"]: row
        for row in read_jsonl(
            Path("outputs/label_regeneration/v1/post_generation/per_sample_route_summary_v1.jsonl")
        )
    }
    parts = args.output.parent / "validation_parts_v1"
    parts.mkdir(parents=True, exist_ok=True)
    processor, base, wrapped, device = load_frozen_model(
        config["base_model"]["model_id"], config["base_model"]["revision"], 0
    )
    config_digest = file_sha256(args.config)
    checkpoint_digest = {epoch: file_sha256(path) for epoch, path in checkpoints.items()}
    for index, row in enumerate(tqdm(rows, desc="execute five validation epochs", unit="sample")):
        part = parts / f"part_{index:05d}.json"
        if part.exists():
            if not args.resume:
                raise FileExistsError(f"existing validation part requires --resume: {part}")
            cached = json.loads(part.read_text(encoding="utf-8"))
            if (
                cached.get("uid") != row["uid"]
                or cached.get("config_sha256") != config_digest
                or {int(key): value for key, value in cached.get("checkpoint_sha256", {}).items()}
                != checkpoint_digest
            ):
                raise RuntimeError(f"stale validation part: {part}")
            continue
        evaluator = RouteEvaluator(
            processor=processor,
            base_model=base,
            wrapped_model=wrapped,
            sample=source[row["uid"]],
            device=device,
        )
        executed_by_mask = {}
        epoch_results = {}
        selected_keys = {route["key"] for route in row["valid_routes"]}
        for epoch in range(1, epochs + 1):
            mask = predictions[epoch][row["uid"]]
            key = "".join(map(str, mask))
            if key not in executed_by_mask:
                executed_by_mask[key] = evaluator.evaluate(
                    tuple(mask), f"cap_nll5_validation_epoch_{epoch}"
                )
            executed = executed_by_mask[key]
            if executed.get("scoring_timed_out"):
                raise RuntimeError(f"validation scoring timed out for {row['uid']}")
            epoch_results[str(epoch)] = {
                "epoch": epoch,
                "predicted_mask": mask,
                "predicted_mask_key": key,
                "num_visual_on_layers": sum(mask),
                "predicted_mask_in_selected_valid_set": key in selected_keys,
                "generated_ids": executed["generated_ids"],
                "prediction": executed["prediction"],
                "score": executed["score"],
                "predicted_correct": executed["result_correct"],
            }
        base_row = baseline[row["uid"]]
        write_json(
            part,
            {
                "schema_version": "binary_cap_nll5_validation_part_v1",
                "uid": row["uid"],
                "benchmark": row["benchmark"],
                "split_group": row["split_group"],
                "config_sha256": config_digest,
                "checkpoint_sha256": checkpoint_digest,
                "baseline_prediction": base_row["current_all_on_prediction"],
                "baseline_score": base_row["current_all_on_score"],
                "baseline_correct": base_row["current_all_on_status"] == "correct",
                "baseline_source": "frozen_current_executor_all_on_cache",
                "epochs": epoch_results,
            },
        )

    part_paths = sorted(parts.glob("part_*.json"))
    if len(part_paths) != len(rows):
        raise RuntimeError("validation execution did not produce one part per record")
    by_epoch = {epoch: [] for epoch in range(1, epochs + 1)}
    for expected, part in zip(rows, part_paths):
        payload = json.loads(part.read_text(encoding="utf-8"))
        if payload["uid"] != expected["uid"]:
            raise RuntimeError("validation part ordering/UID mismatch")
        for epoch in range(1, epochs + 1):
            result = payload["epochs"][str(epoch)]
            by_epoch[epoch].append(
                {
                    "uid": payload["uid"],
                    "benchmark": payload["benchmark"],
                    "split_group": payload["split_group"],
                    "baseline_prediction": payload["baseline_prediction"],
                    "baseline_score": payload["baseline_score"],
                    "baseline_correct": payload["baseline_correct"],
                    **result,
                }
            )
    validation_losses = {
        int(item["epoch"]): float(item["validation"]["overall"]["objective_loss"])
        for item in history
    }
    epoch_payloads = []
    for epoch in range(1, epochs + 1):
        current = by_epoch[epoch]
        epoch_payloads.append(
            {
                "epoch": epoch,
                "checkpoint": str(checkpoints[epoch]),
                "checkpoint_sha256": checkpoint_digest[epoch],
                "validation_objective_loss": validation_losses[epoch],
                "summary": {
                    "overall": summarize(current),
                    "by_benchmark": {
                        benchmark: summarize(
                            [row for row in current if row["benchmark"] == benchmark]
                        )
                        for benchmark in BENCHMARKS
                    },
                },
                "rows": current,
            }
        )
    selected = select_epoch(epoch_payloads)
    output_payload = {
        "schema_version": "binary_cap_nll5_executed_validation_v1",
        "integrity_status": "PASS",
        "config": str(args.config),
        "config_sha256": config_digest,
        "selection_rule": (
            "max_executed_accuracy_then_min_mean_visual_on_then_"
            "min_validation_set_nll_then_earlier_epoch"
        ),
        "records": len(rows),
        "epochs": epoch_payloads,
        "selected_epoch": selected["epoch"],
    }
    if args.output.exists() or args.selection_output.exists():
        raise FileExistsError("refusing to overwrite final validation/selection artifacts")
    write_json(args.output, output_payload)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{file_sha256(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    selection = {
        "schema_version": "binary_cap_nll5_checkpoint_selection_v1",
        "passed": True,
        "objective": "exact_set_nll",
        "criterion": output_payload["selection_rule"],
        "validation_execution": {
            "path": str(args.output),
            "sha256": file_sha256(args.output),
        },
        "epoch": selected["epoch"],
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "metrics": selected["summary"]["overall"],
    }
    write_json(args.selection_output, selection)
    args.selection_output.with_suffix(args.selection_output.suffix + ".sha256").write_text(
        f"{file_sha256(args.selection_output)}  {args.selection_output.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": True, "selected_epoch": selected["epoch"]}))


if __name__ == "__main__":
    main()
