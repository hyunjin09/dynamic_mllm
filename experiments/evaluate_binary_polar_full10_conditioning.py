#!/usr/bin/env python3
"""Evaluate frozen full10 selected checkpoints under aligned/shuffled inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import yaml

from binary_policy.dataset import route_weights, make_set_collator
from binary_policy.decode import topk_factorized_masks
from binary_policy.evaluation import batch_offline_metrics
from binary_policy.losses import multi_valid_set_nll
from binary_policy.multimodal import make_multimodal_set_collator
from binary_policy.predictor import BinaryPolarBackbone, FrozenHFTokenEncoder
from experiments.train_binary_polar import file_sha256


BENCHMARKS = ("gqa", "textvqa", "chartqa")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def padded_valid_sets(rows: list[dict], weighting: str):
    maximum = max(len(row["valid_routes"]) for row in rows)
    width = len(rows[0]["valid_routes"][0]["mask"])
    masks = torch.zeros(len(rows), maximum, width)
    valid = torch.zeros(len(rows), maximum, dtype=torch.bool)
    weights = torch.zeros(len(rows), maximum)
    for sample_index, row in enumerate(rows):
        current = route_weights(row["valid_routes"], weighting)
        for route_index, (route, weight) in enumerate(zip(row["valid_routes"], current)):
            masks[sample_index, route_index] = torch.tensor(route["mask"])
            valid[sample_index, route_index] = True
            weights[sample_index, route_index] = weight
    return masks, valid, weights


def metrics(rows: list[dict], logits: torch.Tensor, weighting: str) -> dict:
    masks, valid, weights = padded_valid_sets(rows, weighting)
    result = batch_offline_metrics(logits, masks, valid, top_k=5)
    result["set_nll"] = float(
        multi_valid_set_nll(logits, masks, valid_mask=valid, route_weights=weights)
    )
    result["examples"] = len(rows)
    return result


def condition_rows(targets: list[dict], by_uid: dict, mapping: dict, condition: str) -> list[dict]:
    output = []
    for target in targets:
        row = dict(target)
        donors = mapping[target["uid"]]
        if condition == "aligned":
            row["feature_uid"] = target["uid"]
        elif condition == "question_shuffled":
            row["question"] = by_uid[donors["question_uid"]]["question"]
            row["feature_uid"] = target["uid"]
        elif condition == "image_shuffled":
            row["feature_uid"] = donors["image_uid"]
        elif condition == "both_shuffled":
            row["question"] = by_uid[donors["both_question_uid"]]["question"]
            row["feature_uid"] = donors["both_image_uid"]
        else:
            raise ValueError(condition)
        output.append(row)
    return output


@torch.inference_mode()
def score_rows(
    config,
    encoder,
    tokenizer,
    checkpoint_path: Path,
    rows: list[dict],
    feature_index: dict,
    modality: str,
    device: torch.device,
    description: str,
) -> torch.Tensor:
    architecture = dict(
        num_layers=int(config["policy"]["num_layers"]),
        input_dim=encoder.output_dim,
        d_model=int(config["predictor"]["d_model"]),
        num_heads=int(config["predictor"]["num_heads"]),
        num_layer_blocks=int(config["predictor"]["num_layer_blocks"]),
        dropout=float(config["predictor"]["dropout"]),
    )
    predictor = BinaryPolarBackbone(
        **architecture,
        **(
            {"image_dim": int(config["visual_features"]["feature_width"])}
            if modality == "image_question"
            else {}
        ),
    ).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    predictor.load_state_dict(checkpoint["predictor"], strict=True)
    collator = (
        make_multimodal_set_collator(
            tokenizer,
            feature_index,
            max_length=int(config["data"]["max_question_tokens"]),
            route_weighting=config["data"]["route_weighting"],
        )
        if modality == "image_question"
        else make_set_collator(
            tokenizer,
            max_length=int(config["data"]["max_question_tokens"]),
            route_weighting=config["data"]["route_weighting"],
        )
    )
    outputs = []
    batch_size = int(config["training"]["physical_batch_size"])
    iterator = range(0, len(rows), batch_size)
    for start in tqdm(iterator, desc=description, unit="batch", dynamic_ncols=True):
        batch = collator(rows[start : start + batch_size])
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        question = encoder(batch["input_ids"], batch["attention_mask"])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if modality == "question":
                logits = predictor(question, batch["attention_mask"])
            else:
                logits = predictor(
                    question,
                    batch["attention_mask"],
                    batch["image_features"],
                    batch["image_attention_mask"],
                )
        outputs.append(logits.float().cpu())
    del predictor
    torch.cuda.empty_cache()
    return torch.cat(outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--question-dir", required=True)
    parser.add_argument("--image-question-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["data"]["manifest"])
    rows = read_jsonl(manifest_path)
    by_uid = {row["uid"]: row for row in rows}
    validation = [row for row in rows if row["split"] == "validation" and row["valid_routes"]]
    if len(validation) != 874:
        raise RuntimeError("full10 conditioning population is not 874")
    feature_path = Path(config["visual_features"]["manifest"])
    feature_index = {row["uid"]: row for row in read_jsonl(feature_path)}
    permutation_path = Path(config["visual_features"]["validation_permutations"])
    if file_sha256(permutation_path) != config["visual_features"]["validation_permutations_sha256"]:
        raise RuntimeError("full10 validation permutation checksum mismatch")
    mapping = json.loads(permutation_path.read_text(encoding="utf-8"))["mapping"]
    smoke = json.loads(Path(config["execution"]["manifest"]).read_text(encoding="utf-8"))
    execution_rows = [by_uid[uid] for uid in smoke["execution_validation_uids"]]

    device = torch.device("cuda")
    encoder_path = config["predictor"]["embedding_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_path, padding_side="left", local_files_only=True
    )
    encoder = FrozenHFTokenEncoder(encoder_path, dtype=torch.bfloat16).to(device).eval()
    weighting = config["data"]["route_weighting"]
    outputs = {}
    execution_predictions = {}
    for modality, directory_value in (
        ("question", args.question_dir),
        ("image_question", args.image_question_dir),
    ):
        directory = Path(directory_value)
        summary = json.loads((directory / "training_summary.json").read_text(encoding="utf-8"))
        selected = {
            name: epoch
            for name, epoch in summary["selections"].items()
            if name in {"best_hit_at_1", "best_set_nll", "final"}
        }
        outputs[modality] = {}
        execution_predictions[modality] = {}
        for selection_name, epoch in selected.items():
            checkpoint = directory / f"epoch_{epoch:02d}" / "checkpoint.pt"
            conditions = (
                ("aligned", "question_shuffled")
                if modality == "question"
                else ("aligned", "question_shuffled", "image_shuffled", "both_shuffled")
            )
            outputs[modality][selection_name] = {"epoch": epoch, "conditions": {}}
            for condition in conditions:
                conditioned = condition_rows(validation, by_uid, mapping, condition)
                logits = score_rows(
                    config,
                    encoder,
                    tokenizer,
                    checkpoint,
                    conditioned,
                    feature_index,
                    modality,
                    device,
                    f"{modality} {selection_name} {condition}",
                )
                outputs[modality][selection_name]["conditions"][condition] = {
                    "overall": metrics(validation, logits, weighting),
                    "by_benchmark": {
                        benchmark: metrics(
                            [row for row in validation if row["benchmark"] == benchmark],
                            logits[
                                torch.tensor(
                                    [row["benchmark"] == benchmark for row in validation]
                                )
                            ],
                            weighting,
                        )
                        for benchmark in BENCHMARKS
                    },
                }
            if selection_name in {"best_hit_at_1", "final"}:
                logits = score_rows(
                    config,
                    encoder,
                    tokenizer,
                    checkpoint,
                    execution_rows,
                    feature_index,
                    modality,
                    device,
                    f"{modality} {selection_name} execution decode",
                )
                decoded = topk_factorized_masks(logits, top_k=1)
                execution_predictions[modality][selection_name] = {
                    "epoch": epoch,
                    "rows": [
                        {
                            "uid": row["uid"],
                            "benchmark": row["benchmark"],
                            "mask": list(candidates[0].mask),
                            "mask_key": "".join(map(str, candidates[0].mask)),
                            "visual_on_layers": sum(candidates[0].mask),
                        }
                        for row, candidates in zip(execution_rows, decoded)
                    ],
                }
    payload = {
        "schema_version": "binary_polar_full10_conditioning_v1",
        "passed": True,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "validation_records": len(validation),
        "metrics": outputs,
        "execution_predictions": execution_predictions,
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite full10 conditioning: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
