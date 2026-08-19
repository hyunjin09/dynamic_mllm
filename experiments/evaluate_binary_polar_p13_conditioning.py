#!/usr/bin/env python3
"""Run the frozen P13 three-model/four-condition modality diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from transformers import AutoTokenizer

from binary_policy.dataset import route_weights
from binary_policy.decode import topk_factorized_masks
from binary_policy.evaluation import batch_offline_metrics
from binary_policy.losses import multi_valid_set_nll
from binary_policy.multimodal import MODALITIES, make_multimodal_set_collator, resolve_modality_inputs
from binary_policy.predictor import BinaryPolarBackbone, FrozenHFTokenEncoder
from experiments.train_binary_polar import file_sha256, validate_gate, validate_readiness_bundle


BENCHMARKS = ("gqa", "textvqa", "chartqa")
CONDITIONS = ("aligned", "question_shuffled", "image_shuffled", "both_shuffled")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def padded_valid_sets(rows, weighting):
    maximum = max(len(row["valid_routes"]) for row in rows)
    width = len(rows[0]["valid_routes"][0]["mask"])
    masks = torch.zeros(len(rows), maximum, width)
    valid = torch.zeros(len(rows), maximum, dtype=torch.bool)
    weights = torch.zeros(len(rows), maximum)
    for sample_index, row in enumerate(rows):
        current_weights = route_weights(row["valid_routes"], weighting)
        for route_index, (route, weight) in enumerate(zip(row["valid_routes"], current_weights)):
            masks[sample_index, route_index] = torch.tensor(route["mask"])
            valid[sample_index, route_index] = True
            weights[sample_index, route_index] = weight
    return masks, valid, weights


def metrics(rows, logits, weighting, top_k):
    masks, valid, weights = padded_valid_sets(rows, weighting)
    result = batch_offline_metrics(logits, masks, valid, top_k=top_k)
    result["set_nll"] = float(
        multi_valid_set_nll(logits, masks, valid_mask=valid, route_weights=weights)
    )
    result["examples"] = len(rows)
    return result


def condition_rows(target_rows, by_uid, mapping, condition):
    output = []
    for target in target_rows:
        donors = mapping[target["uid"]]
        row = dict(target)
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
def score(config, checkpoint, rows, feature_index, modality, device):
    encoder_path = config["predictor"]["embedding_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, padding_side="left", local_files_only=True)
    encoder = FrozenHFTokenEncoder(encoder_path, dtype=torch.bfloat16).to(device)
    predictor = BinaryPolarBackbone(
        num_layers=int(config["policy"]["num_layers"]),
        input_dim=encoder.output_dim,
        image_dim=int(config["p13"]["visual_feature_width"]),
        d_model=int(config["predictor"]["d_model"]),
        num_heads=int(config["predictor"]["num_heads"]),
        num_layer_blocks=int(config["predictor"]["num_layer_blocks"]),
        dropout=float(config["predictor"]["dropout"]),
    ).to(device).eval()
    predictor.load_state_dict(checkpoint["predictor"], strict=True)
    collator = make_multimodal_set_collator(
        tokenizer,
        feature_index,
        max_length=int(config["data"]["max_question_tokens"]),
        route_weighting=str(config["data"]["route_weighting"]),
    )
    logits = []
    batch_size = int(config["training"]["batch_size"])
    for start in range(0, len(rows), batch_size):
        batch = collator(rows[start : start + batch_size])
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        if modality == "image":
            question_features = batch["image_features"].new_zeros(
                batch["input_ids"].shape[0], 1, encoder.output_dim
            )
            question_mask = batch["attention_mask"].new_zeros(batch["input_ids"].shape[0], 1)
        else:
            question_features = encoder(batch["input_ids"], batch["attention_mask"])
            question_mask = batch["attention_mask"]
        inputs = resolve_modality_inputs(
            modality,
            question_features,
            question_mask,
            batch["image_features"],
            batch["image_attention_mask"],
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits.append(predictor(*inputs).float().cpu())
    del predictor, encoder
    torch.cuda.empty_cache()
    return torch.cat(logits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--readiness-gate", required=True)
    parser.add_argument("--question-checkpoint", required=True)
    parser.add_argument("--image-checkpoint", required=True)
    parser.add_argument("--image-question-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confirm-gates", action="store_true")
    args = parser.parse_args()
    if not args.confirm_gates:
        raise RuntimeError("P13 conditioning requires --confirm-gates")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    readiness = validate_readiness_bundle(args.readiness_gate, config_path)
    if not readiness["ready_for_bounded_smoke"]:
        raise RuntimeError("P13 readiness does not authorize conditioning")
    for name, spec in config["gates"].items():
        validate_gate(name, spec)
    manifest_path = Path(config["data"]["manifest"])
    smoke_path = Path(config["smoke"]["manifest"])
    feature_path = Path(config["p13"]["feature_manifest"])
    permutation_path = Path(config["p13"]["modality_permutations"])
    for path, expected in (
        (manifest_path, config["data"]["manifest_sha256"]),
        (smoke_path, config["smoke"]["manifest_sha256"]),
        (feature_path, config["p13"]["feature_manifest_sha256"]),
        (permutation_path, config["p13"]["modality_permutations_sha256"]),
    ):
        if file_sha256(path) != expected:
            raise RuntimeError(f"P13 checksum mismatch: {path}")
    all_rows = read_jsonl(manifest_path)
    by_uid = {row["uid"]: row for row in all_rows}
    feature_index = {row["uid"]: row for row in read_jsonl(feature_path)}
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    targets = [by_uid[uid] for uid in smoke["validation_positive_uids"]]
    execution_targets = [by_uid[uid] for uid in smoke["execution_validation_uids"]]
    permutations = json.loads(permutation_path.read_text(encoding="utf-8"))["mapping"]
    selections = {
        "question": Path(args.question_checkpoint),
        "image": Path(args.image_checkpoint),
        "image_question": Path(args.image_question_checkpoint),
    }
    device = torch.device("cuda")
    result = {}
    aligned_predictions = {}
    execution_predictions = {}
    weighting = str(config["data"]["route_weighting"])
    top_k = int(config["evaluation"]["top_k"])
    for modality in MODALITIES:
        selection = json.loads(selections[modality].read_text(encoding="utf-8"))
        if selection["modality"] != modality or selection["mode"] != "smoke":
            raise RuntimeError(f"P13 checkpoint modality mismatch for {modality}")
        checkpoint = torch.load(selection["checkpoint"], map_location="cpu", weights_only=False)
        result[modality] = {}
        for condition in CONDITIONS:
            rows = condition_rows(targets, by_uid, permutations, condition)
            current_logits = score(config, checkpoint, rows, feature_index, modality, device)
            current = {
                "overall": metrics(targets, current_logits, weighting, top_k),
                "by_benchmark": {
                    benchmark: metrics(
                        [row for row in targets if row["benchmark"] == benchmark],
                        current_logits[
                            torch.tensor([row["benchmark"] == benchmark for row in targets])
                        ],
                        weighting,
                        top_k,
                    )
                    for benchmark in BENCHMARKS
                },
            }
            result[modality][condition] = current
            if condition == "aligned":
                decoded = topk_factorized_masks(current_logits, top_k=1)
                aligned_predictions[modality] = [
                    {
                        "uid": row["uid"],
                        "benchmark": row["benchmark"],
                        "mask": list(candidates[0].mask),
                        "mask_key": "".join(map(str, candidates[0].mask)),
                        "visual_on_layers": sum(candidates[0].mask),
                    }
                    for row, candidates in zip(targets, decoded)
                ]
        execution_logits = score(
            config, checkpoint, execution_targets, feature_index, modality, device
        )
        execution_decoded = topk_factorized_masks(execution_logits, top_k=1)
        execution_predictions[modality] = [
            {
                "uid": row["uid"],
                "benchmark": row["benchmark"],
                "mask": list(candidates[0].mask),
                "mask_key": "".join(map(str, candidates[0].mask)),
                "visual_on_layers": sum(candidates[0].mask),
            }
            for row, candidates in zip(execution_targets, execution_decoded)
        ]
    q = result["question"]["aligned"]["overall"]
    iq = result["image_question"]["aligned"]["overall"]
    iq_qs = result["image_question"]["question_shuffled"]["overall"]
    iq_is = result["image_question"]["image_shuffled"]["overall"]
    gate_checks = {
        "aligned_nll_below_question_shuffled": iq["set_nll"] < iq_qs["set_nll"],
        "aligned_nll_below_image_shuffled": iq["set_nll"] < iq_is["set_nll"],
        "all_on_at_most_0_90": iq["fraction_top1_all_on"] <= 0.90,
        "at_least_10_unique_masks": iq["unique_top1_masks"] >= 10,
        "hit_or_hamming_improves": (
            iq["top1_valid_route_coverage"] - q["top1_valid_route_coverage"] >= 0.03
            or q["nearest_valid_hamming"] - iq["nearest_valid_hamming"] >= 0.25
        ),
        "hit_not_materially_worse": (
            iq["top1_valid_route_coverage"] - q["top1_valid_route_coverage"] >= -0.02
        ),
        "hamming_not_materially_worse": (
            iq["nearest_valid_hamming"] - q["nearest_valid_hamming"] <= 0.25
        ),
    }
    payload = {
        "schema_version": "binary_polar_p13_conditioning_v1",
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint_records": {key: str(value) for key, value in selections.items()},
        "permutation_sha256": file_sha256(permutation_path),
        "metrics": result,
        "image_question_minus_question_aligned": {
            key: iq[key] - q[key]
            for key in (
                "set_nll",
                "top1_valid_route_coverage",
                "topk_valid_route_coverage",
                "nearest_valid_hamming",
                "fraction_top1_all_on",
                "average_predicted_visual_on",
            )
        },
        "execution_admission": {
            "spec": "workspace/binary_polar_p13_execution_admission_gate.md",
            "checks": gate_checks,
            "passed": all(gate_checks.values()),
        },
        "aligned_predictions": aligned_predictions,
        "execution_predictions": execution_predictions,
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P13 conditioning: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
