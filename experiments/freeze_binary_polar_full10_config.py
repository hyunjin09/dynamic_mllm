#!/usr/bin/env python3
"""Freeze the checksum-bound full10 config after the visual cache passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from binary_policy.multimodal import deterministic_group_disjoint_modality_permutations
from experiments.train_binary_polar import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-audit", required=True)
    parser.add_argument("--permutation-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cache_path = Path(args.cache_audit)
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("passed") is not True:
        raise RuntimeError("full10 visual cache did not pass")
    required = {
        "positive_records": 6917,
        "train_positive_records": 6043,
        "validation_positive_records": 874,
        "unique_image_groups": 6574,
    }
    for key, expected in required.items():
        if cache.get(key) != expected:
            raise RuntimeError(f"full10 cache {key} mismatch: {cache.get(key)} != {expected}")
    manifest = Path(cache["manifest"])
    if file_sha256(manifest) != cache["manifest_sha256"]:
        raise RuntimeError("full10 feature-manifest checksum mismatch")

    plan = Path("plans/full_train.md")
    predictor_manifest = Path(
        "outputs/label_regeneration/v1/post_generation/binary_predictor_manifest_v1.jsonl"
    )
    validation_rows = []
    with predictor_manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] == "validation" and row["valid_routes"]:
                validation_rows.append(row)
    if len(validation_rows) != 874:
        raise RuntimeError("full10 validation population is not 874")
    mapping = deterministic_group_disjoint_modality_permutations(
        validation_rows, seed=20260813
    )
    by_uid = {row["uid"]: row for row in validation_rows}
    for uid, donors in mapping.items():
        for donor in donors.values():
            if donor == uid or by_uid[donor]["split_group"] == by_uid[uid]["split_group"]:
                raise RuntimeError("full10 permutation retains target UID or image group")
    permutations = Path(args.permutation_output)
    if permutations.exists():
        raise FileExistsError(f"refusing to overwrite full10 permutations: {permutations}")
    permutations.parent.mkdir(parents=True, exist_ok=True)
    permutations.write_text(
        json.dumps(
            {
                "schema_version": "binary_polar_full10_group_disjoint_permutations_v2",
                "seed": 20260813,
                "same_uid_pairs": 0,
                "same_image_group_pairs": 0,
                "validation_uids": [row["uid"] for row in validation_rows],
                "mapping": mapping,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    permutations.with_suffix(permutations.suffix + ".sha256").write_text(
        f"{file_sha256(permutations)}  {permutations.name}\n", encoding="utf-8"
    )
    execution_manifest = Path("outputs/binary_polar/preflight/p11_smoke_manifest_v1.json")
    config = {
        "protocol_version": "binary_polar_full10_polar_matched_v1",
        "authorization": "full10_question_and_image_question_only",
        "source_plan": {"path": str(plan), "sha256": file_sha256(plan)},
        "gates": {
            "regenerated_label_cache": {
                "path": "outputs/label_regeneration/v1/post_generation/cache_audit_v1.json",
                "sha256": "0afc2e62a0b20b5821bc847d8be2080d0f9cc9cef3cd94b829ff0e924a353cf2",
            },
            "image_group_splits": {
                "path": "outputs/label_regeneration/v1/post_generation/predictor_split_audit_v1.json",
                "sha256": "2f60e4688e7727d5f6d715c255daba4cba4a2f0e10476b390371522c0b1ad84e",
            },
            "derived_valid_set": {
                "path": "outputs/label_regeneration/v1/post_generation/derived_supervision_audit_v1.json",
                "sha256": "29ed03efdb548fa19fc6eddccd03612e6f348b8f382ce11237cbeb03f9b54856",
            },
            "objective_sanity": {
                "path": "outputs/binary_polar/preflight/loss_comparison_sanity_v1.json",
                "sha256": "4efdbd9b5725eacff2e0a85fb062679f19852d9375907839abc28882ae6daa0a",
            },
            "full_visual_cache": {"path": str(cache_path), "sha256": file_sha256(cache_path)},
        },
        "base_model": {
            "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "precision": "bfloat16",
        },
        "policy": {
            "num_layers": 28,
            "representation": "direct_factorized_binary_mask",
            "decode_threshold": 0.5,
        },
        "data": {
            "manifest": str(predictor_manifest),
            "manifest_sha256": file_sha256(predictor_manifest),
            "train_positive_records": 6043,
            "validation_positive_records": 874,
            "max_valid_routes_per_sample": 50,
            "route_cap_policy": "frozen_diverse_50_polar_matched",
            "route_weighting": "polar_full_downweight_0.3",
            "max_question_tokens": 512,
        },
        "visual_features": {
            "manifest": str(manifest),
            "manifest_sha256": file_sha256(manifest),
            "validation_permutations": str(permutations),
            "validation_permutations_sha256": file_sha256(permutations),
            "source": "projected_visual_rows_entering_decoder_layer_0",
            "feature_width": 3584,
            "dtype": "bfloat16",
            "unpooled": True,
        },
        "predictor": {
            "embedding_model_id": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_revision": "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418",
            "embedding_model_path": "/home/hyunjin/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418",
            "frozen_embedding_model": True,
            "d_model": 256,
            "num_heads": 4,
            "num_layer_blocks": 2,
            "dropout": 0.1,
            "head": "direct_28_binary_logits",
        },
        "training": {
            "seed": 20260809,
            "epochs": 10,
            "physical_batch_size": 128,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": 128,
            "learning_rate": 0.0005,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "scheduler": "cosine",
            "warmup_steps": 10,
            "num_workers": 4,
            "gradient_clip_norm": 1.0,
            "precision": "bfloat16",
            "deterministic_algorithms": True,
            "no_early_stopping": True,
            "save_every_epoch": True,
        },
        "execution": {
            "manifest": str(execution_manifest),
            "manifest_sha256": file_sha256(execution_manifest),
            "records": 60,
        },
        "evaluation": {
            "top_k": 5,
            "dataset_wise": ["gqa", "textvqa", "chartqa"],
            "conditioning_selections": ["best_hit_at_1", "best_set_nll", "final"],
            "execution_selections": ["best_hit_at_1", "final"],
        },
        "source_sha256": {
            path: file_sha256(Path(path))
            for path in (
                "binary_policy/dataset.py",
                "binary_policy/decode.py",
                "binary_policy/evaluation.py",
                "binary_policy/losses.py",
                "binary_policy/multimodal.py",
                "binary_policy/predictor.py",
                "experiments/train_binary_polar_full10.py",
                "experiments/evaluate_binary_polar_full10_conditioning.py",
                "experiments/evaluate_binary_polar_full10_execution.py",
            )
        },
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite full10 config: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
