#!/usr/bin/env python3
"""Static/data/model readiness gate for four-action Image+Question POLAR."""

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
from transformers import AutoConfig, AutoTokenizer
import yaml

from experiments.train_binary_polar import file_sha256
from four_action_policy.actions import FOUR_ACTIONS
from four_action_policy.dataset import FourActionManifestDataset
from four_action_policy.external import EXPECTED_COUNTS, TOTAL_RECORDS, load_active_rows
from four_action_policy.feature_cache import (
    load_verified_feature_index,
    visual_cache_contract,
)
from four_action_policy.predictor import FourActionPolarBackbone


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_frozen_contract(config: dict[str, Any]) -> None:
    if config.get("modality") != "image_question":
        raise RuntimeError("only Image+Question conditioning is authorized")
    if tuple(config["policy"]["action_order"]) != FOUR_ACTIONS:
        raise RuntimeError("config action order differs from the unified executor")
    if int(config["policy"]["num_layers"]) != 28 or int(config["policy"]["num_actions"]) != 4:
        raise RuntimeError("four-action policy geometry must be 28 layers by 4 actions")
    objective = str(config["training"]["objective"])
    if objective not in {"duplicated_action_bce", "exact_set_nll"}:
        raise RuntimeError("unsupported four-action training objective")
    persistent = config.get("protocol_version") == "four_action_persistent_polar_v1"
    exact_training = {
        "epochs": 20 if persistent else 10,
        "physical_batch_size": 128,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 128,
        "duplicated_route_microbatch_size": 32,
        "warmup_steps": 10,
    }
    for key, expected in exact_training.items():
        if int(config["training"][key]) != expected:
            raise RuntimeError(f"training setting {key} differs from binary full10: {expected}")
    if (
        float(config["training"]["learning_rate"]) != 5e-4
        or float(config["training"]["weight_decay"]) != 0.01
        or float(config["training"]["gradient_clip_norm"]) != 1.0
    ):
        raise RuntimeError("optimizer settings differ from binary full10")
    if config["data"].get("route_cap") is not None:
        raise RuntimeError("four-action supervision must retain all valid routes")
    expected_validation = 256 if persistent else 866
    if int(config["validation"]["expected_records"]) != expected_validation:
        raise RuntimeError(
            f"full per-epoch validation must contain {expected_validation} records"
        )
    if persistent and (
        float(config["training"].get("boundary_lambda", -1)) != 1.0
        or int(config["training"].get("boundary_events_per_epoch", -1)) != 512
        or int(config["training"].get("world_size", -1)) != 4
    ):
        raise RuntimeError("persistent POLAR intervention contract differs from plan")
    if dict(config["external_evaluation"]["benchmark_counts"]) != EXPECTED_COUNTS:
        raise RuntimeError("external benchmark contract differs from the prospective restriction")
    if int(config["external_evaluation"]["expected_records"]) != TOTAL_RECORDS:
        raise RuntimeError("external evaluation total must be 14,960")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-visual-cache", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_frozen_contract(config)

    manifest_path = Path(config["data"]["manifest"])
    audit_path = Path(config["data"]["manifest_audit"])
    if file_sha256(manifest_path) != config["data"]["manifest_sha256"]:
        raise RuntimeError("training manifest checksum mismatch")
    if file_sha256(audit_path) != config["data"]["manifest_audit_sha256"]:
        raise RuntimeError("training manifest audit checksum mismatch")
    audit = read_json(audit_path)
    if audit.get("passed") is not True:
        raise RuntimeError("training manifest audit did not pass")
    manifest = FourActionManifestDataset(manifest_path)
    split_counts = Counter(row["split"] for row in manifest.rows)
    if split_counts != Counter(
        {
            "train": int(config["data"]["train_records"]),
            "validation": int(config["data"]["validation_records"]),
        }
    ):
        raise RuntimeError(f"training split counts differ from config: {split_counts}")
    if sum(len(row["valid_routes"]) for row in manifest.rows) != int(config["data"]["valid_routes"]):
        raise RuntimeError("valid-route count differs from config")

    encoder_path = Path(config["predictor"]["embedding_model_path"])
    if not encoder_path.is_dir():
        raise FileNotFoundError(f"Qwen3 embedding snapshot is missing: {encoder_path}")
    encoder_config = AutoConfig.from_pretrained(encoder_path, local_files_only=True)
    AutoTokenizer.from_pretrained(
        encoder_path, padding_side="left", local_files_only=True
    )
    safetensors = sorted(encoder_path.glob("*.safetensors"))
    if not safetensors:
        raise FileNotFoundError("Qwen3 embedding snapshot has no safetensors weights")
    predictor = FourActionPolarBackbone(
        num_layers=int(config["policy"]["num_layers"]),
        input_dim=int(encoder_config.hidden_size),
        image_dim=int(config["visual_features"]["feature_width"]),
        d_model=int(config["predictor"]["d_model"]),
        num_heads=int(config["predictor"]["num_heads"]),
        num_layer_blocks=int(config["predictor"]["num_layer_blocks"]),
        dropout=0.0,
    ).eval()
    with torch.inference_mode():
        shape = list(
            predictor(
                torch.zeros(2, 3, int(encoder_config.hidden_size)),
                torch.ones(2, 3, dtype=torch.bool),
                torch.zeros(2, 4, int(config["visual_features"]["feature_width"])),
                torch.ones(2, 4, dtype=torch.bool),
            ).shape
        )
    if shape != [2, 28, 4]:
        raise RuntimeError(f"predictor smoke geometry mismatch: {shape}")

    base_model_path = Path(config["base_model"]["path"])
    if not base_model_path.is_dir() or not any(base_model_path.glob("*.safetensors")):
        raise FileNotFoundError(f"pinned Qwen2.5-VL snapshot is incomplete: {base_model_path}")

    external = config["external_evaluation"]
    source_files = {
        "heldout_lmms_recommended_v1": Path(external["data_root"]) / "heldout_lmms_recommended_v1/samples.jsonl",
        "heldout_mmstar_mmmu_final_v2": Path(external["data_root"]) / "heldout_mmstar_mmmu_final_v2/samples.jsonl",
        "heldout_pope_v1": Path(external["data_root"]) / "heldout_pope_v1/samples.jsonl",
    }
    for name, path in source_files.items():
        if file_sha256(path) != external["source_sha256"][name]:
            raise RuntimeError(f"external source checksum mismatch: {name}")
    external_rows = load_active_rows(external["data_root"])
    protocol_path = Path(external["protocol"])
    if not (protocol_path / "code/dvr_qwen/eval_metrics.py").is_file():
        raise FileNotFoundError("shared-prefix evaluation code is incomplete")

    cache_status: dict[str, Any]
    ready_for_training = False
    persistent = config.get("protocol_version") == "four_action_persistent_polar_v1"
    if persistent:
        source_cache_audit_path = Path(config["visual_features"]["source_cache_audit"])
        source_cache_audit = read_json(source_cache_audit_path)
        if source_cache_audit.get("passed") is not True:
            raise RuntimeError("source visual cache audit did not pass")
        feature_manifest = Path(config["visual_features"]["manifest"])
        observed_feature_sha = file_sha256(feature_manifest)
        if observed_feature_sha != config["visual_features"]["manifest_sha256"]:
            raise RuntimeError("selected visual feature manifest checksum mismatch")
        load_verified_feature_index(
            feature_manifest,
            manifest_sha256=observed_feature_sha,
            expected_uids={str(row["uid"]) for row in manifest.rows},
            expected_feature_width=int(config["visual_features"]["feature_width"]),
            verify_tensors=False,
        )
        cache_status = {
            "present": True,
            "audit": str(source_cache_audit_path),
            "audit_sha256": file_sha256(source_cache_audit_path),
            "manifest": str(feature_manifest),
            "manifest_sha256": observed_feature_sha,
            "selected_from_passed_parent_cache": True,
        }
        ready_for_training = True
        cache_audit_path = source_cache_audit_path
    else:
        cache_audit_path = Path(config["visual_features"]["cache_audit"])
    if not persistent and cache_audit_path.is_file():
        cache_audit = read_json(cache_audit_path)
        expected_contract = visual_cache_contract(config)
        if cache_audit.get("passed") is not True:
            raise RuntimeError("visual cache audit did not pass")
        if cache_audit.get("cache_contract", {}).get("sha256") != expected_contract["sha256"]:
            raise RuntimeError("visual cache contract differs from the training config")
        feature_manifest = Path(config["visual_features"]["manifest"])
        observed_feature_sha = file_sha256(feature_manifest)
        if observed_feature_sha != cache_audit.get("manifest_sha256"):
            raise RuntimeError("visual cache audit does not bind the feature manifest")
        reused_without_copy = bool(cache_audit.get("tensor_files_reused_without_copy"))
        if reused_without_copy:
            source_audit_path = Path(cache_audit["source_feature_audit"])
            source_manifest_path = Path(cache_audit["source_feature_manifest"])
            source_audit = read_json(source_audit_path)
            if (
                file_sha256(source_audit_path)
                != cache_audit["source_feature_audit_sha256"]
                or file_sha256(source_manifest_path)
                != cache_audit["source_feature_manifest_sha256"]
                or source_audit.get("passed") is not True
                or source_audit.get("manifest_sha256")
                != cache_audit["source_feature_manifest_sha256"]
            ):
                raise RuntimeError("reused visual tensors lack valid parent cache provenance")
        load_verified_feature_index(
            feature_manifest,
            manifest_sha256=observed_feature_sha,
            expected_uids={str(row["uid"]) for row in manifest.rows},
            expected_feature_width=int(config["visual_features"]["feature_width"]),
            verify_tensors=not reused_without_copy,
        )
        cache_status = {
            "present": True,
            "audit": str(cache_audit_path),
            "audit_sha256": file_sha256(cache_audit_path),
            "manifest": str(feature_manifest),
            "manifest_sha256": observed_feature_sha,
            "contract_sha256": expected_contract["sha256"],
        }
        ready_for_training = True
    else:
        cache_status = {
            "present": False,
            "required_path": str(cache_audit_path),
            "contract": visual_cache_contract(config),
        }
        if args.require_visual_cache:
            raise FileNotFoundError(f"fresh visual cache is missing: {cache_audit_path}")

    payload = {
        "schema_version": "four_action_polar_preflight_v1",
        "passed": True,
        "ready_for_visual_cache_extraction": True,
        "ready_for_training": ready_for_training,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "objective": config["training"]["objective"],
        "modality": "image_question",
        "training_population": {
            "records": len(manifest),
            "split_counts": dict(split_counts),
            "routes": int(config["data"]["valid_routes"]),
            "zero_valid_route_exclusions": int(config["data"]["zero_valid_route_exclusions"]),
        },
        "predictor_smoke_shape": shape,
        "embedding_snapshot": str(encoder_path),
        "embedding_weight_files": [str(path) for path in safetensors],
        "base_model_snapshot": str(base_model_path.resolve()),
        "visual_cache": cache_status,
        "external_evaluation": {
            "records": len(external_rows),
            "benchmark_counts": dict(Counter(row["benchmark"] for row in external_rows)),
            "protocol": str(protocol_path),
        },
        "jobs_launched": [],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{file_sha256(output_path)}  {output_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
