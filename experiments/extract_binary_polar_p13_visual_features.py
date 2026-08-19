#!/usr/bin/env python3
"""Cache native pre-decoder projected visual rows for frozen P13 identities."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from binary_policy.executor.inputs import build_binary_inputs
from binary_policy.multimodal import deterministic_modality_permutations
from experiments.train_binary_polar import file_sha256
from label_regeneration.runtime import (
    build_native_processor_inputs,
    configure_determinism,
    load_frozen_model,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def distribution(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)

    def quantile(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    return {
        "minimum": min(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": quantile(0.5),
        "q90": quantile(0.9),
        "q95": quantile(0.95),
        "maximum": max(ordered),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p11-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    p11_config_path = Path(args.p11_config)
    config = yaml.safe_load(p11_config_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite P13 feature cache: {output_dir}")
    output_dir.mkdir(parents=True)
    tensor_dir = output_dir / "tensors"
    tensor_dir.mkdir()

    smoke_path = Path(config["smoke"]["manifest"])
    if file_sha256(smoke_path) != config["smoke"]["manifest_sha256"]:
        raise RuntimeError("P13 source smoke manifest checksum mismatch")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    selected_uids = list(
        dict.fromkeys(
            smoke["train_positive_uids"]
            + smoke["validation_positive_uids"]
            + smoke["execution_validation_uids"]
        )
    )
    predictor_path = Path(config["data"]["manifest"])
    if file_sha256(predictor_path) != config["data"]["manifest_sha256"]:
        raise RuntimeError("P13 predictor manifest checksum mismatch")
    predictor = {row["uid"]: row for row in read_jsonl(predictor_path)}
    source_path = Path("outputs/label_regeneration/v1/source_manifest_v1.jsonl")
    source = {row["uid"]: row for row in read_jsonl(source_path)}
    if set(selected_uids) - predictor.keys() or set(selected_uids) - source.keys():
        raise RuntimeError("P13 selected identity is absent from source or predictor manifest")

    groups: dict[str, list[str]] = {}
    for uid in selected_uids:
        row = predictor[uid]
        groups.setdefault(row["split_group"], []).append(uid)
        if row["prompt"] != source[uid]["prompt"]:
            raise RuntimeError(f"prompt mismatch for {uid}")
        if Path(row["image_path"]).resolve() != Path(source[uid]["local_image_path"]).resolve():
            raise RuntimeError(f"image path mismatch for {uid}")

    configure_determinism(int(config["training"]["seed"]))
    processor, base, wrapped, device = load_frozen_model(
        config["base_model"]["model_id"], config["base_model"]["revision"], 0
    )
    records_by_group = {}
    exact_repeat_checks = 0
    for group_index, (group, uids) in enumerate(sorted(groups.items())):
        uid = sorted(uids)[0]
        inputs, metadata = build_native_processor_inputs(processor, source[uid], device)
        prepared = build_binary_inputs(wrapped, inputs)
        feature = prepared.visual_states[0, prepared.visual_valid_mask[0]].detach().cpu().to(torch.bfloat16).contiguous()
        if feature.ndim != 2 or feature.shape[0] < 1 or not bool(torch.isfinite(feature).all()):
            raise RuntimeError(f"invalid projected visual feature for {uid}")
        if group_index < 3:
            repeated = build_binary_inputs(wrapped, inputs)
            repeated_feature = repeated.visual_states[0, repeated.visual_valid_mask[0]].detach().cpu().to(torch.bfloat16)
            if not torch.equal(feature, repeated_feature):
                raise RuntimeError(f"nondeterministic projected visual rows for {uid}")
            exact_repeat_checks += 1
        filename = f"{sha256(group.encode()).hexdigest()}.pt"
        tensor_path = tensor_dir / filename
        torch.save(feature, tensor_path)
        records_by_group[group] = {
            "representative_uid": uid,
            "path": str(tensor_path),
            "sha256": file_sha256(tensor_path),
            "shape": list(feature.shape),
            "dtype": str(feature.dtype),
            "visual_tokens": int(feature.shape[0]),
            "feature_width": int(feature.shape[1]),
            "prompt_sha256_used_for_native_layout": metadata["prompt_sha256"],
            "image_path": source[uid]["local_image_path"],
        }
        print(json.dumps({"group": group, "completed": group_index + 1, "total": len(groups)}), flush=True)

    manifest_path = output_dir / "feature_manifest_v1.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for uid in sorted(selected_uids):
            row = predictor[uid]
            record = records_by_group[row["split_group"]]
            handle.write(
                json.dumps(
                    {
                        "uid": uid,
                        "benchmark": row["benchmark"],
                        "split": row["split"],
                        "split_group": row["split_group"],
                        **record,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    validation_rows = [predictor[uid] for uid in smoke["validation_positive_uids"]]
    permutations = deterministic_modality_permutations(
        validation_rows, seed=int(config["p11"]["shuffle_seed"])
    )
    permutations_path = output_dir / "modality_permutations_v1.json"
    permutations_path.write_text(
        json.dumps(
            {
                "schema_version": "binary_polar_p13_modality_permutations_v1",
                "seed": int(config["p11"]["shuffle_seed"]),
                "validation_uids": smoke["validation_positive_uids"],
                "mapping": permutations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    counts = [record["visual_tokens"] for record in records_by_group.values()]
    widths = {record["feature_width"] for record in records_by_group.values()}
    audit = {
        "schema_version": "binary_polar_p13_visual_feature_cache_v1",
        "passed": True,
        "p11_config": str(p11_config_path),
        "p11_config_sha256": file_sha256(p11_config_path),
        "source_manifest": str(source_path),
        "source_manifest_sha256": file_sha256(source_path),
        "smoke_manifest": str(smoke_path),
        "smoke_manifest_sha256": file_sha256(smoke_path),
        "selected_records": len(selected_uids),
        "unique_image_groups": len(groups),
        "duplicate_group_records": len(selected_uids) - len(groups),
        "records_by_role": {
            "train": len(smoke["train_positive_uids"]),
            "validation": len(smoke["validation_positive_uids"]),
            "execution": len(smoke["execution_validation_uids"]),
            "execution_already_in_validation": len(
                set(smoke["execution_validation_uids"]) & set(smoke["validation_positive_uids"])
            ),
        },
        "records_by_benchmark": dict(Counter(predictor[uid]["benchmark"] for uid in selected_uids)),
        "feature_source": "Qwen2.5-VL get_image_features pooler_output projected rows entering decoder layer 0",
        "feature_dtype": "torch.bfloat16",
        "feature_widths": sorted(widths),
        "visual_token_count": distribution(counts),
        "visual_tokens_total_unique_images": sum(counts),
        "exact_repeat_checks": exact_repeat_checks,
        "exact_repeat_checks_passed": exact_repeat_checks,
        "answer_fields_consumed": [],
        "route_outcome_fields_consumed": [],
        "decoder_layers_executed": 0,
        "external_vision_model": None,
        "normal_inference_already_computes_feature": True,
        "custom_max_image_tokens": None,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "permutations": str(permutations_path),
        "permutations_sha256": file_sha256(permutations_path),
        "tensor_files": len(records_by_group),
        "tensor_checksums_sha256": sha256(
            "\n".join(
                f"{group} {records_by_group[group]['sha256']}" for group in sorted(records_by_group)
            ).encode()
        ).hexdigest(),
    }
    audit_path = output_dir / "cache_audit_v1.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in (manifest_path, permutations_path, audit_path):
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{file_sha256(path)}  {path.name}\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
