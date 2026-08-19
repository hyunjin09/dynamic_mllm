#!/usr/bin/env python3
"""Cache native pre-decoder visual rows for every positive full10 input."""

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
from tqdm.auto import tqdm
import yaml

from binary_policy.executor.inputs import build_binary_inputs
from binary_policy.multimodal import deterministic_group_disjoint_modality_permutations
from experiments.train_binary_polar import file_sha256
from label_regeneration.runtime import configure_determinism, load_frozen_model


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
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--reuse-feature-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config_path = Path(args.source_config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite full visual cache: {output_dir}")
    output_dir.mkdir(parents=True)
    tensor_dir = output_dir / "tensors"
    tensor_dir.mkdir()

    predictor_path = Path(config["data"]["manifest"])
    if file_sha256(predictor_path) != config["data"]["manifest_sha256"]:
        raise RuntimeError("predictor manifest checksum mismatch")
    predictor_rows = read_jsonl(predictor_path)
    positives = [row for row in predictor_rows if row.get("valid_routes")]
    train = [row for row in positives if row["split"] == "train"]
    validation = [row for row in positives if row["split"] == "validation"]
    if len(train) != 6043 or len(validation) != 874:
        raise RuntimeError(
            f"unexpected positive population: train={len(train)} validation={len(validation)}"
        )
    source_path = Path("outputs/label_regeneration/v1/source_manifest_v1.jsonl")
    source = {row["uid"]: row for row in read_jsonl(source_path)}
    if {row["uid"] for row in positives} - source.keys():
        raise RuntimeError("positive predictor identity is absent from source manifest")

    groups: dict[str, list[dict]] = {}
    for row in positives:
        uid = row["uid"]
        groups.setdefault(row["split_group"], []).append(row)
        if row["prompt"] != source[uid]["prompt"]:
            raise RuntimeError(f"prompt mismatch for {uid}")
        if Path(row["image_path"]).resolve() != Path(source[uid]["local_image_path"]).resolve():
            raise RuntimeError(f"image mismatch for {uid}")
    if len(groups) != 6574:
        raise RuntimeError(f"unexpected positive image-group count: {len(groups)}")

    reuse_path = Path(args.reuse_feature_manifest)
    reused_by_group: dict[str, dict] = {}
    for row in read_jsonl(reuse_path):
        group = row["split_group"]
        prior = reused_by_group.setdefault(group, row)
        if prior["sha256"] != row["sha256"] or prior["path"] != row["path"]:
            raise RuntimeError(f"inconsistent reusable feature declarations for {group}")
    reusable = set(groups) & reused_by_group.keys()
    for group in reusable:
        record = reused_by_group[group]
        path = Path(record["path"])
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"reusable visual tensor checksum mismatch: {path}")

    configure_determinism(int(config["training"]["seed"]))
    processor, _base, wrapped, device = load_frozen_model(
        config["base_model"]["model_id"], config["base_model"]["revision"], 0
    )
    records_by_group: dict[str, dict] = {
        group: {
            key: reused_by_group[group][key]
            for key in (
                "representative_uid",
                "path",
                "sha256",
                "shape",
                "dtype",
                "visual_tokens",
                "feature_width",
                "prompt_sha256_used_for_native_layout",
                "image_path",
            )
        }
        for group in reusable
    }
    pending = [group for group in sorted(groups) if group not in reusable]
    exact_repeat_checks = 0
    progress = tqdm(pending, desc="full visual cache", unit="image", dynamic_ncols=True)
    for group in progress:
        row = sorted(groups[group], key=lambda item: item["uid"])[0]
        uid = row["uid"]
        from label_regeneration.runtime import build_native_processor_inputs

        inputs, metadata = build_native_processor_inputs(processor, source[uid], device)
        prepared = build_binary_inputs(wrapped, inputs)
        feature = (
            prepared.visual_states[0, prepared.visual_valid_mask[0]]
            .detach()
            .cpu()
            .to(torch.bfloat16)
            .contiguous()
        )
        if feature.ndim != 2 or feature.shape[0] < 1 or not bool(torch.isfinite(feature).all()):
            raise RuntimeError(f"invalid projected visual feature for {uid}")
        if exact_repeat_checks < 3:
            repeated = build_binary_inputs(wrapped, inputs)
            repeated_feature = (
                repeated.visual_states[0, repeated.visual_valid_mask[0]]
                .detach()
                .cpu()
                .to(torch.bfloat16)
            )
            if not torch.equal(feature, repeated_feature):
                raise RuntimeError(f"nondeterministic projected visual rows for {uid}")
            exact_repeat_checks += 1
        tensor_path = tensor_dir / f"{sha256(group.encode()).hexdigest()}.pt"
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
        progress.set_postfix(done=len(records_by_group), total=len(groups))

    manifest_path = output_dir / "feature_manifest_v1.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in sorted(positives, key=lambda item: item["uid"]):
            record = records_by_group[row["split_group"]]
            handle.write(
                json.dumps(
                    {
                        "uid": row["uid"],
                        "benchmark": row["benchmark"],
                        "split": row["split"],
                        "split_group": row["split_group"],
                        **record,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    permutations = deterministic_group_disjoint_modality_permutations(
        validation, seed=int(config["p13"]["shuffle_seed"])
    )
    permutation_path = output_dir / "validation_modality_permutations_v1.json"
    permutation_path.write_text(
        json.dumps(
            {
                "schema_version": "binary_polar_full10_validation_permutations_v1",
                "seed": int(config["p13"]["shuffle_seed"]),
                "validation_uids": [row["uid"] for row in validation],
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
        "schema_version": "binary_polar_full10_visual_feature_cache_v1",
        "passed": True,
        "source_config": str(config_path),
        "source_config_sha256": file_sha256(config_path),
        "predictor_manifest": str(predictor_path),
        "predictor_manifest_sha256": file_sha256(predictor_path),
        "source_manifest": str(source_path),
        "source_manifest_sha256": file_sha256(source_path),
        "positive_records": len(positives),
        "train_positive_records": len(train),
        "validation_positive_records": len(validation),
        "unique_image_groups": len(groups),
        "reused_verified_image_groups": len(reusable),
        "newly_extracted_image_groups": len(pending),
        "records_by_benchmark": dict(Counter(row["benchmark"] for row in positives)),
        "feature_source": "projected visual rows entering decoder layer 0",
        "feature_dtype": "torch.bfloat16",
        "feature_widths": sorted(widths),
        "visual_token_count": distribution(counts),
        "exact_repeat_checks": exact_repeat_checks,
        "exact_repeat_checks_passed": exact_repeat_checks,
        "answer_fields_consumed": [],
        "route_outcome_fields_consumed": [],
        "decoder_layers_executed": 0,
        "external_vision_model": None,
        "custom_max_image_tokens": None,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "validation_permutations": str(permutation_path),
        "validation_permutations_sha256": file_sha256(permutation_path),
        "tensor_files_total": len(records_by_group),
    }
    audit_path = output_dir / "cache_audit_v1.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in (manifest_path, permutation_path, audit_path):
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{file_sha256(path)}  {path.name}\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
