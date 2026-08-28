#!/usr/bin/env python3
"""Resumably extract fresh pre-decoder visual rows for four-action POLAR.

This cache is intentionally rebuilt from the checksum-bound eligible manifest;
it never reuses the imported binary/P13 cache and consumes no answer or route
outcome fields.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm.auto import tqdm
import yaml

from binary_policy.executor.inputs import build_binary_inputs
from experiments.train_binary_polar import file_sha256
from four_action_policy.feature_cache import visual_cache_contract
from label_regeneration.runtime import (
    build_native_processor_inputs,
    configure_determinism,
    load_frozen_model,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def distribution(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)

    def quantile(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return float(ordered[low])
        fraction = position - low
        return float(ordered[low] * (1 - fraction) + ordered[high] * fraction)

    return {
        "minimum": float(ordered[0]),
        "mean": sum(ordered) / len(ordered),
        "median": quantile(0.5),
        "p90": quantile(0.9),
        "p95": quantile(0.95),
        "maximum": float(ordered[-1]),
    }


def _content_sha256(path: Path) -> str:
    return file_sha256(path)


def _verified_group_images(groups: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Prove that differently named copies in one image group have equal bytes."""
    output = {}
    path_hashes: dict[str, str] = {}
    for group, rows in tqdm(
        sorted(groups.items()), desc="verify source images", unit="group", dynamic_ncols=True
    ):
        paths = sorted({str(row["image_path"]) for row in rows})
        hashes = set()
        for value in paths:
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(f"source image is missing: {path}")
            hashes.add(path_hashes.setdefault(value, _content_sha256(path)))
        if len(hashes) != 1:
            raise RuntimeError(f"split group maps to different image bytes: {group}")
        output[group] = {
            "representative_image_path": paths[0],
            "source_image_paths": paths,
            "image_content_sha256": next(iter(hashes)),
        }
    return output


def _load_completed_record(
    record_path: Path,
    *,
    group: str,
    image_content_sha256: str,
    expected_width: int,
) -> dict[str, Any] | None:
    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (
        record.get("split_group") != group
        or record.get("image_content_sha256") != image_content_sha256
        or int(record.get("feature_width", -1)) != expected_width
        or record.get("dtype") != "torch.bfloat16"
    ):
        raise RuntimeError(f"completed visual record has a stale contract: {record_path}")
    tensor_path = Path(record["path"])
    if not tensor_path.is_file() or file_sha256(tensor_path) != record.get("sha256"):
        raise RuntimeError(f"completed visual tensor failed checksum: {tensor_path}")
    tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if (
        not torch.is_tensor(tensor)
        or list(tensor.shape) != record.get("shape")
        or str(tensor.dtype) != record.get("dtype")
        or not bool(torch.isfinite(tensor).all())
    ):
        raise RuntimeError(f"completed visual tensor failed content audit: {tensor_path}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards > 0 and 0 <= shard_index < num_shards")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("modality") != "image_question":
        raise RuntimeError("four-action training config must freeze Image+Question modality")
    manifest_path = Path(config["data"]["manifest"])
    if file_sha256(manifest_path) != config["data"]["manifest_sha256"]:
        raise RuntimeError("four-action predictor manifest checksum mismatch")
    rows = read_jsonl(manifest_path)
    expected_train = int(config["data"]["train_records"])
    expected_validation = int(config["data"]["validation_records"])
    counts = Counter(str(row["split"]) for row in rows)
    if counts != Counter({"train": expected_train, "validation": expected_validation}):
        raise RuntimeError(f"manifest population differs from config: {counts}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["split_group"]), []).append(row)
    expected_groups = int(config["data"]["unique_image_groups"])
    if len(groups) != expected_groups:
        raise RuntimeError(f"image-group count differs from config: {len(groups)}")
    group_images = _verified_group_images(groups)

    output_dir = Path(args.output_dir)
    audit_path = output_dir / "cache_audit_v1.json"
    if audit_path.exists():
        raise FileExistsError(f"completed visual cache already exists: {audit_path}")
    if output_dir.exists() and not args.resume and args.num_shards == 1 and not args.finalize_only:
        raise FileExistsError(
            f"visual cache directory exists; pass --resume after inspection: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir = output_dir / "tensors"
    record_dir = output_dir / "records"
    tensor_dir.mkdir(exist_ok=True)
    record_dir.mkdir(exist_ok=True)

    feature_width = int(config["visual_features"]["feature_width"])
    records_by_group: dict[str, dict[str, Any]] = {}
    pending_all = []
    for group in sorted(groups):
        key = sha256(group.encode()).hexdigest()
        record = _load_completed_record(
            record_dir / f"{key}.json",
            group=group,
            image_content_sha256=group_images[group]["image_content_sha256"],
            expected_width=feature_width,
        )
        if record is None:
            pending_all.append(group)
        else:
            records_by_group[group] = record

    verified_before = len(records_by_group)
    sorted_groups = sorted(groups)
    assigned_groups = {
        group
        for index, group in enumerate(sorted_groups)
        if index % args.num_shards == args.shard_index
    }
    pending = [group for group in pending_all if group in assigned_groups]
    shard_metadata_path = output_dir / (
        f"extraction_shard_{args.shard_index:03d}_of_{args.num_shards:03d}.json"
    )
    if not args.finalize_only and shard_metadata_path.exists():
        if args.resume:
            print(shard_metadata_path.read_text(encoding="utf-8").strip())
            return
        raise FileExistsError(f"visual extraction shard is already complete: {shard_metadata_path}")

    configure_determinism(int(config["training"]["seed"]))
    if pending and not args.finalize_only:
        processor, _base, wrapped, device = load_frozen_model(
            config["base_model"]["path"],
            config["base_model"]["revision"],
            args.device_index,
        )
        exact_repeat_checks = 0
        progress = tqdm(pending, desc="four-action visual cache", unit="image", dynamic_ncols=True)
        for group in progress:
            row = sorted(groups[group], key=lambda item: str(item["uid"]))[0]
            group_image = group_images[group]
            sample = {
                "local_image_path": group_image["representative_image_path"],
                "image_content_sha256": group_image["image_content_sha256"],
                "prompt": row["prompt"],
            }
            inputs, metadata = build_native_processor_inputs(processor, sample, device)
            prepared = build_binary_inputs(wrapped, inputs)
            feature = (
                prepared.visual_states[0, prepared.visual_valid_mask[0]]
                .detach()
                .cpu()
                .to(torch.bfloat16)
                .contiguous()
            )
            if (
                feature.ndim != 2
                or feature.shape[0] < 1
                or feature.shape[1] != feature_width
                or not bool(torch.isfinite(feature).all())
            ):
                raise RuntimeError(f"invalid projected visual feature for {row['uid']}")
            if exact_repeat_checks < 3:
                repeated = build_binary_inputs(wrapped, inputs)
                repeated_feature = (
                    repeated.visual_states[0, repeated.visual_valid_mask[0]]
                    .detach()
                    .cpu()
                    .to(torch.bfloat16)
                )
                if not torch.equal(feature, repeated_feature):
                    raise RuntimeError(f"nondeterministic projected visual rows for {row['uid']}")
                exact_repeat_checks += 1
            key = sha256(group.encode()).hexdigest()
            tensor_path = tensor_dir / f"{key}.pt"
            temporary_tensor = tensor_path.with_suffix(".pt.tmp")
            torch.save(feature, temporary_tensor)
            temporary_tensor.replace(tensor_path)
            record = {
                "split_group": group,
                "representative_uid": row["uid"],
                "path": str(tensor_path),
                "sha256": file_sha256(tensor_path),
                "shape": list(feature.shape),
                "dtype": str(feature.dtype),
                "visual_tokens": int(feature.shape[0]),
                "feature_width": int(feature.shape[1]),
                "image_content_sha256": group_image["image_content_sha256"],
                "representative_image_path": group_image["representative_image_path"],
                "source_image_paths": group_image["source_image_paths"],
                "prompt_sha256_used_for_native_layout": metadata["prompt_sha256"],
            }
            write_json_atomic(record_dir / f"{key}.json", record)
            records_by_group[group] = record
            progress.set_postfix(done=len(records_by_group), total=len(groups))
    else:
        exact_repeat_checks = 0

    if not args.finalize_only:
        shard_metadata = {
            "schema_version": "four_action_polar_visual_extraction_shard_v1",
            "passed": True,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "assigned_image_groups": len(assigned_groups),
            "already_verified_image_groups": len(assigned_groups) - len(pending),
            "newly_extracted_image_groups": len(pending),
            "exact_repeat_checks": exact_repeat_checks,
            "cache_contract_sha256": visual_cache_contract(config)["sha256"],
        }
        write_json_atomic(shard_metadata_path, shard_metadata)
        if args.num_shards > 1:
            print(json.dumps(shard_metadata, sort_keys=True))
            return

    # Finalization is deliberately CPU-only and rescans every group so that a
    # multi-GPU extraction cannot publish a partial cache as complete.
    records_by_group = {}
    missing_groups = []
    for group in sorted_groups:
        key = sha256(group.encode()).hexdigest()
        record = _load_completed_record(
            record_dir / f"{key}.json",
            group=group,
            image_content_sha256=group_images[group]["image_content_sha256"],
            expected_width=feature_width,
        )
        if record is None:
            missing_groups.append(group)
        else:
            records_by_group[group] = record
    if missing_groups:
        raise RuntimeError(
            f"cannot finalize incomplete visual cache: {len(missing_groups)} image groups missing"
        )
    shard_metadata_files = sorted(output_dir.glob("extraction_shard_*_of_*.json"))
    shard_metadata_rows = [json.loads(path.read_text(encoding="utf-8")) for path in shard_metadata_files]
    if args.finalize_only and args.num_shards > 1:
        indices = {
            int(row["shard_index"])
            for row in shard_metadata_rows
            if int(row["num_shards"]) == args.num_shards
        }
        if indices != set(range(args.num_shards)):
            raise RuntimeError("cannot finalize: extraction shard metadata is incomplete")

    manifest_output = output_dir / "feature_manifest_v1.jsonl"
    temporary_manifest = manifest_output.with_suffix(".jsonl.tmp")
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: str(item["uid"])):
            record = records_by_group[str(row["split_group"])]
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
    temporary_manifest.replace(manifest_output)
    token_counts = [int(record["visual_tokens"]) for record in records_by_group.values()]
    audit = {
        "schema_version": "four_action_polar_visual_feature_cache_v1",
        "passed": True,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "cache_contract": visual_cache_contract(config),
        "predictor_manifest": str(manifest_path),
        "predictor_manifest_sha256": file_sha256(manifest_path),
        "records": len(rows),
        "train_records": counts["train"],
        "validation_records": counts["validation"],
        "unique_image_groups": len(groups),
        "resumed_verified_image_groups": verified_before,
        "newly_extracted_image_groups_this_invocation": (
            0 if args.finalize_only else len(pending)
        ),
        "extraction_shards": shard_metadata_rows,
        "records_by_benchmark": dict(Counter(row["benchmark"] for row in rows)),
        "feature_source": "projected visual rows entering decoder layer 0",
        "feature_dtype": "torch.bfloat16",
        "feature_width": feature_width,
        "visual_token_count": distribution(token_counts),
        "exact_repeat_checks_this_invocation": exact_repeat_checks,
        "exact_repeat_checks_across_shards": sum(
            int(row.get("exact_repeat_checks", 0)) for row in shard_metadata_rows
        ),
        "answer_fields_consumed": [],
        "route_fields_consumed": [],
        "decoder_layers_executed": 0,
        "old_visual_cache_reused": False,
        "manifest": str(manifest_output),
        "manifest_sha256": file_sha256(manifest_output),
        "tensor_files_total": len(records_by_group),
    }
    write_json_atomic(audit_path, audit)
    for path in (manifest_output, audit_path):
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{file_sha256(path)}  {path.name}\n", encoding="utf-8"
        )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
