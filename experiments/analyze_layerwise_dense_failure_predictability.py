#!/usr/bin/env python3
"""Run the frozen 28-layer current-dense failure linear-probe diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dense_failure_stage1.layerwise_probe import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    FEATURE_NAMES,
    binary_metrics,
    compose_layer_features,
    fit_linear_probe,
    preservation_operating_point,
    score_linear_probe,
    validate_checkpoint_provenance,
    validate_complete_layer_records,
)
from tools.research_analysis.dense_failure_stage1 import (  # noqa: E402
    build_image_group_disjoint_split,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/layerwise_dense_failure_probe_v1.json"
SPLITS = ("train", "val", "test")
DATASETS = ("gqa", "chartqa", "textvqa")
BOUND_CODE_PATHS = (
    "configs/layerwise_dense_failure_probe_v1.json",
    "dense_failure_stage1/layerwise_probe.py",
    "experiments/analyze_layerwise_dense_failure_predictability.py",
    "tools/research_analysis/dense_failure_stage1.py",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    _atomic_bytes(path, payload.encode("utf-8"))


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_once_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite incompatible frozen artifact: {path}")
        return
    _atomic_bytes(path, payload)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")


def resolve_repo_path(value: str) -> Path:
    path = (PROJECT_ROOT / value).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path escapes project root: {value}")
    return path


def command_output(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result.stdout.strip()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def load_static_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "layerwise_dense_failure_probe_config_v1":
        raise ValueError("unsupported static layerwise-probe config")
    if tuple(config["input"]["features"]) != FEATURE_NAMES:
        raise ValueError("static config does not use the frozen feature order")
    if int(config["input"]["input_size"]) != 3 * int(
        config["input"]["hidden_size_per_summary"]
    ):
        raise ValueError("static config input size is inconsistent")
    if sum(int(value) for value in config["split"]["targets"].values()) != int(
        config["population"]["records"]
    ):
        raise ValueError("static split targets do not cover the population")
    return config


def _population_rows(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dense_path = resolve_repo_path(config["sources"]["dense_outputs"])
    index_path = resolve_repo_path(config["sources"]["feature_index"])
    schema_path = resolve_repo_path(config["sources"]["feature_schema"])
    dense_rows = read_jsonl(dense_path)
    index_rows = read_jsonl(index_path)
    schema = read_json(schema_path)
    expected_records = int(config["population"]["records"])
    if len(dense_rows) != expected_records or len(index_rows) != expected_records:
        raise ValueError("dense output and feature index must each cover 7,999 rows")
    dense_by_uid = {str(row["uid"]): row for row in dense_rows}
    index_by_uid = {str(row["uid"]): row for row in index_rows}
    if len(dense_by_uid) != len(dense_rows) or len(index_by_uid) != len(index_rows):
        raise ValueError("duplicate UID in dense outputs or feature index")
    if set(dense_by_uid) != set(index_by_uid):
        raise ValueError("dense output and feature-index UID sets differ")
    if schema.get("layers") != list(range(28)) or int(schema.get("hidden_size", -1)) != 3584:
        raise ValueError("feature schema layer/hidden dimensions differ")
    for name in FEATURE_NAMES:
        if not isinstance(schema.get(name), str):
            raise ValueError(f"feature schema is missing {name}")

    rows = []
    for uid in sorted(dense_by_uid):
        dense = dense_by_uid[uid]
        index = index_by_uid[uid]
        dataset = str(dense["dataset"])
        if dataset not in DATASETS:
            raise ValueError(f"unexpected Stage-1 dataset: {dataset}")
        correct = bool(dense["current_dense_correct"])
        wrong = bool(dense["current_dense_wrong"])
        if correct == wrong:
            raise ValueError(f"dense labels are not complementary for {uid}")
        if int(index["layers"]) != 28:
            raise ValueError(f"feature index has wrong layer count for {uid}")
        shard_path = resolve_repo_path(str(index["shard"]))
        if not shard_path.is_file():
            raise ValueError(f"feature shard is missing: {shard_path}")
        rows.append(
            {
                "uid": uid,
                "dataset": dataset,
                "image_group_id": str(dense["image_group_id"]),
                "image_content_sha256": str(dense["image_content_sha256"]),
                "current_dense_correct": correct,
                "current_dense_wrong": wrong,
                "feature_shard": str(index["shard"]),
                "feature_row_index": int(index["row_index"]),
            }
        )
    observed = {
        "records": len(rows),
        "correct": sum(row["current_dense_correct"] for row in rows),
        "wrong": sum(row["current_dense_wrong"] for row in rows),
        "datasets": dict(sorted(Counter(row["dataset"] for row in rows).items())),
    }
    expected = config["population"]
    if observed != {
        "records": int(expected["records"]),
        "correct": int(expected["correct"]),
        "wrong": int(expected["wrong"]),
        "datasets": {key: int(value) for key, value in sorted(expected["datasets"].items())},
    }:
        raise ValueError(f"Stage-1 population differs: {observed}")
    return rows, schema


def split_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    split_uids = {
        split: {str(row["uid"]) for row in rows if row["split"] == split}
        for split in SPLITS
    }
    split_groups = {
        split: {str(row["image_group_id"]) for row in rows if row["split"] == split}
        for split in SPLITS
    }
    uid_overlap = sum(
        len(split_uids[left] & split_uids[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    )
    group_overlap = sum(
        len(split_groups[left] & split_groups[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    )
    counts = {}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        counts[split] = {
            "records": len(selected),
            "image_groups": len(split_groups[split]),
            "correct": sum(bool(row["current_dense_correct"]) for row in selected),
            "wrong": sum(bool(row["current_dense_wrong"]) for row in selected),
            "cells": {
                f"{dataset}_{outcome}": sum(
                    row["dataset"] == dataset
                    and bool(row["current_dense_wrong"]) == (outcome == "wrong")
                    for row in selected
                )
                for dataset in DATASETS
                for outcome in ("correct", "wrong")
            },
        }
    return {
        "schema_version": "layerwise_dense_failure_split_audit_v1",
        "passed": len({str(row["uid"]) for row in rows}) == len(rows)
        and uid_overlap == 0
        and group_overlap == 0,
        "records": len(rows),
        "unique_uids": len({str(row["uid"]) for row in rows}),
        "unique_image_groups": len({str(row["image_group_id"]) for row in rows}),
        "uid_overlap": uid_overlap,
        "image_group_overlap": group_overlap,
        "splits": counts,
    }


def render_split_audit(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Stage-1 Layer-Wise Probe Split Audit",
        "",
        f"- Passed: `{audit['passed']}`",
        f"- Unique UIDs: `{audit['unique_uids']:,}`",
        f"- Unique image groups: `{audit['unique_image_groups']:,}`",
        f"- UID overlap: `{audit['uid_overlap']}`",
        f"- Image-group overlap: `{audit['image_group_overlap']}`",
        "",
        "| Split | GQA C | GQA W | ChartQA C | ChartQA W | TextVQA C | TextVQA W | Total | Groups |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        row = audit["splits"][split]
        cells = row["cells"]
        lines.append(
            f"| {split} | {cells['gqa_correct']} | {cells['gqa_wrong']} | "
            f"{cells['chartqa_correct']} | {cells['chartqa_wrong']} | "
            f"{cells['textvqa_correct']} | {cells['textvqa_wrong']} | "
            f"{row['records']} | {row['image_groups']} |"
        )
    return "\n".join(lines) + "\n"


def prepare(config_path: Path) -> None:
    static = load_static_config(config_path)
    output_root = resolve_repo_path(static["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    for directory in ("checkpoints", "figures", "training_history", "workers"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    population, feature_schema = _population_rows(static)
    split_rows = build_image_group_disjoint_split(
        population,
        targets={key: int(value) for key, value in static["split"]["targets"].items()},
        seed=int(static["seed"]),
    )
    split_rows = [
        {"schema_version": "layerwise_dense_failure_split_v1", **row}
        for row in split_rows
    ]
    audit = split_audit(split_rows)
    if not audit["passed"]:
        raise RuntimeError(f"image-group split audit failed: {audit}")
    observed_targets = Counter(str(row["split"]) for row in split_rows)
    if observed_targets != Counter(
        {key: int(value) for key, value in static["split"]["targets"].items()}
    ):
        raise RuntimeError(f"split target counts differ: {observed_targets}")

    split_path = output_root / "split_manifest.jsonl"
    split_payload = _jsonl_bytes(split_rows)
    write_once_or_verify(split_path, split_payload)
    audit_path = output_root / "split_audit.md"
    write_once_or_verify(audit_path, render_split_audit(audit).encode("utf-8"))
    atomic_json(output_root / "split_audit.json", audit)

    shard_paths = sorted({str(row["feature_shard"]) for row in split_rows})
    shard_rows = []
    for relative in shard_paths:
        path = resolve_repo_path(relative)
        shard_rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    shard_manifest = {
        "schema_version": "layerwise_dense_failure_feature_shard_manifest_v1",
        "shards": shard_rows,
        "records": len(split_rows),
    }
    shard_manifest_path = output_root / "feature_shard_manifest.json"
    write_once_or_verify(shard_manifest_path, _json_bytes(shard_manifest))

    status = command_output(["git", "status", "--porcelain=v1", "--untracked-files=all"])
    source_hashes = {
        name: file_sha256(resolve_repo_path(path)) for name, path in static["sources"].items()
    }
    bound_hashes = {
        path: file_sha256(resolve_repo_path(path)) for path in BOUND_CODE_PATHS
    }
    frozen = json.loads(json.dumps(static))
    frozen["schema_version"] = "layerwise_dense_failure_probe_frozen_contract_v1"
    frozen["provenance"] = {
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_branch": command_output(["git", "branch", "--show-current"]),
        "git_status_porcelain_at_freeze": status.splitlines() if status else [],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "matplotlib": package_version("matplotlib"),
        "cuda_runtime": torch.version.cuda,
        "gpu_inventory": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines(),
        "static_config_sha256": file_sha256(config_path),
        "source_sha256": source_hashes,
        "bound_code_sha256": bound_hashes,
        "split_manifest_sha256": file_sha256(split_path),
        "feature_shard_manifest_sha256": file_sha256(shard_manifest_path),
        "feature_schema_payload_sha256": canonical_hash(feature_schema),
    }
    frozen["contract_sha256"] = canonical_hash(frozen)
    contract_path = output_root / "probe_config.json"
    write_once_or_verify(contract_path, _json_bytes(frozen))
    atomic_json(
        output_root / "preparation_audit.json",
        {
            "passed": True,
            "contract_sha256": frozen["contract_sha256"],
            "split_manifest_sha256": file_sha256(split_path),
            "feature_shards": len(shard_rows),
            "feature_shard_bytes": sum(int(row["bytes"]) for row in shard_rows),
            "split_audit": audit,
        },
    )
    print(
        json.dumps(
            {
                "contract_sha256": frozen["contract_sha256"],
                "split_counts": dict(observed_targets),
                "feature_shards": len(shard_rows),
            },
            sort_keys=True,
        )
    )


def load_frozen_contract(config_path: Path) -> tuple[dict[str, Any], Path]:
    static = load_static_config(config_path)
    output_root = resolve_repo_path(static["output_root"])
    contract = read_json(output_root / "probe_config.json")
    if contract.get("schema_version") != "layerwise_dense_failure_probe_frozen_contract_v1":
        raise ValueError("unsupported frozen probe contract")
    if contract.get("contract_sha256") != canonical_hash(contract):
        raise ValueError("frozen probe contract hash differs")
    provenance = contract["provenance"]
    if provenance["static_config_sha256"] != file_sha256(config_path):
        raise ValueError("static probe config differs from frozen contract")
    for path, expected in provenance["bound_code_sha256"].items():
        if file_sha256(resolve_repo_path(path)) != expected:
            raise ValueError(f"bound probe source differs: {path}")
    for name, expected in provenance["source_sha256"].items():
        if file_sha256(resolve_repo_path(contract["sources"][name])) != expected:
            raise ValueError(f"probe input source differs: {name}")
    split_path = output_root / "split_manifest.jsonl"
    shard_manifest_path = output_root / "feature_shard_manifest.json"
    if file_sha256(split_path) != provenance["split_manifest_sha256"]:
        raise ValueError("frozen split manifest differs")
    if file_sha256(shard_manifest_path) != provenance["feature_shard_manifest_sha256"]:
        raise ValueError("frozen feature shard manifest differs")
    return contract, output_root


def checkpoint_expected(contract: Mapping[str, Any]) -> dict[str, str]:
    source = contract["provenance"]["source_sha256"]
    return {
        "contract_sha256": str(contract["contract_sha256"]),
        "split_manifest_sha256": str(contract["provenance"]["split_manifest_sha256"]),
        "feature_schema_sha256": str(source["feature_schema"]),
        "feature_index_sha256": str(source["feature_index"]),
        "dense_outputs_sha256": str(source["dense_outputs"]),
    }


def load_verified_shard(path: Path, expected_sha256: str) -> dict[str, Any]:
    payload = path.read_bytes()
    if bytes_sha256(payload) != expected_sha256:
        raise ValueError(f"feature shard hash differs: {path}")
    value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("layer_ids") != list(range(28)):
        raise ValueError(f"feature shard schema/layers differ: {path}")
    if int(value.get("records", -1)) != len(value.get("uids", [])):
        raise ValueError(f"feature shard record count differs: {path}")
    return value


def load_layer_matrices(
    contract: Mapping[str, Any],
    output_root: Path,
    *,
    layers: Sequence[int],
    selected_splits: set[str],
) -> tuple[list[dict[str, Any]], dict[int, torch.Tensor]]:
    split_rows = [
        row
        for row in read_jsonl(output_root / "split_manifest.jsonl")
        if str(row["split"]) in selected_splits
    ]
    split_rows.sort(key=lambda row: str(row["uid"]))
    if not split_rows:
        raise ValueError("no rows selected for feature loading")
    input_size = int(contract["input"]["input_size"])
    matrices = {
        int(layer): torch.empty((len(split_rows), input_size), dtype=torch.bfloat16)
        for layer in layers
    }
    filled = torch.zeros(len(split_rows), dtype=torch.bool)
    by_shard: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for destination, row in enumerate(split_rows):
        by_shard[str(row["feature_shard"])].append((destination, row))
    shard_manifest = read_json(output_root / "feature_shard_manifest.json")
    shard_hashes = {str(row["path"]): str(row["sha256"]) for row in shard_manifest["shards"]}
    if set(by_shard) - set(shard_hashes):
        raise ValueError("selected split references an unfrozen feature shard")

    for relative in sorted(by_shard):
        shard = load_verified_shard(resolve_repo_path(relative), shard_hashes[relative])
        selections = by_shard[relative]
        destinations = torch.tensor([item[0] for item in selections], dtype=torch.long)
        row_indices = [int(item[1]["feature_row_index"]) for item in selections]
        for (_, row), source_index in zip(selections, row_indices):
            if not 0 <= source_index < len(shard["uids"]):
                raise ValueError(f"feature row index lies outside shard for {row['uid']}")
            if str(shard["uids"][source_index]) != str(row["uid"]):
                raise ValueError(f"feature shard UID mismatch for {row['uid']}")
        for layer in layers:
            values = compose_layer_features(shard, row_indices=row_indices, layer=int(layer))
            if values.shape != (len(selections), input_size):
                raise ValueError(f"composed feature shape differs for layer {layer}")
            matrices[int(layer)].index_copy_(0, destinations, values.to(torch.bfloat16))
        filled.index_fill_(0, destinations, True)
    if not bool(filled.all()):
        raise RuntimeError("not every selected UID received one feature row")
    return split_rows, matrices


def _jsonable_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in metrics.items()
    }


def _dataset_metrics(
    rows: Sequence[Mapping[str, Any]], labels: np.ndarray, scores: np.ndarray
) -> dict[str, dict[str, Any]]:
    output = {}
    for dataset in DATASETS:
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row["dataset"] == dataset],
            dtype=np.int64,
        )
        output[dataset] = _jsonable_metrics(binary_metrics(labels[indices], scores[indices]))
    return output


def train_worker(config_path: Path, *, rank: int, world_size: int) -> None:
    contract, output_root = load_frozen_contract(config_path)
    if world_size != 4 or not 0 <= rank < world_size:
        raise ValueError("the frozen run requires four direct workers")
    assigned = list(range(rank, 28, world_size))
    expected = checkpoint_expected(contract)
    summaries = []
    missing_layers = []
    for layer in assigned:
        checkpoint_path = output_root / "checkpoints" / f"layer_{layer:02d}.pt"
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            validate_checkpoint_provenance(checkpoint, layer=layer, expected=expected)
            summary = dict(checkpoint["summary"])
            summary["checkpoint_sha256"] = file_sha256(checkpoint_path)
            summaries.append(summary)
        else:
            missing_layers.append(layer)

    device = torch.device(f"cuda:{rank}")
    if missing_layers:
        if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
            raise RuntimeError("four CUDA devices are not visible to the probe worker")
        torch.cuda.set_device(device)
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(8)
        rows, matrices = load_layer_matrices(
            contract,
            output_root,
            layers=missing_layers,
            selected_splits={"train", "val"},
        )
        train_indices = torch.tensor(
            [index for index, row in enumerate(rows) if row["split"] == "train"],
            dtype=torch.long,
        )
        val_indices = torch.tensor(
            [index for index, row in enumerate(rows) if row["split"] == "val"],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [int(bool(row["current_dense_wrong"])) for row in rows], dtype=torch.int64
        )
        training_config = dict(contract["training"])
        training_config["normalization_std_floor"] = float(
            contract["input"]["normalization_std_floor"]
        )
        for layer in missing_layers:
            matrix = matrices.pop(layer)
            fit = fit_linear_probe(
                matrix.index_select(0, train_indices),
                labels.index_select(0, train_indices),
                matrix.index_select(0, val_indices),
                labels.index_select(0, val_indices),
                config=training_config,
                seed=int(contract["seed"]) + int(layer),
                device=device,
            )
            train_scores = score_linear_probe(
                matrix.index_select(0, train_indices), fit, device=device
            ).numpy()
            val_scores = score_linear_probe(
                matrix.index_select(0, val_indices), fit, device=device
            ).numpy()
            train_labels = labels.index_select(0, train_indices).numpy()
            val_labels = labels.index_select(0, val_indices).numpy()
            val_rows = [rows[int(index)] for index in val_indices.tolist()]
            train_metrics = _jsonable_metrics(binary_metrics(train_labels, train_scores))
            validation_metrics = _jsonable_metrics(binary_metrics(val_labels, val_scores))
            selective = {
                f"{float(target):.2f}": _jsonable_metrics(
                    preservation_operating_point(
                        val_labels, val_scores, target_preservation=float(target)
                    )
                )
                for target in contract["evaluation"]["preservation_targets"]
            }
            summary = {
                "schema_version": "layerwise_dense_failure_probe_layer_summary_v1",
                "layer": layer,
                "worker_rank": rank,
                "seed": int(contract["seed"]) + layer,
                "best_epoch": int(fit["best_epoch"]),
                "epochs_ran": int(fit["epochs_ran"]),
                "train_loss": float(fit["train_loss"]),
                "validation_loss": float(fit["validation_loss"]),
                "train": train_metrics,
                "validation": validation_metrics,
                "validation_selective": selective,
                "validation_by_dataset": _dataset_metrics(val_rows, val_labels, val_scores),
            }
            checkpoint = {
                "schema_version": CHECKPOINT_SCHEMA,
                "layer": layer,
                "contract_sha256": expected["contract_sha256"],
                "split_manifest_sha256": expected["split_manifest_sha256"],
                "feature_schema_sha256": expected["feature_schema_sha256"],
                "feature_index_sha256": expected["feature_index_sha256"],
                "dense_outputs_sha256": expected["dense_outputs_sha256"],
                "input_features": list(FEATURE_NAMES),
                "input_size": int(contract["input"]["input_size"]),
                "normalization_mean": fit["mean"],
                "normalization_std": fit["std"],
                "weight": fit["weight"],
                "bias": fit["bias"],
                "summary": summary,
            }
            checkpoint_path = output_root / "checkpoints" / f"layer_{layer:02d}.pt"
            atomic_torch_save(checkpoint_path, checkpoint)
            summary["checkpoint_sha256"] = file_sha256(checkpoint_path)
            atomic_json(
                output_root / "training_history" / f"layer_{layer:02d}.json",
                {
                    "layer": layer,
                    "contract_sha256": expected["contract_sha256"],
                    "best_epoch": int(fit["best_epoch"]),
                    "epochs_ran": int(fit["epochs_ran"]),
                    "history": fit["history"],
                },
            )
            summaries.append(summary)
            del matrix, fit, checkpoint
            torch.cuda.empty_cache()

    summaries = sorted(summaries, key=lambda row: int(row["layer"]))
    if [int(row["layer"]) for row in summaries] != assigned:
        raise RuntimeError("worker does not have exactly its assigned layer summaries")
    worker_path = output_root / "workers" / f"train_rank{rank:02d}.jsonl"
    atomic_jsonl(worker_path, summaries)
    atomic_json(
        output_root / "workers" / f"train_rank{rank:02d}.complete.json",
        {
            "passed": True,
            "rank": rank,
            "world_size": world_size,
            "contract_sha256": contract["contract_sha256"],
            "layers": assigned,
            "records": len(summaries),
            "result_sha256": file_sha256(worker_path),
        },
    )
    print(json.dumps({"rank": rank, "layers": assigned, "passed": True}))


def aggregate_validation(config_path: Path) -> None:
    contract, output_root = load_frozen_contract(config_path)
    records = []
    expected = checkpoint_expected(contract)
    for rank in range(4):
        complete = read_json(output_root / "workers" / f"train_rank{rank:02d}.complete.json")
        worker_path = output_root / "workers" / f"train_rank{rank:02d}.jsonl"
        if (
            complete.get("passed") is not True
            or complete.get("contract_sha256") != contract["contract_sha256"]
            or complete.get("result_sha256") != file_sha256(worker_path)
        ):
            raise RuntimeError(f"training worker {rank} is incomplete or incompatible")
        records.extend(read_jsonl(worker_path))
    records = validate_complete_layer_records(records)
    for record in records:
        layer = int(record["layer"])
        checkpoint_path = output_root / "checkpoints" / f"layer_{layer:02d}.pt"
        if record.get("checkpoint_sha256") != file_sha256(checkpoint_path):
            raise RuntimeError(f"checkpoint hash differs for layer {layer}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        validate_checkpoint_provenance(checkpoint, layer=layer, expected=expected)

    metric_rows = []
    selective_rows = []
    dataset_rows = []
    for record in records:
        layer = int(record["layer"])
        validation = record["validation"]
        metric_rows.append(
            {
                "layer": layer,
                "best_epoch": int(record["best_epoch"]),
                "epochs_ran": int(record["epochs_ran"]),
                "train_loss": float(record["train_loss"]),
                "validation_loss": float(record["validation_loss"]),
                "validation_auroc": float(validation["auroc"]),
                "validation_auprc": float(validation["auprc"]),
                "validation_balanced_accuracy": float(validation["balanced_accuracy"]),
            }
        )
        selective_row: dict[str, Any] = {"layer": layer}
        for target in contract["evaluation"]["preservation_targets"]:
            key = f"{float(target):.2f}"
            suffix = str(int(round(float(target) * 100)))
            point = record["validation_selective"][key]
            selective_row.update(
                {
                    f"threshold_at_{suffix}_preserve": float(point["threshold"]),
                    f"actual_correct_preservation_at_{suffix}": float(
                        point["correct_preservation"]
                    ),
                    f"wrong_recall_at_{suffix}_preserve": float(point["wrong_recall"]),
                    f"failure_precision_at_{suffix}_preserve": float(
                        point["failure_precision"]
                    ),
                    f"false_deviation_rate_at_{suffix}_preserve": float(
                        point["false_deviation_rate"]
                    ),
                }
            )
        selective_rows.append(selective_row)
        for dataset in DATASETS:
            values = record["validation_by_dataset"][dataset]
            dataset_rows.append(
                {
                    "split": "validation",
                    "layer": layer,
                    "dataset": dataset,
                    "records": int(values["records"]),
                    "correct": int(values["correct"]),
                    "wrong": int(values["wrong"]),
                    "auroc": float(values["auroc"]),
                    "auprc": float(values["auprc"]),
                    "balanced_accuracy": float(values["balanced_accuracy"]),
                }
            )
    atomic_csv(output_root / "layerwise_probe_metrics.csv", metric_rows)
    atomic_csv(output_root / "layerwise_selective_metrics.csv", selective_rows)
    atomic_csv(output_root / "dataset_layerwise_metrics.csv", dataset_rows)
    atomic_json(
        output_root / "validation_results.json",
        {
            "schema_version": "layerwise_dense_failure_validation_results_v1",
            "contract_sha256": contract["contract_sha256"],
            "test_evaluated": False,
            "layers": records,
            "metric_table_sha256": file_sha256(output_root / "layerwise_probe_metrics.csv"),
            "selective_table_sha256": file_sha256(
                output_root / "layerwise_selective_metrics.csv"
            ),
        },
    )
    print(
        json.dumps(
            {
                "layers": len(records),
                "best_validation_layer": max(
                    records, key=lambda row: float(row["validation"]["auroc"])
                )["layer"],
                "test_evaluated": False,
            }
        )
    )


def freeze_region(
    config_path: Path, *, start: int, end: int, rationale: str
) -> None:
    contract, output_root = load_frozen_contract(config_path)
    if not 0 <= start <= end <= 27:
        raise ValueError("candidate region must lie within layers 0-27")
    validation_path = output_root / "validation_results.json"
    validation = read_json(validation_path)
    if validation.get("test_evaluated") is not False:
        raise ValueError("candidate region must be frozen before test evaluation")
    if any((output_root / "workers" / f"test_rank{rank:02d}.complete.json").exists() for rank in range(4)):
        raise RuntimeError("test artifacts already exist before region freeze")
    decision = {
        "schema_version": "layerwise_dense_failure_validation_region_v1",
        "contract_sha256": contract["contract_sha256"],
        "validation_results_sha256": file_sha256(validation_path),
        "selected_start": int(start),
        "selected_end": int(end),
        "selected_layers": list(range(int(start), int(end) + 1)),
        "rationale": rationale,
        "test_metrics_seen": False,
    }
    path = output_root / "validation_region_decision.json"
    write_once_or_verify(path, _json_bytes(decision))
    print(json.dumps(decision, sort_keys=True))


def test_worker(config_path: Path, *, rank: int, world_size: int) -> None:
    contract, output_root = load_frozen_contract(config_path)
    if world_size != 4 or not 0 <= rank < world_size:
        raise ValueError("the frozen test run requires four direct workers")
    region_path = output_root / "validation_region_decision.json"
    region = read_json(region_path)
    if (
        region.get("contract_sha256") != contract["contract_sha256"]
        or region.get("test_metrics_seen") is not False
    ):
        raise ValueError("validation region is not a compatible pre-test decision")
    worker_path = output_root / "workers" / f"test_rank{rank:02d}.jsonl"
    complete_path = output_root / "workers" / f"test_rank{rank:02d}.complete.json"
    if worker_path.is_file() and complete_path.is_file():
        complete = read_json(complete_path)
        if (
            complete.get("passed") is True
            and complete.get("contract_sha256") == contract["contract_sha256"]
            and complete.get("validation_region_sha256") == file_sha256(region_path)
            and complete.get("result_sha256") == file_sha256(worker_path)
        ):
            print(json.dumps({"rank": rank, "reused": True, "passed": True}))
            return
        raise RuntimeError(f"incompatible partial test-worker result for rank {rank}")

    assigned = list(range(rank, 28, world_size))
    expected = checkpoint_expected(contract)
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError("four CUDA devices are not visible to the test worker")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)
    rows, matrices = load_layer_matrices(
        contract, output_root, layers=assigned, selected_splits={"test"}
    )
    labels = np.asarray(
        [int(bool(row["current_dense_wrong"])) for row in rows], dtype=np.int64
    )
    results = []
    for layer in assigned:
        checkpoint_path = output_root / "checkpoints" / f"layer_{layer:02d}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        validate_checkpoint_provenance(checkpoint, layer=layer, expected=expected)
        state = {
            "mean": checkpoint["normalization_mean"],
            "std": checkpoint["normalization_std"],
            "weight": checkpoint["weight"],
            "bias": checkpoint["bias"],
        }
        scores = score_linear_probe(matrices.pop(layer), state, device=device).numpy()
        selective = {}
        for target in contract["evaluation"]["preservation_targets"]:
            key = f"{float(target):.2f}"
            threshold = float(
                checkpoint["summary"]["validation_selective"][key]["threshold"]
            )
            values = _jsonable_metrics(binary_metrics(labels, scores, threshold=threshold))
            values["target_validation_preservation"] = float(target)
            selective[key] = values
        results.append(
            {
                "schema_version": "layerwise_dense_failure_probe_test_summary_v1",
                "layer": layer,
                "worker_rank": rank,
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "overall": _jsonable_metrics(binary_metrics(labels, scores)),
                "selective_from_validation": selective,
                "by_dataset": _dataset_metrics(rows, labels, scores),
            }
        )
        torch.cuda.empty_cache()
    atomic_jsonl(worker_path, results)
    atomic_json(
        complete_path,
        {
            "passed": True,
            "rank": rank,
            "world_size": world_size,
            "contract_sha256": contract["contract_sha256"],
            "validation_region_sha256": file_sha256(region_path),
            "layers": assigned,
            "records": len(results),
            "result_sha256": file_sha256(worker_path),
        },
    )
    print(json.dumps({"rank": rank, "layers": assigned, "passed": True}))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _plot_results(
    contract: Mapping[str, Any],
    output_root: Path,
    validation_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    region: Mapping[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_root / "figures"
    layers = np.arange(28)
    start = int(region["selected_start"])
    end = int(region["selected_end"])
    val_auroc = np.asarray(
        [float(row["validation"]["auroc"]) for row in validation_records]
    )
    test_auroc = np.asarray([float(row["overall"]["auroc"]) for row in test_records])
    val_auprc = np.asarray(
        [float(row["validation"]["auprc"]) for row in validation_records]
    )
    test_auprc = np.asarray([float(row["overall"]["auprc"]) for row in test_records])

    def base_figure(ylabel: str):
        figure, axis = plt.subplots(figsize=(8.2, 4.8))
        axis.axvspan(start - 0.5, end + 0.5, color="#e6ab02", alpha=0.13, label="validation-selected region")
        axis.set_xlabel("Decoder layer")
        axis.set_ylabel(ylabel)
        axis.set_xticks(np.arange(0, 28, 2))
        axis.grid(alpha=0.25)
        return figure, axis

    figure, axis = base_figure("Failure-prediction AUROC")
    axis.axhline(0.5, color="black", linestyle=":", linewidth=1, label="chance")
    axis.plot(layers, val_auroc, marker="o", markersize=3, label="validation")
    axis.plot(layers, test_auroc, marker="s", markersize=3, label="test")
    axis.set_ylim(0.45, min(1.0, max(val_auroc.max(), test_auroc.max()) + 0.04))
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figures / "layerwise_failure_auroc.png", dpi=180)
    plt.close(figure)

    figure, axis = base_figure("Failure-prediction AUPRC")
    prevalence = float(contract["population"]["wrong"]) / float(
        contract["population"]["records"]
    )
    axis.axhline(prevalence, color="black", linestyle=":", linewidth=1, label="population prevalence")
    axis.plot(layers, val_auprc, marker="o", markersize=3, label="validation")
    axis.plot(layers, test_auprc, marker="s", markersize=3, label="test")
    axis.set_ylim(max(0.0, prevalence - 0.08), min(1.0, max(val_auprc.max(), test_auprc.max()) + 0.04))
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figures / "layerwise_failure_auprc.png", dpi=180)
    plt.close(figure)

    figure, axis = base_figure("Wrong-sample recall")
    colors = {"0.99": "#1b9e77", "0.98": "#7570b3", "0.95": "#d95f02"}
    for target in contract["evaluation"]["preservation_targets"]:
        key = f"{float(target):.2f}"
        val_values = [
            float(row["validation_selective"][key]["wrong_recall"])
            for row in validation_records
        ]
        test_values = [
            float(row["selective_from_validation"][key]["wrong_recall"])
            for row in test_records
        ]
        label = str(int(round(float(target) * 100)))
        axis.plot(
            layers,
            val_values,
            color=colors[key],
            alpha=0.45,
            linestyle="--",
            label=f"validation @{label}% preserve",
        )
        axis.plot(
            layers,
            test_values,
            color=colors[key],
            marker="o",
            markersize=2.5,
            label=f"test @{label}% preserve",
        )
    axis.set_ylim(-0.02, 1.02)
    axis.legend(loc="best", ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "wrong_recall_at_fixed_preservation.png", dpi=180)
    plt.close(figure)

    figure, axis = base_figure("Test failure-prediction AUROC")
    for dataset, color in zip(DATASETS, ("#1b9e77", "#d95f02", "#7570b3")):
        values = [float(row["by_dataset"][dataset]["auroc"]) for row in test_records]
        axis.plot(layers, values, marker="o", markersize=2.5, label=dataset, color=color)
    axis.axhline(0.5, color="black", linestyle=":", linewidth=1)
    axis.set_ylim(0.4, 1.0)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figures / "dataset_layerwise_auroc.png", dpi=180)
    plt.close(figure)

    answer_rows = _read_csv(
        resolve_repo_path(contract["sources"]["answer_emergence_correct"])
    )
    answer_top1 = np.asarray([float(row["gt_top1_fraction"]) for row in answer_rows])
    if len(answer_top1) != 28:
        raise ValueError("corrected answer-emergence curve does not cover 28 layers")
    figure, axis = base_figure("Fraction / AUROC")
    axis.plot(layers, test_auroc, color="#1b9e77", marker="o", markersize=3, label="test failure AUROC")
    axis.plot(
        layers,
        answer_top1,
        color="#d95f02",
        marker="s",
        markersize=3,
        label="correct GT token raw top-1 fraction",
    )
    axis.axvline(25, color="gray", linestyle=":", linewidth=1, label="late answer region begins")
    axis.set_ylim(-0.02, 1.02)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figures / "failure_vs_answer_emergence.png", dpi=180)
    plt.close(figure)


def _render_analysis_summary(
    contract: Mapping[str, Any],
    region: Mapping[str, Any],
    validation_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    split_audit_value: Mapping[str, Any],
) -> str:
    start = int(region["selected_start"])
    end = int(region["selected_end"])
    best_test = max(test_records, key=lambda row: float(row["overall"]["auroc"]))
    best_pre_answer = max(
        (row for row in test_records if int(row["layer"]) < 25),
        key=lambda row: float(row["overall"]["auroc"]),
    )
    start_test = test_records[start]
    region_test_aurocs = [
        float(test_records[layer]["overall"]["auroc"]) for layer in range(start, end + 1)
    ]
    before_answer = start < 25
    selective_best = {}
    for target in contract["evaluation"]["preservation_targets"]:
        key = f"{float(target):.2f}"
        row = max(
            test_records,
            key=lambda item: float(item["selective_from_validation"][key]["wrong_recall"]),
        )
        selective_best[key] = row
    dataset_best = {
        dataset: max(
            test_records, key=lambda row: float(row["by_dataset"][dataset]["auroc"])
        )
        for dataset in DATASETS
    }
    lines = [
        "# Layer-Wise Dense-Failure Predictability Analysis",
        "",
        "## Contract and validity",
        "",
        f"- Frozen contract: `{contract['contract_sha256']}`",
        f"- Population: `{contract['population']['records']:,}` current native-dense LMMS records "
        f"(`{contract['population']['correct']:,}` correct / `{contract['population']['wrong']:,}` wrong).",
        f"- Split: train `{split_audit_value['splits']['train']['records']:,}`, validation "
        f"`{split_audit_value['splits']['val']['records']:,}`, test `{split_audit_value['splits']['test']['records']:,}`; "
        f"UID overlap `{split_audit_value['uid_overlap']}`, image-group overlap `{split_audit_value['image_group_overlap']}`.",
        "- Input at every layer: train-normalized concatenation of `text_final`, `text_mean`, and `visual_mean` "
        "(10,752 values). No dataset ID, answer, historical label, W→C, or route input was used.",
        "- Twenty-eight independent linear probes used identical settings. Selective thresholds and the candidate region "
        "were fixed from validation before test evaluation; test was evaluated once.",
        "",
        "## Test metrics",
        "",
        "| Layer | AUROC | AUPRC | Balanced Acc |",
        "|---:|---:|---:|---:|",
    ]
    for row in test_records:
        metrics = row["overall"]
        lines.append(
            f"| {int(row['layer'])} | {float(metrics['auroc']):.4f} | "
            f"{float(metrics['auprc']):.4f} | {float(metrics['balanced_accuracy']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Answers to the plan questions",
            "",
            "### Q1. At what depth does failure first become meaningfully predictable?",
            "",
            f"The validation-frozen descriptive region is layers **{start}-{end}**. At its first layer, "
            f"test AUROC is `{float(start_test['overall']['auroc']):.4f}` and AUPRC is "
            f"`{float(start_test['overall']['auprc']):.4f}`; mean test AUROC across the region is "
            f"`{float(np.mean(region_test_aurocs)):.4f}`. Rationale frozen before test: {region['rationale']}",
            "",
            "### Q2. Is useful failure predictability present before answer emergence at layers 25-27?",
            "",
            (
                f"Yes. The validation-selected region begins at layer {start}, before layer 25. The strongest "
                f"pre-25 test layer is {int(best_pre_answer['layer'])} with AUROC "
                f"`{float(best_pre_answer['overall']['auroc']):.4f}`."
                if before_answer
                else f"No defensible pre-25 region was selected; the validation-selected region begins at layer {start}."
            ),
            "",
            "### Q3. What wrong detection is possible at fixed dense-correct preservation?",
            "",
            "Thresholds were chosen on validation and transferred unchanged to test. Full per-layer results are in "
            "`layerwise_selective_metrics.csv` and `test_metrics.csv`.",
            "",
            "| Validation preservation target | Best test wrong recall | Layer | Test correct preservation | Failure precision |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for target in contract["evaluation"]["preservation_targets"]:
        key = f"{float(target):.2f}"
        row = selective_best[key]
        point = row["selective_from_validation"][key]
        lines.append(
            f"| {float(target):.0%} | {float(point['wrong_recall']):.4f} | {int(row['layer'])} | "
            f"{float(point['correct_preservation']):.4f} | {float(point['failure_precision']):.4f} |"
        )
    lines.extend(
        [
            "",
            "### Q4. Is failure-awareness depth consistent across datasets?",
            "",
            "Dataset-wise held-out peaks are shown below; the full curves are in `dataset_layerwise_metrics.csv`.",
            "",
            "| Dataset | Best test AUROC | Layer | AUROC at selected start |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        row = dataset_best[dataset]
        lines.append(
            f"| {dataset} | {float(row['by_dataset'][dataset]['auroc']):.4f} | {int(row['layer'])} | "
            f"{float(start_test['by_dataset'][dataset]['auroc']):.4f} |"
        )
    lines.extend(
        [
            "",
            "### Q5. What depth range is defensible for later Stage-1 supervision and gating?",
            "",
            f"Layers **{start}-{end}** are the validation-frozen candidate range for a later training comparison. "
            "This is descriptive evidence from regularized linear accessibility, not authorization to train the shared predictor "
            "and not proof that every layer in the range is equally causal or useful for intervention.",
            "",
            "### Q6. What should the next training experiment compare?",
            "",
            "A separately authorized Stage-1 training experiment should compare all-layer supervision, informative-depth-only "
            f"supervision on the frozen candidate range {start}-{end}, and random-k sampling from that same range. Keep the "
            "split, labels, compact feature definition, optimizer budget, and evaluation protocol fixed so the comparison isolates "
            "the supervision-depth policy.",
            "",
            "## Important interpretation limits",
            "",
            "- The corrected answer-emergence curve is an unlearned final-head token readout at the true assistant answer position; "
            "the failure curve is a learned held-out decoder from compact states. Their depth ordering is informative, but the y-axes "
            "are not the same quantity.",
            "- The concatenated probe does not attribute signal to text-final, text-mean, or visual-mean components individually.",
            "- The test maximum is descriptive. Probe settings, thresholds, and the candidate region were not retuned on test.",
            "",
            f"Best overall held-out AUROC is `{float(best_test['overall']['auroc']):.4f}` at layer "
            f"`{int(best_test['layer'])}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def finalize(config_path: Path) -> None:
    contract, output_root = load_frozen_contract(config_path)
    region_path = output_root / "validation_region_decision.json"
    region = read_json(region_path)
    if region.get("contract_sha256") != contract["contract_sha256"]:
        raise ValueError("validation region contract differs")
    validation = read_json(output_root / "validation_results.json")
    validation_records = validate_complete_layer_records(validation["layers"])
    test_records = []
    for rank in range(4):
        complete_path = output_root / "workers" / f"test_rank{rank:02d}.complete.json"
        worker_path = output_root / "workers" / f"test_rank{rank:02d}.jsonl"
        complete = read_json(complete_path)
        if (
            complete.get("passed") is not True
            or complete.get("contract_sha256") != contract["contract_sha256"]
            or complete.get("validation_region_sha256") != file_sha256(region_path)
            or complete.get("result_sha256") != file_sha256(worker_path)
        ):
            raise RuntimeError(f"test worker {rank} is incomplete or incompatible")
        test_records.extend(read_jsonl(worker_path))
    test_records = validate_complete_layer_records(test_records)
    expected = checkpoint_expected(contract)
    for record in test_records:
        layer = int(record["layer"])
        checkpoint_path = output_root / "checkpoints" / f"layer_{layer:02d}.pt"
        if record["checkpoint_sha256"] != file_sha256(checkpoint_path):
            raise RuntimeError(f"test checkpoint hash differs for layer {layer}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        validate_checkpoint_provenance(checkpoint, layer=layer, expected=expected)

    test_rows = []
    for record in test_records:
        metrics = record["overall"]
        row: dict[str, Any] = {
            "layer": int(record["layer"]),
            "test_auroc": float(metrics["auroc"]),
            "test_auprc": float(metrics["auprc"]),
            "test_balanced_accuracy": float(metrics["balanced_accuracy"]),
        }
        for target in contract["evaluation"]["preservation_targets"]:
            key = f"{float(target):.2f}"
            suffix = str(int(round(float(target) * 100)))
            point = record["selective_from_validation"][key]
            row.update(
                {
                    f"validation_threshold_at_{suffix}_preserve": float(point["threshold"]),
                    f"test_correct_preservation_at_{suffix}": float(
                        point["correct_preservation"]
                    ),
                    f"test_wrong_recall_at_{suffix}_preserve": float(point["wrong_recall"]),
                    f"test_failure_precision_at_{suffix}_preserve": float(
                        point["failure_precision"]
                    ),
                    f"test_false_deviation_rate_at_{suffix}_preserve": float(
                        point["false_deviation_rate"]
                    ),
                }
            )
        test_rows.append(row)
    atomic_csv(output_root / "test_metrics.csv", test_rows)

    dataset_rows: list[dict[str, Any]] = [dict(row) for row in _read_csv(output_root / "dataset_layerwise_metrics.csv")]
    for record in test_records:
        for dataset in DATASETS:
            values = record["by_dataset"][dataset]
            dataset_rows.append(
                {
                    "split": "test",
                    "layer": int(record["layer"]),
                    "dataset": dataset,
                    "records": int(values["records"]),
                    "correct": int(values["correct"]),
                    "wrong": int(values["wrong"]),
                    "auroc": float(values["auroc"]),
                    "auprc": float(values["auprc"]),
                    "balanced_accuracy": float(values["balanced_accuracy"]),
                }
            )
    atomic_csv(output_root / "dataset_layerwise_metrics.csv", dataset_rows)
    split_audit_value = read_json(output_root / "split_audit.json")
    _plot_results(contract, output_root, validation_records, test_records, region)
    summary = _render_analysis_summary(
        contract, region, validation_records, test_records, split_audit_value
    )
    _atomic_bytes(output_root / "analysis_summary.md", summary.encode("utf-8"))
    results = {
        "schema_version": "layerwise_dense_failure_probe_analysis_v1",
        "contract_sha256": contract["contract_sha256"],
        "validation_region": region,
        "split_audit": split_audit_value,
        "validation_layers": validation_records,
        "test_layers": test_records,
        "best_test_layer": int(
            max(test_records, key=lambda row: float(row["overall"]["auroc"]))["layer"]
        ),
    }
    atomic_json(output_root / "analysis_results.json", results)
    required = [
        "split_manifest.jsonl",
        "split_audit.md",
        "probe_config.json",
        "layerwise_probe_metrics.csv",
        "layerwise_selective_metrics.csv",
        "dataset_layerwise_metrics.csv",
        "test_metrics.csv",
        "analysis_summary.md",
        "figures/layerwise_failure_auroc.png",
        "figures/layerwise_failure_auprc.png",
        "figures/wrong_recall_at_fixed_preservation.png",
        "figures/dataset_layerwise_auroc.png",
        "figures/failure_vs_answer_emergence.png",
    ]
    missing = [path for path in required if not (output_root / path).is_file()]
    if missing:
        raise RuntimeError(f"required final artifacts are missing: {missing}")
    manifest = {
        "schema_version": "layerwise_dense_failure_probe_artifact_manifest_v1",
        "contract_sha256": contract["contract_sha256"],
        "required_files": {
            path: file_sha256(output_root / path) for path in required
        },
        "checkpoints": {
            f"layer_{layer:02d}.pt": file_sha256(
                output_root / "checkpoints" / f"layer_{layer:02d}.pt"
            )
            for layer in range(28)
        },
        "test_evaluated_once": True,
        "complete_layers": 28,
    }
    atomic_json(output_root / "artifact_manifest.json", manifest)
    print(
        json.dumps(
            {
                "passed": True,
                "contract_sha256": contract["contract_sha256"],
                "best_test_layer": results["best_test_layer"],
                "candidate_region": region["selected_layers"],
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    worker = subparsers.add_parser("train-worker")
    worker.add_argument("--rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    worker.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    subparsers.add_parser("aggregate-validation")
    region = subparsers.add_parser("freeze-region")
    region.add_argument("--start", type=int, required=True)
    region.add_argument("--end", type=int, required=True)
    region.add_argument("--rationale", required=True)
    test = subparsers.add_parser("test-worker")
    test.add_argument("--rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    test.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    subparsers.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_relative_to(PROJECT_ROOT):
        raise ValueError("config must lie inside the project root")
    if args.command == "prepare":
        prepare(config_path)
    elif args.command == "train-worker":
        train_worker(config_path, rank=args.rank, world_size=args.world_size)
    elif args.command == "aggregate-validation":
        aggregate_validation(config_path)
    elif args.command == "freeze-region":
        freeze_region(
            config_path, start=args.start, end=args.end, rationale=args.rationale
        )
    elif args.command == "test-worker":
        test_worker(config_path, rank=args.rank, world_size=args.world_size)
    elif args.command == "finalize":
        finalize(config_path)
    else:
        raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
