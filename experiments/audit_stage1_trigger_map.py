#!/usr/bin/env python3
"""Reconstruct and audit the frozen Shared Random-4 Stage-1 trigger map."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dense_failure_stage1.layerwise_probe import compose_layer_features  # noqa: E402
from dense_failure_stage1.shared_global_gate import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    SharedFailurePredictor,
)
from dense_failure_stage1.trigger_map import (  # noqa: E402
    DATASETS,
    SPLITS,
    build_trigger_map,
    cumulative_trigger_rows,
    dataset_breakdown_rows,
    select_reproducibility_rows,
    split_summary,
    trigger_by_depth_bin_rows,
    trigger_by_layer_rows,
    trigger_score_stats_rows,
)
from experiments.analyze_layerwise_dense_failure_predictability import (  # noqa: E402
    load_frozen_contract as load_phase48_contract,
    load_verified_shard,
)
from experiments.analyze_shared_stage1_global_risk_gate import (  # noqa: E402
    load_contract as load_shared_contract,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/stage1_trigger_map_audit_v1.json"
PHASE48_CONFIG = PROJECT_ROOT / "configs/layerwise_dense_failure_probe_v1.json"
SHARED_CONFIG = PROJECT_ROOT / "configs/shared_stage1_global_risk_gate_v1.json"
BOUND_CODE_PATHS = (
    "configs/stage1_trigger_map_audit_v1.json",
    "dense_failure_stage1/trigger_map.py",
    "experiments/audit_stage1_trigger_map.py",
    "dense_failure_stage1/shared_global_gate.py",
    "dense_failure_stage1/layerwise_probe.py",
    "experiments/analyze_shared_stage1_global_risk_gate.py",
    "experiments/analyze_layerwise_dense_failure_predictability.py",
)
EXPECTED_CHECKPOINT_SHA256 = "1aeeaa278a9ac2aa99343b4ea26fad43a8b097c05b497c27554289eaef311211"
EXPECTED_NORMALIZATION_SHA256 = "ce4b63ab7f503d879db20f94aae01116804095eb87fd03b16a818d5904f1aed3"
EXPECTED_SHARED_CONTRACT = "264a9407e5acb35e19bd4b53436ae8bf1175ad989f8c974cf7d8e859565e095b"
EXPECTED_PHASE48_CONTRACT = "3cf49a46d0a47a0ae1955f8a518f5a74bef65b7de8736980a6e3d26e170234cd"
EXPECTED_PHASE52_CONTRACT = "f065d3728ebce2c95667a566e09ab49d5a4ead901b0334e5e7566df0f2288135"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def frozen_contract_payload(
    config: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind both the exact static snapshot and convenient runtime projections."""

    static = json.loads(json.dumps(config))
    contract = json.loads(json.dumps(config))
    contract["schema_version"] = "stage1_trigger_map_audit_frozen_contract_v1"
    contract["static_config"] = static
    contract["provenance"] = json.loads(json.dumps(provenance))
    contract["contract_sha256"] = canonical_hash(contract)
    return contract


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(f"path escapes project root: {value}")
    return resolved


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
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite incompatible artifact: {path}")
        return
    _atomic_bytes(path, payload)


def command_output(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result.stdout.strip()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def load_static(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "stage1_trigger_map_audit_config_v1":
        raise ValueError("unsupported trigger-map audit config")
    if int(config["world_size"]) != 4 or tuple(config["population"]["splits"]) != SPLITS:
        raise ValueError("trigger-map world size or split order differs")
    gate = config["gate"]
    if (
        gate.get("name") != "shared_random4"
        or gate.get("variant") != "state_layer_random4"
        or gate.get("control_type") != "global_raw_threshold"
        or gate.get("comparison") != "strict_greater_than"
        or gate.get("layers") != list(range(28))
    ):
        raise ValueError("trigger-map gate differs from Shared Random-4")
    return config


def _validate_split(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = read_jsonl(resolve_path(config["sources"]["split_manifest"]))
    if len(rows) != int(config["population"]["records"]):
        raise ValueError("split manifest record count differs")
    if len({str(row["uid"]) for row in rows}) != len(rows):
        raise ValueError("split manifest contains duplicate UIDs")
    counts = Counter(str(row["split"]) for row in rows)
    if counts != Counter({key: int(value) for key, value in config["population"]["splits"].items()}):
        raise ValueError(f"split counts differ: {counts}")
    for row in rows:
        if str(row["dataset"]) not in DATASETS:
            raise ValueError(f"unexpected dataset for {row['uid']}")
        if bool(row["current_dense_correct"]) == bool(row["current_dense_wrong"]):
            raise ValueError(f"dense labels are not complementary for {row['uid']}")
        if not resolve_path(row["feature_shard"]).is_file():
            raise ValueError(f"feature shard is missing for {row['uid']}")
    groups = {
        split: {str(row["image_group_id"]) for row in rows if row["split"] == split}
        for split in SPLITS
    }
    if any(groups[left] & groups[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("image group crosses split boundaries")
    return rows


def _load_saved_scores(
    config: Mapping[str, Any], split_rows: Sequence[Mapping[str, Any]], split: str
) -> list[dict[str, Any]]:
    if split == "val":
        rows = read_jsonl(resolve_path(config["sources"]["validation_scores"]))
    elif split == "test":
        rows = [
            row
            for row in read_jsonl(resolve_path(config["sources"]["test_scores"]))
            if row.get("variant") == "state_layer_random4"
        ]
    else:
        raise ValueError("saved scores exist only for val/test")
    expected = sorted(
        [row for row in split_rows if str(row["split"]) == split],
        key=lambda row: str(row["uid"]),
    )
    rows.sort(key=lambda row: str(row["uid"]))
    if len(rows) != 800 or [row["uid"] for row in rows] != [row["uid"] for row in expected]:
        raise ValueError(f"saved {split} score UID coverage differs")
    for score, source in zip(rows, expected):
        if (
            str(score["dataset"]) != str(source["dataset"])
            or str(score["image_group_id"]) != str(source["image_group_id"])
            or bool(score["current_dense_wrong"]) != bool(source["current_dense_wrong"])
            or any(not np.isfinite(float(score[f"p_{layer}"])) for layer in range(28))
        ):
            raise ValueError(f"saved {split} score metadata differs for {source['uid']}")
    return rows


def _source_rank(row: Mapping[str, Any]) -> int:
    match = re.search(r"/rank(\d{3})_", str(row["feature_shard"]))
    if match is None or int(match.group(1)) not in range(4):
        raise ValueError(f"cannot resolve source rank for {row['uid']}")
    return int(match.group(1))


def _protocol_markdown(contract: Mapping[str, Any]) -> str:
    gate = contract["static_config"]["gate"]
    return "\n".join(
        [
            "# Frozen Stage-1 Trigger Map Audit Protocol",
            "",
            f"- Audit contract: `{contract['contract_sha256']}`",
            f"- Git commit: `{contract['provenance']['git_commit']}` on `{contract['provenance']['git_branch']}`; the full dirty-worktree snapshot is recorded in `frozen_protocol.json`.",
            "- Population: the frozen current-runtime dense FULL split (6,399 train / 800 validation / 800 test), with current LMMS-Eval correctness labels and zero image-group overlap.",
            "- Gate: Shared Random-4 (`state_layer_random4`), unchanged checkpoint and global raw threshold.",
            f"- Checkpoint SHA-256: `{contract['provenance']['checkpoint_sha256']}`.",
            f"- Normalization SHA-256: `{contract['provenance']['normalization_sha256']}`.",
            f"- Trigger rule: first layer in 0-27 with `score > {float(gate['threshold']):.17g}` (strict crossing).",
            "- Validation/test: reuse frozen saved 28-layer trajectories and verify them against the Phase-53 trigger manifest and Phase-52 aggregate metrics.",
            "- Train: recompute only the lightweight gate trajectory from hash-verified stored dense feature shards; four direct GPU workers own source ranks 000-003.",
            f"- Reproducibility check: {contract['static_config']['reproducibility_records']} deterministic val/test records, stratified by split, dataset, and dense class.",
            "- No Qwen forward pass, four-action search, Stage-2 labeling/training, threshold change, persistence, or EMA is permitted.",
            "- Future usage: only train triggered-W may enter corrective label search; validation is model-selection/treatment-only and test remains held out.",
            "",
        ]
    )


def prepare(config_path: Path) -> None:
    config = load_static(config_path)
    output_root = resolve_path(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")

    phase48, _ = load_phase48_contract(PHASE48_CONFIG)
    shared, _, _, _ = load_shared_contract(SHARED_CONFIG)
    if phase48["contract_sha256"] != EXPECTED_PHASE48_CONTRACT:
        raise RuntimeError("Phase-48 contract identity differs")
    if shared["contract_sha256"] != EXPECTED_SHARED_CONTRACT:
        raise RuntimeError("shared-gate contract identity differs")
    sources = config["sources"]
    checkpoint_path = resolve_path(sources["shared_checkpoint"])
    normalization_path = resolve_path(sources["shared_normalization"])
    if file_sha256(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Shared Random-4 checkpoint hash differs")
    if file_sha256(normalization_path) != EXPECTED_NORMALIZATION_SHA256:
        raise RuntimeError("shared normalization hash differs")
    complete = read_json(resolve_path(sources["shared_training_complete"]))
    if not complete.get("passed") or complete.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Shared Random-4 training completion is invalid")
    selection = read_json(resolve_path(sources["phase52_selection"]))
    control = selection["candidate_controls"]["shared_random4"]
    if (
        selection.get("contract_sha256") != EXPECTED_PHASE52_CONTRACT
        or control.get("control_type") != "global_raw_threshold"
        or control.get("layers") != list(range(28))
        or float(control.get("threshold")) != float(config["gate"]["threshold"])
    ):
        raise RuntimeError("Phase-52 Shared Random-4 control differs")

    split_rows = _validate_split(config)
    _load_saved_scores(config, split_rows, "val")
    _load_saved_scores(config, split_rows, "test")
    phase53 = read_jsonl(resolve_path(sources["phase53_trigger_manifest"]))
    if len(phase53) != 1600 or len({(row["split"], row["uid"]) for row in phase53}) != 1600:
        raise RuntimeError("Phase-53 Shared Random-4 trigger manifest is incomplete")
    repro = select_reproducibility_rows(
        split_rows,
        seed=int(config["seed"]),
        per_cell=int(config["reproducibility"]["records_per_cell"]),
    )
    if len(repro) != int(config["reproducibility_records"]):
        raise RuntimeError("reproducibility sample count differs")
    repro_rows = [
        {
            "schema_version": "stage1_trigger_map_reproducibility_row_v1",
            "uid": row["uid"],
            "dataset": row["dataset"],
            "split": row["split"],
            "current_dense_wrong": bool(row["current_dense_wrong"]),
            "feature_shard": row["feature_shard"],
            "feature_row_index": int(row["feature_row_index"]),
            "worker_rank": _source_rank(row),
        }
        for row in repro
    ]
    atomic_jsonl(output_root / "reproducibility_manifest.jsonl", repro_rows)

    source_hashes = {name: file_sha256(resolve_path(path)) for name, path in sources.items()}
    bound_hashes = {path: file_sha256(resolve_path(path)) for path in BOUND_CODE_PATHS}
    status = command_output(("git", "status", "--porcelain=v1", "--untracked-files=all"))
    provenance = {
        "git_commit": command_output(("git", "rev-parse", "HEAD")),
        "git_branch": command_output(("git", "branch", "--show-current")),
        "git_status_porcelain_at_freeze": status.splitlines() if status else [],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "matplotlib": package_version("matplotlib"),
        "cuda_runtime": torch.version.cuda,
        "gpu_inventory": command_output(
            (
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            )
        ).splitlines(),
        "static_config_sha256": file_sha256(config_path),
        "source_sha256": source_hashes,
        "bound_code_sha256": bound_hashes,
        "reproducibility_manifest_sha256": file_sha256(output_root / "reproducibility_manifest.jsonl"),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
        "phase48_contract_sha256": phase48["contract_sha256"],
        "shared_contract_sha256": shared["contract_sha256"],
        "phase52_contract_sha256": selection["contract_sha256"],
    }
    contract = frozen_contract_payload(config, provenance)
    write_once(
        output_root / "frozen_protocol.json",
        (json.dumps(contract, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    write_once(output_root / "protocol.md", _protocol_markdown(contract).encode("utf-8"))
    atomic_json(
        output_root / "preparation_audit.json",
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "population_records": len(split_rows),
            "split_counts": dict(Counter(row["split"] for row in split_rows)),
            "reproducibility_records": len(repro_rows),
            "validation_saved_scores": 800,
            "test_saved_scores": 800,
            "train_saved_scores": 0,
            "qwen_forward_required": False,
        },
    )
    print(json.dumps({"passed": True, "contract_sha256": contract["contract_sha256"]}))


def load_contract(config_path: Path) -> tuple[dict[str, Any], Path]:
    config = load_static(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if (
        contract.get("schema_version") != "stage1_trigger_map_audit_frozen_contract_v1"
        or contract.get("contract_sha256") != canonical_hash(contract)
        or contract["provenance"]["static_config_sha256"] != file_sha256(config_path)
    ):
        raise RuntimeError("trigger-map frozen contract is invalid")
    for name, expected in contract["provenance"]["source_sha256"].items():
        if file_sha256(resolve_path(contract["sources"][name])) != expected:
            raise RuntimeError(f"trigger-map source hash differs: {name}")
    for path, expected in contract["provenance"]["bound_code_sha256"].items():
        if file_sha256(resolve_path(path)) != expected:
            raise RuntimeError(f"trigger-map bound code hash differs: {path}")
    if file_sha256(output_root / "reproducibility_manifest.jsonl") != contract["provenance"]["reproducibility_manifest_sha256"]:
        raise RuntimeError("reproducibility manifest hash differs")
    return contract, output_root


def _require_gpu(rank: int, world_size: int) -> torch.device:
    if world_size != 4 or rank not in range(4):
        raise ValueError("trigger-map workers require ranks 0-3")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise RuntimeError("four CUDA GPUs are not visible")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return device


def _build_frozen_model(contract: Mapping[str, Any], device: torch.device):
    shared = read_json(resolve_path(contract["sources"]["shared_contract"]))
    architecture = shared["architecture"]
    model = SharedFailurePredictor(
        variant="state_layer_random4",
        input_size=int(shared["input"]["input_size"]),
        projection_size=int(architecture["projection_size"]),
        layer_embedding_size=int(architecture["layer_embedding_size"]),
        hidden_size=int(architecture["hidden_size"]),
    ).to(device=device, dtype=torch.float32)
    checkpoint = torch.load(
        resolve_path(contract["sources"]["shared_checkpoint"]),
        map_location="cpu",
        weights_only=True,
    )
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("variant") != "state_layer_random4"
        or checkpoint.get("contract_sha256") != EXPECTED_SHARED_CONTRACT
        or checkpoint.get("phase48_contract_sha256") != EXPECTED_PHASE48_CONTRACT
        or checkpoint.get("global_normalization_sha256") != EXPECTED_NORMALIZATION_SHA256
    ):
        raise RuntimeError("Shared Random-4 checkpoint provenance differs")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalization = torch.load(
        resolve_path(contract["sources"]["shared_normalization"]),
        map_location="cpu",
        weights_only=True,
    )
    if (
        normalization.get("schema_version") != "shared_stage1_global_normalization_v1"
        or tuple(normalization["mean"].shape) != (10752,)
        or tuple(normalization["std"].shape) != (10752,)
    ):
        raise RuntimeError("shared normalization schema differs")
    return model, normalization["mean"].float().to(device), normalization["std"].float().to(device)


def _score_feature_rows(
    model: SharedFailurePredictor,
    mean: torch.Tensor,
    std: torch.Tensor,
    shard: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    indices = [int(row["feature_row_index"]) for row in rows]
    for row, index in zip(rows, indices):
        if not 0 <= index < len(shard["uids"]) or str(shard["uids"][index]) != str(row["uid"]):
            raise RuntimeError(f"feature UID/index mismatch for {row['uid']}")
    features = torch.stack(
        [compose_layer_features(shard, row_indices=indices, layer=layer) for layer in range(28)],
        dim=1,
    )
    logits_rows = []
    all_layers = torch.arange(28, dtype=torch.long, device=device)[None, :]
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            values = features[start : start + batch_size]
            records = len(values)
            states = values.reshape(records * 28, -1).to(device=device, dtype=torch.float32)
            states = (states - mean) / std
            layers = all_layers.expand(records, -1).reshape(-1)
            logits_rows.append(model(states, layers).reshape(records, 28).cpu())
    logits = torch.cat(logits_rows).numpy().astype(np.float64)
    scores = np.empty_like(logits)
    positive = logits >= 0
    scores[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    scores[~positive] = exponential / (1.0 + exponential)
    output = []
    for row, values in zip(rows, scores):
        record = {
            "schema_version": "shared_stage1_score_trajectory_v1",
            "variant": "state_layer_random4",
            "uid": str(row["uid"]),
            "dataset": str(row["dataset"]),
            "image_group_id": str(row["image_group_id"]),
            "current_dense_wrong": bool(row["current_dense_wrong"]),
        }
        record.update({f"p_{layer}": float(values[layer]) for layer in range(28)})
        output.append(record)
    return output


def score_worker(config_path: Path, *, rank: int, world_size: int) -> None:
    contract, output_root = load_contract(config_path)
    device = _require_gpu(rank, world_size)
    result_path = output_root / f"work/train_scores_rank{rank:02d}.jsonl"
    repro_path = output_root / f"work/repro_scores_rank{rank:02d}.jsonl"
    complete_path = output_root / f"work/rank{rank:02d}.complete.json"
    if complete_path.exists():
        complete = read_json(complete_path)
        if (
            complete.get("passed") is not True
            or complete.get("contract_sha256") != contract["contract_sha256"]
            or complete.get("train_scores_sha256") != file_sha256(result_path)
            or complete.get("repro_scores_sha256") != file_sha256(repro_path)
        ):
            raise RuntimeError(f"worker {rank} resume artifacts are incompatible")
        print(json.dumps({"passed": True, "rank": rank, "resumed": True}))
        return

    split_rows = _validate_split(contract["static_config"])
    train_rows = sorted(
        [row for row in split_rows if row["split"] == "train" and _source_rank(row) == rank],
        key=lambda row: (str(row["feature_shard"]), str(row["uid"])),
    )
    by_uid = {str(row["uid"]): row for row in split_rows}
    repro_manifest = read_jsonl(output_root / "reproducibility_manifest.jsonl")
    repro_rows = [by_uid[str(row["uid"])] for row in repro_manifest if int(row["worker_rank"]) == rank]
    selected = [("train", row) for row in train_rows] + [("repro", row) for row in repro_rows]
    by_shard: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for purpose, row in selected:
        by_shard[str(row["feature_shard"])].append((purpose, row))
    shard_manifest = read_json(resolve_path(contract["sources"]["feature_shard_manifest"]))
    shard_hashes = {str(row["path"]): str(row["sha256"]) for row in shard_manifest["shards"]}
    if set(by_shard) - set(shard_hashes):
        raise RuntimeError(f"worker {rank} references an unfrozen shard")

    model, mean, std = _build_frozen_model(contract, device)
    train_scores = []
    repro_scores = []
    for relative in sorted(by_shard):
        shard = load_verified_shard(resolve_path(relative), shard_hashes[relative])
        assignments = by_shard[relative]
        rows = [row for _, row in assignments]
        scores = _score_feature_rows(model, mean, std, shard, rows, device=device)
        for (purpose, _), score in zip(assignments, scores):
            (train_scores if purpose == "train" else repro_scores).append(score)
    train_scores.sort(key=lambda row: str(row["uid"]))
    repro_scores.sort(key=lambda row: str(row["uid"]))
    expected_train = sum(row["split"] == "train" and _source_rank(row) == rank for row in split_rows)
    if len(train_scores) != expected_train or len({row["uid"] for row in train_scores}) != expected_train:
        raise RuntimeError(f"worker {rank} train score coverage differs")
    expected_repro = sum(int(row["worker_rank"]) == rank for row in repro_manifest)
    if len(repro_scores) != expected_repro or len({row["uid"] for row in repro_scores}) != expected_repro:
        raise RuntimeError(f"worker {rank} reproducibility score coverage differs")
    atomic_jsonl(result_path, train_scores)
    atomic_jsonl(repro_path, repro_scores)
    atomic_json(
        complete_path,
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "rank": rank,
            "world_size": world_size,
            "train_records": len(train_scores),
            "reproducibility_records": len(repro_scores),
            "train_scores_sha256": file_sha256(result_path),
            "repro_scores_sha256": file_sha256(repro_path),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "normalization_sha256": EXPECTED_NORMALIZATION_SHA256,
        },
    )
    print(json.dumps({"passed": True, "rank": rank, "train": len(train_scores), "repro": len(repro_scores)}))


def _worker_outputs(
    contract: Mapping[str, Any], output_root: Path, split_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_scores = []
    repro_scores = []
    for rank in range(4):
        result_path = output_root / f"work/train_scores_rank{rank:02d}.jsonl"
        repro_path = output_root / f"work/repro_scores_rank{rank:02d}.jsonl"
        complete = read_json(output_root / f"work/rank{rank:02d}.complete.json")
        if (
            complete.get("passed") is not True
            or complete.get("contract_sha256") != contract["contract_sha256"]
            or int(complete.get("rank", -1)) != rank
            or int(complete.get("world_size", -1)) != 4
            or complete.get("train_scores_sha256") != file_sha256(result_path)
            or complete.get("repro_scores_sha256") != file_sha256(repro_path)
            or complete.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
            or complete.get("normalization_sha256") != EXPECTED_NORMALIZATION_SHA256
        ):
            raise RuntimeError(f"worker {rank} completion is incompatible")
        worker_train = read_jsonl(result_path)
        worker_repro = read_jsonl(repro_path)
        if len(worker_train) != int(complete["train_records"]) or len(worker_repro) != int(complete["reproducibility_records"]):
            raise RuntimeError(f"worker {rank} completion counts differ")
        train_scores.extend(worker_train)
        repro_scores.extend(worker_repro)
    train_scores.sort(key=lambda row: str(row["uid"]))
    expected_train = sorted(
        [row for row in split_rows if row["split"] == "train"], key=lambda row: str(row["uid"])
    )
    if (
        len(train_scores) != 6399
        or len({row["uid"] for row in train_scores}) != 6399
        or [row["uid"] for row in train_scores] != [row["uid"] for row in expected_train]
    ):
        raise RuntimeError("four-worker train score aggregation is incomplete")
    manifest = read_jsonl(output_root / "reproducibility_manifest.jsonl")
    repro_scores.sort(key=lambda row: str(row["uid"]))
    if (
        len(repro_scores) != len(manifest)
        or len({row["uid"] for row in repro_scores}) != len(manifest)
        or {row["uid"] for row in repro_scores} != {row["uid"] for row in manifest}
    ):
        raise RuntimeError("four-worker reproducibility aggregation is incomplete")
    return train_scores, repro_scores


def _reproducibility_audit(
    contract: Mapping[str, Any],
    output_root: Path,
    split_rows: Sequence[Mapping[str, Any]],
    repro_scores: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    saved = {}
    for split in ("val", "test"):
        saved.update(
            {
                str(row["uid"]): row
                for row in _load_saved_scores(contract["static_config"], split_rows, split)
            }
        )
    tolerance = float(contract["reproducibility"]["absolute_probability_tolerance"])
    comparison_rows = []
    exact_values = 0
    max_error = 0.0
    for row in sorted(repro_scores, key=lambda value: str(value["uid"])):
        source = saved[str(row["uid"])]
        errors = [abs(float(row[f"p_{layer}"]) - float(source[f"p_{layer}"])) for layer in range(28)]
        exact = sum(float(row[f"p_{layer}"]) == float(source[f"p_{layer}"]) for layer in range(28))
        exact_values += exact
        row_max = max(errors)
        max_error = max(max_error, row_max)
        comparison_rows.append(
            {
                "uid": row["uid"],
                "dataset": row["dataset"],
                "split": next(item["split"] for item in split_rows if item["uid"] == row["uid"]),
                "current_dense_wrong": bool(row["current_dense_wrong"]),
                "exact_layer_scores": exact,
                "max_absolute_error": row_max,
                "within_tolerance": row_max <= tolerance,
            }
        )
    passed = len(comparison_rows) == 24 and all(row["within_tolerance"] for row in comparison_rows)
    audit = {
        "passed": passed,
        "contract_sha256": contract["contract_sha256"],
        "records": len(comparison_rows),
        "layer_scores": len(comparison_rows) * 28,
        "exact_layer_scores": exact_values,
        "absolute_probability_tolerance": tolerance,
        "maximum_absolute_error": max_error,
        "trigger_decisions_reproducible": None,
    }
    threshold = float(contract["gate"]["threshold"])
    saved_map = {
        row["uid"]: row
        for split in ("val", "test")
        for row in build_trigger_map(
            [item for item in split_rows if item["split"] == split and item["uid"] in {value["uid"] for value in comparison_rows}],
            [saved[item["uid"]] for item in split_rows if item["split"] == split and item["uid"] in {value["uid"] for value in comparison_rows}],
            threshold=threshold,
            score_source="saved_phase51_scores",
        )
    }
    recomputed_by_uid = {row["uid"]: row for row in repro_scores}
    recomputed_map = {}
    for split in ("val", "test"):
        selected = [item for item in split_rows if item["split"] == split and item["uid"] in recomputed_by_uid]
        for row in build_trigger_map(
            selected,
            [recomputed_by_uid[item["uid"]] for item in selected],
            threshold=threshold,
            score_source="recomputed_frozen_features",
        ):
            recomputed_map[row["uid"]] = row
    trigger_match = all(
        saved_map[uid]["first_trigger_layer"] == recomputed_map[uid]["first_trigger_layer"]
        for uid in saved_map
    )
    audit["trigger_decisions_reproducible"] = trigger_match
    audit["passed"] = bool(audit["passed"] and trigger_match)
    atomic_csv(output_root / "reproducibility_rows.csv", comparison_rows)
    atomic_json(output_root / "reproducibility_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("stored-feature score reproducibility check failed")
    return audit


def _phase53_consistency(
    contract: Mapping[str, Any], trigger_maps: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    rows = read_jsonl(resolve_path(contract["sources"]["phase53_trigger_manifest"]))
    by_key = {(str(row["split"]), str(row["uid"])): row for row in rows}
    checked = 0
    for split in ("val", "test"):
        for row in trigger_maps[split]:
            old = by_key[(split, str(row["uid"]))]
            trajectory = [float(row[f"score_l{layer}"]) for layer in range(28)]
            if (
                bool(old["triggered"]) != bool(row["triggered"])
                or old["trigger_layer"] != row["first_trigger_layer"]
                or old["risk_at_trigger"] != row["score_at_trigger"]
                or [float(value) for value in old["risk_trajectory"]] != trajectory
            ):
                raise RuntimeError(f"Phase-53 trigger mismatch for {split}/{row['uid']}")
            checked += 1
    if checked != 1600:
        raise RuntimeError("Phase-53 trigger consistency did not cover 1,600 rows")
    return {"passed": True, "records": checked, "exact_trigger_rows": checked}


def compare_prior_summaries(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    validation_point: Mapping[str, Any],
    test_row: Mapping[str, Any],
) -> dict[str, Any]:
    expected_rows = {"val": validation_point, "test": test_row}
    mapping = {
        "records": "records",
        "dense_correct": "correct",
        "dense_wrong": "wrong",
        "triggered": "triggered",
        "dense_correct_trigger": "correct_false_triggers",
        "dense_wrong_trigger": "wrong_detected",
        "correct_preservation": "correct_preservation",
        "wrong_trigger_recall": "wrong_detection_recall",
        "trigger_precision": "failure_precision",
    }
    checks = []
    for split in ("val", "test"):
        observed = summaries[split]
        expected = expected_rows[split]
        for current_key, prior_key in mapping.items():
            value = observed[current_key]
            if current_key in {
                "records",
                "dense_correct",
                "dense_wrong",
                "triggered",
                "dense_correct_trigger",
                "dense_wrong_trigger",
            }:
                matches = int(value) == int(expected[prior_key])
            else:
                matches = abs(float(value) - float(expected[prior_key])) <= 1e-15
            checks.append(
                {
                    "split": split,
                    "metric": current_key,
                    "observed": value,
                    "prior": expected[prior_key],
                    "matches": matches,
                }
            )
    passed = all(row["matches"] for row in checks)
    if not passed:
        raise RuntimeError("trigger-map aggregate differs from Phase-52 Shared Random-4")
    return {"passed": True, "checks": checks, "exact_checks": len(checks)}


def _prior_aggregate_consistency(
    contract: Mapping[str, Any], summaries: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    selection = read_json(resolve_path(contract["sources"]["phase52_selection"]))
    validation_point = selection["candidate_validation_points"]["shared_random4"]
    test_row = next(
        row
        for row in read_csv(resolve_path(contract["sources"]["phase52_test_comparison"]))
        if row["candidate"] == "shared_random4"
    )
    return compare_prior_summaries(
        summaries, validation_point=validation_point, test_row=test_row
    )


def _candidate_rows(rows: Sequence[Mapping[str, Any]], *, wrong: bool) -> list[dict[str, Any]]:
    role = "future_corrective_suffix_search" if wrong else "future_full_suffix_preservation"
    return [
        {
            "schema_version": "stage2_future_trigger_candidate_v1",
            "uid": row["uid"],
            "dataset": row["dataset"],
            "split": row["split"],
            "group_id": row["group_id"],
            "first_trigger_layer": row["first_trigger_layer"],
            "dense_correct": bool(row["dense_correct"]),
            "dense_wrong": bool(row["dense_wrong"]),
            "future_role": role,
        }
        for row in rows
        if bool(row["triggered"]) and bool(row["dense_wrong"]) is wrong
    ]


def _save_figures(
    output_root: Path,
    trigger_maps: Mapping[str, Sequence[Mapping[str, Any]]],
    dataset_rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    for class_name, key, filename in (
        ("Dense-W", "dense_wrong", "trigger_hist_wrong.png"),
        ("Dense-C", "dense_correct", "trigger_hist_correct.png"),
    ):
        figure, axis = plt.subplots(figsize=(9, 4.8))
        for split in SPLITS:
            counts = Counter(
                int(row["first_trigger_layer"])
                for row in trigger_maps[split]
                if bool(row[key]) and bool(row["triggered"])
            )
            axis.plot(range(28), [counts[layer] for layer in range(28)], marker="o", ms=3, label=split, color=colors[split])
        axis.set(xlabel="First trigger layer", ylabel="Count", title=f"{class_name} first-trigger distribution")
        axis.set_xticks(range(0, 28, 3))
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(figure_root / filename, dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True, sharey=True)
    for axis, split in zip(axes, SPLITS):
        rows = cumulative_trigger_rows(trigger_maps[split])
        axis.plot([row["layer"] for row in rows], [row["wrong_cumulative_trigger"] for row in rows], label="Dense-W", color="#D62728")
        axis.plot([row["layer"] for row in rows], [row["correct_cumulative_trigger"] for row in rows], label="Dense-C", color="#1F77B4")
        axis.set(title=split, xlabel="Layer", ylim=(0, 1))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Cumulative fraction triggered")
    axes[-1].legend()
    figure.suptitle("Frozen Shared Random-4 cumulative admissions")
    figure.tight_layout()
    figure.savefig(figure_root / "trigger_cumulative_c_vs_w.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for axis, split in zip(axes, SPLITS):
        correct = [row["score_at_trigger"] for row in trigger_maps[split] if row["dense_correct"] and row["triggered"]]
        wrong = [row["score_at_trigger"] for row in trigger_maps[split] if row["dense_wrong"] and row["triggered"]]
        axis.boxplot([correct, wrong], tick_labels=["Dense-C", "Dense-W"], showfliers=False)
        axis.axhline(float(next(iter(trigger_maps[split]))["threshold"]), color="black", linestyle="--", linewidth=1)
        axis.set(title=split, ylabel="Score at first trigger" if split == "train" else "")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("First-crossing score distributions")
    figure.tight_layout()
    figure.savefig(figure_root / "trigger_score_distribution.png", dpi=180)
    plt.close(figure)

    metrics = ("correct_preservation", "wrong_trigger_recall", "trigger_precision")
    labels = ("C preservation", "W recall", "precision")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    x = np.arange(len(DATASETS))
    width = 0.25
    for axis, metric, label in zip(axes, metrics, labels):
        for index, split in enumerate(SPLITS):
            values = [
                float(next(row for row in dataset_rows if row["split"] == split and row["dataset"] == dataset)[metric])
                for dataset in DATASETS
            ]
            axis.bar(x + (index - 1) * width, values, width, label=split, color=colors[split])
        axis.set_xticks(x, DATASETS)
        axis.set(title=label, ylim=(0, 1))
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Fraction")
    axes[-1].legend()
    figure.suptitle("Dataset-specific trigger behavior")
    figure.tight_layout()
    figure.savefig(figure_root / "dataset_trigger_breakdown.png", dpi=180)
    plt.close(figure)


def _mode_layers(rows: Sequence[Mapping[str, Any]], class_key: str) -> tuple[list[int], int]:
    counts = Counter(
        int(row["first_trigger_layer"])
        for row in rows
        if bool(row[class_key]) and bool(row["triggered"])
    )
    maximum = max(counts.values())
    return sorted(layer for layer, count in counts.items() if count == maximum), maximum


def _decision_summary(
    contract: Mapping[str, Any],
    trigger_maps: Mapping[str, Sequence[Mapping[str, Any]]],
    summaries: Mapping[str, Mapping[str, Any]],
    dataset_rows: Sequence[Mapping[str, Any]],
    reproducibility: Mapping[str, Any],
) -> str:
    lines = [
        "# Stage-1 Trigger Map Decision Summary",
        "",
        f"Frozen audit contract: `{contract['contract_sha256']}`. The gate is Shared Random-4 with strict `score > {float(contract['gate']['threshold']):.17g}` over layers 0-27.",
        "",
        "| Split | Dense-C retained | Dense-W detected | Trigger precision | C trigger | W trigger |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        row = summaries[split]
        lines.append(
            f"| {split} | {row['correct_preservation']:.4f} | {row['wrong_trigger_recall']:.4f} | {row['trigger_precision']:.4f} | {row['dense_correct_trigger']} | {row['dense_wrong_trigger']} |"
        )
    lines.extend(["", "## Required answers", ""])
    lines.append("1. Dense-C retention without triggering is " + ", ".join(f"{split} {summaries[split]['correct_preservation']:.2%}" for split in SPLITS) + ".")
    lines.append("2. Dense-W detection is " + ", ".join(f"{split} {summaries[split]['wrong_trigger_recall']:.2%}" for split in SPLITS) + ".")
    lines.append("3. The triggered population is Dense-W at rates " + ", ".join(f"{split} {summaries[split]['trigger_precision']:.2%}" for split in SPLITS) + ".")
    wrong_modes = {split: _mode_layers(trigger_maps[split], "dense_wrong") for split in SPLITS}
    correct_modes = {split: _mode_layers(trigger_maps[split], "dense_correct") for split in SPLITS}
    lines.append("4. Modal Dense-W first-trigger layers are " + ", ".join(f"{split} {layers} ({count} samples each)" for split, (layers, count) in wrong_modes.items()) + ".")
    lines.append("5. Modal false-admission Dense-C first-trigger layers are " + ", ".join(f"{split} {layers} ({count} samples each)" for split, (layers, count) in correct_modes.items()) + ".")
    curve_bits = []
    for split in SPLITS:
        curve = cumulative_trigger_rows(trigger_maps[split])
        gaps = [row["wrong_cumulative_trigger"] - row["correct_cumulative_trigger"] for row in curve]
        layer = int(np.argmax(gaps))
        curve_bits.append(f"{split} maximum W-minus-C gap {gaps[layer]:.4f} at layer {layer} (final gap {gaps[-1]:.4f})")
    lines.append("6. The cumulative curves separate as follows: " + "; ".join(curve_bits) + ".")
    lines.append("7. Dataset behavior is materially nonuniform: " + "; ".join(
        f"{split} W recall " + "/".join(f"{dataset}={next(row for row in dataset_rows if row['split']==split and row['dataset']==dataset)['wrong_trigger_recall']:.3f}" for dataset in DATASETS)
        for split in SPLITS
    ) + ". No per-dataset calibration was applied.")
    lines.append(f"8. Future train corrective suffix search contains exactly **{summaries['train']['dense_wrong_trigger']}** triggered Dense-W samples.")
    lines.append(f"9. Future train preservation supervision contains exactly **{summaries['train']['dense_correct_trigger']}** triggered Dense-C samples, whose default suffix is FULL.")
    lines.append("10. Yes. Validation and test counts, preservation, recall, and precision match all 18 checked Phase-52 Shared Random-4 aggregate fields exactly; all 1,600 Phase-53 trigger rows also match exactly.")
    lines.extend(
        [
            "",
            "## Provenance and boundaries",
            "",
            f"The deterministic 24-record stored-feature check passed with maximum absolute probability error `{float(reproducibility['maximum_absolute_error']):.3g}` at tolerance `{float(reproducibility['absolute_probability_tolerance']):.3g}`; {reproducibility['exact_layer_scores']}/{reproducibility['layer_scores']} layer scores were bit-exact, and every first-trigger decision matched.",
            "Validation/test trajectories came from saved Phase-51 scores. Train trajectories were generated only by the frozen lightweight gate from hash-verified stored dense features on four direct GPUs. No Qwen or four-action execution occurred.",
            "The map describes who triggers and when. It does not show that early triggers are better, that any triggered failure is fixable, or that Stage 2 improves final accuracy.",
            "Stopped at the plan boundary: no corrective suffix search, Stage-2 labels/training, threshold changes, persistence/EMA, W-to-C repair, or external evaluation ran.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_manifest(output_root: Path, contract_sha256: str) -> dict[str, Any]:
    paths = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name not in {"artifact_manifest.json", "completion.json"}
    )
    return {
        "schema_version": "stage1_trigger_map_artifact_manifest_v1",
        "passed": True,
        "contract_sha256": contract_sha256,
        "files": {str(path.relative_to(output_root)): file_sha256(path) for path in paths},
    }


def aggregate(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    split_rows = _validate_split(contract["static_config"])
    train_scores, repro_scores = _worker_outputs(contract, output_root, split_rows)
    reproducibility = _reproducibility_audit(contract, output_root, split_rows, repro_scores)
    saved_scores = {
        "val": _load_saved_scores(contract["static_config"], split_rows, "val"),
        "test": _load_saved_scores(contract["static_config"], split_rows, "test"),
    }
    threshold = float(contract["gate"]["threshold"])
    trigger_maps = {
        "train": build_trigger_map(
            [row for row in split_rows if row["split"] == "train"],
            train_scores,
            threshold=threshold,
            score_source="recomputed_frozen_features",
        ),
        "val": build_trigger_map(
            [row for row in split_rows if row["split"] == "val"],
            saved_scores["val"],
            threshold=threshold,
            score_source="saved_phase51_scores",
        ),
        "test": build_trigger_map(
            [row for row in split_rows if row["split"] == "test"],
            saved_scores["test"],
            threshold=threshold,
            score_source="saved_phase51_scores",
        ),
    }
    if sum(len(rows) for rows in trigger_maps.values()) != 7999 or len(
        {row["uid"] for rows in trigger_maps.values() for row in rows}
    ) != 7999:
        raise RuntimeError("global trigger map does not cover 7,999 unique UIDs")
    phase53 = _phase53_consistency(contract, trigger_maps)
    summaries = {split: split_summary(trigger_maps[split]) for split in SPLITS}
    prior = _prior_aggregate_consistency(contract, summaries)

    manifest_root = output_root / "manifests"
    for split in SPLITS:
        atomic_jsonl(manifest_root / f"trigger_map_{split}.jsonl", trigger_maps[split])
        atomic_jsonl(manifest_root / f"{split}_triggered_correct.jsonl", _candidate_rows(trigger_maps[split], wrong=False))
        atomic_jsonl(manifest_root / f"{split}_triggered_wrong.jsonl", _candidate_rows(trigger_maps[split], wrong=True))
    partition_uids = {
        split: {
            "dense_correct_no_trigger": [row["uid"] for row in trigger_maps[split] if row["dense_correct"] and not row["triggered"]],
            "dense_correct_trigger": [row["uid"] for row in trigger_maps[split] if row["dense_correct"] and row["triggered"]],
            "dense_wrong_no_trigger": [row["uid"] for row in trigger_maps[split] if row["dense_wrong"] and not row["triggered"]],
            "dense_wrong_trigger": [row["uid"] for row in trigger_maps[split] if row["dense_wrong"] and row["triggered"]],
        }
        for split in SPLITS
    }
    atomic_json(manifest_root / "partition_uid_lists.json", partition_uids)

    layer_rows = [row for split in SPLITS for row in trigger_by_layer_rows(trigger_maps[split])]
    bin_rows = [row for split in SPLITS for row in trigger_by_depth_bin_rows(trigger_maps[split])]
    cumulative_rows = [row for split in SPLITS for row in cumulative_trigger_rows(trigger_maps[split])]
    dataset_rows = [row for split in SPLITS for row in dataset_breakdown_rows(trigger_maps[split])]
    score_rows = [row for split in SPLITS for row in trigger_score_stats_rows(trigger_maps[split])]
    metric_root = output_root / "metrics"
    atomic_csv(metric_root / "split_summary.csv", [summaries[split] for split in SPLITS])
    atomic_csv(metric_root / "trigger_by_layer.csv", layer_rows)
    atomic_csv(metric_root / "trigger_by_depth_bin.csv", bin_rows)
    atomic_csv(metric_root / "cumulative_trigger.csv", cumulative_rows)
    atomic_csv(metric_root / "dataset_breakdown.csv", dataset_rows)
    atomic_csv(metric_root / "trigger_score_stats.csv", score_rows)
    atomic_json(
        output_root / "consistency_audit.json",
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "population_records": 7999,
            "unique_uids": 7999,
            "phase53": phase53,
            "phase52": prior,
            "reproducibility": reproducibility,
            "partition_checks": {
                split: sum(
                    summaries[split][key]
                    for key in (
                        "dense_correct_no_trigger",
                        "dense_correct_trigger",
                        "dense_wrong_no_trigger",
                        "dense_wrong_trigger",
                    )
                )
                for split in SPLITS
            },
        },
    )
    _save_figures(output_root, trigger_maps, dataset_rows)
    write_once(
        output_root / "decision_summary.md",
        _decision_summary(contract, trigger_maps, summaries, dataset_rows, reproducibility).encode("utf-8"),
    )
    artifact = _artifact_manifest(output_root, contract["contract_sha256"])
    atomic_json(output_root / "artifact_manifest.json", artifact)
    atomic_json(
        output_root / "completion.json",
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "records": 7999,
            "splits": {split: summaries[split] for split in SPLITS},
            "required_artifacts": len(artifact["files"]),
            "artifact_manifest_sha256": file_sha256(output_root / "artifact_manifest.json"),
            "qwen_forward_executed": False,
            "four_action_search_executed": False,
            "stage2_training_executed": False,
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "contract_sha256": contract["contract_sha256"],
                "train_triggered_correct": summaries["train"]["dense_correct_trigger"],
                "train_triggered_wrong": summaries["train"]["dense_wrong_trigger"],
                "reproducibility_max_error": reproducibility["maximum_absolute_error"],
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    worker = subparsers.add_parser("score-worker")
    worker.add_argument("--rank", type=int, required=True)
    worker.add_argument("--world-size", type=int, default=4)
    subparsers.add_parser("aggregate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if args.command == "prepare":
        prepare(config_path)
    elif args.command == "score-worker":
        score_worker(config_path, rank=args.rank, world_size=args.world_size)
    else:
        aggregate(config_path)


if __name__ == "__main__":
    main()
