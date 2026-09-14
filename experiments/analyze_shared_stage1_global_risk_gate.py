#!/usr/bin/env python3
"""Train and evaluate the frozen shared Stage-1 global-risk gate."""

from __future__ import annotations

import argparse
from collections import Counter
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
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dense_failure_stage1.layerwise_probe import binary_metrics  # noqa: E402
from dense_failure_stage1.shared_global_gate import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    MAIN_VARIANTS,
    VARIANTS,
    SharedFailurePredictor,
    deterministic_random_layers,
    first_trigger_layers,
    layerwise_ranking_metrics,
    score_space_rows,
    select_candidate,
    threshold_sweep,
    trajectory_operating_point,
    validate_score_matrix,
)
from dense_failure_stage1.sequential_gate import gate_metrics  # noqa: E402
from experiments.analyze_layerwise_dense_failure_predictability import (  # noqa: E402
    load_frozen_contract as load_phase48_contract,
    load_layer_matrices,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/shared_stage1_global_risk_gate_v1.json"
PHASE48_CONFIG = PROJECT_ROOT / "configs/layerwise_dense_failure_probe_v1.json"
DATASETS = ("gqa", "chartqa", "textvqa")
BOUND_CODE_PATHS = (
    "configs/shared_stage1_global_risk_gate_v1.json",
    "dense_failure_stage1/shared_global_gate.py",
    "experiments/analyze_shared_stage1_global_risk_gate.py",
    "dense_failure_stage1/layerwise_probe.py",
    "dense_failure_stage1/sequential_gate.py",
    "experiments/analyze_layerwise_dense_failure_predictability.py",
)


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
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def atomic_torch(path: Path, value: Any) -> None:
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


def resolve_path(value: str) -> Path:
    path = (PROJECT_ROOT / value).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path escapes project root: {value}")
    return path


def load_static_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "shared_stage1_global_risk_gate_config_v1":
        raise ValueError("unsupported shared-gate static config")
    if [row["name"] for row in config["variants"]] != list(VARIANTS):
        raise ValueError("shared-gate variants differ from the frozen four-config contract")
    if [int(row["worker_rank"]) for row in config["variants"]] != list(range(4)):
        raise ValueError("shared-gate variants must map one-to-one to four workers")
    if int(config["input"]["input_size"]) != 10752 or int(config["input"]["layers"]) != 28:
        raise ValueError("shared-gate input shape differs")
    if config["evaluation"]["test_models"] != list(MAIN_VARIANTS):
        raise ValueError("only the two state+layer schemes may receive new test scores")
    return config


def _phase48() -> tuple[dict[str, Any], Path]:
    contract, root = load_phase48_contract(PHASE48_CONFIG)
    if contract.get("contract_sha256") != "3cf49a46d0a47a0ae1955f8a518f5a74bef65b7de8736980a6e3d26e170234cd":
        raise ValueError("Phase-48 contract identity differs")
    return contract, root


def _variant_config(static: Mapping[str, Any], variant: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "shared_stage1_variant_config_v1",
        "name": variant["name"],
        "uses_state": bool(variant["uses_state"]),
        "uses_layer": bool(variant["uses_layer"]),
        "scheme": variant["scheme"],
        "worker_rank": int(variant["worker_rank"]),
        "input": static["input"],
        "architecture": static["architecture"],
        "training": static["training"],
    }


def _load_split_rows(phase48_root: Path, selected: set[str]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(phase48_root / "split_manifest.jsonl")
        if str(row["split"]) in selected
    ]
    rows.sort(key=lambda row: str(row["uid"]))
    return rows


def _compute_global_normalization(
    phase48: Mapping[str, Any], phase48_root: Path, *, std_floor: float
) -> dict[str, Any]:
    torch.set_num_threads(8)
    rows, matrices = load_layer_matrices(
        phase48, phase48_root, layers=range(28), selected_splits={"train"}
    )
    if len(rows) != 6399 or any(row["split"] != "train" for row in rows):
        raise RuntimeError("global normalization did not receive the frozen train split")
    mean_sum = torch.zeros(10752, dtype=torch.float64)
    second_sum = torch.zeros(10752, dtype=torch.float64)
    for layer in range(28):
        values = matrices.pop(layer).float()
        layer_mean = values.mean(dim=0).double()
        layer_var = values.var(dim=0, unbiased=False).double()
        mean_sum += layer_mean
        second_sum += layer_var + layer_mean.square()
        del values, layer_mean, layer_var
    mean = mean_sum / 28.0
    variance = torch.clamp(second_sum / 28.0 - mean.square(), min=0.0)
    std = variance.sqrt()
    std = torch.where(std < float(std_floor), torch.ones_like(std), std)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise RuntimeError("global normalization contains non-finite values")
    return {
        "schema_version": "shared_stage1_global_normalization_v1",
        "train_records": 6399,
        "layers_per_record": 28,
        "observations": 6399 * 28,
        "input_size": 10752,
        "method": "pooled_train_only_population_mean_std",
        "std_floor": float(std_floor),
        "mean": mean.float(),
        "std": std.float(),
    }


def _protocol_markdown(contract: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Shared Stage-1 Global Risk Gate Protocol",
            "",
            f"- Frozen contract: `{contract['contract_sha256']}`",
            f"- Phase-48 contract: `{contract['provenance']['phase48_contract_sha256']}`",
            "- Exact frozen split: 6,399 train / 800 validation / 800 test; zero UID and image-group overlap.",
            "- One train-only normalization is pooled across all train samples and all 28 layers.",
            "- Four configs only: state-only All-28, layer-only All-28, state+layer All-28, state+layer Random-4.",
            "- Architecture: 10,752 -> 256 projection, 32-dimensional layer embedding when used, 256-dimensional GELU head, no dropout.",
            "- Optimization: deterministic FP32 AdamW, lr 5e-4, weight decay 0.01, cosine schedule, 10 epochs; selected by minimum full-28 validation BCE.",
            "- Global thresholds are the most permissive tie-safe validation trajectory thresholds under strict `p > tau` at 99%/98%/95% correct preservation.",
            "- State/layer-only controls remain validation-only. The two state+layer schemes and their windows are frozen before one aggregate test pass.",
            "- The validation-designated primary scheme cannot be switched from test results.",
            "- A trigger is admission only; no treatment or OOD evaluation is executed.",
            "",
        ]
    )


def prepare(config_path: Path) -> None:
    static = load_static_config(config_path)
    output_root = resolve_path(static["output_root"])
    phase48, phase48_root = _phase48()
    split_audit = read_json(phase48_root / "split_audit.json")
    if (
        split_audit.get("passed") is not True
        or int(split_audit["uid_overlap"]) != 0
        or int(split_audit["image_group_overlap"]) != 0
        or {key: int(value["records"]) for key, value in split_audit["splits"].items()}
        != {"train": 6399, "val": 800, "test": 800}
    ):
        raise RuntimeError("Phase-48 split audit differs from the shared-gate contract")
    phase50 = read_json(resolve_path(static["sources"]["phase50_manifest"]))
    if phase50.get("passed") is not True or not phase50.get(
        "test_evaluated_once_after_validation_freeze"
    ):
        raise RuntimeError("Phase-50 baseline manifest is invalid")

    generated_hashes = {}
    for variant in static["variants"]:
        payload = (
            json.dumps(
                _variant_config(static, variant),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        relative = f"configs/{variant['name']}.json"
        write_once_or_verify(output_root / relative, payload)
        generated_hashes[relative] = file_sha256(output_root / relative)

    normalization_path = output_root / "training/global_normalization.pt"
    if not normalization_path.exists():
        normalization = _compute_global_normalization(
            phase48,
            phase48_root,
            std_floor=float(static["input"]["normalization_std_floor"]),
        )
        atomic_torch(normalization_path, normalization)
    normalization = torch.load(normalization_path, map_location="cpu", weights_only=True)
    if (
        normalization.get("schema_version") != "shared_stage1_global_normalization_v1"
        or tuple(normalization["mean"].shape) != (10752,)
        or tuple(normalization["std"].shape) != (10752,)
    ):
        raise RuntimeError("shared normalization artifact is invalid")

    status = command_output(["git", "status", "--porcelain=v1", "--untracked-files=all"])
    source_hashes = {
        name: file_sha256(resolve_path(path)) for name, path in static["sources"].items()
    }
    bound_hashes = {path: file_sha256(resolve_path(path)) for path in BOUND_CODE_PATHS}
    frozen = json.loads(json.dumps(static))
    frozen["schema_version"] = "shared_stage1_global_risk_gate_frozen_contract_v1"
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
        "generated_config_sha256": generated_hashes,
        "global_normalization_sha256": file_sha256(normalization_path),
        "phase48_contract_sha256": phase48["contract_sha256"],
        "phase50_contract_sha256": phase50["contract_sha256"],
    }
    frozen["contract_sha256"] = canonical_hash(frozen)
    write_once_or_verify(
        output_root / "frozen_protocol.json",
        (json.dumps(frozen, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    write_once_or_verify(
        output_root / "protocol.md", _protocol_markdown(frozen).encode("utf-8")
    )
    atomic_json(
        output_root / "preparation_audit.json",
        {
            "passed": True,
            "contract_sha256": frozen["contract_sha256"],
            "phase48_contract_sha256": phase48["contract_sha256"],
            "phase50_contract_sha256": phase50["contract_sha256"],
            "global_normalization_sha256": file_sha256(normalization_path),
            "generated_configs": len(generated_hashes),
            "split_records": {"train": 6399, "validation": 800, "test": 800},
        },
    )
    print(json.dumps({"passed": True, "contract_sha256": frozen["contract_sha256"]}))


def load_contract(
    config_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    static = load_static_config(config_path)
    output_root = resolve_path(static["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if (
        contract.get("schema_version")
        != "shared_stage1_global_risk_gate_frozen_contract_v1"
        or contract.get("contract_sha256") != canonical_hash(contract)
    ):
        raise ValueError("shared-gate frozen contract is invalid")
    provenance = contract["provenance"]
    if provenance["static_config_sha256"] != file_sha256(config_path):
        raise ValueError("shared-gate static config differs")
    for path, expected in provenance["bound_code_sha256"].items():
        if file_sha256(resolve_path(path)) != expected:
            raise ValueError(f"bound shared-gate code differs: {path}")
    for name, expected in provenance["source_sha256"].items():
        if file_sha256(resolve_path(contract["sources"][name])) != expected:
            raise ValueError(f"shared-gate source differs: {name}")
    for relative, expected in provenance["generated_config_sha256"].items():
        if file_sha256(output_root / relative) != expected:
            raise ValueError(f"generated variant config differs: {relative}")
    normalization_path = output_root / "training/global_normalization.pt"
    if file_sha256(normalization_path) != provenance["global_normalization_sha256"]:
        raise ValueError("global normalization hash differs")
    phase48, phase48_root = _phase48()
    if phase48["contract_sha256"] != provenance["phase48_contract_sha256"]:
        raise ValueError("Phase-48 identity differs from shared-gate freeze")
    return contract, output_root, phase48, phase48_root


def _variant(contract: Mapping[str, Any], name: str) -> dict[str, Any]:
    matches = [row for row in contract["variants"] if row["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"variant is not uniquely frozen: {name}")
    return dict(matches[0])


def _normalization(output_root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.load(
        output_root / "training/global_normalization.pt",
        map_location="cpu",
        weights_only=True,
    )
    return value["mean"].float(), value["std"].float()


def _feature_tensor(
    phase48: Mapping[str, Any], phase48_root: Path, *, splits: set[str]
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    rows, matrices = load_layer_matrices(
        phase48, phase48_root, layers=range(28), selected_splits=splits
    )
    features = torch.stack([matrices.pop(layer) for layer in range(28)], dim=1)
    if features.shape != (len(rows), 28, 10752):
        raise RuntimeError("shared-gate feature tensor shape differs")
    return rows, features


def _build_model(contract: Mapping[str, Any], variant: str) -> SharedFailurePredictor:
    architecture = contract["architecture"]
    return SharedFailurePredictor(
        variant=variant,
        input_size=int(contract["input"]["input_size"]),
        projection_size=int(architecture["projection_size"]),
        layer_embedding_size=int(architecture["layer_embedding_size"]),
        hidden_size=int(architecture["hidden_size"]),
    )


def _batch_logits(
    model: SharedFailurePredictor,
    features: torch.Tensor | None,
    sample_indices: torch.Tensor,
    layer_choices: torch.Tensor,
    *,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    batch = len(sample_indices)
    count = layer_choices.shape[1]
    layers = layer_choices.reshape(-1).to(device=device, dtype=torch.long)
    states = None
    if model.uses_state:
        if features is None or mean is None or std is None:
            raise RuntimeError("state model has no features or normalization")
        selected = features.index_select(0, sample_indices)
        row = torch.arange(batch, dtype=torch.long)[:, None]
        selected = selected[row, layer_choices]
        states = selected.reshape(batch * count, -1).to(device=device, dtype=torch.float32)
        states = (states - mean) / std
    return model(states, layers)


def _score_full28(
    model: SharedFailurePredictor,
    features: torch.Tensor | None,
    *,
    records: int,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_rows = []
    all_layers = torch.arange(28, dtype=torch.long)[None, :]
    with torch.inference_mode():
        for start in range(0, records, batch_size):
            stop = min(start + batch_size, records)
            indices = torch.arange(start, stop, dtype=torch.long)
            layers = all_layers.expand(stop - start, -1)
            logits = _batch_logits(
                model,
                features,
                indices,
                layers,
                mean=mean,
                std=std,
                device=device,
            )
            logits_rows.append(logits.reshape(stop - start, 28).cpu())
    logits = torch.cat(logits_rows).numpy().astype(np.float64)
    scores = np.empty_like(logits)
    positive = logits >= 0
    scores[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    scores[~positive] = exponential / (1.0 + exponential)
    return logits, scores


def _score_rows(
    rows: Sequence[Mapping[str, Any]], scores: np.ndarray, *, variant: str
) -> list[dict[str, Any]]:
    if scores.shape != (len(rows), 28):
        raise RuntimeError("score rows do not cover N x 28")
    output = []
    for row, values in zip(rows, scores):
        record = {
            "schema_version": "shared_stage1_score_trajectory_v1",
            "variant": variant,
            "uid": str(row["uid"]),
            "dataset": str(row["dataset"]),
            "image_group_id": str(row["image_group_id"]),
            "current_dense_wrong": bool(row["current_dense_wrong"]),
        }
        record.update({f"p_{layer}": float(values[layer]) for layer in range(28)})
        output.append(record)
    return output


def _arrays(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(
        [[float(row[f"p_{layer}"]) for layer in range(28)] for row in rows],
        dtype=np.float64,
    )
    labels = np.asarray([int(bool(row["current_dense_wrong"])) for row in rows])
    return validate_score_matrix(scores, labels)


def _require_gpu(rank: int, world_size: int) -> torch.device:
    if world_size != 4 or not 0 <= rank < 4:
        raise ValueError("shared-gate execution requires four ranks")
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


def train_worker(config_path: Path, *, rank: int, world_size: int) -> None:
    contract, output_root, phase48, phase48_root = load_contract(config_path)
    device = _require_gpu(rank, world_size)
    variant_spec = [
        row for row in contract["variants"] if int(row["worker_rank"]) == rank
    ]
    if len(variant_spec) != 1:
        raise RuntimeError("rank does not map to exactly one variant")
    spec = variant_spec[0]
    variant = str(spec["name"])
    complete_path = output_root / f"training/histories/{variant}.complete.json"
    checkpoint_path = output_root / f"training/checkpoints/{variant}.pt"
    score_path = output_root / f"evaluation/validation_scores/{variant}.jsonl"
    if complete_path.exists():
        complete = read_json(complete_path)
        if (
            complete.get("contract_sha256") != contract["contract_sha256"]
            or complete.get("checkpoint_sha256") != file_sha256(checkpoint_path)
            or complete.get("validation_scores_sha256") != file_sha256(score_path)
        ):
            raise RuntimeError(f"existing {variant} worker artifacts are incompatible")
        print(json.dumps({"passed": True, "variant": variant, "resumed": True}))
        return

    torch.manual_seed(int(contract["seed"]) + rank)
    torch.cuda.manual_seed_all(int(contract["seed"]) + rank)
    rows = _load_split_rows(phase48_root, {"train", "val"})
    features = None
    if bool(spec["uses_state"]):
        loaded_rows, features = _feature_tensor(
            phase48, phase48_root, splits={"train", "val"}
        )
        if [row["uid"] for row in loaded_rows] != [row["uid"] for row in rows]:
            raise RuntimeError("feature and split row ordering differs")
    train_indices = torch.tensor(
        [index for index, row in enumerate(rows) if row["split"] == "train"],
        dtype=torch.long,
    )
    val_indices = torch.tensor(
        [index for index, row in enumerate(rows) if row["split"] == "val"],
        dtype=torch.long,
    )
    if len(train_indices) != 6399 or len(val_indices) != 800:
        raise RuntimeError("train/validation split counts differ")
    labels = torch.tensor(
        [int(bool(row["current_dense_wrong"])) for row in rows], dtype=torch.float32
    )
    train_labels = labels.index_select(0, train_indices)
    val_labels = labels.index_select(0, val_indices)
    train_features = features.index_select(0, train_indices) if features is not None else None
    val_features = features.index_select(0, val_indices) if features is not None else None
    del features

    model = _build_model(contract, variant).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(contract["training"]["learning_rate"]),
        weight_decay=float(contract["training"]["weight_decay"]),
    )
    epochs = int(contract["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    batch_size = int(contract["training"]["batch_size_samples"])
    mean_cpu, std_cpu = _normalization(output_root)
    mean = mean_cpu.to(device) if model.uses_state else None
    std = std_cpu.to(device) if model.uses_state else None
    permutation_generator = torch.Generator(device="cpu").manual_seed(
        int(contract["seed"]) + 10_000 + rank
    )
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    history = []
    all_layers = torch.arange(28, dtype=torch.long)[None, :]

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(train_indices), generator=permutation_generator)
        random_layers = (
            deterministic_random_layers(
                len(train_indices),
                epoch=epoch,
                seed=int(contract["seed"]) + rank,
                count=int(contract["training"]["random_layers_per_sample"]),
            )
            if spec["scheme"] == "random4"
            else None
        )
        accumulated = 0.0
        for start in range(0, len(permutation), batch_size):
            local = permutation[start : start + batch_size]
            choices = (
                random_layers.index_select(0, local)
                if random_layers is not None
                else all_layers.expand(len(local), -1)
            )
            optimizer.zero_grad(set_to_none=True)
            logits = _batch_logits(
                model,
                train_features,
                local,
                choices,
                mean=mean,
                std=std,
                device=device,
            )
            targets = train_labels.index_select(0, local).to(device).repeat_interleave(
                choices.shape[1]
            )
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()
            accumulated += float(loss.detach().cpu()) * len(local)

        validation_logits, validation_scores = _score_full28(
            model,
            val_features,
            records=len(val_indices),
            mean=mean,
            std=std,
            device=device,
            batch_size=batch_size,
        )
        repeated_labels = np.repeat(val_labels.numpy().astype(np.int64), 28)
        flat_logits = validation_logits.reshape(-1)
        validation_bce = float(
            np.mean(
                np.maximum(flat_logits, 0)
                - flat_logits * repeated_labels
                + np.log1p(np.exp(-np.abs(flat_logits)))
            )
        )
        ranking = binary_metrics(repeated_labels, validation_scores.reshape(-1))
        train_loss = accumulated / len(train_indices)
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": float(train_loss),
                "validation_full28_bce": validation_bce,
                "validation_pooled_auroc": float(ranking["auroc"]),
                "validation_pooled_auprc": float(ranking["auprc"]),
            }
        )
        if validation_bce < best_loss - 1e-12:
            best_loss = validation_bce
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        scheduler.step()

    if best_state is None or best_epoch < 0:
        raise RuntimeError("training did not select a validation checkpoint")
    model.load_state_dict(best_state)
    _, validation_scores = _score_full28(
        model,
        val_features,
        records=len(val_indices),
        mean=mean,
        std=std,
        device=device,
        batch_size=batch_size,
    )
    val_rows = [rows[int(index)] for index in val_indices.tolist()]
    score_rows = _score_rows(val_rows, validation_scores, variant=variant)
    atomic_jsonl(score_path, score_rows)
    history_path = output_root / f"training/histories/{variant}.json"
    atomic_json(
        history_path,
        {
            "schema_version": "shared_stage1_training_history_v1",
            "contract_sha256": contract["contract_sha256"],
            "variant": variant,
            "best_epoch": best_epoch,
            "best_validation_full28_bce": best_loss,
            "history": history,
        },
    )
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "contract_sha256": contract["contract_sha256"],
        "variant": variant,
        "worker_rank": rank,
        "phase48_contract_sha256": contract["provenance"]["phase48_contract_sha256"],
        "global_normalization_sha256": contract["provenance"]["global_normalization_sha256"],
        "variant_config_sha256": contract["provenance"]["generated_config_sha256"][
            f"configs/{variant}.json"
        ],
        "best_epoch": best_epoch,
        "best_validation_full28_bce": best_loss,
        "model_state_dict": best_state,
    }
    atomic_torch(checkpoint_path, checkpoint)
    atomic_json(
        complete_path,
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "variant": variant,
            "worker_rank": rank,
            "best_epoch": best_epoch,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "history_sha256": file_sha256(history_path),
            "validation_scores_sha256": file_sha256(score_path),
            "validation_records": len(score_rows),
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "variant": variant,
                "best_epoch": best_epoch,
                "validation_bce": best_loss,
            }
        )
    )


def _load_checkpoint(
    contract: Mapping[str, Any], output_root: Path, variant: str
) -> dict[str, Any]:
    path = output_root / f"training/checkpoints/{variant}.pt"
    complete = read_json(output_root / f"training/histories/{variant}.complete.json")
    if (
        complete.get("passed") is not True
        or complete.get("contract_sha256") != contract["contract_sha256"]
        or complete.get("variant") != variant
        or complete.get("checkpoint_sha256") != file_sha256(path)
    ):
        raise RuntimeError(f"{variant} training is incomplete or incompatible")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("contract_sha256") != contract["contract_sha256"]
        or checkpoint.get("variant") != variant
        or checkpoint.get("phase48_contract_sha256")
        != contract["provenance"]["phase48_contract_sha256"]
        or checkpoint.get("global_normalization_sha256")
        != contract["provenance"]["global_normalization_sha256"]
    ):
        raise RuntimeError(f"{variant} checkpoint provenance differs")
    return checkpoint


def _load_validation_scores(
    contract: Mapping[str, Any], output_root: Path, phase48_root: Path, variant: str
) -> list[dict[str, Any]]:
    complete = read_json(output_root / f"training/histories/{variant}.complete.json")
    path = output_root / f"evaluation/validation_scores/{variant}.jsonl"
    if complete.get("validation_scores_sha256") != file_sha256(path):
        raise RuntimeError(f"{variant} validation score hash differs")
    rows = read_jsonl(path)
    expected = _load_split_rows(phase48_root, {"val"})
    if (
        len(rows) != 800
        or len({row["uid"] for row in rows}) != 800
        or [row["uid"] for row in rows] != [row["uid"] for row in expected]
        or any(row.get("variant") != variant for row in rows)
    ):
        raise RuntimeError(f"{variant} validation scores do not cover the frozen UIDs")
    _arrays(rows)
    return rows


def _ranking_rows(
    rows: Sequence[Mapping[str, Any]], *, split: str, variant: str
) -> list[dict[str, Any]]:
    scores, labels = _arrays(rows)
    output = []
    for dataset in ("overall", *DATASETS):
        indices = np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if dataset == "overall" or row["dataset"] == dataset
            ],
            dtype=np.int64,
        )
        for metric in layerwise_ranking_metrics(scores[indices], labels[indices]):
            output.append(
                {
                    "split": split,
                    "variant": variant,
                    "dataset": dataset,
                    "layer": int(metric["layer"]),
                    "records": int(metric["records"]),
                    "correct": int(metric["correct"]),
                    "wrong": int(metric["wrong"]),
                    "auroc": float(metric["auroc"]),
                    "auprc": float(metric["auprc"]),
                }
            )
    return output


def _independent_ranking_rows(
    contract: Mapping[str, Any], *, split: str
) -> list[dict[str, Any]]:
    if split not in {"validation", "test"}:
        raise ValueError("independent ranking split is invalid")
    output = []
    if split == "validation":
        source = read_csv(resolve_path(contract["sources"]["phase48_validation_metrics"]))
        for row in source:
            output.append(
                {
                    "split": split,
                    "variant": "independent_linear",
                    "dataset": "overall",
                    "layer": int(row["layer"]),
                    "records": 800,
                    "correct": 400,
                    "wrong": 400,
                    "auroc": float(row["validation_auroc"]),
                    "auprc": float(row["validation_auprc"]),
                }
            )
    else:
        source = read_csv(resolve_path(contract["sources"]["phase48_test_metrics"]))
        for row in source:
            output.append(
                {
                    "split": split,
                    "variant": "independent_linear",
                    "dataset": "overall",
                    "layer": int(row["layer"]),
                    "records": 800,
                    "correct": 400,
                    "wrong": 400,
                    "auroc": float(row["test_auroc"]),
                    "auprc": float(row["test_auprc"]),
                }
            )
    for row in read_csv(resolve_path(contract["sources"]["phase48_dataset_metrics"])):
        if row["split"] != split:
            continue
        output.append(
            {
                "split": split,
                "variant": "independent_linear",
                "dataset": row["dataset"],
                "layer": int(row["layer"]),
                "records": int(row["records"]),
                "correct": int(row["correct"]),
                "wrong": int(row["wrong"]),
                "auroc": float(row["auroc"]),
                "auprc": float(row["auprc"]),
            }
        )
    return output


def _flat_gate_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"window", "thresholds"}
    }


def _fixed_result_row(
    *,
    source: str,
    split: str,
    variant: str,
    layer: int,
    target: float,
    threshold: float,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source": source,
        "split": split,
        "variant": variant,
        "layer": int(layer),
        "target_preservation": float(target),
        "threshold": float(threshold),
        "records": int(metrics["records"]),
        "correct": int(metrics["correct"]),
        "wrong": int(metrics["wrong"]),
        "correct_preservation": float(metrics["correct_preservation"]),
        "wrong_detection_recall": float(metrics["wrong_detection_recall"]),
        "failure_precision": float(metrics["failure_precision"]),
        "trigger_rate": float(metrics["trigger_rate"]),
        "median_first_trigger_layer": metrics["median_first_trigger_layer"],
        "mean_first_trigger_layer": metrics["mean_first_trigger_layer"],
        "no_trigger_fraction": float(metrics["no_trigger_fraction"]),
        "correct_false_triggers": int(metrics["correct_false_triggers"]),
        "wrong_detected": int(metrics["wrong_detected"]),
    }


def _dataset_gate_rows(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    split: str,
    variant: str,
    window_name: str,
    window: Sequence[int],
    target: float,
    threshold: float,
) -> list[dict[str, Any]]:
    output = []
    for dataset in DATASETS:
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row["dataset"] == dataset],
            dtype=np.int64,
        )
        first = first_trigger_layers(
            scores[indices], threshold=threshold, window=window
        )
        output.append(
            {
                "split": split,
                "variant": variant,
                "window": window_name,
                "target_preservation": target,
                "threshold": threshold,
                "dataset": dataset,
                **gate_metrics(labels[indices], first),
            }
        )
    return output


def _trigger_distribution_rows(
    labels: np.ndarray,
    first: np.ndarray,
    *,
    split: str,
    variant: str,
    window_name: str,
    target: float,
) -> list[dict[str, Any]]:
    output = []
    for class_name, class_value in (("correct", 0), ("wrong", 1)):
        selected = first[labels == class_value]
        total = len(selected)
        for layer in [-1, *range(28)]:
            count = int((selected == layer).sum())
            output.append(
                {
                    "split": split,
                    "variant": variant,
                    "window": window_name,
                    "target_preservation": target,
                    "class": class_name,
                    "bucket_type": "layer",
                    "bucket": "never" if layer == -1 else layer,
                    "count": count,
                    "fraction_of_class": count / total,
                }
            )
        for name, mask in (
            ("early", np.logical_and(selected >= 0, selected <= 8)),
            ("middle", np.logical_and(selected >= 9, selected <= 18)),
            ("late", np.logical_and(selected >= 19, selected <= 27)),
            ("never", selected == -1),
        ):
            count = int(mask.sum())
            output.append(
                {
                    "split": split,
                    "variant": variant,
                    "window": window_name,
                    "target_preservation": target,
                    "class": class_name,
                    "bucket_type": "region",
                    "bucket": name,
                    "count": count,
                    "fraction_of_class": count / total,
                }
            )
    return output


def _score_space_table(
    rows: Sequence[Mapping[str, Any]], *, split: str, variant: str
) -> list[dict[str, Any]]:
    scores, labels = _arrays(rows)
    return [
        {"split": split, "variant": variant, **row}
        for row in score_space_rows(scores, labels)
    ]


def _validation_architecture_summary(
    rows: Sequence[Mapping[str, Any]], variant: str
) -> dict[str, Any]:
    scores, labels = _arrays(rows)
    repeated = np.repeat(labels, 28)
    pooled = binary_metrics(repeated, scores.reshape(-1))
    layers = layerwise_ranking_metrics(scores, labels)
    return {
        "variant": variant,
        "pooled_auroc": float(pooled["auroc"]),
        "pooled_auprc": float(pooled["auprc"]),
        "mean_layer_auroc": float(np.mean([row["auroc"] for row in layers])),
        "mean_layer_auprc": float(np.mean([row["auprc"] for row in layers])),
        "minimum_layer_auroc": float(min(row["auroc"] for row in layers)),
        "maximum_layer_auroc": float(max(row["auroc"] for row in layers)),
    }


def freeze_validation(config_path: Path) -> None:
    contract, output_root, _, phase48_root = load_contract(config_path)
    selection_path = output_root / "evaluation/selected_operating_points.json"
    if selection_path.exists():
        raise RuntimeError("validation operating points are already frozen")
    validation = {}
    checkpoint_hashes = {}
    ranking_rows = []
    score_space = []
    architecture = []
    for variant in VARIANTS:
        _load_checkpoint(contract, output_root, variant)
        rows = _load_validation_scores(contract, output_root, phase48_root, variant)
        validation[variant] = rows
        checkpoint_hashes[variant] = file_sha256(
            output_root / f"training/checkpoints/{variant}.pt"
        )
        ranking_rows.extend(_ranking_rows(rows, split="validation", variant=variant))
        score_space.extend(_score_space_table(rows, split="validation", variant=variant))
        architecture.append(_validation_architecture_summary(rows, variant))
    ranking_rows.extend(_independent_ranking_rows(contract, split="validation"))

    windows = {
        name: [int(layer) for layer in layers]
        for name, layers in contract["evaluation"]["gate_windows"].items()
    }
    targets = [float(value) for value in contract["evaluation"]["preservation_targets"]]
    sweep_rows = []
    dataset_rows = []
    trigger_rows = []
    fixed_rows = []
    all_points: dict[str, Any] = {}
    selected_windows = {}
    fixed_points: dict[str, Any] = {}
    for variant in MAIN_VARIANTS:
        rows = validation[variant]
        scores, labels = _arrays(rows)
        all_points[variant] = {}
        for window_name, window in windows.items():
            sweep = threshold_sweep(scores, labels, window=window)
            sweep_rows.extend(
                {
                    "variant": variant,
                    "window": window_name,
                    **_flat_gate_metrics(row),
                }
                for row in sweep
            )
            points = {}
            for target in targets:
                point = trajectory_operating_point(
                    scores, labels, target_preservation=target, window=window
                )
                points[str(int(round(target * 100)))] = point
                first = first_trigger_layers(
                    scores, threshold=float(point["threshold"]), window=window
                )
                dataset_rows.extend(
                    _dataset_gate_rows(
                        rows,
                        scores,
                        labels,
                        split="validation",
                        variant=variant,
                        window_name=window_name,
                        window=window,
                        target=target,
                        threshold=float(point["threshold"]),
                    )
                )
                trigger_rows.extend(
                    _trigger_distribution_rows(
                        labels,
                        first,
                        split="validation",
                        variant=variant,
                        window_name=window_name,
                        target=target,
                    )
                )
            all_points[variant][window_name] = points
        candidates = []
        for window_name in windows:
            point = all_points[variant][window_name]["99"]
            candidates.append({"name": window_name, **point})
        selected_windows[variant] = select_candidate(
            candidates, preferred="all_0_27"
        )["name"]

        fixed_points[variant] = {}
        for layer in (14, 21, 27):
            fixed_points[variant][str(layer)] = {}
            for target in targets:
                point = trajectory_operating_point(
                    scores,
                    labels,
                    target_preservation=target,
                    window=[layer],
                )
                fixed_points[variant][str(layer)][str(int(round(target * 100)))] = point
                fixed_rows.append(
                    _fixed_result_row(
                        source="shared_predictor",
                        split="validation",
                        variant=variant,
                        layer=layer,
                        target=target,
                        threshold=float(point["threshold"]),
                        metrics=point,
                    )
                )

    scheme_candidates = []
    for variant in MAIN_VARIANTS:
        window_name = selected_windows[variant]
        point = all_points[variant][window_name]["99"]
        scheme_candidates.append(
            {
                "name": variant,
                **{key: value for key, value in point.items() if key != "window"},
                "window_name": window_name,
            }
        )
    primary = select_candidate(
        scheme_candidates, preferred="state_layer_all28"
    )

    phase50_fixed = read_csv(resolve_path(contract["sources"]["phase50_single_layer"]))
    for row in phase50_fixed:
        if not str(row["gate"]).startswith("layer_"):
            continue
        fixed_rows.append(
            _fixed_result_row(
                source="phase50_independent",
                split=str(row["split"]),
                variant="independent_linear",
                layer=int(str(row["gate"]).split("_")[-1]),
                target=float(row["target_preservation"]),
                threshold=float(row["threshold"]),
                metrics=row,
            )
        )
    independent_rows = []
    for source_name in ("phase50_validation_results", "phase50_test_results"):
        for row in read_csv(resolve_path(contract["sources"][source_name])):
            independent_rows.append({"source_file": source_name, **row})

    atomic_csv(output_root / "evaluation/layerwise_metrics.csv", ranking_rows)
    atomic_csv(output_root / "evaluation/global_threshold_sweep.csv", sweep_rows)
    atomic_csv(output_root / "evaluation/dataset_breakdown.csv", dataset_rows)
    atomic_csv(output_root / "evaluation/trigger_layer_distribution.csv", trigger_rows)
    atomic_csv(output_root / "evaluation/calibration_analysis.csv", score_space)
    atomic_csv(output_root / "baselines/fixed_layer_comparison.csv", fixed_rows)
    atomic_csv(
        output_root / "baselines/independent_gate_comparison.csv", independent_rows
    )
    atomic_json(
        output_root / "evaluation/architecture_validation.json",
        {
            "contract_sha256": contract["contract_sha256"],
            "models": architecture,
        },
    )
    selection = {
        "schema_version": "shared_stage1_selected_operating_points_v1",
        "contract_sha256": contract["contract_sha256"],
        "test_evaluated": False,
        "checkpoint_sha256": checkpoint_hashes,
        "validation_scores_sha256": {
            variant: file_sha256(
                output_root / f"evaluation/validation_scores/{variant}.jsonl"
            )
            for variant in VARIANTS
        },
        "selection_rules": {
            "checkpoint": contract["training"]["checkpoint_selection"],
            "window": contract["evaluation"]["window_selection"],
            "scheme": contract["evaluation"]["primary_scheme_selection"],
            "threshold": "most_permissive_tie_safe_complete_validation_trajectory_threshold_under_strict_greater_than",
        },
        "windows": windows,
        "operating_points": all_points,
        "selected_window_by_variant": selected_windows,
        "fixed_layer_points": fixed_points,
        "primary_variant": primary["name"],
        "primary_window": primary["window_name"],
        "validation_primary_99": {
            key: value
            for key, value in primary.items()
            if key not in {"thresholds"}
        },
    }
    write_once_or_verify(
        selection_path,
        (json.dumps(selection, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "passed": True,
                "primary_variant": selection["primary_variant"],
                "primary_window": selection["primary_window"],
                "validation_primary_99": selection["validation_primary_99"],
                "test_evaluated": False,
            }
        )
    )


def test_worker(config_path: Path, *, rank: int, world_size: int) -> None:
    contract, output_root, phase48, phase48_root = load_contract(config_path)
    device = _require_gpu(rank, world_size)
    if (output_root / "evaluation/test_completion.json").exists():
        raise RuntimeError("shared-gate test evaluation is already finalized")
    selection_path = output_root / "evaluation/selected_operating_points.json"
    selection = read_json(selection_path)
    if (
        selection.get("contract_sha256") != contract["contract_sha256"]
        or selection.get("test_evaluated") is not False
    ):
        raise RuntimeError("test scoring requires compatible pre-test selection")
    assignments = {
        0: ("state_layer_all28", 0),
        1: ("state_layer_random4", 0),
        2: ("state_layer_all28", 1),
        3: ("state_layer_random4", 1),
    }
    variant, shard = assignments[rank]
    result_path = output_root / f"evaluation/test_workers/rank{rank:02d}.jsonl"
    complete_path = output_root / f"evaluation/test_workers/rank{rank:02d}.complete.json"
    selection_hash = file_sha256(selection_path)
    if complete_path.exists():
        complete = read_json(complete_path)
        if (
            complete.get("contract_sha256") != contract["contract_sha256"]
            or complete.get("selection_sha256") != selection_hash
            or complete.get("result_sha256") != file_sha256(result_path)
        ):
            raise RuntimeError(f"existing test rank {rank} is incompatible")
        print(json.dumps({"passed": True, "rank": rank, "resumed": True}))
        return

    rows, features = _feature_tensor(phase48, phase48_root, splits={"test"})
    if len(rows) != 800:
        raise RuntimeError("test feature load does not cover 800 frozen UIDs")
    selected_indices = torch.arange(shard, len(rows), 2, dtype=torch.long)
    selected_rows = [rows[int(index)] for index in selected_indices.tolist()]
    selected_features = features.index_select(0, selected_indices)
    del features
    checkpoint = _load_checkpoint(contract, output_root, variant)
    if selection["checkpoint_sha256"][variant] != file_sha256(
        output_root / f"training/checkpoints/{variant}.pt"
    ):
        raise RuntimeError("selected checkpoint hash differs before test")
    model = _build_model(contract, variant).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model_state_dict"])
    mean_cpu, std_cpu = _normalization(output_root)
    mean = mean_cpu.to(device)
    std = std_cpu.to(device)
    _, scores = _score_full28(
        model,
        selected_features,
        records=len(selected_rows),
        mean=mean,
        std=std,
        device=device,
        batch_size=int(contract["training"]["batch_size_samples"]),
    )
    score_rows = _score_rows(selected_rows, scores, variant=variant)
    atomic_jsonl(result_path, score_rows)
    atomic_json(
        complete_path,
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "selection_sha256": selection_hash,
            "rank": rank,
            "world_size": world_size,
            "variant": variant,
            "shard": shard,
            "records": len(score_rows),
            "result_sha256": file_sha256(result_path),
        },
    )
    print(
        json.dumps(
            {"passed": True, "rank": rank, "variant": variant, "records": len(score_rows)}
        )
    )


def _aggregate_test(
    contract: Mapping[str, Any], output_root: Path, phase48_root: Path
) -> dict[str, list[dict[str, Any]]]:
    selection_path = output_root / "evaluation/selected_operating_points.json"
    selection_hash = file_sha256(selection_path)
    combined: dict[str, list[dict[str, Any]]] = {variant: [] for variant in MAIN_VARIANTS}
    for rank in range(4):
        complete_path = output_root / f"evaluation/test_workers/rank{rank:02d}.complete.json"
        result_path = output_root / f"evaluation/test_workers/rank{rank:02d}.jsonl"
        complete = read_json(complete_path)
        if (
            complete.get("passed") is not True
            or complete.get("contract_sha256") != contract["contract_sha256"]
            or complete.get("selection_sha256") != selection_hash
            or complete.get("result_sha256") != file_sha256(result_path)
            or int(complete.get("records", -1)) != 400
        ):
            raise RuntimeError(f"test worker {rank} is incomplete or incompatible")
        combined[str(complete["variant"])].extend(read_jsonl(result_path))
    expected = _load_split_rows(phase48_root, {"test"})
    expected_uids = [str(row["uid"]) for row in expected]
    for variant, rows in combined.items():
        rows.sort(key=lambda row: str(row["uid"]))
        if (
            len(rows) != 800
            or len({row["uid"] for row in rows}) != 800
            or [row["uid"] for row in rows] != expected_uids
            or any(row["variant"] != variant for row in rows)
        ):
            raise RuntimeError(f"{variant} test aggregation is incomplete")
        _arrays(rows)
    return combined


def _gate_result_row(
    *,
    source: str,
    variant: str,
    window: str,
    target: float,
    threshold: float | str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source": source,
        "variant": variant,
        "window": window,
        "target_preservation": float(target),
        "threshold": threshold,
        "records": int(metrics["records"]),
        "correct": int(metrics["correct"]),
        "wrong": int(metrics["wrong"]),
        "correct_preservation": float(metrics["correct_preservation"]),
        "wrong_detection_recall": float(metrics["wrong_detection_recall"]),
        "failure_precision": float(metrics["failure_precision"]),
        "trigger_rate": float(metrics["trigger_rate"]),
        "median_first_trigger_layer": metrics["median_first_trigger_layer"],
        "mean_first_trigger_layer": metrics["mean_first_trigger_layer"],
        "no_trigger_fraction": float(metrics["no_trigger_fraction"]),
        "correct_false_triggers": int(metrics["correct_false_triggers"]),
        "wrong_detected": int(metrics["wrong_detected"]),
    }


def _dataset_spreads(
    dataset_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    variant: str,
    window: str,
    target: float,
) -> tuple[float, float]:
    selected = [
        row
        for row in dataset_rows
        if row["split"] == split
        and row["variant"] == variant
        and row["window"] == window
        and math.isclose(float(row["target_preservation"]), target)
    ]
    if len(selected) != 3:
        raise RuntimeError("dataset spread requires exactly three dataset rows")
    preservation = [float(row["correct_preservation"]) for row in selected]
    recall = [float(row["wrong_detection_recall"]) for row in selected]
    return max(preservation) - min(preservation), max(recall) - min(recall)


def _plots(
    output_root: Path,
    *,
    selection: Mapping[str, Any],
    layerwise: Sequence[Mapping[str, Any]],
    test_results: Sequence[Mapping[str, Any]],
    dataset_rows: Sequence[Mapping[str, Any]],
    trigger_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(9, 5))
    for variant, label in (
        ("independent_linear", "independent linear"),
        ("state_layer_all28", "shared All-28"),
        ("state_layer_random4", "shared Random-4"),
    ):
        rows = sorted(
            [
                row
                for row in layerwise
                if row["split"] == "test"
                and row["variant"] == variant
                and row["dataset"] == "overall"
            ],
            key=lambda row: int(row["layer"]),
        )
        axis.plot(
            [int(row["layer"]) for row in rows],
            [float(row["auroc"]) for row in rows],
            marker="o",
            markersize=3,
            label=label,
        )
    axis.set(xlabel="Layer", ylabel="Test AUROC", ylim=(0.45, 1.0))
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_root / "shared_vs_independent_layerwise_auroc.png", dpi=160)
    plt.close(fig)

    sweeps = read_csv(output_root / "evaluation/global_threshold_sweep.csv")
    fig, axis = plt.subplots(figsize=(8, 6))
    for variant in MAIN_VARIANTS:
        window = selection["selected_window_by_variant"][variant]
        rows = [row for row in sweeps if row["variant"] == variant and row["window"] == window]
        axis.plot(
            [float(row["correct_preservation"]) for row in rows],
            [float(row["wrong_detection_recall"]) for row in rows],
            label=f"{variant} ({window})",
        )
    axis.set(
        xlabel="Validation sample-level correct preservation",
        ylabel="Validation wrong detection recall",
        xlim=(0.7, 1.005),
        ylim=(0.0, 1.0),
    )
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_root / "global_preservation_vs_wrong_detection.png", dpi=160)
    plt.close(fig)

    primary_variant = str(selection["primary_variant"])
    primary_window = str(selection["primary_window"])
    rows = [
        row
        for row in trigger_rows
        if row["split"] == "test"
        and row["variant"] == primary_variant
        and row["window"] == primary_window
        and math.isclose(float(row["target_preservation"]), 0.99)
        and row["bucket_type"] == "layer"
        and row["bucket"] != "never"
    ]
    fig, axis = plt.subplots(figsize=(10, 5))
    width = 0.4
    layers = np.arange(28)
    for offset, class_name in ((-width / 2, "correct"), (width / 2, "wrong")):
        counts = {
            int(row["bucket"]): int(row["count"])
            for row in rows
            if row["class"] == class_name
        }
        axis.bar(
            layers + offset,
            [counts.get(layer, 0) for layer in layers],
            width=width,
            label=class_name,
        )
    axis.set(xlabel="First trigger layer", ylabel="Test samples")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_root / "shared_trigger_layer_distribution.png", dpi=160)
    plt.close(fig)

    shared = [
        row
        for row in dataset_rows
        if row["split"] == "test"
        and row["variant"] == primary_variant
        and row["window"] == primary_window
        and math.isclose(float(row["target_preservation"]), 0.99)
    ]
    independent = [
        row
        for row in read_csv(resolve_path(contract["sources"]["phase50_dataset_breakdown"]))
        if row["split"] == "test" and math.isclose(float(row["target_preservation"]), 0.99)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    positions = np.arange(3)
    width = 0.35
    for axis, metric, title in (
        (axes[0], "correct_preservation", "Correct preservation"),
        (axes[1], "wrong_detection_recall", "Wrong recall"),
    ):
        axis.bar(
            positions - width / 2,
            [float(next(row for row in independent if row["dataset"] == dataset)[metric]) for dataset in DATASETS],
            width,
            label="independent sequential",
        )
        axis.bar(
            positions + width / 2,
            [float(next(row for row in shared if row["dataset"] == dataset)[metric]) for dataset in DATASETS],
            width,
            label="shared global",
        )
        axis.set_xticks(positions, DATASETS)
        axis.set_title(title)
        axis.set_ylim(0, 1.0)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figure_root / "dataset_calibration_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    targets = (0.99, 0.98, 0.95)
    for axis, metric, title in (
        (axes[0], "correct_preservation", "Test correct preservation"),
        (axes[1], "wrong_detection_recall", "Test wrong recall"),
    ):
        for offset, variant in ((-width / 2, "state_layer_all28"), (width / 2, "state_layer_random4")):
            selected = [
                next(
                    row
                    for row in test_results
                    if row["source"] == "shared_global"
                    and row["variant"] == variant
                    and math.isclose(float(row["target_preservation"]), target)
                )
                for target in targets
            ]
            axis.bar(
                np.arange(3) + offset,
                [float(row[metric]) for row in selected],
                width,
                label=variant.replace("state_layer_", ""),
            )
        axis.set_xticks(np.arange(3), ["99%", "98%", "95%"])
        axis.set_title(title)
        axis.set_ylim(0, 1.0)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figure_root / "all28_vs_random4.png", dpi=160)
    plt.close(fig)


def _decision_summary(
    contract: Mapping[str, Any],
    output_root: Path,
    *,
    selection: Mapping[str, Any],
    test_results: Sequence[Mapping[str, Any]],
    dataset_rows: Sequence[Mapping[str, Any]],
    layerwise: Sequence[Mapping[str, Any]],
    fixed_rows: Sequence[Mapping[str, Any]],
) -> str:
    primary_variant = str(selection["primary_variant"])
    primary_window = str(selection["primary_window"])
    shared_rows = {
        int(round(float(row["target_preservation"]) * 100)): row
        for row in test_results
        if row["source"] == "shared_global" and row["variant"] == primary_variant
    }
    independent_rows = {
        int(round(float(row["target_preservation"]) * 100)): row
        for row in test_results
        if row["source"] == "phase50_independent_sequential"
    }
    validation99 = selection["operating_points"][primary_variant][primary_window]["99"]
    test99 = shared_rows[99]
    preservation_drift = abs(
        float(validation99["correct_preservation"])
        - float(test99["correct_preservation"])
    )
    preservation_spread, recall_spread = _dataset_spreads(
        dataset_rows,
        split="test",
        variant=primary_variant,
        window=primary_window,
        target=0.99,
    )
    phase50_preservation_spread, phase50_recall_spread = _dataset_spreads(
        [
            {
                "split": row["split"],
                "variant": "phase50",
                "window": "independent",
                **row,
            }
            for row in read_csv(resolve_path(contract["sources"]["phase50_dataset_breakdown"]))
        ],
        split="test",
        variant="phase50",
        window="independent",
        target=0.99,
    )
    readiness = contract["evaluation"]["readiness_99"]
    ready_checks = {
        "test_preservation": float(test99["correct_preservation"])
        >= float(readiness["minimum_test_correct_preservation"]),
        "wrong_recall": float(test99["wrong_detection_recall"])
        >= float(independent_rows[99]["wrong_detection_recall"])
        + float(readiness["minimum_wrong_recall_relative_to_phase50"]),
        "preservation_drift": preservation_drift
        <= float(readiness["maximum_validation_test_preservation_drift"]),
        "dataset_preservation_spread": preservation_spread < phase50_preservation_spread,
    }
    ready = all(ready_checks.values())

    def mean_test_auroc(variant: str) -> float:
        values = [
            float(row["auroc"])
            for row in layerwise
            if row["split"] == "test"
            and row["variant"] == variant
            and row["dataset"] == "overall"
        ]
        return float(np.mean(values))

    primary_fixed_validation = [
        row
        for row in fixed_rows
        if row["source"] == "shared_predictor"
        and row["split"] == "validation"
        and row["variant"] == primary_variant
        and math.isclose(float(row["target_preservation"]), 0.99)
    ]
    selected_fixed = max(
        primary_fixed_validation,
        key=lambda row: (float(row["wrong_detection_recall"]), -int(row["layer"])),
    )
    fixed_test = next(
        row
        for row in fixed_rows
        if row["source"] == "shared_predictor"
        and row["split"] == "test"
        and row["variant"] == primary_variant
        and int(row["layer"]) == int(selected_fixed["layer"])
        and math.isclose(float(row["target_preservation"]), 0.99)
    )
    architecture = read_json(output_root / "evaluation/architecture_validation.json")["models"]
    architecture_by_name = {row["variant"]: row for row in architecture}
    lines = [
        "# Shared Stage-1 Global Risk Gate Decision Summary",
        "",
        f"Frozen contract: `{contract['contract_sha256']}`.",
        f"Validation-designated primary: `{primary_variant}` with `{primary_window}`. Test results did not change this selection.",
        "",
        "| Gate | Target | Test preservation | Wrong recall | Precision | Median trigger |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target in (99, 98, 95):
        for label, row in (
            ("Independent sequential", independent_rows[target]),
            ("Shared All-28", next(item for item in test_results if item["source"] == "shared_global" and item["variant"] == "state_layer_all28" and int(round(float(item["target_preservation"]) * 100)) == target)),
            ("Shared Random-4", next(item for item in test_results if item["source"] == "shared_global" and item["variant"] == "state_layer_random4" and int(round(float(item["target_preservation"]) * 100)) == target)),
        ):
            lines.append(
                f"| {label} | {target}% | {float(row['correct_preservation']):.4f} | {float(row['wrong_detection_recall']):.4f} | {float(row['failure_precision']):.4f} | {row['median_first_trigger_layer']} |"
            )
    lines.extend(
        [
            "",
            "## Q1. Can the shared predictor match the independent probes?",
            "",
            f"Mean test layer AUROC is `{mean_test_auroc(primary_variant):.4f}` for the primary shared predictor versus `{mean_test_auroc('independent_linear'):.4f}` for the 28 independent probes. Validation state+layer mean layer AUROC is `{float(architecture_by_name[primary_variant]['mean_layer_auroc']):.4f}`, state-only is `{float(architecture_by_name['state_only']['mean_layer_auroc']):.4f}`, and layer-only is `{float(architecture_by_name['layer_only']['mean_layer_auroc']):.4f}`.",
            "",
            "## Q2. All-28 versus Random-4",
            "",
            f"Validation selected `{primary_variant}` under the frozen 99%-risk rule. Both schemes were scored once on test as predeclared confirmatory arms; the selection was not switched post hoc.",
            "",
            "## Q3. Global threshold stability",
            "",
            f"At the primary 99% point, validation/test preservation is `{float(validation99['correct_preservation']):.4f}` / `{float(test99['correct_preservation']):.4f}` and validation/test wrong recall is `{float(validation99['wrong_detection_recall']):.4f}` / `{float(test99['wrong_detection_recall']):.4f}`.",
            "",
            "## Q4. Validation-to-test preservation drift",
            "",
            f"Primary absolute preservation drift at 99% is `{preservation_drift:.4f}`, versus Phase-50 drift `{abs(1.0 - float(independent_rows[99]['correct_preservation'])):.4f}`.",
            "",
            "## Q5. Dataset calibration spread",
            "",
            f"Primary shared 99% test preservation/wrong-recall spreads are `{preservation_spread:.4f}` / `{recall_spread:.4f}`; Phase-50 values are `{phase50_preservation_spread:.4f}` / `{phase50_recall_spread:.4f}`.",
            "",
            "## Q6. Sequential versus a shared fixed strong layer",
            "",
            f"Validation selected shared fixed layer `{int(selected_fixed['layer'])}` among L14/L21/L27. On test its preservation/recall is `{float(fixed_test['correct_preservation']):.4f}` / `{float(fixed_test['wrong_detection_recall']):.4f}`, versus shared sequential `{float(test99['correct_preservation']):.4f}` / `{float(test99['wrong_detection_recall']):.4f}`.",
            "",
            "## Q7. Gate window",
            "",
            f"Validation selected `{selection['selected_window_by_variant']['state_layer_all28']}` for All-28 and `{selection['selected_window_by_variant']['state_layer_random4']}` for Random-4. The late window was a validation sensitivity only and was not separately opened on test.",
            "",
            "## Q8. Ready for Stage-1 admission?",
            "",
            f"`{'YES' if ready else 'NO'}` under the frozen 99%-risk readiness checks: "
            + ", ".join(f"{name}={'pass' if value else 'fail'}" for name, value in ready_checks.items())
            + ".",
            "",
            "## Scope limits",
            "",
            "- Test was scored in one aggregate pass after all checkpoints, windows, and thresholds were frozen.",
            "- State-only and layer-only remained validation controls and received no new test scores.",
            "- No OOD follow-up, treatment, W-to-C repair, four-action routing, MCTS, or external evaluation ran.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(config_path: Path) -> None:
    contract, output_root, _, phase48_root = load_contract(config_path)
    completion_path = output_root / "evaluation/test_completion.json"
    if completion_path.exists():
        raise RuntimeError("shared-gate test evaluation is already finalized")
    selection_path = output_root / "evaluation/selected_operating_points.json"
    selection = read_json(selection_path)
    if selection.get("test_evaluated") is not False:
        raise RuntimeError("selected operating points are not a pre-test freeze")
    test = _aggregate_test(contract, output_root, phase48_root)
    combined_test_rows = [row for variant in MAIN_VARIANTS for row in test[variant]]
    atomic_jsonl(output_root / "evaluation/test_scores.jsonl", combined_test_rows)

    layerwise = read_csv(output_root / "evaluation/layerwise_metrics.csv")
    calibration = read_csv(output_root / "evaluation/calibration_analysis.csv")
    dataset_rows = read_csv(output_root / "evaluation/dataset_breakdown.csv")
    trigger_rows = read_csv(output_root / "evaluation/trigger_layer_distribution.csv")
    fixed_rows = read_csv(output_root / "baselines/fixed_layer_comparison.csv")
    layerwise.extend(_independent_ranking_rows(contract, split="test"))
    targets = [float(value) for value in contract["evaluation"]["preservation_targets"]]
    test_results = []
    for variant in MAIN_VARIANTS:
        rows = test[variant]
        scores, labels = _arrays(rows)
        layerwise.extend(_ranking_rows(rows, split="test", variant=variant))
        calibration.extend(_score_space_table(rows, split="test", variant=variant))
        window_name = str(selection["selected_window_by_variant"][variant])
        window = selection["windows"][window_name]
        for target in targets:
            key = str(int(round(target * 100)))
            point = selection["operating_points"][variant][window_name][key]
            threshold = float(point["threshold"])
            first = first_trigger_layers(scores, threshold=threshold, window=window)
            metrics = gate_metrics(labels, first)
            test_results.append(
                _gate_result_row(
                    source="shared_global",
                    variant=variant,
                    window=window_name,
                    target=target,
                    threshold=threshold,
                    metrics=metrics,
                )
            )
            dataset_rows.extend(
                _dataset_gate_rows(
                    rows,
                    scores,
                    labels,
                    split="test",
                    variant=variant,
                    window_name=window_name,
                    window=window,
                    target=target,
                    threshold=threshold,
                )
            )
            trigger_rows.extend(
                _trigger_distribution_rows(
                    labels,
                    first,
                    split="test",
                    variant=variant,
                    window_name=window_name,
                    target=target,
                )
            )
        for layer in (14, 21, 27):
            for target in targets:
                key = str(int(round(target * 100)))
                point = selection["fixed_layer_points"][variant][str(layer)][key]
                threshold = float(point["threshold"])
                first = first_trigger_layers(scores, threshold=threshold, window=[layer])
                metrics = gate_metrics(labels, first)
                fixed_rows.append(
                    _fixed_result_row(
                        source="shared_predictor",
                        split="test",
                        variant=variant,
                        layer=layer,
                        target=target,
                        threshold=threshold,
                        metrics=metrics,
                    )
                )

    for row in read_csv(resolve_path(contract["sources"]["phase50_test_results"])):
        test_results.append(
            _gate_result_row(
                source="phase50_independent_sequential",
                variant="independent_linear",
                window="layer_specific_0_27",
                target=float(row["target_preservation"]),
                threshold="layer_specific",
                metrics=row,
            )
        )
    phase50_fixed = [
        row
        for row in fixed_rows
        if row["source"] == "phase50_independent" and row["split"] == "test"
    ]
    for target in targets:
        validation_candidates = [
            row
            for row in fixed_rows
            if row["source"] == "phase50_independent"
            and row["split"] == "validation"
            and math.isclose(float(row["target_preservation"]), target)
        ]
        selected_layer = int(
            max(
                validation_candidates,
                key=lambda row: (
                    float(row["wrong_detection_recall"]),
                    -int(row["layer"]),
                ),
            )["layer"]
        )
        row = next(
            item
            for item in phase50_fixed
            if int(item["layer"]) == selected_layer
            and math.isclose(float(item["target_preservation"]), target)
        )
        test_results.append(
            _gate_result_row(
                source="phase50_validation_selected_fixed",
                variant="independent_linear",
                window=f"layer_{selected_layer}",
                target=target,
                threshold=float(row["threshold"]),
                metrics=row,
            )
        )

    atomic_csv(output_root / "evaluation/layerwise_metrics.csv", layerwise)
    atomic_csv(output_root / "evaluation/calibration_analysis.csv", calibration)
    atomic_csv(output_root / "evaluation/dataset_breakdown.csv", dataset_rows)
    atomic_csv(output_root / "evaluation/trigger_layer_distribution.csv", trigger_rows)
    atomic_csv(output_root / "baselines/fixed_layer_comparison.csv", fixed_rows)
    atomic_csv(output_root / "evaluation/test_results.csv", test_results)
    _plots(
        output_root,
        selection=selection,
        layerwise=layerwise,
        test_results=test_results,
        dataset_rows=dataset_rows,
        trigger_rows=trigger_rows,
        contract=contract,
    )
    summary = _decision_summary(
        contract,
        output_root,
        selection=selection,
        test_results=test_results,
        dataset_rows=dataset_rows,
        layerwise=layerwise,
        fixed_rows=fixed_rows,
    )
    atomic_json(
        completion_path,
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "selection_sha256": file_sha256(selection_path),
            "test_scores_sha256": file_sha256(output_root / "evaluation/test_scores.jsonl"),
            "test_variants": list(MAIN_VARIANTS),
            "records_per_variant": 800,
            "test_evaluated_once_after_validation_freeze": True,
        },
    )
    atomic_json(
        output_root / "evaluation/test_evaluation_audit.json",
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "test_rows": len(combined_test_rows),
            "unique_variant_uid_pairs": len(
                {(row["variant"], row["uid"]) for row in combined_test_rows}
            ),
            "models": list(MAIN_VARIANTS),
            "state_and_layer_controls_tested": False,
            "selection_unchanged_after_test": True,
        },
    )
    atomic_json(output_root / "evaluation/final_selection_snapshot.json", selection)
    _atomic_bytes(output_root / "decision_summary.md", summary.encode("utf-8"))

    required = [
        "protocol.md",
        "frozen_protocol.json",
        "preparation_audit.json",
        *[f"configs/{variant}.json" for variant in VARIANTS],
        *[f"training/histories/{variant}.json" for variant in VARIANTS],
        *[f"training/checkpoints/{variant}.pt" for variant in VARIANTS],
        "evaluation/layerwise_metrics.csv",
        "evaluation/global_threshold_sweep.csv",
        "evaluation/selected_operating_points.json",
        "evaluation/test_results.csv",
        "evaluation/test_scores.jsonl",
        "evaluation/dataset_breakdown.csv",
        "evaluation/trigger_layer_distribution.csv",
        "evaluation/calibration_analysis.csv",
        "evaluation/test_completion.json",
        "evaluation/test_evaluation_audit.json",
        "baselines/fixed_layer_comparison.csv",
        "baselines/independent_gate_comparison.csv",
        "figures/shared_vs_independent_layerwise_auroc.png",
        "figures/global_preservation_vs_wrong_detection.png",
        "figures/shared_trigger_layer_distribution.png",
        "figures/dataset_calibration_comparison.png",
        "figures/all28_vs_random4.png",
        "decision_summary.md",
    ]
    hashes = {relative: file_sha256(output_root / relative) for relative in required}
    atomic_json(
        output_root / "artifact_manifest.json",
        {
            "schema_version": "shared_stage1_global_risk_gate_artifact_manifest_v1",
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "required_files": hashes,
            "required_file_count": len(hashes),
            "test_evaluated_once_after_validation_freeze": True,
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "contract_sha256": contract["contract_sha256"],
                "primary_variant": selection["primary_variant"],
                "primary_window": selection["primary_window"],
                "required_files": len(hashes),
            }
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("prepare", "train-worker", "freeze-validation", "test-worker", "finalize")
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if args.command == "prepare":
        prepare(config_path)
    elif args.command == "train-worker":
        if args.rank is None:
            raise SystemExit("train-worker requires --rank")
        train_worker(config_path, rank=args.rank, world_size=args.world_size)
    elif args.command == "freeze-validation":
        freeze_validation(config_path)
    elif args.command == "test-worker":
        if args.rank is None:
            raise SystemExit("test-worker requires --rank")
        test_worker(config_path, rank=args.rank, world_size=args.world_size)
    else:
        finalize(config_path)


if __name__ == "__main__":
    main()
