#!/usr/bin/env python3
"""Run mixed-source Stage-1 training and fixed dataset-LODO evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dense_failure_stage1.all_source_robustness import (  # noqa: E402
    bootstrap_auc_interval,
    choose_robustness_decision,
    four_cell_epoch_indices,
    source_stratified_validation_uids,
    validate_lodo_training_rows,
)
from dense_failure_stage1.layerwise_probe import binary_metrics  # noqa: E402
from dense_failure_stage1.shared_global_gate import (  # noqa: E402
    SharedFailurePredictor,
    deterministic_random_layers,
)
from experiments.analyze_layerwise_dense_failure_predictability import (  # noqa: E402
    load_frozen_contract as load_historical_feature_contract,
    load_layer_matrices,
)


OUTPUT_ROOT = PROJECT_ROOT / "analysis/dense_failure_stage1/all_source_robustness"
DATASETS = ("gqa", "chartqa", "textvqa")
LAYERS = tuple(range(28))
BLOCKS = ("text_final", "text_mean", "visual_mean")
SEED = 20260903
MODEL_SEED = 20260831
FOLDS = 5
INNER_VALIDATION_FRACTION = 0.125
BOOTSTRAP_DRAWS = 5000
PHASE49_REFERENCES = {"chartqa": 0.4502, "gqa": 0.6160, "textvqa": 0.7236}

SOURCES = {
    "plan": "plans/stage1_all_source_mixed_training_and_ood_plan_v2.md",
    "phase61_protocol": "analysis/dense_failure_stage1/canonical_refit_diagnostic/frozen_protocol.json",
    "phase61_artifact_manifest": "analysis/dense_failure_stage1/canonical_refit_diagnostic/artifact_manifest.json",
    "canonical_fold_manifest": "analysis/dense_failure_stage1/canonical_refit_diagnostic/splits/fold_manifest.jsonl",
    "canonical_feature_index": "analysis/dense_failure_stage2/data_scale_search/dense/features/feature_index.jsonl",
    "historical_feature_config": "configs/layerwise_dense_failure_probe_v1.json",
    "historical_feature_protocol": "analysis/dense_failure_stage1/layerwise_failure_probe/probe_config.json",
    "historical_split": "analysis/dense_failure_stage1/layerwise_failure_probe/split_manifest.jsonl",
    "historical_feature_manifest": "analysis/dense_failure_stage1/layerwise_failure_probe/feature_shard_manifest.json",
    "old_normalization": "analysis/dense_failure_stage1/shared_global_gate/training/global_normalization.pt",
    "historical_specialist_val": "analysis/dense_failure_stage1/trigger_map/manifests/trigger_map_val.jsonl",
    "historical_specialist_test": "analysis/dense_failure_stage1/trigger_map/manifests/trigger_map_test.jsonl",
    "historical_specialist_canonical": "analysis/dense_failure_stage1/canonical_refit_diagnostic/predictions/frozen_old_head_canonical_scores.jsonl",
    "canonical_specialist_canonical": "analysis/dense_failure_stage1/canonical_refit_diagnostic/predictions/canonical_oof_scores.jsonl",
    "canonical_specialist_historical": "analysis/dense_failure_stage1/canonical_refit_diagnostic/predictions/full_canonical_head_historical_scores.jsonl",
    "phase49_decision": "analysis/dense_failure_stage1/ood_signal_diagnostic/decision_summary.md",
}
BOUND_CODE = (
    "experiments/run_stage1_all_source_robustness.py",
    "dense_failure_stage1/all_source_robustness.py",
    "dense_failure_stage1/shared_global_gate.py",
    "dense_failure_stage1/layerwise_probe.py",
    "experiments/analyze_layerwise_dense_failure_predictability.py",
)
ARCHITECTURE = {
    "input_size": 10752,
    "projection_size": 256,
    "layer_embedding_size": 32,
    "hidden_size": 256,
    "activation": "GELU",
    "dropout": 0.0,
}
TRAINING = {
    "epochs": 10,
    "optimizer": "AdamW",
    "learning_rate": 5e-4,
    "weight_decay": 0.01,
    "scheduler": "CosineAnnealingLR_per_epoch",
    "batch_size_samples": 32,
    "random_layers_per_sample": 4,
    "checkpoint_selection": "minimum_full28_internal_validation_bce_then_earliest_epoch",
    "compute_dtype": "torch.float32",
    "allow_tf32": False,
    "deterministic_algorithms": True,
    "sampler": "exact equal Historical-C/Historical-W/Canonical-C/Canonical-W epoch quotas",
    "epoch_records": "smallest multiple of four not below training-partition records",
    "within_cell": "uniform; no dataset balancing; replacement only when cell smaller than quota",
}


def resolve(relative: str | Path) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    external = Path("/mnt/hyemin").resolve()
    if not (path.is_relative_to(PROJECT_ROOT) or path.is_relative_to(external)):
        raise ValueError(f"path escapes allowed roots: {relative}")
    return path


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{number}")
            rows.append(value)
    return rows


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    _atomic_bytes(path, payload.encode())


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode())


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def command_output(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result.stdout.strip()


def _source_paths() -> dict[str, Path]:
    paths = {name: resolve(value) for name, value in SOURCES.items()}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing sources: {missing}")
    return paths


def _combined_manifest_rows(paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    historical = read_jsonl(paths["historical_split"])
    canonical = read_jsonl(paths["canonical_fold_manifest"])
    if len(historical) != 7999 or len(canonical) != 4000:
        raise RuntimeError("historical/canonical population count differs")
    rows = []
    for row in historical:
        rows.append(
            {
                "schema_version": "stage1_all_source_fold_manifest_v1",
                "uid": str(row["uid"]),
                "source_regime": "historical",
                "dataset": str(row["dataset"]),
                "current_dense_correct": bool(row["current_dense_correct"]),
                "current_dense_wrong": bool(row["current_dense_wrong"]),
                "image_group_id": str(row["image_group_id"]),
                "image_content_sha256": str(row["image_content_sha256"]),
                "historical_split": str(row["split"]),
                "canonical_fold": None,
            }
        )
    for row in canonical:
        rows.append(
            {
                "schema_version": "stage1_all_source_fold_manifest_v1",
                "uid": str(row["uid"]),
                "source_regime": "canonical",
                "dataset": str(row["dataset"]),
                "current_dense_correct": bool(row["current_dense_correct"]),
                "current_dense_wrong": bool(row["current_dense_wrong"]),
                "image_group_id": str(row["image_group_id"]),
                "image_content_sha256": str(row["image_content_sha256"]),
                "historical_split": None,
                "canonical_fold": int(row["fold"]),
            }
        )
    if len({row["uid"] for row in rows}) != len(rows):
        raise RuntimeError("combined manifest has duplicate UIDs")
    historical_groups = {
        row["image_group_id"] for row in rows if row["source_regime"] == "historical"
    }
    canonical_groups = {
        row["image_group_id"] for row in rows if row["source_regime"] == "canonical"
    }
    historical_hashes = {
        row["image_content_sha256"]
        for row in rows
        if row["source_regime"] == "historical"
    }
    canonical_hashes = {
        row["image_content_sha256"]
        for row in rows
        if row["source_regime"] == "canonical"
    }
    if historical_groups & canonical_groups:
        raise RuntimeError("historical/canonical image groups overlap")
    if historical_hashes & canonical_hashes:
        raise RuntimeError("historical/canonical image SHA values overlap")
    return sorted(rows, key=lambda row: (str(row["source_regime"]), str(row["uid"])))


def _fold_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for fold in range(FOLDS):
        training = [
            row
            for row in rows
            if (row["source_regime"] == "historical" and row["historical_split"] == "train")
            or (row["source_regime"] == "canonical" and row["canonical_fold"] != fold)
        ]
        canonical_eval = [
            row
            for row in rows
            if row["source_regime"] == "canonical" and row["canonical_fold"] == fold
        ]
        entry: dict[str, Any] = {
            "fold": fold,
            "training_pool_records_before_internal_validation": len(training),
            "canonical_oof_records": len(canonical_eval),
            "historical_heldout_records": sum(
                row["source_regime"] == "historical"
                and row["historical_split"] in {"val", "test"}
                for row in rows
            ),
        }
        cells = Counter(
            (str(row["source_regime"]), bool(row["current_dense_wrong"]))
            for row in training
        )
        for source in ("historical", "canonical"):
            for wrong, outcome in ((False, "correct"), (True, "wrong")):
                entry[f"train_pool_{source}_{outcome}"] = cells[(source, wrong)]
        output.append(entry)
    return output


def prepare() -> None:
    paths = _source_paths()
    phase61 = read_json(paths["phase61_artifact_manifest"])
    if phase61.get("passed") is not True or phase61.get("canonical_oof_records") != 4000:
        raise RuntimeError("Phase-61 artifact manifest is not complete")
    rows = _combined_manifest_rows(paths)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fold_path = OUTPUT_ROOT / "main_all/folds/fold_manifest.jsonl"
    atomic_jsonl(fold_path, rows)
    atomic_csv(OUTPUT_ROOT / "main_all/folds/fold_summary.csv", _fold_summary(rows))
    source_hashes = {name: file_sha256(path) for name, path in paths.items()}
    code_hashes = {name: file_sha256(resolve(name)) for name in BOUND_CODE}
    phase61_contract = read_json(paths["phase61_protocol"])
    canonical_shards = dict(
        phase61_contract["provenance"]["canonical_feature_shard_sha256"]
    )
    historical_feature_manifest = read_json(paths["historical_feature_manifest"])
    historical_shards = {
        str(row["path"]): str(row["sha256"])
        for row in historical_feature_manifest["shards"]
    }
    if source_hashes["old_normalization"] != "ce4b63ab7f503d879db20f94aae01116804095eb87fd03b16a818d5904f1aed3":
        raise RuntimeError("old normalization differs from the frozen artifact")
    status = command_output(["git", "status", "--porcelain=v1", "--untracked-files=all"])
    contract: dict[str, Any] = {
        "schema_version": "stage1_all_source_robustness_frozen_protocol_v1",
        "run_id": "stage1_all_source_robustness_v2",
        "objective": "one mixed-source Shared Random-4 boundary plus three target-blind dataset-LODO evaluations",
        "seed": SEED,
        "model_seed": MODEL_SEED,
        "input": {
            "blocks": list(BLOCKS),
            "layers": 28,
            "input_size": 10752,
            "normalization": "exact frozen historical global normalization",
            "normalization_sha256": source_hashes["old_normalization"],
            "prohibited_model_inputs": ["source_id", "dataset_id", "visual_token_count", "metadata"],
        },
        "architecture": ARCHITECTURE,
        "training": TRAINING,
        "population": {
            "historical_records": 7999,
            "historical_train": 6399,
            "historical_validation": 800,
            "historical_test": 800,
            "canonical_records": 4000,
            "canonical_folds": FOLDS,
            "internal_validation_fraction": INNER_VALIDATION_FRACTION,
            "main_fold_training": "historical train plus four canonical folds",
            "main_fold_evaluation": "one canonical OOF fold plus untouched historical validation and test",
        },
        "evaluation": {
            "positive_class": "current_dense_wrong",
            "trajectory_score": "maximum probability across layers 0-27",
            "historical_ensemble": "arithmetic mean of five fold probabilities at each layer, then maximum",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "phase49_references": PHASE49_REFERENCES,
            "decision_rule": {
                "source_robust": "both main Historical and Canonical max-score AUROC >= 0.70",
                "broad_ood": "every target worst-source AUROC >= 0.60 and mean Historical LODO AUROC >= mean Phase-49 reference",
                "decision_a": "source_robust and broad_ood",
                "decision_b": "source_robust and not broad_ood",
                "decision_c": "not source_robust",
            },
            "no_threshold_calibration": True,
        },
        "lodo": {
            "targets": list(DATASETS),
            "training": "historical train and all canonical rows from the two non-target datasets",
            "evaluation": "all Historical and all Canonical rows from the completely excluded target dataset",
            "target_data_prohibited_from": ["training", "internal_validation", "checkpoint_selection", "layer_selection"],
        },
        "sources": SOURCES,
        "provenance": {
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "git_branch": command_output(["git", "branch", "--show-current"]),
            "git_status_porcelain_at_freeze": status.splitlines() if status else [],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "source_sha256": source_hashes,
            "bound_code_sha256": code_hashes,
            "main_fold_manifest_sha256": file_sha256(fold_path),
            "canonical_feature_shard_sha256": canonical_shards,
            "historical_feature_shard_sha256": historical_shards,
        },
        "prohibited_actions": [
            "threshold calibration",
            "trigger-map regeneration",
            "Stage-2 label changes",
            "corrective search",
            "Stage-2 training",
            "external evaluation",
        ],
    }
    contract["contract_sha256"] = canonical_hash(contract)
    frozen = OUTPUT_ROOT / "frozen_protocol.json"
    if frozen.exists():
        if read_json(frozen).get("contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError("refusing to overwrite a different frozen protocol")
    else:
        atomic_json(frozen, contract)
    protocol = f"""# Stage-1 ALL-Source Robustness Protocol

- Frozen contract: `{contract['contract_sha256']}`
- Exact historical Shared Random-4 feature/head architecture and old normalization.
- Main fold training: Historical train plus four Canonical folds; evaluation: the fifth Canonical fold and untouched Historical validation/test.
- Every training epoch has exact 25% quotas for Historical-C, Historical-W, Canonical-C, and Canonical-W. Sampling is uniform within each cell, preserving its dataset mixture in expectation; no dataset balancing is used.
- Internal validation is selected only from each run's allowed training pool and stratified by source, dataset, and current dense correctness.
- Historical primary scores average the five main-fold probabilities per layer before computing the max trajectory score.
- Each LODO head excludes its target dataset from training, internal validation, checkpoint selection, and layer selection.
- Source robustness requires both aggregate source AUROCs >= 0.70. Broad OOD requires every target's worst-source AUROC >= 0.60 and non-inferior mean Historical LODO AUROC versus the Phase-49 references.
- Ranking only: no threshold calibration, trigger map, Stage-2 change, search, or external evaluation.
"""
    _atomic_bytes(OUTPUT_ROOT / "protocol.md", protocol.encode())
    print(json.dumps({"passed": True, "contract_sha256": contract["contract_sha256"]}))


def load_contract() -> dict[str, Any]:
    contract = read_json(OUTPUT_ROOT / "frozen_protocol.json")
    if (
        contract.get("schema_version") != "stage1_all_source_robustness_frozen_protocol_v1"
        or contract.get("contract_sha256") != canonical_hash(contract)
    ):
        raise RuntimeError("ALL-source frozen protocol is invalid")
    provenance = contract["provenance"]
    if command_output(["git", "rev-parse", "HEAD"]) != provenance["git_commit"]:
        raise RuntimeError("git commit differs from frozen protocol")
    for name, expected in provenance["source_sha256"].items():
        if file_sha256(resolve(contract["sources"][name])) != expected:
            raise RuntimeError(f"frozen source differs: {name}")
    for name, expected in provenance["bound_code_sha256"].items():
        if file_sha256(resolve(name)) != expected:
            raise RuntimeError(f"bound code differs: {name}")
    fold_path = OUTPUT_ROOT / "main_all/folds/fold_manifest.jsonl"
    if file_sha256(fold_path) != provenance["main_fold_manifest_sha256"]:
        raise RuntimeError("main fold manifest differs")
    if file_sha256(resolve(SOURCES["old_normalization"])) != contract["input"]["normalization_sha256"]:
        raise RuntimeError("old normalization differs")
    return contract


def _gpu() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker requires exactly one GPU via CUDA_VISIBLE_DEVICES")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return device


def _model() -> SharedFailurePredictor:
    return SharedFailurePredictor(
        variant="state_layer_random4",
        input_size=ARCHITECTURE["input_size"],
        projection_size=ARCHITECTURE["projection_size"],
        layer_embedding_size=ARCHITECTURE["layer_embedding_size"],
        hidden_size=ARCHITECTURE["hidden_size"],
    )


def _normalization() -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.load(resolve(SOURCES["old_normalization"]), map_location="cpu", weights_only=True)
    if (
        value.get("schema_version") != "shared_stage1_global_normalization_v1"
        or tuple(value["mean"].shape) != (10752,)
        or tuple(value["std"].shape) != (10752,)
    ):
        raise RuntimeError("old normalization schema differs")
    return value["mean"].float(), value["std"].float()


def _load_canonical_features(
    rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> torch.Tensor:
    phase_rows = {
        str(row["uid"]): row
        for row in read_jsonl(resolve(SOURCES["canonical_fold_manifest"]))
    }
    selected = []
    for row in rows:
        source = phase_rows[str(row["uid"])]
        selected.append(
            {
                **row,
                "feature_shard": str(source["feature_shard"]),
                "feature_row_index": int(source["feature_row_index"]),
            }
        )
    target = torch.empty((len(selected), 28, 10752), dtype=torch.bfloat16)
    filled = torch.zeros(len(selected), dtype=torch.bool)
    by_shard: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for destination, row in enumerate(selected):
        by_shard[str(row["feature_shard"])].append((destination, row))
    frozen = contract["provenance"]["canonical_feature_shard_sha256"]
    for name in sorted(by_shard):
        path = resolve(name)
        if file_sha256(path) != frozen.get(name):
            raise RuntimeError(f"canonical feature shard hash differs: {name}")
        shard = torch.load(path, map_location="cpu", weights_only=True)
        if shard.get("layer_ids") != list(LAYERS):
            raise RuntimeError(f"canonical feature layers differ: {name}")
        matrix = torch.cat([shard[block] for block in BLOCKS], dim=-1)
        if tuple(matrix.shape[1:]) != (28, 10752):
            raise RuntimeError(f"canonical feature shape differs: {name}")
        for destination, row in by_shard[name]:
            source_index = int(row["feature_row_index"])
            if str(shard["uids"][source_index]) != str(row["uid"]):
                raise RuntimeError(f"canonical feature UID differs: {row['uid']}")
            target[destination].copy_(matrix[source_index])
            filled[destination] = True
    if not bool(filled.all()) or not bool(torch.isfinite(target.float()).all()):
        raise RuntimeError("canonical feature load is incomplete or non-finite")
    return target


def _load_historical_features(
    contract: Mapping[str, Any], selected_splits: set[str]
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    feature_contract, root = load_historical_feature_contract(
        resolve(SOURCES["historical_feature_config"])
    )
    rows, matrices = load_layer_matrices(
        feature_contract, root, layers=LAYERS, selected_splits=selected_splits
    )
    features = torch.stack([matrices.pop(layer) for layer in LAYERS], dim=1)
    for row in rows:
        row["source_regime"] = "historical"
        row["historical_split"] = str(row["split"])
        row["canonical_fold"] = None
    expected_shards = contract["provenance"]["historical_feature_shard_sha256"]
    manifest = read_json(resolve(SOURCES["historical_feature_manifest"]))
    observed = {str(row["path"]): str(row["sha256"]) for row in manifest["shards"]}
    if observed != expected_shards:
        raise RuntimeError("historical feature shard manifest differs")
    return rows, features


def _load_all_features(
    contract: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    historical_rows, historical_features = _load_historical_features(
        contract, {"train", "val", "test"}
    )
    canonical_rows = read_jsonl(resolve(SOURCES["canonical_fold_manifest"]))
    for row in canonical_rows:
        row["source_regime"] = "canonical"
        row["historical_split"] = None
        row["canonical_fold"] = int(row["fold"])
    canonical_features = _load_canonical_features(canonical_rows, contract)
    rows = historical_rows + canonical_rows
    features = torch.cat((historical_features, canonical_features), dim=0)
    if features.shape != (11999, 28, 10752):
        raise RuntimeError("combined feature shape differs")
    if len({str(row["uid"]) for row in rows}) != 11999:
        raise RuntimeError("combined feature rows have duplicate UIDs")
    return rows, features


def _batch_logits(
    model: SharedFailurePredictor,
    features: torch.Tensor,
    sample_indices: torch.Tensor,
    layer_choices: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch, count = len(sample_indices), int(layer_choices.shape[1])
    selected = features.index_select(0, sample_indices)
    row_indices = torch.arange(batch, dtype=torch.long)[:, None]
    selected = selected[row_indices, layer_choices]
    states = selected.reshape(batch * count, -1).to(device=device, dtype=torch.float32)
    states = (states - mean) / std
    layers = layer_choices.reshape(-1).to(device=device, dtype=torch.long)
    return model(states, layers)


def _score_with_logits(
    model: SharedFailurePredictor,
    features: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = []
    all_layers = torch.arange(28, dtype=torch.long)[None, :]
    with torch.inference_mode():
        for start in range(0, len(features), TRAINING["batch_size_samples"]):
            stop = min(start + TRAINING["batch_size_samples"], len(features))
            indices = torch.arange(start, stop, dtype=torch.long)
            layers = all_layers.expand(stop - start, -1)
            values = _batch_logits(
                model,
                features,
                indices,
                layers,
                mean=mean,
                std=std,
                device=device,
            )
            logits.append(values.reshape(stop - start, 28).cpu())
    logit_values = torch.cat(logits).numpy().astype(np.float64)
    scores = np.empty_like(logit_values)
    positive = logit_values >= 0
    scores[positive] = 1.0 / (1.0 + np.exp(-logit_values[positive]))
    exponential = np.exp(logit_values[~positive])
    scores[~positive] = exponential / (1.0 + exponential)
    return logit_values, scores


def _score(
    model: SharedFailurePredictor,
    features: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    return _score_with_logits(
        model, features, mean=mean, std=std, device=device
    )[1]


def _validation_bce(logits: np.ndarray, labels: np.ndarray) -> float:
    if logits.ndim != 2 or logits.shape[0] != len(labels):
        raise ValueError("validation logits and labels are not aligned")
    flat = logits.reshape(-1)
    repeated = np.repeat(labels.astype(np.float64), logits.shape[1])
    return float(
        np.mean(
            np.maximum(flat, 0)
            - flat * repeated
            + np.log1p(np.exp(-np.abs(flat)))
        )
    )


def _train(
    rows: Sequence[Mapping[str, Any]],
    features: torch.Tensor,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int, float, int]:
    training_rows = [rows[index] for index in training_indices]
    epoch_records = 4 * math.ceil(len(training_rows) / 4)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _model().to(device=device, dtype=torch.float32)
    mean_cpu, std_cpu = _normalization()
    mean, std = mean_cpu.to(device), std_cpu.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAINING["epochs"]
    )
    labels = np.asarray(
        [int(bool(row["current_dense_wrong"])) for row in rows], dtype=np.int64
    )
    history = []
    best_state = None
    best_epoch = -1
    best_bce = float("inf")
    for epoch in range(TRAINING["epochs"]):
        model.train()
        selected_local = four_cell_epoch_indices(
            training_rows, records=epoch_records, seed=seed + 10000, epoch=epoch
        )
        selected = training_indices[selected_local]
        layer_choices = deterministic_random_layers(
            len(selected),
            epoch=epoch,
            seed=seed,
            count=TRAINING["random_layers_per_sample"],
        )
        accumulated = 0.0
        for start in range(0, len(selected), TRAINING["batch_size_samples"]):
            stop = min(start + TRAINING["batch_size_samples"], len(selected))
            indices = torch.as_tensor(selected[start:stop], dtype=torch.long)
            choices = layer_choices[start:stop]
            optimizer.zero_grad(set_to_none=True)
            logits = _batch_logits(
                model,
                features,
                indices,
                choices,
                mean=mean,
                std=std,
                device=device,
            )
            targets = torch.as_tensor(labels[selected[start:stop]], dtype=torch.float32)
            targets = targets.to(device).repeat_interleave(choices.shape[1])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()
            accumulated += float(loss.detach().cpu()) * (stop - start)
        validation_logits, validation_scores = _score_with_logits(
            model,
            features.index_select(0, torch.as_tensor(validation_indices)),
            mean=mean,
            std=std,
            device=device,
        )
        validation_labels = labels[validation_indices]
        bce = _validation_bce(validation_logits, validation_labels)
        ranking = binary_metrics(validation_labels, validation_scores.max(axis=1))
        cells = Counter(
            (
                str(rows[index]["source_regime"]),
                bool(rows[index]["current_dense_wrong"]),
            )
            for index in selected
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": accumulated / len(selected),
                "epoch_records": len(selected),
                "cell_counts": {f"{source}_{'wrong' if wrong else 'correct'}": cells[(source, wrong)] for source in ("historical", "canonical") for wrong in (False, True)},
                "validation_full28_bce": bce,
                "validation_max_auroc": float(ranking["auroc"]),
                "validation_max_auprc": float(ranking["auprc"]),
            }
        )
        if bce < best_bce - 1e-12:
            best_bce = bce
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        scheduler.step()
    if best_state is None or best_epoch < 0:
        raise RuntimeError("training did not select an internal-validation checkpoint")
    return best_state, history, best_epoch, best_bce, epoch_records


def _prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    *,
    model_id: str,
    fold: int | str,
) -> list[dict[str, Any]]:
    if scores.shape != (len(rows), 28):
        raise RuntimeError("prediction shape differs")
    output = []
    for row, values in zip(rows, scores):
        entry = {
            "schema_version": "stage1_all_source_score_v1",
            "uid": str(row["uid"]),
            "source_regime": str(row["source_regime"]),
            "dataset": str(row["dataset"]),
            "image_group_id": str(row["image_group_id"]),
            "current_dense_correct": not bool(row["current_dense_wrong"]),
            "current_dense_wrong": bool(row["current_dense_wrong"]),
            "model_id": model_id,
            "fold": fold,
            "score_max": float(values.max()),
        }
        if row.get("historical_split"):
            entry["historical_split"] = str(row["historical_split"])
        if row.get("canonical_fold") is not None:
            entry["canonical_fold"] = int(row["canonical_fold"])
        entry.update({f"p_{layer}": float(values[layer]) for layer in LAYERS})
        output.append(entry)
    return output


def _partition_indices(
    rows: Sequence[Mapping[str, Any]], base_indices: Sequence[int], *, run_id: str
) -> tuple[np.ndarray, np.ndarray]:
    base_rows = [rows[index] for index in base_indices]
    validation_uids = source_stratified_validation_uids(
        base_rows,
        fraction=INNER_VALIDATION_FRACTION,
        seed=SEED,
        run_id=run_id,
    )
    train = np.asarray(
        [index for index in base_indices if str(rows[index]["uid"]) not in validation_uids],
        dtype=np.int64,
    )
    validation = np.asarray(
        [index for index in base_indices if str(rows[index]["uid"]) in validation_uids],
        dtype=np.int64,
    )
    if set(train) & set(validation) or set(train) | set(validation) != set(base_indices):
        raise RuntimeError("training/internal-validation partition differs")
    train_groups = {str(rows[index]["image_group_id"]) for index in train}
    validation_groups = {str(rows[index]["image_group_id"]) for index in validation}
    if train_groups & validation_groups:
        raise RuntimeError("training/internal-validation image groups overlap")
    return train, validation


def _write_complete(directory: Path, contract: Mapping[str, Any], files: Mapping[str, str], extra: Mapping[str, Any]) -> None:
    complete: dict[str, Any] = {
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        **extra,
    }
    for key, name in files.items():
        complete[f"{key}_file"] = name
        complete[f"{key}_sha256"] = file_sha256(directory / name)
    atomic_json(directory / "complete.json", complete)


def _resume(directory: Path, contract: Mapping[str, Any]) -> bool:
    path = directory / "complete.json"
    if not path.exists():
        return False
    complete = read_json(path)
    if complete.get("passed") is not True or complete.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError(f"incompatible completion marker: {directory}")
    for key, value in complete.items():
        if key.endswith("_file"):
            stem = key[: -len("_file")]
            if file_sha256(directory / str(value)) != complete[f"{stem}_sha256"]:
                raise RuntimeError(f"resume artifact hash differs: {directory}/{value}")
        elif key.endswith("_predictions"):
            if file_sha256(resolve(str(value))) != complete[f"{key}_sha256"]:
                raise RuntimeError(f"resume prediction hash differs: {value}")
    return True


def train_main_fold(fold: int) -> None:
    if fold not in range(FOLDS):
        raise ValueError("main fold must lie in 0..4")
    contract = load_contract()
    directory = OUTPUT_ROOT / f"main_all/training/fold_{fold}"
    if _resume(directory, contract):
        print(json.dumps({"passed": True, "resumed": True, "main_fold": fold}))
        return
    device = _gpu()
    rows, features = _load_all_features(contract)
    base = [
        index
        for index, row in enumerate(rows)
        if (
            row["source_regime"] == "historical"
            and row["historical_split"] == "train"
        )
        or (
            row["source_regime"] == "canonical"
            and int(row["canonical_fold"]) != fold
        )
    ]
    canonical_eval = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["source_regime"] == "canonical"
            and int(row["canonical_fold"]) == fold
        ],
        dtype=np.int64,
    )
    historical_eval = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["source_regime"] == "historical"
            and row["historical_split"] in {"val", "test"}
        ],
        dtype=np.int64,
    )
    training, validation = _partition_indices(rows, base, run_id=f"main_fold_{fold}")
    eval_groups = {
        str(rows[index]["image_group_id"])
        for index in np.concatenate((canonical_eval, historical_eval))
    }
    if eval_groups & {
        str(rows[index]["image_group_id"])
        for index in np.concatenate((training, validation))
    }:
        raise RuntimeError("main fold has training/evaluation image-group leakage")
    seed = MODEL_SEED + 200 + fold
    best_state, history, best_epoch, best_bce, epoch_records = _train(
        rows,
        features,
        training,
        validation,
        seed=seed,
        device=device,
    )
    model = _model().to(device=device, dtype=torch.float32)
    model.load_state_dict(best_state)
    mean_cpu, std_cpu = _normalization()
    mean, std = mean_cpu.to(device), std_cpu.to(device)
    canonical_scores = _score(
        model,
        features.index_select(0, torch.as_tensor(canonical_eval)),
        mean=mean,
        std=std,
        device=device,
    )
    historical_scores = _score(
        model,
        features.index_select(0, torch.as_tensor(historical_eval)),
        mean=mean,
        std=std,
        device=device,
    )
    canonical_rows = [rows[index] for index in canonical_eval]
    historical_rows = [rows[index] for index in historical_eval]
    split_rows = [
        {"uid": str(rows[index]["uid"]), "role": role}
        for role, values in (
            ("train", training),
            ("internal_val", validation),
            ("canonical_oof", canonical_eval),
            ("historical_heldout", historical_eval),
        )
        for index in values
    ]
    split_rows.sort(key=lambda row: (row["role"], row["uid"]))
    atomic_jsonl(directory / "inner_split.jsonl", split_rows)
    atomic_jsonl(
        directory / "canonical_scores.jsonl",
        _prediction_rows(
            canonical_rows,
            canonical_scores,
            model_id=f"all_source_fold_{fold}",
            fold=fold,
        ),
    )
    atomic_jsonl(
        directory / "historical_scores.jsonl",
        _prediction_rows(
            historical_rows,
            historical_scores,
            model_id=f"all_source_fold_{fold}",
            fold=fold,
        ),
    )
    atomic_json(
        directory / "history.json",
        {
            "contract_sha256": contract["contract_sha256"],
            "run": f"main_fold_{fold}",
            "seed": seed,
            "best_epoch": best_epoch,
            "best_internal_validation_full28_bce": best_bce,
            "training_records": len(training),
            "internal_validation_records": len(validation),
            "canonical_oof_records": len(canonical_eval),
            "historical_heldout_records": len(historical_eval),
            "epoch_records": epoch_records,
            "history": history,
        },
    )
    atomic_torch(
        directory / "checkpoint.pt",
        {
            "schema_version": "stage1_all_source_checkpoint_v1",
            "contract_sha256": contract["contract_sha256"],
            "run": f"main_fold_{fold}",
            "seed": seed,
            "best_epoch": best_epoch,
            "best_internal_validation_full28_bce": best_bce,
            "old_normalization_sha256": contract["input"]["normalization_sha256"],
            "model_state_dict": best_state,
        },
    )
    _write_complete(
        directory,
        contract,
        {
            "checkpoint": "checkpoint.pt",
            "history": "history.json",
            "inner_split": "inner_split.jsonl",
            "canonical_scores": "canonical_scores.jsonl",
            "historical_scores": "historical_scores.jsonl",
        },
        {"main_fold": fold, "best_epoch": best_epoch},
    )
    print(
        json.dumps(
            {
                "passed": True,
                "main_fold": fold,
                "best_epoch": best_epoch,
                "canonical_records": len(canonical_eval),
                "historical_records": len(historical_eval),
            }
        )
    )


def train_lodo(target: str) -> None:
    if target not in DATASETS:
        raise ValueError(f"unsupported LODO target: {target}")
    contract = load_contract()
    directory = OUTPUT_ROOT / f"lodo/{target}/training"
    if _resume(directory, contract):
        print(json.dumps({"passed": True, "resumed": True, "lodo_target": target}))
        return
    device = _gpu()
    rows, features = _load_all_features(contract)
    base = [
        index
        for index, row in enumerate(rows)
        if row["dataset"] != target
        and (
            (
                row["source_regime"] == "historical"
                and row["historical_split"] == "train"
            )
            or row["source_regime"] == "canonical"
        )
    ]
    validate_lodo_training_rows([rows[index] for index in base], target_dataset=target)
    historical_eval = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["dataset"] == target
            and row["source_regime"] == "historical"
        ],
        dtype=np.int64,
    )
    canonical_eval = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row["dataset"] == target and row["source_regime"] == "canonical"
        ],
        dtype=np.int64,
    )
    training, validation = _partition_indices(rows, base, run_id=f"lodo_{target}")
    if any(rows[index]["dataset"] == target for index in np.concatenate((training, validation))):
        raise RuntimeError("target dataset entered LODO training or validation")
    training_groups = {
        str(rows[index]["image_group_id"])
        for index in np.concatenate((training, validation))
    }
    evaluation_groups = {
        str(rows[index]["image_group_id"])
        for index in np.concatenate((historical_eval, canonical_eval))
    }
    if training_groups & evaluation_groups:
        raise RuntimeError("LODO training/evaluation image groups overlap")
    seed = MODEL_SEED + 300 + DATASETS.index(target)
    best_state, history, best_epoch, best_bce, epoch_records = _train(
        rows,
        features,
        training,
        validation,
        seed=seed,
        device=device,
    )
    model = _model().to(device=device, dtype=torch.float32)
    model.load_state_dict(best_state)
    mean_cpu, std_cpu = _normalization()
    mean, std = mean_cpu.to(device), std_cpu.to(device)
    historical_scores = _score(
        model,
        features.index_select(0, torch.as_tensor(historical_eval)),
        mean=mean,
        std=std,
        device=device,
    )
    canonical_scores = _score(
        model,
        features.index_select(0, torch.as_tensor(canonical_eval)),
        mean=mean,
        std=std,
        device=device,
    )
    historical_rows = [rows[index] for index in historical_eval]
    canonical_rows = [rows[index] for index in canonical_eval]
    prediction_root = OUTPUT_ROOT / f"lodo/{target}/predictions"
    split_rows = [
        {"uid": str(rows[index]["uid"]), "role": role}
        for role, values in (
            ("train", training),
            ("internal_val", validation),
            ("historical_target", historical_eval),
            ("canonical_target", canonical_eval),
        )
        for index in values
    ]
    split_rows.sort(key=lambda row: (row["role"], row["uid"]))
    atomic_jsonl(directory / "inner_split.jsonl", split_rows)
    atomic_jsonl(
        prediction_root / "historical_target_scores.jsonl",
        _prediction_rows(
            historical_rows,
            historical_scores,
            model_id=f"all_source_lodo_{target}",
            fold="target_blind",
        ),
    )
    atomic_jsonl(
        prediction_root / "canonical_target_scores.jsonl",
        _prediction_rows(
            canonical_rows,
            canonical_scores,
            model_id=f"all_source_lodo_{target}",
            fold="target_blind",
        ),
    )
    atomic_json(
        directory / "history.json",
        {
            "contract_sha256": contract["contract_sha256"],
            "run": f"lodo_{target}",
            "target_dataset": target,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_internal_validation_full28_bce": best_bce,
            "training_records": len(training),
            "internal_validation_records": len(validation),
            "historical_target_records": len(historical_eval),
            "canonical_target_records": len(canonical_eval),
            "epoch_records": epoch_records,
            "history": history,
        },
    )
    atomic_torch(
        directory / "checkpoint.pt",
        {
            "schema_version": "stage1_all_source_checkpoint_v1",
            "contract_sha256": contract["contract_sha256"],
            "run": f"lodo_{target}",
            "target_dataset": target,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_internal_validation_full28_bce": best_bce,
            "old_normalization_sha256": contract["input"]["normalization_sha256"],
            "model_state_dict": best_state,
        },
    )
    protocol = f"""# ALL-Source LODO {target.upper()} Protocol

- Frozen parent contract: `{contract['contract_sha256']}`
- `{target}` is absent from training, internal validation, checkpoint selection, and layer selection.
- Training uses Historical train plus all Canonical rows from `{', '.join(dataset for dataset in DATASETS if dataset != target)}`.
- Every epoch uses exact 25% Historical-C/Historical-W/Canonical-C/Canonical-W quotas with natural within-cell dataset mixture.
- Evaluation uses every Historical and Canonical `{target}` row; all are target-blind because `{target}` is completely absent from fitting. No threshold is selected.
"""
    _atomic_bytes(OUTPUT_ROOT / f"lodo/{target}/protocol.md", protocol.encode())
    _write_complete(
        directory,
        contract,
        {
            "checkpoint": "checkpoint.pt",
            "history": "history.json",
            "inner_split": "inner_split.jsonl",
        },
        {
            "lodo_target": target,
            "best_epoch": best_epoch,
            "historical_predictions": str(
                (prediction_root / "historical_target_scores.jsonl").relative_to(PROJECT_ROOT)
            ),
            "historical_predictions_sha256": file_sha256(
                prediction_root / "historical_target_scores.jsonl"
            ),
            "canonical_predictions": str(
                (prediction_root / "canonical_target_scores.jsonl").relative_to(PROJECT_ROOT)
            ),
            "canonical_predictions_sha256": file_sha256(
                prediction_root / "canonical_target_scores.jsonl"
            ),
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "lodo_target": target,
                "best_epoch": best_epoch,
                "historical_records": len(historical_eval),
                "canonical_records": len(canonical_eval),
            }
        )
    )


def _score_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    matrix = np.asarray(
        [[float(row[f"p_{layer}"]) for layer in LAYERS] for row in rows],
        dtype=np.float64,
    )
    if matrix.shape != (len(rows), 28) or not np.isfinite(matrix).all():
        raise RuntimeError("score matrix is incomplete or non-finite")
    return matrix


def _ranking(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = np.asarray(
        [int(bool(row["current_dense_wrong"])) for row in rows], dtype=np.int64
    )
    scores = _score_matrix(rows).max(axis=1)
    return dict(binary_metrics(labels, scores))


def _ensemble_historical(
    fold_rows: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    if len(fold_rows) != FOLDS:
        raise RuntimeError("historical ensemble requires five fold predictions")
    maps = [{str(row["uid"]): row for row in rows} for rows in fold_rows]
    expected = set(maps[0])
    if len(expected) != 1600 or any(set(values) != expected for values in maps):
        raise RuntimeError("historical fold predictions are not aligned over 1,600 UIDs")
    output = []
    for uid in sorted(expected):
        references = [values[uid] for values in maps]
        signatures = {
            (
                str(row["dataset"]),
                bool(row["current_dense_wrong"]),
                str(row["image_group_id"]),
            )
            for row in references
        }
        if len(signatures) != 1:
            raise RuntimeError(f"historical ensemble metadata differs for {uid}")
        mean_scores = np.mean(
            [[float(row[f"p_{layer}"]) for layer in LAYERS] for row in references],
            axis=0,
        )
        reference = references[0]
        output.append(
            {
                "schema_version": "stage1_all_source_historical_ensemble_v1",
                "uid": uid,
                "source_regime": "historical",
                "dataset": str(reference["dataset"]),
                "image_group_id": str(reference["image_group_id"]),
                "historical_split": str(reference["historical_split"]),
                "current_dense_correct": bool(reference["current_dense_correct"]),
                "current_dense_wrong": bool(reference["current_dense_wrong"]),
                "model_id": "all_source_five_fold_probability_ensemble",
                "score_max": float(mean_scores.max()),
                **{f"p_{layer}": float(mean_scores[layer]) for layer in LAYERS},
            }
        )
    return output


def _old_historical_specialist_rows() -> list[dict[str, Any]]:
    output = []
    for split in ("val", "test"):
        for row in read_jsonl(resolve(SOURCES[f"historical_specialist_{split}"])):
            values = np.asarray(
                [float(row[f"score_l{layer}"]) for layer in LAYERS], dtype=np.float64
            )
            output.append(
                {
                    "uid": str(row["uid"]),
                    "source_regime": "historical",
                    "dataset": str(row["dataset"]),
                    "historical_split": split,
                    "image_group_id": str(row["image_group_id"]),
                    "current_dense_correct": bool(row["dense_correct"]),
                    "current_dense_wrong": bool(row["dense_wrong"]),
                    "score_max": float(values.max()),
                    **{f"p_{layer}": float(values[layer]) for layer in LAYERS},
                }
            )
    return sorted(output, key=lambda row: str(row["uid"]))


def _normalized_specialist_rows(path: Path, source: str) -> list[dict[str, Any]]:
    output = []
    for row in read_jsonl(path):
        value = dict(row)
        value["source_regime"] = source
        if "dense_wrong" in value and "current_dense_wrong" not in value:
            value["current_dense_wrong"] = bool(value["dense_wrong"])
            value["current_dense_correct"] = not bool(value["dense_wrong"])
        if "score_max" not in value:
            value["score_max"] = max(float(value[f"p_{layer}"]) for layer in LAYERS)
        output.append(value)
    return sorted(output, key=lambda row: str(row["uid"]))


def _metric_columns(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "records": int(metrics["records"]),
        "correct": int(metrics["correct"]),
        "wrong": int(metrics["wrong"]),
        "auroc": float(metrics["auroc"]),
        "auprc": float(metrics["auprc"]),
    }


def _source_summary(
    collections: Mapping[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = []
    robustness = []
    for head, (historical, canonical) in collections.items():
        historical_metrics = _ranking(historical)
        canonical_metrics = _ranking(canonical)
        summary.append(
            {
                "head": head,
                "historical_records": historical_metrics["records"],
                "historical_auroc": historical_metrics["auroc"],
                "historical_auprc": historical_metrics["auprc"],
                "canonical_records": canonical_metrics["records"],
                "canonical_auroc": canonical_metrics["auroc"],
                "canonical_auprc": canonical_metrics["auprc"],
            }
        )
        average = (float(historical_metrics["auroc"]) + float(canonical_metrics["auroc"])) / 2
        worst = min(float(historical_metrics["auroc"]), float(canonical_metrics["auroc"]))
        robustness.append(
            {
                "head": head,
                "average_source_auroc": average,
                "worst_source_auroc": worst,
                "historical_auroc": historical_metrics["auroc"],
                "canonical_auroc": canonical_metrics["auroc"],
            }
        )
    return summary, robustness


def _dataset_source_breakdown(
    historical: Sequence[Mapping[str, Any]], canonical: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    counter = 0
    for source, rows in (("historical", historical), ("canonical", canonical)):
        for dataset in DATASETS:
            selected = [row for row in rows if row["dataset"] == dataset]
            labels = np.asarray(
                [int(bool(row["current_dense_wrong"])) for row in selected]
            )
            scores = _score_matrix(selected).max(axis=1)
            metrics = binary_metrics(labels, scores)
            interval = bootstrap_auc_interval(
                labels,
                scores,
                draws=BOOTSTRAP_DRAWS,
                seed=SEED + 1000 + counter,
            )
            output.append(
                {
                    "source_regime": source,
                    "dataset": dataset,
                    **_metric_columns(metrics),
                    "auroc_ci_low": interval["ci_low"],
                    "auroc_ci_high": interval["ci_high"],
                    "bootstrap_draws": interval["draws"],
                }
            )
            counter += 1
    return output


def _main_layerwise(
    historical: Sequence[Mapping[str, Any]], canonical: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrices = {
        "historical": _score_matrix(historical),
        "canonical": _score_matrix(canonical),
    }
    labels = {
        "historical": np.asarray(
            [int(bool(row["current_dense_wrong"])) for row in historical]
        ),
        "canonical": np.asarray(
            [int(bool(row["current_dense_wrong"])) for row in canonical]
        ),
    }
    auroc_rows = []
    auprc_rows = []
    for layer in LAYERS:
        metrics = {
            source: binary_metrics(values, matrices[source][:, layer])
            for source, values in labels.items()
        }
        auroc_rows.append(
            {
                "layer": layer,
                "historical_auroc": metrics["historical"]["auroc"],
                "canonical_auroc": metrics["canonical"]["auroc"],
                "average_source_auroc": (
                    float(metrics["historical"]["auroc"])
                    + float(metrics["canonical"]["auroc"])
                )
                / 2,
                "worst_source_auroc": min(
                    float(metrics["historical"]["auroc"]),
                    float(metrics["canonical"]["auroc"]),
                ),
            }
        )
        auprc_rows.append(
            {
                "layer": layer,
                "historical_auprc": metrics["historical"]["auprc"],
                "canonical_auprc": metrics["canonical"]["auprc"],
                "average_source_auprc": (
                    float(metrics["historical"]["auprc"])
                    + float(metrics["canonical"]["auprc"])
                )
                / 2,
                "worst_source_auprc": min(
                    float(metrics["historical"]["auprc"]),
                    float(metrics["canonical"]["auprc"]),
                ),
            }
        )
    return auroc_rows, auprc_rows


def _source_conditioned_stats(
    heads: Mapping[str, tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]]
) -> list[dict[str, Any]]:
    output = []
    for head, (historical, canonical) in heads.items():
        for dataset in ("overall", *DATASETS):
            for wrong, outcome in ((False, "correct"), (True, "wrong")):
                source_values = {}
                for source, rows in (("historical", historical), ("canonical", canonical)):
                    selected = [
                        float(row["score_max"])
                        for row in rows
                        if bool(row["current_dense_wrong"]) == wrong
                        and (dataset == "overall" or row["dataset"] == dataset)
                    ]
                    values = np.asarray(selected, dtype=np.float64)
                    source_values[source] = values
                output.append(
                    {
                        "head": head,
                        "dataset": dataset,
                        "outcome": outcome,
                        "historical_records": len(source_values["historical"]),
                        "historical_mean": float(source_values["historical"].mean()),
                        "historical_median": float(np.median(source_values["historical"])),
                        "historical_p95": float(np.quantile(source_values["historical"], 0.95)),
                        "canonical_records": len(source_values["canonical"]),
                        "canonical_mean": float(source_values["canonical"].mean()),
                        "canonical_median": float(np.median(source_values["canonical"])),
                        "canonical_p95": float(np.quantile(source_values["canonical"], 0.95)),
                        "canonical_minus_historical_mean": float(
                            source_values["canonical"].mean()
                            - source_values["historical"].mean()
                        ),
                    }
                )
    return output


def _roc(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    true_positive = np.r_[0, np.cumsum(sorted_labels == 1)]
    false_positive = np.r_[0, np.cumsum(sorted_labels == 0)]
    return false_positive / false_positive[-1], true_positive / true_positive[-1]


def _main_figures(
    all_historical: Sequence[Mapping[str, Any]],
    all_canonical: Sequence[Mapping[str, Any]],
    source_summary: Sequence[Mapping[str, Any]],
    layerwise: Sequence[Mapping[str, Any]],
) -> None:
    root = OUTPUT_ROOT / "main_all/figures"
    root.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    for source, rows, color in (
        ("Historical", all_historical, "#4c72b0"),
        ("Canonical", all_canonical, "#dd8452"),
    ):
        labels = np.asarray([int(row["current_dense_wrong"]) for row in rows])
        scores = _score_matrix(rows).max(axis=1)
        fpr, tpr = _roc(labels, scores)
        auc = binary_metrics(labels, scores)["auroc"]
        plt.plot(fpr, tpr, label=f"{source} ({auc:.3f})", color=color)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "historical_vs_canonical_roc.png", dpi=180)
    plt.close()

    names = [str(row["head"]) for row in source_summary]
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width / 2, [row["historical_auroc"] for row in source_summary], width, label="Historical")
    plt.bar(x + width / 2, [row["canonical_auroc"] for row in source_summary], width, label="Canonical")
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.xticks(x, names, rotation=15, ha="right")
    plt.ylabel("Max-score AUROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "source_robustness_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot([row["layer"] for row in layerwise], [row["historical_auroc"] for row in layerwise], label="Historical")
    plt.plot([row["layer"] for row in layerwise], [row["canonical_auroc"] for row in layerwise], label="Canonical")
    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "layerwise_source_auroc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot([row["layer"] for row in layerwise], [row["worst_source_auroc"] for row in layerwise], color="#55a868")
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Layer")
    plt.ylabel("Worst-source AUROC")
    plt.tight_layout()
    plt.savefig(root / "worst_source_auroc_by_layer.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for source, rows, color in (
        ("Historical", all_historical, "#4c72b0"),
        ("Canonical", all_canonical, "#dd8452"),
    ):
        for wrong, linestyle, outcome in ((False, "-", "C"), (True, "--", "W")):
            values = [float(row["score_max"]) for row in rows if bool(row["current_dense_wrong"]) == wrong]
            plt.hist(values, bins=30, density=True, histtype="step", linewidth=1.7, color=color, linestyle=linestyle, label=f"{source}-{outcome}")
    plt.xlabel("Max layer risk score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "source_conditioned_score_distributions.png", dpi=180)
    plt.close()


def _aggregate_main(
    contract: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    canonical = []
    historical_folds = []
    checkpoints = []
    per_fold_metrics = []
    for fold in range(FOLDS):
        directory = OUTPUT_ROOT / f"main_all/training/fold_{fold}"
        if not _resume(directory, contract):
            raise RuntimeError(f"main fold {fold} is incomplete")
        complete = read_json(directory / "complete.json")
        fold_canonical = read_jsonl(directory / complete["canonical_scores_file"])
        fold_historical = read_jsonl(directory / complete["historical_scores_file"])
        if len(fold_canonical) != 800 or len(fold_historical) != 1600:
            raise RuntimeError(f"main fold {fold} prediction count differs")
        canonical.extend(fold_canonical)
        historical_folds.append(fold_historical)
        historical_metric = _ranking(fold_historical)
        per_fold_metrics.append(
            {
                "fold": fold,
                "historical_auroc": historical_metric["auroc"],
                "historical_auprc": historical_metric["auprc"],
            }
        )
        checkpoints.append(
            {
                "model_id": f"all_source_fold_{fold}",
                "path": str((directory / complete["checkpoint_file"]).relative_to(PROJECT_ROOT)),
                "sha256": complete["checkpoint_sha256"],
                "contract_sha256": contract["contract_sha256"],
            }
        )
    expected_canonical = {
        str(row["uid"])
        for row in read_jsonl(resolve(SOURCES["canonical_fold_manifest"]))
    }
    if (
        len(canonical) != 4000
        or len({str(row["uid"]) for row in canonical}) != 4000
        or {str(row["uid"]) for row in canonical} != expected_canonical
    ):
        raise RuntimeError("canonical OOF coverage differs")
    canonical.sort(key=lambda row: str(row["uid"]))
    historical = _ensemble_historical(historical_folds)
    atomic_jsonl(OUTPUT_ROOT / "main_all/predictions/canonical_oof_scores.jsonl", canonical)
    atomic_jsonl(OUTPUT_ROOT / "main_all/predictions/historical_ensemble_scores.jsonl", historical)
    atomic_json(
        OUTPUT_ROOT / "main_all/training/checkpoint_manifest.json",
        {"contract_sha256": contract["contract_sha256"], "checkpoints": checkpoints},
    )

    historical_specialist_historical = _old_historical_specialist_rows()
    historical_specialist_canonical = _normalized_specialist_rows(
        resolve(SOURCES["historical_specialist_canonical"]), "canonical"
    )
    canonical_specialist_historical = _normalized_specialist_rows(
        resolve(SOURCES["canonical_specialist_historical"]), "historical"
    )
    canonical_specialist_canonical = _normalized_specialist_rows(
        resolve(SOURCES["canonical_specialist_canonical"]), "canonical"
    )
    collections = {
        "historical_specialist": (
            historical_specialist_historical,
            historical_specialist_canonical,
        ),
        "canonical_specialist": (
            canonical_specialist_historical,
            canonical_specialist_canonical,
        ),
        "all_source": (historical, canonical),
    }
    source_summary, robustness = _source_summary(collections)
    fold_values = np.asarray([row["historical_auroc"] for row in per_fold_metrics])
    for row in robustness:
        row["historical_per_fold_auroc_mean"] = ""
        row["historical_per_fold_auroc_std"] = ""
    all_robustness = next(row for row in robustness if row["head"] == "all_source")
    all_robustness["historical_per_fold_auroc_mean"] = float(fold_values.mean())
    all_robustness["historical_per_fold_auroc_std"] = float(fold_values.std())
    atomic_csv(OUTPUT_ROOT / "main_all/metrics/source_summary.csv", source_summary)
    atomic_csv(OUTPUT_ROOT / "main_all/metrics/robustness_metrics.csv", robustness)
    atomic_csv(OUTPUT_ROOT / "main_all/metrics/historical_per_fold_metrics.csv", per_fold_metrics)
    breakdown = _dataset_source_breakdown(historical, canonical)
    atomic_csv(OUTPUT_ROOT / "main_all/metrics/dataset_source_breakdown.csv", breakdown)
    layerwise_auroc, layerwise_auprc = _main_layerwise(historical, canonical)
    atomic_csv(OUTPUT_ROOT / "main_all/metrics/layerwise_auroc.csv", layerwise_auroc)
    atomic_csv(OUTPUT_ROOT / "main_all/metrics/layerwise_auprc.csv", layerwise_auprc)
    score_stats = _source_conditioned_stats(
        {
            "historical_specialist": (
                historical_specialist_historical,
                historical_specialist_canonical,
            ),
            "all_source": (historical, canonical),
        }
    )
    atomic_csv(
        OUTPUT_ROOT / "main_all/metrics/source_conditioned_score_stats.csv",
        score_stats,
    )
    _main_figures(historical, canonical, source_summary, layerwise_auroc)
    return canonical, historical, source_summary, robustness, breakdown


def _aggregate_lodo(
    contract: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    summary = []
    comparison = []
    layerwise_by_target: dict[str, list[dict[str, Any]]] = {}
    for target_index, target in enumerate(DATASETS):
        directory = OUTPUT_ROOT / f"lodo/{target}/training"
        if not _resume(directory, contract):
            raise RuntimeError(f"LODO target is incomplete: {target}")
        complete = read_json(directory / "complete.json")
        historical_path = resolve(complete["historical_predictions"])
        canonical_path = resolve(complete["canonical_predictions"])
        if file_sha256(historical_path) != complete["historical_predictions_sha256"] or file_sha256(canonical_path) != complete["canonical_predictions_sha256"]:
            raise RuntimeError(f"LODO target prediction hash differs: {target}")
        historical = read_jsonl(historical_path)
        canonical = read_jsonl(canonical_path)
        if any(row["dataset"] != target for row in historical + canonical):
            raise RuntimeError(f"LODO target predictions contain another dataset: {target}")
        metric_rows = []
        source_metrics = {}
        for source_index, (source, rows) in enumerate(
            (("historical", historical), ("canonical", canonical))
        ):
            labels = np.asarray([int(row["current_dense_wrong"]) for row in rows])
            scores = _score_matrix(rows).max(axis=1)
            metrics = binary_metrics(labels, scores)
            interval = bootstrap_auc_interval(
                labels,
                scores,
                draws=BOOTSTRAP_DRAWS,
                seed=SEED + 2000 + 10 * target_index + source_index,
            )
            source_metrics[source] = metrics
            metric_rows.append(
                {
                    "target_dataset": target,
                    "source_regime": source,
                    **_metric_columns(metrics),
                    "auroc_ci_low": interval["ci_low"],
                    "auroc_ci_high": interval["ci_high"],
                    "bootstrap_draws": interval["draws"],
                }
            )
        average = (
            float(source_metrics["historical"]["auroc"])
            + float(source_metrics["canonical"]["auroc"])
        ) / 2
        worst = min(
            float(source_metrics["historical"]["auroc"]),
            float(source_metrics["canonical"]["auroc"]),
        )
        for row in metric_rows:
            row["average_target_source_auroc"] = average
            row["worst_target_source_auroc"] = worst
        atomic_csv(OUTPUT_ROOT / f"lodo/{target}/metrics.csv", metric_rows)

        historical_matrix = _score_matrix(historical)
        canonical_matrix = _score_matrix(canonical)
        historical_labels = np.asarray(
            [int(row["current_dense_wrong"]) for row in historical]
        )
        canonical_labels = np.asarray(
            [int(row["current_dense_wrong"]) for row in canonical]
        )
        layer_rows = []
        for layer in LAYERS:
            historical_metrics = binary_metrics(
                historical_labels, historical_matrix[:, layer]
            )
            canonical_metrics = binary_metrics(
                canonical_labels, canonical_matrix[:, layer]
            )
            layer_rows.append(
                {
                    "target_dataset": target,
                    "layer": layer,
                    "historical_auroc": historical_metrics["auroc"],
                    "canonical_auroc": canonical_metrics["auroc"],
                    "average_target_source_auroc": (
                        float(historical_metrics["auroc"])
                        + float(canonical_metrics["auroc"])
                    )
                    / 2,
                    "worst_target_source_auroc": min(
                        float(historical_metrics["auroc"]),
                        float(canonical_metrics["auroc"]),
                    ),
                }
            )
        layerwise_by_target[target] = layer_rows
        atomic_csv(OUTPUT_ROOT / f"lodo/{target}/layerwise_auroc.csv", layer_rows)
        train_datasets = "+".join(dataset for dataset in DATASETS if dataset != target)
        summary.append(
            {
                "train_datasets": train_datasets,
                "ood_target": target,
                "historical_target_records": source_metrics["historical"]["records"],
                "historical_target_auroc": source_metrics["historical"]["auroc"],
                "historical_target_auprc": source_metrics["historical"]["auprc"],
                "canonical_target_records": source_metrics["canonical"]["records"],
                "canonical_target_auroc": source_metrics["canonical"]["auroc"],
                "canonical_target_auprc": source_metrics["canonical"]["auprc"],
                "average_target_source_auroc": average,
                "worst_target_source_auroc": worst,
            }
        )
        comparison.append(
            {
                "target": target,
                "phase49_historical_only_auroc": PHASE49_REFERENCES[target],
                "all_source_lodo_historical_auroc": source_metrics["historical"]["auroc"],
                "all_source_lodo_canonical_auroc": source_metrics["canonical"]["auroc"],
                "historical_difference_vs_phase49": float(
                    source_metrics["historical"]["auroc"]
                )
                - PHASE49_REFERENCES[target],
                "protocol_qualification": "Phase49 used independent probes and source-selected single layers; this phase uses Shared Random-4 max trajectories",
            }
        )
    atomic_csv(OUTPUT_ROOT / "lodo/lodo_summary.csv", summary)
    atomic_csv(OUTPUT_ROOT / "lodo/phase49_comparison.csv", comparison)
    return summary, comparison, layerwise_by_target


def _lodo_figures(
    summary: Sequence[Mapping[str, Any]],
    comparison: Sequence[Mapping[str, Any]],
    layerwise: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    root = OUTPUT_ROOT / "lodo/figures"
    root.mkdir(parents=True, exist_ok=True)
    targets = [str(row["ood_target"]) for row in summary]
    x = np.arange(len(targets))
    width = 0.35
    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width / 2, [row["historical_target_auroc"] for row in summary], width, label="Historical")
    plt.bar(x + width / 2, [row["canonical_target_auroc"] for row in summary], width, label="Canonical")
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.xticks(x, targets)
    plt.ylabel("OOD max-score AUROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "lodo_auroc_matrix.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width, [row["phase49_historical_only_auroc"] for row in comparison], width, label="Phase49 historical")
    plt.bar(x, [row["all_source_lodo_historical_auroc"] for row in comparison], width, label="ALL LODO historical")
    plt.bar(x + width, [row["all_source_lodo_canonical_auroc"] for row in comparison], width, label="ALL LODO canonical")
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.xticks(x, [row["target"] for row in comparison])
    plt.ylabel("AUROC")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(root / "phase49_vs_all_source_lodo.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for target, rows in layerwise.items():
        plt.plot([row["layer"] for row in rows], [row["average_target_source_auroc"] for row in rows], label=target)
    plt.xlabel("Layer")
    plt.ylabel("Average target-source AUROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "lodo_layerwise_auroc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for target, rows in layerwise.items():
        plt.plot([row["layer"] for row in rows], [row["worst_target_source_auroc"] for row in rows], label=target)
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Layer")
    plt.ylabel("Worst target-source AUROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "lodo_worst_source_auroc.png", dpi=180)
    plt.close()


def _summaries(
    source_summary: Sequence[Mapping[str, Any]],
    robustness: Sequence[Mapping[str, Any]],
    breakdown: Sequence[Mapping[str, Any]],
    lodo_summary: Sequence[Mapping[str, Any]],
    comparison: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> None:
    sources = {str(row["head"]): row for row in source_summary}
    robust = {str(row["head"]): row for row in robustness}
    all_row = sources["all_source"]
    all_robust = robust["all_source"]
    cells = {
        (str(row["source_regime"]), str(row["dataset"])): row for row in breakdown
    }
    weakest_cell = min(breakdown, key=lambda row: float(row["auroc"]))
    layer_rows = read_jsonl(OUTPUT_ROOT / "main_all/predictions/canonical_oof_scores.jsonl")
    del layer_rows
    layerwise = []
    with (OUTPUT_ROOT / "main_all/metrics/layerwise_auroc.csv").open() as handle:
        layerwise = list(csv.DictReader(handle))
    best_layers = sorted(
        layerwise, key=lambda row: float(row["worst_source_auroc"]), reverse=True
    )[:5]
    score_stats = []
    with (OUTPUT_ROOT / "main_all/metrics/source_conditioned_score_stats.csv").open() as handle:
        score_stats = list(csv.DictReader(handle))
    canonical_correct_all = next(
        row
        for row in score_stats
        if row["head"] == "all_source"
        and row["dataset"] == "overall"
        and row["outcome"] == "correct"
    )
    canonical_correct_old = next(
        row
        for row in score_stats
        if row["head"] == "historical_specialist"
        and row["dataset"] == "overall"
        and row["outcome"] == "correct"
    )
    all_summary = f"""# ALL-Source Training Summary

The ALL-source five-fold head ensemble achieved Historical held-out AUROC **{all_row['historical_auroc']:.4f}** and Canonical OOF AUROC **{all_row['canonical_auroc']:.4f}**. Average-source AUROC is **{all_robust['average_source_auroc']:.4f}** and worst-source AUROC is **{all_robust['worst_source_auroc']:.4f}**.

| Head | Historical held-out AUROC | Canonical held-out AUROC | Average source | Worst source |
|---|---:|---:|---:|---:|
| Historical specialist | {sources['historical_specialist']['historical_auroc']:.4f} | {sources['historical_specialist']['canonical_auroc']:.4f} | {robust['historical_specialist']['average_source_auroc']:.4f} | {robust['historical_specialist']['worst_source_auroc']:.4f} |
| Canonical specialist | {sources['canonical_specialist']['historical_auroc']:.4f} | {sources['canonical_specialist']['canonical_auroc']:.4f} | {robust['canonical_specialist']['average_source_auroc']:.4f} | {robust['canonical_specialist']['worst_source_auroc']:.4f} |
| ALL-source | {all_row['historical_auroc']:.4f} | {all_row['canonical_auroc']:.4f} | {all_robust['average_source_auroc']:.4f} | {all_robust['worst_source_auroc']:.4f} |

The weakest dataset×source cell is **{weakest_cell['source_regime']} {weakest_cell['dataset']}** at AUROC {weakest_cell['auroc']:.4f} (C={weakest_cell['correct']}, W={weakest_cell['wrong']}). Canonical ChartQA AUROC is {cells[('canonical','chartqa')]['auroc']:.4f}, so it {'remains non-inverted' if float(cells[('canonical','chartqa')]['auroc']) > .5 else 'is inverted'}.

For canonical Dense-C, the historical-specialist mean/p95 maximum risk was {float(canonical_correct_old['canonical_mean']):.4f}/{float(canonical_correct_old['canonical_p95']):.4f}; under ALL-source training it is {float(canonical_correct_all['canonical_mean']):.4f}/{float(canonical_correct_all['canonical_p95']):.4f}. The five most source-robust layers by worst-source AUROC are {', '.join(f"L{row['layer']} ({float(row['worst_source_auroc']):.3f})" for row in best_layers)}.

Conclusion: one shared Stage-1 boundary {'is viable across both observed source regimes for the aggregate in-scope mixture' if decision['source_robust'] else 'does not maintain the frozen minimum ranking on both observed source regimes'}. No threshold was selected.
"""
    _atomic_bytes(
        OUTPUT_ROOT / "summaries/all_source_training_summary.md",
        all_summary.encode(),
    )

    lodo = {str(row["ood_target"]): row for row in lodo_summary}
    comparisons = {str(row["target"]): row for row in comparison}
    least = min(lodo_summary, key=lambda row: float(row["worst_target_source_auroc"]))
    ood_summary = f"""# Dataset-OOD Summary

| Train datasets | OOD target | Historical AUROC | Canonical AUROC | Average source | Worst source |
|---|---|---:|---:|---:|---:|
| GQA + TextVQA | ChartQA | {lodo['chartqa']['historical_target_auroc']:.4f} | {lodo['chartqa']['canonical_target_auroc']:.4f} | {lodo['chartqa']['average_target_source_auroc']:.4f} | {lodo['chartqa']['worst_target_source_auroc']:.4f} |
| GQA + ChartQA | TextVQA | {lodo['textvqa']['historical_target_auroc']:.4f} | {lodo['textvqa']['canonical_target_auroc']:.4f} | {lodo['textvqa']['average_target_source_auroc']:.4f} | {lodo['textvqa']['worst_target_source_auroc']:.4f} |
| ChartQA + TextVQA | GQA | {lodo['gqa']['historical_target_auroc']:.4f} | {lodo['gqa']['canonical_target_auroc']:.4f} | {lodo['gqa']['average_target_source_auroc']:.4f} | {lodo['gqa']['worst_target_source_auroc']:.4f} |

Against the historical-only Phase-49 references, Historical-target differences are ChartQA {comparisons['chartqa']['historical_difference_vs_phase49']:+.4f}, TextVQA {comparisons['textvqa']['historical_difference_vs_phase49']:+.4f}, and GQA {comparisons['gqa']['historical_difference_vs_phase49']:+.4f}. This comparison is qualified because Phase 49 selected single independent-probe layers, whereas this phase evaluates Shared Random-4 maximum trajectories.

The least transferable target is **{least['ood_target']}** by worst-source AUROC ({least['worst_target_source_auroc']:.4f}). Canonical TextVQA has only 19 wrong examples; its point estimate and bootstrap interval in `lodo/textvqa/metrics.csv` remain high-uncertainty.

Interpretation: Stage-1 is {'broadly benchmark-transferable under the frozen rule' if decision['broad_ood_transfer'] else 'not broadly benchmark-transferable under the frozen rule'}. Source diversity alone does not justify a universal failure-detector claim when any target/source cell remains weak.
"""
    _atomic_bytes(
        OUTPUT_ROOT / "summaries/dataset_ood_summary.md", ood_summary.encode()
    )

    if str(decision["decision"]).startswith("A"):
        consequence = "A separately authorized next phase may calibrate a robust threshold on the ALL mixture."
    elif str(decision["decision"]).startswith("B"):
        consequence = "A separately authorized next phase may calibrate an in-scope ALL-mixture threshold, while explicitly avoiding a benchmark-universal claim."
    else:
        consequence = "Stop before threshold calibration; a separately authorized diagnostic must inspect shared-head conflict, sampling, support, and score geometry."
    decision_summary = f"""# Stage-1 Robustness Decision

## {decision['decision']}

- ALL Historical held-out AUROC: {all_row['historical_auroc']:.4f}
- ALL Canonical OOF AUROC: {all_row['canonical_auroc']:.4f}
- Worst-source AUROC: {all_robust['worst_source_auroc']:.4f}
- LODO worst-source AUROCs: ChartQA {lodo['chartqa']['worst_target_source_auroc']:.4f}, TextVQA {lodo['textvqa']['worst_target_source_auroc']:.4f}, GQA {lodo['gqa']['worst_target_source_auroc']:.4f}

Frozen rule outcome: source robust = `{str(decision['source_robust']).lower()}`; broad OOD transfer = `{str(decision['broad_ood_transfer']).lower()}`.

{consequence}

This phase stops here. No threshold, trigger map, Stage-2 artifact, corrective search, or deployment evaluation was produced.
"""
    _atomic_bytes(
        OUTPUT_ROOT / "summaries/stage1_robustness_decision.md",
        decision_summary.encode(),
    )


def aggregate() -> None:
    contract = load_contract()
    canonical, historical, source_summary, robustness, breakdown = _aggregate_main(
        contract
    )
    lodo_summary, comparison, lodo_layerwise = _aggregate_lodo(contract)
    _lodo_figures(lodo_summary, comparison, lodo_layerwise)
    all_source = next(row for row in source_summary if row["head"] == "all_source")
    historical_lodo_mean = float(
        np.mean([float(row["historical_target_auroc"]) for row in lodo_summary])
    )
    phase49_mean = float(np.mean(list(PHASE49_REFERENCES.values())))
    decision = choose_robustness_decision(
        historical_auroc=float(all_source["historical_auroc"]),
        canonical_auroc=float(all_source["canonical_auroc"]),
        lodo_worst_source_aurocs=[
            float(row["worst_target_source_auroc"]) for row in lodo_summary
        ],
        historical_lodo_mean=historical_lodo_mean,
        phase49_historical_mean=phase49_mean,
    )
    decision["historical_lodo_mean"] = historical_lodo_mean
    decision["phase49_historical_mean"] = phase49_mean
    _summaries(source_summary, robustness, breakdown, lodo_summary, comparison, decision)

    required = [
        "protocol.md",
        "frozen_protocol.json",
        "main_all/folds/fold_manifest.jsonl",
        "main_all/folds/fold_summary.csv",
        "main_all/training/checkpoint_manifest.json",
        "main_all/predictions/canonical_oof_scores.jsonl",
        "main_all/predictions/historical_ensemble_scores.jsonl",
        "main_all/metrics/source_summary.csv",
        "main_all/metrics/robustness_metrics.csv",
        "main_all/metrics/dataset_source_breakdown.csv",
        "main_all/metrics/layerwise_auroc.csv",
        "main_all/metrics/layerwise_auprc.csv",
        "main_all/metrics/source_conditioned_score_stats.csv",
        "main_all/figures/historical_vs_canonical_roc.png",
        "main_all/figures/source_robustness_comparison.png",
        "main_all/figures/layerwise_source_auroc.png",
        "main_all/figures/worst_source_auroc_by_layer.png",
        "main_all/figures/source_conditioned_score_distributions.png",
        "lodo/lodo_summary.csv",
        "lodo/phase49_comparison.csv",
        "lodo/figures/lodo_auroc_matrix.png",
        "lodo/figures/phase49_vs_all_source_lodo.png",
        "lodo/figures/lodo_layerwise_auroc.png",
        "lodo/figures/lodo_worst_source_auroc.png",
        "summaries/all_source_training_summary.md",
        "summaries/dataset_ood_summary.md",
        "summaries/stage1_robustness_decision.md",
    ]
    for target in DATASETS:
        required.extend(
            [
                f"lodo/{target}/protocol.md",
                f"lodo/{target}/training/checkpoint.pt",
                f"lodo/{target}/training/history.json",
                f"lodo/{target}/predictions/historical_target_scores.jsonl",
                f"lodo/{target}/predictions/canonical_target_scores.jsonl",
                f"lodo/{target}/metrics.csv",
                f"lodo/{target}/layerwise_auroc.csv",
            ]
        )
    missing = [relative for relative in required if not (OUTPUT_ROOT / relative).is_file()]
    if missing:
        raise RuntimeError(f"required ALL-source artifacts are missing: {missing}")
    all_files = sorted(
        path
        for path in OUTPUT_ROOT.rglob("*")
        if path.is_file()
        and path.name != "artifact_manifest.json"
        and ".tmp." not in path.name
    )
    artifacts = [
        {
            "path": str(path.relative_to(OUTPUT_ROOT)),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in all_files
    ]
    atomic_json(
        OUTPUT_ROOT / "artifact_manifest.json",
        {
            "schema_version": "stage1_all_source_robustness_artifact_manifest_v1",
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "canonical_oof_records": len(canonical),
            "historical_ensemble_records": len(historical),
            "uid_overlap": 0,
            "image_group_overlap": 0,
            "image_sha_overlap": 0,
            "decision": decision,
            "artifacts": artifacts,
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "decision": decision["decision"],
                "historical_auroc": all_source["historical_auroc"],
                "canonical_auroc": all_source["canonical_auroc"],
            }
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    main_fold = subparsers.add_parser("train-main-fold")
    main_fold.add_argument("--fold", required=True, type=int)
    lodo = subparsers.add_parser("train-lodo")
    lodo.add_argument("--target", required=True, choices=DATASETS)
    subparsers.add_parser("aggregate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "train-main-fold":
        train_main_fold(args.fold)
    elif args.command == "train-lodo":
        train_lodo(args.target)
    elif args.command == "aggregate":
        aggregate()


if __name__ == "__main__":
    main()
