#!/usr/bin/env python3
"""Run the same-head canonical Stage-1 refit diagnostic (Arm A only)."""

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

from dense_failure_stage1.canonical_refit import (  # noqa: E402
    assign_group_folds,
    balanced_epoch_indices,
    inner_validation_uids,
    paired_bootstrap_auc_difference,
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


OUTPUT_ROOT = PROJECT_ROOT / "analysis/dense_failure_stage1/canonical_refit_diagnostic"
DATASETS = ("gqa", "chartqa", "textvqa")
LAYERS = tuple(range(28))
BLOCKS = ("text_final", "text_mean", "visual_mean")
SEED = 20260903
HISTORICAL_SEED = 20260831
FOLDS = 5
INNER_VALIDATION_FRACTION = 0.125
BOOTSTRAP_DRAWS = 5000

SOURCES = {
    "plan": "plans/stage1_canonical_refit_diagnostic_plan.md",
    "candidate": "analysis/dense_failure_stage2/data_scale_search/manifests/new_candidate_manifest.jsonl",
    "dense": "analysis/dense_failure_stage2/data_scale_search/manifests/new_dense_results.jsonl",
    "canonical_feature_index": "analysis/dense_failure_stage2/data_scale_search/dense/features/feature_index.jsonl",
    "old_canonical_scores": "analysis/dense_failure_stage2/data_scale_search/manifests/new_trigger_map.jsonl",
    "old_normalization": "analysis/dense_failure_stage1/shared_global_gate/training/global_normalization.pt",
    "old_checkpoint": "analysis/dense_failure_stage1/shared_global_gate/training/checkpoints/state_layer_random4.pt",
    "historical_config": "configs/shared_stage1_global_risk_gate_v1.json",
    "historical_feature_config": "configs/layerwise_dense_failure_probe_v1.json",
    "historical_split": "analysis/dense_failure_stage1/layerwise_failure_probe/split_manifest.jsonl",
    "old_historical_val_scores": "analysis/dense_failure_stage1/trigger_map/manifests/trigger_map_val.jsonl",
    "old_historical_test_scores": "analysis/dense_failure_stage1/trigger_map/manifests/trigger_map_test.jsonl",
}
BOUND_CODE = (
    "experiments/run_stage1_canonical_refit_diagnostic.py",
    "dense_failure_stage1/canonical_refit.py",
    "dense_failure_stage1/shared_global_gate.py",
    "dense_failure_stage1/layerwise_probe.py",
    "experiments/analyze_shared_stage1_global_risk_gate.py",
    "experiments/analyze_layerwise_dense_failure_predictability.py",
)
EXPECTED_COUNTS = {
    ("gqa", False): 1264,
    ("gqa", True): 736,
    ("chartqa", False): 884,
    ("chartqa", True): 116,
    ("textvqa", False): 981,
    ("textvqa", True): 19,
}
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
    "sampling": "all minority-class records plus equal no-replacement majority subsample each epoch",
}


def resolve(relative: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    allowed_external = Path("/mnt/hyemin").resolve()
    if not (path.is_relative_to(PROJECT_ROOT) or path.is_relative_to(allowed_external)):
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
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{number}")
            rows.append(row)
    return rows


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_bytes(
        path,
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows).encode(
            "utf-8"
        ),
    )


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode("utf-8"))


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


def _canonical_rows(paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    candidates = {str(row["uid"]): row for row in read_jsonl(paths["candidate"])}
    dense = {str(row["uid"]): row for row in read_jsonl(paths["dense"])}
    old = {str(row["uid"]): row for row in read_jsonl(paths["old_canonical_scores"])}
    features = {
        str(row["uid"]): row for row in read_jsonl(paths["canonical_feature_index"])
    }
    if not (set(candidates) == set(dense) == set(old) == set(features)):
        raise RuntimeError("canonical candidate/dense/old-score/feature UIDs differ")
    if len(dense) != 4000:
        raise RuntimeError(f"canonical population differs from 4,000: {len(dense)}")
    rows = []
    for uid in sorted(dense):
        candidate, outcome, score, feature = (
            candidates[uid],
            dense[uid],
            old[uid],
            features[uid],
        )
        signatures = {
            str(candidate["dataset"]),
            str(outcome["dataset"]),
            str(score["dataset"]),
        }
        if len(signatures) != 1:
            raise RuntimeError(f"dataset mismatch for {uid}")
        label = bool(outcome["current_dense_wrong"])
        if label != bool(score["dense_wrong"]):
            raise RuntimeError(f"canonical label/old-score mismatch for {uid}")
        groups = {
            str(candidate["image_group_id"]),
            str(outcome["image_group_id"]),
            str(score["image_group_id"]),
        }
        if len(groups) != 1:
            raise RuntimeError(f"image-group mismatch for {uid}")
        rows.append(
            {
                "schema_version": "stage1_canonical_refit_fold_v1",
                "uid": uid,
                "dataset": signatures.pop(),
                "current_dense_correct": not label,
                "current_dense_wrong": label,
                "image_group_id": groups.pop(),
                "image_content_sha256": str(outcome["image_content_sha256"]),
                "feature_shard": str(feature["shard"]),
                "feature_row_index": int(feature["row_index"]),
            }
        )
    counts = Counter((row["dataset"], row["current_dense_wrong"]) for row in rows)
    if counts != Counter(EXPECTED_COUNTS):
        raise RuntimeError(f"canonical class counts differ: {counts}")
    if len({row["image_content_sha256"] for row in rows}) != 4000:
        raise RuntimeError("canonical population does not have 4,000 unique image hashes")
    if len({row["image_group_id"] for row in rows}) != 4000:
        raise RuntimeError("canonical population does not have 4,000 unique image groups")
    return rows


def _canonical_shard_hashes(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    paths = sorted({str(row["feature_shard"]) for row in rows})
    hashes = {}
    for name in paths:
        path = resolve(name)
        hashes[name] = file_sha256(path)
    return hashes


def _fold_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for fold in range(FOLDS):
        selected = [row for row in rows if int(row["fold"]) == fold]
        groups = {str(row["image_group_id"]) for row in selected}
        entry: dict[str, Any] = {
            "fold": fold,
            "records": len(selected),
            "image_groups": len(groups),
        }
        for dataset in DATASETS:
            for label, suffix in ((False, "correct"), (True, "wrong")):
                entry[f"{dataset}_{suffix}"] = sum(
                    row["dataset"] == dataset
                    and bool(row["current_dense_wrong"]) == label
                    for row in selected
                )
        output.append(entry)
    return output


def prepare() -> None:
    paths = _source_paths()
    rows = assign_group_folds(_canonical_rows(paths), folds=FOLDS, seed=SEED)
    if len({str(row["uid"]) for row in rows}) != 4000:
        raise RuntimeError("fold assignment lost or duplicated UIDs")
    group_folds: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        group_folds[str(row["image_group_id"])].add(int(row["fold"]))
    if any(len(value) != 1 for value in group_folds.values()):
        raise RuntimeError("an image group crosses outer folds")

    status = command_output(["git", "status", "--porcelain=v1", "--untracked-files=all"])
    source_hashes = {name: file_sha256(path) for name, path in paths.items()}
    code_hashes = {name: file_sha256(resolve(name)) for name in BOUND_CODE}
    shard_hashes = _canonical_shard_hashes(rows)
    contract: dict[str, Any] = {
        "schema_version": "stage1_canonical_refit_frozen_protocol_v1",
        "run_id": "stage1_canonical_refit_diagnostic_arm_a_v1",
        "objective": "same historical Random-4 head and old normalization refit on canonical current-runtime dense correctness",
        "seed": SEED,
        "historical_training_seed": HISTORICAL_SEED,
        "population": {
            "records": 4000,
            "folds": FOLDS,
            "fold_assignment": "deterministic image-group-disjoint dataset-by-current-label stratification",
            "internal_validation_fraction": INNER_VALIDATION_FRACTION,
            "class_counts": {
                f"{dataset}_{'wrong' if wrong else 'correct'}": count
                for (dataset, wrong), count in EXPECTED_COUNTS.items()
            },
        },
        "input": {
            "blocks": list(BLOCKS),
            "layers": 28,
            "input_size": 10752,
            "normalization": "exact frozen historical global normalization",
            "normalization_sha256": source_hashes["old_normalization"],
        },
        "architecture": ARCHITECTURE,
        "training": TRAINING,
        "evaluation": {
            "positive_class": "current_dense_wrong",
            "primary_score": "maximum probability across layers 0-27",
            "canonical_claims": "out-of-fold predictions only",
            "bootstrap": "paired UID resampling",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "decision_rule": {
                "overall_above_chance": "95% bootstrap lower bound of AUROC-minus-0.5 > 0",
                "material_old_head_improvement": "OOF max-score AUROC gain >= 0.05 and paired 95% lower bound > 0",
                "chartqa_inversion_removed": "ChartQA OOF max-score AUROC > 0.5",
            },
            "prohibited": [
                "threshold calibration",
                "trigger-map regeneration",
                "Stage-2 changes",
                "Arm-B canonical normalization",
                "mixed historical-canonical training",
            ],
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
            "canonical_feature_shard_sha256": shard_hashes,
        },
    }
    if source_hashes["old_normalization"] != "ce4b63ab7f503d879db20f94aae01116804095eb87fd03b16a818d5904f1aed3":
        raise RuntimeError("old normalization is not the expected frozen artifact")
    if source_hashes["old_checkpoint"] != "1aeeaa278a9ac2aa99343b4ea26fad43a8b097c05b497c27554289eaef311211":
        raise RuntimeError("old Random-4 checkpoint is not the expected frozen artifact")
    contract["contract_sha256"] = canonical_hash(contract)
    protocol = "\n".join(
        [
            "# Canonical Stage-1 Refit Diagnostic Protocol",
            "",
            f"- Frozen contract: `{contract['contract_sha256']}`",
            "- Arm A only: exact historical Shared Random-4 architecture and exact old global normalization.",
            "- Target: frozen current-runtime LMMS dense-wrong label; wrong is positive.",
            "- Five deterministic dataset/label-stratified image-group-disjoint outer folds.",
            "- Each outer-training partition alone supplies its group-disjoint 12.5% internal validation subset.",
            "- Training epochs are C/W balanced by using every minority row and an equal no-replacement majority sample; held-out prevalence remains natural.",
            "- Ten FP32 AdamW epochs, lr 5e-4, weight decay 0.01, epoch-wise cosine schedule, four unique random layers per selected sample, minimum full-28 validation BCE checkpoint.",
            "- Primary canonical evidence is the concatenated 4,000-row OOF max-trajectory score. No threshold is selected.",
            "- The full-canonical model is fit only for historical validation/test cross-evaluation and is not used for canonical claims.",
            "- Arm B, mixed-source training, Stage-2 work, corrective search, and trigger-map regeneration are prohibited.",
            "",
        ]
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frozen_path = OUTPUT_ROOT / "frozen_protocol.json"
    if frozen_path.exists():
        existing = read_json(frozen_path)
        if existing.get("contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError("refusing to overwrite a different frozen protocol")
    else:
        atomic_json(frozen_path, contract)
    atomic_jsonl(OUTPUT_ROOT / "splits/fold_manifest.jsonl", rows)
    atomic_csv(OUTPUT_ROOT / "splits/fold_summary.csv", _fold_summary(rows))
    _atomic_bytes(OUTPUT_ROOT / "protocol.md", protocol.encode("utf-8"))
    print(json.dumps({"passed": True, "contract_sha256": contract["contract_sha256"]}))


def load_contract() -> dict[str, Any]:
    contract = read_json(OUTPUT_ROOT / "frozen_protocol.json")
    if (
        contract.get("schema_version") != "stage1_canonical_refit_frozen_protocol_v1"
        or contract.get("contract_sha256") != canonical_hash(contract)
    ):
        raise RuntimeError("canonical-refit frozen protocol is invalid")
    provenance = contract["provenance"]
    if command_output(["git", "rev-parse", "HEAD"]) != provenance["git_commit"]:
        raise RuntimeError("git commit differs from frozen protocol")
    for name, expected in provenance["source_sha256"].items():
        if file_sha256(resolve(contract["sources"][name])) != expected:
            raise RuntimeError(f"frozen source differs: {name}")
    for name, expected in provenance["bound_code_sha256"].items():
        if file_sha256(resolve(name)) != expected:
            raise RuntimeError(f"bound code differs: {name}")
    for name, expected in provenance["canonical_feature_shard_sha256"].items():
        if file_sha256(resolve(name)) != expected:
            raise RuntimeError(f"canonical feature shard differs: {name}")
    return contract


def load_canonical_features(
    rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> torch.Tensor:
    target = torch.empty((len(rows), 28, 10752), dtype=torch.bfloat16)
    filled = torch.zeros(len(rows), dtype=torch.bool)
    by_shard: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for destination, row in enumerate(rows):
        by_shard[str(row["feature_shard"])].append((destination, row))
    frozen_hashes = contract["provenance"]["canonical_feature_shard_sha256"]
    for name in sorted(by_shard):
        path = resolve(name)
        if file_sha256(path) != frozen_hashes.get(name):
            raise RuntimeError(f"feature shard fails frozen hash check: {name}")
        shard = torch.load(path, map_location="cpu", weights_only=True)
        if shard.get("layer_ids") != list(LAYERS) or int(shard.get("records", -1)) != len(
            shard.get("uids", [])
        ):
            raise RuntimeError(f"feature shard schema differs: {name}")
        matrix = torch.cat([shard[block] for block in BLOCKS], dim=-1)
        if matrix.ndim != 3 or tuple(matrix.shape[1:]) != (28, 10752):
            raise RuntimeError(f"feature shard tensor shape differs: {name}")
        for destination, row in by_shard[name]:
            source = int(row["feature_row_index"])
            if str(shard["uids"][source]) != str(row["uid"]):
                raise RuntimeError(f"feature UID mismatch: {row['uid']}")
            target[destination].copy_(matrix[source])
            filled[destination] = True
    if not bool(filled.all()) or not bool(torch.isfinite(target.float()).all()):
        raise RuntimeError("canonical feature loading is incomplete or non-finite")
    return target


def _model() -> SharedFailurePredictor:
    return SharedFailurePredictor(
        variant="state_layer_random4",
        input_size=ARCHITECTURE["input_size"],
        projection_size=ARCHITECTURE["projection_size"],
        layer_embedding_size=ARCHITECTURE["layer_embedding_size"],
        hidden_size=ARCHITECTURE["hidden_size"],
    )


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


def _normalization() -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.load(resolve(SOURCES["old_normalization"]), map_location="cpu", weights_only=True)
    if (
        value.get("schema_version") != "shared_stage1_global_normalization_v1"
        or tuple(value["mean"].shape) != (10752,)
        or tuple(value["std"].shape) != (10752,)
    ):
        raise RuntimeError("old normalization schema differs")
    return value["mean"].float(), value["std"].float()


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
    rows = torch.arange(batch, dtype=torch.long)[:, None]
    selected = selected[rows, layer_choices]
    states = selected.reshape(batch * count, -1).to(device=device, dtype=torch.float32)
    states = (states - mean) / std
    return model(states, layer_choices.reshape(-1).to(device=device, dtype=torch.long))


def _score(
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
            choices = all_layers.expand(stop - start, -1)
            values = _batch_logits(
                model,
                features,
                indices,
                choices,
                mean=mean,
                std=std,
                device=device,
            )
            logits.append(values.reshape(stop - start, 28).cpu())
    logit_values = torch.cat(logits).numpy().astype(np.float64)
    scores = np.empty_like(logit_values)
    positive = logit_values >= 0
    scores[positive] = 1.0 / (1.0 + np.exp(-logit_values[positive]))
    exp = np.exp(logit_values[~positive])
    scores[~positive] = exp / (1.0 + exp)
    return logit_values, scores


def _validation_bce(logits: np.ndarray, labels: np.ndarray) -> float:
    flat = logits.reshape(-1)
    repeated = np.repeat(labels.astype(np.int64), 28)
    return float(
        np.mean(np.maximum(flat, 0) - flat * repeated + np.log1p(np.exp(-np.abs(flat))))
    )


def _run_epochs(
    model: SharedFailurePredictor,
    features: torch.Tensor,
    labels: np.ndarray,
    training_indices: np.ndarray,
    validation_indices: np.ndarray | None,
    *,
    seed: int,
    epochs: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor] | None, int, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAINING["epochs"]
    )
    history = []
    best_state = None
    best_epoch = -1
    best_bce = float("inf")
    local_labels = labels[training_indices]
    for epoch in range(epochs):
        model.train()
        balanced_local = balanced_epoch_indices(local_labels, seed=seed + 10000, epoch=epoch)
        epoch_indices = training_indices[balanced_local]
        layer_choices = deterministic_random_layers(
            len(epoch_indices),
            epoch=epoch,
            seed=seed,
            count=TRAINING["random_layers_per_sample"],
        )
        accumulated = 0.0
        for start in range(0, len(epoch_indices), TRAINING["batch_size_samples"]):
            stop = min(start + TRAINING["batch_size_samples"], len(epoch_indices))
            indices = torch.as_tensor(epoch_indices[start:stop], dtype=torch.long)
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
            targets = torch.as_tensor(labels[epoch_indices[start:stop]], dtype=torch.float32)
            targets = targets.to(device).repeat_interleave(choices.shape[1])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()
            accumulated += float(loss.detach().cpu()) * (stop - start)
        entry: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "balanced_training_records": len(epoch_indices),
            "training_correct": int((labels[epoch_indices] == 0).sum()),
            "training_wrong": int((labels[epoch_indices] == 1).sum()),
            "train_loss": accumulated / len(epoch_indices),
        }
        if validation_indices is not None:
            val_logits, val_scores = _score(
                model,
                features.index_select(0, torch.as_tensor(validation_indices)),
                mean=mean,
                std=std,
                device=device,
            )
            val_labels = labels[validation_indices]
            bce = _validation_bce(val_logits, val_labels)
            entry["validation_full28_bce"] = bce
            entry["validation_max_auroc"] = float(
                binary_metrics(val_labels, val_scores.max(axis=1))["auroc"]
            )
            if bce < best_bce - 1e-12:
                best_bce = bce
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        history.append(entry)
        scheduler.step()
    if validation_indices is not None and (best_state is None or best_epoch < 0):
        raise RuntimeError("training failed to select a checkpoint")
    return history, best_state, best_epoch, best_bce


def _prediction_rows(
    rows: Sequence[Mapping[str, Any]], scores: np.ndarray, *, model_id: str, fold: int | str
) -> list[dict[str, Any]]:
    if scores.shape != (len(rows), 28):
        raise RuntimeError("prediction shape differs")
    output = []
    for row, values in zip(rows, scores):
        entry = {
            "schema_version": "stage1_canonical_refit_score_v1",
            "uid": str(row["uid"]),
            "dataset": str(row["dataset"]),
            "image_group_id": str(row["image_group_id"]),
            "current_dense_correct": not bool(row["current_dense_wrong"]),
            "current_dense_wrong": bool(row["current_dense_wrong"]),
            "model_id": model_id,
            "fold": fold,
            "score_max": float(values.max()),
        }
        if "split" in row:
            entry["split"] = str(row["split"])
        entry.update({f"p_{layer}": float(values[layer]) for layer in LAYERS})
        output.append(entry)
    return output


def reproduction_check() -> None:
    contract = load_contract()
    device = _gpu()
    folded = read_jsonl(OUTPUT_ROOT / "splits/fold_manifest.jsonl")
    selected = []
    for dataset in DATASETS:
        for label in (False, True):
            selected.extend(
                [
                    row
                    for row in folded
                    if row["dataset"] == dataset
                    and bool(row["current_dense_wrong"]) == label
                ][:6]
            )
    features = load_canonical_features(selected, contract)
    checkpoint = torch.load(resolve(SOURCES["old_checkpoint"]), map_location="cpu", weights_only=True)
    if checkpoint.get("variant") != "state_layer_random4":
        raise RuntimeError("old checkpoint variant differs")
    model = _model().to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model_state_dict"])
    mean_cpu, std_cpu = _normalization()
    _, scores = _score(
        model,
        features,
        mean=mean_cpu.to(device),
        std=std_cpu.to(device),
        device=device,
    )
    frozen = {str(row["uid"]): row for row in read_jsonl(resolve(SOURCES["old_canonical_scores"]))}
    expected = np.asarray(
        [[float(frozen[str(row["uid"])][f"p_{layer}"]) for layer in LAYERS] for row in selected]
    )
    maximum_error = float(np.abs(scores - expected).max())
    if maximum_error > 1e-6:
        raise RuntimeError(f"historical recipe reproduction differs: {maximum_error}")
    atomic_json(
        OUTPUT_ROOT / "training/historical_recipe_reproduction_check.json",
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "records": len(selected),
            "layers": 28,
            "maximum_absolute_score_error": maximum_error,
            "tolerance": 1e-6,
            "old_checkpoint_sha256": file_sha256(resolve(SOURCES["old_checkpoint"])),
            "old_normalization_sha256": file_sha256(resolve(SOURCES["old_normalization"])),
        },
    )
    print(json.dumps({"passed": True, "maximum_absolute_score_error": maximum_error}))


def train_fold(fold: int) -> None:
    if fold not in range(FOLDS):
        raise ValueError("fold must lie in 0..4")
    contract = load_contract()
    reproduction = read_json(OUTPUT_ROOT / "training/historical_recipe_reproduction_check.json")
    if reproduction.get("passed") is not True:
        raise RuntimeError("historical recipe reproduction check has not passed")
    directory = OUTPUT_ROOT / f"training/fold_{fold}"
    complete_path = directory / "complete.json"
    if complete_path.exists():
        complete = read_json(complete_path)
        for key in ("checkpoint", "oof_scores", "history", "inner_split"):
            if file_sha256(directory / complete[f"{key}_file"]) != complete[f"{key}_sha256"]:
                raise RuntimeError(f"fold {fold} resume artifact differs: {key}")
        if complete.get("contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError(f"fold {fold} resume contract differs")
        print(json.dumps({"passed": True, "fold": fold, "resumed": True}))
        return

    device = _gpu()
    rows = read_jsonl(OUTPUT_ROOT / "splits/fold_manifest.jsonl")
    features = load_canonical_features(rows, contract)
    labels = np.asarray([int(bool(row["current_dense_wrong"])) for row in rows], dtype=np.int64)
    outer_test = np.asarray([i for i, row in enumerate(rows) if int(row["fold"]) == fold])
    outer_training_rows = [row for row in rows if int(row["fold"]) != fold]
    inner_uids = inner_validation_uids(
        outer_training_rows,
        fraction=INNER_VALIDATION_FRACTION,
        seed=SEED,
        outer_fold=fold,
    )
    inner_val = np.asarray([i for i, row in enumerate(rows) if str(row["uid"]) in inner_uids])
    training = np.asarray(
        [
            i
            for i, row in enumerate(rows)
            if int(row["fold"]) != fold and str(row["uid"]) not in inner_uids
        ]
    )
    sets = [set(training.tolist()), set(inner_val.tolist()), set(outer_test.tolist())]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("fold partitions overlap")
    if set.union(*sets) != set(range(4000)):
        raise RuntimeError("fold partitions are incomplete")
    group_sets = [
        {str(rows[index]["image_group_id"]) for index in values} for values in sets
    ]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("fold image groups overlap")
    inner_rows = []
    for index, role in [(i, "train") for i in training] + [(i, "internal_val") for i in inner_val] + [(i, "outer_test") for i in outer_test]:
        inner_rows.append({"uid": rows[index]["uid"], "role": role})
    inner_rows.sort(key=lambda row: str(row["uid"]))
    atomic_jsonl(directory / "inner_split.jsonl", inner_rows)

    seed = HISTORICAL_SEED + fold
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _model().to(device=device, dtype=torch.float32)
    mean_cpu, std_cpu = _normalization()
    mean, std = mean_cpu.to(device), std_cpu.to(device)
    history, best_state, best_epoch, best_bce = _run_epochs(
        model,
        features,
        labels,
        training,
        inner_val,
        seed=seed,
        epochs=TRAINING["epochs"],
        mean=mean,
        std=std,
        device=device,
    )
    assert best_state is not None
    model.load_state_dict(best_state)
    _, scores = _score(
        model,
        features.index_select(0, torch.as_tensor(outer_test)),
        mean=mean,
        std=std,
        device=device,
    )
    test_rows = [rows[index] for index in outer_test]
    prediction_rows = _prediction_rows(test_rows, scores, model_id=f"canonical_refit_fold_{fold}", fold=fold)
    checkpoint = {
        "schema_version": "stage1_canonical_refit_checkpoint_v1",
        "contract_sha256": contract["contract_sha256"],
        "fold": fold,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_internal_validation_full28_bce": best_bce,
        "old_normalization_sha256": contract["input"]["normalization_sha256"],
        "model_state_dict": best_state,
    }
    atomic_torch(directory / "checkpoint.pt", checkpoint)
    atomic_json(
        directory / "history.json",
        {
            "contract_sha256": contract["contract_sha256"],
            "fold": fold,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_internal_validation_full28_bce": best_bce,
            "partition_counts": {
                "train": len(training),
                "internal_val": len(inner_val),
                "outer_test": len(outer_test),
            },
            "training_sampling_probability": {
                "wrong": min(1.0, int((labels[training] == 0).sum()) / int((labels[training] == 1).sum())),
                "correct": min(1.0, int((labels[training] == 1).sum()) / int((labels[training] == 0).sum())),
            },
            "history": history,
        },
    )
    atomic_jsonl(directory / "oof_scores.jsonl", prediction_rows)
    files = {
        "checkpoint": "checkpoint.pt",
        "oof_scores": "oof_scores.jsonl",
        "history": "history.json",
        "inner_split": "inner_split.jsonl",
    }
    complete = {
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        "fold": fold,
        "outer_test_records": len(test_rows),
    }
    for key, name in files.items():
        complete[f"{key}_file"] = name
        complete[f"{key}_sha256"] = file_sha256(directory / name)
    atomic_json(complete_path, complete)
    print(json.dumps({"passed": True, "fold": fold, "best_epoch": best_epoch, "oof_records": len(test_rows)}))


def _historical_features() -> tuple[list[dict[str, Any]], torch.Tensor]:
    config = resolve(SOURCES["historical_feature_config"])
    contract, root = load_historical_feature_contract(config)
    rows, matrices = load_layer_matrices(
        contract, root, layers=LAYERS, selected_splits={"val", "test"}
    )
    features = torch.stack([matrices.pop(layer) for layer in LAYERS], dim=1)
    if features.shape != (1600, 28, 10752):
        raise RuntimeError("historical validation/test feature shape differs")
    return rows, features


def train_full() -> None:
    contract = load_contract()
    for fold in range(FOLDS):
        if read_json(OUTPUT_ROOT / f"training/fold_{fold}/complete.json").get("passed") is not True:
            raise RuntimeError("all OOF folds must complete before the full fit")
    directory = OUTPUT_ROOT / "training/full_canonical_fit"
    complete_path = directory / "complete.json"
    if complete_path.exists():
        complete = read_json(complete_path)
        for key in ("checkpoint", "historical_scores", "canonical_scores", "history"):
            if file_sha256(directory / complete[f"{key}_file"]) != complete[f"{key}_sha256"]:
                raise RuntimeError(f"full-fit resume artifact differs: {key}")
        if complete.get("contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError("full-fit resume contract differs")
        print(json.dumps({"passed": True, "resumed": True, "stage": "full_fit"}))
        return

    device = _gpu()
    rows = read_jsonl(OUTPUT_ROOT / "splits/fold_manifest.jsonl")
    features = load_canonical_features(rows, contract)
    labels = np.asarray([int(bool(row["current_dense_wrong"])) for row in rows], dtype=np.int64)
    inner_uids = inner_validation_uids(
        rows,
        fraction=INNER_VALIDATION_FRACTION,
        seed=SEED,
        outer_fold=FOLDS,
    )
    selection_train = np.asarray([i for i, row in enumerate(rows) if str(row["uid"]) not in inner_uids])
    selection_val = np.asarray([i for i, row in enumerate(rows) if str(row["uid"]) in inner_uids])
    seed = HISTORICAL_SEED + 100
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    selection_model = _model().to(device=device, dtype=torch.float32)
    mean_cpu, std_cpu = _normalization()
    mean, std = mean_cpu.to(device), std_cpu.to(device)
    selection_history, _, best_epoch, best_bce = _run_epochs(
        selection_model,
        features,
        labels,
        selection_train,
        selection_val,
        seed=seed,
        epochs=TRAINING["epochs"],
        mean=mean,
        std=std,
        device=device,
    )
    del selection_model
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _model().to(device=device, dtype=torch.float32)
    refit_history, _, _, _ = _run_epochs(
        model,
        features,
        labels,
        np.arange(len(rows)),
        None,
        seed=seed,
        epochs=best_epoch + 1,
        mean=mean,
        std=std,
        device=device,
    )
    _, canonical_scores = _score(model, features, mean=mean, std=std, device=device)
    canonical_prediction_rows = _prediction_rows(
        rows, canonical_scores, model_id="full_canonical_fit_descriptive_only", fold="in_sample"
    )
    historical_rows, historical_features = _historical_features()
    _, historical_scores = _score(
        model,
        historical_features,
        mean=mean,
        std=std,
        device=device,
    )
    historical_prediction_rows = _prediction_rows(
        historical_rows,
        historical_scores,
        model_id="full_canonical_fit",
        fold="historical_heldout",
    )
    checkpoint = {
        "schema_version": "stage1_canonical_refit_checkpoint_v1",
        "contract_sha256": contract["contract_sha256"],
        "fold": "full_canonical",
        "seed": seed,
        "epoch_selection_best_epoch": best_epoch,
        "epoch_selection_best_internal_validation_full28_bce": best_bce,
        "full_population_epochs": best_epoch + 1,
        "old_normalization_sha256": contract["input"]["normalization_sha256"],
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    atomic_torch(directory / "checkpoint.pt", checkpoint)
    atomic_jsonl(directory / "historical_scores.jsonl", historical_prediction_rows)
    atomic_jsonl(directory / "canonical_scores_descriptive_only.jsonl", canonical_prediction_rows)
    atomic_json(
        directory / "history.json",
        {
            "contract_sha256": contract["contract_sha256"],
            "seed": seed,
            "epoch_selection": {
                "training_records": len(selection_train),
                "validation_records": len(selection_val),
                "best_epoch": best_epoch,
                "best_validation_full28_bce": best_bce,
                "history": selection_history,
            },
            "full_population_refit": {
                "records": len(rows),
                "epochs": best_epoch + 1,
                "history": refit_history,
            },
        },
    )
    files = {
        "checkpoint": "checkpoint.pt",
        "historical_scores": "historical_scores.jsonl",
        "canonical_scores": "canonical_scores_descriptive_only.jsonl",
        "history": "history.json",
    }
    complete = {"passed": True, "contract_sha256": contract["contract_sha256"]}
    for key, name in files.items():
        complete[f"{key}_file"] = name
        complete[f"{key}_sha256"] = file_sha256(directory / name)
    atomic_json(complete_path, complete)
    print(json.dumps({"passed": True, "stage": "full_fit", "selected_epochs": best_epoch + 1}))


def _scores(rows: Sequence[Mapping[str, Any]], prefix: str = "p_") -> np.ndarray:
    return np.asarray(
        [[float(row[f"{prefix}{layer}"]) for layer in LAYERS] for row in rows],
        dtype=np.float64,
    )


def _metric_row(
    labels: np.ndarray, scores: np.ndarray, *, head: str, dataset: str
) -> dict[str, Any]:
    metrics = binary_metrics(labels, scores)
    return {"head": head, "dataset": dataset, **metrics}


def _roc(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    tpr, fpr = [], []
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    for threshold in thresholds:
        predicted = scores >= threshold
        tpr.append(float(np.logical_and(predicted, labels == 1).sum() / positives))
        fpr.append(float(np.logical_and(predicted, labels == 0).sum() / negatives))
    return np.asarray(fpr), np.asarray(tpr)


def _distribution_rows(
    canonical_old: Sequence[Mapping[str, Any]],
    canonical_refit: Sequence[Mapping[str, Any]],
    historical_old: Sequence[Mapping[str, Any]],
    historical_refit: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    collections = (
        ("canonical", "old_historical_head", canonical_old),
        ("canonical", "canonical_refit_oof", canonical_refit),
        ("historical_val_test", "old_historical_head", historical_old),
        ("historical_val_test", "full_canonical_fit", historical_refit),
    )
    output = []
    for population, head, rows in collections:
        for dataset in ("overall", *DATASETS):
            for label, outcome in ((False, "correct"), (True, "wrong")):
                selected = [
                    row
                    for row in rows
                    if bool(row["current_dense_wrong"]) == label
                    and (dataset == "overall" or row["dataset"] == dataset)
                ]
                if not selected:
                    continue
                values = np.asarray([float(row["score_max"]) for row in selected])
                output.append(
                    {
                        "population": population,
                        "head": head,
                        "dataset": dataset,
                        "outcome": outcome,
                        "records": len(values),
                        "mean": float(values.mean()),
                        "median": float(np.median(values)),
                        "p90": float(np.quantile(values, 0.90)),
                        "p95": float(np.quantile(values, 0.95)),
                        "p99": float(np.quantile(values, 0.99)),
                        "maximum": float(values.max()),
                    }
                )
    return output


def _save_figures(
    old: list[dict[str, Any]], refit: list[dict[str, Any]], layer_rows: list[dict[str, Any]], cross_rows: list[dict[str, Any]]
) -> None:
    figure_root = OUTPUT_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([int(row["current_dense_wrong"]) for row in old])
    plt.figure(figsize=(6, 5))
    for name, rows, color in (("Frozen old", old, "#c44e52"), ("Canonical refit OOF", refit, "#4c72b0")):
        scores = np.asarray([float(row["score_max"]) for row in rows])
        fpr, tpr = _roc(labels, scores)
        auc = binary_metrics(labels, scores)["auroc"]
        plt.plot(fpr, tpr, label=f"{name} (AUROC={auc:.3f})", color=color)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_root / "canonical_old_vs_refit_roc.png", dpi=180)
    plt.close()

    for metric in ("auroc", "auprc"):
        plt.figure(figsize=(7, 4.5))
        for head, color in (("old_historical_head", "#c44e52"), ("canonical_refit_oof", "#4c72b0")):
            selected = sorted(
                [row for row in layer_rows if row["head"] == head and row["dataset"] == "overall"],
                key=lambda row: int(row["layer"]),
            )
            plt.plot([row["layer"] for row in selected], [row[metric] for row in selected], marker="o", markersize=2, label=head)
        plt.xlabel("Layer")
        plt.ylabel(metric.upper())
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_root / f"canonical_layerwise_{metric}_recovery.png", dpi=180)
        plt.close()

    for dataset in ("chartqa", "textvqa"):
        plt.figure(figsize=(7, 4.5))
        for rows, head, color in ((old, "Frozen old", "#c44e52"), (refit, "Canonical refit OOF", "#4c72b0")):
            for label, linestyle, outcome in ((False, "-", "correct"), (True, "--", "wrong")):
                values = [float(row["score_max"]) for row in rows if row["dataset"] == dataset and bool(row["current_dense_wrong"]) == label]
                plt.hist(values, bins=25, density=True, histtype="step", linewidth=1.7, color=color, linestyle=linestyle, label=f"{head} {outcome}")
        plt.xlabel("Max layer risk score")
        plt.ylabel("Density")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figure_root / f"{dataset}_score_distribution_old_vs_refit.png", dpi=180)
        plt.close()

    selected = [row for row in cross_rows if row["dataset"] == "overall"]
    labels_text = [f"{row['training_regime']}\n{row['evaluation_population']}" for row in selected]
    plt.figure(figsize=(8, 4.5))
    plt.bar(np.arange(len(selected)), [row["auroc"] for row in selected], color=["#c44e52", "#4c72b0"] * 3)
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.xticks(np.arange(len(selected)), labels_text, rotation=25, ha="right", fontsize=8)
    plt.ylabel("Max-score AUROC")
    plt.tight_layout()
    plt.savefig(figure_root / "canonical_refit_cross_regime_comparison.png", dpi=180)
    plt.close()


def aggregate() -> None:
    contract = load_contract()
    reproduction = read_json(OUTPUT_ROOT / "training/historical_recipe_reproduction_check.json")
    if reproduction.get("passed") is not True:
        raise RuntimeError("historical recipe reproduction check did not pass")
    oof = []
    checkpoints = []
    for fold in range(FOLDS):
        directory = OUTPUT_ROOT / f"training/fold_{fold}"
        complete = read_json(directory / "complete.json")
        if complete.get("passed") is not True or complete.get("contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError(f"fold {fold} is incomplete")
        path = directory / complete["oof_scores_file"]
        if file_sha256(path) != complete["oof_scores_sha256"]:
            raise RuntimeError(f"fold {fold} OOF hash differs")
        oof.extend(read_jsonl(path))
        checkpoints.append(
            {
                "model_id": f"canonical_refit_fold_{fold}",
                "path": str((directory / complete["checkpoint_file"]).relative_to(PROJECT_ROOT)),
                "sha256": complete["checkpoint_sha256"],
                "contract_sha256": contract["contract_sha256"],
            }
        )
    full_directory = OUTPUT_ROOT / "training/full_canonical_fit"
    full_complete = read_json(full_directory / "complete.json")
    if full_complete.get("passed") is not True or full_complete.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("full canonical fit is incomplete")
    checkpoints.append(
        {
            "model_id": "full_canonical_fit",
            "path": str((full_directory / full_complete["checkpoint_file"]).relative_to(PROJECT_ROOT)),
            "sha256": full_complete["checkpoint_sha256"],
            "contract_sha256": contract["contract_sha256"],
        }
    )
    manifest_rows = read_jsonl(OUTPUT_ROOT / "splits/fold_manifest.jsonl")
    expected_uids = {str(row["uid"]) for row in manifest_rows}
    if len(oof) != 4000 or len({str(row["uid"]) for row in oof}) != 4000 or {str(row["uid"]) for row in oof} != expected_uids:
        raise RuntimeError("OOF predictions are incomplete or duplicated")
    by_uid_oof = {str(row["uid"]): row for row in oof}
    oof = [by_uid_oof[uid] for uid in sorted(by_uid_oof)]
    old_source = {str(row["uid"]): row for row in read_jsonl(resolve(SOURCES["old_canonical_scores"]))}
    old = []
    for row in manifest_rows:
        uid = str(row["uid"])
        source = old_source[uid]
        values = np.asarray([float(source[f"p_{layer}"]) for layer in LAYERS])
        old.extend(_prediction_rows([row], values[None, :], model_id="frozen_old_historical_head", fold="frozen"))
    old.sort(key=lambda row: str(row["uid"]))
    atomic_jsonl(OUTPUT_ROOT / "predictions/canonical_oof_scores.jsonl", oof)
    atomic_jsonl(OUTPUT_ROOT / "predictions/frozen_old_head_canonical_scores.jsonl", old)

    historical_refit = read_jsonl(full_directory / full_complete["historical_scores_file"])
    historical_refit.sort(key=lambda row: (str(row["split"] if "split" in row else ""), str(row["uid"])))
    atomic_jsonl(OUTPUT_ROOT / "predictions/full_canonical_head_historical_scores.jsonl", historical_refit)
    atomic_json(OUTPUT_ROOT / "training/checkpoint_manifest.json", {"contract_sha256": contract["contract_sha256"], "checkpoints": checkpoints})

    labels = np.asarray([int(row["current_dense_wrong"]) for row in old])
    old_matrix, refit_matrix = _scores(old), _scores(oof)
    old_max, refit_max = old_matrix.max(axis=1), refit_matrix.max(axis=1)
    summary_rows = []
    dataset_rows = []
    for head, values in (("frozen_old_head", old_max), ("canonical_refit_oof", refit_max)):
        summary_rows.append(_metric_row(labels, values, head=head, dataset="overall"))
        for dataset in DATASETS:
            indices = np.asarray([i for i, row in enumerate(old) if row["dataset"] == dataset])
            dataset_rows.append(_metric_row(labels[indices], values[indices], head=head, dataset=dataset))
    atomic_csv(OUTPUT_ROOT / "metrics/canonical_oof_summary.csv", summary_rows)
    atomic_csv(OUTPUT_ROOT / "metrics/canonical_dataset_breakdown.csv", dataset_rows)

    layer_rows = []
    for dataset in ("overall", *DATASETS):
        indices = np.arange(len(old)) if dataset == "overall" else np.asarray([i for i, row in enumerate(old) if row["dataset"] == dataset])
        for layer in LAYERS:
            for head, matrix in (("old_historical_head", old_matrix), ("canonical_refit_oof", refit_matrix)):
                metrics = binary_metrics(labels[indices], matrix[indices, layer])
                layer_rows.append({"dataset": dataset, "head": head, "layer": layer, "records": len(indices), "correct": int((labels[indices] == 0).sum()), "wrong": int((labels[indices] == 1).sum()), "auroc": metrics["auroc"], "auprc": metrics["auprc"]})
    atomic_csv(OUTPUT_ROOT / "metrics/layerwise_auroc.csv", [{key: row[key] for key in ("dataset", "head", "layer", "records", "correct", "wrong", "auroc")} for row in layer_rows])
    atomic_csv(OUTPUT_ROOT / "metrics/layerwise_auprc.csv", [{key: row[key] for key in ("dataset", "head", "layer", "records", "correct", "wrong", "auprc")} for row in layer_rows])

    bootstrap_rows = []
    for dataset in ("overall", *DATASETS):
        indices = np.arange(len(old)) if dataset == "overall" else np.asarray([i for i, row in enumerate(old) if row["dataset"] == dataset])
        paired = paired_bootstrap_auc_difference(labels[indices], old_max[indices], refit_max[indices], draws=BOOTSTRAP_DRAWS, seed=SEED + len(bootstrap_rows))
        bootstrap_rows.append({"dataset": dataset, "comparison": "canonical_refit_oof_minus_frozen_old_head", "records": len(indices), "correct": int((labels[indices] == 0).sum()), "wrong": int((labels[indices] == 1).sum()), **paired})
        chance = paired_bootstrap_auc_difference(labels[indices], np.zeros(len(indices)), refit_max[indices], draws=BOOTSTRAP_DRAWS, seed=SEED + 100 + len(bootstrap_rows))
        bootstrap_rows.append({"dataset": dataset, "comparison": "canonical_refit_oof_minus_chance", "records": len(indices), "correct": int((labels[indices] == 0).sum()), "wrong": int((labels[indices] == 1).sum()), **chance})
    atomic_csv(OUTPUT_ROOT / "metrics/paired_bootstrap_difference.csv", bootstrap_rows)

    historical_old = []
    for split_name in ("val", "test"):
        for row in read_jsonl(resolve(SOURCES[f"old_historical_{split_name}_scores"])):
            values = np.asarray([float(row[f"score_l{layer}"]) for layer in LAYERS])
            historical_old.append(
                {
                    "uid": str(row["uid"]), "dataset": str(row["dataset"]), "split": split_name,
                    "image_group_id": str(row["image_group_id"]), "current_dense_wrong": bool(row["dense_wrong"]),
                    "current_dense_correct": bool(row["dense_correct"]), "score_max": float(values.max()),
                    **{f"p_{layer}": float(values[layer]) for layer in LAYERS},
                }
            )
    by_uid_refit = {str(row["uid"]): row for row in historical_refit}
    historical_refit = [by_uid_refit[str(row["uid"])] for row in historical_old]
    for row, source in zip(historical_refit, historical_old):
        row["split"] = source["split"]
    cross_rows = []
    regimes = (("historical_old_head", historical_old), ("full_canonical_fit", historical_refit))
    for split_name in ("val", "test", "val_test"):
        for regime, values in regimes:
            selected = [row for row in values if split_name == "val_test" or row["split"] == split_name]
            for dataset in ("overall", *DATASETS):
                subset = [row for row in selected if dataset == "overall" or row["dataset"] == dataset]
                y = np.asarray([int(row["current_dense_wrong"]) for row in subset])
                score = np.asarray([float(row["score_max"]) for row in subset])
                cross_rows.append({"training_regime": regime, "evaluation_population": f"historical_{split_name}", "dataset": dataset, **binary_metrics(y, score)})
    for regime, values in (("historical_old_head", old), ("canonical_refit_oof", oof)):
        for dataset in ("overall", *DATASETS):
            subset = [row for row in values if dataset == "overall" or row["dataset"] == dataset]
            y = np.asarray([int(row["current_dense_wrong"]) for row in subset])
            score = np.asarray([float(row["score_max"]) for row in subset])
            cross_rows.append({"training_regime": regime, "evaluation_population": "canonical_oof", "dataset": dataset, **binary_metrics(y, score)})
    atomic_csv(OUTPUT_ROOT / "metrics/historical_cross_eval.csv", cross_rows)
    distributions = _distribution_rows(old, oof, historical_old, historical_refit)
    atomic_csv(OUTPUT_ROOT / "metrics/score_distribution_summary.csv", distributions)
    _save_figures(old, oof, layer_rows, cross_rows)

    overall_old = summary_rows[0]
    overall_refit = summary_rows[1]
    dataset_refit = {row["dataset"]: row for row in dataset_rows if row["head"] == "canonical_refit_oof"}
    paired_overall = next(row for row in bootstrap_rows if row["dataset"] == "overall" and row["comparison"].endswith("old_head"))
    chance_overall = next(row for row in bootstrap_rows if row["dataset"] == "overall" and row["comparison"].endswith("chance"))
    decision_a = (
        chance_overall["ci_low"] > 0
        and paired_overall["observed_difference"] >= 0.05
        and paired_overall["ci_low"] > 0
        and dataset_refit["chartqa"]["auroc"] > 0.5
    )
    decision = "A — Same architecture is adequate" if decision_a else "B — Old normalization is the next suspect"
    best_recovery = sorted(
        [
            (layer, float(binary_metrics(labels, refit_matrix[:, layer])["auroc"]) - float(binary_metrics(labels, old_matrix[:, layer])["auroc"]))
            for layer in LAYERS
        ],
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    old_correct = old_max[labels == 0]
    refit_correct = refit_max[labels == 0]
    text_boot = next(row for row in bootstrap_rows if row["dataset"] == "textvqa" and row["comparison"].endswith("old_head"))
    historical_overall = {
        (row["training_regime"], row["evaluation_population"]): row
        for row in cross_rows if row["dataset"] == "overall"
    }
    summary = f"""# Canonical Refit Summary

## Outcome

The exact historical Shared Random-4 architecture, refit with canonical current-runtime labels and the frozen old normalization, achieved **{overall_refit['auroc']:.4f} OOF AUROC** versus **{overall_old['auroc']:.4f}** for the frozen old head on the same 4,000 samples. The paired difference was **{paired_overall['observed_difference']:+.4f}** (95% bootstrap CI [{paired_overall['ci_low']:+.4f}, {paired_overall['ci_high']:+.4f}]).

| Head | Overall AUROC | GQA | ChartQA | TextVQA | Overall AUPRC |
|---|---:|---:|---:|---:|---:|
| Frozen old head | {overall_old['auroc']:.4f} | {next(r['auroc'] for r in dataset_rows if r['head']=='frozen_old_head' and r['dataset']=='gqa'):.4f} | {next(r['auroc'] for r in dataset_rows if r['head']=='frozen_old_head' and r['dataset']=='chartqa'):.4f} | {next(r['auroc'] for r in dataset_rows if r['head']=='frozen_old_head' and r['dataset']=='textvqa'):.4f} | {overall_old['auprc']:.4f} |
| Canonical refit, old norm | {overall_refit['auroc']:.4f} | {dataset_refit['gqa']['auroc']:.4f} | {dataset_refit['chartqa']['auroc']:.4f} | {dataset_refit['textvqa']['auroc']:.4f} | {overall_refit['auprc']:.4f} |

ChartQA inversion {'disappeared' if dataset_refit['chartqa']['auroc'] > 0.5 else 'did not disappear'} (OOF AUROC {dataset_refit['chartqa']['auroc']:.4f}). The five largest layerwise AUROC gains were {', '.join(f'L{layer} ({gain:+.3f})' for layer, gain in best_recovery)}.

Canonical Dense-C maximum-risk scores changed from mean {old_correct.mean():.4f} / p95 {np.quantile(old_correct, .95):.4f} to mean {refit_correct.mean():.4f} / p95 {np.quantile(refit_correct, .95):.4f}; this {'reduces' if refit_correct.mean() < old_correct.mean() else 'does not reduce'} the extreme canonical-correct high-risk behavior descriptively.

TextVQA has only 19 wrong samples. Its OOF AUROC is {dataset_refit['textvqa']['auroc']:.4f}, and its paired refit-minus-old interval is [{text_boot['ci_low']:+.4f}, {text_boot['ci_high']:+.4f}]; the point estimate should not be treated as precise.

## Historical cross-evaluation

On the frozen historical validation+test population, the full-canonical fit achieved AUROC {historical_overall[('full_canonical_fit','historical_val_test')]['auroc']:.4f}; the old historical head achieved {historical_overall[('historical_old_head','historical_val_test')]['auroc']:.4f}. This cross-regime result did not select the canonical model.

## Interpretation

Prospective decision: **{decision}**. {'Canonical OOF recovery meets every frozen criterion, supporting the historical fitted-boundary/population-regime explanation over an inherent representational inability of this head.' if decision_a else 'The same architecture with the old normalization does not meet every frozen recovery criterion; the plan therefore points to a train-fold canonical-normalization diagnostic next, not an architecture change.'}

No threshold was calibrated and no trigger map, Stage-2 dataset, corrective search, or routing artifact was changed.
"""
    decision_text = f"""# Stage-1 Repair Decision

## {decision}

Evidence:

- Canonical refit OOF AUROC: {overall_refit['auroc']:.4f}; chance-difference 95% CI [{chance_overall['ci_low']:+.4f}, {chance_overall['ci_high']:+.4f}].
- Refit-minus-old paired AUROC difference: {paired_overall['observed_difference']:+.4f}, 95% CI [{paired_overall['ci_low']:+.4f}, {paired_overall['ci_high']:+.4f}].
- ChartQA canonical OOF AUROC: {dataset_refit['chartqa']['auroc']:.4f}.

{'The same architecture is adequate for canonical failure ranking. The next separately authorized phase would test source-balanced historical + canonical robust Stage-1 training; do not regenerate Stage-2 labels yet.' if decision_a else 'Arm A remained insufficient under the frozen decision rule. The smallest next experiment would keep the head/features fixed and recompute normalization from each canonical training fold (Arm B), but it is not authorized or executed here.'}

This phase stops here. No threshold selection or downstream execution was performed.
"""
    _atomic_bytes(OUTPUT_ROOT / "summaries/canonical_refit_summary.md", summary.encode("utf-8"))
    _atomic_bytes(OUTPUT_ROOT / "summaries/stage1_repair_decision.md", decision_text.encode("utf-8"))

    required = [
        "protocol.md", "frozen_protocol.json", "splits/fold_manifest.jsonl", "splits/fold_summary.csv",
        "training/checkpoint_manifest.json", "predictions/canonical_oof_scores.jsonl",
        "predictions/frozen_old_head_canonical_scores.jsonl", "predictions/full_canonical_head_historical_scores.jsonl",
        "metrics/canonical_oof_summary.csv", "metrics/canonical_dataset_breakdown.csv", "metrics/layerwise_auroc.csv",
        "metrics/layerwise_auprc.csv", "metrics/paired_bootstrap_difference.csv", "metrics/historical_cross_eval.csv",
        "metrics/score_distribution_summary.csv", "figures/canonical_old_vs_refit_roc.png",
        "figures/canonical_layerwise_auroc_recovery.png", "figures/canonical_layerwise_auprc_recovery.png",
        "figures/chartqa_score_distribution_old_vs_refit.png", "figures/textvqa_score_distribution_old_vs_refit.png",
        "figures/canonical_refit_cross_regime_comparison.png", "summaries/canonical_refit_summary.md",
        "summaries/stage1_repair_decision.md",
    ]
    artifact_rows = []
    for relative in required:
        path = OUTPUT_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"required artifact missing: {relative}")
        artifact_rows.append({"path": relative, "sha256": file_sha256(path), "bytes": path.stat().st_size})
    atomic_json(
        OUTPUT_ROOT / "artifact_manifest.json",
        {
            "schema_version": "stage1_canonical_refit_artifact_manifest_v1",
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "canonical_oof_records": len(oof),
            "canonical_oof_unique_uids": len({row["uid"] for row in oof}),
            "image_group_overlap": 0,
            "decision": decision,
            "artifacts": artifact_rows,
        },
    )
    print(json.dumps({"passed": True, "decision": decision, "canonical_oof_auroc": overall_refit["auroc"]}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("reproduction-check")
    fold = subparsers.add_parser("train-fold")
    fold.add_argument("--fold", type=int, required=True)
    subparsers.add_parser("train-full")
    subparsers.add_parser("aggregate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "reproduction-check":
        reproduction_check()
    elif args.command == "train-fold":
        train_fold(args.fold)
    elif args.command == "train-full":
        train_full()
    elif args.command == "aggregate":
        aggregate()


if __name__ == "__main__":
    main()
