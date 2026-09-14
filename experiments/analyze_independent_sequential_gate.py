#!/usr/bin/env python3
"""Evaluate a sequential gate built from the frozen Phase-48 probes."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import io
import json
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
    score_linear_probe,
    validate_checkpoint_provenance,
)
from dense_failure_stage1.sequential_gate import (  # noqa: E402
    empirical_alpha_grid,
    first_trigger_layers,
    fixed_layer_threshold,
    gate_metrics,
    region_name,
    select_operating_point,
    sweep_shared_alpha,
)
from experiments.analyze_layerwise_dense_failure_predictability import (  # noqa: E402
    checkpoint_expected as phase48_checkpoint_expected,
    load_frozen_contract as load_phase48_contract,
    load_layer_matrices,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/independent_layerwise_sequential_gate_v1.json"
CONTRACT_FILE = "frozen_protocol.json"
DATASETS = ("gqa", "chartqa", "textvqa")
BOUND_CODE_PATHS = (
    "configs/independent_layerwise_sequential_gate_v1.json",
    "dense_failure_stage1/sequential_gate.py",
    "dense_failure_stage1/layerwise_probe.py",
    "experiments/analyze_independent_sequential_gate.py",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
    )


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_bytes(
        path,
        "".join(
            json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ).encode(),
    )


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode())


def write_once_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite frozen artifact: {path}")
        return
    _atomic_bytes(path, payload)


def command_output(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result.stdout.strip()


def resolve_path(value: str) -> Path:
    path = (PROJECT_ROOT / value).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"path escapes project root: {value}")
    return path


def load_static_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "independent_layerwise_sequential_gate_config_v1":
        raise ValueError("unsupported sequential-gate config")
    if int(config["gate"]["layers"]) != 28 or int(config["world_size"]) != 4:
        raise ValueError("sequential gate requires 28 layers and four workers")
    if config["calibration"]["quantile_method"] != "higher":
        raise ValueError("sequential gate requires the frozen higher quantile")
    if config["gate"]["trigger_comparison"] != "strict_greater_than":
        raise ValueError("sequential gate requires strict threshold crossing")
    if tuple(config["single_layer_baselines"]) != (14, 21, 27):
        raise ValueError("fixed-layer baseline set differs")
    return config


def _split_rows(phase48_root: Path, split: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(phase48_root / "split_manifest.jsonl")
        if str(row["split"]) == split
    ]
    rows.sort(key=lambda row: str(row["uid"]))
    if len(rows) != 800 or len({str(row["uid"]) for row in rows}) != 800:
        raise ValueError(f"Phase-48 {split} split does not contain 800 unique UIDs")
    if Counter(bool(row["current_dense_wrong"]) for row in rows) != Counter({False: 400, True: 400}):
        raise ValueError(f"Phase-48 {split} label counts differ")
    return rows


def _protocol_markdown(contract: Mapping[str, Any]) -> str:
    grid = contract["calibration"]["alpha_values"]
    return "\n".join(
        [
            "# Independent Layer-Wise Sequential Gate Protocol",
            "",
            f"- Frozen contract: `{contract['contract_sha256']}`",
            f"- Reused Phase-48 contract: `{contract['provenance']['phase48_contract_sha256']}`",
            "- No predictor is retrained. The 28 frozen Phase-48 linear probabilities are rescored on its exact 800 validation and 800 test UIDs.",
            "- Test scores remain unopened until validation freezes every sequential and fixed-layer threshold.",
            "- At each layer, the threshold is the empirical higher `(1-alpha)` quantile of validation-correct scores. The gate uses strict `p_l > tau_l` and stops at the first crossing.",
            f"- Shared-alpha grid: `{len(grid)}` values from `{grid[0]:.8f}` through `{grid[-1]:.8f}`, covering every attainable 400-correct empirical tail breakpoint up to 10% plus exact 10%.",
            "- For each 99%/98%/95% target, select the largest alpha whose full sequential validation trajectory retains at least the target fraction of correct samples.",
            "- Fixed L14/L21/L27 thresholds are tie-safe strict-crossing thresholds calibrated independently on validation at the same preservation targets.",
            "- The same global sequential threshold vector is used across GQA, ChartQA, and TextVQA. No dataset-specific calibration is allowed.",
            "- A trigger is admission for possible intervention only; no treatment is executed.",
            "",
            "Finite-sample note: empirical higher quantiles may produce coarse steps or a zero-trigger conservative point. This limitation is reported directly rather than hidden with interpolated tail resolution.",
            "",
        ]
    )


def prepare(config_path: Path) -> None:
    config = load_static_config(config_path)
    output_root = resolve_path(config["output_root"])
    for directory in ("validation_workers", "test_workers", "figures"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    phase48, phase48_root = load_phase48_contract(
        resolve_path(config["sources"]["phase48_static_config"])
    )
    validation = _split_rows(phase48_root, "val")
    test = _split_rows(phase48_root, "test")
    expected_checkpoint = phase48_checkpoint_expected(phase48)
    checkpoint_hashes = {}
    for layer in range(28):
        path = phase48_root / "checkpoints" / f"layer_{layer:02d}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        validate_checkpoint_provenance(
            checkpoint, layer=layer, expected=expected_checkpoint
        )
        checkpoint_hashes[f"layer_{layer:02d}.pt"] = file_sha256(path)
    alpha_values = empirical_alpha_grid(
        sum(not bool(row["current_dense_wrong"]) for row in validation),
        maximum_alpha=float(config["calibration"]["maximum_alpha"]),
    )
    contract = json.loads(json.dumps(config))
    contract["schema_version"] = "independent_layerwise_sequential_gate_frozen_protocol_v1"
    contract["calibration"]["alpha_values"] = alpha_values
    git_status = command_output(["git", "status", "--porcelain=v1", "--untracked-files=all"])
    contract["provenance"] = {
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_branch": command_output(["git", "branch", "--show-current"]),
        "git_status_porcelain_at_freeze": git_status.splitlines() if git_status else [],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gpu_inventory": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines(),
        "static_config_sha256": file_sha256(config_path),
        "bound_code_sha256": {
            path: file_sha256(resolve_path(path)) for path in BOUND_CODE_PATHS
        },
        "source_sha256": {
            name: file_sha256(resolve_path(path))
            for name, path in config["sources"].items()
        },
        "phase48_contract_sha256": phase48["contract_sha256"],
        "phase48_checkpoint_sha256": checkpoint_hashes,
        "validation_uids_sha256": sha256(
            "\n".join(str(row["uid"]) for row in validation).encode()
        ).hexdigest(),
        "test_uids_sha256": sha256(
            "\n".join(str(row["uid"]) for row in test).encode()
        ).hexdigest(),
    }
    contract["contract_sha256"] = canonical_hash(contract)
    write_once_or_verify(
        output_root / CONTRACT_FILE,
        (json.dumps(contract, indent=2, sort_keys=True) + "\n").encode(),
    )
    write_once_or_verify(
        output_root / "protocol.md", _protocol_markdown(contract).encode()
    )
    atomic_json(
        output_root / "preparation_audit.json",
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "phase48_contract_sha256": phase48["contract_sha256"],
            "checkpoints": 28,
            "validation_records": len(validation),
            "test_records": len(test),
            "alpha_values": len(alpha_values),
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "contract_sha256": contract["contract_sha256"],
                "alpha_values": len(alpha_values),
            },
            sort_keys=True,
        )
    )


def load_contract(config_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    static = load_static_config(config_path)
    output_root = resolve_path(static["output_root"])
    contract = read_json(output_root / CONTRACT_FILE)
    if contract.get("schema_version") != "independent_layerwise_sequential_gate_frozen_protocol_v1":
        raise ValueError("unsupported frozen sequential-gate protocol")
    if contract.get("contract_sha256") != canonical_hash(contract):
        raise ValueError("frozen sequential-gate protocol hash differs")
    provenance = contract["provenance"]
    if file_sha256(config_path) != provenance["static_config_sha256"]:
        raise ValueError("sequential-gate config differs from freeze")
    for path, expected in provenance["bound_code_sha256"].items():
        if file_sha256(resolve_path(path)) != expected:
            raise ValueError(f"bound sequential-gate code differs: {path}")
    for name, expected in provenance["source_sha256"].items():
        if file_sha256(resolve_path(contract["sources"][name])) != expected:
            raise ValueError(f"sequential-gate source differs: {name}")
    phase48, phase48_root = load_phase48_contract(
        resolve_path(contract["sources"]["phase48_static_config"])
    )
    if phase48["contract_sha256"] != provenance["phase48_contract_sha256"]:
        raise ValueError("Phase-48 contract identity differs")
    return contract, output_root, phase48, phase48_root


def _require_gpu(rank: int, world_size: int) -> torch.device:
    if world_size != 4 or not 0 <= rank < 4:
        raise ValueError("score extraction requires four ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise RuntimeError("four CUDA GPUs are not visible")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # Four score workers load and compose disjoint layer matrices concurrently.
    # Leaving PyTorch at the host-wide default oversubscribes the CPU and can make
    # this deterministic preprocessing path effectively stall.
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return device


def score_worker(
    config_path: Path, *, split: str, rank: int, world_size: int
) -> None:
    contract, output_root, phase48, phase48_root = load_contract(config_path)
    if split not in {"val", "test"}:
        raise ValueError("score split must be val or test")
    decision_path = output_root / "selected_operating_points.json"
    if split == "test":
        decision = read_json(decision_path)
        if (
            decision.get("contract_sha256") != contract["contract_sha256"]
            or decision.get("test_evaluated") is not False
        ):
            raise ValueError("test scoring requires compatible pre-test operating points")
    device = _require_gpu(rank, world_size)
    assigned = list(range(rank, 28, world_size))
    rows, matrices = load_layer_matrices(
        phase48,
        phase48_root,
        layers=assigned,
        selected_splits={split},
    )
    expected = phase48_checkpoint_expected(phase48)
    results = []
    for layer in assigned:
        checkpoint_path = phase48_root / "checkpoints" / f"layer_{layer:02d}.pt"
        if file_sha256(checkpoint_path) != contract["provenance"]["phase48_checkpoint_sha256"][f"layer_{layer:02d}.pt"]:
            raise RuntimeError(f"Phase-48 checkpoint hash differs at layer {layer}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        validate_checkpoint_provenance(checkpoint, layer=layer, expected=expected)
        state = {
            "mean": checkpoint["normalization_mean"],
            "std": checkpoint["normalization_std"],
            "weight": checkpoint["weight"],
            "bias": checkpoint["bias"],
        }
        scores = score_linear_probe(matrices.pop(layer), state, device=device).numpy()
        results.extend(
            {
                "schema_version": "independent_sequential_gate_layer_score_v1",
                "uid": row["uid"],
                "dataset": row["dataset"],
                "current_dense_wrong": bool(row["current_dense_wrong"]),
                "layer": layer,
                "score": float(score),
                "worker_rank": rank,
            }
            for row, score in zip(rows, scores)
        )
    directory = "validation_workers" if split == "val" else "test_workers"
    worker_path = output_root / directory / f"rank{rank:02d}.jsonl"
    atomic_jsonl(worker_path, results)
    complete = {
        "passed": True,
        "split": split,
        "rank": rank,
        "world_size": world_size,
        "contract_sha256": contract["contract_sha256"],
        "layers": assigned,
        "records": len(results),
        "result_sha256": file_sha256(worker_path),
    }
    if split == "test":
        complete["selected_operating_points_sha256"] = file_sha256(decision_path)
    atomic_json(output_root / directory / f"rank{rank:02d}.complete.json", complete)
    print(json.dumps({"passed": True, "split": split, "rank": rank, "layers": assigned}))


def aggregate_scores(
    contract: Mapping[str, Any],
    output_root: Path,
    phase48_root: Path,
    *,
    split: str,
) -> list[dict[str, Any]]:
    directory = "validation_workers" if split == "val" else "test_workers"
    decision_path = output_root / "selected_operating_points.json"
    partial = []
    for rank in range(4):
        worker_path = output_root / directory / f"rank{rank:02d}.jsonl"
        complete = read_json(output_root / directory / f"rank{rank:02d}.complete.json")
        valid = (
            complete.get("passed") is True
            and complete.get("split") == split
            and complete.get("contract_sha256") == contract["contract_sha256"]
            and complete.get("result_sha256") == file_sha256(worker_path)
        )
        if split == "test":
            valid = valid and complete.get("selected_operating_points_sha256") == file_sha256(decision_path)
        if not valid:
            raise RuntimeError(f"{split} score worker {rank} is incomplete or incompatible")
        partial.extend(read_jsonl(worker_path))
    expected_rows = _split_rows(phase48_root, split)
    expected = {(str(row["uid"]), layer) for row in expected_rows for layer in range(28)}
    observed = Counter((str(row["uid"]), int(row["layer"])) for row in partial)
    if set(observed) != expected or any(observed[key] != 1 for key in expected):
        raise RuntimeError(f"{split} score workers do not cover every UID/layer exactly once")
    scores = {(str(row["uid"]), int(row["layer"])): float(row["score"]) for row in partial}
    wide = []
    for row in expected_rows:
        uid = str(row["uid"])
        values = [scores[(uid, layer)] for layer in range(28)]
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite {split} score trajectory for {uid}")
        wide.append(
            {
                "schema_version": "independent_sequential_gate_score_trajectory_v1",
                "uid": uid,
                "dataset": row["dataset"],
                "image_group_id": row["image_group_id"],
                "current_dense_wrong": bool(row["current_dense_wrong"]),
                **{f"p_{layer}": values[layer] for layer in range(28)},
            }
        )
    return wide


def score_arrays(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [[float(row[f"p_{layer}"]) for layer in range(28)] for row in rows],
        dtype=np.float64,
    )
    labels = np.asarray([int(bool(row["current_dense_wrong"])) for row in rows], dtype=np.int64)
    return matrix, labels


def _flat_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.item() if isinstance(value, np.generic) else value for key, value in metrics.items()}


def _result_row(
    *,
    target: float,
    alpha: float | None,
    metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row = {
        "target_preservation": float(target),
        "alpha": "" if alpha is None else float(alpha),
        "available": metrics is not None,
    }
    fields = (
        "records",
        "correct",
        "wrong",
        "correct_preservation",
        "wrong_detection_recall",
        "failure_precision",
        "trigger_rate",
        "median_first_trigger_layer",
        "mean_first_trigger_layer",
        "no_trigger_fraction",
        "correct_false_triggers",
        "wrong_detected",
    )
    for field in fields:
        row[field] = "" if metrics is None or metrics.get(field) is None else metrics[field]
    return row


def _dataset_gate_rows(
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    first: np.ndarray,
    *,
    split: str,
    target: float,
) -> list[dict[str, Any]]:
    output = []
    for dataset in DATASETS:
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row["dataset"] == dataset],
            dtype=np.int64,
        )
        metrics = gate_metrics(labels[indices], first[indices])
        output.append(
            {
                "split": split,
                "target_preservation": target,
                "dataset": dataset,
                **_flat_metrics(metrics),
            }
        )
    return output


def _distribution_rows(
    labels: np.ndarray,
    first: np.ndarray,
    *,
    split: str,
    target: float,
) -> list[dict[str, Any]]:
    output = []
    for class_name, class_value in (("correct", 0), ("wrong", 1)):
        selected = first[labels == class_value]
        denominator = len(selected)
        for layer in range(-1, 28):
            count = int((selected == layer).sum())
            output.append(
                {
                    "split": split,
                    "target_preservation": target,
                    "class": class_name,
                    "bucket_type": "layer",
                    "bucket": "never" if layer == -1 else str(layer),
                    "count": count,
                    "fraction_of_class": count / denominator,
                }
            )
        for region in ("early", "middle", "late", "never"):
            count = sum(region_name(int(layer)) == region for layer in selected)
            output.append(
                {
                    "split": split,
                    "target_preservation": target,
                    "class": class_name,
                    "bucket_type": "region",
                    "bucket": region,
                    "count": count,
                    "fraction_of_class": count / denominator,
                }
            )
    return output


def calibrate(config_path: Path) -> None:
    contract, output_root, _phase48, phase48_root = load_contract(config_path)
    if any((output_root / "test_workers" / f"rank{rank:02d}.complete.json").exists() for rank in range(4)):
        raise RuntimeError("test score artifacts exist before validation calibration freeze")
    rows = aggregate_scores(contract, output_root, phase48_root, split="val")
    score_path = output_root / "validation_scores.jsonl"
    atomic_jsonl(score_path, rows)
    matrix, labels = score_arrays(rows)
    sweep = sweep_shared_alpha(matrix, labels, contract["calibration"]["alpha_values"])
    sweep_rows = [
        {
            "alpha": row["alpha"],
            "correct_preservation": row["correct_preservation"],
            "wrong_detection_recall": row["wrong_detection_recall"],
            "failure_precision": row["failure_precision"],
            "trigger_rate": row["trigger_rate"],
            "median_first_trigger_layer": "" if row["median_first_trigger_layer"] is None else row["median_first_trigger_layer"],
            "mean_first_trigger_layer": "" if row["mean_first_trigger_layer"] is None else row["mean_first_trigger_layer"],
            "no_trigger_fraction": row["no_trigger_fraction"],
            "correct_false_triggers": row["correct_false_triggers"],
            "wrong_detected": row["wrong_detected"],
        }
        for row in sweep
    ]
    atomic_csv(output_root / "threshold_sweep.csv", sweep_rows)
    selected = {}
    validation_results = []
    thresholds_rows = []
    validation_dataset = []
    validation_distribution = []
    for target in contract["calibration"]["preservation_targets"]:
        key = str(int(round(float(target) * 100)))
        point = select_operating_point(sweep, target_preservation=float(target))
        if point is None:
            selected[key] = None
            validation_results.append(_result_row(target=float(target), alpha=None, metrics=None))
            continue
        thresholds = np.asarray(point["thresholds"], dtype=np.float64)
        first = first_trigger_layers(matrix, thresholds)
        metrics = gate_metrics(labels, first)
        selected[key] = {
            "target_preservation": float(target),
            "alpha": float(point["alpha"]),
            "thresholds": thresholds.tolist(),
            "validation_metrics": _flat_metrics(metrics),
        }
        validation_results.append(
            _result_row(target=float(target), alpha=float(point["alpha"]), metrics=metrics)
        )
        thresholds_rows.extend(
            {
                "target_preservation": float(target),
                "alpha": float(point["alpha"]),
                "layer": layer,
                "threshold": float(thresholds[layer]),
            }
            for layer in range(28)
        )
        validation_dataset.extend(
            _dataset_gate_rows(rows, labels, first, split="validation", target=float(target))
        )
        validation_distribution.extend(
            _distribution_rows(labels, first, split="validation", target=float(target))
        )

    fixed = {}
    fixed_validation_rows = []
    for layer in contract["single_layer_baselines"]:
        fixed[str(layer)] = {}
        for target in contract["calibration"]["preservation_targets"]:
            key = str(int(round(float(target) * 100)))
            threshold = fixed_layer_threshold(
                matrix[:, int(layer)], labels, target_preservation=float(target)
            )
            first = np.where(matrix[:, int(layer)] > threshold, int(layer), -1)
            metrics = gate_metrics(labels, first)
            fixed[str(layer)][key] = {
                "target_preservation": float(target),
                "threshold": threshold,
                "validation_metrics": _flat_metrics(metrics),
            }
            fixed_validation_rows.append(
                {
                    "split": "validation",
                    "target_preservation": float(target),
                    "gate": f"layer_{layer}",
                    "threshold": threshold,
                    **_flat_metrics(metrics),
                }
            )
    for row in validation_results:
        if row["available"]:
            fixed_validation_rows.append(
                {
                    "split": "validation",
                    "target_preservation": row["target_preservation"],
                    "gate": "sequential_layer_specific",
                    "threshold": "shared_alpha",
                    **{key: value for key, value in row.items() if key not in {"target_preservation", "alpha", "available"}},
                }
            )
    decision = {
        "schema_version": "independent_sequential_gate_operating_points_v1",
        "contract_sha256": contract["contract_sha256"],
        "validation_scores_sha256": file_sha256(score_path),
        "selection_rule": contract["calibration"]["operating_point_selection"],
        "quantile_method": contract["calibration"]["quantile_method"],
        "trigger_comparison": contract["gate"]["trigger_comparison"],
        "test_evaluated": False,
        "sequential": selected,
        "fixed_layer": fixed,
    }
    write_once_or_verify(
        output_root / "selected_operating_points.json",
        (json.dumps(decision, indent=2, sort_keys=True) + "\n").encode(),
    )
    atomic_csv(output_root / "layer_specific_thresholds.csv", thresholds_rows)
    atomic_csv(output_root / "validation_results.csv", validation_results)
    atomic_json(output_root / "validation_dataset_breakdown.json", {"rows": validation_dataset})
    atomic_json(output_root / "validation_trigger_distribution.json", {"rows": validation_distribution})
    atomic_json(output_root / "validation_single_layer_comparison.json", {"rows": fixed_validation_rows})
    print(
        json.dumps(
            {
                "passed": True,
                "validation_records": len(rows),
                "selected": {
                    key: None if value is None else {
                        "alpha": value["alpha"],
                        "correct_preservation": value["validation_metrics"]["correct_preservation"],
                        "wrong_detection_recall": value["validation_metrics"]["wrong_detection_recall"],
                    }
                    for key, value in selected.items()
                },
                "test_evaluated": False,
            },
            sort_keys=True,
        )
    )


def _trajectory_examples(
    rows: Sequence[Mapping[str, Any]], labels: np.ndarray, first: np.ndarray
) -> dict[str, int]:
    categories = {
        "correct + never trigger": lambda index: labels[index] == 0 and first[index] == -1,
        "correct + false trigger": lambda index: labels[index] == 0 and first[index] >= 0,
        "wrong + early trigger": lambda index: labels[index] == 1 and region_name(int(first[index])) == "early",
        "wrong + late trigger": lambda index: labels[index] == 1 and region_name(int(first[index])) == "late",
        "wrong + never trigger": lambda index: labels[index] == 1 and first[index] == -1,
    }
    selected = {}
    ordered = sorted(range(len(rows)), key=lambda index: str(rows[index]["uid"]))
    for name, predicate in categories.items():
        match = next((index for index in ordered if predicate(index)), None)
        if match is not None:
            selected[name] = match
    return selected


def _plot_results(
    output_root: Path,
    contract: Mapping[str, Any],
    validation_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
    validation_results: Sequence[Mapping[str, Any]],
    test_results: Sequence[Mapping[str, Any]],
    dataset_rows: Sequence[Mapping[str, Any]],
    distribution_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_root / "figures"
    sweep = list(csv.DictReader((output_root / "threshold_sweep.csv").open()))
    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    axis.plot(
        [float(row["correct_preservation"]) for row in sweep],
        [float(row["wrong_detection_recall"]) for row in sweep],
        marker="o",
        markersize=3,
        color="#1b9e77",
    )
    for row in validation_results:
        if row["available"] in {True, "True", "true", 1, "1"}:
            axis.scatter(
                [float(row["correct_preservation"])],
                [float(row["wrong_detection_recall"])],
                s=55,
                edgecolor="black",
                label=f"selected {float(row['target_preservation']):.0%}",
            )
    axis.set_xlabel("Validation sample-level correct preservation")
    axis.set_ylabel("Validation wrong detection recall")
    axis.set_xlim(0, 1.01)
    axis.set_ylim(0, 1.01)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figures / "preservation_vs_wrong_detection.png", dpi=180)
    plt.close(figure)

    target_key = "99"
    target = float(selected["sequential"][target_key]["target_preservation"])
    dist = [
        row
        for row in distribution_rows
        if row["split"] == "test"
        and float(row["target_preservation"]) == target
        and row["bucket_type"] == "layer"
        and row["bucket"] != "never"
    ]
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    layers = np.arange(28)
    width = 0.42
    for offset, class_name, color in ((-width / 2, "correct", "#d95f02"), (width / 2, "wrong", "#1b9e77")):
        values = [
            next(int(row["count"]) for row in dist if row["class"] == class_name and int(row["bucket"]) == layer)
            for layer in layers
        ]
        axis.bar(layers + offset, values, width, label=class_name, color=color)
    axis.set_xlabel("First trigger layer")
    axis.set_ylabel("Test samples")
    axis.set_xticks(range(0, 28, 2))
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figures / "trigger_layer_distribution.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharex=True)
    x = np.arange(len(DATASETS))
    width = 0.24
    colors = {0.99: "#1b9e77", 0.98: "#7570b3", 0.95: "#d95f02"}
    for offset_index, preservation in enumerate(contract["calibration"]["preservation_targets"]):
        chosen = [
            row
            for row in dataset_rows
            if row["split"] == "test" and float(row["target_preservation"]) == float(preservation)
        ]
        chosen_by_dataset = {row["dataset"]: row for row in chosen}
        offset = (offset_index - 1) * width
        axes[0].bar(
            x + offset,
            [float(chosen_by_dataset[dataset]["correct_preservation"]) for dataset in DATASETS],
            width,
            label=f"{float(preservation):.0%} target",
            color=colors[float(preservation)],
        )
        axes[1].bar(
            x + offset,
            [float(chosen_by_dataset[dataset]["wrong_detection_recall"]) for dataset in DATASETS],
            width,
            color=colors[float(preservation)],
        )
    axes[0].set_ylabel("Correct preservation")
    axes[1].set_ylabel("Wrong detection recall")
    for axis in axes:
        axis.set_xticks(x, DATASETS)
        axis.set_ylim(0, 1)
    axes[0].legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(figures / "dataset_preservation_recall.png", dpi=180)
    plt.close(figure)

    test_comparison = [row for row in comparison_rows if row["split"] == "test"]
    gates = ["layer_14", "layer_21", "layer_27", "sequential_layer_specific"]
    x = np.arange(len(gates))
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    for offset_index, preservation in enumerate(contract["calibration"]["preservation_targets"]):
        by_gate = {
            row["gate"]: row
            for row in test_comparison
            if float(row["target_preservation"]) == float(preservation)
        }
        axis.bar(
            x + (offset_index - 1) * width,
            [float(by_gate[gate]["wrong_detection_recall"]) for gate in gates],
            width,
            label=f"validation {float(preservation):.0%}",
        )
    axis.set_xticks(x, ["L14", "L21", "L27", "Sequential"])
    axis.set_ylabel("Test wrong detection recall")
    axis.set_ylim(0, 1)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figures / "sequential_vs_single_layer.png", dpi=180)
    plt.close(figure)

    example_key = "95" if selected["sequential"].get("95") is not None else target_key
    thresholds = np.asarray(selected["sequential"][example_key]["thresholds"], dtype=np.float64)
    matrix, labels = score_arrays(test_rows)
    first = first_trigger_layers(matrix, thresholds)
    examples = _trajectory_examples(test_rows, labels, first)
    figure, axes = plt.subplots(len(examples), 1, figsize=(9.0, max(3.0, 2.2 * len(examples))), sharex=True)
    if len(examples) == 1:
        axes = [axes]
    for axis, (name, index) in zip(axes, examples.items()):
        axis.plot(range(28), matrix[index], marker="o", markersize=2.5, label="failure score")
        axis.plot(range(28), thresholds, linestyle="--", color="#d95f02", label="layer threshold")
        axis.set_ylabel("score")
        axis.set_title(f"{name}: {test_rows[index]['uid']}", fontsize=9)
        axis.set_ylim(-0.02, 1.02)
    axes[-1].set_xlabel("Layer")
    axes[0].legend(loc="best")
    figure.tight_layout()
    figure.savefig(figures / "score_trajectory_examples.png", dpi=180)
    plt.close(figure)


def _decision_summary(
    contract: Mapping[str, Any],
    validation_results: Sequence[Mapping[str, Any]],
    test_results: Sequence[Mapping[str, Any]],
    dataset_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    distribution_rows: Sequence[Mapping[str, Any]],
) -> str:
    test_by_target = {int(round(float(row["target_preservation"]) * 100)): row for row in test_results}
    validation_by_target = {int(round(float(row["target_preservation"]) * 100)): row for row in validation_results}
    comparison_test = [row for row in comparison_rows if row["split"] == "test"]
    meaningful_floor = float(contract["interpretation"]["meaningful_wrong_recall_floor"])
    latest_useful = int(contract["interpretation"]["useful_remaining_depth_latest_median_trigger"])
    main = test_by_target[99]
    main_median = None if main["median_first_trigger_layer"] == "" else float(main["median_first_trigger_layer"])
    useful = (
        float(main["correct_preservation"]) >= 0.99
        and float(main["wrong_detection_recall"]) >= meaningful_floor
        and main_median is not None
        and main_median <= latest_useful
    )
    gains = {}
    for target in (99, 98, 95):
        selected = [row for row in comparison_test if int(round(float(row["target_preservation"]) * 100)) == target]
        sequential = next(row for row in selected if row["gate"] == "sequential_layer_specific")
        best_single = max(
            (row for row in selected if row["gate"] != "sequential_layer_specific"),
            key=lambda row: float(row["wrong_detection_recall"]),
        )
        gains[target] = (sequential, best_single, float(sequential["wrong_detection_recall"]) - float(best_single["wrong_detection_recall"]))
    dataset_test = [row for row in dataset_rows if row["split"] == "test"]
    mismatch = False
    spreads = {}
    for target in (99, 98, 95):
        rows = [row for row in dataset_test if int(round(float(row["target_preservation"]) * 100)) == target]
        preservation_spread = max(float(row["correct_preservation"]) for row in rows) - min(float(row["correct_preservation"]) for row in rows)
        recall_spread = max(float(row["wrong_detection_recall"]) for row in rows) - min(float(row["wrong_detection_recall"]) for row in rows)
        spreads[target] = (preservation_spread, recall_spread)
        mismatch = mismatch or max(preservation_spread, recall_spread) >= float(contract["interpretation"]["dataset_metric_spread_warning"])
    lines = [
        "# Independent Layer-Wise Sequential Gate Decision Summary",
        "",
        f"Frozen protocol: `{contract['contract_sha256']}`.",
        "",
        "| Validation target | Alpha | Validation preservation | Validation wrong recall | Test preservation | Test wrong recall | Test precision | Median test trigger |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in (99, 98, 95):
        validation = validation_by_target[target]
        test = test_by_target[target]
        lines.append(
            f"| {target}% | {float(validation['alpha']):.8f} | {float(validation['correct_preservation']):.4f} | "
            f"{float(validation['wrong_detection_recall']):.4f} | {float(test['correct_preservation']):.4f} | "
            f"{float(test['wrong_detection_recall']):.4f} | {float(test['failure_precision']):.4f} | "
            f"{test['median_first_trigger_layer']} |"
        )
    lines.extend(
        [
            "",
            "## Q1. Can the 28 probes form a conservative sequential gate?",
            "",
            (
                "Yes under the frozen 99% criterion: the test policy preserves at least 99% of correct samples, detects at least 5% of wrong samples, and has a median trigger by layer 18."
                if useful
                else "Not under the complete frozen 99% usefulness criterion. At least one of preservation, nontrivial wrong recall, or sufficiently early median triggering fails."
            ),
            "",
            "## Q2. Detection at the three preservation targets",
            "",
            "The table above gives both validation-selected and untouched-test results. Counts are available in `validation_results.csv` and `test_results.csv`.",
            "",
            "## Q3. Where do failures first trigger?",
            "",
        ]
    )
    for target in (99, 98, 95):
        regions = [
            row
            for row in distribution_rows
            if row["split"] == "test"
            and int(round(float(row["target_preservation"]) * 100)) == target
            and row["class"] == "wrong"
            and row["bucket_type"] == "region"
        ]
        values = {row["bucket"]: int(row["count"]) for row in regions}
        lines.append(
            f"- {target}% target: early `{values['early']}`, middle `{values['middle']}`, late `{values['late']}`, never `{values['never']}` wrong samples."
        )
    lines.extend(
        [
            "",
            "## Q4. Does sequential gating outperform one fixed strong layer?",
            "",
            "| Target | Sequential recall | Best fixed gate | Best fixed recall | Sequential gain |",
            "|---:|---:|---|---:|---:|",
        ]
    )
    for target in (99, 98, 95):
        sequential, fixed, gain = gains[target]
        lines.append(
            f"| {target}% | {float(sequential['wrong_detection_recall']):.4f} | {fixed['gate']} | "
            f"{float(fixed['wrong_detection_recall']):.4f} | {gain:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Q5. Is the common calibration rule consistent across datasets?",
            "",
        ]
    )
    for target in (99, 98, 95):
        preservation_spread, recall_spread = spreads[target]
        lines.append(
            f"- {target}% target: dataset preservation spread `{preservation_spread:.4f}`, wrong-recall spread `{recall_spread:.4f}`."
        )
    lines.extend(
        [
            "",
            (
                "The frozen 0.10 spread warning is reached, so the shared quantile rule has a material dataset calibration mismatch. No dataset-specific threshold was added."
                if mismatch
                else "Neither dataset preservation nor recall spread reaches the frozen 0.10 warning threshold."
            ),
            "",
            "## Q6. Is the independent gate sufficient?",
            "",
            (
                "The independent sequential gate is already a defensible Stage-1 admission rule under the frozen criteria. A more complex shared/global-budget model is not justified by necessity in this phase."
                if useful and all(gain > float(contract["interpretation"]["little_sequential_gain_ceiling"]) for _, _, gain in gains.values()) and not mismatch
                else "The independent gate is not sufficient as a robust final design under the frozen criteria. The observed limitation motivates—but does not authorize—a separately planned shared predictor or global risk-budget formulation."
            ),
            "",
            "## Scope limits",
            "",
            "- This is an in-domain Phase-48 split analysis, not evidence that calibration transfers OOD.",
            "- Test was evaluated once after all thresholds were frozen from validation.",
            "- Triggering does not demonstrate that any treatment would improve the answer.",
            "- No shared predictor, global risk-budget model, or intervention was trained or executed.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(config_path: Path) -> None:
    contract, output_root, _phase48, phase48_root = load_contract(config_path)
    decision_path = output_root / "selected_operating_points.json"
    selected = read_json(decision_path)
    if selected.get("contract_sha256") != contract["contract_sha256"] or selected.get("test_evaluated") is not False:
        raise ValueError("operating points are not a compatible pre-test freeze")
    validation_rows = read_jsonl(output_root / "validation_scores.jsonl")
    if selected.get("validation_scores_sha256") != file_sha256(output_root / "validation_scores.jsonl"):
        raise RuntimeError("validation scores differ from operating-point freeze")
    test_rows = aggregate_scores(contract, output_root, phase48_root, split="test")
    test_score_path = output_root / "test_scores.jsonl"
    atomic_jsonl(test_score_path, test_rows)
    test_matrix, test_labels = score_arrays(test_rows)
    validation_results = list(csv.DictReader((output_root / "validation_results.csv").open()))
    test_results = []
    dataset_rows = list(read_json(output_root / "validation_dataset_breakdown.json")["rows"])
    distribution_rows = list(read_json(output_root / "validation_trigger_distribution.json")["rows"])
    comparison_rows = list(read_json(output_root / "validation_single_layer_comparison.json")["rows"])
    for target in contract["calibration"]["preservation_targets"]:
        key = str(int(round(float(target) * 100)))
        point = selected["sequential"].get(key)
        if point is None:
            test_results.append(_result_row(target=float(target), alpha=None, metrics=None))
            continue
        thresholds = np.asarray(point["thresholds"], dtype=np.float64)
        first = first_trigger_layers(test_matrix, thresholds)
        metrics = gate_metrics(test_labels, first)
        test_results.append(_result_row(target=float(target), alpha=float(point["alpha"]), metrics=metrics))
        dataset_rows.extend(
            _dataset_gate_rows(test_rows, test_labels, first, split="test", target=float(target))
        )
        distribution_rows.extend(
            _distribution_rows(test_labels, first, split="test", target=float(target))
        )
        comparison_rows.append(
            {
                "split": "test",
                "target_preservation": float(target),
                "gate": "sequential_layer_specific",
                "threshold": "shared_alpha",
                **_flat_metrics(metrics),
            }
        )
        for layer in contract["single_layer_baselines"]:
            threshold = float(selected["fixed_layer"][str(layer)][key]["threshold"])
            fixed_first = np.where(test_matrix[:, int(layer)] > threshold, int(layer), -1)
            fixed_metrics = gate_metrics(test_labels, fixed_first)
            comparison_rows.append(
                {
                    "split": "test",
                    "target_preservation": float(target),
                    "gate": f"layer_{layer}",
                    "threshold": threshold,
                    **_flat_metrics(fixed_metrics),
                }
            )
    atomic_csv(output_root / "test_results.csv", test_results)
    atomic_csv(output_root / "dataset_breakdown.csv", dataset_rows)
    atomic_csv(output_root / "trigger_layer_distribution.csv", distribution_rows)
    atomic_csv(output_root / "single_layer_comparison.csv", comparison_rows)
    _plot_results(
        output_root,
        contract,
        validation_rows,
        test_rows,
        selected,
        validation_results,
        test_results,
        dataset_rows,
        distribution_rows,
        comparison_rows,
    )
    summary = _decision_summary(
        contract,
        validation_results,
        test_results,
        dataset_rows,
        comparison_rows,
        distribution_rows,
    )
    write_once_or_verify(output_root / "decision_summary.md", summary.encode())
    required = [
        "protocol.md",
        "validation_scores.jsonl",
        "test_scores.jsonl",
        "threshold_sweep.csv",
        "selected_operating_points.json",
        "layer_specific_thresholds.csv",
        "validation_results.csv",
        "test_results.csv",
        "dataset_breakdown.csv",
        "trigger_layer_distribution.csv",
        "single_layer_comparison.csv",
        "figures/preservation_vs_wrong_detection.png",
        "figures/trigger_layer_distribution.png",
        "figures/dataset_preservation_recall.png",
        "figures/sequential_vs_single_layer.png",
        "figures/score_trajectory_examples.png",
        "decision_summary.md",
    ]
    missing = [path for path in required if not (output_root / path).is_file()]
    if missing:
        raise RuntimeError(f"required sequential-gate artifacts are missing: {missing}")
    manifest = {
        "schema_version": "independent_sequential_gate_artifact_manifest_v1",
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        "selected_operating_points_sha256": file_sha256(decision_path),
        "test_evaluated_once_after_validation_freeze": True,
        "required_files": {path: file_sha256(output_root / path) for path in required},
    }
    atomic_json(output_root / "artifact_manifest.json", manifest)
    print(json.dumps({"passed": True, "contract_sha256": contract["contract_sha256"], "required_files": len(required)}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    for name in ("validation-score-worker", "test-score-worker"):
        worker = subparsers.add_parser(name)
        worker.add_argument("--rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
        worker.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    subparsers.add_parser("calibrate")
    subparsers.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_relative_to(PROJECT_ROOT):
        raise ValueError("config must lie inside the project root")
    if args.command == "prepare":
        prepare(config_path)
    elif args.command == "validation-score-worker":
        score_worker(config_path, split="val", rank=args.rank, world_size=args.world_size)
    elif args.command == "calibrate":
        calibrate(config_path)
    elif args.command == "test-score-worker":
        score_worker(config_path, split="test", rank=args.rank, world_size=args.world_size)
    elif args.command == "finalize":
        finalize(config_path)
    else:
        raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
