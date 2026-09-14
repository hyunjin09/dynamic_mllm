#!/usr/bin/env python3
"""Build robust Stage-1 trigger maps and audit prior Stage-2 label compatibility."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
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
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from binary_policy.executor import (  # noqa: E402
    capture_four_action_suffix_from_full_baseline,
    capture_full_baseline,
)
from binary_policy.executor.inputs import build_binary_inputs  # noqa: E402
from dense_failure_stage1.operating_point_compatibility import (  # noqa: E402
    build_trigger_rows,
    compatibility_category,
    select_retained_operating_points,
    structural_route_status,
    transition_category,
)
from dense_failure_stage1.threshold_calibration import (  # noqa: E402
    first_trigger_layers,
    most_permissive_at_preservation,
    operating_metrics,
)
from dense_failure_stage1.runtime import (  # noqa: E402
    build_dense_inputs,
    configure_dense_determinism,
)
from dense_failure_stage2.corrective_search import trigger_depth_bin  # noqa: E402
from experiments.run_stage1_all_source_robustness import (  # noqa: E402
    BLOCKS,
    LAYERS,
    SOURCES as PHASE62_SOURCES,
    _gpu as stage1_gpu,
    _load_all_features,
    _model as stage1_model,
    _normalization as stage1_normalization,
    _prediction_rows,
    _score as stage1_score,
    file_sha256,
    load_contract as load_phase62_contract,
)
from experiments.run_trigger_conditioned_corrective_search_pilot import (  # noqa: E402
    _generate_output,
    _load_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/robust_stage1_operating_points_compatibility_v1.json"
DATASETS = ("gqa", "chartqa", "textvqa")
SOURCES = ("historical", "canonical")
DEPTH_BINS = ("L0", "L1-8", "L9-18", "L19-27")
BOUND_CODE = (
    "configs/robust_stage1_operating_points_compatibility_v1.json",
    "dense_failure_stage1/operating_point_compatibility.py",
    "dense_failure_stage1/threshold_calibration.py",
    "experiments/analyze_robust_stage1_operating_points_compatibility.py",
    "experiments/run_stage1_all_source_robustness.py",
    "dense_failure_stage1/all_source_robustness.py",
    "dense_failure_stage1/shared_global_gate.py",
    "dense_failure_stage1/layerwise_probe.py",
    "dense_failure_stage1/runtime.py",
    "dense_failure_stage1/lmms_scoring.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/model.py",
    "experiments/run_trigger_conditioned_corrective_search_pilot.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    allowed = (PROJECT_ROOT.resolve(), Path("/mnt/hyemin").resolve())
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed):
        raise ValueError(f"path escapes allowed roots: {value}")
    return resolved


def canonical_hash(value: Mapping[str, Any], *, excluded: Sequence[str] = ("contract_sha256",)) -> str:
    payload = {key: item for key, item in value.items() if key not in excluded}
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
                raise ValueError(f"expected object at {path}:{line_number}")
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
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_bytes(
        path,
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows).encode(),
    )


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode())


def write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite frozen artifact: {path}")
        return
    _atomic_bytes(path, payload)


def write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    write_once(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def command_output(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result.stdout.strip()


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "robust_stage1_operating_points_compatibility_config_v1":
        raise ValueError("unsupported compatibility config")
    if int(config["world_size"]) != 4 or int(config["trigger"]["layers"]) != 28:
        raise ValueError("world size/layer count differs")
    if config["trigger"]["comparison"] != "strict_greater_than":
        raise ValueError("trigger comparison must remain strict")
    if config["thresholds"]["retention_default"] != ["P98", "P95", "P90"]:
        raise ValueError("prospective retention default differs")
    return config


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "cuda_runtime": torch.version.cuda,
    }


def _manifest_files(manifest: Mapping[str, Any]) -> dict[str, str]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        artifacts = manifest.get("artifacts")
        if isinstance(artifacts, list):
            files = {str(row["path"]): str(row["sha256"]) for row in artifacts}
    if not isinstance(files, dict):
        raise ValueError("unsupported artifact manifest file table")
    return {str(key): str(value) for key, value in files.items()}


def _verify_manifest_file(root: Path, manifest: Mapping[str, Any], path: Path) -> None:
    relative = str(path.resolve().relative_to(root.resolve()))
    expected = _manifest_files(manifest).get(relative)
    if expected is None or not path.is_file() or file_sha256(path) != expected:
        raise RuntimeError(f"artifact hash mismatch: {path}")


def _model_snapshot_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    historical_contract = read_json(
        resolve_path("analysis/dense_failure_stage2/full_corrective_labels/frozen_protocol.json")
    )
    expected = {str(key): str(value) for key, value in historical_contract["model_snapshot_sha256"].items()}
    root = resolve_path(config["model"]["snapshot_path"])
    actual_files = sorted(path.name for path in root.iterdir() if path.is_file())
    if actual_files != sorted(expected):
        raise RuntimeError("model snapshot inventory differs from prior replay contract")
    for name, digest in expected.items():
        if file_sha256(root / name) != digest:
            raise RuntimeError(f"model snapshot file differs: {name}")
    return expected


def prepare(config_path: Path) -> None:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    sources = {name: resolve_path(value) for name, value in config["sources"].items()}
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing sources: {missing}")

    phase63_manifest = read_json(sources["phase63_artifact_manifest"])
    phase63_gate = read_json(sources["phase63_gate"])
    phase63_gate_hash = canonical_hash(
        phase63_gate,
        excluded=("contract_sha256", "head_identity_sha256", "gate_sha256"),
    )
    if not phase63_manifest.get("passed") or phase63_gate_hash != phase63_gate.get("gate_sha256"):
        raise RuntimeError("Phase-63 gate/manifest is invalid")
    _verify_manifest_file(sources["phase63_artifact_manifest"].parent, phase63_manifest, sources["phase63_gate"])
    if float(phase63_gate["global_threshold"]) != float(config["thresholds"]["p98_anchor"]):
        raise RuntimeError("P98 anchor differs from Phase-63 gate")

    phase62_manifest = read_json(sources["phase62_artifact_manifest"])
    phase62_protocol = read_json(sources["phase62_protocol"])
    if not phase62_manifest.get("passed") or phase62_manifest["contract_sha256"] != phase63_gate["phase62_contract_sha256"]:
        raise RuntimeError("Phase-62 source contract differs")
    checkpoint_manifest = read_json(sources["phase62_checkpoint_manifest"])
    if checkpoint_manifest["checkpoints"] != phase63_gate["checkpoints"]:
        raise RuntimeError("Phase-62 checkpoint identity differs from frozen gate")
    for checkpoint in phase63_gate["checkpoints"]:
        path = resolve_path(checkpoint["path"])
        if file_sha256(path) != checkpoint["sha256"]:
            raise RuntimeError(f"Stage-1 checkpoint differs: {path}")
    norm = resolve_path(phase63_gate["normalization"]["path"])
    if file_sha256(norm) != phase63_gate["normalization"]["sha256"]:
        raise RuntimeError("Stage-1 normalization differs")

    historical_trigger_manifest = read_json(sources["historical_trigger_artifact_manifest"])
    historical_label_manifest = read_json(sources["historical_label_artifact_manifest"])
    canonical_label_manifest = read_json(sources["canonical_label_artifact_manifest"])
    if not all(
        manifest.get("passed")
        for manifest in (historical_trigger_manifest, historical_label_manifest, canonical_label_manifest)
    ):
        raise RuntimeError("an upstream trigger/label manifest is not passing")
    checks = (
        ("historical_old_trigger_map", sources["historical_trigger_artifact_manifest"].parent, historical_trigger_manifest),
        ("historical_single_labels", sources["historical_label_artifact_manifest"].parent, historical_label_manifest),
        ("historical_mcts_labels", sources["historical_label_artifact_manifest"].parent, historical_label_manifest),
        ("historical_unresolved", sources["historical_label_artifact_manifest"].parent, historical_label_manifest),
        ("historical_single_routes", sources["historical_label_artifact_manifest"].parent, historical_label_manifest),
        ("historical_mcts_routes", sources["historical_label_artifact_manifest"].parent, historical_label_manifest),
        ("canonical_single_labels", sources["canonical_label_artifact_manifest"].parent, canonical_label_manifest),
        ("canonical_mcts_labels", sources["canonical_label_artifact_manifest"].parent, canonical_label_manifest),
        ("canonical_unresolved", sources["canonical_label_artifact_manifest"].parent, canonical_label_manifest),
        ("canonical_single_routes", sources["canonical_label_artifact_manifest"].parent, canonical_label_manifest),
        ("canonical_mcts_routes", sources["canonical_label_artifact_manifest"].parent, canonical_label_manifest),
    )
    for name, root, manifest in checks:
        _verify_manifest_file(root, manifest, sources[name])

    historical_split = [row for row in read_jsonl(sources["historical_split"]) if row["split"] == "train"]
    canonical = read_jsonl(sources["canonical_fold_manifest"])
    old_h = read_jsonl(sources["historical_old_trigger_map"])
    old_c = read_jsonl(sources["canonical_old_trigger_map"])
    if len(historical_split) != 6399 or len(canonical) != 4000 or len(old_h) != 6399 or len(old_c) != 4000:
        raise RuntimeError("train population count differs")
    if {row["uid"] for row in historical_split} != {row["uid"] for row in old_h}:
        raise RuntimeError("Historical train/old-trigger identities differ")
    if {row["uid"] for row in canonical} != {row["uid"] for row in old_c}:
        raise RuntimeError("Canonical train/old-trigger identities differ")

    model_hashes = _model_snapshot_hashes(config)
    source_hashes = {name: file_sha256(path) for name, path in sources.items()}
    protocol = f"""# Robust Stage-1 operating-point and label-compatibility protocol

- Frozen Stage-1 system: Phase-63 five-checkpoint ALL-source system, per-layer probability mean; no singleton checkpoint exists or is trained here.
- Strict trigger rule: first layer 0-27 with `score > tau`.
- P98 is anchored exactly at `{config['thresholds']['p98_anchor']:.17g}`; P99/P97/P95/P90 are least-permissive discrete calibration breakpoints satisfying both-source preservation.
- Default retained set P98/P95/P90 is accepted only if all three are fold-stable, thresholds are distinct, and each relaxation adds at least {config['thresholds']['minimum_incremental_pooled_wrong_recall']:.0%} pooled calibration W recall.
- Fold-stable means threshold IQR <= {config['thresholds']['maximum_fold_iqr']} and range <= {config['thresholds']['maximum_fold_range']} across five existing cross-fit calibrations.
- Train maps are descriptive in-sample maps: 6,399 Historical train plus 4,000 Canonical train rows scored by the exact five-head mean.
- Existing single route structural compatibility: new trigger <= intervention layer. Existing MCTS route structural compatibility: new trigger <= first non-FULL layer.
- Every structurally compatible route/handoff pair is rerun with exact Qwen suffix execution and current LMMS scoring; current correct output plus exact stored-token parity is required for reuse.
- Stage-2 outcomes are not used to choose thresholds. No new single/MCTS search or Stage-2 training is permitted.
"""
    write_once(output_root / "protocol.md", protocol.encode())
    contract = {
        "schema_version": "robust_stage1_operating_points_compatibility_contract_v1",
        "static_config": config,
        "git": {
            "commit": command_output(("git", "rev-parse", "HEAD")),
            "branch": command_output(("git", "branch", "--show-current")),
            "worktree_status_at_freeze": command_output(("git", "status", "--short")),
        },
        "runtime": _runtime_metadata(),
        "source_sha256": source_hashes,
        "bound_code_sha256": {path: file_sha256(resolve_path(path)) for path in BOUND_CODE},
        "model_snapshot_sha256": model_hashes,
        "phase63_gate_sha256": phase63_gate["gate_sha256"],
        "phase62_contract_sha256": phase62_protocol["contract_sha256"],
        "population": {
            "historical_train": 6399,
            "canonical_train": 4000,
            "historical_existing_single_routes": 7628,
            "historical_existing_mcts_routes": 442,
            "canonical_existing_single_routes": 950,
            "canonical_existing_mcts_routes": 66,
        },
        "protocol_sha256": file_sha256(output_root / "protocol.md"),
    }
    contract["contract_sha256"] = canonical_hash(contract)
    write_once_json(output_root / "frozen_protocol.json", contract)
    print(json.dumps({"prepared": True, "contract_sha256": contract["contract_sha256"]}))


def load_contract(config_path: Path, *, verify_model: bool = False) -> tuple[dict[str, Any], Path]:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if canonical_hash(contract) != contract.get("contract_sha256") or contract["static_config"] != config:
        raise RuntimeError("frozen compatibility contract is invalid")
    if command_output(("git", "rev-parse", "HEAD")) != contract["git"]["commit"]:
        raise RuntimeError("git commit differs from frozen contract")
    if _runtime_metadata() != contract["runtime"]:
        raise RuntimeError("runtime differs from frozen contract")
    for name, expected in contract["source_sha256"].items():
        if file_sha256(resolve_path(config["sources"][name])) != expected:
            raise RuntimeError(f"source differs after freeze: {name}")
    for path, expected in contract["bound_code_sha256"].items():
        if file_sha256(resolve_path(path)) != expected:
            raise RuntimeError(f"bound code differs after freeze: {path}")
    if file_sha256(output_root / "protocol.md") != contract["protocol_sha256"]:
        raise RuntimeError("protocol differs after freeze")
    if verify_model:
        root = resolve_path(config["model"]["snapshot_path"])
        for name, expected in contract["model_snapshot_sha256"].items():
            if file_sha256(root / name) != expected:
                raise RuntimeError(f"model snapshot differs: {name}")
    return contract, output_root


def _point_from_sweep(rows: Sequence[Mapping[str, Any]], target: float) -> dict[str, Any]:
    parsed = [{key: (float(value) if key != "threshold" and _is_float(value) else value) for key, value in row.items()} for row in rows]
    for row in parsed:
        row["threshold"] = float(row["threshold"])
        row["worst_source_c_preservation"] = float(row["worst_source_c_preservation"])
    return most_permissive_at_preservation(parsed, target)


def _is_float(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def thresholds(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    config = contract["static_config"]
    source = resolve_path(config["sources"]["phase63_frontier"])
    frontier = read_csv(source)
    atomic_csv(output_root / "thresholds/frontier.csv", frontier)
    named = []
    fold_rows = []
    phase63_stability = read_csv(resolve_path(config["sources"]["phase63_fold_stability"]))
    p98_folds = {int(row["fold"]): float(row["threshold"]) for row in phase63_stability}
    for target in config["thresholds"]["named_preservation_targets"]:
        name = f"P{int(round(100 * float(target)))}"
        if name == "P98":
            tau = float(config["thresholds"]["p98_anchor"])
            selected = next(row for row in frontier if float(row["threshold"]) == tau)
        else:
            selected = _point_from_sweep(frontier, float(target))
            tau = float(selected["threshold"])
        folds = []
        for fold in range(5):
            if name == "P98":
                fold_tau = p98_folds[fold]
            else:
                sweep = read_csv(output_root.parent / "all_source_threshold_calibration" / "crossfit" / f"fold_{fold}_threshold_sweep.csv")
                fold_tau = float(_point_from_sweep(sweep, float(target))["threshold"])
            folds.append(fold_tau)
            fold_rows.append({"operating_point": name, "preservation_target": target, "fold": fold, "threshold": fold_tau})
        values = np.asarray(folds)
        iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        span = float(values.max() - values.min())
        stable = bool(
            iqr <= float(config["thresholds"]["maximum_fold_iqr"])
            and span <= float(config["thresholds"]["maximum_fold_range"])
        )
        named.append(
            {
                "operating_point": name,
                "preservation_target": target,
                "threshold": tau,
                "historical_c_preservation": float(selected["historical_c_preservation"]),
                "canonical_c_preservation": float(selected["canonical_c_preservation"]),
                "historical_w_recall": float(selected["historical_w_recall"]),
                "canonical_w_recall": float(selected["canonical_w_recall"]),
                "pooled_w_recall": float(selected["pooled_w_recall"]),
                "historical_trigger_precision": float(selected["historical_trigger_precision"]),
                "canonical_trigger_precision": float(selected["canonical_trigger_precision"]),
                "pooled_trigger_precision": float(selected["pooled_trigger_precision"]),
                "median_first_trigger_layer": float(selected["pooled_median_first_trigger_layer"]),
                "worst_source_c_preservation": float(selected["worst_source_c_preservation"]),
                "worst_source_w_recall": float(selected["worst_source_w_recall"]),
                "fold_threshold_median": float(np.median(values)),
                "fold_threshold_min": float(values.min()),
                "fold_threshold_max": float(values.max()),
                "fold_threshold_iqr": iqr,
                "fold_threshold_range": span,
                "fold_stable": stable,
            }
        )
    retained, reason = select_retained_operating_points(
        named,
        default_names=config["thresholds"]["retention_default"],
        minimum_recall_increment=float(config["thresholds"]["minimum_incremental_pooled_wrong_recall"]),
        maximum_points=int(config["thresholds"]["maximum_retained_points"]),
    )
    for row in named:
        row["retained"] = row["operating_point"] in retained
        row["retention_reason"] = reason if row["retained"] else "not_retained"
    atomic_csv(output_root / "thresholds/named_operating_points.csv", named)
    atomic_csv(output_root / "thresholds/fold_stability.csv", fold_rows)
    binding = {
        "schema_version": "robust_stage1_operating_point_binding_v1",
        "contract_sha256": contract["contract_sha256"],
        "retained_operating_points": retained,
        "retention_reason": reason,
        "named_operating_points_sha256": file_sha256(output_root / "thresholds/named_operating_points.csv"),
        "fold_stability_sha256": file_sha256(output_root / "thresholds/fold_stability.csv"),
        "frontier_sha256": file_sha256(output_root / "thresholds/frontier.csv"),
    }
    atomic_json(output_root / "work/threshold_binding.json", binding)
    print(json.dumps({"retained": retained, "reason": reason}))


def _load_threshold_binding(contract: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    binding = read_json(output_root / "work/threshold_binding.json")
    if binding["contract_sha256"] != contract["contract_sha256"]:
        raise RuntimeError("threshold binding contract mismatch")
    for key, relative in (
        ("named_operating_points_sha256", "thresholds/named_operating_points.csv"),
        ("fold_stability_sha256", "thresholds/fold_stability.csv"),
        ("frontier_sha256", "thresholds/frontier.csv"),
    ):
        if file_sha256(output_root / relative) != binding[key]:
            raise RuntimeError(f"threshold artifact differs: {relative}")
    return binding


def score_worker(config_path: Path, fold: int) -> None:
    contract, output_root = load_contract(config_path)
    binding = _load_threshold_binding(contract, output_root)
    if fold not in range(5):
        raise ValueError("fold must lie in 0..4")
    phase62 = load_phase62_contract()
    device = stage1_gpu()
    rows, features = _load_all_features(phase62)
    model = stage1_model().to(device=device, dtype=torch.float32)
    gate = read_json(resolve_path(contract["static_config"]["sources"]["phase63_gate"]))
    checkpoint_meta = next(row for row in gate["checkpoints"] if row["model_id"] == f"all_source_fold_{fold}")
    checkpoint = torch.load(resolve_path(checkpoint_meta["path"]), map_location="cpu", weights_only=True)
    if checkpoint["contract_sha256"] != phase62["contract_sha256"]:
        raise RuntimeError("checkpoint contract mismatch")
    model.load_state_dict(checkpoint["model_state_dict"])
    mean_cpu, std_cpu = stage1_normalization()
    scores = stage1_score(
        model,
        features,
        mean=mean_cpu.to(device),
        std=std_cpu.to(device),
        device=device,
    )
    predictions = _prediction_rows(rows, scores, model_id=f"all_source_fold_{fold}", fold=fold)

    existing_h = {
        row["uid"]: row
        for row in read_jsonl(
            resolve_path(f"analysis/dense_failure_stage1/all_source_robustness/main_all/training/fold_{fold}/historical_scores.jsonl")
        )
    }
    existing_c = {
        row["uid"]: row
        for row in read_jsonl(
            resolve_path(f"analysis/dense_failure_stage1/all_source_robustness/main_all/training/fold_{fold}/canonical_scores.jsonl")
        )
    }
    max_error = 0.0
    verified = 0
    train_rows = []
    for row in predictions:
        prior = None
        if row.get("historical_split") in {"val", "test"}:
            prior = existing_h[row["uid"]]
        elif row.get("canonical_fold") == fold:
            prior = existing_c[row["uid"]]
        if prior is not None:
            error = max(abs(float(row[f"p_{layer}"]) - float(prior[f"p_{layer}"])) for layer in range(28))
            max_error = max(max_error, error)
            verified += 1
        if row.get("historical_split") == "train" or row["source_regime"] == "canonical":
            train_rows.append({**row, "compatibility_contract_sha256": contract["contract_sha256"]})
    if verified != 2400 or max_error > 1e-6 or len(train_rows) != 10399:
        raise RuntimeError(f"score reproducibility/coverage failed: verified={verified}, error={max_error}, train={len(train_rows)}")
    path = output_root / f"work/scores/fold_{fold}.jsonl"
    atomic_jsonl(path, train_rows)
    atomic_json(
        output_root / f"work/scores/fold_{fold}.complete.json",
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "threshold_binding_sha256": file_sha256(output_root / "work/threshold_binding.json"),
            "fold": fold,
            "records": len(train_rows),
            "reproduced_heldout_records": verified,
            "maximum_absolute_score_error": max_error,
            "scores_sha256": file_sha256(path),
            "checkpoint_sha256": checkpoint_meta["sha256"],
        },
    )
    print(json.dumps({"fold": fold, "records": len(train_rows), "max_error": max_error}))


def _retained_points(output_root: Path) -> list[dict[str, Any]]:
    rows = read_csv(output_root / "thresholds/named_operating_points.csv")
    return [
        {**row, "threshold": float(row["threshold"]), "preservation_target": float(row["preservation_target"])}
        for row in rows
        if row["retained"] == "True"
    ]


def aggregate_scores(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    binding = _load_threshold_binding(contract, output_root)
    folds = []
    for fold in range(5):
        complete = read_json(output_root / f"work/scores/fold_{fold}.complete.json")
        path = output_root / f"work/scores/fold_{fold}.jsonl"
        if not complete.get("passed") or complete["contract_sha256"] != contract["contract_sha256"] or file_sha256(path) != complete["scores_sha256"]:
            raise RuntimeError(f"fold score output is invalid: {fold}")
        folds.append({row["uid"]: row for row in read_jsonl(path)})
    expected = set(folds[0])
    if len(expected) != 10399 or any(set(rows) != expected for rows in folds):
        raise RuntimeError("fold train-score identities differ")
    ensemble = []
    for uid in sorted(expected):
        rows = [values[uid] for values in folds]
        signature = {
            (row["source_regime"], row["dataset"], row["image_group_id"], bool(row["current_dense_wrong"]))
            for row in rows
        }
        if len(signature) != 1:
            raise RuntimeError(f"fold score metadata differs: {uid}")
        scores = np.mean([[float(row[f"p_{layer}"]) for layer in range(28)] for row in rows], axis=0)
        reference = rows[0]
        ensemble.append(
            {
                "uid": uid,
                "dataset": reference["dataset"],
                "source_regime": reference["source_regime"],
                "image_group_id": reference["image_group_id"],
                "current_dense_wrong": bool(reference["current_dense_wrong"]),
                "current_dense_correct": bool(reference["current_dense_correct"]),
                "score_max": float(scores.max()),
                **{f"p_{layer}": float(scores[layer]) for layer in range(28)},
            }
        )
    points = _retained_points(output_root)
    by_source = {source: [row for row in ensemble if row["source_regime"] == source] for source in SOURCES}
    if len(by_source["historical"]) != 6399 or len(by_source["canonical"]) != 4000:
        raise RuntimeError("ensemble source count mismatch")
    maps = {}
    for source in SOURCES:
        rows = by_source[source]
        source_index = {row["uid"]: row for row in rows}
        score_matrix = np.asarray([[row[f"p_{layer}"] for layer in range(28)] for row in rows])
        mapped = build_trigger_rows(rows, score_matrix, points)
        for item in mapped:
            source_row = source_index[item["uid"]]
            item.update({f"p_{layer}": source_row[f"p_{layer}"] for layer in range(28)})
            item["schema_version"] = "robust_stage1_train_trigger_map_v1"
            item["head_identity"] = "phase63_five_checkpoint_probability_mean"
        maps[source] = mapped
        atomic_jsonl(output_root / f"trigger_maps/{source}_train.jsonl", mapped)
    score_audit = {
        "schema_version": "robust_stage1_train_score_audit_v1",
        "contract_sha256": contract["contract_sha256"],
        "threshold_binding_sha256": file_sha256(output_root / "work/threshold_binding.json"),
        "retained_operating_points": binding["retained_operating_points"],
        "historical_records_per_point": 6399,
        "canonical_records_per_point": 4000,
        "fold_score_files": {f"fold_{fold}": file_sha256(output_root / f"work/scores/fold_{fold}.jsonl") for fold in range(5)},
        "historical_trigger_map_sha256": file_sha256(output_root / "trigger_maps/historical_train.jsonl"),
        "canonical_trigger_map_sha256": file_sha256(output_root / "trigger_maps/canonical_train.jsonl"),
    }
    atomic_json(output_root / "work/trigger_binding.json", score_audit)
    print(json.dumps({"historical": 6399, "canonical": 4000, "operating_points": binding["retained_operating_points"]}))


def _load_trigger_maps(contract: Mapping[str, Any], output_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    binding = read_json(output_root / "work/trigger_binding.json")
    if binding["contract_sha256"] != contract["contract_sha256"]:
        raise RuntimeError("trigger binding contract mismatch")
    result = {}
    for source in SOURCES:
        path = output_root / f"trigger_maps/{source}_train.jsonl"
        if file_sha256(path) != binding[f"{source}_trigger_map_sha256"]:
            raise RuntimeError(f"trigger map differs: {source}")
        for row in read_jsonl(path):
            key = (str(row["uid"]), str(row["threshold_name"]))
            if key in result:
                raise RuntimeError(f"duplicate robust trigger row: {key}")
            result[key] = row
    return result


def _source_rows(config: Mapping[str, Any], prefix: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidates = {row["uid"]: row for row in read_jsonl(resolve_path(config["sources"][f"{prefix}_candidates"]))}
    dense = {row["uid"]: row for row in read_jsonl(resolve_path(config["sources"][f"{prefix}_dense"]))}
    old = {row["uid"]: row for row in read_jsonl(resolve_path(config["sources"][f"{prefix}_old_trigger_map"]))}
    if set(old) - set(candidates) or set(old) - set(dense):
        raise RuntimeError(f"{prefix} old trigger rows do not join candidate/dense data")
    return candidates, dense, old


def _label_map(config: Mapping[str, Any]) -> dict[str, str]:
    result = {}
    for source in SOURCES:
        for suffix, label in (
            ("single_labels", "SINGLE_FIXABLE"),
            ("mcts_labels", "MCTS_ONLY_FIXABLE"),
            ("unresolved", "UNRESOLVED"),
        ):
            for row in read_jsonl(resolve_path(config["sources"][f"{source}_{suffix}"])):
                uid = str(row["uid"])
                if uid in result:
                    raise RuntimeError(f"duplicate prior label: {uid}")
                result[uid] = label
    return result


def _route_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for source in SOURCES:
        for route_source in ("single", "mcts"):
            rows = read_jsonl(resolve_path(config["sources"][f"{source}_{route_source}_routes"]))
            for row in rows:
                actions = list(row["actions"])
                changed = [index for index, action in enumerate(actions) if action != "FULL"]
                if (
                    len(actions) != 28
                    or not changed
                    or int(row["first_non_full_layer"]) != changed[0]
                    or not bool(row["final_lmms_correct"])
                    or not bool(row["replay_token_parity"])
                    or str(row["route_source"]) != route_source
                ):
                    raise RuntimeError(f"invalid prior route: {row.get('route_id')}")
                output.append({**row, "source_regime": source})
    route_ids = [(row["uid"], row["route_id"]) for row in output]
    if len(route_ids) != len(set(route_ids)):
        raise RuntimeError("duplicate prior route identity")
    return output


def prepare_compatibility(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    config = contract["static_config"]
    trigger_maps = _load_trigger_maps(contract, output_root)
    retained = [row["operating_point"] for row in _retained_points(output_root)]
    source_data = {source: _source_rows(config, source) for source in SOURCES}
    old_maps = {uid: {**row, "source_regime": source} for source, (_, _, old) in source_data.items() for uid, row in old.items()}
    transition_rows = []
    for (uid, point), new in sorted(trigger_maps.items()):
        old = old_maps[uid]
        old_layer = int(old["first_trigger_layer"]) if bool(old["triggered"]) else None
        new_layer = int(new["first_trigger_layer"]) if bool(new["triggered"]) else None
        category, delta = transition_category(old_layer, new_layer)
        transition_rows.append(
            {
                "uid": uid,
                "dataset": new["dataset"],
                "source_regime": new["source_regime"],
                "dense_correct": new["dense_correct"],
                "dense_wrong": new["dense_wrong"],
                "operating_point": point,
                "old_triggered": old_layer is not None,
                "old_first_trigger_layer": old_layer,
                "new_triggered": new_layer is not None,
                "new_first_trigger_layer": new_layer,
                "transition": category,
                "delta_layer": delta,
            }
        )
    atomic_jsonl(output_root / "compatibility/old_vs_new_trigger_map.jsonl", transition_rows)

    routes = _route_rows(config)
    structural = {"single": [], "mcts": []}
    candidate_by_job: dict[tuple[str, str, int], dict[str, Any]] = {}
    for route in routes:
        uid = str(route["uid"])
        route_source = str(route["route_source"])
        for point in retained:
            trigger = trigger_maps[(uid, point)]
            new_layer = int(trigger["first_trigger_layer"]) if trigger["triggered"] else None
            status = structural_route_status(
                new_trigger_layer=new_layer,
                first_non_full_layer=int(route["first_non_full_layer"]),
            )
            row = {
                "uid": uid,
                "dataset": route["dataset"],
                "source_regime": route["source_regime"],
                "operating_point": point,
                "threshold": trigger["threshold_value"],
                "new_trigger_layer": new_layer,
                "old_trigger_layer": int(route["trigger_layer"]),
                "route_id": route["route_id"],
                "route_source": route_source,
                "first_non_full_layer": int(route["first_non_full_layer"]),
                "intervention_layer": route.get("intervention_layer"),
                "status": status,
            }
            structural[route_source].append(row)
            if status == "STRUCTURALLY_COMPATIBLE":
                key = (uid, str(route["route_id"]), int(new_layer))
                if key not in candidate_by_job:
                    candidate_by_job[key] = {
                        "uid": uid,
                        "route_id": route["route_id"],
                        "route_source": route_source,
                        "source_regime": route["source_regime"],
                        "dataset": route["dataset"],
                        "new_trigger_layer": int(new_layer),
                        "operating_points": [],
                        "actions": route["actions"],
                        "route_key": route["route_key"],
                        "expected_generated_token_ids": route["generated_token_ids"],
                        "expected_generated_answer": route["generated_answer"],
                        "expected_lmms_metric": route["lmms_metric"],
                        "expected_lmms_score": route["lmms_score"],
                    }
                candidate_by_job[key]["operating_points"].append(point)
    atomic_jsonl(output_root / "compatibility/single_structural_compatibility.jsonl", structural["single"])
    atomic_jsonl(output_root / "compatibility/mcts_structural_compatibility.jsonl", structural["mcts"])

    jobs_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in candidate_by_job.values():
        job["operating_points"] = sorted(job["operating_points"])
        jobs_by_uid[job["uid"]].append(job)
    uid_rows = []
    for uid, jobs in jobs_by_uid.items():
        source = str(jobs[0]["source_regime"])
        candidates, dense, _ = source_data[source]
        candidate = candidates[uid]
        sample = {
            key: candidate[key]
            for key in (
                "uid", "sample_id", "dataset", "prompt", "question", "answer",
                "all_answer_norms", "local_image_path", "image_content_sha256",
                "image_group_id", "max_new_tokens",
            )
        }
        uid_rows.append(
            {
                "uid": uid,
                "source_regime": source,
                "dataset": jobs[0]["dataset"],
                "sample": sample,
                "dense_output": dense[uid],
                "jobs": sorted(jobs, key=lambda row: (row["new_trigger_layer"], row["route_source"], row["route_id"])),
            }
        )
    uid_rows.sort(key=lambda row: row["uid"])
    ranked = sorted(uid_rows, key=lambda row: (-len(row["jobs"]), sha256(f"assign:{config['seed']}:{row['uid']}".encode()).hexdigest()))
    loads = [0] * int(config["world_size"])
    for row in ranked:
        rank = min(range(len(loads)), key=lambda value: (loads[value], value))
        row["worker_rank"] = rank
        loads[rank] += len(row["jobs"])
    uid_rows.sort(key=lambda row: row["uid"])
    smoke_candidates = sorted(
        uid_rows,
        key=lambda row: (
            row["source_regime"],
            0 if any(job["route_source"] == "mcts" for job in row["jobs"]) else 1,
            sha256(f"smoke:{config['seed']}:{row['uid']}".encode()).hexdigest(),
        ),
    )
    smoke_uids = set()
    for source in SOURCES:
        choices = [row for row in smoke_candidates if row["source_regime"] == source]
        smoke_uids.update(row["uid"] for row in choices[:4])
    smoke = [row for row in uid_rows if row["uid"] in smoke_uids]
    full = [row for row in uid_rows if row["uid"] not in smoke_uids]
    atomic_jsonl(output_root / "work/replay_smoke_manifest.jsonl", smoke)
    atomic_jsonl(output_root / "work/replay_full_manifest.jsonl", full)
    binding = {
        "schema_version": "robust_stage1_replay_binding_v1",
        "contract_sha256": contract["contract_sha256"],
        "trigger_binding_sha256": file_sha256(output_root / "work/trigger_binding.json"),
        "retained_operating_points": retained,
        "unique_replay_jobs": len(candidate_by_job),
        "replay_uids": len(uid_rows),
        "smoke_uids": len(smoke),
        "full_uids": len(full),
        "files": {
            "old_vs_new": file_sha256(output_root / "compatibility/old_vs_new_trigger_map.jsonl"),
            "single_structural": file_sha256(output_root / "compatibility/single_structural_compatibility.jsonl"),
            "mcts_structural": file_sha256(output_root / "compatibility/mcts_structural_compatibility.jsonl"),
            "smoke_manifest": file_sha256(output_root / "work/replay_smoke_manifest.jsonl"),
            "full_manifest": file_sha256(output_root / "work/replay_full_manifest.jsonl"),
        },
        "worker_job_loads": loads,
    }
    atomic_json(output_root / "work/replay_binding.json", binding)
    print(json.dumps({"replay_jobs": len(candidate_by_job), "replay_uids": len(uid_rows), "worker_loads": loads, "smoke_uids": len(smoke)}))


def _load_replay_binding(contract: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    binding = read_json(output_root / "work/replay_binding.json")
    if binding["contract_sha256"] != contract["contract_sha256"]:
        raise RuntimeError("replay binding contract mismatch")
    paths = {
        "old_vs_new": "compatibility/old_vs_new_trigger_map.jsonl",
        "single_structural": "compatibility/single_structural_compatibility.jsonl",
        "mcts_structural": "compatibility/mcts_structural_compatibility.jsonl",
        "smoke_manifest": "work/replay_smoke_manifest.jsonl",
        "full_manifest": "work/replay_full_manifest.jsonl",
    }
    for key, relative in paths.items():
        if file_sha256(output_root / relative) != binding["files"][key]:
            raise RuntimeError(f"replay binding source differs: {relative}")
    return binding


def _safe_uid(uid: str) -> str:
    return sha256(uid.encode()).hexdigest()[:24]


def replay_worker(config_path: Path, *, mode: str, rank: int, resume: bool) -> None:
    contract, output_root = load_contract(config_path, verify_model=True)
    config = contract["static_config"]
    binding = _load_replay_binding(contract, output_root)
    if mode not in {"smoke", "full"} or rank not in range(int(config["world_size"])):
        raise ValueError("invalid replay mode/rank")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("replay worker requires exactly one visible GPU")
    if mode == "full":
        completion = read_json(output_root / "work/replay_smoke_completion.json")
        if not completion.get("passed"):
            raise RuntimeError("full replay requires passing smoke")
    manifest = read_jsonl(output_root / f"work/replay_{mode}_manifest.jsonl")
    rows = [row for row in manifest if int(row["worker_rank"]) == rank]
    root = output_root / f"work/replay_{mode}/rank{rank:02d}"
    sample_dir = root / "samples"
    complete_path = root / "complete.json"
    if complete_path.exists():
        raise RuntimeError(f"replay worker already complete: {complete_path}")
    existing = list(sample_dir.glob("*.json")) if sample_dir.exists() else []
    if existing and not resume:
        raise RuntimeError("partial replay rows require --resume")
    completed = {}
    for path in existing:
        row = read_json(path)
        if row.get("contract_sha256") != contract["contract_sha256"] or not row.get("passed"):
            raise RuntimeError(f"incompatible replay resume row: {path}")
        completed[row["uid"]] = row

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    backend = read_json(resolve_path(config["sources"]["historical_dense_config"]))["backend_settings"]
    configure_dense_determinism(int(config["seed"]) + rank, backend)
    processor, _base, wrapped = _load_model(config, device)
    for index, row in enumerate(rows):
        uid = str(row["uid"])
        if uid in completed:
            continue
        started = time.monotonic()
        results = []
        try:
            sample = row["sample"]
            inputs, metadata = build_dense_inputs(processor, sample, device)
            if metadata["consumed_image_sha256"] != sample["image_content_sha256"]:
                raise RuntimeError("consumed image SHA differs")
            prepared = build_binary_inputs(wrapped, inputs)
            baseline = capture_full_baseline(
                wrapped,
                inputs,
                prepared_inputs=prepared,
                use_cache=True,
                native_causal=bool(config["replay"]["native_full_row_dispatch"]),
            )
            dense_state = _generate_output(processor, wrapped, baseline, inputs, sample)
            expected_dense = row["dense_output"]
            baseline_ok = (
                dense_state["generated_ids"] == expected_dense["generated_token_ids"]
                and dense_state["generated_answer"] == expected_dense["generated_answer"]
                and dense_state["lmms_score"] == float(expected_dense["lmms_eval_per_sample_score"])
                and dense_state["correct"] == bool(expected_dense["current_dense_correct"])
            )
            if not baseline_ok:
                raise RuntimeError("current dense baseline parity failed")
            for job in row["jobs"]:
                try:
                    start = int(job["new_trigger_layer"])
                    actions = tuple(job["actions"])
                    if any(action != "FULL" for action in actions[:start]):
                        raise RuntimeError("route requires intervention before new trigger")
                    output = capture_four_action_suffix_from_full_baseline(
                        wrapped, baseline, start, actions[start:]
                    )
                    state = _generate_output(processor, wrapped, output, inputs, sample)
                    parity = (
                        state["generated_ids"] == job["expected_generated_token_ids"]
                        and state["generated_answer"] == job["expected_generated_answer"]
                        and state["lmms_metric"] == job["expected_lmms_metric"]
                        and state["lmms_score"] == float(job["expected_lmms_score"])
                    )
                    if state["correct"] and parity:
                        status = "REPLAY_COMPATIBLE_CORRECT"
                    elif not state["correct"]:
                        status = "REPLAY_INCOMPATIBLE_WRONG"
                    else:
                        status = "REPLAY_ERROR"
                    results.append(
                        {
                            **job,
                            "status": status,
                            "current_generated_token_ids": state["generated_ids"],
                            "current_generated_answer": state["generated_answer"],
                            "current_lmms_metric": state["lmms_metric"],
                            "current_lmms_score": state["lmms_score"],
                            "current_correct": state["correct"],
                            "exact_stored_output_parity": parity,
                            "error": None,
                        }
                    )
                except Exception as exc:
                    results.append({**job, "status": "REPLAY_ERROR", "error": str(exc)})
            final = {
                "schema_version": "robust_stage1_route_replay_sample_v1",
                "contract_sha256": contract["contract_sha256"],
                "uid": uid,
                "source_regime": row["source_regime"],
                "dataset": row["dataset"],
                "worker_rank": rank,
                "mode": mode,
                "image_sha_verified": True,
                "dense_baseline_parity": True,
                "results": results,
                "elapsed_seconds": time.monotonic() - started,
                "passed": True,
            }
        except Exception as exc:
            final = {
                "schema_version": "robust_stage1_route_replay_sample_v1",
                "contract_sha256": contract["contract_sha256"],
                "uid": uid,
                "source_regime": row["source_regime"],
                "dataset": row["dataset"],
                "worker_rank": rank,
                "mode": mode,
                "image_sha_verified": False,
                "dense_baseline_parity": False,
                "results": [{**job, "status": "REPLAY_ERROR", "error": str(exc)} for job in row["jobs"]],
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started,
                "passed": True,
            }
        atomic_json(sample_dir / f"{_safe_uid(uid)}.json", final)
        completed[uid] = final
        print(json.dumps({"mode": mode, "rank": rank, "sample": index + 1, "total": len(rows), "uid": uid, "jobs": len(row["jobs"])}), flush=True)
        torch.cuda.empty_cache()
    expected = {row["uid"] for row in rows}
    if set(completed) != expected:
        raise RuntimeError("replay worker completion mismatch")
    statuses = Counter(result["status"] for row in completed.values() for result in row["results"])
    atomic_json(
        complete_path,
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "replay_binding_sha256": file_sha256(output_root / "work/replay_binding.json"),
            "mode": mode,
            "rank": rank,
            "uids": len(completed),
            "jobs": sum(len(row["results"]) for row in completed.values()),
            "statuses": dict(statuses),
            "completed_at": utc_now(),
        },
    )


def finish_smoke(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    binding = _load_replay_binding(contract, output_root)
    expected_rows = read_jsonl(output_root / "work/replay_smoke_manifest.jsonl")
    totals = Counter()
    uids = 0
    for rank in range(4):
        complete = read_json(output_root / f"work/replay_smoke/rank{rank:02d}/complete.json")
        if not complete.get("passed") or complete["contract_sha256"] != contract["contract_sha256"]:
            raise RuntimeError(f"smoke rank {rank} incomplete")
        uids += int(complete["uids"])
        totals.update(complete["statuses"])
    if uids != len(expected_rows) or totals["REPLAY_ERROR"] or totals["REPLAY_INCOMPATIBLE_WRONG"]:
        raise RuntimeError(f"replay smoke failed: uids={uids}, statuses={totals}")
    atomic_json(
        output_root / "work/replay_smoke_completion.json",
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "replay_binding_sha256": file_sha256(output_root / "work/replay_binding.json"),
            "uids": uids,
            "jobs": sum(totals.values()),
            "statuses": dict(totals),
        },
    )
    print(json.dumps({"smoke_passed": True, "uids": uids, "statuses": dict(totals)}))


def _collect_replays(contract: Mapping[str, Any], output_root: Path) -> list[dict[str, Any]]:
    expected = {}
    for mode in ("smoke", "full"):
        for row in read_jsonl(output_root / f"work/replay_{mode}_manifest.jsonl"):
            expected[row["uid"]] = row
    collected = {}
    for mode in ("smoke", "full"):
        for rank in range(4):
            complete = read_json(output_root / f"work/replay_{mode}/rank{rank:02d}/complete.json")
            if not complete.get("passed") or complete["contract_sha256"] != contract["contract_sha256"]:
                raise RuntimeError(f"replay worker incomplete: {mode}/{rank}")
            sample_dir = output_root / f"work/replay_{mode}/rank{rank:02d}/samples"
            for path in sample_dir.glob("*.json"):
                row = read_json(path)
                if row["uid"] in collected:
                    raise RuntimeError(f"duplicate replay UID: {row['uid']}")
                collected[row["uid"]] = row
    if set(collected) != set(expected):
        raise RuntimeError("global replay UID coverage mismatch")
    results = []
    for uid, row in collected.items():
        for result in row["results"]:
            for point in result["operating_points"]:
                results.append(
                    {
                        "uid": uid,
                        "dataset": row["dataset"],
                        "source_regime": row["source_regime"],
                        "operating_point": point,
                        "new_trigger_layer": result["new_trigger_layer"],
                        "route_id": result["route_id"],
                        "route_source": result["route_source"],
                        "status": result["status"],
                        "exact_stored_output_parity": result.get("exact_stored_output_parity", False),
                        "current_generated_token_ids": result.get("current_generated_token_ids"),
                        "current_generated_answer": result.get("current_generated_answer"),
                        "current_lmms_metric": result.get("current_lmms_metric"),
                        "current_lmms_score": result.get("current_lmms_score"),
                        "error": result.get("error"),
                    }
                )
    return sorted(results, key=lambda row: (row["operating_point"], row["uid"], row["route_source"], row["route_id"]))


def _trigger_summaries(trigger_maps: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary = []
    dataset_rows = []
    depth_rows = []
    points = sorted({row["threshold_name"] for row in trigger_maps})
    for point in points:
        point_rows = [row for row in trigger_maps if row["threshold_name"] == point]
        for source in SOURCES:
            source_rows = [row for row in point_rows if row["source_regime"] == source]
            counts = Counter(("W" if row["dense_wrong"] else "C", bool(row["triggered"])) for row in source_rows)
            triggered_layers = [int(row["first_trigger_layer"]) for row in source_rows if row["triggered"]]
            summary.append(
                {
                    "operating_point": point,
                    "source_regime": source,
                    "c_no_trigger": counts[("C", False)],
                    "c_trigger": counts[("C", True)],
                    "w_no_trigger": counts[("W", False)],
                    "w_trigger": counts[("W", True)],
                    "c_preservation": counts[("C", False)] / (counts[("C", False)] + counts[("C", True)]),
                    "w_recall": counts[("W", True)] / (counts[("W", False)] + counts[("W", True)]),
                    "trigger_precision": counts[("W", True)] / max(1, counts[("W", True)] + counts[("C", True)]),
                    "median_trigger_layer": float(np.median(triggered_layers)) if triggered_layers else float("nan"),
                    "l0_trigger_fraction": sum(layer == 0 for layer in triggered_layers) / max(1, len(triggered_layers)),
                }
            )
            for dataset in DATASETS:
                subset = [row for row in source_rows if row["dataset"] == dataset]
                cell = Counter(("W" if row["dense_wrong"] else "C", bool(row["triggered"])) for row in subset)
                dataset_rows.append(
                    {
                        "operating_point": point,
                        "source_regime": source,
                        "dataset": dataset,
                        "c_no_trigger": cell[("C", False)],
                        "c_trigger": cell[("C", True)],
                        "w_no_trigger": cell[("W", False)],
                        "w_trigger": cell[("W", True)],
                        "total": len(subset),
                    }
                )
            for outcome, predicate in (
                ("triggered_c", lambda row: row["dense_correct"] and row["triggered"]),
                ("triggered_w", lambda row: row["dense_wrong"] and row["triggered"]),
            ):
                layers = [int(row["first_trigger_layer"]) for row in source_rows if predicate(row)]
                for depth in DEPTH_BINS:
                    count = sum(trigger_depth_bin(layer) == depth for layer in layers)
                    depth_rows.append(
                        {
                            "operating_point": point,
                            "source_regime": source,
                            "outcome": outcome,
                            "depth_bin": depth,
                            "count": count,
                            "fraction": count / max(1, len(layers)),
                            "triggered_records": len(layers),
                        }
                    )
    return summary, dataset_rows, depth_rows


def aggregate(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    config = contract["static_config"]
    _load_replay_binding(contract, output_root)
    replay = _collect_replays(contract, output_root)
    atomic_jsonl(output_root / "compatibility/replay_results.jsonl", replay)
    trigger_rows = read_jsonl(output_root / "trigger_maps/historical_train.jsonl") + read_jsonl(output_root / "trigger_maps/canonical_train.jsonl")
    trigger_summary, dataset_summary, depth_summary = _trigger_summaries(trigger_rows)
    atomic_csv(output_root / "trigger_maps/trigger_summary.csv", trigger_summary)
    atomic_csv(output_root / "trigger_maps/dataset_source_breakdown.csv", dataset_summary)
    atomic_csv(output_root / "trigger_maps/trigger_depth_breakdown.csv", depth_summary)

    transition_rows = read_jsonl(output_root / "compatibility/old_vs_new_trigger_map.jsonl")
    transition_counts = Counter(
        (row["operating_point"], row["source_regime"], "W" if row["dense_wrong"] else "C", row["transition"])
        for row in transition_rows
    )
    transition_matrix = []
    for key, count in sorted(transition_counts.items()):
        transition_matrix.append(
            {"operating_point": key[0], "source_regime": key[1], "dense_outcome": key[2], "transition": key[3], "count": count}
        )
    atomic_csv(output_root / "metrics/trigger_transition_matrix.csv", transition_matrix)

    labels = _label_map(config)
    replay_correct = defaultdict(lambda: {"single": set(), "mcts": set()})
    for row in replay:
        if row["status"] == "REPLAY_COMPATIBLE_CORRECT":
            replay_correct[(row["uid"], row["operating_point"])][row["route_source"]].add(row["route_id"])
    per_sample = []
    for row in trigger_rows:
        if not row["dense_wrong"] or not row["triggered"]:
            continue
        key = (row["uid"], row["threshold_name"])
        available = replay_correct[key]
        category = compatibility_category(
            has_replay_single=bool(available["single"]),
            has_replay_mcts=bool(available["mcts"]),
            prior_label=labels.get(row["uid"]),
        )
        per_sample.append(
            {
                "uid": row["uid"],
                "dataset": row["dataset"],
                "source_regime": row["source_regime"],
                "operating_point": row["threshold_name"],
                "new_trigger_layer": row["first_trigger_layer"],
                "trigger_depth_bin": trigger_depth_bin(int(row["first_trigger_layer"])),
                "prior_label": labels.get(row["uid"]),
                "replay_compatible_single_routes": len(available["single"]),
                "replay_compatible_mcts_routes": len(available["mcts"]),
                "compatibility_category": category,
                "has_reusable_corrective_route": bool(available["single"] or available["mcts"]),
                "requires_new_search": not bool(available["single"] or available["mcts"]),
            }
        )
    atomic_jsonl(output_root / "compatibility/per_sample_compatibility.jsonl", per_sample)

    structural = read_jsonl(output_root / "compatibility/single_structural_compatibility.jsonl") + read_jsonl(output_root / "compatibility/mcts_structural_compatibility.jsonl")
    reuse_rows = []
    for point in sorted({row["operating_point"] for row in structural}):
        for source in SOURCES:
            for route_source in ("single", "mcts"):
                subset = [row for row in structural if row["operating_point"] == point and row["source_regime"] == source and row["route_source"] == route_source]
                replay_subset = [row for row in replay if row["operating_point"] == point and row["source_regime"] == source and row["route_source"] == route_source]
                compatible = [row for row in replay_subset if row["status"] == "REPLAY_COMPATIBLE_CORRECT"]
                reuse_rows.append(
                    {
                        "operating_point": point,
                        "source_regime": source,
                        "route_source": route_source,
                        "existing_routes": len(subset),
                        "structurally_compatible_routes": sum(row["status"] == "STRUCTURALLY_COMPATIBLE" for row in subset),
                        "structurally_incompatible_routes": sum(row["status"] == "STRUCTURALLY_INCOMPATIBLE" for row in subset),
                        "not_new_triggered_routes": sum(row["status"] == "NOT_NEW_TRIGGERED" for row in subset),
                        "replay_compatible_correct_routes": len(compatible),
                        "replay_incompatible_wrong_routes": sum(row["status"] == "REPLAY_INCOMPATIBLE_WRONG" for row in replay_subset),
                        "replay_error_routes": sum(row["status"] == "REPLAY_ERROR" for row in replay_subset),
                        "samples_with_replay_compatible_routes": len({row["uid"] for row in compatible}),
                    }
                )
    atomic_csv(output_root / "metrics/route_reuse_summary.csv", reuse_rows)

    preservation = []
    for point in sorted({row["threshold_name"] for row in trigger_rows}):
        for source in SOURCES:
            subset = [row for row in transition_rows if row["operating_point"] == point and row["source_regime"] == source and row["dense_correct"]]
            for transition in ("SAME_TRIGGER", "NEW_EARLIER", "NEW_LATER", "OLD_TRIGGER_ONLY", "NEW_TRIGGER_ONLY", "NEITHER_TRIGGER"):
                count = sum(row["transition"] == transition for row in subset)
                preservation.append(
                    {
                        "operating_point": point,
                        "source_regime": source,
                        "transition": transition,
                        "correct_records": len(subset),
                        "count": count,
                        "new_preservation_full_required": transition in {"SAME_TRIGGER", "NEW_EARLIER", "NEW_LATER", "NEW_TRIGGER_ONLY"},
                        "direct_old_state_reuse": transition == "SAME_TRIGGER",
                    }
                )
    atomic_csv(output_root / "metrics/preservation_population.csv", preservation)

    total_wrong = Counter((row["source_regime"], row["threshold_name"]) for row in trigger_rows if row["dense_wrong"])
    lower = []
    workload = []
    group_keys = sorted({(row["operating_point"], row["source_regime"], row["dataset"], row["trigger_depth_bin"]) for row in per_sample})
    for point in sorted({row["operating_point"] for row in per_sample}):
        for source in (*SOURCES, "pooled"):
            selected = [row for row in per_sample if row["operating_point"] == point and (source == "pooled" or row["source_regime"] == source)]
            denominator = sum(total_wrong[(item, point)] for item in SOURCES) if source == "pooled" else total_wrong[(source, point)]
            reusable = sum(row["has_reusable_corrective_route"] for row in selected)
            lower.append(
                {
                    "operating_point": point,
                    "source_regime": source,
                    "all_dense_wrong": denominator,
                    "new_triggered_wrong": len(selected),
                    "wrong_with_replay_compatible_route": reusable,
                    "known_repairable_wrong_recall_lower_bound": reusable / denominator,
                    "conditional_known_repairable_given_triggered_wrong": reusable / max(1, len(selected)),
                }
            )
    for point, source, dataset, depth in group_keys:
        selected = [row for row in per_sample if row["operating_point"] == point and row["source_regime"] == source and row["dataset"] == dataset and row["trigger_depth_bin"] == depth]
        counts = Counter(row["compatibility_category"] for row in selected if row["requires_new_search"])
        workload.append(
            {
                "operating_point": point,
                "source_regime": source,
                "dataset": dataset,
                "trigger_depth_bin": depth,
                "new_triggered_wrong": len(selected),
                "with_reusable_route": sum(not row["requires_new_search"] for row in selected),
                "new_search_required": sum(row["requires_new_search"] for row in selected),
                "no_prior_search": counts["NEW_TRIGGER_NO_EXISTING_LABEL"],
                "prior_unresolved": counts["EXISTING_UNRESOLVED"],
                "prior_route_incompatible_or_replay_failed": counts["EXISTING_LABEL_BUT_INCOMPATIBLE"],
            }
        )
    atomic_csv(output_root / "metrics/known_repairable_lower_bound.csv", lower)
    atomic_csv(output_root / "metrics/new_search_workload.csv", workload)

    named = {row["operating_point"]: row for row in _retained_points(output_root)}
    op_summary = []
    for point, values in named.items():
        pooled_lower = next(row for row in lower if row["operating_point"] == point and row["source_regime"] == "pooled")
        h_train = next(row for row in trigger_summary if row["operating_point"] == point and row["source_regime"] == "historical")
        c_train = next(row for row in trigger_summary if row["operating_point"] == point and row["source_regime"] == "canonical")
        op_summary.append(
            {
                "operating_point": point,
                "threshold": values["threshold"],
                "calibration_historical_c_preservation": values["historical_c_preservation"],
                "calibration_canonical_c_preservation": values["canonical_c_preservation"],
                "calibration_pooled_w_recall": values["pooled_w_recall"],
                "calibration_median_trigger_layer": values["median_first_trigger_layer"],
                "train_historical_c_preservation": h_train["c_preservation"],
                "train_canonical_c_preservation": c_train["c_preservation"],
                "train_historical_w_recall": h_train["w_recall"],
                "train_canonical_w_recall": c_train["w_recall"],
                "known_repairable_w_recall": pooled_lower["known_repairable_wrong_recall_lower_bound"],
                "conditional_known_repairable_given_triggered_w": pooled_lower["conditional_known_repairable_given_triggered_wrong"],
                "new_w_search_required": pooled_lower["new_triggered_wrong"] - pooled_lower["wrong_with_replay_compatible_route"],
            }
        )
    atomic_csv(output_root / "metrics/operating_point_summary.csv", op_summary)
    _figures(output_root, named, op_summary, transition_rows, reuse_rows)
    _summaries(output_root, named, op_summary, trigger_summary, transition_rows, reuse_rows, replay, per_sample)
    atomic_json(
        output_root / "run_summary.json",
        {
            "contract_sha256": contract["contract_sha256"],
            "retained_operating_points": list(named),
            "replay_jobs": len(replay),
            "replay_statuses": dict(Counter(row["status"] for row in replay)),
            "new_search_required": {row["operating_point"]: row["new_w_search_required"] for row in op_summary},
            "passed": True,
        },
    )
    print(json.dumps(read_json(output_root / "run_summary.json"), sort_keys=True))


def _figures(output_root: Path, named: Mapping[str, Mapping[str, Any]], op_summary: Sequence[Mapping[str, Any]], transitions: Sequence[Mapping[str, Any]], reuse: Sequence[Mapping[str, Any]]) -> None:
    figures = output_root / "figures"; figures.mkdir(parents=True, exist_ok=True)
    all_named = read_csv(output_root / "thresholds/named_operating_points.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    x = [float(row["worst_source_c_preservation"]) for row in all_named]
    y = [float(row["pooled_w_recall"]) for row in all_named]
    ax.plot(x, y, marker="o")
    for row, xv, yv in zip(all_named, x, y): ax.annotate(row["operating_point"], (xv, yv))
    ax.set(xlabel="Worst-source C preservation", ylabel="Pooled W recall", title="Named robust operating points"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figures / "preservation_vs_wrong_recall.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([float(row["worst_source_c_preservation"]) for row in all_named], [float(row["median_first_trigger_layer"]) for row in all_named], marker="o")
    ax.set(xlabel="Worst-source C preservation", ylabel="Median first-trigger layer", title="Preservation vs trigger depth"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figures / "preservation_vs_trigger_depth.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    order = sorted(all_named, key=lambda row: float(row["threshold"]))
    ax.plot([float(row["threshold"]) for row in order], [float(row["historical_w_recall"]) for row in order], marker="o", label="Historical W recall")
    ax.plot([float(row["threshold"]) for row in order], [float(row["canonical_w_recall"]) for row in order], marker="o", label="Canonical W recall")
    ax.set(xlabel="Threshold", ylabel="W recall", title="Threshold frontier by source"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figures / "threshold_frontier_by_source.png", dpi=180); plt.close(fig)

    selected = [row for row in transitions if row["old_triggered"] and row["new_triggered"]]
    fig, ax = plt.subplots(figsize=(6, 6)); ax.scatter([row["old_first_trigger_layer"] for row in selected], [row["new_first_trigger_layer"] for row in selected], s=5, alpha=.15)
    ax.plot([0,27],[0,27],color="black",linewidth=1); ax.set(xlabel="Old trigger layer", ylabel="New trigger layer", title="Old vs new trigger layer"); fig.tight_layout(); fig.savefig(figures / "old_vs_new_trigger_layer.png", dpi=180); plt.close(fig)

    points = list(named)
    fig, ax = plt.subplots(figsize=(7, 5))
    single = [sum(int(row["replay_compatible_correct_routes"]) for row in reuse if row["operating_point"] == point and row["route_source"] == "single") for point in points]
    mcts = [sum(int(row["replay_compatible_correct_routes"]) for row in reuse if row["operating_point"] == point and row["route_source"] == "mcts") for point in points]
    ax.bar(points, single, label="single"); ax.bar(points, mcts, bottom=single, label="MCTS"); ax.set(ylabel="Replay-compatible routes", title="Route reuse by threshold"); ax.legend(); fig.tight_layout(); fig.savefig(figures / "route_reuse_by_threshold.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5)); ax.bar([row["operating_point"] for row in op_summary], [row["new_w_search_required"] for row in op_summary]); ax.set(ylabel="Triggered-W requiring new search", title="New search workload"); fig.tight_layout(); fig.savefig(figures / "search_workload_by_threshold.png", dpi=180); plt.close(fig)


def _summaries(output_root: Path, named: Mapping[str, Mapping[str, Any]], op_summary: Sequence[Mapping[str, Any]], trigger_summary: Sequence[Mapping[str, Any]], transitions: Sequence[Mapping[str, Any]], reuse: Sequence[Mapping[str, Any]], replay: Sequence[Mapping[str, Any]], per_sample: Sequence[Mapping[str, Any]]) -> None:
    all_named = read_csv(output_root / "thresholds/named_operating_points.csv")
    text = "# Robust Stage-1 operating-point summary\n\n"
    text += "The frozen Stage-1 object is the Phase-63 five-checkpoint per-layer probability-mean system. Train-map rates below are descriptive in-sample admission rates, not held-out calibration claims.\n\n"
    text += "| Point | Tau | Hist C preserve | Canon C preserve | Hist W recall | Canon W recall | Pooled W recall | Median layer | Fold range | Retained |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    for row in all_named:
        text += f"| {row['operating_point']} | {float(row['threshold']):.6f} | {float(row['historical_c_preservation']):.4f} | {float(row['canonical_c_preservation']):.4f} | {float(row['historical_w_recall']):.4f} | {float(row['canonical_w_recall']):.4f} | {float(row['pooled_w_recall']):.4f} | {float(row['median_first_trigger_layer']):.1f} | {float(row['fold_threshold_range']):.4f} | {row['retained']} |\n"
    text += "\nRetained operating points are " + ", ".join(named) + ". They represent conservative, middle, and permissive Stage-1 candidates because they passed the prospective fold-stability, distinctness, and >=5-point incremental pooled-W-recall rules. None is a final deployment threshold: only future measured W→C and C→W can select that.\n"
    _atomic_bytes(output_root / "summaries/robust_operating_point_summary.md", text.encode())

    comp = "# Stage-2 label compatibility summary\n\n"
    comp += f"Exact current-runtime replay statuses over route×operating-point rows: {dict(Counter(row['status'] for row in replay))}.\n\n"
    comp += "| Point | Triggered W | Same | Earlier | Later | Compatible single routes | Compatible MCTS routes | W with reusable route | New search | New triggered C |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for point in named:
        triggered_w = [row for row in per_sample if row["operating_point"] == point]
        w_trans = [row for row in transitions if row["operating_point"] == point and row["dense_wrong"]]
        reuse_point = [row for row in reuse if row["operating_point"] == point]
        new_c = sum(row["operating_point"] == point and row["dense_correct"] and row["new_triggered"] for row in transitions)
        comp += f"| {point} | {len(triggered_w)} | {sum(row['transition']=='SAME_TRIGGER' for row in w_trans)} | {sum(row['transition']=='NEW_EARLIER' for row in w_trans)} | {sum(row['transition']=='NEW_LATER' for row in w_trans)} | {sum(int(row['replay_compatible_correct_routes']) for row in reuse_point if row['route_source']=='single')} | {sum(int(row['replay_compatible_correct_routes']) for row in reuse_point if row['route_source']=='mcts')} | {sum(row['has_reusable_corrective_route'] for row in triggered_w)} | {sum(row['requires_new_search'] for row in triggered_w)} | {new_c} |\n"
    comp += "\nStructural compatibility was only a pre-filter; every counted reusable route passed current Qwen/LMMS execution and exact stored-token parity. A missing compatible label is not evidence of unfixability.\n"
    _atomic_bytes(output_root / "summaries/stage2_label_compatibility_summary.md", comp.encode())

    missing = {point: {row["uid"] for row in per_sample if row["operating_point"] == point and row["requires_new_search"]} for point in named}
    union = set().union(*missing.values()) if missing else set()
    recommendation = "# Next corrective-search recommendation\n\n"
    recommendation += "Do not rerun all historical searches. Reuse every exact-replay-compatible route and search only triggered-W rows still missing coverage.\n\n"
    for point in named:
        recommendation += f"- {point}: {len(missing[point])} W samples require new search.\n"
    recommendation += f"- Union across retained points: {len(union)} unique W samples.\n"
    if "P90" in missing and all(missing[point] <= missing["P90"] for point in missing):
        recommendation += "- Missing sample identities are nested inside P90. One future P90-envelope search can share sample execution, but each discovered route must still be tagged against the later P95/P98 handoff because an intervention before a later trigger is not reusable there.\n"
    else:
        recommendation += "- Missing identities are not fully nested; a future plan must explicitly deduplicate their union and preserve operating-point handoff tags.\n"
    recommendation += "\nThis phase did not authorize or execute that search.\n"
    _atomic_bytes(output_root / "summaries/next_search_recommendation.md", recommendation.encode())


def finalize(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    required = [
        "protocol.md", "frozen_protocol.json",
        "thresholds/frontier.csv", "thresholds/named_operating_points.csv", "thresholds/fold_stability.csv",
        "trigger_maps/historical_train.jsonl", "trigger_maps/canonical_train.jsonl", "trigger_maps/trigger_summary.csv", "trigger_maps/dataset_source_breakdown.csv", "trigger_maps/trigger_depth_breakdown.csv",
        "compatibility/old_vs_new_trigger_map.jsonl", "compatibility/single_structural_compatibility.jsonl", "compatibility/mcts_structural_compatibility.jsonl", "compatibility/replay_results.jsonl", "compatibility/per_sample_compatibility.jsonl",
        "metrics/trigger_transition_matrix.csv", "metrics/route_reuse_summary.csv", "metrics/preservation_population.csv", "metrics/known_repairable_lower_bound.csv", "metrics/new_search_workload.csv", "metrics/operating_point_summary.csv",
        "figures/preservation_vs_wrong_recall.png", "figures/preservation_vs_trigger_depth.png", "figures/threshold_frontier_by_source.png", "figures/old_vs_new_trigger_layer.png", "figures/route_reuse_by_threshold.png", "figures/search_workload_by_threshold.png",
        "summaries/robust_operating_point_summary.md", "summaries/stage2_label_compatibility_summary.md", "summaries/next_search_recommendation.md", "run_summary.json",
    ]
    missing = [path for path in required if not (output_root / path).is_file()]
    if missing:
        raise RuntimeError(f"missing final artifacts: {missing}")
    run = read_json(output_root / "run_summary.json")
    if not run.get("passed") or run["contract_sha256"] != contract["contract_sha256"]:
        raise RuntimeError("run summary is invalid")
    trigger_rows = read_jsonl(output_root / "trigger_maps/historical_train.jsonl") + read_jsonl(output_root / "trigger_maps/canonical_train.jsonl")
    if len(trigger_rows) != 10399 * len(run["retained_operating_points"]):
        raise RuntimeError("final trigger-map coverage mismatch")
    replay = read_jsonl(output_root / "compatibility/replay_results.jsonl")
    if len(replay) != int(run["replay_jobs"]):
        raise RuntimeError("final replay coverage mismatch")
    artifacts = []
    for relative in required:
        path = output_root / relative
        artifacts.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    manifest = {
        "schema_version": "robust_stage1_operating_points_compatibility_artifact_manifest_v1",
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        "retained_operating_points": run["retained_operating_points"],
        "replay_statuses": run["replay_statuses"],
        "artifacts": artifacts,
    }
    atomic_json(output_root / "artifact_manifest.json", manifest)
    for row in artifacts:
        if file_sha256(output_root / row["path"]) != row["sha256"]:
            raise RuntimeError(f"artifact changed after manifest: {row['path']}")
    print(json.dumps({"finalized": True, "artifacts": len(artifacts), "replay_statuses": run["replay_statuses"]}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "thresholds", "score-worker", "aggregate-scores", "prepare-compatibility", "replay-worker", "finish-smoke", "aggregate", "finalize"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--mode", choices=("smoke", "full"))
    parser.add_argument("--rank", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "prepare": prepare(args.config)
    elif args.action == "thresholds": thresholds(args.config)
    elif args.action == "score-worker": score_worker(args.config, args.fold)
    elif args.action == "aggregate-scores": aggregate_scores(args.config)
    elif args.action == "prepare-compatibility": prepare_compatibility(args.config)
    elif args.action == "replay-worker": replay_worker(args.config, mode=args.mode, rank=args.rank, resume=args.resume)
    elif args.action == "finish-smoke": finish_smoke(args.config)
    elif args.action == "aggregate": aggregate(args.config)
    elif args.action == "finalize": finalize(args.config)


if __name__ == "__main__":
    main()
