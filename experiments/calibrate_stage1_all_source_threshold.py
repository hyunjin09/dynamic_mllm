#!/usr/bin/env python3
"""Calibrate the frozen ALL-source Shared Random-4 Stage-1 gate."""

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
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dense_failure_stage1.threshold_calibration import (  # noqa: E402
    DEPTH_BINS,
    choose_freeze_decision,
    first_trigger_layers,
    group_bootstrap_metrics,
    most_permissive_at_preservation,
    operating_metrics,
    robust_threshold_sweep,
    select_primary_threshold,
    wilson_interval,
)


OUTPUT_ROOT = PROJECT_ROOT / "analysis/dense_failure_stage1/all_source_threshold_calibration"
PHASE62_ROOT = PROJECT_ROOT / "analysis/dense_failure_stage1/all_source_robustness"
PLAN = PROJECT_ROOT / "plans/stage1_all_source_robust_threshold_calibration_plan.md"
PHASE62_PROTOCOL = PHASE62_ROOT / "frozen_protocol.json"
PHASE62_MANIFEST = PHASE62_ROOT / "artifact_manifest.json"
CHECKPOINT_MANIFEST = PHASE62_ROOT / "main_all/training/checkpoint_manifest.json"
CANONICAL_SCORES = PHASE62_ROOT / "main_all/predictions/canonical_oof_scores.jsonl"
HISTORICAL_SCORES = PHASE62_ROOT / "main_all/predictions/historical_ensemble_scores.jsonl"
OLD_GATE_PROTOCOL = PROJECT_ROOT / "analysis/dense_failure_stage1/trigger_map/frozen_protocol.json"
NORMALIZATION = PROJECT_ROOT / "analysis/dense_failure_stage1/shared_global_gate/training/global_normalization.pt"
BOUND_CODE = (
    "dense_failure_stage1/threshold_calibration.py",
    "experiments/calibrate_stage1_all_source_threshold.py",
)
DATASETS = ("gqa", "chartqa", "textvqa")
SOURCES = ("historical", "canonical")
REFERENCE_TARGETS = (0.99, 0.98, 0.95)
PRIMARY_PRESERVATION = 0.98
OLD_THRESHOLD = 0.9253839280601031
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260903
SANITY_SEED = 20260903
ADEQUATE_CELL_CORRECT_N = 30
USEFUL_POOLED_W_RECALL = 0.05
CROSSFIT_MAX_IQR = 0.05
CROSSFIT_MAX_RANGE = 0.10
CROSSFIT_MIN_HELDOUT_C_PRESERVATION = 0.95


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = {
        key: item
        for key, item in value.items()
        if key not in {"contract_sha256", "head_identity_sha256", "gate_sha256"}
    }
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


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def _artifact_hash_map(manifest: Mapping[str, Any]) -> dict[str, str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Phase-62 artifact manifest lacks artifact list")
    result = {}
    for row in artifacts:
        path = str(row["path"])
        if path in result:
            raise ValueError(f"duplicate Phase-62 artifact path: {path}")
        result[path] = str(row["sha256"])
    return result


def verify_phase62() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    manifest = read_json(PHASE62_MANIFEST)
    protocol = read_json(PHASE62_PROTOCOL)
    if not manifest.get("passed"):
        raise RuntimeError("Phase-62 artifact manifest is not passing")
    if manifest.get("contract_sha256") != protocol.get("contract_sha256"):
        raise RuntimeError("Phase-62 contract mismatch")
    if protocol.get("contract_sha256") != "08dbf1464f5b1d2bdd4c9398ff11ce961a7d59aa1c52967e3505c7fac890a4fa":
        raise RuntimeError("unexpected Phase-62 contract identity")
    hashes = _artifact_hash_map(manifest)
    for path_string, expected in hashes.items():
        path = PHASE62_ROOT / path_string
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"Phase-62 artifact mismatch: {path_string}")
    return manifest, protocol, hashes


def _normalize_score_rows(
    rows: Sequence[Mapping[str, Any]], *, source: str
) -> list[dict[str, Any]]:
    normalized = []
    seen = set()
    for row in rows:
        uid = str(row["uid"])
        if uid in seen:
            raise ValueError(f"duplicate score UID: {uid}")
        seen.add(uid)
        dataset = str(row["dataset"]).lower()
        if dataset not in DATASETS or str(row["source_regime"]) != source:
            raise ValueError(f"invalid source/dataset for {uid}")
        correct = bool(row["current_dense_correct"])
        wrong = bool(row["current_dense_wrong"])
        if correct == wrong:
            raise ValueError(f"non-complementary label fields for {uid}")
        scores = np.asarray([float(row[f"p_{layer}"]) for layer in range(28)])
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError(f"invalid score trajectory for {uid}")
        score_max = float(scores.max())
        if not math.isclose(score_max, float(row["score_max"]), abs_tol=1e-12):
            raise ValueError(f"score_max mismatch for {uid}")
        if source == "canonical":
            fold = int(row["canonical_fold"])
            if fold not in range(5) or int(row["fold"]) != fold:
                raise ValueError(f"invalid canonical fold for {uid}")
            if str(row["model_id"]) != f"all_source_fold_{fold}":
                raise ValueError(f"canonical score uses stale/wrong head for {uid}")
            historical_split = None
        else:
            fold = None
            historical_split = str(row["historical_split"])
            if historical_split not in {"val", "test"}:
                raise ValueError(f"invalid Historical split for {uid}")
            if str(row["model_id"]) != "all_source_five_fold_probability_ensemble":
                raise ValueError(f"Historical score uses stale/wrong head for {uid}")
        normalized.append(
            {
                "uid": uid,
                "dataset": dataset,
                "source_regime": source,
                "historical_split": historical_split,
                "canonical_fold": fold,
                "image_group_id": str(row["image_group_id"]),
                "current_dense_correct": correct,
                "current_dense_wrong": wrong,
                "label": int(wrong),
                "scores": scores,
                "score_max": score_max,
                "model_id": str(row["model_id"]),
            }
        )
    return sorted(normalized, key=lambda row: row["uid"])


def load_populations() -> dict[str, list[dict[str, Any]]]:
    historical = _normalize_score_rows(read_jsonl(HISTORICAL_SCORES), source="historical")
    canonical = _normalize_score_rows(read_jsonl(CANONICAL_SCORES), source="canonical")
    historical_val = [row for row in historical if row["historical_split"] == "val"]
    historical_test = [row for row in historical if row["historical_split"] == "test"]
    if len(canonical) != 4000 or len(historical_val) != 800 or len(historical_test) != 800:
        raise RuntimeError("calibration/evaluation population count mismatch")
    if Counter(row["label"] for row in historical_val) != Counter({0: 400, 1: 400}):
        raise RuntimeError("Historical validation label balance mismatch")
    if Counter(row["label"] for row in historical_test) != Counter({0: 400, 1: 400}):
        raise RuntimeError("Historical test label balance mismatch")
    if Counter(row["canonical_fold"] for row in canonical) != Counter({k: 800 for k in range(5)}):
        raise RuntimeError("Canonical fold count mismatch")
    all_uids = [row["uid"] for row in historical + canonical]
    if len(all_uids) != len(set(all_uids)):
        raise RuntimeError("cross-source UID overlap")
    historical_groups = {row["image_group_id"] for row in historical}
    canonical_groups = {row["image_group_id"] for row in canonical}
    if historical_groups & canonical_groups:
        raise RuntimeError("cross-source image-group overlap")
    return {
        "historical_validation": historical_val,
        "historical_test": historical_test,
        "canonical_oof": canonical,
    }


def _matrix(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([np.asarray(row["scores"], dtype=np.float64) for row in rows]),
        np.asarray([int(row["label"]) for row in rows], dtype=np.int64),
    )


def _population_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "correct": sum(int(row["label"] == 0) for row in rows),
        "wrong": sum(int(row["label"] == 1) for row in rows),
        "image_groups": len({row["image_group_id"] for row in rows}),
        "datasets": dict(sorted(Counter(row["dataset"] for row in rows).items())),
    }


def prepare() -> None:
    _, phase62_protocol, phase62_hashes = verify_phase62()
    populations = load_populations()
    checkpoint_manifest = read_json(CHECKPOINT_MANIFEST)
    checkpoints = checkpoint_manifest.get("checkpoints", [])
    if len(checkpoints) != 5:
        raise RuntimeError("expected exactly five ALL-source fold checkpoints")
    for checkpoint in checkpoints:
        path = PROJECT_ROOT / str(checkpoint["path"])
        if checkpoint.get("contract_sha256") != phase62_protocol["contract_sha256"]:
            raise RuntimeError("checkpoint contract mismatch")
        if not path.is_file() or file_sha256(path) != checkpoint.get("sha256"):
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
    normalization_hash = file_sha256(NORMALIZATION)
    expected_norm = phase62_protocol["input"]["normalization_sha256"]
    if normalization_hash != expected_norm:
        raise RuntimeError("normalization artifact mismatch")
    old_gate = read_json(OLD_GATE_PROTOCOL)
    old_tau = float(old_gate["gate"]["threshold"])
    if old_tau != OLD_THRESHOLD:
        raise RuntimeError("old threshold differs from frozen reference")

    head_manifest: dict[str, Any] = {
        "schema_version": "stage1_all_source_head_manifest_v1",
        "phase62_contract_sha256": phase62_protocol["contract_sha256"],
        "candidate": "Shared Random-4 ALL-source five-fold system",
        "architecture": phase62_protocol["architecture"],
        "normalization": {
            "path": relative(NORMALIZATION),
            "sha256": normalization_hash,
            "identity": "frozen historical global normalization",
        },
        "checkpoints": checkpoints,
        "score_semantics": {
            "canonical_oof": "one held-out-fold ALL-source head per UID",
            "historical": "arithmetic mean of five ALL-source fold probabilities at each layer",
            "deployment_candidate": "the same five checkpoint identities with per-layer probability mean",
            "trajectory": "28 probabilities, layers 0-27",
        },
    }
    head_manifest["head_identity_sha256"] = canonical_hash(head_manifest)

    score_manifest = {
        "schema_version": "stage1_all_source_score_manifest_v1",
        "phase62_contract_sha256": phase62_protocol["contract_sha256"],
        "sources": {
            "canonical_oof": {
                "path": relative(CANONICAL_SCORES),
                "sha256": file_sha256(CANONICAL_SCORES),
                "phase62_expected_sha256": phase62_hashes[relative(CANONICAL_SCORES).removeprefix(relative(PHASE62_ROOT) + "/")],
                "records": 4000,
            },
            "historical_ensemble": {
                "path": relative(HISTORICAL_SCORES),
                "sha256": file_sha256(HISTORICAL_SCORES),
                "phase62_expected_sha256": phase62_hashes[relative(HISTORICAL_SCORES).removeprefix(relative(PHASE62_ROOT) + "/")],
                "records": 1600,
            },
        },
        "strict_comparison": "score(layer) > tau",
        "sample_score": "max of layers 0-27 for threshold sweep",
        "positive_class": "current_dense_wrong",
    }

    identity_rows = []
    for population, rows in populations.items():
        for row in rows:
            identity_rows.append(
                {
                    "uid": row["uid"],
                    "population": population,
                    "source_regime": row["source_regime"],
                    "dataset": row["dataset"],
                    "current_dense_wrong": bool(row["label"]),
                    "image_group_id": row["image_group_id"],
                    "historical_split": row["historical_split"],
                    "canonical_fold": row["canonical_fold"],
                    "model_id": row["model_id"],
                }
            )
    identity_rows.sort(key=lambda row: (row["population"], row["uid"]))
    identity_hash = sha256(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in identity_rows).encode()
    ).hexdigest()
    population_manifest = {
        "schema_version": "stage1_all_source_calibration_population_manifest_v1",
        "selection": {
            "historical": "validation only",
            "canonical": "all five existing out-of-fold partitions",
            "historical_test": "held out from every threshold choice",
        },
        "counts": {name: _population_counts(rows) for name, rows in populations.items()},
        "identity_rows_sha256": identity_hash,
        "records": identity_rows,
    }

    protocol_text = f"""# Stage-1 ALL-source robust threshold calibration protocol

- Candidate: frozen Phase-62 Shared Random-4 ALL-source five-fold system.
- Head identity: `{head_manifest['head_identity_sha256']}`.
- Threshold calibration data: Historical validation (800) plus Canonical OOF (4,000).
- Historical test (800) is excluded from threshold selection and used only after freeze.
- Trigger rule: first layer 0-27 with strict `score > tau`; sample admission is the true any-layer rule.
- Sweep: every observed calibration trajectory-maximum breakpoint plus the all-trigger terminal point.
- Primary rule: maximize pooled W recall subject to Historical and Canonical C preservation both >=98%; use the plan's fixed tie-break order.
- References: most permissive thresholds satisfying 99%, 98%, and 95% worst-source C preservation.
- Useful-W criterion for Decision A: pooled W recall >= {USEFUL_POOLED_W_RECALL:.0%} and nonzero W detections in both calibration sources.
- Adequately supported catastrophic-cell check: N_C >= {ADEQUATE_CELL_CORRECT_N}; flag C preservation <90%.
- Cross-fit stability is declared only when threshold IQR <= {CROSSFIT_MAX_IQR:.2f}, range <= {CROSSFIT_MAX_RANGE:.2f}, the full threshold lies inside the fold range, and every held-out Canonical fold preserves >= {CROSSFIT_MIN_HELDOUT_C_PRESERVATION:.0%} of C.
- Bootstrap: {BOOTSTRAP_DRAWS:,} source-stratified image-group cluster replicates; seed {BOOTSTRAP_SEED}.
- Canonical TextVQA rates receive Wilson 95% intervals because only 19 W records exist.
- Old reference threshold: `{OLD_THRESHOLD:.17g}`, applied to these same frozen ALL-head trajectories.
- No threshold is optimized for trigger depth or downstream Stage-2 outcomes.
"""

    inputs = OUTPUT_ROOT / "inputs"
    write_once_json(inputs / "head_manifest.json", head_manifest)
    write_once_json(inputs / "score_manifest.json", score_manifest)
    write_once_json(inputs / "calibration_population_manifest.json", population_manifest)
    write_once(OUTPUT_ROOT / "protocol.md", protocol_text.encode())
    code_hashes = {path: file_sha256(PROJECT_ROOT / path) for path in BOUND_CODE}
    contract = {
        "schema_version": "stage1_all_source_threshold_calibration_contract_v1",
        "run_id": "stage1_all_source_threshold_calibration_v1",
        "git_commit": command_output(("git", "rev-parse", "HEAD")),
        "git_branch": command_output(("git", "branch", "--show-current")),
        "git_status_porcelain_at_freeze": command_output(("git", "status", "--short")),
        "plan_sha256": file_sha256(PLAN),
        "phase62_contract_sha256": phase62_protocol["contract_sha256"],
        "head_manifest_sha256": file_sha256(inputs / "head_manifest.json"),
        "score_manifest_sha256": file_sha256(inputs / "score_manifest.json"),
        "population_manifest_sha256": file_sha256(inputs / "calibration_population_manifest.json"),
        "protocol_sha256": file_sha256(OUTPUT_ROOT / "protocol.md"),
        "bound_code_sha256": code_hashes,
        "selection": {
            "primary_minimum_worst_source_c_preservation": PRIMARY_PRESERVATION,
            "reference_targets": list(REFERENCE_TARGETS),
            "strict_comparison": "score > tau",
            "useful_pooled_w_recall_minimum": USEFUL_POOLED_W_RECALL,
            "adequate_cell_correct_n": ADEQUATE_CELL_CORRECT_N,
            "crossfit_max_iqr": CROSSFIT_MAX_IQR,
            "crossfit_max_range": CROSSFIT_MAX_RANGE,
            "crossfit_min_heldout_c_preservation": CROSSFIT_MIN_HELDOUT_C_PRESERVATION,
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    write_once_json(inputs / "calibration_contract.json", contract)
    print(json.dumps({"prepared": True, "contract_sha256": contract["contract_sha256"]}))


def verify_prepared() -> dict[str, Any]:
    verify_phase62()
    inputs = OUTPUT_ROOT / "inputs"
    contract = read_json(inputs / "calibration_contract.json")
    if canonical_hash(contract) != contract.get("contract_sha256"):
        raise RuntimeError("calibration contract hash mismatch")
    checks = {
        "head_manifest_sha256": inputs / "head_manifest.json",
        "score_manifest_sha256": inputs / "score_manifest.json",
        "population_manifest_sha256": inputs / "calibration_population_manifest.json",
        "protocol_sha256": OUTPUT_ROOT / "protocol.md",
    }
    for key, path in checks.items():
        if file_sha256(path) != contract[key]:
            raise RuntimeError(f"prepared input mismatch: {path}")
    if file_sha256(PLAN) != contract["plan_sha256"]:
        raise RuntimeError("plan changed after calibration freeze")
    for path, expected in contract["bound_code_sha256"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"calibration code changed after freeze: {path}")
    score_manifest = read_json(inputs / "score_manifest.json")
    for source in score_manifest["sources"].values():
        path = PROJECT_ROOT / source["path"]
        if file_sha256(path) != source["sha256"]:
            raise RuntimeError(f"score input changed after freeze: {path}")
    head_manifest = read_json(inputs / "head_manifest.json")
    if canonical_hash(head_manifest) != head_manifest["head_identity_sha256"]:
        raise RuntimeError("head identity hash mismatch")
    if file_sha256(PROJECT_ROOT / head_manifest["normalization"]["path"]) != head_manifest["normalization"]["sha256"]:
        raise RuntimeError("normalization changed after freeze")
    for checkpoint in head_manifest["checkpoints"]:
        if file_sha256(PROJECT_ROOT / checkpoint["path"]) != checkpoint["sha256"]:
            raise RuntimeError("checkpoint changed after freeze")
    return contract


def _flatten_metrics(prefix: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _evaluate_sources(
    populations: Mapping[str, Sequence[Mapping[str, Any]]], threshold: float
) -> list[dict[str, Any]]:
    rows = []
    named = {
        "historical_validation": populations["historical_validation"],
        "canonical_oof": populations["canonical_oof"],
        "historical_test": populations["historical_test"],
        "pooled_calibration": list(populations["historical_validation"])
        + list(populations["canonical_oof"]),
    }
    for name, values in named.items():
        scores, labels = _matrix(values)
        rows.append({"population": name, **operating_metrics(scores, labels, threshold)})
    return rows


def _operating_points(sweep: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points = []
    primary = select_primary_threshold(sweep, minimum_preservation=PRIMARY_PRESERVATION)
    for target in REFERENCE_TARGETS:
        row = most_permissive_at_preservation(sweep, target)
        points.append(
            {
                "operating_point": f"reference_{int(target * 100)}",
                "target_worst_source_c_preservation": target,
                "is_primary": bool(target == PRIMARY_PRESERVATION and row["threshold"] == primary["threshold"]),
                **row,
            }
        )
    points.append(
        {
            "operating_point": "primary_98",
            "target_worst_source_c_preservation": PRIMARY_PRESERVATION,
            "is_primary": True,
            **primary,
        }
    )
    return points


def _crossfit(
    historical: Sequence[Mapping[str, Any]], canonical: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    historical_scores, historical_labels = _matrix(historical)
    stability = []
    for fold in range(5):
        calibration = [row for row in canonical if int(row["canonical_fold"]) != fold]
        heldout = [row for row in canonical if int(row["canonical_fold"]) == fold]
        c_scores, c_labels = _matrix(calibration)
        heldout_scores, heldout_labels = _matrix(heldout)
        sweep = robust_threshold_sweep(
            historical_scores,
            historical_labels,
            c_scores,
            c_labels,
            include_depth=False,
        )
        selected = select_primary_threshold(sweep, minimum_preservation=PRIMARY_PRESERVATION)
        output_rows = []
        for row in sweep:
            heldout_metrics = operating_metrics(
                heldout_scores, heldout_labels, float(row["threshold"]), include_depth=False
            )
            output_rows.append(
                {
                    "fold": fold,
                    "selected": bool(row["threshold"] == selected["threshold"]),
                    **row,
                    **_flatten_metrics("heldout_canonical", heldout_metrics),
                }
            )
        atomic_csv(OUTPUT_ROOT / "crossfit" / f"fold_{fold}_threshold_sweep.csv", output_rows)
        selected_heldout = operating_metrics(
            heldout_scores, heldout_labels, float(selected["threshold"])
        )
        stability.append(
            {
                "fold": fold,
                "threshold": selected["threshold"],
                "calibration_worst_source_c_preservation": selected["worst_source_c_preservation"],
                "calibration_historical_c_preservation": selected["historical_c_preservation"],
                "calibration_canonical_c_preservation": selected["canonical_c_preservation"],
                "calibration_historical_w_recall": selected["historical_w_recall"],
                "calibration_canonical_w_recall": selected["canonical_w_recall"],
                "heldout_canonical_records": selected_heldout["records"],
                "heldout_canonical_c_preservation": selected_heldout["c_preservation"],
                "heldout_canonical_w_recall": selected_heldout["w_recall"],
                "heldout_canonical_trigger_precision": selected_heldout["trigger_precision"],
                "heldout_canonical_trigger_rate": selected_heldout["trigger_rate"],
                "heldout_canonical_median_first_trigger_layer": selected_heldout["median_first_trigger_layer"],
            }
        )
    thresholds = np.asarray([float(row["threshold"]) for row in stability])
    summary = {
        "mean": float(np.mean(thresholds)),
        "median": float(np.median(thresholds)),
        "q25": float(np.quantile(thresholds, 0.25)),
        "q75": float(np.quantile(thresholds, 0.75)),
        "iqr": float(np.quantile(thresholds, 0.75) - np.quantile(thresholds, 0.25)),
        "minimum": float(thresholds.min()),
        "maximum": float(thresholds.max()),
        "range": float(thresholds.max() - thresholds.min()),
        "minimum_heldout_canonical_c_preservation": min(
            float(row["heldout_canonical_c_preservation"]) for row in stability
        ),
    }
    return stability, summary


def _cell_rows(
    populations: Mapping[str, Sequence[Mapping[str, Any]]],
    points: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for point in points:
        threshold = float(point["threshold"])
        for source, values in (
            ("historical", populations["historical_validation"]),
            ("canonical", populations["canonical_oof"]),
        ):
            for dataset in DATASETS:
                subset = [row for row in values if row["dataset"] == dataset]
                scores, labels = _matrix(subset)
                metrics = operating_metrics(scores, labels, threshold)
                c_preserved = metrics["correct"] - metrics["correct_false_admissions"]
                c_low, c_high = wilson_interval(c_preserved, metrics["correct"])
                w_low, w_high = wilson_interval(metrics["wrong_detected"], metrics["wrong"])
                p_low, p_high = wilson_interval(metrics["wrong_detected"], metrics["triggered"])
                rows.append(
                    {
                        "operating_point": point["operating_point"],
                        "threshold": threshold,
                        "source": source,
                        "dataset": dataset,
                        "n_correct": metrics["correct"],
                        "n_wrong": metrics["wrong"],
                        "c_preservation": metrics["c_preservation"],
                        "c_preservation_ci_low": c_low,
                        "c_preservation_ci_high": c_high,
                        "w_recall": metrics["w_recall"],
                        "w_recall_ci_low": w_low,
                        "w_recall_ci_high": w_high,
                        "trigger_precision": metrics["trigger_precision"],
                        "trigger_precision_ci_low": p_low,
                        "trigger_precision_ci_high": p_high,
                        "median_first_trigger_layer": metrics["median_first_trigger_layer"],
                        "adequately_supported_c_cell": metrics["correct"] >= ADEQUATE_CELL_CORRECT_N,
                        "catastrophic_c_preservation_warning": bool(
                            metrics["correct"] >= ADEQUATE_CELL_CORRECT_N
                            and metrics["c_preservation"] < 0.90
                        ),
                    }
                )
    return rows


def _depth_rows(
    populations: Mapping[str, Sequence[Mapping[str, Any]]],
    points: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for point in points:
        threshold = float(point["threshold"])
        for source, values in (
            ("historical", populations["historical_validation"]),
            ("canonical", populations["canonical_oof"]),
        ):
            scores, labels = _matrix(values)
            first = first_trigger_layers(scores, threshold)
            for label, outcome in ((0, "false_trigger_c"), (1, "true_trigger_w")):
                selected = first[(labels == label) & (first >= 0)]
                for depth_bin, layers in DEPTH_BINS.items():
                    count = int(np.isin(selected, layers).sum())
                    rows.append(
                        {
                            "operating_point": point["operating_point"],
                            "threshold": threshold,
                            "source": source,
                            "outcome": outcome,
                            "depth_bin": depth_bin,
                            "triggered_records": int(len(selected)),
                            "count": count,
                            "fraction": float(count / len(selected)) if len(selected) else float("nan"),
                            "median_first_trigger_layer": float(np.median(selected)) if len(selected) else float("nan"),
                            "q25_first_trigger_layer": float(np.quantile(selected, 0.25)) if len(selected) else float("nan"),
                            "q75_first_trigger_layer": float(np.quantile(selected, 0.75)) if len(selected) else float("nan"),
                        }
                    )
    return rows


def _bootstrap_rows(
    calibration_rows: Sequence[Mapping[str, Any]], points: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    metrics = (
        "historical_c_preservation",
        "canonical_c_preservation",
        "historical_w_recall",
        "canonical_w_recall",
        "pooled_w_recall",
        "trigger_precision",
        "c_preservation_source_difference",
        "w_recall_source_difference",
    )
    for index, point in enumerate(points):
        draws = group_bootstrap_metrics(
            calibration_rows,
            threshold=float(point["threshold"]),
            draws=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED + index,
        )
        for metric in metrics:
            values = np.asarray([row[metric] for row in draws], dtype=np.float64)
            values = values[np.isfinite(values)]
            if not len(values):
                raise RuntimeError(f"bootstrap produced no finite {metric} values")
            output.append(
                {
                    "operating_point": point["operating_point"],
                    "threshold": point["threshold"],
                    "metric": metric,
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "draws": len(values),
                    "seed": BOOTSTRAP_SEED + index,
                    "resampling_unit": "image_group_cluster_stratified_by_source",
                }
            )
    return output


def _sanity_rows(
    calibration_rows: Sequence[Mapping[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SANITY_SEED)
    categories: dict[str, list[Mapping[str, Any]]] = {
        "triggered_c": [row for row in calibration_rows if row["label"] == 0 and row["score_max"] > threshold],
        "triggered_w": [row for row in calibration_rows if row["label"] == 1 and row["score_max"] > threshold],
        "near_threshold_c": sorted(
            [row for row in calibration_rows if row["label"] == 0],
            key=lambda row: (abs(row["score_max"] - threshold), row["uid"]),
        )[:10],
        "near_threshold_w": sorted(
            [row for row in calibration_rows if row["label"] == 1],
            key=lambda row: (abs(row["score_max"] - threshold), row["uid"]),
        )[:10],
    }
    output = []
    for category, candidates in categories.items():
        if category.startswith("triggered") and len(candidates) > 10:
            indices = sorted(rng.choice(len(candidates), size=10, replace=False).tolist())
            candidates = [candidates[index] for index in indices]
        for row in candidates:
            first = int(first_trigger_layers(np.asarray(row["scores"])[None, :], threshold)[0])
            expected = [
                layer for layer, score in enumerate(row["scores"]) if float(score) > threshold
            ]
            if first != (expected[0] if expected else -1):
                raise RuntimeError("first-trigger sanity mismatch")
            if not row["model_id"].startswith("all_source_"):
                raise RuntimeError("sanity row uses a stale head")
            output.append(
                {
                    "category": category,
                    "uid": row["uid"],
                    "source_regime": row["source_regime"],
                    "dataset": row["dataset"],
                    "label": row["label"],
                    "model_id": row["model_id"],
                    "threshold": threshold,
                    "score_max": row["score_max"],
                    "triggered": bool(first >= 0),
                    "first_trigger_layer": first,
                    "score_at_first_trigger": float(row["scores"][first]) if first >= 0 else None,
                    "strict_comparison_verified": True,
                }
            )
    return output


def _plot_outputs(
    sweep: Sequence[Mapping[str, Any]],
    points: Sequence[Mapping[str, Any]],
    stability: Sequence[Mapping[str, Any]],
    dataset_rows: Sequence[Mapping[str, Any]],
    depth_rows: Sequence[Mapping[str, Any]],
) -> None:
    figures = OUTPUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    c = np.asarray([float(row["worst_source_c_preservation"]) for row in sweep])
    hw = np.asarray([float(row["historical_w_recall"]) for row in sweep])
    cw = np.asarray([float(row["canonical_w_recall"]) for row in sweep])
    pw = np.asarray([float(row["pooled_w_recall"]) for row in sweep])
    thresholds = np.asarray([float(row["threshold"]) for row in sweep])
    precision = np.asarray([float(row["pooled_trigger_precision"]) for row in sweep])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(c, hw, label="Historical W recall")
    ax.plot(c, cw, label="Canonical W recall")
    ax.plot(c, pw, label="Pooled W recall", linewidth=2)
    for point in points[:3]:
        ax.scatter(point["worst_source_c_preservation"], point["pooled_w_recall"], s=35)
    ax.set(xlabel="Worst-source C preservation", ylabel="W recall", title="Preservation vs wrong recall")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "preservation_vs_wrong_recall.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([row["historical_c_preservation"] for row in sweep], hw, label="Historical")
    ax.plot([row["canonical_c_preservation"] for row in sweep], cw, label="Canonical")
    ax.set(xlabel="Source-specific C preservation", ylabel="Source-specific W recall", title="Source-specific Pareto curves")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(figures / "source_specific_pareto.png", dpi=180); plt.close(fig)

    for filename, y1, y2, ylabel, title in (
        ("threshold_vs_source_preservation.png", [row["historical_c_preservation"] for row in sweep], [row["canonical_c_preservation"] for row in sweep], "C preservation", "Threshold vs source preservation"),
        ("threshold_vs_source_wrong_recall.png", hw, cw, "W recall", "Threshold vs source wrong recall"),
    ):
        order = np.argsort(thresholds)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(thresholds[order], np.asarray(y1)[order], label="Historical")
        ax.plot(thresholds[order], np.asarray(y2)[order], label="Canonical")
        ax.set(xlabel="Threshold", ylabel=ylabel, title=title); ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
        fig.savefig(figures / filename, dpi=180); plt.close(fig)

    order = np.argsort(thresholds)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds[order], precision[order])
    ax.set(xlabel="Threshold", ylabel="Pooled trigger precision", title="Threshold vs trigger precision"); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(figures / "threshold_vs_trigger_precision.png", dpi=180); plt.close(fig)

    primary_depth = [row for row in depth_rows if row["operating_point"] == "primary_98"]
    labels = []
    stacks = {name: [] for name in DEPTH_BINS}
    for source in SOURCES:
        for outcome in ("true_trigger_w", "false_trigger_c"):
            labels.append(f"{source[:4]}\n{outcome.replace('_trigger_', '-')}" )
            for depth_bin in DEPTH_BINS:
                match = next(row for row in primary_depth if row["source"] == source and row["outcome"] == outcome and row["depth_bin"] == depth_bin)
                stacks[depth_bin].append(0.0 if not math.isfinite(float(match["fraction"])) else float(match["fraction"]))
    fig, ax = plt.subplots(figsize=(8, 5)); bottom = np.zeros(len(labels))
    for name in DEPTH_BINS:
        values = np.asarray(stacks[name]); ax.bar(labels, values, bottom=bottom, label=name); bottom += values
    ax.set(ylabel="Fraction among triggers", title="Primary-gate trigger depth"); ax.legend(); fig.tight_layout()
    fig.savefig(figures / "trigger_depth_by_source_and_outcome.png", dpi=180); plt.close(fig)

    primary_cells = [row for row in dataset_rows if row["operating_point"] == "primary_98"]
    for metric, filename, title in (
        ("c_preservation", "dataset_source_preservation_heatmap.png", "Primary C preservation"),
        ("w_recall", "dataset_source_wrong_recall_heatmap.png", "Primary W recall"),
    ):
        matrix = np.asarray([[next(float(row[metric]) for row in primary_cells if row["source"] == source and row["dataset"] == dataset) for dataset in DATASETS] for source in SOURCES])
        fig, ax = plt.subplots(figsize=(6, 3.5)); image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(3), DATASETS); ax.set_yticks(range(2), SOURCES)
        for i in range(2):
            for j in range(3): ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", color="white" if matrix[i,j] < 0.65 else "black")
        ax.set_title(title); fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(figures / filename, dpi=180); plt.close(fig)

    fold_ids = [int(row["fold"]) for row in stability]
    fold_tau = [float(row["threshold"]) for row in stability]
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.plot(fold_ids, fold_tau, marker="o")
    ax.set(xticks=fold_ids, xlabel="Held-out Canonical fold", ylabel="Selected threshold", title="Cross-fold threshold stability"); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(figures / "cross_fold_threshold_stability.png", dpi=180); plt.close(fig)


def run() -> None:
    contract = verify_prepared()
    populations = load_populations()
    h_scores, h_labels = _matrix(populations["historical_validation"])
    c_scores, c_labels = _matrix(populations["canonical_oof"])
    sweep = robust_threshold_sweep(
        h_scores, h_labels, c_scores, c_labels, include_depth=True
    )
    points = _operating_points(sweep)
    primary = next(row for row in points if row["operating_point"] == "primary_98")
    atomic_csv(OUTPUT_ROOT / "metrics/full_threshold_sweep.csv", sweep)
    atomic_csv(OUTPUT_ROOT / "metrics/reference_operating_points.csv", points)

    stability, stability_summary = _crossfit(
        populations["historical_validation"], populations["canonical_oof"]
    )
    full_tau = float(primary["threshold"])
    crossfit_stable = bool(
        stability_summary["iqr"] <= CROSSFIT_MAX_IQR
        and stability_summary["range"] <= CROSSFIT_MAX_RANGE
        and stability_summary["minimum"] <= full_tau <= stability_summary["maximum"]
        and stability_summary["minimum_heldout_canonical_c_preservation"]
        >= CROSSFIT_MIN_HELDOUT_C_PRESERVATION
    )
    stability_summary["full_selected_threshold"] = full_tau
    stability_summary["stable"] = crossfit_stable
    stability_summary["criteria"] = {
        "max_iqr": CROSSFIT_MAX_IQR,
        "max_range": CROSSFIT_MAX_RANGE,
        "full_threshold_inside_fold_range": True,
        "minimum_heldout_canonical_c_preservation": CROSSFIT_MIN_HELDOUT_C_PRESERVATION,
    }
    atomic_csv(OUTPUT_ROOT / "crossfit/threshold_stability.csv", stability)
    atomic_json(OUTPUT_ROOT / "crossfit/threshold_stability_summary.json", stability_summary)

    named_points = list(points)
    named_points.append({"operating_point": "old_threshold", "threshold": OLD_THRESHOLD})
    source_rows = []
    for point in named_points:
        for metrics in _evaluate_sources(populations, float(point["threshold"])):
            source_rows.append(
                {
                    "operating_point": point["operating_point"],
                    "threshold": point["threshold"],
                    **metrics,
                }
            )
    atomic_csv(OUTPUT_ROOT / "metrics/source_breakdown.csv", source_rows)

    cell_rows = _cell_rows(populations, named_points)
    atomic_csv(OUTPUT_ROOT / "metrics/dataset_source_breakdown.csv", cell_rows)
    calibration_rows = list(populations["historical_validation"]) + list(populations["canonical_oof"])
    bootstrap_rows = _bootstrap_rows(calibration_rows, points)
    atomic_csv(OUTPUT_ROOT / "metrics/bootstrap_intervals.csv", bootstrap_rows)
    depth_rows = _depth_rows(populations, named_points)
    atomic_csv(OUTPUT_ROOT / "metrics/trigger_depth_breakdown.csv", depth_rows)

    comparison = [
        row
        for row in source_rows
        if row["operating_point"] in {"primary_98", "old_threshold"}
    ]
    atomic_csv(OUTPUT_ROOT / "metrics/old_vs_new_gate_comparison.csv", comparison)
    sanity = _sanity_rows(calibration_rows, full_tau)
    atomic_jsonl(OUTPUT_ROOT / "metrics/score_trajectory_sanity.jsonl", sanity)

    primary_cells = [
        row for row in cell_rows if row["operating_point"] == "primary_98"
    ]
    catastrophic = [row for row in primary_cells if row["catastrophic_c_preservation_warning"]]
    useful_w = bool(
        float(primary["pooled_w_recall"]) >= USEFUL_POOLED_W_RECALL
        and int(primary["historical_wrong_detected"]) > 0
        and int(primary["canonical_wrong_detected"]) > 0
    )
    decision = choose_freeze_decision(
        primary_metrics=primary,
        useful_w_detection=useful_w,
        crossfit_stable=crossfit_stable,
        catastrophic_cells=len(catastrophic),
    )
    _plot_outputs(sweep, points, stability, cell_rows, depth_rows)

    reference_by_name = {row["operating_point"]: row for row in points}
    hist_test = next(
        row for row in source_rows if row["operating_point"] == "primary_98" and row["population"] == "historical_test"
    )
    canonical_chart = next(
        row for row in primary_cells if row["source"] == "canonical" and row["dataset"] == "chartqa"
    )
    canonical_text = next(
        row for row in primary_cells if row["source"] == "canonical" and row["dataset"] == "textvqa"
    )
    primary_depth_rows = [
        row for row in depth_rows if row["operating_point"] == "primary_98"
    ]
    canonical_l0_false = sum(
        1
        for row in populations["canonical_oof"]
        if row["label"] == 0 and float(row["scores"][0]) > full_tau
    )
    old_canonical_l0_false = sum(
        1
        for row in populations["canonical_oof"]
        if row["label"] == 0 and float(row["scores"][0]) > OLD_THRESHOLD
    )
    summary = f"""# ALL-source robust threshold calibration summary

Calibration contract: `{contract['contract_sha256']}`. The selection population is Historical validation (800) plus Canonical OOF (4,000); Historical test is untouched by selection.

## Decision

**{decision}.** The primary threshold is `{full_tau:.17g}` under strict `score > tau`.

- Worst-source C preservation: {primary['worst_source_c_preservation']:.4f} (Historical {primary['historical_c_preservation']:.4f}; Canonical {primary['canonical_c_preservation']:.4f}).
- W recall: Historical {primary['historical_w_recall']:.4f}; Canonical {primary['canonical_w_recall']:.4f}; pooled {primary['pooled_w_recall']:.4f}.
- Trigger precision: {primary['pooled_trigger_precision']:.4f}; median first-trigger layer: {primary['pooled_median_first_trigger_layer']:.1f}.
- Untouched Historical test: C preservation {hist_test['c_preservation']:.4f}, W recall {hist_test['w_recall']:.4f}, trigger precision {hist_test['trigger_precision']:.4f}.

## Reference operating points

| Constraint | Threshold | Hist C preserve | Canon C preserve | Hist W recall | Canon W recall | Pooled W recall | Precision |
|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for name in ("reference_99", "reference_98", "reference_95"):
        row = reference_by_name[name]
        summary += f"| {name.removeprefix('reference_')}% | {row['threshold']:.6f} | {row['historical_c_preservation']:.4f} | {row['canonical_c_preservation']:.4f} | {row['historical_w_recall']:.4f} | {row['canonical_w_recall']:.4f} | {row['pooled_w_recall']:.4f} | {row['pooled_trigger_precision']:.4f} |\n"
    summary += f"""

## Stability and cell safety

- Cross-fit thresholds: mean {stability_summary['mean']:.6f}, median {stability_summary['median']:.6f}, IQR {stability_summary['iqr']:.6f}, range [{stability_summary['minimum']:.6f}, {stability_summary['maximum']:.6f}]; fixed stability result: **{crossfit_stable}**.
- Minimum held-out Canonical-fold C preservation: {stability_summary['minimum_heldout_canonical_c_preservation']:.4f}.
- Catastrophic adequately supported cells: {len(catastrophic)}.
- Canonical ChartQA: C preservation {canonical_chart['c_preservation']:.4f}, W recall {canonical_chart['w_recall']:.4f}, precision {canonical_chart['trigger_precision']:.4f}.
- Canonical TextVQA (N_W={canonical_text['n_wrong']}): W recall {canonical_text['w_recall']:.4f}, Wilson 95% CI [{canonical_text['w_recall_ci_low']:.4f}, {canonical_text['w_recall_ci_high']:.4f}].

## Old-gate context and trigger timing

- The old threshold `{OLD_THRESHOLD:.6f}` and the new threshold are evaluated on identical ALL-head trajectories in `metrics/old_vs_new_gate_comparison.csv`.
- Canonical Dense-C first-layer-0 false triggers under these ALL-head scores: old threshold {old_canonical_l0_false}; new threshold {canonical_l0_false}. This comparison concerns threshold behavior on the repaired ALL head, not reuse of the stale Historical-only head.
- Full early/middle/late distributions for true-trigger W and false-trigger C are in `metrics/trigger_depth_breakdown.csv`; threshold timing was not optimized.

## Answers to the plan questions

1. The complete empirical Pareto sweep is `metrics/full_threshold_sweep.csv` and the main preservation-vs-recall view is `figures/preservation_vs_wrong_recall.png`.
2. The 99/98/95 thresholds and metrics are listed above and frozen in `metrics/reference_operating_points.csv`.
3. Historical and Canonical W recall at each reference point are reported above.
4. The primary 98%-constraint threshold is `{full_tau:.17g}`.
5. Cross-fit stability is **{crossfit_stable}** under the prospective criteria in `protocol.md`.
6. Source-specific preservation/recall differences and group-bootstrap intervals are in `metrics/bootstrap_intervals.csv`.
7. {len(catastrophic)} adequately supported dataset/source cell(s) are catastrophic at the primary point.
8. Canonical ChartQA safety is {'acceptable under the >=90% cell rule' if not canonical_chart['catastrophic_c_preservation_warning'] else 'not acceptable under the fixed cell rule'}.
9. The ALL-head layer-0 canonical false-trigger count changes from {old_canonical_l0_false} at the old threshold to {canonical_l0_false} at the new threshold.
10. The old/new same-population comparison is frozen in `metrics/old_vs_new_gate_comparison.csv`.
"""
    _atomic_bytes(OUTPUT_ROOT / "summaries/threshold_calibration_summary.md", summary.encode())

    decision_text = f"""# Stage-1 robust gate freeze decision

## {decision}

- Calibration contract: `{contract['contract_sha256']}`.
- Primary strict any-layer threshold: `{full_tau:.17g}`.
- Worst-source C preservation: `{primary['worst_source_c_preservation']:.6f}`.
- Pooled W recall: `{primary['pooled_w_recall']:.6f}`; Historical/Canonical: `{primary['historical_w_recall']:.6f}` / `{primary['canonical_w_recall']:.6f}`.
- Cross-fit stable: `{crossfit_stable}`.
- Catastrophic adequately supported cells: `{len(catastrophic)}`.
- Useful-W criterion passed: `{useful_w}`.

"""
    if decision.startswith("A"):
        decision_text += "The gate is frozen for a separately authorized trigger-map and Stage-2 compatibility audit. That next phase was not executed.\n"
        head_manifest = read_json(OUTPUT_ROOT / "inputs/head_manifest.json")
        population_manifest = read_json(OUTPUT_ROOT / "inputs/calibration_population_manifest.json")
        gate = {
            "schema_version": "robust_stage1_gate_v1",
            "calibration_contract_sha256": contract["contract_sha256"],
            "phase62_contract_sha256": contract["phase62_contract_sha256"],
            "head_identity_sha256": head_manifest["head_identity_sha256"],
            "checkpoints": head_manifest["checkpoints"],
            "normalization": head_manifest["normalization"],
            "architecture": head_manifest["architecture"],
            "global_threshold": full_tau,
            "comparison": "strict greater-than",
            "trigger_rule": "first layer 0-27 whose score exceeds global_threshold",
            "calibration_population_sha256": population_manifest["identity_rows_sha256"],
            "calibration_score_manifest_sha256": contract["score_manifest_sha256"],
            "selection_rule": "maximum pooled ALL W recall subject to >=98% Historical and Canonical C preservation, then frozen tie-breaks",
            "metrics": {
                "worst_source_c_preservation": primary["worst_source_c_preservation"],
                "historical_c_preservation": primary["historical_c_preservation"],
                "canonical_c_preservation": primary["canonical_c_preservation"],
                "historical_w_recall": primary["historical_w_recall"],
                "canonical_w_recall": primary["canonical_w_recall"],
                "pooled_w_recall": primary["pooled_w_recall"],
                "trigger_precision": primary["pooled_trigger_precision"],
            },
            "scope": "observed Historical+Canonical GQA/ChartQA/TextVQA mixture; not benchmark-universal",
            "code_commit": contract["git_commit"],
            "bound_code_sha256": contract["bound_code_sha256"],
        }
        gate["gate_sha256"] = canonical_hash(gate)
        write_once_json(OUTPUT_ROOT / "frozen/robust_stage1_gate.json", gate)
    else:
        decision_text += "No robust gate artifact was frozen. The 95% point remains diagnostic and requires explicit review before use.\n"
        if (OUTPUT_ROOT / "frozen/robust_stage1_gate.json").exists():
            raise RuntimeError("a gate artifact exists despite a non-freeze decision")
    _atomic_bytes(OUTPUT_ROOT / "summaries/stage1_gate_freeze_decision.md", decision_text.encode())
    atomic_json(
        OUTPUT_ROOT / "run_summary.json",
        {
            "contract_sha256": contract["contract_sha256"],
            "decision": decision,
            "primary_threshold": full_tau,
            "crossfit_stable": crossfit_stable,
            "useful_w_detection": useful_w,
            "catastrophic_cells": len(catastrophic),
            "historical_test_excluded_from_selection": True,
        },
    )
    print(json.dumps(read_json(OUTPUT_ROOT / "run_summary.json"), sort_keys=True))


def finalize() -> None:
    contract = verify_prepared()
    required = [
        "protocol.md",
        "inputs/head_manifest.json",
        "inputs/score_manifest.json",
        "inputs/calibration_population_manifest.json",
        "inputs/calibration_contract.json",
        *(f"crossfit/fold_{fold}_threshold_sweep.csv" for fold in range(5)),
        "crossfit/threshold_stability.csv",
        "crossfit/threshold_stability_summary.json",
        "metrics/full_threshold_sweep.csv",
        "metrics/reference_operating_points.csv",
        "metrics/source_breakdown.csv",
        "metrics/dataset_source_breakdown.csv",
        "metrics/bootstrap_intervals.csv",
        "metrics/trigger_depth_breakdown.csv",
        "metrics/old_vs_new_gate_comparison.csv",
        "metrics/score_trajectory_sanity.jsonl",
        "figures/preservation_vs_wrong_recall.png",
        "figures/source_specific_pareto.png",
        "figures/threshold_vs_source_preservation.png",
        "figures/threshold_vs_source_wrong_recall.png",
        "figures/threshold_vs_trigger_precision.png",
        "figures/trigger_depth_by_source_and_outcome.png",
        "figures/dataset_source_preservation_heatmap.png",
        "figures/dataset_source_wrong_recall_heatmap.png",
        "figures/cross_fold_threshold_stability.png",
        "summaries/threshold_calibration_summary.md",
        "summaries/stage1_gate_freeze_decision.md",
        "run_summary.json",
    ]
    summary = read_json(OUTPUT_ROOT / "run_summary.json")
    if summary["decision"].startswith("A"):
        required.append("frozen/robust_stage1_gate.json")
        gate = read_json(OUTPUT_ROOT / "frozen/robust_stage1_gate.json")
        if canonical_hash(gate) != gate.get("gate_sha256"):
            raise RuntimeError("frozen gate hash mismatch")
    missing = [path for path in required if not (OUTPUT_ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"missing calibration artifacts: {missing}")
    for fold in range(5):
        rows = read_csv(OUTPUT_ROOT / "crossfit" / f"fold_{fold}_threshold_sweep.csv")
        if sum(row["selected"] == "True" for row in rows) != 1:
            raise RuntimeError(f"fold {fold} lacks one selected threshold")
    if len(read_csv(OUTPUT_ROOT / "metrics/reference_operating_points.csv")) != 4:
        raise RuntimeError("reference operating-point count mismatch")
    if len(read_jsonl(OUTPUT_ROOT / "metrics/score_trajectory_sanity.jsonl")) < 20:
        raise RuntimeError("score-trajectory sanity sample is too small")
    artifacts = []
    for path_string in required:
        path = OUTPUT_ROOT / path_string
        artifacts.append(
            {"path": path_string, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        )
    manifest = {
        "schema_version": "stage1_all_source_threshold_calibration_artifact_manifest_v1",
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        "decision": summary["decision"],
        "primary_threshold": summary["primary_threshold"],
        "historical_test_excluded_from_selection": True,
        "artifacts": artifacts,
    }
    atomic_json(OUTPUT_ROOT / "artifact_manifest.json", manifest)
    for row in artifacts:
        if file_sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"post-write artifact mismatch: {row['path']}")
    print(json.dumps({"finalized": True, "artifacts": len(artifacts), "decision": summary["decision"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "finalize", "all"))
    args = parser.parse_args()
    if args.action in {"prepare", "all"}:
        prepare()
    if args.action in {"run", "all"}:
        run()
    if args.action in {"finalize", "all"}:
        finalize()


if __name__ == "__main__":
    main()
