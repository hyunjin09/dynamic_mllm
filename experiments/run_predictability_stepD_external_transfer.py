#!/usr/bin/env python3
"""Freeze and execute Predictability Step-D external transfer.

This runner keeps external state construction, prediction freezing, and
counterfactual measurement as separate commands so utility labels cannot leak
into refits or predictions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from binary_policy.executor import (  # noqa: E402
    capture_four_action_route,
    capture_four_action_suffix_from_full_baseline,
    greedy_generate_from_cached_prompt,
)
from binary_policy.executor.four_action import score_token_ids_from_cached_prompt  # noqa: E402
from binary_policy.executor.inputs import build_binary_inputs  # noqa: E402
from dense_failure_stage1.contract import model_file_hashes  # noqa: E402
from dense_failure_stage2.predictability_external_transfer import (  # noqa: E402
    EXTERNAL_BENCHMARK_COUNTS,
    benchmark_family,
    canonical_question_text,
    external_answer_specs,
    file_sha256,
    freeze_prediction_hashes,
    stage1_transfer_nuisance,
    stage2_transfer_nuisance,
    strict_first_trigger,
    validate_external_population,
    validate_prediction_census,
)
from dense_failure_stage2.predictability_generalization import (  # noqa: E402
    last_token_pool,
    normalize_question,
)
from dense_failure_stage2.predictability_learnability import (  # noqa: E402
    binary_classification_metrics,
    regression_metrics,
    robust_target_scale,
    select_preservation_threshold,
)
from dense_failure_stage2.predictability_measurement import (  # noqa: E402
    ACTIONS,
    aggregate_reference_mean_logprobs,
    derive_utility_row,
    validate_complete_state_results,
)
from experiments import run_full_benchmark_end_to_end_eval as phase69  # noqa: E402
from experiments import run_predictability_stepA_measurement as stepa  # noqa: E402
from experiments import run_predictability_stepB_learnability as stepb  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/predictability_stepD_external_transfer_v1.json"
ALLOWED_ROOTS = (REPO_ROOT.resolve(), Path("/mnt/hyemin").resolve())
BOUND_CODE = (
    "configs/predictability_stepD_external_transfer_v1.json",
    "dense_failure_stage2/predictability_external_transfer.py",
    "experiments/run_predictability_stepD_external_transfer.py",
    "experiments/aggregate_predictability_stepD_external_transfer.py",
    "experiments/run_predictability_stepA_measurement.py",
    "experiments/run_predictability_stepB_learnability.py",
    "experiments/run_predictability_stepC_generalization.py",
    "experiments/run_full_benchmark_end_to_end_eval.py",
    "dense_failure_stage2/predictability_measurement.py",
    "dense_failure_stage2/predictability_learnability.py",
    "dense_failure_stage2/predictability_generalization.py",
    "dense_failure_stage2/full_benchmark_eval.py",
    "dense_failure_stage1/shared_global_gate.py",
    "dense_failure_stage1/runtime.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/model.py",
    "scoring/benchmark_metrics.py",
    "scoring/reference_likelihood.py",
    "eval/reference/shared_prefix_eval_20260812/EVAL_PROTOCOL.md",
    "eval/reference/shared_prefix_eval_20260812/README.md",
    "eval/reference/shared_prefix_eval_20260812/REFERENCE_RESULT.md",
    "plans/predictability_phase_stepD_full_external_transfer_plan.md",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    if not any(resolved == root or resolved.is_relative_to(root) for root in ALLOWED_ROOTS):
        raise ValueError(f"path escapes allowed roots: {value}")
    return resolved


def read_json(value: str | Path) -> dict[str, Any]:
    with resolve_path(value).open() as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {value}")
    return result


def read_jsonl(value: str | Path) -> list[dict[str, Any]]:
    rows = []
    with resolve_path(value).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"expected JSON object at {value}:{line_number}")
                rows.append(row)
    return rows


def read_csv(value: str | Path) -> list[dict[str, str]]:
    with resolve_path(value).open(newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic(path, "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows).encode())


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    # Several evidence tables intentionally combine classification and
    # regression rows, whose metric schemas differ.  Preserve first-seen field
    # order while retaining every declared column.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    import io

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    _atomic(path, output.getvalue().encode())


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def command_output(arguments: Sequence[str]) -> str:
    return subprocess.check_output(list(arguments), cwd=REPO_ROOT, text=True).strip()


def git_state() -> dict[str, str]:
    return {
        "commit": command_output(("git", "rev-parse", "HEAD")),
        "branch": command_output(("git", "branch", "--show-current")),
        "worktree_status": command_output(("git", "status", "--short")),
    }


def runtime_state() -> dict[str, Any]:
    packages = (
        "torch", "transformers", "numpy", "pandas", "matplotlib", "lmms-eval",
        "qwen-vl-utils", "Pillow", "av",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return {
        "python": sys.version.split()[0],
        "packages": versions,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }


def _verify_artifact_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    entries = manifest["files"]
    if isinstance(entries, dict):
        iterable = entries.items()
    else:
        iterable = ((str(row["path"]), str(row["sha256"])) for row in entries)
    checked = 0
    for relative, expected in iterable:
        path = root / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"parent artifact differs: {path}")
        checked += 1
    # Historical phases used more than one convention for a manifest's
    # optional self-hash (for example, hashing before adding the self-hash).
    # Verify every declared artifact byte-for-byte and bind the complete
    # manifest file itself into the Step-D contract instead of guessing that
    # historical convention here.
    return {
        "path": str(manifest_path),
        "sha256": file_sha256(manifest_path),
        "declared_artifact_manifest_sha256": manifest.get("artifact_manifest_sha256"),
        "files_verified": checked,
    }


def _robust_head_artifacts(config: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve and verify every file used by the frozen robust Stage-1 gate."""

    manifest_path = resolve_path(config["robust_p90"]["head_manifest"])
    manifest = read_json(manifest_path)
    checkpoints = list(manifest.get("checkpoints", []))
    if len(checkpoints) != 5:
        raise RuntimeError("robust Stage-1 head manifest does not contain five checkpoints")
    artifacts = {
        "robust_head_manifest": manifest_path,
        "robust_normalization": resolve_path(config["robust_p90"]["normalization"]),
    }
    for index, row in enumerate(checkpoints):
        path = resolve_path(row["path"])
        if file_sha256(path) != str(row["sha256"]):
            raise RuntimeError(f"robust Stage-1 checkpoint {index} hash differs")
        artifacts[f"robust_checkpoint_{index}"] = path
    if file_sha256(artifacts["robust_normalization"]) != str(manifest["normalization"]["sha256"]):
        raise RuntimeError("robust Stage-1 normalization hash differs")
    return artifacts


def _phase69_dense_index(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = []
    for path in config["external"]["dense_results"].values():
        rows.extend(read_jsonl(path))
    result = {str(row["uid"]): row for row in rows}
    if len(result) != int(config["external"]["expected_total"]) or len(result) != len(rows):
        raise RuntimeError("Phase-69 dense result census differs")
    return result


def _phase69_paired_index(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(config["external"]["paired_results"])
    result = {str(row["uid"]): row for row in rows}
    if len(result) != int(config["external"]["expected_total"]) or len(result) != len(rows):
        raise RuntimeError("Phase-69 paired result census differs")
    return result


def _collect_stepb_epochs(config: Mapping[str, Any]) -> dict[str, Any]:
    root = resolve_path(config["parents"]["stepB_root"]) / "cache/models"
    records = []
    for path in root.glob("*/*/*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        task = payload["task"]
        records.append(
            {
                "domain": str(task["domain"]),
                "model": str(task["model"]),
                "target": str(task["target"]),
                "seed": int(task["seed"]),
                "epoch": int(payload["best_epoch"]),
            }
        )

    def median_epoch(domain: str, model: str, target: str, seed: int) -> int:
        values = sorted(
            row["epoch"] for row in records
            if (row["domain"], row["model"], row["target"], row["seed"])
            == (domain, model, target, int(seed))
        )
        if len(values) != 5:
            raise RuntimeError(f"missing five Step-B fold epochs: {domain}/{model}/{target}/{seed}")
        return int(statistics.median_low(values))

    observed: dict[str, Any] = {"stage1": {}, "stage2": {"read": {}, "write": {}}}
    stage1 = config["refit"]["stage1"]
    observed["stage1"]["m0_x"] = median_epoch("stage1", "m0_nuisance", "dense_wrong", stage1["m0_x"]["seed"])
    observed["stage1"]["m1"] = median_epoch("stage1", "m1_linear", "dense_wrong", stage1["m1"]["seed"])
    observed["stage1"]["m3"] = {
        str(row["seed"]): median_epoch("stage1", "m3_current_head", "dense_wrong", row["seed"])
        for row in stage1["m3"]
    }
    for target in ("read", "write"):
        spec = config["refit"]["stage2"][target]
        observed["stage2"][target]["m0_x"] = median_epoch("stage2_dense", "m0_nuisance", target, spec["m0_x"]["seed"])
        observed["stage2"][target]["m1"] = median_epoch("stage2_dense", "m1_text_visual", target, spec["m1"]["seed"])
        observed["stage2"][target]["m3"] = {
            str(row["seed"]): median_epoch("stage2_dense", "m3_z_RW", target, row["seed"])
            for row in spec["m3"]
        }
    expected = {
        "stage1": {
            "m0_x": int(stage1["m0_x"]["epochs"]),
            "m1": int(stage1["m1"]["epochs"]),
            "m3": {str(row["seed"]): int(row["epochs"]) for row in stage1["m3"]},
        },
        "stage2": {
            target: {
                "m0_x": int(config["refit"]["stage2"][target]["m0_x"]["epochs"]),
                "m1": int(config["refit"]["stage2"][target]["m1"]["epochs"]),
                "m3": {
                    str(row["seed"]): int(row["epochs"])
                    for row in config["refit"]["stage2"][target]["m3"]
                },
            }
            for target in ("read", "write")
        },
    }
    if observed != expected:
        raise RuntimeError(f"frozen full-refit epochs differ from Step-B medians: {observed} != {expected}")
    return observed


def _refit_tasks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = []

    def add(domain: str, target: str, label: str, model: str, input_name: str, seed: int, epochs: int, cost: int) -> None:
        tasks.append(
            {
                "task_id": f"{domain}__{target}__{label}__s{int(seed)}",
                "domain": domain,
                "target": target,
                "label": label,
                "model": model,
                "input": input_name,
                "seed": int(seed),
                "epochs": int(epochs),
                "estimated_cost": int(cost) * int(epochs),
            }
        )

    stage1 = config["refit"]["stage1"]
    add("stage1", "dense_wrong", "m0_x", "m0_nuisance", "nuisance", **stage1["m0_x"], cost=1)
    add("stage1", "dense_wrong", "m1", "m1_linear", "state", **stage1["m1"], cost=3)
    for row in stage1["m3"]:
        add("stage1", "dense_wrong", "m3", "m3_current_head", "state_layer", **row, cost=6)
    for target in ("read", "write"):
        spec = config["refit"]["stage2"][target]
        add("stage2_dense", target, "m0_x", "m0_nuisance", "nuisance", **spec["m0_x"], cost=1)
        add("stage2_dense", target, "m1", "m1_text_visual", "text_visual", **spec["m1"], cost=3)
        for row in spec["m3"]:
            add("stage2_dense", target, "m3", "m3_z_RW", "z_RW", **row, cost=24)
    loads = [0] * int(config["world_size"])
    for task in sorted(tasks, key=lambda row: (-int(row["estimated_cost"]), str(row["task_id"]))):
        rank = min(range(len(loads)), key=lambda value: (loads[value], value))
        task["worker_rank"] = rank
        loads[rank] += int(task["estimated_cost"])
    return sorted(tasks, key=lambda row: str(row["task_id"]))


def prepare(config_path: Path) -> None:
    config = read_json(config_path)
    output_root = resolve_path(config["output_root"])
    external_root = resolve_path(config["external_output_root"])
    for relative in (
        "refit", "external_manifests", "predictions_frozen_before_labels", "stage1",
        "stage2_measurement", "stage2_predictability", "question_semantics", "statistics",
        "figures", "summaries", "validation", "work/refit", "work/external_states",
        "work/embeddings", "work/predictions", "work/measurement",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    for relative in ("refit", "external_states", "embeddings", "predictions", "measurement"):
        (external_root / relative).mkdir(parents=True, exist_ok=True)
    cache_link = output_root / "cache"
    if cache_link.exists() or cache_link.is_symlink():
        if not cache_link.is_symlink() or cache_link.resolve() != external_root:
            raise RuntimeError("Step-D cache path already exists with a different target")
    else:
        cache_link.symlink_to(external_root, target_is_directory=True)

    parent_checks = {}
    for name in ("stepA", "stepB", "stepC", "phase69"):
        root = resolve_path(config["parents"][f"{name}_root"])
        manifest = resolve_path(config["parents"][f"{name}_artifact_manifest"])
        parent_checks[name] = _verify_artifact_manifest(root, manifest)
    external_rows = read_jsonl(config["external"]["prepared_manifest"])
    for row in external_rows:
        question, source_field = canonical_question_text(row)
        row["question"] = question
        row["question_source_field"] = source_field
    population = validate_external_population(external_rows)
    if population["counts"] != dict(config["external"]["expected_counts"]):
        raise RuntimeError("external count config differs from prepared manifest")
    dense_index = _phase69_dense_index(config)
    paired_index = _phase69_paired_index(config)
    if set(dense_index) != {str(row["uid"]) for row in external_rows} or set(paired_index) != set(dense_index):
        raise RuntimeError("Phase-69 UID sets differ from Step-D population")
    for row in external_rows:
        uid = str(row["uid"])
        if (
            dense_index[uid]["generated_token_ids"] != paired_index[uid]["dense_generated_token_ids"]
            or bool(dense_index[uid]["correct"]) != bool(paired_index[uid]["dense_correct"])
            or float(dense_index[uid]["score"]) != float(paired_index[uid]["dense_score"])
        ):
            raise RuntimeError(f"Phase-69 dense/paired parity differs: {uid}")
    epoch_proof = _collect_stepb_epochs(config)
    tasks = _refit_tasks(config)
    atomic_jsonl(output_root / "work/refit_tasks.jsonl", tasks)
    schedule = []
    for index, row in enumerate(external_rows):
        schedule.append({"uid": str(row["uid"]), "benchmark": str(row["benchmark"]), "worker_rank": index % int(config["world_size"])})
    atomic_jsonl(output_root / "work/external_state_schedule.jsonl", schedule)
    atomic_jsonl(output_root / "external_manifests/benchmark_manifest.jsonl", external_rows)

    source_paths = {
        "external_manifest": config["external"]["prepared_manifest"],
        "external_paired": config["external"]["paired_results"],
        "internal_samples": config["internal"]["sample_manifest"],
        "internal_stage1_states": config["internal"]["stage1_state_manifest"],
        "internal_stage1_labels": config["internal"]["stage1_labels"],
        "internal_stage2_states": config["internal"]["stage2_state_manifest"],
        "internal_stage2_utilities": config["internal"]["stage2_utility_labels"],
        "internal_embeddings": config["internal"]["question_embeddings"],
        "internal_embedding_index": config["internal"]["uid_embedding_index"],
        "encoder_contract": config["internal"]["encoder_contract"],
    }
    source_paths.update(_robust_head_artifacts(config))
    for name, path in config["external"]["dense_results"].items():
        source_paths[f"dense_{name}"] = path
    contract = {
        "schema_version": "predictability_stepD_external_transfer_contract_v1",
        "created_at": utc_now(),
        "static_config": config,
        "config_sha256": file_sha256(config_path),
        "bound_code_sha256": {path: file_sha256(resolve_path(path)) for path in BOUND_CODE},
        "source_sha256": {name: file_sha256(resolve_path(path)) for name, path in source_paths.items()},
        "prepared_sha256": {
            "benchmark_manifest.jsonl": file_sha256(output_root / "external_manifests/benchmark_manifest.jsonl"),
            "refit_tasks.jsonl": file_sha256(output_root / "work/refit_tasks.jsonl"),
            "external_state_schedule.jsonl": file_sha256(output_root / "work/external_state_schedule.jsonl"),
        },
        "parents": parent_checks,
        "parent_contract_sha256": {
            name: read_json(config["parents"][f"{name}_contract"])["contract_sha256"]
            for name in ("stepA", "stepB", "stepC", "phase69")
        },
        "model_snapshot_sha256": model_file_hashes(resolve_path(config["model"]["snapshot_path"])),
        "refit_epoch_proof": epoch_proof,
        "population": population,
        "git": git_state(),
        "runtime": runtime_state(),
        "review_reconciliation": {
            "verdict": "stable",
            "reuse_boundary": "Phase69 identity/dense/trigger parity oracle only; recompute all Step-D features, states, and four-branch q labels",
            "minimum_gate": "native_hooked_dense_feature_trigger_FULL_q parity before full execution",
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_contract.json", contract)
    protocol = f"""# Predictability Step-D frozen protocol

- Contract: `{contract['contract_sha256']}`
- Population: 19,960 exact established UIDs (ChartQA 2,500; TextVQA 5,000; MMMU-Pro 3,460; POPE 9,000).
- Full refits: all internal states, fixed epochs selected as the median-low Step-B fold best epoch before external scoring.
- Stage-1 target: eventual external Dense failure under the frozen task scorer.
- Stage-2 domain: first strict robust score `> 0.9061332901863008`, then all Dense pre-action states through layer 27.
- Primary utilities: READ = q_FULL - q_WRITE_ONLY; WRITE = q_FULL - q_READ_ONLY.
- External-label firewall: predictions and their hashes are frozen before four-branch q/correctness labels execute.
- Reuse: Phase-69 rows are parity oracles only; Step-D features, states, and counterfactual labels are newly measured.
- Stop: final external-transfer evidence and one method implication; no routing redesign or deployment training.
"""
    _atomic(output_root / "protocol.md", protocol.encode())
    print(json.dumps({"prepared": True, "contract_sha256": contract["contract_sha256"], "population": population, "refit_tasks": len(tasks)}, sort_keys=True))


def verify_contract(config_path: Path, *, verify_model: bool = False) -> tuple[dict[str, Any], Path, Path]:
    config = read_json(config_path)
    output_root = resolve_path(config["output_root"])
    external_root = resolve_path(config["external_output_root"])
    contract = read_json(output_root / "frozen_contract.json")
    if canonical_hash(contract) != contract.get("contract_sha256"):
        raise RuntimeError("Step-D contract self-hash differs")
    if contract.get("static_config") != config or contract.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError("Step-D config differs from frozen contract")
    if git_state() != contract["git"]:
        raise RuntimeError("Git/worktree state differs from frozen Step-D contract")
    if runtime_state() != contract["runtime"]:
        raise RuntimeError("runtime differs from frozen Step-D contract")
    for path, expected in contract["bound_code_sha256"].items():
        if file_sha256(resolve_path(path)) != expected:
            raise RuntimeError(f"Step-D bound code differs: {path}")
    source_lookup = {
        "external_manifest": config["external"]["prepared_manifest"],
        "external_paired": config["external"]["paired_results"],
        "internal_samples": config["internal"]["sample_manifest"],
        "internal_stage1_states": config["internal"]["stage1_state_manifest"],
        "internal_stage1_labels": config["internal"]["stage1_labels"],
        "internal_stage2_states": config["internal"]["stage2_state_manifest"],
        "internal_stage2_utilities": config["internal"]["stage2_utility_labels"],
        "internal_embeddings": config["internal"]["question_embeddings"],
        "internal_embedding_index": config["internal"]["uid_embedding_index"],
        "encoder_contract": config["internal"]["encoder_contract"],
        **{f"dense_{name}": path for name, path in config["external"]["dense_results"].items()},
    }
    source_lookup.update(_robust_head_artifacts(config))
    for name, expected in contract["source_sha256"].items():
        if file_sha256(resolve_path(source_lookup[name])) != expected:
            raise RuntimeError(f"Step-D source differs: {name}")
    for relative, expected in contract["prepared_sha256"].items():
        path = output_root / ("external_manifests" if relative == "benchmark_manifest.jsonl" else "work") / relative
        if file_sha256(path) != expected:
            raise RuntimeError(f"Step-D prepared file differs: {relative}")
    if not (output_root / "cache").is_symlink() or (output_root / "cache").resolve() != external_root:
        raise RuntimeError("Step-D external cache binding differs")
    if verify_model and model_file_hashes(resolve_path(config["model"]["snapshot_path"])) != contract["model_snapshot_sha256"]:
        raise RuntimeError("Step-D model snapshot differs")
    return contract, output_root, external_root


def _internal_data(config: Mapping[str, Any], domain: str) -> dict[str, Any]:
    stepb_root = resolve_path(config["parents"]["stepB_root"])
    parent = read_json(config["parents"]["stepB_contract"])
    data = stepb._domain_data(parent, stepb_root, domain)
    sample_index = {str(row["uid"]): row for row in read_jsonl(config["internal"]["sample_manifest"])}
    for row in data["rows"]:
        source = sample_index[str(row["uid"])]
        row["question"] = str(source["question"])
    return data


def _full_nuisance(data: Mapping[str, Any], domain: str) -> stepb.TensorStore:
    values = (
        stage1_transfer_nuisance(data["rows"])
        if domain == "stage1"
        else stage2_transfer_nuisance(data["rows"])
    )
    return stepb.TensorStore(torch.from_numpy(values))


def _train_full_task(
    contract: Mapping[str, Any],
    task: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    config = contract["static_config"]
    parent_config = read_json(config["parents"]["stepB_contract"])["static_config"]
    stepb.configure_determinism(int(task["seed"]))
    indices = np.arange(len(data["rows"]), dtype=np.int64)
    classification = str(task["domain"]) == "stage1"
    nuisance_store = _full_nuisance(data, str(task["domain"])) if str(task["label"]) == "m0_x" else None
    if nuisance_store is not None:
        input_width = nuisance_store.input_width(str(task["input"]))
        mean, std = stepb._stream_standardizer(nuisance_store, str(task["input"]), indices, device=device)
    elif str(task["model"]).startswith("m3_z_"):
        input_width = int(parent_config["inputs"]["stage2_summary_width_each"])
        mean = std = None
    else:
        input_width = data["fixed"].input_width(str(task["input"]))
        mean, std = stepb._stream_standardizer(data["fixed"], str(task["input"]), indices, device=device)
    if mean is not None:
        mean, std = mean.to(device), std.to(device)
    target = np.asarray(data["targets"][str(task["target"])], dtype=np.float64)
    scale = None if classification else robust_target_scale(target, floor=float(parent_config["targets"]["target_scale_floor"]))
    training_target = target if classification else scale.transform(target)
    model = stepb._model_for_task(task, input_width, parent_config).to(device=device, dtype=torch.float32)
    spec = stepb._training_spec(task, parent_config)
    batch_size = int(spec.get("batch_size", spec.get("state_microbatch")))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"]))
    weights = stepb._subset_uid_weights(data["rows"], indices)
    generator = torch.Generator(device="cpu").manual_seed(int(task["seed"]))
    history = []
    started = time.monotonic()
    for epoch in range(int(task["epochs"])):
        model.train()
        order = torch.randperm(len(indices), generator=generator).numpy()
        local_batches = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
        batches = [indices[local] for local in local_batches]
        if str(task["model"]).startswith("m3_z_"):
            iterator = (
                (batch, packed)
                for batch, packed in stepb.iter_prefetched_packed_batches(data["packed"], batches, pin_memory=True)
            )
        else:
            iterator = ((batch, None) for batch in batches)
        total_loss = total_weight = 0.0
        for number, (batch_indices, packed) in enumerate(iterator):
            local = local_batches[number]
            batch_weights = torch.tensor(weights[local], dtype=torch.float32, device=device)
            batch_target = torch.tensor(training_target[batch_indices], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = stepb._forward_task(
                model, task, data, batch_indices, mean=mean, std=std,
                nuisance_store=nuisance_store, device=device, packed_batch=packed,
            )
            losses = (
                torch.nn.functional.binary_cross_entropy_with_logits(prediction, batch_target, reduction="none")
                if classification
                else torch.nn.functional.huber_loss(prediction, batch_target, delta=1.0, reduction="none")
            )
            loss = (losses * batch_weights).sum() / batch_weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(parent_config["training"]["gradient_clip_norm"]))
            optimizer.step()
            total_loss += float((losses.detach() * batch_weights).sum().cpu())
            total_weight += float(batch_weights.sum().cpu())
        history.append({"epoch": epoch + 1, "fit_loss": total_loss / total_weight, "elapsed_seconds": time.monotonic() - started})
    raw = stepb._predict_indices(
        model, task, data, indices, mean=mean, std=std,
        nuisance_store=nuisance_store, device=device, batch_size=batch_size,
    )
    prediction = 1.0 / (1.0 + np.exp(-np.clip(raw, -50, 50))) if classification else scale.inverse(raw)
    checkpoint = {
        "schema_version": "predictability_stepD_full_refit_v1",
        "contract_sha256": contract["contract_sha256"],
        "task": dict(task),
        "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "normalization_mean": None if mean is None else mean.detach().cpu(),
        "normalization_std": None if std is None else std.detach().cpu(),
        "target_center": None if scale is None else float(scale.center),
        "target_scale": None if scale is None else float(scale.scale),
        "internal_prediction": torch.from_numpy(prediction.astype(np.float32)),
        "epochs": int(task["epochs"]),
        "history": history,
        "fit_states": len(indices),
    }
    return checkpoint


def refit_worker(config_path: Path, rank: int, *, resume: bool) -> None:
    contract, output_root, external_root = verify_contract(config_path)
    config = contract["static_config"]
    parent_config = read_json(config["parents"]["stepB_contract"])["static_config"]
    # PyTorch permits setting the inter-op thread pool only before parallel
    # work starts.  Configure it once per worker process, not once per task.
    stepb.configure_worker_runtime(parent_config)
    rank = int(rank)
    if not 0 <= rank < int(config["world_size"]):
        raise ValueError("invalid Step-D refit rank")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    tasks = [row for row in read_jsonl(output_root / "work/refit_tasks.jsonl") if int(row["worker_rank"]) == rank]
    rank_root = output_root / f"work/refit/rank{rank:02d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    data_cache = {}
    completed = 0
    started = time.monotonic()
    for task in tasks:
        completion_path = rank_root / f"{task['task_id']}.json"
        checkpoint_path = external_root / "refit" / f"{task['task_id']}.pt"
        if resume and completion_path.is_file() and checkpoint_path.is_file():
            old = read_json(completion_path)
            if old.get("contract_sha256") == contract["contract_sha256"] and old.get("checkpoint_sha256") == file_sha256(checkpoint_path):
                completed += 1
                continue
        domain = str(task["domain"])
        if domain not in data_cache:
            data_cache[domain] = _internal_data(config, domain)
        payload = _train_full_task(contract, task, data_cache[domain], device=device)
        atomic_torch(checkpoint_path, payload)
        result = {
            "schema_version": "predictability_stepD_refit_completion_v1",
            "contract_sha256": contract["contract_sha256"],
            "task_id": task["task_id"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "fit_states": int(payload["fit_states"]),
            "epochs": int(payload["epochs"]),
            "final_fit_loss": float(payload["history"][-1]["fit_loss"]),
        }
        atomic_json(completion_path, result)
        completed += 1
        torch.cuda.empty_cache()
        print(json.dumps({"rank": rank, "completed": completed, "assigned": len(tasks), "task_id": task["task_id"], "elapsed_seconds": time.monotonic() - started}, sort_keys=True), flush=True)
    atomic_json(rank_root / "complete.json", {"schema_version": "predictability_stepD_refit_rank_complete_v1", "contract_sha256": contract["contract_sha256"], "rank": rank, "expected_tasks": len(tasks), "completed_tasks": completed, "completed_at": utc_now(), "elapsed_seconds": time.monotonic() - started})


def _refit_completions(contract: Mapping[str, Any], output_root: Path) -> list[dict[str, Any]]:
    config = contract["static_config"]
    tasks = read_jsonl(output_root / "work/refit_tasks.jsonl")
    completions = []
    for rank in range(int(config["world_size"])):
        rank_root = output_root / f"work/refit/rank{rank:02d}"
        marker = read_json(rank_root / "complete.json")
        expected = sum(int(row["worker_rank"]) == rank for row in tasks)
        if marker.get("contract_sha256") != contract["contract_sha256"] or int(marker.get("completed_tasks", -1)) != expected:
            raise RuntimeError(f"incomplete Step-D refit rank {rank}")
    for task in tasks:
        row = read_json(output_root / f"work/refit/rank{int(task['worker_rank']):02d}/{task['task_id']}.json")
        checkpoint = resolve_path(row["checkpoint"])
        if row.get("contract_sha256") != contract["contract_sha256"] or file_sha256(checkpoint) != row.get("checkpoint_sha256"):
            raise RuntimeError(f"invalid Step-D refit completion: {task['task_id']}")
        completions.append({**task, **row})
    return completions


def finalize_refits(config_path: Path) -> None:
    contract, output_root, _ = verify_contract(config_path)
    completions = _refit_completions(contract, output_root)
    config = contract["static_config"]
    stage1_data = _internal_data(config, "stage1")
    labels = np.asarray(stage1_data["targets"]["dense_wrong"], dtype=np.int64)
    uids = np.asarray([str(row["uid"]) for row in stage1_data["rows"]])
    layers = np.asarray([int(row["layer"]) for row in stage1_data["rows"]])
    thresholds = []
    m3_predictions = []
    refit_rows = []
    for completion in completions:
        payload = torch.load(completion["checkpoint"], map_location="cpu", weights_only=False)
        prediction = payload["internal_prediction"].numpy().astype(np.float64)
        metric = (
            binary_classification_metrics(truth=labels, prediction=prediction)
            if completion["domain"] == "stage1"
            else regression_metrics(
                truth=_internal_data(config, "stage2_dense")["targets"][completion["target"]],
                prediction=prediction,
            )
        )
        refit_rows.append({"task_id": completion["task_id"], "domain": completion["domain"], "target": completion["target"], "model": completion["label"], "seed": completion["seed"], "epochs": completion["epochs"], "fit_states": completion["fit_states"], **metric})
        if completion["domain"] == "stage1" and completion["label"] == "m3":
            m3_predictions.append((int(completion["seed"]), prediction))
    if len(m3_predictions) != 3:
        raise RuntimeError("Step-D requires three Stage-1 M3 refits")

    def calibrate(name: str, prediction: np.ndarray) -> None:
        by_uid = defaultdict(list)
        truth = {}
        for uid, layer, label, score in zip(uids, layers, labels, prediction):
            by_uid[str(uid)].append((int(layer), float(score)))
            truth[str(uid)] = int(label)
        uid_order = sorted(by_uid)
        maxima = []
        uid_truth = []
        for uid in uid_order:
            values = sorted(by_uid[uid])
            if [layer for layer, _ in values] != list(range(28)):
                raise RuntimeError(f"internal Stage-1 layer census differs: {uid}")
            maxima.append(max(score for _, score in values))
            uid_truth.append(truth[uid])
        for target in config["evaluation"]["stage1_preservation_targets"]:
            thresholds.append({"model": "m3", "seed_or_ensemble": name, "nominal_c_preservation": float(target), "threshold": select_preservation_threshold(maxima, uid_truth, target_correct_preservation=float(target)), "comparison": "strict_greater_than", "calibration_population": "all_internal_dense_C_full_refit_scores"})

    for seed, prediction in m3_predictions:
        calibrate(str(seed), prediction)
    calibrate("ensemble", np.mean([value for _, value in m3_predictions], axis=0))
    atomic_csv(output_root / "refit/stage1_internal_calibration_thresholds.csv", thresholds)
    atomic_csv(output_root / "refit/refit_metrics.csv", refit_rows)
    manifest = {
        "schema_version": "predictability_stepD_refit_manifest_v1",
        "contract_sha256": contract["contract_sha256"],
        "tasks": [
            {key: row[key] for key in ("task_id", "domain", "target", "label", "seed", "epochs", "checkpoint", "checkpoint_sha256", "fit_states")}
            for row in completions
        ],
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    atomic_json(output_root / "refit/stage1_full_refit_manifest.json", {**manifest, "tasks": [row for row in manifest["tasks"] if row["domain"] == "stage1"]})
    atomic_json(output_root / "refit/stage2_seed_checkpoints.json", {**manifest, "tasks": [row for row in manifest["tasks"] if row["domain"] == "stage2_dense"]})
    atomic_json(output_root / "refit/stage1_seed_checkpoints.json", {**manifest, "tasks": [row for row in manifest["tasks"] if row["domain"] == "stage1" and row["label"] == "m3"]})
    for target in ("read", "write"):
        atomic_json(output_root / f"refit/stage2_{target}_full_refit_manifest.json", {**manifest, "target": target, "tasks": [row for row in manifest["tasks"] if row["domain"] == "stage2_dense" and row["target"] == target]})
    lines = ["# Step-D full-internal refit summary", "", f"- Contract: `{contract['contract_sha256']}`", f"- Completed refits: {len(completions)}/{len(read_jsonl(output_root / 'work/refit_tasks.jsonl'))}", "- Training population: all internal states; no external labels or early stopping.", "- Epochs: median-low selected epoch from the five corresponding Step-B folds, frozen before external scoring.", f"- Stage-1 states: {len(stage1_data['rows']):,}; Stage-2 Dense states: {len(_internal_data(config, 'stage2_dense')['rows']):,}.", "- Internal-fit thresholds: numerical strict-greater values are in `stage1_internal_calibration_thresholds.csv`; Step-B OOF remains the unbiased ID reference."]
    _atomic(output_root / "refit/refit_training_summary.md", ("\n".join(lines) + "\n").encode())
    print(json.dumps({"refits_complete": True, "tasks": len(completions), "thresholds": len(thresholds)}, sort_keys=True))


class DenseRuntime:
    """External dense runtime with the frozen robust gate but no Stage-2 policy."""

    def __init__(self, config: Mapping[str, Any], device_index: int):
        self.config = config
        self.device = torch.device(f"cuda:{int(device_index)}")
        torch.cuda.set_device(self.device)
        self.processor, self.base, self.wrapped = phase69._load_model(config, self.device)
        head_manifest = read_json(config["robust_p90"]["head_manifest"])
        self.stage1 = []
        for checkpoint_row in head_manifest["checkpoints"]:
            model = phase69.SharedFailurePredictor(
                variant="state_layer_random4",
                input_size=10752,
                projection_size=256,
                layer_embedding_size=32,
                hidden_size=256,
            ).to(self.device, dtype=torch.float32).eval()
            checkpoint = torch.load(resolve_path(checkpoint_row["path"]), map_location="cpu", weights_only=True)
            if checkpoint.get("schema_version") != "stage1_all_source_checkpoint_v1" or checkpoint.get("contract_sha256") != checkpoint_row["contract_sha256"]:
                raise RuntimeError("loaded robust Stage-1 checkpoint provenance differs")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.stage1.append(model)
        normalization = torch.load(resolve_path(config["robust_p90"]["normalization"]), map_location="cpu", weights_only=True)
        if normalization.get("schema_version") != "shared_stage1_global_normalization_v1":
            raise RuntimeError("loaded robust Stage-1 normalization provenance differs")
        self.mean = normalization["mean"].float().to(self.device)
        self.std = normalization["std"].float().to(self.device)


@torch.inference_mode()
def _robust_stage1_scores(
    runtime: DenseRuntime, features: torch.Tensor
) -> tuple[list[float], int | None]:
    normalized = (features.to(runtime.device, dtype=torch.float32) - runtime.mean) / runtime.std
    layers = torch.arange(28, device=runtime.device, dtype=torch.long)
    probabilities = torch.stack(
        [torch.sigmoid(model(normalized, layers)) for model in runtime.stage1], dim=0
    ).mean(dim=0)
    if tuple(probabilities.shape) != (28,) or not bool(torch.isfinite(probabilities).all()):
        raise RuntimeError("robust Stage-1 ensemble produced an invalid trajectory")
    scores = [float(value) for value in probabilities.detach().cpu().tolist()]
    return scores, strict_first_trigger(scores, float(runtime.config["robust_p90"]["threshold"]))


def _decode_external(runtime: DenseRuntime, generated: torch.Tensor) -> tuple[list[int], str]:
    return phase69._decode(runtime, generated)


def _measure_external_output(
    runtime: DenseRuntime,
    output: Any,
    input_ids: torch.Tensor,
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    if output.cache is None:
        raise RuntimeError("Step-D measurement output lacks a prompt cache")
    answer_specs = external_answer_specs(sample)
    answer_rows = []
    mean_logprobs = []
    for answer in answer_specs:
        token_ids = runtime.processor.tokenizer(
            str(answer["text"]), add_special_tokens=False, return_tensors="pt"
        ).input_ids[0].to(output.prompt_logits.device)
        scored = score_token_ids_from_cached_prompt(
            runtime.wrapped, output.prompt_logits, output.inputs, output.cache, token_ids
        )
        mean_logprobs.append(float(scored.mean_logprob))
        answer_rows.append(
            {
                "text": str(answer["text"]),
                "weight": float(answer["weight"]),
                "token_ids": scored.token_ids,
                "token_logprobs": scored.token_logprobs,
                "sequence_logprob": float(scored.sequence_logprob),
                "mean_logprob": float(scored.mean_logprob),
            }
        )
    q = aggregate_reference_mean_logprobs(answer_specs, mean_logprobs)
    generated = greedy_generate_from_cached_prompt(
        runtime.wrapped,
        output.prompt_logits,
        output.inputs,
        output.cache,
        input_ids,
        max_new_tokens=int(sample["max_new_tokens"]),
        eos_token_ids=list(runtime.config["generation"]["eos_token_ids"]),
        repetition_penalty=float(runtime.config["generation"]["repetition_penalty"]),
    ).generated_ids
    generated_ids, generated_answer = _decode_external(runtime, generated)
    score = phase69._score(runtime.config, sample, generated_answer)
    return {
        "mean_logprob": float(q),
        "accepted_answer_scores": answer_rows,
        "generated_token_ids": generated_ids,
        "generated_answer": generated_answer,
        "metric": score["metric"],
        "score": float(score["score"]),
        "threshold": float(score["threshold"]),
        "correct": bool(score["correct"]),
    }


def _full_baseline(
    runtime: DenseRuntime, sample: Mapping[str, Any]
) -> tuple[dict[str, Any], Any, Any, torch.Tensor, int]:
    consumed = phase69._verify_images(sample)
    inputs = phase69._build_inputs(runtime, sample)
    prepared = build_binary_inputs(runtime.wrapped, inputs)
    output = capture_four_action_route(
        runtime.wrapped,
        {},
        ["FULL"] * int(runtime.config["model"]["decoder_layers"]),
        prepared_inputs=prepared,
        use_cache=True,
        native_full_rows=True,
    )
    measured = _measure_external_output(runtime, output, inputs["input_ids"], sample)
    if list(consumed) != list(sample["image_content_sha256s"]):
        raise RuntimeError(f"Step-D binary baseline image hash differs: {sample['uid']}")
    instruction_count = int(sum(bool(value) for value in inputs["instruction_token_mask"][0].tolist()))
    return measured, output, prepared, inputs["input_ids"], instruction_count


def _branch_actions(
    runtime: DenseRuntime,
    sample: Mapping[str, Any],
    baseline: Any,
    input_ids: torch.Tensor,
    layer: int,
    dense: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    branches = {}
    full_parity = None
    for action in ACTIONS:
        suffix = [action] + ["FULL"] * (27 - int(layer))
        output = capture_four_action_suffix_from_full_baseline(
            runtime.wrapped, baseline, int(layer), suffix
        )
        stepa._branch_semantics(output, int(layer), action)
        measured = _measure_external_output(runtime, output, input_ids, sample)
        branches[action] = measured
        if action == "FULL":
            full_parity = {
                "token_exact": measured["generated_token_ids"] == dense["generated_token_ids"],
                "score_exact": measured["score"] == dense["score"],
                "correctness_exact": measured["correct"] == dense["correct"],
                "q_abs_difference": abs(float(measured["mean_logprob"]) - float(dense["mean_logprob"])),
            }
            full_parity["passed"] = (
                full_parity["token_exact"]
                and full_parity["score_exact"]
                and full_parity["correctness_exact"]
                and full_parity["q_abs_difference"] <= float(runtime.config["measurement"]["q_absolute_tolerance"])
            )
            if not full_parity["passed"]:
                raise RuntimeError(f"Step-D FULL branch parity failed: {sample['uid']} L{layer}")
        del output
    assert full_parity is not None
    return branches, full_parity


def _smoke_rows(manifest: Sequence[Mapping[str, Any]], paired: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for benchmark in EXTERNAL_BENCHMARK_COUNTS:
        rows = [row for row in manifest if str(row["benchmark"]) == benchmark]
        selected.append(min(rows, key=lambda row: sha256(str(row["uid"]).encode()).hexdigest()))
    for family in ("chartqa", "textvqa", "mmmu_pro"):
        rows = [
            row for row in manifest
            if str(row["benchmark_family"]) == family and bool(paired[str(row["uid"])]["triggered"])
        ]
        if not rows:
            raise RuntimeError(f"Step-D smoke lacks a triggered {family} row")
        candidate = min(rows, key=lambda row: sha256(("triggered|" + str(row["uid"])).encode()).hexdigest())
        if str(candidate["uid"]) not in {str(row["uid"]) for row in selected}:
            selected.append(candidate)
    return selected


def smoke(config_path: Path, device_index: int) -> None:
    contract, output_root, _ = verify_contract(config_path, verify_model=True)
    if not (output_root / "refit/stage1_full_refit_manifest.json").is_file():
        raise RuntimeError("Step-D smoke requires completed full-internal refits")
    config = contract["static_config"]
    phase69.configure_dense_determinism(int(config["seed"]), config["backend_settings"])
    runtime = DenseRuntime(config, int(device_index))
    manifest = read_jsonl(output_root / "external_manifests/benchmark_manifest.jsonl")
    prior_dense = _phase69_dense_index(config)
    prior_paired = _phase69_paired_index(config)
    selected = _smoke_rows(manifest, prior_paired)
    rows = []
    triggered_checked = set()
    for sample in selected:
        uid = str(sample["uid"])
        unhooked = phase69._native_dense_generation(runtime, sample, extract_features=False)
        hooked = phase69._native_dense_generation(runtime, sample, extract_features=True)
        features = hooked.pop("features")
        scores, trigger = _robust_stage1_scores(runtime, features)
        prior = prior_dense[uid]
        paired = prior_paired[uid]
        score_delta = max(abs(float(left) - float(right)) for left, right in zip(scores, paired["stage1_scores"]))
        item = {
            "uid": uid,
            "benchmark": sample["benchmark"],
            "native_hooked_token_exact": unhooked["generated_token_ids"] == hooked["generated_token_ids"],
            "native_hooked_score_exact": unhooked["score"] == hooked["score"],
            "reference_token_exact": hooked["generated_token_ids"] == prior["generated_token_ids"],
            "reference_score_exact": hooked["score"] == prior["score"],
            "reference_correctness_exact": hooked["correct"] == prior["correct"],
            "robust_score_max_abs_difference": score_delta,
            "trigger_exact": trigger == paired["trigger_layer"],
            "image_hash_exact": hooked["consumed_image_sha256s"] == sample["image_content_sha256s"],
            "feature_shape": list(features.shape),
            "stage2_checked": False,
        }
        family = str(sample["benchmark_family"])
        if trigger is not None and family not in triggered_checked:
            first, baseline1, prepared1, input_ids1, _instruction_count1 = _full_baseline(runtime, sample)
            second, baseline2, prepared2, input_ids2, _instruction_count2 = _full_baseline(runtime, sample)
            state1 = stepa._cached_state(
                *baseline1.pre_layer_states[int(trigger)],
                prepared1.text_valid_mask,
                prepared1.visual_valid_mask,
            )
            state2 = stepa._cached_state(
                *baseline2.pre_layer_states[int(trigger)],
                prepared2.text_valid_mask,
                prepared2.visual_valid_mask,
            )
            branches, parity = _branch_actions(runtime, sample, baseline1, input_ids1, int(trigger), first)
            item.update(
                {
                    "stage2_checked": True,
                    "binary_dense_token_exact": first["generated_token_ids"] == hooked["generated_token_ids"],
                    "binary_dense_score_exact": first["score"] == hooked["score"],
                    "state_hash_repeat_exact": state1["state_sha256"] == state2["state_sha256"],
                    "full_branch_parity": parity,
                    "four_actions_complete": set(branches) == set(ACTIONS),
                }
            )
            triggered_checked.add(family)
            del baseline1, baseline2, prepared1, prepared2, input_ids1, input_ids2, branches
        item["passed"] = (
            item["native_hooked_token_exact"]
            and item["native_hooked_score_exact"]
            and item["reference_token_exact"]
            and item["reference_score_exact"]
            and item["reference_correctness_exact"]
            and item["robust_score_max_abs_difference"] <= 1e-7
            and item["trigger_exact"]
            and item["image_hash_exact"]
            and item["feature_shape"] == [28, 10752]
            and (
                not item["stage2_checked"]
                or (
                    item["binary_dense_token_exact"]
                    and item["binary_dense_score_exact"]
                    and item["state_hash_repeat_exact"]
                    and item["full_branch_parity"]["passed"]
                    and item["four_actions_complete"]
                )
            )
        )
        rows.append(item)
        if not item["passed"]:
            raise RuntimeError(f"Step-D smoke failed: {uid} {item}")
        del features
        torch.cuda.empty_cache()
    if triggered_checked != {"chartqa", "textvqa", "mmmu_pro"}:
        raise RuntimeError(f"Step-D smoke Stage-2 family coverage differs: {triggered_checked}")
    atomic_jsonl(output_root / "validation/smoke_rows.jsonl", rows)
    report = {
        "schema_version": "predictability_stepD_smoke_v1",
        "contract_sha256": contract["contract_sha256"],
        "passed": all(bool(row["passed"]) for row in rows),
        "samples": len(rows),
        "task_variants": sorted({str(row["benchmark"]) for row in rows}),
        "stage2_families": sorted(triggered_checked),
        "native_hooked_token_parity": all(bool(row["native_hooked_token_exact"]) for row in rows),
        "reference_dense_parity": all(bool(row["reference_token_exact"]) and bool(row["reference_score_exact"]) for row in rows),
        "robust_trigger_parity": all(bool(row["trigger_exact"]) for row in rows),
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "validation/smoke_report.json", report)
    print(json.dumps(report, sort_keys=True))


def _require_smoke(contract: Mapping[str, Any], output_root: Path) -> None:
    report = read_json(output_root / "validation/smoke_report.json")
    if report.get("contract_sha256") != contract["contract_sha256"] or not bool(report.get("passed")):
        raise RuntimeError("Step-D requires a passing bound smoke")


def external_state_worker(config_path: Path, rank: int, *, resume: bool) -> None:
    contract, output_root, external_root = verify_contract(config_path, verify_model=True)
    _require_smoke(contract, output_root)
    config = contract["static_config"]
    rank = int(rank)
    phase69.configure_dense_determinism(int(config["seed"]) + rank, config["backend_settings"])
    runtime = DenseRuntime(config, rank)
    manifest = {str(row["uid"]): row for row in read_jsonl(output_root / "external_manifests/benchmark_manifest.jsonl")}
    prior_dense = _phase69_dense_index(config)
    prior_paired = _phase69_paired_index(config)
    schedule = [row for row in read_jsonl(output_root / "work/external_state_schedule.jsonl") if int(row["worker_rank"]) == rank]
    rank_root = output_root / f"work/external_states/rank{rank:02d}"
    state_root = external_root / "external_states" / f"rank{rank:02d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    started = time.monotonic()
    for item in schedule:
        uid = str(item["uid"])
        slug = sha256(uid.encode()).hexdigest()[:24]
        result_path = rank_root / f"{slug}.json"
        state_path = state_root / f"{slug}.pt"
        if resume and result_path.is_file() and state_path.is_file():
            old = read_json(result_path)
            if old.get("contract_sha256") == contract["contract_sha256"] and old.get("state_file_sha256") == file_sha256(state_path):
                completed += 1
                continue
        sample = manifest[uid]
        prior = prior_dense[uid]
        paired = prior_paired[uid]
        metadata_inputs = phase69._build_inputs(runtime, sample)
        user_text_token_count = int(
            sum(bool(value) for value in metadata_inputs["instruction_token_mask"][0].tolist())
        )
        del metadata_inputs
        dense = phase69._native_dense_generation(runtime, sample, extract_features=True)
        live_features = dense.pop("features")
        robust_scores, trigger = _robust_stage1_scores(runtime, live_features)
        features = live_features.detach().cpu().to(torch.bfloat16).contiguous()
        del live_features
        score_delta = max(abs(float(left) - float(right)) for left, right in zip(robust_scores, paired["stage1_scores"]))
        if (
            dense["generated_token_ids"] != prior["generated_token_ids"]
            or dense["generated_answer"] != prior["generated_answer"]
            or dense["score"] != prior["score"]
            or dense["correct"] != prior["correct"]
            or dense["consumed_image_sha256s"] != sample["image_content_sha256s"]
            or score_delta > 1e-7
            or trigger != paired["trigger_layer"]
        ):
            raise RuntimeError(f"Step-D external dense/trigger parity failed: {uid}")
        states = {}
        state_rows = []
        binary_dense = None
        if trigger is not None:
            binary_dense, baseline, prepared, input_ids, instruction_count = _full_baseline(runtime, sample)
            if instruction_count != user_text_token_count:
                raise RuntimeError(f"Step-D instruction token count is unstable: {uid}")
            if (
                binary_dense["generated_token_ids"] != dense["generated_token_ids"]
                or binary_dense["score"] != dense["score"]
                or binary_dense["correct"] != dense["correct"]
            ):
                raise RuntimeError(f"Step-D binary FULL baseline differs: {uid}")
            for layer in range(int(trigger), 28):
                state_id = sha256(f"stepD|{uid}|L{layer}|dense".encode()).hexdigest()
                cached = stepa._cached_state(
                    *baseline.pre_layer_states[layer],
                    prepared.text_valid_mask,
                    prepared.visual_valid_mask,
                )
                states[state_id] = {**cached, "state_id": state_id, "uid": uid, "layer": layer, "trigger_layer": int(trigger), "prefix_actions": ["FULL"] * layer}
                state_rows.append({"state_id": state_id, "uid": uid, "benchmark": sample["benchmark"], "benchmark_family": sample["benchmark_family"], "image_group_id": sample["image_group_id"], "layer": layer, "trigger_layer": int(trigger), "trigger_relative_depth": layer - int(trigger), "state_sha256": cached["state_sha256"], "tensor_sha256": cached["tensor_sha256"], "text_tokens": int(cached["text_states"].shape[1]), "visual_tokens": int(cached["visual_states"].shape[1]), "prompt_token_count": int(dense["prompt_tokens"]), "user_text_token_count": instruction_count, "question": str(sample["question"])})
            del baseline, prepared, input_ids
        payload = {
            "schema_version": "predictability_stepD_external_uid_states_v1",
            "contract_sha256": contract["contract_sha256"],
            "uid": uid,
            "stage1_features": features,
            "stage2_states": states,
        }
        atomic_torch(state_path, payload)
        result = {
            "schema_version": "predictability_stepD_external_state_completion_v1",
            "contract_sha256": contract["contract_sha256"],
            "uid": uid,
            "benchmark": sample["benchmark"],
            "benchmark_family": sample["benchmark_family"],
            "image_group_id": sample["image_group_id"],
            "dense_generated_token_ids": dense["generated_token_ids"],
            "dense_generated_answer": dense["generated_answer"],
            "dense_score": dense["score"],
            "dense_correct": dense["correct"],
            "prompt_token_count": dense["prompt_tokens"],
            "visual_token_count": dense["visual_tokens"],
            "user_text_token_count": user_text_token_count,
            "question": sample["question"],
            "robust_scores": robust_scores,
            "robust_score_max_abs_difference": score_delta,
            "triggered": trigger is not None,
            "trigger_layer": trigger,
            "stage2_state_rows": state_rows,
            "state_file": str(state_path),
            "state_file_sha256": file_sha256(state_path),
            "consumed_image_sha256s": dense["consumed_image_sha256s"],
        }
        atomic_json(result_path, result)
        completed += 1
        torch.cuda.empty_cache()
        if completed % 25 == 0 or completed == len(schedule):
            print(json.dumps({"rank": rank, "completed": completed, "assigned": len(schedule), "triggered": int(trigger is not None), "elapsed_seconds": time.monotonic() - started}, sort_keys=True), flush=True)
    atomic_json(rank_root / "complete.json", {"schema_version": "predictability_stepD_external_state_rank_complete_v1", "contract_sha256": contract["contract_sha256"], "rank": rank, "expected_uids": len(schedule), "completed_uids": completed, "completed_at": utc_now(), "elapsed_seconds": time.monotonic() - started})


def _external_state_results(contract: Mapping[str, Any], output_root: Path) -> list[dict[str, Any]]:
    config = contract["static_config"]
    schedule = read_jsonl(output_root / "work/external_state_schedule.jsonl")
    rows = []
    for rank in range(int(config["world_size"])):
        assigned = [row for row in schedule if int(row["worker_rank"]) == rank]
        marker = read_json(output_root / f"work/external_states/rank{rank:02d}/complete.json")
        if marker.get("contract_sha256") != contract["contract_sha256"] or int(marker.get("completed_uids", -1)) != len(assigned):
            raise RuntimeError(f"incomplete Step-D external-state rank {rank}")
        for item in assigned:
            slug = sha256(str(item["uid"]).encode()).hexdigest()[:24]
            row = read_json(output_root / f"work/external_states/rank{rank:02d}/{slug}.json")
            if row.get("uid") != item["uid"] or row.get("contract_sha256") != contract["contract_sha256"] or file_sha256(row["state_file"]) != row["state_file_sha256"]:
                raise RuntimeError(f"invalid Step-D external-state result: {item['uid']}")
            rows.append(row)
    if len(rows) != int(config["external"]["expected_total"]) or len({row["uid"] for row in rows}) != len(rows):
        raise RuntimeError("Step-D external-state global census differs")
    return rows


def finalize_external_states(config_path: Path) -> None:
    contract, output_root, _ = verify_contract(config_path)
    rows = _external_state_results(contract, output_root)
    config = contract["static_config"]
    dense_manifest = []
    trigger_manifest = []
    stage2_manifest = []
    for row in sorted(rows, key=lambda value: str(value["uid"])):
        for layer in range(28):
            dense_manifest.append({"state_id": sha256(f"stepD|{row['uid']}|L{layer}|stage1".encode()).hexdigest(), "uid": row["uid"], "benchmark": row["benchmark"], "benchmark_family": row["benchmark_family"], "image_group_id": row["image_group_id"], "layer": layer, "dense_correct": row["dense_correct"], "dense_wrong": not bool(row["dense_correct"]), "question": row["question"], "prompt_token_count": row["prompt_token_count"], "user_text_token_count": row["user_text_token_count"], "visual_token_count": row["visual_token_count"], "state_file": row["state_file"], "state_file_sha256": row["state_file_sha256"], "feature_row_index": layer})
        trigger_manifest.append({"uid": row["uid"], "benchmark": row["benchmark"], "benchmark_family": row["benchmark_family"], "image_group_id": row["image_group_id"], "dense_correct": row["dense_correct"], "dense_wrong": not bool(row["dense_correct"]), "scores": row["robust_scores"], "max_score": max(row["robust_scores"]), "threshold": config["robust_p90"]["threshold"], "comparison": "strict_greater_than", "triggered": row["triggered"], "first_trigger_layer": row["trigger_layer"], "post_trigger_state_count": len(row["stage2_state_rows"])})
        for state in row["stage2_state_rows"]:
            stage2_manifest.append({**state, "dense_correct": row["dense_correct"], "dense_wrong": not bool(row["dense_correct"]), "state_file": row["state_file"], "state_file_sha256": row["state_file_sha256"]})
    expected_dense = int(config["external"]["expected_total"]) * 28
    if len(dense_manifest) != expected_dense or len({row["state_id"] for row in dense_manifest}) != expected_dense:
        raise RuntimeError("Step-D dense-state census differs")
    trigger_count = sum(bool(row["triggered"]) for row in trigger_manifest)
    pope_triggers = sum(bool(row["triggered"]) and row["benchmark_family"] == "pope" for row in trigger_manifest)
    if trigger_count != int(config["robust_p90"]["reference_trigger_count"]) or pope_triggers != int(config["robust_p90"]["reference_pope_trigger_count"]):
        raise RuntimeError(f"Step-D robust trigger census differs: total={trigger_count}, pope={pope_triggers}")
    expected_states = sum(28 - int(row["first_trigger_layer"]) for row in trigger_manifest if row["triggered"])
    if len(stage2_manifest) != expected_states or len({row["state_id"] for row in stage2_manifest}) != len(stage2_manifest):
        raise RuntimeError("Step-D Stage-2 state census differs")
    atomic_jsonl(output_root / "external_manifests/dense_state_manifest.jsonl", dense_manifest)
    atomic_jsonl(output_root / "external_manifests/p90_trigger_manifest.jsonl", trigger_manifest)
    atomic_jsonl(output_root / "external_manifests/stage2_dense_state_manifest.jsonl", stage2_manifest)
    parity = {
        "schema_version": "predictability_stepD_external_parity_v1",
        "contract_sha256": contract["contract_sha256"],
        "external_uids": len(rows),
        "dense_states": len(dense_manifest),
        "triggered_uids": trigger_count,
        "stage2_states": len(stage2_manifest),
        "pope_triggers": pope_triggers,
        "dense_token_score_correctness_parity": all(float(row["robust_score_max_abs_difference"]) <= 1e-7 for row in rows),
        "image_hash_parity": all(bool(row["consumed_image_sha256s"]) for row in rows),
        "passed": True,
    }
    atomic_json(output_root / "external_manifests/parity_report.json", parity)
    _atomic(output_root / "external_manifests/parity_report.md", (f"# Step-D external parity report\n\n- Contract: `{contract['contract_sha256']}`\n- Exact external UIDs: {len(rows):,}\n- Dense layer states: {len(dense_manifest):,}\n- Robust P90 triggers: {trigger_count:,}; POPE: {pope_triggers}\n- Dense post-trigger states: {len(stage2_manifest):,}\n- Phase-69 dense token/score/correctness and trigger-trace parity: PASS\n- Image SHA checks: PASS\n").encode())
    print(json.dumps(parity, sort_keys=True))


def embedding_worker(config_path: Path, rank: int) -> None:
    contract, output_root, external_root = verify_contract(config_path)
    config = contract["static_config"]
    rank = int(rank)
    if not 0 <= rank < int(config["world_size"]):
        raise ValueError("invalid Step-D embedding rank")
    rows = read_jsonl(output_root / "external_manifests/benchmark_manifest.jsonl")
    encoder_config = {"encoder": dict(config["question_encoder"])}
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    tokenizer, model = __import__(
        "experiments.run_predictability_stepC_generalization", fromlist=["_load_encoder"]
    )._load_encoder(encoder_config, device)
    encode = __import__(
        "experiments.run_predictability_stepC_generalization", fromlist=["_encode_questions"]
    )._encode_questions
    indices = np.arange(rank, len(rows), int(config["world_size"]), dtype=np.int64)
    questions = [normalize_question(str(rows[int(index)]["question"])) for index in indices]
    embeddings = encode(questions, encoder_config, tokenizer, model, device)
    if rank == 0:
        smoke_indices = indices[: min(16, len(indices))]
        smoke_questions = [normalize_question(str(rows[int(index)]["question"])) for index in smoke_indices]
        first = encode(smoke_questions, encoder_config, tokenizer, model, device)
        second = encode(smoke_questions, encoder_config, tokenizer, model, device)
        if not np.array_equal(first, second) or not np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=2e-5):
            raise RuntimeError("Step-D question-embedding repeatability failed")
        atomic_json(output_root / "validation/embedding_smoke.json", {"schema_version": "predictability_stepD_embedding_smoke_v1", "contract_sha256": contract["contract_sha256"], "encoder_contract_sha256": read_json(config["question_encoder"]["contract"])["contract_sha256"], "exact_repeat": True, "unit_norm": True, "uids": [rows[int(index)]["uid"] for index in smoke_indices]})
    shard = external_root / "embeddings" / f"rank{rank:02d}.npz"
    with tempfile.NamedTemporaryFile(dir=shard.parent, delete=False, suffix=".npz") as handle:
        temporary = Path(handle.name)
    np.savez(temporary, indices=indices, embeddings=embeddings)
    os.replace(temporary, shard)
    atomic_json(output_root / f"work/embeddings/rank{rank:02d}.json", {"schema_version": "predictability_stepD_embedding_rank_v1", "contract_sha256": contract["contract_sha256"], "rank": rank, "rows": len(indices), "shard": str(shard), "shard_sha256": file_sha256(shard)})
    print(json.dumps({"rank": rank, "embedded": len(indices)}, sort_keys=True))


def finalize_embeddings(config_path: Path) -> None:
    contract, output_root, external_root = verify_contract(config_path)
    config = contract["static_config"]
    rows = read_jsonl(output_root / "external_manifests/benchmark_manifest.jsonl")
    arrays = []
    observed = np.zeros(len(rows), dtype=bool)
    width = None
    for rank in range(int(config["world_size"])):
        meta = read_json(output_root / f"work/embeddings/rank{rank:02d}.json")
        if meta.get("contract_sha256") != contract["contract_sha256"] or file_sha256(meta["shard"]) != meta["shard_sha256"]:
            raise RuntimeError(f"invalid Step-D embedding rank {rank}")
        shard = np.load(meta["shard"])
        indices = shard["indices"].astype(np.int64)
        values = shard["embeddings"].astype(np.float32)
        if width is None:
            width = int(values.shape[1])
            merged = np.empty((len(rows), width), dtype=np.float32)
        if values.shape != (len(indices), width) or observed[indices].any():
            raise RuntimeError(f"Step-D embedding shard differs: rank {rank}")
        merged[indices] = values
        observed[indices] = True
        arrays.append(meta)
    if not observed.all() or not np.allclose(np.linalg.norm(merged, axis=1), 1.0, atol=2e-5):
        raise RuntimeError("Step-D embedding global census/norm differs")
    embedding_path = external_root / "embeddings/external_question_embeddings.npy"
    with tempfile.NamedTemporaryFile(dir=embedding_path.parent, delete=False, suffix=".npy") as handle:
        temporary = Path(handle.name)
    np.save(temporary, merged)
    os.replace(temporary, embedding_path)
    link = output_root / "question_semantics/external_question_embeddings.npy"
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != embedding_path:
            raise RuntimeError("Step-D external embedding link differs")
    else:
        link.symlink_to(embedding_path)
    internal_index = read_jsonl(config["internal"]["uid_embedding_index"])
    internal_embeddings = np.load(resolve_path(config["internal"]["question_embeddings"])).astype(np.float32)
    if len(internal_index) != len(internal_embeddings):
        raise RuntimeError("internal Step-C embedding index differs")
    sample_index = {str(row["uid"]): row for row in read_jsonl(config["internal"]["sample_manifest"])}
    internal_labels = np.asarray([int(bool(sample_index[str(row["uid"])]["dense_wrong"])) for row in internal_index], dtype=np.float64)
    generalization = __import__("dense_failure_stage2.predictability_generalization", fromlist=["cosine_nearest", "exact_knn_prediction"])
    similarities, nearest = generalization.cosine_nearest(merged, internal_embeddings, block_size=1024)
    knn = generalization.exact_knn_prediction(merged, internal_embeddings, internal_labels, k=int(config["question_encoder"]["knn_k"]), block_size=256)
    semantic_rows = []
    embedding_index = []
    for index, row in enumerate(rows):
        semantic_rows.append({"uid": row["uid"], "benchmark": row["benchmark"], "benchmark_family": row["benchmark_family"], "image_group_id": row["image_group_id"], "nearest_internal_similarity": float(similarities[index]), "nearest_internal_uid": internal_index[int(nearest[index])]["uid"], "question_knn_failure_prediction": float(knn[index])})
        embedding_index.append({"uid": row["uid"], "benchmark": row["benchmark"], "benchmark_family": row["benchmark_family"], "row_index": index})
    atomic_jsonl(output_root / "question_semantics/nearest_internal_similarity.jsonl", semantic_rows)
    atomic_jsonl(output_root / "question_semantics/external_embedding_index.jsonl", embedding_index)
    atomic_json(output_root / "question_semantics/embedding_manifest.json", {"schema_version": "predictability_stepD_embedding_manifest_v1", "contract_sha256": contract["contract_sha256"], "encoder_contract_sha256": read_json(config["question_encoder"]["contract"])["contract_sha256"], "rows": len(rows), "width": width, "embedding_path": str(embedding_path), "embedding_sha256": file_sha256(embedding_path), "index_sha256": file_sha256(output_root / "question_semantics/external_embedding_index.jsonl"), "semantic_sha256": file_sha256(output_root / "question_semantics/nearest_internal_similarity.jsonl")})
    print(json.dumps({"embeddings_complete": True, "rows": len(rows), "width": width}, sort_keys=True))


def prepare_predictions(config_path: Path) -> None:
    contract, output_root, _ = verify_contract(config_path)
    config = contract["static_config"]
    if list((output_root / "work/measurement").glob("rank*/complete.json")) or (output_root / "stage2_measurement/four_branch_results.jsonl").exists():
        raise RuntimeError("external-label firewall violated: measurement exists before prediction freeze")
    _refit_completions(contract, output_root)
    states = read_jsonl(output_root / "external_manifests/stage2_dense_state_manifest.jsonl")
    external_index = read_jsonl(output_root / "question_semantics/external_embedding_index.jsonl")
    external_embeddings = np.load(output_root / "question_semantics/external_question_embeddings.npy").astype(np.float32)
    ext_row = {str(row["uid"]): int(row["row_index"]) for row in external_index}
    internal_embedding_index = read_jsonl(config["internal"]["uid_embedding_index"])
    internal_embeddings = np.load(resolve_path(config["internal"]["question_embeddings"])).astype(np.float32)
    int_row = {str(row["uid"]): int(row["row_index"]) for row in internal_embedding_index}
    internal_states = read_jsonl(config["internal"]["stage2_state_manifest"])
    utilities = {str(row["state_id"]): row for row in read_csv(config["internal"]["stage2_utility_labels"])}
    generalization = __import__("dense_failure_stage2.predictability_generalization", fromlist=["exact_knn_prediction"])
    output = []
    for layer in range(28):
        query_rows = [row for row in states if int(row["layer"]) == layer]
        if not query_rows:
            continue
        reference_rows = [row for row in internal_states if int(row["layer"]) == layer]
        if len(reference_rows) < int(config["question_encoder"]["knn_k"]):
            raise RuntimeError(f"under-supported Step-D exact-layer kNN: L{layer}")
        query = np.stack([external_embeddings[ext_row[str(row["uid"])]] for row in query_rows])
        reference = np.stack([internal_embeddings[int_row[str(row["uid"])]] for row in reference_rows])
        for target, column in (("read", "u_read_w1"), ("write", "u_write_r1")):
            truth = np.asarray([float(utilities[str(row["state_id"])][column]) for row in reference_rows])
            prediction = generalization.exact_knn_prediction(query, reference, truth, k=int(config["question_encoder"]["knn_k"]), block_size=512)
            for row, value in zip(query_rows, prediction):
                output.append({"state_id": row["state_id"], "uid": row["uid"], "layer": layer, "target": target, "prediction": float(value), "reference_states": len(reference_rows), "k": int(config["question_encoder"]["knn_k"])})
    if len(output) != 2 * len(states):
        raise RuntimeError("Step-D Stage-2 question-kNN census differs")
    atomic_jsonl(output_root / "question_semantics/stage2_question_knn_predictions.jsonl", output)
    atomic_json(output_root / "work/prediction_readiness.json", {"schema_version": "predictability_stepD_prediction_readiness_v1", "contract_sha256": contract["contract_sha256"], "external_stage1_states": int(config["external"]["expected_total"]) * 28, "external_stage2_states": len(states), "stage2_knn_rows": len(output), "measurement_absent": True, "completed_at": utc_now()})
    print(json.dumps({"prediction_ready": True, "stage2_states": len(states), "knn_rows": len(output)}, sort_keys=True))


def _load_refit_models(contract: Mapping[str, Any], output_root: Path, device: torch.device) -> dict[str, list[dict[str, Any]]]:
    parent_config = read_json(contract["static_config"]["parents"]["stepB_contract"])["static_config"]
    result = defaultdict(list)
    for completion in _refit_completions(contract, output_root):
        payload = torch.load(completion["checkpoint"], map_location="cpu", weights_only=False)
        task = payload["task"]
        mean = payload["normalization_mean"]
        if str(task["model"]).startswith("m3_z_"):
            width = int(parent_config["inputs"]["stage2_summary_width_each"])
        else:
            width = int(mean.numel())
        model = stepb._model_for_task(task, width, parent_config).to(device=device, dtype=torch.float32).eval()
        model.load_state_dict(payload["model_state"], strict=True)
        result[str(task["domain"])].append({"completion": completion, "payload": payload, "model": model})
    return dict(result)


def _summary_prediction(entry: Mapping[str, Any], values: torch.Tensor, layers: torch.Tensor | None, device: torch.device) -> np.ndarray:
    payload, task, model = entry["payload"], entry["payload"]["task"], entry["model"]
    mean = payload["normalization_mean"].to(device)
    std = payload["normalization_std"].to(device)
    normalized = (values.to(device=device, dtype=torch.float32) - mean) / std
    with torch.inference_mode():
        raw = model(normalized, layers.to(device)) if str(task["model"]) == "m3_current_head" else model(normalized)
    value = raw.float().cpu().numpy().astype(np.float64)
    if str(task["domain"]) == "stage1":
        return 1.0 / (1.0 + np.exp(-np.clip(value, -50, 50)))
    return value * float(payload["target_scale"]) + float(payload["target_center"])


def prediction_worker(config_path: Path, rank: int, *, resume: bool) -> None:
    contract, output_root, _ = verify_contract(config_path)
    readiness = read_json(output_root / "work/prediction_readiness.json")
    if readiness.get("contract_sha256") != contract["contract_sha256"] or not readiness.get("measurement_absent"):
        raise RuntimeError("Step-D prediction readiness differs")
    rank = int(rank)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    models = _load_refit_models(contract, output_root, device)
    schedule = [row for row in read_jsonl(output_root / "work/external_state_schedule.jsonl") if int(row["worker_rank"]) == rank]
    dense_states = defaultdict(list)
    for row in read_jsonl(output_root / "external_manifests/dense_state_manifest.jsonl"):
        dense_states[str(row["uid"])].append(row)
    stage2_states = defaultdict(list)
    for row in read_jsonl(output_root / "external_manifests/stage2_dense_state_manifest.jsonl"):
        stage2_states[str(row["uid"])].append(row)
    semantic = {str(row["uid"]): row for row in read_jsonl(output_root / "question_semantics/nearest_internal_similarity.jsonl")}
    knn = {(str(row["state_id"]), str(row["target"])): float(row["prediction"]) for row in read_jsonl(output_root / "question_semantics/stage2_question_knn_predictions.jsonl")}
    root = output_root / f"work/predictions/rank{rank:02d}"
    root.mkdir(parents=True, exist_ok=True)
    completed = 0
    for item in schedule:
        uid = str(item["uid"])
        slug = sha256(uid.encode()).hexdigest()[:24]
        path = root / f"{slug}.json"
        if resume and path.is_file() and read_json(path).get("contract_sha256") == contract["contract_sha256"]:
            completed += 1
            continue
        state_row = sorted(dense_states[uid], key=lambda row: int(row["layer"]))
        payload = torch.load(state_row[0]["state_file"], map_location="cpu", weights_only=False)
        if payload.get("contract_sha256") != contract["contract_sha256"] or file_sha256(state_row[0]["state_file"]) != state_row[0]["state_file_sha256"]:
            raise RuntimeError(f"Step-D prediction state provenance differs: {uid}")
        features = payload["stage1_features"].float()
        layers = torch.arange(28, dtype=torch.long)
        nuisance = torch.from_numpy(stage1_transfer_nuisance(state_row))
        stage1_columns = {}
        m3_seed_values = []
        for entry in models["stage1"]:
            label = str(entry["completion"]["label"])
            source = nuisance if label == "m0_x" else features
            values = _summary_prediction(entry, source, layers if label == "m3" else None, device)
            name = f"{label}_s{entry['completion']['seed']}"
            stage1_columns[name] = values.tolist()
            if label == "m3":
                m3_seed_values.append(values)
        stage1_columns["m3_ensemble"] = np.mean(m3_seed_values, axis=0).tolist()
        stage1_output = []
        for index, row in enumerate(state_row):
            stage1_output.append({"state_id": row["state_id"], "uid": uid, "benchmark": row["benchmark"], "benchmark_family": row["benchmark_family"], "image_group_id": row["image_group_id"], "layer": int(row["layer"]), "contract_sha256": contract["contract_sha256"], "prediction_m0_x": float(stage1_columns[[key for key in stage1_columns if key.startswith('m0_x_')][0]][index]), "prediction_m1": float(stage1_columns[[key for key in stage1_columns if key.startswith('m1_')][0]][index]), "prediction_m3_seed1": float(stage1_columns[f"m3_s2026090801"][index]), "prediction_m3_seed2": float(stage1_columns[f"m3_s2026090802"][index]), "prediction_m3_seed3": float(stage1_columns[f"m3_s2026090803"][index]), "prediction_m3_ensemble": float(stage1_columns["m3_ensemble"][index]), "prediction_question_knn": float(semantic[uid]["question_knn_failure_prediction"]), "nearest_internal_similarity": float(semantic[uid]["nearest_internal_similarity"])})
        stage2_output = []
        for row in sorted(stage2_states.get(uid, []), key=lambda value: int(value["layer"])):
            state = payload["stage2_states"][str(row["state_id"])]
            last_text = int(state["text_mask"][0].long().sum().item()) - 1
            if last_text < 0:
                raise RuntimeError(f"Step-D Stage-2 state lacks valid text tokens: {row['state_id']}")
            summary = torch.cat((state["text_states"][0, last_text].float(), state["visual_states"][0][state["visual_mask"][0]].float().mean(dim=0)))[None]
            nuisance2 = torch.from_numpy(stage2_transfer_nuisance([row]))
            target_predictions = {}
            for target in ("read", "write"):
                m3_values = []
                for entry in models["stage2_dense"]:
                    if str(entry["completion"]["target"]) != target:
                        continue
                    label = str(entry["completion"]["label"])
                    if label == "m3":
                        with torch.inference_mode():
                            raw = entry["model"](
                                state["text_states"].to(device=device, dtype=torch.float32),
                                state["visual_states"].to(device=device, dtype=torch.float32),
                                text_mask=state["text_mask"].to(device),
                                visual_mask=state["visual_mask"].to(device),
                            ).float().cpu().numpy().astype(np.float64)
                        value = float(raw[0] * float(entry["payload"]["target_scale"]) + float(entry["payload"]["target_center"]))
                        m3_values.append((int(entry["completion"]["seed"]), value))
                    else:
                        source = nuisance2 if label == "m0_x" else summary
                        value = float(_summary_prediction(entry, source, None, device)[0])
                    target_predictions[f"{target}_{label}_s{entry['completion']['seed']}"] = value
                target_predictions[f"{target}_m3_ensemble"] = float(np.mean([value for _, value in m3_values]))
                target_predictions[f"{target}_question_knn"] = knn[(str(row["state_id"]), target)]
            stage2_output.append({"state_id": row["state_id"], "uid": uid, "benchmark": row["benchmark"], "benchmark_family": row["benchmark_family"], "image_group_id": row["image_group_id"], "layer": int(row["layer"]), "trigger_layer": int(row["trigger_layer"]), "trigger_relative_depth": int(row["trigger_relative_depth"]), "contract_sha256": contract["contract_sha256"], "prediction_read_m3_ensemble": target_predictions["read_m3_ensemble"], "prediction_write_m3_ensemble": target_predictions["write_m3_ensemble"], **target_predictions})
        atomic_json(path, {"schema_version": "predictability_stepD_uid_predictions_v1", "contract_sha256": contract["contract_sha256"], "uid": uid, "stage1": stage1_output, "stage2": stage2_output})
        completed += 1
        if completed % 100 == 0 or completed == len(schedule):
            print(json.dumps({"rank": rank, "predicted_uids": completed, "assigned": len(schedule)}, sort_keys=True), flush=True)
    atomic_json(root / "complete.json", {"schema_version": "predictability_stepD_prediction_rank_complete_v1", "contract_sha256": contract["contract_sha256"], "rank": rank, "expected_uids": len(schedule), "completed_uids": completed, "completed_at": utc_now()})


def finalize_predictions(config_path: Path) -> None:
    contract, output_root, _ = verify_contract(config_path)
    config = contract["static_config"]
    schedule = read_jsonl(output_root / "work/external_state_schedule.jsonl")
    stage1 = []
    stage2 = []
    for rank in range(int(config["world_size"])):
        assigned = [row for row in schedule if int(row["worker_rank"]) == rank]
        root = output_root / f"work/predictions/rank{rank:02d}"
        marker = read_json(root / "complete.json")
        if marker.get("contract_sha256") != contract["contract_sha256"] or int(marker.get("completed_uids", -1)) != len(assigned):
            raise RuntimeError(f"incomplete Step-D prediction rank {rank}")
        for item in assigned:
            row = read_json(root / f"{sha256(str(item['uid']).encode()).hexdigest()[:24]}.json")
            if row.get("uid") != item["uid"] or row.get("contract_sha256") != contract["contract_sha256"]:
                raise RuntimeError(f"invalid Step-D UID prediction: {item['uid']}")
            stage1.extend(row["stage1"])
            stage2.extend(row["stage2"])
    stage1_manifest = read_jsonl(output_root / "external_manifests/dense_state_manifest.jsonl")
    stage2_manifest = read_jsonl(output_root / "external_manifests/stage2_dense_state_manifest.jsonl")
    validate_prediction_census([row["state_id"] for row in stage1_manifest], stage1, contract_sha256=contract["contract_sha256"])
    validate_prediction_census([row["state_id"] for row in stage2_manifest], stage2, contract_sha256=contract["contract_sha256"])
    root = output_root / "predictions_frozen_before_labels"
    stage1_path = root / "stage1_external_predictions.jsonl"
    read_path = root / "stage2_read_external_predictions.jsonl"
    write_path = root / "stage2_write_external_predictions.jsonl"
    atomic_jsonl(stage1_path, sorted(stage1, key=lambda row: (str(row["uid"]), int(row["layer"]))))
    common = ("state_id", "uid", "benchmark", "benchmark_family", "image_group_id", "layer", "trigger_layer", "trigger_relative_depth", "contract_sha256")
    atomic_jsonl(read_path, ({**{key: row[key] for key in common}, **{key.removeprefix("read_"): value for key, value in row.items() if key.startswith("read_")}} for row in sorted(stage2, key=lambda value: (str(value["uid"]), int(value["layer"])))))
    atomic_jsonl(write_path, ({**{key: row[key] for key in common}, **{key.removeprefix("write_"): value for key, value in row.items() if key.startswith("write_")}} for row in sorted(stage2, key=lambda value: (str(value["uid"]), int(value["layer"])))))
    model_manifest_sha = sha256("".join(sorted(row["checkpoint_sha256"] for row in _refit_completions(contract, output_root))).encode()).hexdigest()
    hashes = freeze_prediction_hashes({"stage1": stage1_path, "stage2_read": read_path, "stage2_write": write_path}, contract_sha256=contract["contract_sha256"], model_manifest_sha256=model_manifest_sha)
    hashes["frozen_at"] = utc_now()
    hashes["external_utility_labels_present_at_freeze"] = False
    atomic_json(root / "prediction_hashes.json", hashes)
    print(json.dumps({"predictions_frozen": True, "stage1_states": len(stage1), "stage2_states": len(stage2), "hashes": {key: value["sha256"] for key, value in hashes["files"].items()}}, sort_keys=True))


def _verify_frozen_predictions(contract: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    frozen = read_json(output_root / "predictions_frozen_before_labels/prediction_hashes.json")
    expected = {
        key: frozen[key]
        for key in ("schema_version", "contract_sha256", "model_manifest_sha256", "files")
    }
    paths = {name: row["path"] for name, row in frozen["files"].items()}
    freeze_prediction_hashes(
        paths,
        contract_sha256=contract["contract_sha256"],
        model_manifest_sha256=frozen["model_manifest_sha256"],
        expected=expected,
    )
    if frozen.get("external_utility_labels_present_at_freeze") is not False:
        raise RuntimeError("Step-D prediction freeze did not precede utility labels")
    return frozen


def measurement_worker(config_path: Path, rank: int, *, resume: bool) -> None:
    contract, output_root, _ = verify_contract(config_path, verify_model=True)
    _verify_frozen_predictions(contract, output_root)
    config = contract["static_config"]
    rank = int(rank)
    phase69.configure_dense_determinism(int(config["seed"]) + 100 + rank, config["backend_settings"])
    runtime = DenseRuntime(config, rank)
    manifest = {str(row["uid"]): row for row in read_jsonl(output_root / "external_manifests/benchmark_manifest.jsonl")}
    state_manifest = read_jsonl(output_root / "external_manifests/stage2_dense_state_manifest.jsonl")
    by_uid = defaultdict(list)
    for row in state_manifest:
        by_uid[str(row["uid"])].append(row)
    assigned = [
        row for row in read_jsonl(output_root / "work/external_state_schedule.jsonl")
        if int(row["worker_rank"]) == rank and str(row["uid"]) in by_uid
    ]
    root = output_root / f"work/measurement/rank{rank:02d}"
    root.mkdir(parents=True, exist_ok=True)
    completed = 0
    started = time.monotonic()
    for item in assigned:
        uid = str(item["uid"])
        slug = sha256(uid.encode()).hexdigest()[:24]
        result_path = root / f"{slug}.json"
        if resume and result_path.is_file():
            old = read_json(result_path)
            if old.get("contract_sha256") == contract["contract_sha256"] and len(old.get("state_results", [])) == len(by_uid[uid]):
                completed += 1
                continue
        sample = manifest[uid]
        source_rank = int(item["worker_rank"])
        source = read_json(output_root / f"work/external_states/rank{source_rank:02d}/{slug}.json")
        if file_sha256(source["state_file"]) != source["state_file_sha256"]:
            raise RuntimeError(f"Step-D source state file differs before measurement: {uid}")
        cached_payload = torch.load(source["state_file"], map_location="cpu", weights_only=False)
        dense, baseline, prepared, input_ids, instruction_count = _full_baseline(runtime, sample)
        if (
            dense["generated_token_ids"] != source["dense_generated_token_ids"]
            or dense["score"] != source["dense_score"]
            or dense["correct"] != source["dense_correct"]
            or instruction_count != int(source["user_text_token_count"])
        ):
            raise RuntimeError(f"Step-D measurement baseline differs from state census: {uid}")
        results = []
        parity_rows = []
        for state_row in sorted(by_uid[uid], key=lambda row: int(row["layer"])):
            state_id = str(state_row["state_id"])
            layer = int(state_row["layer"])
            cached = cached_payload["stage2_states"][state_id]
            live = stepa._cached_state(
                *baseline.pre_layer_states[layer],
                prepared.text_valid_mask,
                prepared.visual_valid_mask,
            )
            if live["state_sha256"] != cached["state_sha256"] or live["tensor_sha256"] != cached["tensor_sha256"] or live["state_sha256"] != state_row["state_sha256"]:
                raise RuntimeError(f"Step-D live/cached state parity failed: {state_id}")
            branches, parity = _branch_actions(runtime, sample, baseline, input_ids, layer, dense)
            derived = derive_utility_row(branches)
            results.append({"state_id": state_id, "uid": uid, "layer": layer, "branches": branches, "branch_order": list(ACTIONS), "derived": derived})
            parity_rows.append({"state_id": state_id, "layer": layer, "state_hash_exact": True, **parity})
        validate_complete_state_results([row["state_id"] for row in by_uid[uid]], [{"state_id": row["state_id"], "branches": row["branch_order"]} for row in results])
        atomic_json(result_path, {"schema_version": "predictability_stepD_measurement_uid_v1", "contract_sha256": contract["contract_sha256"], "prediction_hashes_sha256": file_sha256(output_root / "predictions_frozen_before_labels/prediction_hashes.json"), "uid": uid, "benchmark": sample["benchmark"], "benchmark_family": sample["benchmark_family"], "image_group_id": sample["image_group_id"], "dense_correct": source["dense_correct"], "state_ids": [row["state_id"] for row in by_uid[uid]], "state_results": results, "parity": parity_rows})
        completed += 1
        del baseline, prepared, input_ids, cached_payload, results
        torch.cuda.empty_cache()
        if completed % 5 == 0 or completed == len(assigned):
            print(json.dumps({"rank": rank, "measured_uids": completed, "assigned": len(assigned), "elapsed_seconds": time.monotonic() - started}, sort_keys=True), flush=True)
    atomic_json(root / "complete.json", {"schema_version": "predictability_stepD_measurement_rank_complete_v1", "contract_sha256": contract["contract_sha256"], "prediction_hashes_sha256": file_sha256(output_root / "predictions_frozen_before_labels/prediction_hashes.json"), "rank": rank, "expected_uids": len(assigned), "completed_uids": completed, "completed_at": utc_now(), "elapsed_seconds": time.monotonic() - started})


def finalize_measurement(config_path: Path) -> None:
    contract, output_root, _ = verify_contract(config_path)
    frozen = _verify_frozen_predictions(contract, output_root)
    config = contract["static_config"]
    manifest = read_jsonl(output_root / "external_manifests/stage2_dense_state_manifest.jsonl")
    metadata = {str(row["state_id"]): row for row in manifest}
    if len(metadata) != len(manifest):
        raise RuntimeError("Step-D measurement manifest contains duplicate states")
    schedule = read_jsonl(output_root / "work/external_state_schedule.jsonl")
    triggered = {str(row["uid"]) for row in manifest}
    uid_results = []
    for rank in range(int(config["world_size"])):
        assigned = [row for row in schedule if int(row["worker_rank"]) == rank and str(row["uid"]) in triggered]
        root = output_root / f"work/measurement/rank{rank:02d}"
        marker = read_json(root / "complete.json")
        if marker.get("contract_sha256") != contract["contract_sha256"] or int(marker.get("completed_uids", -1)) != len(assigned) or marker.get("prediction_hashes_sha256") != file_sha256(output_root / "predictions_frozen_before_labels/prediction_hashes.json"):
            raise RuntimeError(f"incomplete Step-D measurement rank {rank}")
        for item in assigned:
            row = read_json(root / f"{sha256(str(item['uid']).encode()).hexdigest()[:24]}.json")
            if row.get("contract_sha256") != contract["contract_sha256"] or row.get("prediction_hashes_sha256") != file_sha256(output_root / "predictions_frozen_before_labels/prediction_hashes.json"):
                raise RuntimeError(f"invalid Step-D measurement result: {item['uid']}")
            uid_results.append(row)
    branch_rows = []
    utility_rows = []
    flip_rows = []
    parity_rows = []
    completeness = []
    for uid_result in uid_results:
        for parity in uid_result["parity"]:
            parity_rows.append({"uid": uid_result["uid"], "benchmark": uid_result["benchmark"], **parity})
        for state in uid_result["state_results"]:
            state_id = str(state["state_id"])
            meta = metadata[state_id]
            common = {"state_id": state_id, "uid": meta["uid"], "benchmark": meta["benchmark"], "benchmark_family": meta["benchmark_family"], "image_group_id": meta["image_group_id"], "dense_correct": meta["dense_correct"], "dense_wrong": meta["dense_wrong"], "trigger_layer": meta["trigger_layer"], "layer": meta["layer"], "trigger_relative_depth": meta["trigger_relative_depth"]}
            for action in ACTIONS:
                branch_rows.append({"schema_version": "predictability_stepD_four_branch_result_v1", "contract_sha256": contract["contract_sha256"], **common, "action": action, **state["branches"][action]})
            recomputed = derive_utility_row(state["branches"])
            maximum = max(abs(float(recomputed[key]) - float(state["derived"][key])) for key in ("u_read_w1", "u_read_w0", "u_write_r1", "u_write_r0", "u_read", "u_write", "u_interaction", "full_gap"))
            if maximum > 1e-12:
                raise RuntimeError(f"Step-D utility algebra differs: {state_id}")
            utility_rows.append({**common, **recomputed})
            flip_rows.append({**common, **{key: value for key, value in recomputed.items() if "flip" in key or key in {"local_rescue_exists", "local_regression_exists", "all_four_correct", "all_four_wrong", "correct_action_count"}}})
            completeness.append({"state_id": state_id, "branches": list(state["branch_order"])})
    validate_complete_state_results([row["state_id"] for row in manifest], completeness)
    if len(branch_rows) != 4 * len(manifest) or len(utility_rows) != len(manifest):
        raise RuntimeError("Step-D four-branch global census differs")
    root = output_root / "stage2_measurement"
    atomic_jsonl(root / "four_branch_results.jsonl", branch_rows)
    atomic_csv(root / "utility_labels.csv", utility_rows)
    atomic_csv(root / "correctness_flip_labels.csv", flip_rows)
    atomic_csv(root / "full_branch_parity.csv", parity_rows)
    parity_pass = all(bool(row["passed"]) and bool(row["state_hash_exact"]) for row in parity_rows)
    if not parity_pass:
        raise RuntimeError("Step-D aggregate FULL/state parity failed")
    atomic_json(root / "parity_report.json", {"schema_version": "predictability_stepD_measurement_parity_v1", "contract_sha256": contract["contract_sha256"], "prediction_hashes": {name: row["sha256"] for name, row in frozen["files"].items()}, "triggered_uids": len(uid_results), "states": len(manifest), "branches": len(branch_rows), "full_branch_and_state_parity": parity_pass, "utility_algebra": True, "complete": True})
    _atomic(root / "parity_report.md", (f"# Step-D Stage-2 measurement parity\n\n- Contract: `{contract['contract_sha256']}`\n- Predictions were frozen before labels: PASS\n- Triggered UIDs: {len(uid_results):,}\n- Dense post-trigger states: {len(manifest):,}\n- Four-action branches: {len(branch_rows):,}\n- Live/cached state hashes: PASS\n- FULL answer/score/correctness/q parity: PASS\n- Utility algebra and global completeness: PASS\n").encode())
    print(json.dumps({"measurement_complete": True, "uids": len(uid_results), "states": len(manifest), "branches": len(branch_rows)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    refit = subparsers.add_parser("refit-worker")
    refit.add_argument("--rank", type=int, required=True)
    refit.add_argument("--resume", action="store_true")
    subparsers.add_parser("finalize-refits")
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--device", type=int, default=0)
    state = subparsers.add_parser("external-state-worker")
    state.add_argument("--rank", type=int, required=True)
    state.add_argument("--resume", action="store_true")
    subparsers.add_parser("finalize-external-states")
    embedding = subparsers.add_parser("embedding-worker")
    embedding.add_argument("--rank", type=int, required=True)
    subparsers.add_parser("finalize-embeddings")
    subparsers.add_parser("prepare-predictions")
    prediction = subparsers.add_parser("prediction-worker")
    prediction.add_argument("--rank", type=int, required=True)
    prediction.add_argument("--resume", action="store_true")
    subparsers.add_parser("finalize-predictions")
    measurement = subparsers.add_parser("measurement-worker")
    measurement.add_argument("--rank", type=int, required=True)
    measurement.add_argument("--resume", action="store_true")
    subparsers.add_parser("finalize-measurement")
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare(arguments.config)
    elif arguments.command == "refit-worker":
        refit_worker(arguments.config, arguments.rank, resume=arguments.resume)
    elif arguments.command == "finalize-refits":
        finalize_refits(arguments.config)
    elif arguments.command == "smoke":
        smoke(arguments.config, arguments.device)
    elif arguments.command == "external-state-worker":
        external_state_worker(arguments.config, arguments.rank, resume=arguments.resume)
    elif arguments.command == "finalize-external-states":
        finalize_external_states(arguments.config)
    elif arguments.command == "embedding-worker":
        embedding_worker(arguments.config, arguments.rank)
    elif arguments.command == "finalize-embeddings":
        finalize_embeddings(arguments.config)
    elif arguments.command == "prepare-predictions":
        prepare_predictions(arguments.config)
    elif arguments.command == "prediction-worker":
        prediction_worker(arguments.config, arguments.rank, resume=arguments.resume)
    elif arguments.command == "finalize-predictions":
        finalize_predictions(arguments.config)
    elif arguments.command == "measurement-worker":
        measurement_worker(arguments.config, arguments.rank, resume=arguments.resume)
    elif arguments.command == "finalize-measurement":
        finalize_measurement(arguments.config)
    else:
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
