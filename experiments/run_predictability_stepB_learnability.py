#!/usr/bin/env python3
"""Run the frozen Predictability Step-B in-domain learnability study."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
import copy
import csv
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dense_failure_stage2.predictability_learnability import (  # noqa: E402
    RouterStyleUtilityRegressor,
    SummaryScalarPredictor,
    assign_inner_group_roles,
    assign_shared_group_folds,
    robust_target_scale,
    validate_prediction_roundtrip,
)
from dense_failure_stage1.shared_global_gate import SharedFailurePredictor  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/predictability_stepB_id_learnability_v1.json"
BOUND_CODE = (
    "configs/predictability_stepB_id_learnability_v1.json",
    "dense_failure_stage1/shared_global_gate.py",
    "dense_failure_stage2/v1_router.py",
    "dense_failure_stage2/predictability_learnability.py",
    "experiments/run_predictability_stepB_learnability.py",
    "experiments/aggregate_predictability_stepB_learnability.py",
)
FEATURE_NAMES = ("text_final", "text_mean", "visual_mean")


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_stepa_artifact_path(config: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return resolve_path(config["sources"]["stepA_contract"]).parent / path


def file_sha256(value: str | Path) -> str:
    digest = sha256()
    with resolve_path(value).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = f"{value.dtype}:{tuple(value.shape)}:".encode()
    return sha256(header + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def state_tensor_hashes(state: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: tensor_sha256(state[name])
        for name in ("text_states", "visual_states", "text_mask", "visual_mask")
    }


def combined_state_hash(hashes: Mapping[str, str]) -> str:
    return sha256(
        json.dumps(dict(hashes), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = {
        key: item
        for key, item in value.items()
        if key not in {"contract_sha256", "artifact_manifest_sha256"}
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def read_json(value: str | Path) -> dict[str, Any]:
    with resolve_path(value).open() as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object: {value}")
    return result


def read_jsonl(value: str | Path) -> list[dict[str, Any]]:
    rows = []
    with resolve_path(value).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {value}:{line_number}")
            rows.append(row)
    return rows


def read_csv(value: str | Path) -> list[dict[str, str]]:
    with resolve_path(value).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    _atomic_bytes(path, payload.encode())


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def command_output(arguments: Sequence[str]) -> str:
    return subprocess.run(
        list(arguments), cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def git_state() -> dict[str, str]:
    return {
        "commit": command_output(("git", "rev-parse", "HEAD")),
        "branch": command_output(("git", "branch", "--show-current")),
        "worktree_status": command_output(("git", "status", "--short")),
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def runtime_state() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "scipy": package_version("scipy"),
        "scikit_learn": package_version("scikit-learn"),
    }


def load_config(value: str | Path) -> dict[str, Any]:
    path = resolve_path(value)
    config = read_json(path)
    if config.get("schema_version") != "predictability_stepB_id_learnability_config_v1":
        raise ValueError("unsupported Step-B config schema")
    if int(config["world_size"]) != 4 or int(config["split"]["outer_folds"]) != 5:
        raise ValueError("Step-B requires four workers and five outer folds")
    if list(config["training"]["seeds"]) != [2026090801, 2026090802, 2026090803]:
        raise ValueError("Step-B three-seed contract differs")
    return config


def _verify_stepa(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = read_json(config["sources"]["stepA_contract"])
    manifest = read_json(config["sources"]["stepA_artifact_manifest"])
    expected = "6103b9b91826455e9ebed97c2ee19c5e115a3ca0835006fdf2e5bf049934613d"
    if contract.get("contract_sha256") != expected or canonical_hash(contract) != expected:
        raise RuntimeError("Step-A contract differs from the approved input")
    claimed = manifest.get("artifact_manifest_sha256")
    if claimed != canonical_hash(manifest):
        raise RuntimeError("Step-A artifact manifest self-hash differs")
    if manifest.get("contract_sha256") != expected or not manifest.get("ready_for_step_b"):
        raise RuntimeError("Step-A is not measurement-ready")
    step_a_root = resolve_path(config["sources"]["stepA_contract"]).parent
    for relative, digest in manifest["files"].items():
        if file_sha256(step_a_root / relative) != digest:
            raise RuntimeError(f"Step-A compact artifact hash differs: {relative}")
    return contract, manifest


def _base_population(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    internal = read_jsonl(config["sources"]["internal_samples"])
    rows = []
    seen = set()
    for row in internal:
        uid = str(row["uid"])
        if uid in seen:
            raise RuntimeError(f"duplicate internal UID: {uid}")
        seen.add(uid)
        rows.append(
            {
                "uid": uid,
                "image_group_id": str(row["image_group_id"]),
                "dataset": str(row["dataset"]),
                "source_regime": str(row["source_regime"]),
                "dense_wrong": bool(row["dense_wrong"]),
                "p90_triggered": bool(row["p90"]["triggered"]),
                "visual_token_count": int(row["dense"]["visual_token_count"]),
                "user_text_token_count": int(row["dense"]["user_text_token_count"]),
                "prompt_token_count": int(row["dense"]["prompt_token_count"]),
            }
        )
    if len(rows) != int(config["population"]["internal_uids"]):
        raise RuntimeError("Step-B internal UID population differs")
    return rows


def _fold_support(
    config: Mapping[str, Any], registry: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    fold_by_uid = {str(row["uid"]): int(row["fold"]) for row in registry}
    base_by_uid = {str(row["uid"]): row for row in registry}
    dense_utilities = read_csv(config["sources"]["stage2_dense_utilities"])
    dense_flips = {str(row["state_id"]): row for row in read_csv(config["sources"]["stage2_dense_flips"])}
    routed_utilities = read_csv(config["sources"]["stage2_routed_utilities"])
    support = []
    for fold in range(int(config["split"]["outer_folds"])):
        uids = [uid for uid, value in fold_by_uid.items() if value == fold]
        base = [base_by_uid[uid] for uid in uids]
        dense = [row for row in dense_utilities if fold_by_uid[str(row["uid"])] == fold]
        routed = [row for row in routed_utilities if fold_by_uid[str(row["uid"])] == fold]
        support.append(
            {
                "fold": fold,
                "image_groups": len({str(row["image_group_id"]) for row in base}),
                "uids": len(base),
                "stage1_correct": sum(not bool(row["dense_wrong"]) for row in base),
                "stage1_wrong": sum(bool(row["dense_wrong"]) for row in base),
                "p90_triggered_uids": sum(bool(row["p90_triggered"]) for row in base),
                "stage2_dense_states": len(dense),
                "dense_read_harmful": sum(float(row["u_read_w1"]) < 0.0 for row in dense),
                "dense_read_beneficial": sum(float(row["u_read_w1"]) > 0.0 for row in dense),
                "dense_write_harmful": sum(float(row["u_write_r1"]) < 0.0 for row in dense),
                "dense_write_beneficial": sum(float(row["u_write_r1"]) > 0.0 for row in dense),
                "dense_read_harmful_flips": sum(
                    str(dense_flips[str(row["state_id"])]["read_harmful_flip_w1"]).lower() == "true"
                    for row in dense
                ),
                "dense_write_harmful_flips": sum(
                    str(dense_flips[str(row["state_id"])]["write_harmful_flip_r1"]).lower() == "true"
                    for row in dense
                ),
                "stage2_routed_states": len(routed),
                "routed_read_harmful": sum(float(row["u_read_w1"]) < 0.0 for row in routed),
                "routed_read_beneficial": sum(float(row["u_read_w1"]) > 0.0 for row in routed),
                "routed_write_harmful": sum(float(row["u_write_r1"]) < 0.0 for row in routed),
                "routed_write_beneficial": sum(float(row["u_write_r1"]) > 0.0 for row in routed),
            }
        )
    return support


def _validate_fold_registry(
    config: Mapping[str, Any], registry: Sequence[Mapping[str, Any]], support: Sequence[Mapping[str, Any]]
) -> None:
    expected = int(config["population"]["internal_uids"])
    if len(registry) != expected or len({str(row["uid"]) for row in registry}) != expected:
        raise RuntimeError("shared fold registry UID completeness differs")
    group_folds: dict[str, set[int]] = defaultdict(set)
    for row in registry:
        group_folds[str(row["image_group_id"])].add(int(row["fold"]))
    if any(len(values) != 1 for values in group_folds.values()):
        raise RuntimeError("shared fold registry leaks image groups")
    required_positive = (
        "stage1_correct", "stage1_wrong", "p90_triggered_uids", "stage2_dense_states",
        "dense_read_harmful", "dense_read_beneficial", "dense_write_harmful",
        "dense_write_beneficial", "stage2_routed_states", "routed_read_harmful",
        "routed_read_beneficial", "routed_write_harmful", "routed_write_beneficial",
    )
    for row in support:
        absent = [key for key in required_positive if int(row[key]) < 1]
        if absent:
            raise RuntimeError(f"fold {row['fold']} lacks required support: {absent}")


def _protocol(config: Mapping[str, Any], step_a_contract: Mapping[str, Any]) -> str:
    return f"""# Predictability Step-B frozen protocol

- Run ID: `{config['run_id']}`
- Parent Step-A contract: `{step_a_contract['contract_sha256']}`
- Evaluation: five image-group-disjoint outer folds with one shared base-group registry.
- Inner calibration: deterministic 10% group-disjoint subset of each outer-training population.
- Weighting: every UID contributes equal total training/calibration mass.
- Stage-1 target: current dense all-FULL eventual wrongness.
- Stage-2 primary READ target: `q_F - q_WO` (`u_read_w1`).
- Stage-2 primary WRITE target: `q_F - q_RO` (`u_write_r1`).
- M0: nuisance-only control. M1: frozen summary linear. M2: fixed two-layer summary MLP.
- M3 Stage 1: current `SharedFailurePredictor` architecture, including its structural layer embedding.
- M3 Stage 2: current READ/WRITE cross-modal topology trained from scratch per target/representation/fold/seed.
- M1/M2/M3 receive no dataset/source/trigger-depth/final-outcome/utility/future/search inputs except the structural Stage-1 M3 layer embedding noted above.
- M2/M3 seeds: `{config['training']['seeds']}`.
- Regression target scaling: outer-fit median and `1.4826*MAD`, floor `{config['targets']['target_scale_floor']}`; no clipping.
- Primary dense and secondary selected routed domains remain separate.
- This phase is in-domain only. It performs no nearest-question, clustering, LODO, external, or deployment evaluation.
"""


def _training_tasks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    seeds = [int(value) for value in config["training"]["seeds"]]
    base_seed = int(config["seed"])
    folds = int(config["split"]["outer_folds"])
    tasks: list[dict[str, Any]] = []

    def add(
        domain: str,
        fold: int,
        model: str,
        input_name: str,
        target: str,
        seed: int,
        cost: int,
    ) -> None:
        task_id = f"{domain}__f{fold}__{target}__{model}__{input_name}__s{seed}"
        tasks.append(
            {
                "task_id": task_id,
                "domain": domain,
                "fold": int(fold),
                "model": model,
                "input": input_name,
                "target": target,
                "seed": int(seed),
                "estimated_cost": int(cost),
            }
        )

    for fold in range(folds):
        add("stage1", fold, "m0_nuisance", "nuisance", "dense_wrong", base_seed, 1)
        add("stage1", fold, "m1_linear", "state", "dense_wrong", base_seed, 3)
        for seed in seeds:
            add("stage1", fold, "m2_mlp", "state", "dense_wrong", seed, 5)
            add("stage1", fold, "m3_current_head", "state_layer", "dense_wrong", seed, 6)

    primary_models = (
        ("m0_nuisance", "nuisance", (base_seed,), 1),
        ("m0_dense_outcome_only", "dense_outcome", (base_seed,), 1),
        ("m0_nuisance_plus_dense_outcome", "nuisance_dense_outcome", (base_seed,), 1),
        ("m1_text", "text", (base_seed,), 2),
        ("m1_visual", "visual", (base_seed,), 2),
        ("m1_text_visual", "text_visual", (base_seed,), 3),
        ("m2_text_visual", "text_visual", tuple(seeds), 5),
        ("m3_z_R", "z_R", tuple(seeds), 20),
        ("m3_z_W", "z_W", tuple(seeds), 20),
        ("m3_z_RW", "z_RW", tuple(seeds), 24),
    )
    for domain in ("stage2_dense", "stage2_routed"):
        for fold in range(folds):
            for target in ("read", "write"):
                for model, input_name, model_seeds, cost in primary_models:
                    for seed in model_seeds:
                        add(domain, fold, model, input_name, target, seed, cost)

    secondary_models = (
        ("m0_nuisance", "nuisance", (base_seed,), 1),
        ("m1_text_visual", "text_visual", (base_seed,), 3),
        ("m2_text_visual", "text_visual", tuple(seeds), 5),
        ("m3_z_RW", "z_RW", tuple(seeds), 24),
    )
    for fold in range(folds):
        for target in config["targets"]["secondary"]:
            for model, input_name, model_seeds, cost in secondary_models:
                for seed in model_seeds:
                    add("stage2_dense", fold, model, input_name, str(target), seed, cost)

    seen = {str(task["task_id"]) for task in tasks}
    if len(seen) != len(tasks):
        raise RuntimeError("duplicate Step-B training task ID")
    load = [0] * int(config["world_size"])
    for task in sorted(tasks, key=lambda value: (-int(value["estimated_cost"]), str(value["task_id"]))):
        rank = min(range(len(load)), key=lambda value: (load[value], value))
        task["worker_rank"] = rank
        load[rank] += int(task["estimated_cost"])
    return sorted(tasks, key=lambda value: str(value["task_id"]))


def prepare(config_path: Path) -> None:
    config = load_config(config_path)
    step_a_contract, step_a_manifest = _verify_stepa(config)
    output_root = resolve_path(config["output_root"])
    external_root = resolve_path(config["external_cache_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    external_root.mkdir(parents=True, exist_ok=True)
    existing_contract_path = output_root / "frozen_contract.json"
    if existing_contract_path.is_file():
        existing_contract = read_json(existing_contract_path)
        existing_hash = str(existing_contract.get("contract_sha256", ""))
        if not existing_hash or canonical_hash(existing_contract) != existing_hash:
            raise RuntimeError("refusing to replace an invalid prior Step-B contract")
        archive_root = output_root / "validation/contract_archive"
        atomic_json(archive_root / f"{existing_hash}.json", existing_contract)
        validation_path = output_root / "validation/cache_validation.json"
        if validation_path.is_file():
            validation = read_json(validation_path)
            if validation.get("contract_sha256") != existing_hash:
                raise RuntimeError("prior cache validation does not match prior contract")
            atomic_json(archive_root / f"{existing_hash}_cache_validation.json", validation)
    for relative in (
        "splits", "stage1/model_configs", "stage2_dense/targets", "stage2_dense/models",
        "stage2_routed/models", "statistics", "figures", "summaries", "validation", "work",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    cache_link = output_root / "cache"
    if cache_link.exists() or cache_link.is_symlink():
        if not cache_link.is_symlink() or cache_link.resolve() != external_root:
            raise RuntimeError("Step-B cache link already points elsewhere")
    else:
        cache_link.symlink_to(external_root, target_is_directory=True)
    for domain in ("stage2_dense", "stage2_routed"):
        model_root = output_root / domain / "models"
        for family in ("nuisance", "linear", "mlp", "router_style"):
            external_models = external_root / "models" / domain / family
            external_models.mkdir(parents=True, exist_ok=True)
            link = model_root / family
            if link.exists() or link.is_symlink():
                if not link.is_symlink() or link.resolve() != external_models:
                    raise RuntimeError(f"Step-B model link already points elsewhere: {link}")
            else:
                link.symlink_to(external_models, target_is_directory=True)

    base = _base_population(config)
    registry = assign_shared_group_folds(
        base, folds=int(config["split"]["outer_folds"]), seed=int(config["seed"])
    )
    support = _fold_support(config, registry)
    _validate_fold_registry(config, registry, support)
    atomic_jsonl(output_root / "splits/group_fold_registry.jsonl", registry)
    atomic_csv(output_root / "splits/fold_support.csv", support)
    inner_hashes = {}
    for outer_fold in range(int(config["split"]["outer_folds"])):
        roles = assign_inner_group_roles(
            registry,
            outer_fold=outer_fold,
            calibration_fraction=float(config["split"]["inner_calibration_fraction"]),
            seed=int(config["seed"]),
        )
        path = output_root / f"splits/inner_roles_fold{outer_fold}.jsonl"
        atomic_jsonl(path, roles)
        inner_hashes[str(path.relative_to(output_root))] = file_sha256(path)
    tasks = _training_tasks(config)
    atomic_jsonl(output_root / "work/training_tasks.jsonl", tasks)
    fold_validation = "# Step-B fold validation\n\n"
    fold_validation += f"- UIDs: {len(registry):,}\n"
    fold_validation += f"- Image groups: {len({row['image_group_id'] for row in registry}):,}\n"
    fold_validation += "- UID overlap across held-out folds: 0\n- Image-group overlap across held-out folds: 0\n"
    fold_validation += "- Every fold has Stage-1 C/W, triggered UID, dense READ/WRITE signs, and routed READ/WRITE signs: true\n"
    _atomic_bytes(output_root / "splits/fold_validation.md", fold_validation.encode())
    _atomic_bytes(output_root / "protocol.md", _protocol(config, step_a_contract).encode())
    target_definition = """# Primary Stage-2 target definition

- READ: `u_read_w1 = q_FULL - q_WRITE_ONLY`.
- WRITE: `u_write_r1 = q_FULL - q_READ_ONLY`.
- Continuous Huber regression is primary; sign and correctness-flip metrics are evaluation-only.
- Exact zero is neutral and excluded from harmful-sign AUROC/AUPRC.
"""
    _atomic_bytes(output_root / "stage2_dense/targets/primary_target_definition.md", target_definition.encode())
    for name, payload in (
        ("nuisance", config["models"]["m0"]),
        ("linear", config["models"]["m1"]),
        ("mlp", config["models"]["m2"]),
        ("current_head", config["models"]["m3_stage1"]),
    ):
        atomic_json(output_root / f"stage1/model_configs/{name}.json", payload)

    source_hashes = {key: file_sha256(path) for key, path in config["sources"].items()}
    split_hashes = {
        "splits/group_fold_registry.jsonl": file_sha256(output_root / "splits/group_fold_registry.jsonl"),
        "splits/fold_support.csv": file_sha256(output_root / "splits/fold_support.csv"),
        "splits/fold_validation.md": file_sha256(output_root / "splits/fold_validation.md"),
        "work/training_tasks.jsonl": file_sha256(output_root / "work/training_tasks.jsonl"),
        **inner_hashes,
    }
    contract = {
        "schema_version": "predictability_stepB_id_learnability_contract_v1",
        "created_at": utc_now(),
        "static_config": config,
        "config_sha256": file_sha256(config_path),
        "parent_stepA_contract_sha256": step_a_contract["contract_sha256"],
        "parent_stepA_artifact_manifest_sha256": step_a_manifest["artifact_manifest_sha256"],
        "source_sha256": source_hashes,
        "bound_code_sha256": {path: file_sha256(path) for path in BOUND_CODE},
        "split_sha256": split_hashes,
        "git": git_state(),
        "runtime": runtime_state(),
        "review_reconciliation": {
            "verdict": "stable",
            "selected": "pure_pytorch_full_ladder_three_seeds",
            "m3_initialization": "from_scratch_inside_each_outer_fold",
            "specificity_limitation": "comparative_predictive_evidence_not_causality_or_unique_necessity",
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_contract.json", contract)
    print(json.dumps({"prepared": True, "contract_sha256": contract["contract_sha256"], "uids": len(registry)}, sort_keys=True))


def verify_contract(config_path: Path) -> tuple[dict[str, Any], Path, Path]:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    external_root = resolve_path(config["external_cache_root"])
    contract = read_json(output_root / "frozen_contract.json")
    if contract.get("contract_sha256") != canonical_hash(contract):
        raise RuntimeError("Step-B frozen contract hash differs")
    if contract.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError("Step-B config changed after freeze")
    for path, digest in contract["bound_code_sha256"].items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"Step-B bound code changed after freeze: {path}")
    for key, path in config["sources"].items():
        if file_sha256(path) != contract["source_sha256"][key]:
            raise RuntimeError(f"Step-B source changed after freeze: {key}")
    for relative, digest in contract["split_sha256"].items():
        if file_sha256(output_root / relative) != digest:
            raise RuntimeError(f"Step-B split changed after freeze: {relative}")
    if git_state() != contract["git"]:
        raise RuntimeError("Step-B git/worktree state differs from frozen contract")
    if runtime_state() != contract["runtime"]:
        raise RuntimeError("Step-B runtime differs from frozen contract")
    if not (output_root / "cache").is_symlink() or (output_root / "cache").resolve() != external_root:
        raise RuntimeError("Step-B external cache binding differs")
    return contract, output_root, external_root


def _write_bf16(memmap: np.memmap, index: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().to(torch.bfloat16).contiguous()
    memmap[index] = value.view(torch.uint16).numpy()


def _cache_manifest_valid(path: Path, *, contract_sha256: str) -> bool:
    if not path.is_file():
        return False
    manifest = read_json(path)
    if manifest.get("contract_sha256") != contract_sha256:
        return False
    for value in manifest.get("files", {}).values():
        file_path = resolve_path(value["path"])
        if not file_path.is_file() or file_path.stat().st_size != int(value["bytes"]):
            return False
        if file_sha256(file_path) != str(value["sha256"]):
            return False
    return True


def validate_caches(config_path: Path) -> None:
    contract, output_root, _ = verify_contract(config_path)
    bound = {}
    for name in ("stage1", "stage2_dense", "stage2_routed"):
        manifest_path = output_root / f"work/{name}_cache_manifest.json"
        if not _cache_manifest_valid(manifest_path, contract_sha256=contract["contract_sha256"]):
            raise RuntimeError(f"cache validation failed: {name}")
        manifest = read_json(manifest_path)
        files = {}
        for key, value in manifest["files"].items():
            path = resolve_path(value["path"])
            stat = path.stat()
            files[key] = {
                "path": str(path),
                "bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "inode": int(stat.st_ino),
                "sha256": str(value["sha256"]),
            }
        bound[name] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "files": files,
        }
    report = {
        "schema_version": "predictability_stepB_cache_validation_v1",
        "contract_sha256": contract["contract_sha256"],
        "passed": True,
        "validated_at": utc_now(),
        "caches": bound,
    }
    atomic_json(output_root / "validation/cache_validation.json", report)
    print(json.dumps({"cache_validation": "PASS", "caches": sorted(bound)}, sort_keys=True))


def _verify_validated_caches(contract: Mapping[str, Any], output_root: Path) -> None:
    report = read_json(output_root / "validation/cache_validation.json")
    if report.get("contract_sha256") != contract["contract_sha256"] or not report.get("passed"):
        raise RuntimeError("bound cache validation report differs")
    if set(report.get("caches", {})) != {"stage1", "stage2_dense", "stage2_routed"}:
        raise RuntimeError("bound cache validation population differs")
    for name, binding in report.get("caches", {}).items():
        manifest_path = resolve_path(binding["manifest_path"])
        if file_sha256(manifest_path) != binding["manifest_sha256"]:
            raise RuntimeError(f"cache manifest changed after validation: {name}")
        manifest = read_json(manifest_path)
        if manifest.get("contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError(f"cache contract changed after validation: {name}")
        for key, expected in binding["files"].items():
            path = resolve_path(expected["path"])
            stat = path.stat()
            observed = (int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ino))
            frozen = (int(expected["bytes"]), int(expected["mtime_ns"]), int(expected["inode"]))
            if observed != frozen or str(manifest["files"][key]["sha256"]) != str(expected["sha256"]):
                raise RuntimeError(f"cache binding changed after validation: {name}/{key}")


def rebind_validated_caches(config_path: Path, previous_contract_sha256: str) -> None:
    contract, output_root, _ = verify_contract(config_path)
    previous_hash = str(previous_contract_sha256)
    archive_root = output_root / "validation/contract_archive"
    previous = read_json(archive_root / f"{previous_hash}.json")
    validation = read_json(archive_root / f"{previous_hash}_cache_validation.json")
    if previous.get("contract_sha256") != previous_hash or canonical_hash(previous) != previous_hash:
        raise RuntimeError("archived previous contract differs")
    if validation.get("contract_sha256") != previous_hash or not validation.get("passed"):
        raise RuntimeError("archived cache validation differs")
    old_config = copy.deepcopy(previous["static_config"])
    new_config = copy.deepcopy(contract["static_config"])
    old_config.pop("execution_runtime", None)
    new_config.pop("execution_runtime", None)
    for smoke_config in (old_config.setdefault("smoke", {}), new_config.setdefault("smoke", {})):
        smoke_config.pop("checkpoint_prediction_atol", None)
        smoke_config.pop("checkpoint_prediction_float32_ulp_multiplier", None)
    if old_config != new_config:
        raise RuntimeError(
            "cache rebind permits only smoke-guard and cache-neutral execution-runtime changes"
        )
    for key in (
        "parent_stepA_contract_sha256", "parent_stepA_artifact_manifest_sha256",
        "source_sha256", "split_sha256",
    ):
        if previous[key] != contract[key]:
            raise RuntimeError(f"cache-relevant contract field changed: {key}")
    rebound = {}
    for name in ("stage1", "stage2_dense", "stage2_routed"):
        old_binding = validation["caches"][name]
        manifest_path = resolve_path(old_binding["manifest_path"])
        if file_sha256(manifest_path) != old_binding["manifest_sha256"]:
            raise RuntimeError(f"cache manifest changed before rebind: {name}")
        manifest = read_json(manifest_path)
        if manifest.get("contract_sha256") != previous_hash:
            raise RuntimeError(f"cache manifest previous contract differs: {name}")
        for file_binding in old_binding["files"].values():
            path = resolve_path(file_binding["path"])
            stat = path.stat()
            if (
                int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ino)
            ) != (
                int(file_binding["bytes"]), int(file_binding["mtime_ns"]), int(file_binding["inode"])
            ):
                raise RuntimeError(f"cache file metadata changed before rebind: {name}")
        manifest["contract_sha256"] = contract["contract_sha256"]
        manifest["provenance_rebind"] = {
            "from_contract_sha256": previous_hash,
            "reason": "smoke_checkpoint_roundtrip_guard_only_no_cache_derivation_change",
            "rebound_at": utc_now(),
        }
        atomic_json(manifest_path, manifest)
        rebound[name] = {
            **old_binding,
            "manifest_sha256": file_sha256(manifest_path),
        }
    report = {
        "schema_version": "predictability_stepB_cache_validation_v1",
        "contract_sha256": contract["contract_sha256"],
        "passed": True,
        "validated_at": utc_now(),
        "validation_mode": "metadata_preserving_rebind_after_prior_full_sha256_validation",
        "previous_contract_sha256": previous_hash,
        "caches": rebound,
    }
    atomic_json(output_root / "validation/cache_validation.json", report)
    _verify_validated_caches(contract, output_root)
    print(
        json.dumps(
            {
                "cache_rebind": "PASS",
                "from_contract_sha256": previous_hash,
                "to_contract_sha256": contract["contract_sha256"],
            },
            sort_keys=True,
        )
    )


def build_stage1_cache(config_path: Path) -> None:
    contract, output_root, external_root = verify_contract(config_path)
    manifest_path = output_root / "work/stage1_cache_manifest.json"
    if _cache_manifest_valid(manifest_path, contract_sha256=contract["contract_sha256"]):
        print(json.dumps({"cache": "stage1", "reused": True}, sort_keys=True))
        return
    rows = read_jsonl(contract["static_config"]["sources"]["stage1_features"])
    expected = int(contract["static_config"]["population"]["stage1_states"])
    width = int(contract["static_config"]["inputs"]["stage1_summary_width"])
    if len(rows) != expected or len({str(row["state_id"]) for row in rows}) != expected:
        raise RuntimeError("Stage-1 feature manifest differs")
    root = external_root / "stage1"
    root.mkdir(parents=True, exist_ok=True)
    final = root / "summary.bf16"
    partial = root / "summary.bf16.partial"
    with partial.open("wb") as handle:
        handle.truncate(expected * width * 2)
    values = np.memmap(partial, dtype=np.uint16, mode="r+", shape=(expected, width))
    by_shard: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for output_index, row in enumerate(rows):
        by_shard[str(row["feature_shard"])].append((output_index, row))
    started = time.monotonic()
    source_shard_sha256 = {}
    for shard_index, (shard_path, requested) in enumerate(sorted(by_shard.items())):
        resolved_shard = resolve_path(shard_path)
        source_shard_sha256[str(resolved_shard)] = file_sha256(resolved_shard)
        payload = torch.load(resolved_shard, map_location="cpu", weights_only=False)
        output_indices = [value[0] for value in requested]
        row_indices = torch.tensor([int(value[1]["feature_row_index"]) for value in requested])
        layers = torch.tensor([int(value[1]["feature_layer_index"]) for value in requested])
        pieces = [payload[name][row_indices, layers] for name in FEATURE_NAMES]
        _write_bf16(values, output_indices, torch.cat(pieces, dim=-1))
        if shard_index % 20 == 0:
            print(json.dumps({"cache": "stage1", "shards": shard_index + 1, "total_shards": len(by_shard), "elapsed_seconds": time.monotonic() - started}), flush=True)
    values.flush()
    del values
    os.replace(partial, final)
    index = root / "index.jsonl"
    atomic_jsonl(index, ({**row, "cache_row_index": index_value} for index_value, row in enumerate(rows)))
    manifest = {
        "schema_version": "predictability_stepB_fixed_cache_v1",
        "contract_sha256": contract["contract_sha256"],
        "domain": "stage1",
        "rows": expected,
        "width": width,
        "dtype": "bfloat16_uint16_storage",
        "files": {
            "summary": {"path": str(final), "bytes": final.stat().st_size, "sha256": file_sha256(final)},
            "index": {"path": str(index), "bytes": index.stat().st_size, "sha256": file_sha256(index)},
        },
        "source_shard_sha256": source_shard_sha256,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({"cache": "stage1", "rows": expected, "elapsed_seconds": time.monotonic() - started}, sort_keys=True))


def build_stage2_cache(config_path: Path, domain: str) -> None:
    if domain not in {"dense", "routed"}:
        raise ValueError("Stage-2 cache domain must be dense or routed")
    contract, output_root, external_root = verify_contract(config_path)
    manifest_path = output_root / f"work/stage2_{domain}_cache_manifest.json"
    if _cache_manifest_valid(manifest_path, contract_sha256=contract["contract_sha256"]):
        print(json.dumps({"cache": f"stage2_{domain}", "reused": True}, sort_keys=True))
        return
    source_key = f"stage2_{domain}_features"
    rows = read_jsonl(contract["static_config"]["sources"][source_key])
    expected = int(contract["static_config"]["population"][f"stage2_{domain}_states"])
    hidden = int(contract["static_config"]["inputs"]["stage2_summary_width_each"])
    if len(rows) != expected or len({str(row["state_id"]) for row in rows}) != expected:
        raise RuntimeError(f"Stage-2 {domain} feature manifest differs")
    text_total = sum(int(row["text_tokens"]) for row in rows)
    visual_total = sum(int(row["visual_tokens"]) for row in rows)
    root = external_root / f"stage2_{domain}"
    root.mkdir(parents=True, exist_ok=True)
    specs = {
        "summary": ((expected, 2 * hidden), np.uint16, 2),
        "text": ((text_total, hidden), np.uint16, 2),
        "visual": ((visual_total, hidden), np.uint16, 2),
        "text_mask": ((text_total,), np.uint8, 1),
        "visual_mask": ((visual_total,), np.uint8, 1),
    }
    maps = {}
    partials = {}
    finals = {}
    for name, (shape, dtype, bytes_each) in specs.items():
        final = root / f"{name}.{'bf16' if dtype == np.uint16 else 'u8'}"
        partial = Path(str(final) + ".partial")
        with partial.open("wb") as handle:
            handle.truncate(int(np.prod(shape)) * bytes_each)
        maps[name] = np.memmap(partial, dtype=dtype, mode="r+", shape=shape)
        partials[name] = partial
        finals[name] = final
    text_offsets = []
    visual_offsets = []
    text_cursor = visual_cursor = 0
    for row in rows:
        text_offsets.append(text_cursor)
        visual_offsets.append(visual_cursor)
        text_cursor += int(row["text_tokens"])
        visual_cursor += int(row["visual_tokens"])
    by_file: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for output_index, row in enumerate(rows):
        by_file[str(row["state_file"])].append((output_index, row))
    started = time.monotonic()
    for file_index, (state_path, requested) in enumerate(sorted(by_file.items())):
        payload = torch.load(
            resolve_stepa_artifact_path(contract["static_config"], state_path),
            map_location="cpu",
            weights_only=False,
        )
        states = payload["states"]
        for output_index, row in requested:
            state = states[str(row["state_id"])]
            observed_hashes = state_tensor_hashes(state)
            if observed_hashes != dict(row["tensor_sha256"]):
                raise RuntimeError(f"cached state tensor hash differs: {row['state_id']}")
            if combined_state_hash(observed_hashes) != str(row["state_sha256"]):
                raise RuntimeError(f"cached combined state hash differs: {row['state_id']}")
            text = state["text_states"].squeeze(0).to(torch.bfloat16).contiguous()
            visual = state["visual_states"].squeeze(0).to(torch.bfloat16).contiguous()
            text_mask = state["text_mask"].squeeze(0).bool().contiguous()
            visual_mask = state["visual_mask"].squeeze(0).bool().contiguous()
            nt, nv = len(text), len(visual)
            if nt != int(row["text_tokens"]) or nv != int(row["visual_tokens"]):
                raise RuntimeError(f"cached token count differs: {row['state_id']}")
            to, vo = text_offsets[output_index], visual_offsets[output_index]
            _write_bf16(maps["text"], slice(to, to + nt), text)
            _write_bf16(maps["visual"], slice(vo, vo + nv), visual)
            maps["text_mask"][to : to + nt] = text_mask.numpy().astype(np.uint8)
            maps["visual_mask"][vo : vo + nv] = visual_mask.numpy().astype(np.uint8)
            last = int(text_mask.long().sum()) - 1
            pooled_visual = visual[visual_mask].float().mean(dim=0).to(torch.bfloat16)
            _write_bf16(maps["summary"], output_index, torch.cat((text[last], pooled_visual)))
        if file_index % 25 == 0:
            print(json.dumps({"cache": f"stage2_{domain}", "files": file_index + 1, "total_files": len(by_file), "elapsed_seconds": time.monotonic() - started}), flush=True)
    for value in maps.values():
        value.flush()
    del maps
    for name in specs:
        os.replace(partials[name], finals[name])
    index_rows = []
    for index_value, row in enumerate(rows):
        index_rows.append(
            {
                **row,
                "cache_row_index": index_value,
                "text_offset": text_offsets[index_value],
                "visual_offset": visual_offsets[index_value],
            }
        )
    index = root / "index.jsonl"
    atomic_jsonl(index, index_rows)
    files = {}
    for name, path in {**finals, "index": index}.items():
        files[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
    manifest = {
        "schema_version": "predictability_stepB_packed_state_cache_v1",
        "contract_sha256": contract["contract_sha256"],
        "domain": f"stage2_{domain}",
        "rows": expected,
        "hidden_size": hidden,
        "text_tokens": text_total,
        "visual_tokens": visual_total,
        "dtype": "bfloat16_uint16_storage",
        "files": files,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({"cache": f"stage2_{domain}", "rows": expected, "elapsed_seconds": time.monotonic() - started}, sort_keys=True))


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    torch.save(value, temporary)
    os.replace(temporary, path)


def configure_determinism(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def configure_worker_runtime(config: Mapping[str, Any]) -> None:
    execution = config["execution_runtime"]
    if int(execution["packed_batch_prefetch_depth"]) != 1:
        raise ValueError("Step-B supports exactly one semantics-preserving packed prefetch batch")
    if not bool(execution["pinned_host_batches"]):
        raise ValueError("Step-B packed prefetch requires pinned host batches")
    torch.set_num_threads(int(execution["torch_intraop_threads_per_worker"]))
    torch.set_num_interop_threads(int(execution["torch_interop_threads_per_worker"]))


def _u16_to_bf16(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(value, copy=True)).view(torch.bfloat16)


class FixedBfloat16Store:
    def __init__(self, path: str | Path, *, rows: int, width: int) -> None:
        self.path = resolve_path(path)
        self.rows = int(rows)
        self.width = int(width)
        if self.path.stat().st_size != self.rows * self.width * 2:
            raise RuntimeError(f"fixed BF16 cache size differs: {self.path}")
        self.values = np.memmap(
            self.path, dtype=np.uint16, mode="r", shape=(self.rows, self.width)
        )

    def get(self, indices: Sequence[int] | np.ndarray, input_name: str) -> torch.Tensor:
        selected = _u16_to_bf16(self.values[np.asarray(indices, dtype=np.int64)])
        if self.width == 7168:
            if input_name == "text":
                return selected[:, :3584]
            if input_name == "visual":
                return selected[:, 3584:]
            if input_name != "text_visual":
                raise ValueError(f"unsupported Stage-2 summary input: {input_name}")
        elif input_name not in {"state", "state_layer"}:
            raise ValueError(f"unsupported Stage-1 summary input: {input_name}")
        return selected

    def input_width(self, input_name: str) -> int:
        if self.width == 7168 and input_name in {"text", "visual"}:
            return 3584
        return self.width


class TensorStore:
    def __init__(self, values: torch.Tensor) -> None:
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("tensor store requires a nonempty matrix")
        self.values = values.cpu().float().contiguous()

    def get(self, indices: Sequence[int] | np.ndarray, input_name: str) -> torch.Tensor:
        del input_name
        return self.values.index_select(0, torch.as_tensor(indices, dtype=torch.long))

    def input_width(self, input_name: str) -> int:
        del input_name
        return int(self.values.shape[1])


class PackedStateStore:
    def __init__(self, cache_manifest: Mapping[str, Any], index_rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = [dict(row) for row in index_rows]
        hidden = int(cache_manifest["hidden_size"])
        self.hidden_size = hidden
        files = cache_manifest["files"]
        self.text = np.memmap(
            resolve_path(files["text"]["path"]),
            dtype=np.uint16,
            mode="r",
            shape=(int(cache_manifest["text_tokens"]), hidden),
        )
        self.visual = np.memmap(
            resolve_path(files["visual"]["path"]),
            dtype=np.uint16,
            mode="r",
            shape=(int(cache_manifest["visual_tokens"]), hidden),
        )
        self.text_mask = np.memmap(
            resolve_path(files["text_mask"]["path"]),
            dtype=np.uint8,
            mode="r",
            shape=(int(cache_manifest["text_tokens"]),),
        )
        self.visual_mask = np.memmap(
            resolve_path(files["visual_mask"]["path"]),
            dtype=np.uint8,
            mode="r",
            shape=(int(cache_manifest["visual_tokens"]),),
        )

    def collate(
        self, indices: Sequence[int] | np.ndarray, *, pin_memory: bool = False
    ) -> tuple[torch.Tensor, ...]:
        selected = [self.rows[int(index)] for index in indices]
        max_text = max(int(row["text_tokens"]) for row in selected)
        max_visual = max(int(row["visual_tokens"]) for row in selected)
        text = torch.zeros(
            (len(selected), max_text, self.hidden_size),
            dtype=torch.bfloat16,
            pin_memory=pin_memory,
        )
        visual = torch.zeros(
            (len(selected), max_visual, self.hidden_size),
            dtype=torch.bfloat16,
            pin_memory=pin_memory,
        )
        text_mask = torch.zeros(
            (len(selected), max_text), dtype=torch.bool, pin_memory=pin_memory
        )
        visual_mask = torch.zeros(
            (len(selected), max_visual), dtype=torch.bool, pin_memory=pin_memory
        )
        for output_index, row in enumerate(selected):
            nt, nv = int(row["text_tokens"]), int(row["visual_tokens"])
            to, vo = int(row["text_offset"]), int(row["visual_offset"])
            text[output_index, :nt] = _u16_to_bf16(self.text[to : to + nt])
            visual[output_index, :nv] = _u16_to_bf16(self.visual[vo : vo + nv])
            text_mask[output_index, :nt] = torch.from_numpy(
                np.array(self.text_mask[to : to + nt], copy=True).astype(bool)
            )
            visual_mask[output_index, :nv] = torch.from_numpy(
                np.array(self.visual_mask[vo : vo + nv], copy=True).astype(bool)
            )
        return text, visual, text_mask, visual_mask


def iter_prefetched_packed_batches(
    store: PackedStateStore,
    batches: Iterable[np.ndarray],
    *,
    pin_memory: bool,
) -> Iterable[tuple[np.ndarray, tuple[torch.Tensor, ...]]]:
    """Collate one packed batch ahead without changing batch order or contents."""

    iterator = iter(batches)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="stepb-packed") as executor:
        try:
            first = np.asarray(next(iterator), dtype=np.int64)
        except StopIteration:
            return
        future: Future[tuple[torch.Tensor, ...]] = executor.submit(
            store.collate, first, pin_memory=pin_memory
        )
        current = first
        while True:
            packed = future.result()
            try:
                following = np.asarray(next(iterator), dtype=np.int64)
            except StopIteration:
                yield current, packed
                break
            future = executor.submit(store.collate, following, pin_memory=pin_memory)
            yield current, packed
            current = following


def _domain_data(
    contract: Mapping[str, Any], output_root: Path, domain: str
) -> dict[str, Any]:
    config = contract["static_config"]
    registry = read_jsonl(output_root / "splits/group_fold_registry.jsonl")
    fold_by_uid = {str(row["uid"]): int(row["fold"]) for row in registry}
    internal = read_jsonl(config["sources"]["internal_samples"])
    base = {
        str(row["uid"]): {
            "visual_token_count": int(row["dense"]["visual_token_count"]),
            "user_text_token_count": int(row["dense"]["user_text_token_count"]),
            "prompt_token_count": int(row["dense"]["prompt_token_count"]),
        }
        for row in internal
    }
    if domain == "stage1":
        cache_manifest = read_json(output_root / "work/stage1_cache_manifest.json")
    elif domain in {"stage2_dense", "stage2_routed"}:
        suffix = domain.removeprefix("stage2_")
        cache_manifest = read_json(output_root / f"work/stage2_{suffix}_cache_manifest.json")
    else:
        raise ValueError(f"unknown Step-B domain: {domain}")
    if cache_manifest.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError(f"{domain} cache contract differs")
    index_rows = read_jsonl(cache_manifest["files"]["index"]["path"])
    if len(index_rows) != int(cache_manifest["rows"]):
        raise RuntimeError(f"{domain} cache index size differs")
    state_ids = [str(row["state_id"]) for row in index_rows]
    if len(set(state_ids)) != len(state_ids):
        raise RuntimeError(f"{domain} cache index has duplicate state IDs")
    for row in index_rows:
        uid = str(row["uid"])
        row["fold"] = fold_by_uid[uid]
        row.update(base[uid])
    if domain == "stage1":
        targets = np.asarray([int(bool(row["dense_wrong"])) for row in index_rows], dtype=np.float64)
        target_map = {"dense_wrong": targets}
        fixed = FixedBfloat16Store(
            cache_manifest["files"]["summary"]["path"],
            rows=len(index_rows),
            width=int(cache_manifest["width"]),
        )
        packed = None
    else:
        suffix = domain.removeprefix("stage2_")
        utilities = {
            str(row["state_id"]): row
            for row in read_csv(config["sources"][f"stage2_{suffix}_utilities"])
        }
        target_columns = {
            "read": "u_read_w1",
            "write": "u_write_r1",
            "u_read_w0": "u_read_w0",
            "u_write_r0": "u_write_r0",
            "u_read": "u_read",
            "u_write": "u_write",
            "u_interaction": "u_interaction",
        }
        target_map = {
            name: np.asarray([float(utilities[state_id][column]) for state_id in state_ids])
            for name, column in target_columns.items()
            if domain == "stage2_dense" or name in {"read", "write"}
        }
        fixed = FixedBfloat16Store(
            cache_manifest["files"]["summary"]["path"],
            rows=len(index_rows),
            width=2 * int(cache_manifest["hidden_size"]),
        )
        packed = PackedStateStore(cache_manifest, index_rows)
    return {
        "rows": index_rows,
        "state_ids": state_ids,
        "targets": target_map,
        "fixed": fixed,
        "packed": packed,
    }


def _nuisance_features(rows: Sequence[Mapping[str, Any]], input_name: str) -> torch.Tensor:
    datasets = ("gqa", "chartqa", "textvqa")
    sources = ("historical", "canonical")
    output = []
    for row in rows:
        dense_only = input_name == "dense_outcome"
        pieces: list[float] = []
        if not dense_only:
            pieces.extend(float(str(row["dataset"]) == value) for value in datasets)
            pieces.extend(float(str(row["source_regime"]) == value) for value in sources)
            pieces.extend(float(int(row["layer"]) == value) for value in range(28))
            if "trigger_layer" in row:
                pieces.extend(float(int(row["trigger_layer"]) == value) for value in range(28))
                relative = min(int(row["trigger_relative_depth"]), 27)
                pieces.extend(float(relative == value) for value in range(28))
                pieces.extend(
                    (
                        np.log1p(float(row["visual_tokens"])),
                        np.log1p(float(row["text_tokens"])),
                        np.log1p(float(row["user_text_token_count"])),
                        np.log1p(float(row["prompt_token_count"])),
                    )
                )
            else:
                pieces.extend(
                    (
                        np.log1p(float(row["visual_token_count"])),
                        np.log1p(float(row["user_text_token_count"])),
                        np.log1p(float(row["prompt_token_count"])),
                    )
                )
        if input_name in {"dense_outcome", "nuisance_dense_outcome"}:
            pieces.append(float(bool(row["dense_wrong"])))
        output.append(pieces)
    return torch.tensor(output, dtype=torch.float32)


def _role_indices(output_root: Path, rows: Sequence[Mapping[str, Any]], fold: int) -> dict[str, np.ndarray]:
    roles = {
        str(row["uid"]): str(row["role"])
        for row in read_jsonl(output_root / f"splits/inner_roles_fold{int(fold)}.jsonl")
    }
    result = {}
    for role in ("fit", "calibration", "outer_test"):
        result[role] = np.asarray(
            [index for index, row in enumerate(rows) if roles[str(row["uid"])] == role],
            dtype=np.int64,
        )
        if len(result[role]) == 0:
            raise RuntimeError(f"fold {fold} has no {role} states")
    return result


def _subset_uid_weights(rows: Sequence[Mapping[str, Any]], indices: np.ndarray) -> np.ndarray:
    counts = Counter(str(rows[int(index)]["uid"]) for index in indices)
    weights = np.asarray([1.0 / counts[str(rows[int(index)]["uid"])] for index in indices])
    return weights / weights.mean()


def _stream_standardizer(
    store: FixedBfloat16Store | TensorStore,
    input_name: str,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = store.input_width(input_name)
    total = torch.zeros(width, dtype=torch.float64)
    square = torch.zeros(width, dtype=torch.float64)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = store.get(batch_indices, input_name).to(device=device, dtype=torch.float32)
        total += batch.sum(dim=0).double().cpu()
        square += batch.square().sum(dim=0).double().cpu()
    mean = total / len(indices)
    variance = torch.clamp(square / len(indices) - mean.square(), min=0.0)
    std = variance.sqrt()
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean.float(), std.float()


def _model_for_task(task: Mapping[str, Any], input_width: int, config: Mapping[str, Any]) -> torch.nn.Module:
    model_name = str(task["model"])
    if model_name.startswith("m0_") or model_name.startswith("m1_"):
        return SummaryScalarPredictor(kind="linear", input_size=input_width, hidden_size=1, dropout=0.0)
    if model_name == "m2_mlp" or model_name == "m2_text_visual":
        spec = config["models"]["m2"]
        return SummaryScalarPredictor(
            kind="mlp",
            input_size=input_width,
            hidden_size=int(spec["hidden_size"]),
            dropout=float(spec["dropout"]),
        )
    if model_name == "m3_current_head":
        spec = config["models"]["m3_stage1"]
        return SharedFailurePredictor(
            variant="state_layer_all28",
            input_size=input_width,
            projection_size=int(spec["projection_size"]),
            layer_embedding_size=int(spec["layer_embedding_size"]),
            hidden_size=int(spec["hidden_size"]),
        )
    if model_name.startswith("m3_z_"):
        spec = config["models"]["m3_stage2"]
        return RouterStyleUtilityRegressor(
            hidden_size=int(config["inputs"]["stage2_summary_width_each"]),
            router_size=int(spec["router_size"]),
            num_heads=int(spec["num_heads"]),
            dropout=float(spec["dropout"]),
            readout_hidden_size=int(spec["readout_hidden_size"]),
            representation=str(task["input"]),
        )
    raise ValueError(f"unknown Step-B model: {model_name}")


def _training_spec(task: Mapping[str, Any], config: Mapping[str, Any]) -> Mapping[str, Any]:
    model = str(task["model"])
    if model.startswith("m0_") or model.startswith("m1_"):
        return config["training"]["linear"]
    if model.startswith("m2_"):
        return config["training"]["mlp"]
    if model == "m3_current_head":
        return config["training"]["stage1_m3"]
    return config["training"]["stage2_m3"]


def _forward_task(
    model: torch.nn.Module,
    task: Mapping[str, Any],
    data: Mapping[str, Any],
    indices: np.ndarray,
    *,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
    nuisance_store: TensorStore | None,
    device: torch.device,
    packed_batch: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    model_name = str(task["model"])
    if model_name.startswith("m3_z_"):
        text, visual, text_mask, visual_mask = (
            data["packed"].collate(indices) if packed_batch is None else packed_batch
        )
        non_blocking = bool(text.is_pinned())
        return model(
            text.to(device, non_blocking=non_blocking),
            visual.to(device, non_blocking=non_blocking),
            text_mask=text_mask.to(device, non_blocking=non_blocking),
            visual_mask=visual_mask.to(device, non_blocking=non_blocking),
        )
    store = nuisance_store if nuisance_store is not None else data["fixed"]
    features = store.get(indices, str(task["input"])).to(device=device, dtype=torch.float32)
    if mean is None or std is None:
        raise RuntimeError("summary/nuisance model lacks fold-local standardization")
    features = (features - mean.to(device)) / std.to(device)
    if model_name == "m3_current_head":
        layers = torch.tensor(
            [int(data["rows"][int(index)]["layer"]) for index in indices],
            dtype=torch.long,
            device=device,
        )
        return model(features, layers)
    return model(features)


def _predict_indices(
    model: torch.nn.Module,
    task: Mapping[str, Any],
    data: Mapping[str, Any],
    indices: np.ndarray,
    *,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
    nuisance_store: TensorStore | None,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs = []
    batches = [
        indices[start : start + batch_size] for start in range(0, len(indices), batch_size)
    ]
    if str(task["model"]).startswith("m3_z_"):
        batch_iterator: Iterable[tuple[np.ndarray, tuple[torch.Tensor, ...] | None]] = (
            (batch, packed)
            for batch, packed in iter_prefetched_packed_batches(
                data["packed"], batches, pin_memory=True
            )
        )
    else:
        batch_iterator = ((batch, None) for batch in batches)
    with torch.inference_mode():
        for batch, packed in batch_iterator:
            outputs.append(
                _forward_task(
                    model, task, data, batch, mean=mean, std=std,
                    nuisance_store=nuisance_store, device=device, packed_batch=packed,
                ).float().cpu()
            )
    return torch.cat(outputs).numpy().astype(np.float64)


def train_task(
    contract: Mapping[str, Any],
    output_root: Path,
    external_root: Path,
    task: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    config = contract["static_config"]
    configure_determinism(int(task["seed"]))
    roles = _role_indices(output_root, data["rows"], int(task["fold"]))
    target = np.asarray(data["targets"][str(task["target"])], dtype=np.float64)
    classification = str(task["domain"]) == "stage1"
    model_name = str(task["model"])
    nuisance_store = None
    if model_name.startswith("m0_"):
        nuisance_store = TensorStore(_nuisance_features(data["rows"], str(task["input"])))
        input_width = nuisance_store.input_width(str(task["input"]))
        mean, std = _stream_standardizer(
            nuisance_store, str(task["input"]), roles["fit"], device=device
        )
    elif model_name.startswith("m3_z_"):
        input_width = int(config["inputs"]["stage2_summary_width_each"])
        mean = std = None
    else:
        input_width = data["fixed"].input_width(str(task["input"]))
        mean, std = _stream_standardizer(
            data["fixed"], str(task["input"]), roles["fit"], device=device
        )
    if mean is not None:
        mean = mean.to(device)
        std = std.to(device)
    scale = None if classification else robust_target_scale(
        target[roles["fit"]], floor=float(config["targets"]["target_scale_floor"])
    )
    training_target = target if classification else scale.transform(target)
    model = _model_for_task(task, input_width, config).to(device=device, dtype=torch.float32)
    spec = _training_spec(task, config)
    batch_size = int(spec.get("batch_size", spec.get("state_microbatch")))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    fit_weights = _subset_uid_weights(data["rows"], roles["fit"])
    calibration_weights = _subset_uid_weights(data["rows"], roles["calibration"])
    generator = torch.Generator(device="cpu").manual_seed(int(task["seed"]))
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    history = []
    started = time.monotonic()
    for epoch in range(int(spec["maximum_epochs"])):
        model.train()
        order = torch.randperm(len(roles["fit"]), generator=generator).numpy()
        epoch_sum = 0.0
        epoch_weight = 0.0
        epoch_local_batches = [
            order[start : start + batch_size]
            for start in range(0, len(order), batch_size)
        ]
        epoch_batches = [roles["fit"][local] for local in epoch_local_batches]
        if model_name.startswith("m3_z_"):
            batch_iterator: Iterable[tuple[np.ndarray, tuple[torch.Tensor, ...] | None]] = (
                (batch, packed)
                for batch, packed in iter_prefetched_packed_batches(
                    data["packed"], epoch_batches, pin_memory=True
                )
            )
        else:
            batch_iterator = ((batch, None) for batch in epoch_batches)
        for batch_number, (batch_indices, packed) in enumerate(batch_iterator):
            local = epoch_local_batches[batch_number]
            batch_weights = torch.tensor(fit_weights[local], dtype=torch.float32, device=device)
            batch_target = torch.tensor(
                training_target[batch_indices], dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward_task(
                model, task, data, batch_indices, mean=mean, std=std,
                nuisance_store=nuisance_store, device=device, packed_batch=packed,
            )
            losses = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    prediction, batch_target, reduction="none"
                )
                if classification
                else torch.nn.functional.huber_loss(
                    prediction, batch_target, delta=1.0, reduction="none"
                )
            )
            loss = (losses * batch_weights).sum() / batch_weights.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            epoch_sum += float((losses.detach() * batch_weights).sum().cpu())
            epoch_weight += float(batch_weights.sum().cpu())
        calibration_raw = _predict_indices(
            model, task, data, roles["calibration"], mean=mean, std=std,
            nuisance_store=nuisance_store, device=device, batch_size=batch_size,
        )
        calibration_target = training_target[roles["calibration"]]
        raw_tensor = torch.tensor(calibration_raw, dtype=torch.float64)
        target_tensor = torch.tensor(calibration_target, dtype=torch.float64)
        calibration_losses = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                raw_tensor, target_tensor, reduction="none"
            ).numpy()
            if classification
            else torch.nn.functional.huber_loss(
                raw_tensor, target_tensor, delta=1.0, reduction="none"
            ).numpy()
        )
        calibration_loss = float(
            np.dot(calibration_losses, calibration_weights) / calibration_weights.sum()
        )
        history.append(
            {
                "epoch": epoch + 1,
                "fit_loss": epoch_sum / epoch_weight,
                "calibration_loss": calibration_loss,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch + 1 >= int(spec["minimum_epochs"]) and stale >= int(spec["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"training produced no checkpoint: {task['task_id']}")
    model.load_state_dict(best_state)
    calibration_raw = _predict_indices(
        model, task, data, roles["calibration"], mean=mean, std=std,
        nuisance_store=nuisance_store, device=device, batch_size=batch_size,
    )
    test_raw = _predict_indices(
        model, task, data, roles["outer_test"], mean=mean, std=std,
        nuisance_store=nuisance_store, device=device, batch_size=batch_size,
    )
    if classification:
        calibration_prediction = 1.0 / (1.0 + np.exp(-np.clip(calibration_raw, -50, 50)))
        test_prediction = 1.0 / (1.0 + np.exp(-np.clip(test_raw, -50, 50)))
    else:
        calibration_prediction = scale.inverse(calibration_raw)
        test_prediction = scale.inverse(test_raw)
    payload = {
        "schema_version": "predictability_stepB_training_result_v1",
        "contract_sha256": contract["contract_sha256"],
        "task": dict(task),
        "task_sha256": canonical_hash(task),
        "model_state": best_state,
        "normalization_mean": None if mean is None else mean.detach().cpu(),
        "normalization_std": None if std is None else std.detach().cpu(),
        "target_center": None if scale is None else scale.center,
        "target_scale": None if scale is None else scale.scale,
        "best_epoch": best_epoch,
        "best_calibration_loss": best_loss,
        "history": history,
        "calibration_indices": roles["calibration"],
        "calibration_prediction": calibration_prediction,
        "test_indices": roles["outer_test"],
        "test_prediction": test_prediction,
        "elapsed_seconds": time.monotonic() - started,
    }
    family = (
        "nuisance" if model_name.startswith("m0_") else
        "linear" if model_name.startswith("m1_") else
        "mlp" if model_name.startswith("m2_") else "router_style"
    )
    checkpoint = external_root / "models" / str(task["domain"]) / family / f"{task['task_id']}.pt"
    atomic_torch(checkpoint, payload)
    return {
        "schema_version": "predictability_stepB_task_completion_v1",
        "contract_sha256": contract["contract_sha256"],
        "task_id": task["task_id"],
        "task_sha256": canonical_hash(task),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "best_epoch": best_epoch,
        "best_calibration_loss": best_loss,
        "fit_states": len(roles["fit"]),
        "calibration_states": len(roles["calibration"]),
        "test_states": len(roles["outer_test"]),
        "elapsed_seconds": payload["elapsed_seconds"],
    }


def train_worker(config_path: Path, rank: int, *, resume: bool) -> None:
    contract, output_root, external_root = verify_contract(config_path)
    config = contract["static_config"]
    if not 0 <= int(rank) < int(config["world_size"]):
        raise ValueError("invalid Step-B worker rank")
    smoke_report = read_json(output_root / "validation/smoke_report.json")
    if (
        smoke_report.get("contract_sha256") != contract["contract_sha256"]
        or not smoke_report.get("passed")
    ):
        raise RuntimeError("Step-B full training requires a passing bound smoke report")
    _verify_validated_caches(contract, output_root)
    configure_worker_runtime(config)
    torch.cuda.set_device(int(rank))
    device = torch.device(f"cuda:{int(rank)}")
    tasks = [
        row for row in read_jsonl(output_root / "work/training_tasks.jsonl")
        if int(row["worker_rank"]) == int(rank)
    ]
    rank_root = output_root / f"work/training/rank{int(rank):02d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    data_cache: dict[str, dict[str, Any]] = {}
    completed = 0
    started = time.monotonic()
    for task in tasks:
        result_path = rank_root / f"{task['task_id']}.json"
        if resume and result_path.is_file():
            old = read_json(result_path)
            checkpoint = resolve_path(old.get("checkpoint", ""))
            if (
                old.get("contract_sha256") == contract["contract_sha256"]
                and old.get("task_sha256") == canonical_hash(task)
                and checkpoint.is_file()
                and file_sha256(checkpoint) == old.get("checkpoint_sha256")
            ):
                completed += 1
                continue
        domain = str(task["domain"])
        if domain not in data_cache:
            data_cache[domain] = _domain_data(contract, output_root, domain)
        result = train_task(
            contract, output_root, external_root, task, data_cache[domain], device=device
        )
        atomic_json(result_path, result)
        completed += 1
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "rank": int(rank), "completed": completed, "assigned": len(tasks),
                    "task_id": task["task_id"], "elapsed_seconds": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    atomic_json(
        rank_root / "complete.json",
        {
            "schema_version": "predictability_stepB_worker_complete_v1",
            "contract_sha256": contract["contract_sha256"],
            "rank": int(rank),
            "expected_tasks": len(tasks),
            "completed_tasks": completed,
            "elapsed_seconds": time.monotonic() - started,
            "completed_at": utc_now(),
        },
    )


class RemappedStore:
    """Expose a deterministic local view without copying a large cache."""

    def __init__(self, base: Any, indices: np.ndarray) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def get(self, indices: Sequence[int] | np.ndarray, input_name: str) -> torch.Tensor:
        local = np.asarray(indices, dtype=np.int64)
        return self.base.get(self.indices[local], input_name)

    def input_width(self, input_name: str) -> int:
        return int(self.base.input_width(input_name))


class RemappedPackedStore:
    def __init__(self, base: PackedStateStore, indices: np.ndarray) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def collate(
        self, indices: Sequence[int] | np.ndarray, *, pin_memory: bool = False
    ) -> tuple[torch.Tensor, ...]:
        local = np.asarray(indices, dtype=np.int64)
        return self.base.collate(self.indices[local], pin_memory=pin_memory)


def _smoke_subset(
    output_root: Path,
    data: Mapping[str, Any],
    *,
    fold: int,
    total: int,
    minimum_groups_per_role: int,
) -> dict[str, Any]:
    role_by_uid = {
        str(row["uid"]): str(row["role"])
        for row in read_jsonl(output_root / f"splits/inner_roles_fold{int(fold)}.jsonl")
    }
    quotas = {
        "fit": int(total) // 2,
        "calibration": int(total) // 4,
        "outer_test": int(total) - int(total) // 2 - int(total) // 4,
    }
    selected: list[int] = []
    for role, quota in quotas.items():
        candidates = [
            index for index, row in enumerate(data["rows"])
            if role_by_uid[str(row["uid"])] == role
        ]
        candidates.sort(
            key=lambda index: sha256(
                f"smoke|{role}|{data['rows'][index]['state_id']}".encode()
            ).hexdigest()
        )
        chosen: list[int] = []
        groups: set[str] = set()
        deferred: list[int] = []
        for index in candidates:
            group = str(data["rows"][index]["image_group_id"])
            if group not in groups and len(groups) < int(minimum_groups_per_role):
                chosen.append(index)
                groups.add(group)
            else:
                deferred.append(index)
            if len(chosen) == quota:
                break
        if len(groups) < int(minimum_groups_per_role):
            raise RuntimeError(f"smoke {role} lacks the required image groups")
        if len(chosen) < quota:
            chosen.extend(deferred[: quota - len(chosen)])
        if len(chosen) != quota:
            raise RuntimeError(f"smoke {role} lacks the required states")
        selected.extend(chosen)
    selected_array = np.asarray(selected, dtype=np.int64)
    rows = [dict(data["rows"][int(index)]) for index in selected_array]
    role_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        role_groups[role_by_uid[str(row["uid"])]].add(str(row["image_group_id"]))
    role_names = tuple(role_groups)
    for left_index, left in enumerate(role_names):
        for right in role_names[left_index + 1 :]:
            if role_groups[left].intersection(role_groups[right]):
                raise RuntimeError("smoke subset leaks an image group across roles")
    return {
        "rows": rows,
        "state_ids": [str(row["state_id"]) for row in rows],
        "targets": {
            key: np.asarray(values, dtype=np.float64)[selected_array]
            for key, values in data["targets"].items()
        },
        "fixed": RemappedStore(data["fixed"], selected_array),
        "packed": (
            None if data["packed"] is None else RemappedPackedStore(data["packed"], selected_array)
        ),
    }


def _smoke_tasks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    base_seed = int(config["seed"])
    nonlinear_seed = int(config["training"]["seeds"][0])
    tasks: list[dict[str, Any]] = []

    def add(domain: str, model: str, input_name: str, target: str, seed: int) -> None:
        tasks.append(
            {
                "task_id": f"smoke__{domain}__{target}__{model}__{input_name}__s{seed}",
                "domain": domain,
                "fold": 0,
                "model": model,
                "input": input_name,
                "target": target,
                "seed": seed,
                "estimated_cost": 1,
            }
        )

    for model, input_name, seed in (
        ("m0_nuisance", "nuisance", base_seed),
        ("m1_linear", "state", base_seed),
        ("m2_mlp", "state", nonlinear_seed),
        ("m3_current_head", "state_layer", nonlinear_seed),
    ):
        add("stage1", model, input_name, "dense_wrong", seed)
    for domain in ("stage2_dense", "stage2_routed"):
        for model, input_name, seed in (
            ("m0_nuisance", "nuisance", base_seed),
            ("m1_text", "text", base_seed),
            ("m1_visual", "visual", base_seed),
            ("m1_text_visual", "text_visual", base_seed),
            ("m2_text_visual", "text_visual", nonlinear_seed),
            ("m3_z_R", "z_R", nonlinear_seed),
            ("m3_z_W", "z_W", nonlinear_seed),
            ("m3_z_RW", "z_RW", nonlinear_seed),
        ):
            add(domain, model, input_name, "read", seed)
    for index, task in enumerate(tasks):
        task["worker_rank"] = index % int(config["world_size"])
    return tasks


def _verify_smoke_checkpoint(
    checkpoint: Path,
    task: Mapping[str, Any],
    data: Mapping[str, Any],
    config: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_name = str(task["model"])
    nuisance_store = None
    if model_name.startswith("m0_"):
        nuisance_store = TensorStore(_nuisance_features(data["rows"], str(task["input"])))
        input_width = nuisance_store.input_width(str(task["input"]))
    elif model_name.startswith("m3_z_"):
        input_width = int(config["inputs"]["stage2_summary_width_each"])
    else:
        input_width = data["fixed"].input_width(str(task["input"]))
    model = _model_for_task(task, input_width, config).to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["model_state"])
    roles = _role_indices(output_root, data["rows"], int(task["fold"]))
    kwargs = {
        "mean": (
            None if payload["normalization_mean"] is None else payload["normalization_mean"].to(device)
        ),
        "std": (
            None if payload["normalization_std"] is None else payload["normalization_std"].to(device)
        ),
        "nuisance_store": nuisance_store,
        "device": device,
        "batch_size": 16 if model_name.startswith("m3_z_") else 128,
    }
    first = _predict_indices(model, task, data, roles["outer_test"], **kwargs)
    second = _predict_indices(model, task, data, roles["outer_test"], **kwargs)
    if not np.array_equal(first, second):
        raise RuntimeError(f"smoke repeat prediction differs: {task['task_id']}")
    expected = np.asarray(payload["test_prediction"], dtype=np.float64)
    if str(task["domain"]) == "stage1":
        first = 1.0 / (1.0 + np.exp(-np.clip(first, -50, 50)))
    else:
        first = first * float(payload["target_scale"]) + float(payload["target_center"])
    try:
        roundtrip = validate_prediction_roundtrip(
            expected,
            first,
            float32_ulp_multiplier=float(
                config["smoke"]["checkpoint_prediction_float32_ulp_multiplier"]
            ),
        )
    except ValueError as error:
        raise RuntimeError(f"smoke checkpoint roundtrip differs: {task['task_id']}") from error
    return {
        "task_id": str(task["task_id"]),
        "domain": str(task["domain"]),
        "model": model_name,
        "input": str(task["input"]),
        "test_states": int(len(first)),
        "exact_repeat": True,
        "checkpoint_roundtrip": True,
        "checkpoint_roundtrip_max_abs": roundtrip["maximum_absolute_difference"],
        "checkpoint_roundtrip_max_scaled_float32_ulps": roundtrip[
            "maximum_scaled_float32_ulps"
        ],
    }


def smoke_worker(config_path: Path, rank: int) -> None:
    contract, output_root, external_root = verify_contract(config_path)
    config = contract["static_config"]
    if not 0 <= int(rank) < int(config["world_size"]):
        raise ValueError("invalid smoke worker rank")
    _verify_validated_caches(contract, output_root)
    configure_worker_runtime(config)
    torch.cuda.set_device(int(rank))
    device = torch.device(f"cuda:{int(rank)}")
    smoke_contract = copy.deepcopy(contract)
    for name in ("linear", "mlp", "stage1_m3", "stage2_m3"):
        smoke_contract["static_config"]["training"][name]["minimum_epochs"] = 1
        smoke_contract["static_config"]["training"][name]["maximum_epochs"] = 1
    tasks = [task for task in _smoke_tasks(config) if int(task["worker_rank"]) == int(rank)]
    data_by_domain: dict[str, dict[str, Any]] = {}
    rows = []
    for task in tasks:
        domain = str(task["domain"])
        if domain not in data_by_domain:
            full = _domain_data(contract, output_root, domain)
            total_key = {
                "stage1": "stage1_states",
                "stage2_dense": "stage2_dense_states",
                "stage2_routed": "stage2_routed_states",
            }[domain]
            data_by_domain[domain] = _smoke_subset(
                output_root,
                full,
                fold=0,
                total=int(config["smoke"][total_key]),
                minimum_groups_per_role=int(config["smoke"]["groups_per_fold"]),
            )
        result = train_task(
            smoke_contract,
            output_root,
            external_root / "smoke",
            task,
            data_by_domain[domain],
            device=device,
        )
        rows.append(
            {
                **_verify_smoke_checkpoint(
                    resolve_path(result["checkpoint"]),
                    task,
                    data_by_domain[domain],
                    config,
                    output_root,
                    device,
                ),
                "checkpoint": result["checkpoint"],
                "checkpoint_sha256": result["checkpoint_sha256"],
            }
        )
    atomic_json(
        output_root / f"validation/smoke_rank{int(rank):02d}.json",
        {
            "schema_version": "predictability_stepB_smoke_rank_v1",
            "contract_sha256": contract["contract_sha256"],
            "rank": int(rank),
            "tasks": rows,
            "passed": len(rows) == len(tasks),
        },
    )


def finalize_smoke(config_path: Path) -> None:
    contract, output_root, _ = verify_contract(config_path)
    config = contract["static_config"]
    expected = _smoke_tasks(config)
    observed = []
    for rank in range(int(config["world_size"])):
        payload = read_json(output_root / f"validation/smoke_rank{rank:02d}.json")
        if payload.get("contract_sha256") != contract["contract_sha256"] or not payload.get("passed"):
            raise RuntimeError(f"smoke rank {rank} did not pass")
        observed.extend(payload["tasks"])
    expected_ids = {str(row["task_id"]) for row in expected}
    observed_ids = [str(row["task_id"]) for row in observed]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != expected_ids:
        raise RuntimeError("smoke global task completeness differs")
    report = {
        "schema_version": "predictability_stepB_smoke_report_v1",
        "contract_sha256": contract["contract_sha256"],
        "passed": True,
        "tasks_expected": len(expected),
        "tasks_completed": len(observed),
        "all_model_families": sorted({str(row["model"]).split("_")[0] for row in observed}),
        "exact_repeat_predictions": all(bool(row["exact_repeat"]) for row in observed),
        "checkpoint_roundtrip": all(bool(row["checkpoint_roundtrip"]) for row in observed),
        "image_group_overlap_across_roles": 0,
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "validation/smoke_report.json", report)
    text_report = "# Step-B bounded smoke\n\n" + "\n".join(
        (
            f"- Contract: `{contract['contract_sha256']}`",
            f"- Tasks: {len(observed)}/{len(expected)}",
            "- M0/M1/M2/M3 exercised: true",
            "- Stage-1, dense Stage-2, routed Stage-2 exercised: true",
            "- Image-group overlap across fit/calibration/test roles: 0",
            "- Exact repeated predictions: true",
            "- Checkpoint roundtrip: true",
            "- Result: PASS",
        )
    ) + "\n"
    _atomic_bytes(output_root / "validation/smoke_report.md", text_report.encode())
    print(json.dumps(report, sort_keys=True))


def _validated_training_completions(
    contract: Mapping[str, Any], output_root: Path
) -> dict[str, dict[str, Any]]:
    tasks = read_jsonl(output_root / "work/training_tasks.jsonl")
    observed: dict[str, dict[str, Any]] = {}
    for task in tasks:
        path = output_root / f"work/training/rank{int(task['worker_rank']):02d}/{task['task_id']}.json"
        completion = read_json(path)
        checkpoint = resolve_path(completion.get("checkpoint", ""))
        if (
            completion.get("contract_sha256") != contract["contract_sha256"]
            or completion.get("task_sha256") != canonical_hash(task)
            or not checkpoint.is_file()
            or file_sha256(checkpoint) != completion.get("checkpoint_sha256")
        ):
            raise RuntimeError(f"invalid Step-B task completion: {task['task_id']}")
        task_id = str(task["task_id"])
        if task_id in observed:
            raise RuntimeError(f"duplicate Step-B task completion: {task_id}")
        observed[task_id] = {"task": task, "completion": completion}
    if len(observed) != len(tasks):
        raise RuntimeError("Step-B task completion count differs")
    for rank in range(int(contract["static_config"]["world_size"])):
        marker = read_json(output_root / f"work/training/rank{rank:02d}/complete.json")
        expected = sum(int(task["worker_rank"]) == rank for task in tasks)
        if (
            marker.get("contract_sha256") != contract["contract_sha256"]
            or int(marker.get("expected_tasks", -1)) != expected
            or int(marker.get("completed_tasks", -1)) != expected
        ):
            raise RuntimeError(f"Step-B worker {rank} global completion differs")
    return observed


def _transfer_tasks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    base_seed = int(config["seed"])
    seeds = [int(value) for value in config["training"]["seeds"]]
    tasks = []
    for train_domain, test_domain in (
        ("stage2_dense", "stage2_routed"),
        ("stage2_routed", "stage2_dense"),
    ):
        for fold in range(int(config["split"]["outer_folds"])):
            for target in ("read", "write"):
                for model, input_name, model_seeds, cost in (
                    ("m1_text_visual", "text_visual", (base_seed,), 1),
                    ("m2_text_visual", "text_visual", tuple(seeds), 2),
                    ("m3_z_RW", "z_RW", tuple(seeds), 6),
                ):
                    for seed in model_seeds:
                        task_id = (
                            f"transfer__{train_domain}__to__{test_domain}__f{fold}__"
                            f"{target}__{model}__s{seed}"
                        )
                        tasks.append(
                            {
                                "task_id": task_id,
                                "train_domain": train_domain,
                                "test_domain": test_domain,
                                "fold": fold,
                                "target": target,
                                "model": model,
                                "input": input_name,
                                "seed": seed,
                                "estimated_cost": cost,
                            }
                        )
    load = [0] * int(config["world_size"])
    for task in sorted(tasks, key=lambda row: (-int(row["estimated_cost"]), str(row["task_id"]))):
        rank = min(range(len(load)), key=lambda value: (load[value], value))
        task["worker_rank"] = rank
        load[rank] += int(task["estimated_cost"])
    return sorted(tasks, key=lambda row: str(row["task_id"]))


def transfer_worker(config_path: Path, rank: int, *, resume: bool) -> None:
    contract, output_root, external_root = verify_contract(config_path)
    config = contract["static_config"]
    if not 0 <= int(rank) < int(config["world_size"]):
        raise ValueError("invalid transfer worker rank")
    _verify_validated_caches(contract, output_root)
    training = _validated_training_completions(contract, output_root)
    configure_worker_runtime(config)
    torch.cuda.set_device(int(rank))
    device = torch.device(f"cuda:{int(rank)}")
    tasks = [task for task in _transfer_tasks(config) if int(task["worker_rank"]) == int(rank)]
    rank_root = output_root / f"work/transfer/rank{int(rank):02d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    data_cache: dict[str, dict[str, Any]] = {}
    completed = 0
    started = time.monotonic()
    for task in tasks:
        result_path = rank_root / f"{task['task_id']}.json"
        prediction_path = external_root / "transfer" / f"{task['task_id']}.pt"
        if resume and result_path.is_file() and prediction_path.is_file():
            old = read_json(result_path)
            if (
                old.get("contract_sha256") == contract["contract_sha256"]
                and old.get("task_sha256") == canonical_hash(task)
                and file_sha256(prediction_path) == old.get("prediction_sha256")
            ):
                completed += 1
                continue
        source_id = (
            f"{task['train_domain']}__f{task['fold']}__{task['target']}__"
            f"{task['model']}__{task['input']}__s{task['seed']}"
        )
        source = training[source_id]
        source_payload = torch.load(
            resolve_path(source["completion"]["checkpoint"]), map_location="cpu", weights_only=False
        )
        test_domain = str(task["test_domain"])
        if test_domain not in data_cache:
            data_cache[test_domain] = _domain_data(contract, output_root, test_domain)
        data = data_cache[test_domain]
        indices = _role_indices(output_root, data["rows"], int(task["fold"]))["outer_test"]
        input_width = (
            int(config["inputs"]["stage2_summary_width_each"])
            if str(task["model"]).startswith("m3_z_")
            else data["fixed"].input_width(str(task["input"]))
        )
        source_task = source["task"]
        model = _model_for_task(source_task, input_width, config).to(device=device, dtype=torch.float32)
        model.load_state_dict(source_payload["model_state"])
        raw = _predict_indices(
            model,
            source_task,
            data,
            indices,
            mean=(
                None if source_payload["normalization_mean"] is None
                else source_payload["normalization_mean"].to(device)
            ),
            std=(
                None if source_payload["normalization_std"] is None
                else source_payload["normalization_std"].to(device)
            ),
            nuisance_store=None,
            device=device,
            batch_size=(32 if str(task["model"]).startswith("m3_z_") else 512),
        )
        prediction = (
            raw * float(source_payload["target_scale"]) + float(source_payload["target_center"])
        )
        payload = {
            "schema_version": "predictability_stepB_state_regime_transfer_v1",
            "contract_sha256": contract["contract_sha256"],
            "task": task,
            "task_sha256": canonical_hash(task),
            "source_checkpoint_sha256": source["completion"]["checkpoint_sha256"],
            "test_indices": indices,
            "test_prediction": prediction,
        }
        atomic_torch(prediction_path, payload)
        completion = {
            "schema_version": "predictability_stepB_transfer_completion_v1",
            "contract_sha256": contract["contract_sha256"],
            "task_id": task["task_id"],
            "task_sha256": canonical_hash(task),
            "prediction_path": str(prediction_path),
            "prediction_sha256": file_sha256(prediction_path),
            "test_states": len(indices),
        }
        atomic_json(result_path, completion)
        completed += 1
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "rank": int(rank), "transfer_completed": completed,
                    "assigned": len(tasks), "task_id": task["task_id"],
                    "elapsed_seconds": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    atomic_json(
        rank_root / "complete.json",
        {
            "schema_version": "predictability_stepB_transfer_worker_complete_v1",
            "contract_sha256": contract["contract_sha256"],
            "rank": int(rank),
            "expected_tasks": len(tasks),
            "completed_tasks": completed,
            "completed_at": utc_now(),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare", "cache-stage1", "cache-dense", "cache-routed", "validate-caches",
            "rebind-caches", "smoke-worker", "smoke-finalize", "train-worker", "transfer-worker",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--previous-contract-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.config)
    elif args.command == "cache-stage1":
        build_stage1_cache(args.config)
    elif args.command == "cache-dense":
        build_stage2_cache(args.config, "dense")
    elif args.command == "cache-routed":
        build_stage2_cache(args.config, "routed")
    elif args.command == "validate-caches":
        validate_caches(args.config)
    elif args.command == "rebind-caches":
        if not args.previous_contract_sha256:
            raise ValueError("rebind-caches requires --previous-contract-sha256")
        rebind_validated_caches(args.config, args.previous_contract_sha256)
    elif args.command == "smoke-worker":
        if args.rank is None:
            raise ValueError("smoke-worker requires --rank")
        smoke_worker(args.config, args.rank)
    elif args.command == "smoke-finalize":
        finalize_smoke(args.config)
    elif args.command == "transfer-worker":
        if args.rank is None:
            raise ValueError("transfer-worker requires --rank")
        transfer_worker(args.config, args.rank, resume=args.resume)
    else:
        if args.rank is None:
            raise ValueError("train-worker requires --rank")
        train_worker(args.config, args.rank, resume=args.resume)


if __name__ == "__main__":
    main()
