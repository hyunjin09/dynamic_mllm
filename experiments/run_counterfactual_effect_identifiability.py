#!/usr/bin/env python3
"""Run frozen one-step counterfactual effect extraction and OOF fitting."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
import csv
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from binary_policy.executor import capture_four_action_route, one_step_four_action_from_baseline  # noqa: E402
from binary_policy.executor.inputs import build_binary_inputs  # noqa: E402
from dense_failure_stage1.runtime import build_dense_inputs, configure_dense_determinism  # noqa: E402
from dense_failure_stage2.counterfactual_identifiability import (  # noqa: E402
    ACTIONS,
    PairedTokenComparator,
    construct_summary_feature,
    feature_width,
    matched_random_pair_indices,
    pool_branch,
    tensor_sha256,
)
from dense_failure_stage2.closed_loop_trajectory_set import compact_router_state  # noqa: E402
from dense_failure_stage2.predictability_learnability import (  # noqa: E402
    SummaryScalarPredictor,
    robust_target_scale,
)
from experiments.run_predictability_stepA_measurement import (  # noqa: E402
    _cached_state,
    _load_model,
)


DEFAULT_CONFIG = ROOT / "configs/counterfactual_effect_identifiability_v1.json"
ALLOWED_ROOTS = (ROOT.resolve(), Path("/mnt/hyemin").resolve())
BOUND_CODE = (
    "configs/counterfactual_effect_identifiability_v1.json",
    "plans/counterfactual_effect_identifiability_plan.md",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/cache.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "dense_failure_stage1/runtime.py",
    "dense_failure_stage2/closed_loop_trajectory_set.py",
    "dense_failure_stage2/counterfactual_identifiability.py",
    "dense_failure_stage2/predictability_learnability.py",
    "experiments/run_counterfactual_effect_identifiability.py",
    "experiments/analyze_counterfactual_effect_identifiability.py",
    "experiments/run_predictability_stepA_measurement.py",
    "experiments/run_stage2_v1_training_revised.py",
)
ACTION_INDEX = {name: index for index, name in enumerate(ACTIONS)}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if not any(resolved == allowed or resolved.is_relative_to(allowed) for allowed in ALLOWED_ROOTS):
        raise ValueError(f"path escapes allowed roots: {value}")
    return resolved


def file_sha256(value: str | Path) -> str:
    digest = sha256()
    with resolve_path(value).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    result = json.loads(resolve_path(value).read_text())
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {value}")
    return result


def read_jsonl(value: str | Path) -> list[dict[str, Any]]:
    rows = []
    with resolve_path(value).open() as handle:
        for line_number, line in enumerate(handle, 1):
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


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic(path, "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows).encode())


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic(path, stream.getvalue().encode())


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    torch.save(value, temporary)
    os.replace(temporary, path)


def command_output(args: Sequence[str]) -> str:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed {args}: {result.stderr.strip()}")
    return result.stdout.strip()


def git_state() -> dict[str, str]:
    return {
        "commit": command_output(("git", "rev-parse", "HEAD")),
        "branch": command_output(("git", "branch", "--show-current")),
        "worktree_status": command_output(("git", "status", "--short")),
    }


def runtime_state() -> dict[str, Any]:
    packages = ("torch", "transformers", "numpy", "lmms-eval", "qwen-vl-utils", "Pillow", "av")
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


def model_file_hashes(snapshot: Path) -> dict[str, str]:
    names = (
        "config.json", "generation_config.json", "preprocessor_config.json", "chat_template.json",
        "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
        "model.safetensors.index.json", "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors", "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors", "model-00005-of-00005.safetensors",
    )
    output = {}
    for name in names:
        path = snapshot / name
        if not path.is_file():
            raise RuntimeError(f"model snapshot file is missing: {path}")
        output[name] = file_sha256(path)
    return output


def verify_artifact_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    files = manifest.get("files", {})
    iterable = files.items() if isinstance(files, dict) else (
        (str(row["path"]), str(row["sha256"])) for row in files
    )
    checked = 0
    for relative, expected in iterable:
        path = Path(relative)
        path = path if path.is_absolute() else root / path
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"parent artifact differs: {path}")
        checked += 1
    return {"path": str(manifest_path), "sha256": file_sha256(manifest_path), "files_verified": checked}


def _source_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    keys = (
        "internal_samples", "dense_states", "dense_features", "dense_utilities", "dense_flips",
        "routed_states", "routed_features", "routed_utilities", "routed_flips",
        "routed_program_manifest", "routed_uid_files", "phase76_contract",
    )
    for fold in range(int(config["split"]["folds"])):
        keys += ()
    output = {key: file_sha256(config["sources"][key]) for key in keys}
    output["fold_registry"] = file_sha256(config["split"]["registry"])
    for fold in range(int(config["split"]["folds"])):
        output[f"inner_roles_fold{fold}"] = file_sha256(
            str(config["split"]["inner_roles_pattern"]).format(fold=fold)
        )
    return output


def _index_unique(rows: Sequence[Mapping[str, Any]], key: str, name: str) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        value = str(row[key])
        if value in output:
            raise RuntimeError(f"duplicate {name}: {value}")
        output[value] = dict(row)
    return output


def _utility_index(path: str | Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    output = _index_unique(rows, "state_id", "utility state")
    for row in output.values():
        read_value = float(row["u_read_w1"])
        write_value = float(row["u_write_r1"])
        if not math.isclose(read_value, float(row["q_full"]) - float(row["q_write_only"]), abs_tol=1e-9):
            raise RuntimeError(f"READ target algebra differs: {row['state_id']}")
        if not math.isclose(write_value, float(row["q_full"]) - float(row["q_read_only"]), abs_tol=1e-9):
            raise RuntimeError(f"WRITE target algebra differs: {row['state_id']}")
    return output


def _build_state_index(config: Mapping[str, Any], domain: str) -> list[dict[str, Any]]:
    states = read_jsonl(config["sources"][f"{domain}_states"])
    features = _index_unique(
        read_jsonl(config["sources"][f"{domain}_features"]), "state_id", f"{domain} feature"
    )
    utilities = _utility_index(config["sources"][f"{domain}_utilities"])
    folds = _index_unique(read_jsonl(config["split"]["registry"]), "uid", "fold UID")
    expected = int(config["population"][f"{domain}_states"])
    if len(states) != expected or len({str(row["state_id"]) for row in states}) != expected:
        raise RuntimeError(f"{domain} state census differs")
    output = []
    text_cursor = visual_cursor = 0
    for row_index, row in enumerate(sorted(states, key=lambda value: str(value["state_id"]))):
        state_id = str(row["state_id"])
        uid = str(row["uid"])
        feature = features.get(state_id)
        utility = utilities.get(state_id)
        if feature is None or utility is None or uid not in folds:
            raise RuntimeError(f"{domain} state lacks feature/utility/fold: {state_id}")
        nt, nv = int(feature["text_tokens"]), int(feature["visual_tokens"])
        output.append(
            {
                **dict(row),
                "row_index": row_index,
                "fold": int(folds[uid]["fold"]),
                "text_tokens": nt,
                "visual_tokens": nv,
                "text_offset": text_cursor,
                "visual_offset": visual_cursor,
                "pre_state_file": str(feature["state_file"]),
                "pre_state_sha256": str(feature["state_sha256"]),
                "pre_tensor_sha256": dict(feature["tensor_sha256"]),
                "u_read": float(utility["u_read_w1"]),
                "u_write": float(utility["u_write_r1"]),
            }
        )
        text_cursor += nt
        visual_cursor += nv
    return output


def _balanced_uid_schedule(rows: Sequence[Mapping[str, Any]], world_size: int) -> list[dict[str, Any]]:
    by_uid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_uid[str(row["uid"])].append(row)
    loads = [0] * int(world_size)
    output = []
    for uid, members in sorted(
        by_uid.items(), key=lambda item: (-sum(int(row["visual_tokens"]) for row in item[1]), item[0])
    ):
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        cost = sum(int(row["visual_tokens"]) for row in members)
        loads[rank] += cost
        output.append(
            {
                "uid": uid,
                "worker_rank": rank,
                "state_ids": sorted(str(row["state_id"]) for row in members),
                "estimated_visual_tokens": cost,
            }
        )
    return sorted(output, key=lambda row: (int(row["worker_rank"]), str(row["uid"])))


def _allocate_cache(external_root: Path, domain: str, rows: Sequence[Mapping[str, Any]], hidden: int) -> dict[str, Any]:
    root = external_root / domain
    root.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    text_total = sum(int(row["text_tokens"]) for row in rows)
    visual_total = sum(int(row["visual_tokens"]) for row in rows)
    specs = {
        "post_text": ((text_total, len(ACTIONS), hidden), np.uint16),
        "post_visual": ((visual_total, len(ACTIONS), hidden), np.uint16),
        "pooled": ((n, 4, 2 * hidden), np.uint16),
    }
    files = {}
    for name, (shape, dtype) in specs.items():
        path = root / f"{name}.bf16"
        expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if path.exists() and path.stat().st_size != expected_bytes:
            raise RuntimeError(f"existing cache has wrong size: {path}")
        if not path.exists():
            with path.open("wb") as handle:
                handle.truncate(expected_bytes)
        files[name] = {"path": str(path), "shape": list(shape), "bytes": expected_bytes}
    return {
        "schema_version": "counterfactual_effect_raw_cache_v1",
        "domain": domain,
        "rows": n,
        "hidden_size": hidden,
        "actions": list(ACTIONS),
        "text_tokens": text_total,
        "visual_tokens": visual_total,
        "dtype": "bfloat16_uint16_storage",
        "files": files,
    }


def prepare(config_path: Path) -> None:
    config = read_json(config_path)
    output_root = resolve_path(config["output_root"])
    external_root = resolve_path(config["external_cache_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    external_root.mkdir(parents=True, exist_ok=True)
    cache_link = output_root / "states/cache"
    cache_link.parent.mkdir(parents=True, exist_ok=True)
    if cache_link.exists() or cache_link.is_symlink():
        if not cache_link.is_symlink() or cache_link.resolve() != external_root:
            raise RuntimeError("existing cache link differs")
    else:
        cache_link.symlink_to(external_root, target_is_directory=True)
    parents = {}
    for name in ("stepA", "stepB", "phase76"):
        contract_path = resolve_path(config["sources"][f"{name}_contract"])
        manifest_path = resolve_path(config["sources"][f"{name}_artifact_manifest"])
        parents[name] = {
            "contract_sha256": read_json(contract_path).get("contract_sha256"),
            "contract_file_sha256": file_sha256(contract_path),
            "artifact": verify_artifact_manifest(manifest_path.parent, manifest_path),
        }
    dense = _build_state_index(config, "dense")
    routed = _build_state_index(config, "routed")
    if len({str(row["uid"]) for row in dense}) != int(config["population"]["dense_uids"]):
        raise RuntimeError("dense UID census differs")
    if len({str(row["image_group_id"]) for row in dense}) != int(config["population"]["dense_image_groups"]):
        raise RuntimeError("dense image-group census differs")
    if len({str(row["uid"]) for row in routed}) != int(config["population"]["routed_uids"]):
        raise RuntimeError("routed UID census differs")
    fold_rows = read_jsonl(config["split"]["registry"])
    fold_groups: dict[int, set[str]] = defaultdict(set)
    for row in fold_rows:
        fold_groups[int(row["fold"])].add(str(row["image_group_id"] or row["uid"]))
    for left in range(int(config["split"]["folds"])):
        for right in range(left + 1, int(config["split"]["folds"])):
            if fold_groups[left].intersection(fold_groups[right]):
                raise RuntimeError("inherited fold registry leaks image groups")
    atomic_jsonl(output_root / "states/dense_state_manifest.jsonl", dense)
    atomic_jsonl(output_root / "states/routed_state_manifest.jsonl", routed)
    atomic_jsonl(output_root / "splits/inherited_stepB_fold_registry.jsonl", fold_rows)
    for fold in range(int(config["split"]["folds"])):
        source = read_jsonl(str(config["split"]["inner_roles_pattern"]).format(fold=fold))
        atomic_jsonl(output_root / f"splits/inner_roles_fold{fold}.jsonl", source)
    support = []
    for domain, rows in (("dense", dense), ("routed", routed)):
        for fold in range(int(config["split"]["folds"])):
            subset = [row for row in rows if int(row["fold"]) == fold]
            support.append(
                {
                    "domain": domain,
                    "fold": fold,
                    "uids": len({str(row["uid"]) for row in subset}),
                    "image_groups": len({str(row["image_group_id"]) for row in subset}),
                    "states": len(subset),
                    "read_harmful": sum(float(row["u_read"]) < 0 for row in subset),
                    "write_harmful": sum(float(row["u_write"]) < 0 for row in subset),
                }
            )
    atomic_csv(output_root / "splits/fold_support.csv", support)
    _atomic(
        output_root / "splits/fold_validation.md",
        ("# Fold validation\n\nInherited exact Step-B five-fold registry. UID overlap = 0 across outer folds; "
         "image-group overlap = 0. Inner fit/calibration/outer-test roles are copied byte-equivalently.\n").encode(),
    )
    world = int(config["world_size"])
    atomic_jsonl(output_root / "work/dense_schedule.jsonl", _balanced_uid_schedule(dense, world))
    atomic_jsonl(output_root / "work/routed_schedule.jsonl", _balanced_uid_schedule(routed, world))
    dense_cache = _allocate_cache(external_root, "dense", dense, int(config["model"]["hidden_size"]))
    routed_cache = _allocate_cache(external_root, "routed", routed, int(config["model"]["hidden_size"]))
    source_hashes = _source_hashes(config)
    contract = {
        "schema_version": "counterfactual_effect_identifiability_contract_v1",
        "static_config": config,
        "config_sha256": file_sha256(config_path),
        "git": git_state(),
        "runtime": runtime_state(),
        "parents": parents,
        "source_sha256": source_hashes,
        "bound_code_sha256": {path: file_sha256(path) for path in BOUND_CODE},
        "model_snapshot_sha256": model_file_hashes(resolve_path(config["model"]["snapshot_path"])),
        "population": {
            "dense_states": len(dense), "dense_uids": len({str(row['uid']) for row in dense}),
            "routed_states": len(routed), "routed_uids": len({str(row['uid']) for row in routed}),
        },
        "cache_layout": {"dense": dense_cache, "routed": routed_cache},
        "review_verdict": "stable",
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_contract.json", contract)
    atomic_json(output_root / "work/dense_cache_layout.json", {**dense_cache, "contract_sha256": contract["contract_sha256"]})
    atomic_json(output_root / "work/routed_cache_layout.json", {**routed_cache, "contract_sha256": contract["contract_sha256"]})
    protocol = f"""# Counterfactual effect identifiability protocol

- Frozen contract: `{contract['contract_sha256']}`
- Dense primary: {len(dense)} exact post-trigger states; routed secondary: {len(routed)} exact prefix states.
- Inputs stop immediately after the current layer. No suffix state, final norm, LM head, generated answer, or correctness is an input.
- READ target: `q_FULL-q_WRITE_ONLY`; WRITE target: `q_FULL-q_READ_ONLY` from frozen Step A.
- Primary fixed-capacity comparison: Linear and 128-wide two-layer MLP over PRE/FULL_POST/OFF_POST/PAIR/DELTA/PAIR_PLUS_DELTA.
- Token comparator: shared 64-wide text/visual projection plus one-head attention per branch, then `[FULL;OFF;FULL-OFF]` readout.
- OOF: inherited Step-B five-fold image-group split, fold-local normalization, UID-balanced Huber regression, three frozen seeds.
- External deployment, router training, MCTS, and target redesign are out of scope.
"""
    _atomic(output_root / "protocol.md", protocol.encode())
    _atomic(
        output_root / "features/representation_contract.md",
        ("# Representation contract\n\nFor each branch, `r=[last valid text/control state; mean valid visual-token state]` "
         "immediately after the current layer. Raw compact valid-token tensors are stored in packed BF16 caches. "
         "PAIR order is FULL then OFF; DELTA is FULL minus OFF.\n").encode(),
    )
    print(json.dumps({"prepared": True, "contract_sha256": contract["contract_sha256"], "dense_states": len(dense), "routed_states": len(routed)}, sort_keys=True))


def verify_contract(config_path: Path, *, verify_model: bool = False) -> tuple[dict[str, Any], Path, Path]:
    config = read_json(config_path)
    output_root = resolve_path(config["output_root"])
    external_root = resolve_path(config["external_cache_root"])
    contract = read_json(output_root / "frozen_contract.json")
    if canonical_hash(contract) != contract.get("contract_sha256"):
        raise RuntimeError("contract self-hash differs")
    if contract.get("static_config") != config or contract.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError("config differs from frozen contract")
    if contract.get("git") != git_state() or contract.get("runtime") != runtime_state():
        raise RuntimeError("git/worktree or runtime differs from frozen contract")
    for path, expected in contract["bound_code_sha256"].items():
        if file_sha256(path) != expected:
            raise RuntimeError(f"bound code differs: {path}")
    if _source_hashes(config) != contract["source_sha256"]:
        raise RuntimeError("source artifact differs from frozen contract")
    cache_link = output_root / "states/cache"
    if not cache_link.is_symlink() or cache_link.resolve() != external_root:
        raise RuntimeError("external cache binding differs")
    if verify_model and model_file_hashes(resolve_path(config["model"]["snapshot_path"])) != contract["model_snapshot_sha256"]:
        raise RuntimeError("model snapshot differs")
    return contract, output_root, external_root


def _open_maps(layout: Mapping[str, Any], mode: str = "r+") -> dict[str, np.memmap]:
    return {
        name: np.memmap(
            resolve_path(spec["path"]), dtype=np.uint16, mode=mode, shape=tuple(spec["shape"])
        )
        for name, spec in layout["files"].items()
    }


def _write_bf16(target: np.memmap, index: Any, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().to(torch.bfloat16).contiguous()
    target[index] = tensor.view(torch.uint16).numpy()


def _read_bf16(source: np.memmap, index: Any) -> torch.Tensor:
    return torch.from_numpy(np.array(source[index], copy=True)).view(torch.bfloat16)


def _internal_index(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return _index_unique(read_jsonl(config["sources"]["internal_samples"]), "uid", "internal UID")


def _source_payload(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    return torch.load(resolved, map_location="cpu", weights_only=False)


def _validate_cached_pre(row: Mapping[str, Any], payload: Mapping[str, Any]) -> Mapping[str, Any]:
    state = payload["states"].get(str(row["state_id"]))
    if state is None:
        raise RuntimeError(f"cached state is missing: {row['state_id']}")
    cached = _cached_state(state["text_states"], state["visual_states"], state["text_mask"], state["visual_mask"])
    if cached["state_sha256"] != str(row["pre_state_sha256"]) or cached["tensor_sha256"] != dict(row["pre_tensor_sha256"]):
        raise RuntimeError(f"cached pre-state hash differs: {row['state_id']}")
    return state


def _canonical_post(baseline, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    if int(layer) + 1 < len(baseline.pre_layer_states):
        return baseline.pre_layer_states[int(layer) + 1]
    return baseline.text_hidden_state, baseline.visual_hidden_state


def _execute_state(
    wrapped,
    baseline,
    prepared,
    row: Mapping[str, Any],
    maps: Mapping[str, np.memmap],
    *,
    require_full_canonical: bool,
) -> dict[str, Any]:
    layer = int(row["layer"])
    live_text, live_visual = baseline.pre_layer_states[layer]
    live = _cached_state(live_text, live_visual, prepared.text_valid_mask, prepared.visual_valid_mask)
    if live["state_sha256"] != str(row["pre_state_sha256"]):
        raise RuntimeError(f"live pre-state differs: {row['state_id']}")
    row_index = int(row["row_index"])
    to, vo = int(row["text_offset"]), int(row["visual_offset"])
    nt, nv = int(row["text_tokens"]), int(row["visual_tokens"])
    _write_bf16(maps["pooled"], (row_index, 0), pool_branch(live_text, live_visual))
    branch_rows = []
    for action in ACTIONS:
        output = one_step_four_action_from_baseline(wrapped, baseline, layer, action)
        if output.pre_text_state.data_ptr() != live_text.data_ptr() or output.pre_visual_state.data_ptr() != live_visual.data_ptr():
            raise RuntimeError(f"branch did not share captured pre-state: {row['state_id']} {action}")
        expected_bits = {"FULL": (True, True), "WRITE_ONLY": (False, True), "READ_ONLY": (True, False)}[action]
        if (output.execution.read_on, output.execution.write_on) != expected_bits:
            raise RuntimeError(f"branch action semantics differ: {row['state_id']} {action}")
        raw_text = output.post_text_state
        raw_visual = output.post_visual_state
        if action == "FULL" and require_full_canonical:
            expected_text, expected_visual = _canonical_post(baseline, layer)
            if not torch.equal(raw_text, expected_text) or not torch.equal(raw_visual, expected_visual):
                raise RuntimeError(f"FULL post-state differs from canonical Dense: {row['state_id']}")
        compact = compact_router_state(
            raw_text,
            raw_visual,
            prepared.text_valid_mask,
            prepared.visual_valid_mask,
        )
        # Materialize each stored tensor on CPU exactly once. Repeated
        # hash/write calls on CUDA tensors otherwise trigger redundant device
        # synchronization and copies for every action branch.
        text = compact["text_states"].detach().cpu().to(torch.bfloat16).contiguous()
        visual = compact["visual_states"].detach().cpu().to(torch.bfloat16).contiguous()
        if text.shape[1] != nt or visual.shape[1] != nv:
            raise RuntimeError(
                f"compact post-state token count differs: {row['state_id']} {action}; "
                f"expected=({nt},{nv}) observed=({text.shape[1]},{visual.shape[1]})"
            )
        action_index = ACTION_INDEX[action]
        _write_bf16(maps["post_text"], (slice(to, to + nt), action_index, slice(None)), text.squeeze(0))
        _write_bf16(maps["post_visual"], (slice(vo, vo + nv), action_index, slice(None)), visual.squeeze(0))
        pooled = pool_branch(text, visual)
        _write_bf16(maps["pooled"], (row_index, action_index + 1), pooled)
        text_sha = tensor_sha256(text)
        visual_sha = tensor_sha256(visual)
        branch_rows.append(
            {
                "action": action,
                "read_on": bool(output.execution.read_on),
                "write_on": bool(output.execution.write_on),
                "decoder_calls": int(output.execution.decoder_calls),
                "post_text_sha256": text_sha,
                "post_visual_sha256": visual_sha,
                "post_state_sha256": sha256(
                    f"text:{text_sha}|visual:{visual_sha}".encode()
                ).hexdigest(),
                "pooled_sha256": tensor_sha256(pooled),
            }
        )
        del output
    return {
        "state_id": str(row["state_id"]),
        "uid": str(row["uid"]),
        "layer": layer,
        "pre_state_sha256": live["state_sha256"],
        "full_canonical_exact": bool(require_full_canonical),
        "branches": branch_rows,
    }


def _select_smoke(rows: Sequence[Mapping[str, Any]], count: int, *, require_layer27: bool) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    candidates = [dict(row) for row in rows]
    cells = [(dataset, wrong) for dataset in ("gqa", "chartqa", "textvqa") for wrong in (False, True)]
    for dataset, wrong in cells:
        subset = [row for row in candidates if str(row["dataset"]) == dataset and bool(row["dense_wrong"]) == wrong]
        if not subset:
            raise RuntimeError(f"smoke lacks cell {dataset}/{wrong}")
        row = min(subset, key=lambda value: sha256(str(value["state_id"]).encode()).hexdigest())
        selected.append(row)
        candidates.remove(row)
    if require_layer27 and not any(int(row["layer"]) == 27 for row in selected):
        subset = [row for row in candidates if int(row["layer"]) == 27]
        if not subset:
            raise RuntimeError("smoke lacks layer 27")
        selected.append(min(subset, key=lambda value: sha256(str(value["state_id"]).encode()).hexdigest()))
    while len(selected) < int(count):
        covered_layers = {int(row["layer"]) for row in selected}
        row = min(
            candidates,
            key=lambda value: (
                int(int(value["layer"]) in covered_layers),
                sha256(str(value["state_id"]).encode()).hexdigest(),
            ),
        )
        selected.append(row)
        candidates.remove(row)
    if len(selected) != int(count):
        raise RuntimeError("smoke selection differs")
    return selected


def _load_programs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return _index_unique(read_jsonl(config["sources"]["routed_program_manifest"]), "program_id", "program")


def _baseline_for_routed_state(wrapped, prepared, row: Mapping[str, Any], programs: Mapping[str, Mapping[str, Any]]):
    program = programs[str(row["anchor_program_id"])]
    actions = [str(value) for value in program["full_actions"]]
    layer = int(row["layer"])
    trigger = int(row["trigger_layer"])
    if actions[:trigger] != ["FULL"] * trigger or actions[trigger:layer] != list(row["prefix_actions"]):
        raise RuntimeError(f"routed prefix identity differs: {row['state_id']}")
    return capture_four_action_route(
        wrapped, {}, actions, prepared_inputs=prepared, use_cache=True, native_full_rows=True
    )


def smoke(config_path: Path, device_index: int) -> None:
    contract, output_root, _ = verify_contract(config_path, verify_model=True)
    config = contract["static_config"]
    configure_dense_determinism(int(config["seed"]), config["backend_settings"])
    torch.cuda.set_device(int(device_index))
    device = torch.device(f"cuda:{int(device_index)}")
    processor, _base, wrapped = _load_model(config, device)
    internal = _internal_index(config)
    programs = _load_programs(config)
    report_rows = []
    for domain in ("dense", "routed"):
        rows = read_jsonl(output_root / f"states/{domain}_state_manifest.jsonl")
        selected = _select_smoke(
            rows, int(config["smoke"][f"{domain}_states"]),
            require_layer27=bool(config["smoke"]["require_layer_27"]),
        )
        for row in selected:
            sample = internal[str(row["uid"])]["sample"]
            inputs, metadata = build_dense_inputs(processor, sample, device)
            if metadata["consumed_image_sha256"] != sample["image_content_sha256"]:
                raise RuntimeError(f"smoke image hash differs: {row['uid']}")
            prepared = build_binary_inputs(wrapped, inputs)
            if domain == "dense":
                baseline = capture_four_action_route(
                    wrapped, {}, ["FULL"] * 28, prepared_inputs=prepared,
                    use_cache=True, native_full_rows=True,
                )
            else:
                baseline = _baseline_for_routed_state(wrapped, prepared, row, programs)
            payload = _source_payload(row["pre_state_file"])
            _validate_cached_pre(row, payload)
            live_text, live_visual = baseline.pre_layer_states[int(row["layer"])]
            live = _cached_state(live_text, live_visual, prepared.text_valid_mask, prepared.visual_valid_mask)
            if live["state_sha256"] != str(row["pre_state_sha256"]):
                raise RuntimeError(f"smoke live pre-state differs: {row['state_id']}")
            repeat_sets = []
            for _repeat in range(2):
                branches = []
                for action in ACTIONS:
                    output = one_step_four_action_from_baseline(
                        wrapped, baseline, int(row["layer"]), action
                    )
                    if action == "FULL" and domain == "dense":
                        expected_text, expected_visual = _canonical_post(baseline, int(row["layer"]))
                        if not torch.equal(output.post_text_state, expected_text) or not torch.equal(output.post_visual_state, expected_visual):
                            raise RuntimeError(f"smoke FULL post parity failed: {row['state_id']}")
                    compact = compact_router_state(
                        output.post_text_state,
                        output.post_visual_state,
                        prepared.text_valid_mask,
                        prepared.visual_valid_mask,
                    )
                    if (
                        int(compact["text_states"].shape[1]) != int(row["text_tokens"])
                        or int(compact["visual_states"].shape[1]) != int(row["visual_tokens"])
                    ):
                        raise RuntimeError(f"smoke compact post-state layout differs: {row['state_id']} {action}")
                    expected_bits = {
                        "FULL": (True, True),
                        "WRITE_ONLY": (False, True),
                        "READ_ONLY": (True, False),
                    }[action]
                    action_exact = (
                        bool(output.execution.read_on), bool(output.execution.write_on)
                    ) == expected_bits
                    branches.append(
                        {
                            "action": action,
                            "pre_text_sha256": tensor_sha256(output.pre_text_state),
                            "pre_visual_sha256": tensor_sha256(output.pre_visual_state),
                            "post_text_sha256": tensor_sha256(compact["text_states"]),
                            "post_visual_sha256": tensor_sha256(compact["visual_states"]),
                            "read_on": output.execution.read_on,
                            "write_on": output.execution.write_on,
                            "action_exact": action_exact,
                            "target_layer": output.target_layer,
                        }
                    )
                    del output
                repeat_sets.append(branches)
            exact_repeat = repeat_sets[0] == repeat_sets[1]
            same_pre = len({(row_["pre_text_sha256"], row_["pre_visual_sha256"]) for row_ in repeat_sets[0]}) == 1
            actions_exact = all(branch["action_exact"] for branch in repeat_sets[0])
            item = {
                "domain": domain, "state_id": row["state_id"], "uid": row["uid"],
                "dataset": row["dataset"], "dense_wrong": row["dense_wrong"], "layer": row["layer"],
                "same_pre_state": same_pre, "exact_repeat": exact_repeat,
                "full_canonical_exact": True if domain == "dense" else None,
                "actions_exact": actions_exact,
                "branches": repeat_sets[0],
                "passed": same_pre and exact_repeat and actions_exact,
            }
            if not item["passed"]:
                raise RuntimeError(f"one-step smoke failed: {row['state_id']}")
            report_rows.append(item)
            del baseline, prepared, inputs, payload
            torch.cuda.empty_cache()
    atomic_jsonl(output_root / "smoke/smoke_rows.jsonl", report_rows)
    report = {
        "schema_version": "counterfactual_effect_smoke_v1",
        "contract_sha256": contract["contract_sha256"],
        "dense_states": sum(row["domain"] == "dense" for row in report_rows),
        "routed_states": sum(row["domain"] == "routed" for row in report_rows),
        "covers_layer_27": all(any(row["domain"] == domain and int(row["layer"]) == 27 for row in report_rows) for domain in ("dense", "routed")),
        "covers_all_dataset_outcome_cells": all(
            any(row["domain"] == domain and row["dataset"] == dataset and bool(row["dense_wrong"]) == wrong for row in report_rows)
            for domain in ("dense", "routed") for dataset in ("gqa", "chartqa", "textvqa") for wrong in (False, True)
        ),
        "same_pre_state_all": all(row["same_pre_state"] for row in report_rows),
        "exact_repeat_all": all(row["exact_repeat"] for row in report_rows),
        "actions_exact_all": all(row["actions_exact"] for row in report_rows),
        "dense_full_canonical_exact_all": all(
            row["full_canonical_exact"] for row in report_rows if row["domain"] == "dense"
        ),
        "passed": all(row["passed"] for row in report_rows),
    }
    if not report["covers_layer_27"] or not report["covers_all_dataset_outcome_cells"]:
        raise RuntimeError("smoke coverage gate failed")
    atomic_json(output_root / "smoke/smoke_report.json", report)
    print(json.dumps(report, sort_keys=True))


def _require_smoke(contract: Mapping[str, Any], output_root: Path) -> None:
    report = read_json(output_root / "smoke/smoke_report.json")
    if report.get("contract_sha256") != contract["contract_sha256"] or not report.get("passed"):
        raise RuntimeError("bound smoke has not passed")


def extraction_worker(config_path: Path, domain: str, rank: int, *, resume: bool) -> None:
    if domain not in {"dense", "routed"}:
        raise ValueError("domain must be dense or routed")
    contract, output_root, _ = verify_contract(config_path, verify_model=True)
    _require_smoke(contract, output_root)
    config = contract["static_config"]
    rank = int(rank)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    configure_dense_determinism(int(config["seed"]) + rank, config["backend_settings"])
    processor, _base, wrapped = _load_model(config, device)
    internal = _internal_index(config)
    programs = _load_programs(config) if domain == "routed" else {}
    rows = read_jsonl(output_root / f"states/{domain}_state_manifest.jsonl")
    rows_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_uid[str(row["uid"])].append(row)
    schedule = [
        row for row in read_jsonl(output_root / f"work/{domain}_schedule.jsonl")
        if int(row["worker_rank"]) == rank
    ]
    layout = contract["cache_layout"][domain]
    maps = _open_maps(layout)
    rank_root = output_root / f"work/extraction/{domain}/rank{rank:02d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    started = time.monotonic()
    for item in schedule:
        uid = str(item["uid"])
        result_path = rank_root / f"{sha256(uid.encode()).hexdigest()[:24]}.json"
        if resume and result_path.is_file():
            old = read_json(result_path)
            if old.get("contract_sha256") == contract["contract_sha256"] and old.get("state_ids") == item["state_ids"]:
                completed += 1
                continue
        sample = internal[uid]["sample"]
        inputs, metadata = build_dense_inputs(processor, sample, device)
        if metadata["consumed_image_sha256"] != sample["image_content_sha256"]:
            raise RuntimeError(f"image hash differs: {uid}")
        prepared = build_binary_inputs(wrapped, inputs)
        uid_rows = sorted(rows_by_uid[uid], key=lambda row: (str(row.get("anchor_program_id", "")), int(row["layer"]), str(row["state_id"])))
        payload_by_file = {}
        state_results = []
        if domain == "dense":
            baselines = {"dense": capture_four_action_route(
                wrapped, {}, ["FULL"] * 28, prepared_inputs=prepared,
                use_cache=True, native_full_rows=True,
            )}
            grouped = {"dense": uid_rows}
        else:
            grouped = defaultdict(list)
            for row in uid_rows:
                grouped[str(row["anchor_program_id"])].append(row)
            baselines = {}
        for group, group_rows in sorted(grouped.items()):
            if domain == "routed":
                baselines[group] = _baseline_for_routed_state(wrapped, prepared, group_rows[0], programs)
            baseline = baselines[group]
            for row in group_rows:
                if domain == "routed":
                    program_actions = [str(value) for value in programs[group]["full_actions"]]
                    layer = int(row["layer"])
                    trigger = int(row["trigger_layer"])
                    if (
                        program_actions[:trigger] != ["FULL"] * trigger
                        or program_actions[trigger:layer] != list(row["prefix_actions"])
                    ):
                        raise RuntimeError(f"routed prefix identity differs: {row['state_id']}")
                source_file = str(row["pre_state_file"])
                if source_file not in payload_by_file:
                    payload_by_file[source_file] = _source_payload(source_file)
                _validate_cached_pre(row, payload_by_file[source_file])
                state_results.append(
                    _execute_state(
                        wrapped, baseline, prepared, row, maps,
                        require_full_canonical=domain == "dense",
                    )
                )
            del baselines[group]
        observed_ids = [str(row["state_id"]) for row in state_results]
        if len(observed_ids) != len(set(observed_ids)) or sorted(observed_ids) != sorted(item["state_ids"]):
            raise RuntimeError(f"UID state completion differs: {uid}")
        for value in maps.values():
            value.flush()
        result = {
            "schema_version": "counterfactual_effect_uid_extraction_v1",
            "contract_sha256": contract["contract_sha256"],
            "domain": domain,
            "uid": uid,
            "state_ids": item["state_ids"],
            "image_sha256": metadata["consumed_image_sha256"],
            "state_results": state_results,
        }
        atomic_json(result_path, result)
        completed += 1
        del prepared, inputs, payload_by_file, state_results
        torch.cuda.empty_cache()
        if completed % 5 == 0 or completed == len(schedule):
            print(json.dumps({"domain": domain, "rank": rank, "completed_uids": completed, "assigned_uids": len(schedule), "elapsed_seconds": time.monotonic() - started}), flush=True)
    for value in maps.values():
        value.flush()
    atomic_json(
        rank_root / "complete.json",
        {"contract_sha256": contract["contract_sha256"], "domain": domain, "rank": rank,
         "expected_uids": len(schedule), "completed_uids": completed, "elapsed_seconds": time.monotonic() - started},
    )


def finalize_extraction(config_path: Path, domain: str) -> None:
    contract, output_root, _ = verify_contract(config_path)
    _require_smoke(contract, output_root)
    config = contract["static_config"]
    rows = read_jsonl(output_root / f"states/{domain}_state_manifest.jsonl")
    expected = {str(row["state_id"]): row for row in rows}
    uid_results = []
    for rank in range(int(config["world_size"])):
        rank_root = output_root / f"work/extraction/{domain}/rank{rank:02d}"
        complete = read_json(rank_root / "complete.json")
        if complete.get("contract_sha256") != contract["contract_sha256"] or complete.get("completed_uids") != complete.get("expected_uids"):
            raise RuntimeError(f"partial extraction rank: {domain}/{rank}")
        uid_results.extend(
            read_json(path) for path in sorted(rank_root.glob("*.json")) if path.name != "complete.json"
        )
    observed_uids = [str(row["uid"]) for row in uid_results]
    expected_uids = sorted({str(row["uid"]) for row in rows})
    if len(observed_uids) != len(set(observed_uids)) or sorted(observed_uids) != expected_uids:
        raise RuntimeError(f"global UID extraction differs: {domain}")
    state_results = [state for uid in uid_results for state in uid["state_results"]]
    observed_ids = [str(row["state_id"]) for row in state_results]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != set(expected):
        raise RuntimeError(f"global state extraction differs: {domain}")
    maps = _open_maps(contract["cache_layout"][domain], mode="r")
    branch_manifest = []
    parity_rows = []
    for result in sorted(state_results, key=lambda row: str(row["state_id"])):
        row = expected[str(result["state_id"])]
        ri, to, vo = int(row["row_index"]), int(row["text_offset"]), int(row["visual_offset"])
        nt, nv = int(row["text_tokens"]), int(row["visual_tokens"])
        if str(result["pre_state_sha256"]) != str(row["pre_state_sha256"]):
            raise RuntimeError(f"final pre-state hash differs: {row['state_id']}")
        if [branch["action"] for branch in result["branches"]] != list(ACTIONS):
            raise RuntimeError(f"final branch order differs: {row['state_id']}")
        for branch in result["branches"]:
            ai = ACTION_INDEX[str(branch["action"])]
            text = _read_bf16(maps["post_text"], (slice(to, to + nt), ai, slice(None))).unsqueeze(0)
            visual = _read_bf16(maps["post_visual"], (slice(vo, vo + nv), ai, slice(None))).unsqueeze(0)
            pooled = _read_bf16(maps["pooled"], (ri, ai + 1, slice(None)))
            if tensor_sha256(text) != branch["post_text_sha256"] or tensor_sha256(visual) != branch["post_visual_sha256"] or tensor_sha256(pooled) != branch["pooled_sha256"]:
                raise RuntimeError(f"packed branch hash differs: {row['state_id']} {branch['action']}")
            branch_manifest.append(
                {
                    "contract_sha256": contract["contract_sha256"], "domain": domain,
                    "state_id": row["state_id"], "uid": row["uid"], "layer": row["layer"],
                    "action": branch["action"], "read_on": branch["read_on"], "write_on": branch["write_on"],
                    "post_text_sha256": branch["post_text_sha256"], "post_visual_sha256": branch["post_visual_sha256"],
                    "post_state_sha256": branch["post_state_sha256"], "pooled_sha256": branch["pooled_sha256"],
                    "text_offset": to, "visual_offset": vo, "text_tokens": nt, "visual_tokens": nv, "row_index": ri,
                }
            )
        parity_rows.append(
            {"domain": domain, "state_id": row["state_id"], "uid": row["uid"], "layer": row["layer"],
             "pre_hash_exact": True, "branch_count": 3, "same_pre_state": True,
             "full_canonical_exact": bool(result["full_canonical_exact"]), "passed": True}
        )
    atomic_jsonl(output_root / f"states/{domain}_branch_execution_manifest.jsonl", branch_manifest)
    atomic_csv(output_root / f"states/{domain}_state_hash_parity.csv", parity_rows)
    atomic_jsonl(output_root / f"features/{domain}_pooled_post_features_manifest.jsonl", rows)
    atomic_jsonl(output_root / f"features/{domain}_delta_features_manifest.jsonl", rows)
    atomic_jsonl(output_root / f"features/{domain}_token_feature_manifest.jsonl", rows)
    if domain == "dense":
        # Canonical primary-population names required by the frozen analysis
        # protocol. Domain-qualified files remain available for unambiguous
        # primary/secondary provenance.
        atomic_jsonl(output_root / "states/post_state_manifest.jsonl", branch_manifest)
        atomic_jsonl(output_root / "states/branch_execution_manifest.jsonl", branch_manifest)
        atomic_csv(output_root / "states/state_hash_parity.csv", parity_rows)
        atomic_jsonl(output_root / "features/pooled_post_features_manifest.jsonl", rows)
        atomic_jsonl(output_root / "features/delta_features_manifest.jsonl", rows)
        atomic_jsonl(output_root / "features/token_feature_manifest.jsonl", rows)
    layout = dict(contract["cache_layout"][domain])
    layout["contract_sha256"] = contract["contract_sha256"]
    layout["files"] = {
        name: {**dict(spec), "sha256": file_sha256(spec["path"])}
        for name, spec in layout["files"].items()
    }
    atomic_json(output_root / f"features/{domain}_cache_manifest.json", layout)
    print(json.dumps({"domain": domain, "states": len(rows), "branches": len(branch_manifest), "complete": True}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "smoke", "extract", "finalize-extraction"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--domain", choices=("dense", "routed"))
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config)
    elif args.command == "smoke":
        smoke(args.config, args.rank)
    elif args.command == "extract":
        if args.domain is None:
            parser.error("extract requires --domain")
        extraction_worker(args.config, args.domain, args.rank, resume=args.resume)
    else:
        if args.domain is None:
            parser.error("finalize-extraction requires --domain")
        finalize_extraction(args.config, args.domain)


if __name__ == "__main__":
    main()
