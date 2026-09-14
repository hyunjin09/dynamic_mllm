#!/usr/bin/env python3
"""Prepare, validate, and extract frozen READ short-horizon rollouts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from binary_policy.executor import capture_four_action_route  # noqa: E402
from binary_policy.executor.cache import BinaryRouteCache  # noqa: E402
from binary_policy.executor.four_action import four_action_layer  # noqa: E402
from binary_policy.executor.inputs import build_binary_inputs, resolve_decoder  # noqa: E402
from dense_failure_stage1.runtime import build_dense_inputs, configure_dense_determinism  # noqa: E402
from dense_failure_stage2.closed_loop_trajectory_set import compact_router_state  # noqa: E402
from dense_failure_stage2.counterfactual_identifiability import tensor_sha256  # noqa: E402
from dense_failure_stage2.read_short_horizon import (  # noqa: E402
    BRANCHES,
    HORIZONS,
    eligible_horizons,
    expected_action_trace,
    pool_horizon_state,
    validate_horizon_census,
    validate_order_invariant_hashes,
)
from experiments.run_counterfactual_effect_identifiability import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_jsonl,
    canonical_hash,
    file_sha256,
    git_state,
    model_file_hashes,
    read_json,
    read_jsonl,
    resolve_path,
)
from experiments.run_predictability_stepA_measurement import _cached_state  # noqa: E402
from experiments.run_stage2_v1_training_revised import _load_model  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/read_short_horizon_propagation_v1.json"
BOUND_CODE = (
    "configs/read_short_horizon_propagation_v1.json",
    "plans/read_short_horizon_counterfactual_propagation_plan.md",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/cache.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "dense_failure_stage1/runtime.py",
    "dense_failure_stage2/closed_loop_trajectory_set.py",
    "dense_failure_stage2/counterfactual_identifiability.py",
    "dense_failure_stage2/predictability_learnability.py",
    "dense_failure_stage2/read_short_horizon.py",
    "experiments/run_read_short_horizon_propagation.py",
    "experiments/analyze_read_short_horizon_propagation.py",
    "experiments/run_counterfactual_effect_identifiability.py",
    "experiments/analyze_counterfactual_effect_identifiability.py",
    "experiments/run_predictability_stepA_measurement.py",
)
BRANCH_ACTION = {"ON": "FULL", "OFF": "WRITE_ONLY"}
BRANCH_INDEX = {"ON": 0, "OFF": 1}


def stable_runtime_state() -> dict[str, Any]:
    """Runtime fingerprint independent of per-worker CUDA visibility."""

    packages = {}
    for name in ("torch", "transformers", "numpy", "lmms-eval", "qwen-vl-utils", "Pillow", "av"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    driver = subprocess.run(
        ("nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().splitlines()
    return {
        "python": sys.version.split()[0],
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "physical_gpus": driver,
    }


def _index_unique(rows: Sequence[Mapping[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    output = {}
    for source in rows:
        row = dict(source)
        value = str(row[key])
        if value in output:
            raise RuntimeError(f"duplicate {label}: {value}")
        output[value] = row
    return output


def _parent_record(contract_path: str, manifest_path: str) -> dict[str, Any]:
    contract = read_json(contract_path)
    manifest = read_json(manifest_path)
    if canonical_hash(contract) != contract.get("contract_sha256"):
        raise RuntimeError(f"parent contract self-hash differs: {contract_path}")
    if canonical_hash(manifest) != manifest.get("artifact_manifest_sha256"):
        raise RuntimeError(f"parent artifact manifest self-hash differs: {manifest_path}")
    return {
        "contract_sha256": contract["contract_sha256"],
        "contract_file_sha256": file_sha256(contract_path),
        "artifact_manifest_sha256": manifest["artifact_manifest_sha256"],
        "artifact_manifest_file_sha256": file_sha256(manifest_path),
    }


def _source_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    return {name: file_sha256(path) for name, path in config["sources"].items()}


def _support_row(horizon: int, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    layers: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[f"{row['dataset']}|{row['source_regime']}"] += 1
        layers[str(row["layer"])] += 1
    return {
        "support": "native",
        "horizon": horizon,
        "states": len(rows),
        "uids": len({str(row["uid"]) for row in rows}),
        "image_groups": len({str(row["image_group_id"]) for row in rows}),
        "dense_correct": sum(bool(row["dense_correct"]) for row in rows),
        "dense_wrong": sum(bool(row["dense_wrong"]) for row in rows),
        "layer_distribution_json": json.dumps(dict(sorted(layers.items(), key=lambda item: int(item[0]))), sort_keys=True),
        "dataset_source_json": json.dumps(dict(sorted(counts.items())), sort_keys=True),
    }


def _build_population(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    source = read_jsonl(config["sources"]["phase83_dense_states"])
    if len(source) != int(config["population"]["dense_states"]):
        raise RuntimeError("dense state census differs")
    folds = _index_unique(read_jsonl(config["split"]["registry"]), "uid", "fold UID")
    rows = []
    for row_index, source_row in enumerate(sorted(source, key=lambda row: str(row["state_id"]))):
        row = dict(source_row)
        if not math.isclose(float(row["h_r"]), -float(row["u_read"]), abs_tol=1e-12):
            raise RuntimeError(f"H_R target transform differs: {row['state_id']}")
        uid = str(row["uid"])
        if uid not in folds or int(row["fold"]) != int(folds[uid]["fold"]):
            raise RuntimeError(f"fold identity differs: {uid}")
        row["row_index"] = row_index
        row["eligible_horizons"] = list(eligible_horizons(int(row["layer"]), int(config["model"]["decoder_layers"])))
        row["common_h8"] = 8 in row["eligible_horizons"]
        rows.append(row)
    if len({str(row["state_id"]) for row in rows}) != len(rows):
        raise RuntimeError("dense state IDs are duplicated")
    if len({str(row["uid"]) for row in rows}) != int(config["population"]["dense_uids"]):
        raise RuntimeError("dense UID census differs")
    if len({str(row["image_group_id"]) for row in rows}) != int(config["population"]["dense_image_groups"]):
        raise RuntimeError("dense image-group census differs")
    by_horizon: dict[int, list[dict[str, Any]]] = {}
    for horizon in HORIZONS:
        selected = [row for row in rows if horizon in row["eligible_horizons"]]
        visual_offset = 0
        for horizon_index, row in enumerate(selected):
            row.setdefault("horizon_indices", {})[str(horizon)] = {
                "row_index": horizon_index,
                "visual_offset": visual_offset,
            }
            visual_offset += int(row["visual_tokens"])
        by_horizon[horizon] = selected
    return rows, by_horizon


def _allocate_layout(
    external_root: Path,
    by_horizon: Mapping[int, Sequence[Mapping[str, Any]]],
    hidden: int,
) -> dict[str, Any]:
    layout = {
        "schema_version": "read_short_horizon_cache_layout_v1",
        "hidden_size": hidden,
        "branches": list(BRANCHES),
        "dtype": "bfloat16_uint16_storage",
        "horizons": {},
    }
    for horizon, rows in by_horizon.items():
        root = external_root / "features" / f"h{horizon}"
        root.mkdir(parents=True, exist_ok=True)
        visual_tokens = sum(int(row["visual_tokens"]) for row in rows)
        specs = {
            "pooled": ([len(rows), 2, 2 * hidden], root / "pooled.bf16"),
            "text": ([len(rows), 2, hidden], root / "text.bf16"),
            "visual": ([visual_tokens, 2, hidden], root / "visual.bf16"),
        }
        files = {}
        for name, (shape, path) in specs.items():
            size = math.prod(shape) * 2
            if path.exists() and path.stat().st_size != size:
                raise RuntimeError(f"existing horizon cache size differs: {path}")
            if not path.exists():
                with path.open("wb") as handle:
                    handle.truncate(size)
            files[name] = {"path": str(path), "shape": shape, "bytes": size}
        layout["horizons"][str(horizon)] = {
            "rows": len(rows),
            "visual_tokens": visual_tokens,
            "files": files,
        }
    return layout


def _balanced_schedule(rows: Sequence[Mapping[str, Any]], world_size: int) -> list[dict[str, Any]]:
    by_uid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_uid[str(row["uid"])].append(row)
    loads = [0] * int(world_size)
    schedule = []
    for uid, members in sorted(
        by_uid.items(),
        key=lambda item: (
            -sum(2 * max(row["eligible_horizons"]) * int(row["visual_tokens"]) for row in item[1]),
            item[0],
        ),
    ):
        cost = sum(2 * max(row["eligible_horizons"]) * int(row["visual_tokens"]) for row in members)
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        loads[rank] += cost
        schedule.append({
            "uid": uid,
            "worker_rank": rank,
            "state_ids": sorted(str(row["state_id"]) for row in members),
            "estimated_token_layer_calls": cost,
        })
    return sorted(schedule, key=lambda row: (int(row["worker_rank"]), str(row["uid"])))


def prepare(config_path: Path) -> None:
    config = read_json(config_path)
    if tuple(config["rollout"]["horizons"]) != HORIZONS:
        raise RuntimeError("horizon contract differs")
    if config["features"]["primary_emergence_condition"] != "delta":
        raise RuntimeError("primary emergence condition must remain delta")
    token = config["features"]["token_comparator"]
    if not token["mandatory"] or not token["identical_across_horizons"]:
        raise RuntimeError("token comparator must be frozen and identical across horizons")
    output_root = resolve_path(config["output_root"])
    external_root = resolve_path(config["external_cache_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    external_root.mkdir(parents=True, exist_ok=True)
    cache_link = output_root / "features/cache"
    cache_link.parent.mkdir(parents=True, exist_ok=True)
    if cache_link.exists() or cache_link.is_symlink():
        if not cache_link.is_symlink() or cache_link.resolve() != external_root:
            raise RuntimeError("external cache binding differs")
    else:
        cache_link.symlink_to(external_root, target_is_directory=True)
    parents = {
        "phase83": _parent_record(config["sources"]["phase83_contract"], config["sources"]["phase83_artifact_manifest"]),
        "phase82": _parent_record(config["sources"]["phase82_contract"], config["sources"]["phase82_artifact_manifest"]),
        "stepA": _parent_record(config["sources"]["stepA_contract"], config["sources"]["stepA_artifact_manifest"]),
        "stepB": _parent_record(config["sources"]["stepB_contract"], config["sources"]["stepB_artifact_manifest"]),
    }
    rows, by_horizon = _build_population(config)
    atomic_jsonl(output_root / "population/state_manifest.jsonl", rows)
    atomic_jsonl(output_root / "population/common_support_h8_manifest.jsonl", by_horizon[8])
    support = [_support_row(horizon, by_horizon[horizon]) for horizon in HORIZONS]
    common = by_horizon[8]
    for horizon in (1, 2, 4, 8):
        row = _support_row(horizon, common)
        row["support"] = "common_h8"
        support.append(row)
    atomic_csv(output_root / "population/horizon_support.csv", support)
    fold_rows = read_jsonl(config["split"]["registry"])
    atomic_jsonl(output_root / "splits/inherited_stepB_fold_registry.jsonl", fold_rows)
    for fold in range(int(config["split"]["folds"])):
        atomic_jsonl(
            output_root / f"splits/inner_roles_fold{fold}.jsonl",
            read_jsonl(str(config["split"]["inner_roles_pattern"]).format(fold=fold)),
        )
    fold_support = []
    for support_name, supported in (("native", None), ("common_h8", common)):
        for horizon in HORIZONS:
            subset = by_horizon[horizon] if supported is None else common
            for fold in range(int(config["split"]["folds"])):
                selected = [row for row in subset if int(row["fold"]) == fold]
                fold_support.append({
                    "support": support_name,
                    "horizon": horizon,
                    "fold": fold,
                    "states": len(selected),
                    "uids": len({str(row["uid"]) for row in selected}),
                    "image_groups": len({str(row["image_group_id"]) for row in selected}),
                })
    atomic_csv(output_root / "splits/fold_support_by_horizon.csv", fold_support)
    schedule = _balanced_schedule(rows, int(config["world_size"]))
    atomic_jsonl(output_root / "work/extraction_schedule.jsonl", schedule)
    layout = _allocate_layout(external_root, by_horizon, int(config["model"]["hidden_size"]))
    contract = {
        "schema_version": "read_short_horizon_propagation_contract_v1",
        "static_config": config,
        "config_sha256": file_sha256(config_path),
        "git": git_state(),
        "runtime": stable_runtime_state(),
        "parents": parents,
        "source_sha256": _source_hashes(config),
        "bound_code_sha256": {path: file_sha256(path) for path in BOUND_CODE},
        "model_snapshot_sha256": model_file_hashes(resolve_path(config["model"]["snapshot_path"])),
        "population": {
            "states": len(rows),
            "uids": len({str(row["uid"]) for row in rows}),
            "image_groups": len({str(row["image_group_id"]) for row in rows}),
            "native_support": {str(h): len(by_horizon[h]) for h in HORIZONS},
            "common_h8_states": len(common),
        },
        "cache_layout": layout,
        "independent_review": {
            "verdict": "revise_accepted",
            "requirements": [
                "fresh_cache_repeat",
                "swapped_branch_order_h8_hash_parity",
                "mandatory_identical_token_comparator",
            ],
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_contract.json", contract)
    atomic_json(output_root / "work/cache_layout.json", {**layout, "contract_sha256": contract["contract_sha256"]})
    protocol = f"""# READ short-horizon propagation protocol

- Frozen contract: `{contract['contract_sha256']}`
- Target: `H_R = q_WRITE_ONLY - q_FULL`; positive is harmful READ.
- Population: {len(rows):,} exact dense states / {len({str(row['uid']) for row in rows}):,} UIDs.
- Horizons: H=1/2/4/8; branch ON is FULL at layer l, branch OFF is WRITE_ONLY at l, and every later executed layer is FULL.
- Primary emergence: two-layer MLP on `delta = ON - OFF`, evaluated on common H=8 support against H=1.
- Fixed controls: ON, OFF, PAIR, PAIR+DELTA, text/visual delta, identical token comparator, random pairs, and UID-permuted training targets.
- OOF: inherited Step-B five-fold image-group registry, fold-local normalization, UID-balanced Huber loss, three fixed seeds.
- Bulk extraction requires exact Phase-82 H=1 parity, ON canonical parity at every reached horizon, fresh-cache repeatability, and swapped branch-order parity.
- No WRITE analysis, search, deployment router, generation, external evaluation, or target redesign.
"""
    (output_root / "protocol.md").write_text(protocol)
    print(json.dumps({"prepared": True, "contract_sha256": contract["contract_sha256"], "population": contract["population"]}, sort_keys=True))


def verify_contract(config_path: Path, *, verify_model: bool = False) -> tuple[dict[str, Any], Path, Path]:
    config = read_json(config_path)
    output_root = resolve_path(config["output_root"])
    external_root = resolve_path(config["external_cache_root"])
    contract = read_json(output_root / "frozen_contract.json")
    if canonical_hash(contract) != contract.get("contract_sha256"):
        raise RuntimeError("contract self-hash differs")
    if contract.get("static_config") != config or contract.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError("config differs from frozen contract")
    if contract.get("git") != git_state() or contract.get("runtime") != stable_runtime_state():
        raise RuntimeError("git/worktree or runtime differs from frozen contract")
    for path, digest in contract["bound_code_sha256"].items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"bound code differs: {path}")
    if _source_hashes(config) != contract["source_sha256"]:
        raise RuntimeError("source artifact differs from frozen contract")
    link = output_root / "features/cache"
    if not link.is_symlink() or link.resolve() != external_root:
        raise RuntimeError("external cache binding differs")
    for horizon in HORIZONS:
        for spec in contract["cache_layout"]["horizons"][str(horizon)]["files"].values():
            path = resolve_path(spec["path"])
            if not path.is_file() or path.stat().st_size != int(spec["bytes"]):
                raise RuntimeError(f"horizon cache size differs: {path}")
    if verify_model and model_file_hashes(resolve_path(config["model"]["snapshot_path"])) != contract["model_snapshot_sha256"]:
        raise RuntimeError("model snapshot differs")
    return contract, output_root, external_root


def _open_maps(layout: Mapping[str, Any], mode: str = "r+") -> dict[int, dict[str, np.memmap]]:
    output = {}
    for horizon in HORIZONS:
        files = layout["horizons"][str(horizon)]["files"]
        output[horizon] = {
            name: np.memmap(resolve_path(spec["path"]), dtype=np.uint16, mode=mode, shape=tuple(spec["shape"]))
            for name, spec in files.items()
        }
    return output


def _write_bf16(target: np.memmap, index: Any, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().to(torch.bfloat16).contiguous()
    target[index] = tensor.view(torch.uint16).numpy()


def _read_bf16(target: np.memmap, index: Any) -> torch.Tensor:
    return torch.from_numpy(np.array(target[index], copy=True)).view(torch.bfloat16)


def _internal_index(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return _index_unique(read_jsonl(config["sources"]["internal_samples"]), "uid", "internal UID")


def _phase82_oracle(config: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_jsonl(config["sources"]["phase82_dense_branches"])
    result = {}
    for row in rows:
        action = str(row["action"])
        if action not in {"FULL", "WRITE_ONLY"}:
            continue
        key = (str(row["state_id"]), action)
        if key in result:
            raise RuntimeError(f"duplicate Phase-82 branch oracle: {key}")
        result[key] = row
    return result


def _canonical_compact(baseline, prepared, reached_layer: int) -> dict[str, torch.Tensor]:
    if reached_layer < len(baseline.pre_layer_states):
        text, visual = baseline.pre_layer_states[reached_layer]
    else:
        text, visual = baseline.text_hidden_state, baseline.visual_hidden_state
    return compact_router_state(text, visual, prepared.text_valid_mask, prepared.visual_valid_mask)


@torch.inference_mode()
def rollout_branch(
    wrapped,
    baseline,
    prepared,
    *,
    layer: int,
    branch: str,
    horizons: Sequence[int],
) -> dict[int, dict[str, Any]]:
    """Run one isolated branch to its largest native horizon."""

    decoder = resolve_decoder(wrapped)
    branch = str(branch).upper()
    action0 = BRANCH_ACTION[branch]
    cache = BinaryRouteCache(len(decoder.layers))
    text, visual = baseline.pre_layer_states[int(layer)]
    text = text.detach().clone()
    visual = visual.detach().clone()
    captures = {}
    trace = []
    for step in range(1, max(int(value) for value in horizons) + 1):
        current_layer = int(layer) + step - 1
        action = action0 if step == 1 else "FULL"
        text, visual, execution = four_action_layer(
            wrapped,
            decoder.layers[current_layer],
            text,
            visual,
            prepared,
            action=action,
            layer_index=current_layer,
            cache=cache,
            use_cache=True,
            native_causal=baseline.native_causal,
        )
        trace.append(action)
        expected_bits = (True, True) if action == "FULL" else (False, True)
        if (bool(execution.read_on), bool(execution.write_on)) != expected_bits:
            raise RuntimeError("rollout action semantics differ")
        if step in horizons:
            compact = compact_router_state(
                text,
                visual,
                prepared.text_valid_mask,
                prepared.visual_valid_mask,
            )
            text_cpu = compact["text_states"].detach().cpu().to(torch.bfloat16).contiguous()
            visual_cpu = compact["visual_states"].detach().cpu().to(torch.bfloat16).contiguous()
            pooled = pool_horizon_state(text_cpu, visual_cpu).to(torch.bfloat16).contiguous()
            captures[step] = {
                "text": text_cpu,
                "visual": visual_cpu,
                "pooled": pooled,
                "text_sha256": tensor_sha256(text_cpu),
                "visual_sha256": tensor_sha256(visual_cpu),
                "pooled_sha256": tensor_sha256(pooled),
                "state_sha256": sha256(f"text:{tensor_sha256(text_cpu)}|visual:{tensor_sha256(visual_cpu)}".encode()).hexdigest(),
                "action_trace": list(trace),
                "fresh_cache_object": True,
            }
    return captures


def _validate_capture(
    row: Mapping[str, Any],
    branch: str,
    horizon: int,
    capture: Mapping[str, Any],
    oracle: Mapping[tuple[str, str], Mapping[str, Any]],
    baseline,
    prepared,
) -> dict[str, Any]:
    action = BRANCH_ACTION[branch]
    trace_exact = tuple(capture["action_trace"]) == expected_action_trace(horizon, action)
    phase82_exact = None
    if horizon == 1:
        expected = oracle[(str(row["state_id"]), action)]
        phase82_exact = (
            capture["text_sha256"] == expected["post_text_sha256"]
            and capture["visual_sha256"] == expected["post_visual_sha256"]
            and capture["pooled_sha256"] == expected["pooled_sha256"]
        )
        if not phase82_exact:
            raise RuntimeError(f"H=1 Phase-82 parity differs: {row['state_id']} {branch}")
    on_canonical_exact = None
    if branch == "ON":
        expected = _canonical_compact(baseline, prepared, int(row["layer"]) + int(horizon))
        on_canonical_exact = (
            tensor_sha256(expected["text_states"].to(torch.bfloat16)) == capture["text_sha256"]
            and tensor_sha256(expected["visual_states"].to(torch.bfloat16)) == capture["visual_sha256"]
        )
        if not on_canonical_exact:
            raise RuntimeError(f"ON canonical parity differs: {row['state_id']} H={horizon}")
    if not trace_exact:
        raise RuntimeError(f"action trace differs: {row['state_id']} {branch} H={horizon}")
    return {
        "trace_exact": trace_exact,
        "phase82_h1_exact": phase82_exact,
        "on_canonical_exact": on_canonical_exact,
    }


def _write_capture(
    maps: Mapping[int, Mapping[str, np.memmap]],
    row: Mapping[str, Any],
    branch: str,
    horizon: int,
    capture: Mapping[str, Any],
) -> None:
    index = row["horizon_indices"][str(horizon)]
    ri, vo = int(index["row_index"]), int(index["visual_offset"])
    nv = int(row["visual_tokens"])
    bi = BRANCH_INDEX[branch]
    _write_bf16(maps[horizon]["pooled"], (ri, bi), capture["pooled"])
    _write_bf16(maps[horizon]["text"], (ri, bi), capture["text"][0, -1])
    _write_bf16(maps[horizon]["visual"], (slice(vo, vo + nv), bi, slice(None)), capture["visual"][0])


def _select_smoke(rows: Sequence[Mapping[str, Any]], count: int, boundary_layers: Sequence[int]) -> list[dict[str, Any]]:
    selected = []
    remaining = [dict(row) for row in rows]
    cells = [(dataset, wrong) for dataset in ("gqa", "chartqa", "textvqa") for wrong in (False, True)]
    for dataset, wrong in cells:
        subset = [row for row in remaining if row["dataset"] == dataset and bool(row["dense_wrong"]) == wrong and row["common_h8"]]
        if not subset:
            raise RuntimeError(f"smoke lacks H8 cell: {dataset}/{wrong}")
        choice = min(subset, key=lambda row: sha256(str(row["state_id"]).encode()).hexdigest())
        selected.append(choice)
        remaining.remove(choice)
    for layer in boundary_layers:
        if any(int(row["layer"]) == int(layer) for row in selected):
            continue
        subset = [row for row in remaining if int(row["layer"]) == int(layer)]
        if not subset:
            raise RuntimeError(f"smoke lacks boundary layer {layer}")
        choice = min(subset, key=lambda row: sha256(str(row["state_id"]).encode()).hexdigest())
        selected.append(choice)
        remaining.remove(choice)
    while len(selected) < int(count):
        choice = min(remaining, key=lambda row: sha256(str(row["state_id"]).encode()).hexdigest())
        selected.append(choice)
        remaining.remove(choice)
    return selected


def smoke(config_path: Path, device_index: int) -> None:
    contract, output_root, _ = verify_contract(config_path, verify_model=True)
    config = contract["static_config"]
    configure_dense_determinism(int(config["seed"]), config["backend_settings"])
    torch.cuda.set_device(int(device_index))
    device = torch.device(f"cuda:{int(device_index)}")
    processor, _base, wrapped = _load_model(config, device)
    rows = read_jsonl(output_root / "population/state_manifest.jsonl")
    selected = _select_smoke(
        rows,
        int(config["validation"]["smoke_states"]),
        config["validation"]["require_horizon_boundary_layers"],
    )
    internal = _internal_index(config)
    oracle = _phase82_oracle(config)
    report_rows = []
    for row in selected:
        sample = internal[str(row["uid"])]["sample"]
        inputs, metadata = build_dense_inputs(processor, sample, device)
        if metadata["consumed_image_sha256"] != sample["image_content_sha256"]:
            raise RuntimeError(f"smoke image hash differs: {row['uid']}")
        prepared = build_binary_inputs(wrapped, inputs)
        baseline = capture_four_action_route(
            wrapped, {}, ["FULL"] * 28, prepared_inputs=prepared, use_cache=True, native_full_rows=True
        )
        source = torch.load(resolve_path(row["pre_state_file"]), map_location="cpu", weights_only=False)
        cached = source["states"].get(str(row["state_id"]))
        if cached is None:
            raise RuntimeError(f"smoke source state missing: {row['state_id']}")
        expected_pre = _cached_state(cached["text_states"], cached["visual_states"], cached["text_mask"], cached["visual_mask"])
        live_text, live_visual = baseline.pre_layer_states[int(row["layer"])]
        live_pre = _cached_state(live_text, live_visual, prepared.text_valid_mask, prepared.visual_valid_mask)
        if expected_pre["state_sha256"] != row["pre_state_sha256"] or live_pre["state_sha256"] != row["pre_state_sha256"]:
            raise RuntimeError(f"smoke pre-state differs: {row['state_id']}")
        orders = (("ON", "OFF"), ("ON", "OFF"), ("OFF", "ON"))
        order_hashes = []
        first_captures = None
        for order in orders:
            captures = {}
            for branch in order:
                captures[branch] = rollout_branch(
                    wrapped,
                    baseline,
                    prepared,
                    layer=int(row["layer"]),
                    branch=branch,
                    horizons=row["eligible_horizons"],
                )
            hashes = {
                (branch, int(horizon)): captures[branch][int(horizon)]["state_sha256"]
                for branch in BRANCHES
                for horizon in row["eligible_horizons"]
            }
            order_hashes.append(hashes)
            if first_captures is None:
                first_captures = captures
        validate_order_invariant_hashes(order_hashes[0], order_hashes[1])
        validate_order_invariant_hashes(order_hashes[0], order_hashes[2])
        assert first_captures is not None
        validations = []
        for branch in BRANCHES:
            for horizon in row["eligible_horizons"]:
                validations.append(_validate_capture(row, branch, int(horizon), first_captures[branch][int(horizon)], oracle, baseline, prepared))
        report_rows.append({
            "contract_sha256": contract["contract_sha256"],
            "state_id": row["state_id"],
            "uid": row["uid"],
            "dataset": row["dataset"],
            "dense_wrong": row["dense_wrong"],
            "layer": row["layer"],
            "eligible_horizons": row["eligible_horizons"],
            "image_sha256": metadata["consumed_image_sha256"],
            "fresh_cache_repeat_exact": order_hashes[0] == order_hashes[1],
            "swapped_order_exact": order_hashes[0] == order_hashes[2],
            "h1_phase82_exact": all(v["phase82_h1_exact"] for v in validations if v["phase82_h1_exact"] is not None),
            "on_canonical_exact": all(v["on_canonical_exact"] for v in validations if v["on_canonical_exact"] is not None),
            "action_trace_exact": all(v["trace_exact"] for v in validations),
            "passed": True,
        })
        del baseline, prepared, inputs, source, first_captures
        torch.cuda.empty_cache()
    report = {
        "schema_version": "read_short_horizon_smoke_v1",
        "contract_sha256": contract["contract_sha256"],
        "states": len(report_rows),
        "all_dataset_outcome_cells": all(
            any(row["dataset"] == dataset and bool(row["dense_wrong"]) == wrong for row in report_rows)
            for dataset in ("gqa", "chartqa", "textvqa")
            for wrong in (False, True)
        ),
        "boundary_layers": sorted({int(row["layer"]) for row in report_rows if int(row["layer"]) in config["validation"]["require_horizon_boundary_layers"]}),
        "fresh_cache_repeat_exact": all(row["fresh_cache_repeat_exact"] for row in report_rows),
        "swapped_order_exact": all(row["swapped_order_exact"] for row in report_rows if 8 in row["eligible_horizons"]),
        "h1_phase82_exact": all(row["h1_phase82_exact"] for row in report_rows),
        "on_canonical_exact": all(row["on_canonical_exact"] for row in report_rows),
        "action_trace_exact": all(row["action_trace_exact"] for row in report_rows),
    }
    report["passed"] = all(value for key, value in report.items() if key not in {"schema_version", "contract_sha256", "states", "boundary_layers"}) and report["boundary_layers"] == list(config["validation"]["require_horizon_boundary_layers"])
    if not report["passed"]:
        raise RuntimeError(f"short-horizon smoke failed: {report}")
    atomic_jsonl(output_root / "smoke/smoke_rows.jsonl", report_rows)
    atomic_json(output_root / "smoke/smoke_report.json", report)
    print(json.dumps(report, sort_keys=True))


def _require_smoke(contract: Mapping[str, Any], output_root: Path) -> None:
    report = read_json(output_root / "smoke/smoke_report.json")
    if report.get("contract_sha256") != contract["contract_sha256"] or not report.get("passed"):
        raise RuntimeError("bound short-horizon smoke has not passed")


def extraction_worker(config_path: Path, rank: int, device_index: int, *, resume: bool) -> None:
    contract, output_root, _ = verify_contract(config_path, verify_model=True)
    _require_smoke(contract, output_root)
    config = contract["static_config"]
    configure_dense_determinism(int(config["seed"]) + int(rank), config["backend_settings"])
    torch.cuda.set_device(int(device_index))
    device = torch.device(f"cuda:{int(device_index)}")
    processor, _base, wrapped = _load_model(config, device)
    internal = _internal_index(config)
    oracle = _phase82_oracle(config)
    rows = read_jsonl(output_root / "population/state_manifest.jsonl")
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_uid[str(row["uid"])].append(row)
    schedule = [row for row in read_jsonl(output_root / "work/extraction_schedule.jsonl") if int(row["worker_rank"]) == int(rank)]
    maps = _open_maps(contract["cache_layout"])
    rank_root = output_root / f"work/extraction/rank{int(rank):02d}"
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
        baseline = capture_four_action_route(
            wrapped, {}, ["FULL"] * 28, prepared_inputs=prepared, use_cache=True, native_full_rows=True
        )
        payloads = {}
        state_results = []
        for row in sorted(by_uid[uid], key=lambda value: int(value["layer"])):
            source_file = str(row["pre_state_file"])
            if source_file not in payloads:
                payloads[source_file] = torch.load(resolve_path(source_file), map_location="cpu", weights_only=False)
            source_state = payloads[source_file]["states"].get(str(row["state_id"]))
            if source_state is None:
                raise RuntimeError(f"source state missing: {row['state_id']}")
            expected_pre = _cached_state(source_state["text_states"], source_state["visual_states"], source_state["text_mask"], source_state["visual_mask"])
            live_text, live_visual = baseline.pre_layer_states[int(row["layer"])]
            live_pre = _cached_state(live_text, live_visual, prepared.text_valid_mask, prepared.visual_valid_mask)
            if expected_pre["state_sha256"] != row["pre_state_sha256"] or live_pre["state_sha256"] != row["pre_state_sha256"]:
                raise RuntimeError(f"pre-state differs: {row['state_id']}")
            branches = []
            for branch in BRANCHES:
                captures = rollout_branch(
                    wrapped,
                    baseline,
                    prepared,
                    layer=int(row["layer"]),
                    branch=branch,
                    horizons=row["eligible_horizons"],
                )
                horizon_rows = []
                for horizon in row["eligible_horizons"]:
                    horizon = int(horizon)
                    capture = captures[horizon]
                    validation = _validate_capture(row, branch, horizon, capture, oracle, baseline, prepared)
                    _write_capture(maps, row, branch, horizon, capture)
                    horizon_rows.append({
                        "horizon": horizon,
                        "reached_layer": int(row["layer"]) + horizon - 1,
                        "text_sha256": capture["text_sha256"],
                        "visual_sha256": capture["visual_sha256"],
                        "pooled_sha256": capture["pooled_sha256"],
                        "state_sha256": capture["state_sha256"],
                        "action_trace": capture["action_trace"],
                        **validation,
                    })
                    del capture
                branches.append({"branch": branch, "first_action": BRANCH_ACTION[branch], "horizons": horizon_rows})
                del captures
            state_results.append({"state_id": row["state_id"], "layer": row["layer"], "pre_state_sha256": row["pre_state_sha256"], "branches": branches})
        for horizon_maps in maps.values():
            for value in horizon_maps.values():
                value.flush()
        result = {
            "schema_version": "read_short_horizon_uid_rollout_v1",
            "contract_sha256": contract["contract_sha256"],
            "rank": int(rank),
            "uid": uid,
            "state_ids": item["state_ids"],
            "image_sha256": metadata["consumed_image_sha256"],
            "state_results": state_results,
        }
        atomic_json(result_path, result)
        completed += 1
        del baseline, prepared, inputs, state_results, payloads
        torch.cuda.empty_cache()
        if completed % 5 == 0 or completed == len(schedule):
            print(json.dumps({"rank": rank, "completed_uids": completed, "assigned_uids": len(schedule), "elapsed_seconds": time.monotonic() - started}), flush=True)
    atomic_json(rank_root / "complete.json", {
        "contract_sha256": contract["contract_sha256"],
        "rank": int(rank),
        "expected_uids": len(schedule),
        "completed_uids": completed,
        "elapsed_seconds": time.monotonic() - started,
    })


def finalize_extraction(config_path: Path) -> None:
    contract, output_root, _ = verify_contract(config_path)
    _require_smoke(contract, output_root)
    config = contract["static_config"]
    rows = read_jsonl(output_root / "population/state_manifest.jsonl")
    expected = {str(row["state_id"]): row for row in rows}
    uid_results = []
    for rank in range(int(config["world_size"])):
        root = output_root / f"work/extraction/rank{rank:02d}"
        complete = read_json(root / "complete.json")
        if complete.get("contract_sha256") != contract["contract_sha256"] or complete.get("expected_uids") != complete.get("completed_uids"):
            raise RuntimeError(f"partial extraction rank {rank}")
        uid_results.extend(read_json(path) for path in sorted(root.glob("*.json")) if path.name != "complete.json")
    observed_uids = [str(row["uid"]) for row in uid_results]
    expected_uids = {str(row["uid"]) for row in rows}
    if len(observed_uids) != len(set(observed_uids)) or set(observed_uids) != expected_uids:
        raise RuntimeError("global UID rollout census differs")
    results = {}
    branch_rows = []
    rollout_rows = []
    action_rows = []
    parity_rows = []
    for uid_result in uid_results:
        for state in uid_result["state_results"]:
            state_id = str(state["state_id"])
            if state_id in results:
                raise RuntimeError(f"duplicate rollout state: {state_id}")
            results[state_id] = state
            rollout_rows.append({
                "contract_sha256": contract["contract_sha256"],
                "state_id": state_id,
                "uid": uid_result["uid"],
                "layer": state["layer"],
                "pre_state_sha256": state["pre_state_sha256"],
                "image_sha256": uid_result["image_sha256"],
                "branches": state["branches"],
            })
            for branch in state["branches"]:
                for horizon in branch["horizons"]:
                    record = {
                        "contract_sha256": contract["contract_sha256"],
                        "state_id": state_id,
                        "uid": uid_result["uid"],
                        "layer": state["layer"],
                        "branch": branch["branch"],
                        "first_action": branch["first_action"],
                        **horizon,
                    }
                    branch_rows.append(record)
                    action_rows.append({
                        "state_id": state_id,
                        "uid": uid_result["uid"],
                        "layer": state["layer"],
                        "branch": branch["branch"],
                        "horizon": horizon["horizon"],
                        "reached_layer": horizon["reached_layer"],
                        "action_trace": "|".join(horizon["action_trace"]),
                        "trace_exact": horizon["trace_exact"],
                    })
                    parity_rows.append({
                        "state_id": state_id,
                        "uid": uid_result["uid"],
                        "branch": branch["branch"],
                        "horizon": horizon["horizon"],
                        "phase82_h1_exact": horizon["phase82_h1_exact"],
                        "on_canonical_exact": horizon["on_canonical_exact"],
                        "passed": horizon["trace_exact"] and (horizon["phase82_h1_exact"] is not False) and (horizon["on_canonical_exact"] is not False),
                    })
    if set(results) != set(expected):
        raise RuntimeError("global state rollout census differs")
    validate_horizon_census(rows, branch_rows, num_layers=int(config["model"]["decoder_layers"]))
    maps = _open_maps(contract["cache_layout"], mode="r")
    result_index = {(row["state_id"], int(row["horizon"]), row["branch"]): row for row in branch_rows}
    cache_layout = dict(contract["cache_layout"])
    for horizon in HORIZONS:
        selected = sorted(
            [row for row in rows if horizon in row["eligible_horizons"]],
            key=lambda row: int(row["horizon_indices"][str(horizon)]["row_index"]),
        )
        for row in selected:
            index = row["horizon_indices"][str(horizon)]
            ri, vo, nv = int(index["row_index"]), int(index["visual_offset"]), int(row["visual_tokens"])
            for branch in BRANCHES:
                bi = BRANCH_INDEX[branch]
                record = result_index[(str(row["state_id"]), horizon, branch)]
                pooled = _read_bf16(maps[horizon]["pooled"], (ri, bi))
                text = _read_bf16(maps[horizon]["text"], (ri, bi)).reshape(1, 1, -1)
                visual = _read_bf16(maps[horizon]["visual"], (slice(vo, vo + nv), bi, slice(None))).unsqueeze(0)
                if tensor_sha256(pooled) != record["pooled_sha256"] or tensor_sha256(text) != record["text_sha256"] or tensor_sha256(visual) != record["visual_sha256"]:
                    raise RuntimeError(f"stored horizon tensor differs: {row['state_id']} H={horizon} {branch}")
        horizon_layout = dict(cache_layout["horizons"][str(horizon)])
        horizon_layout["files"] = {
            name: {**dict(spec), "sha256": file_sha256(spec["path"])}
            for name, spec in horizon_layout["files"].items()
        }
        cache_layout["horizons"][str(horizon)] = horizon_layout
    cache_layout["contract_sha256"] = contract["contract_sha256"]
    atomic_json(output_root / "features/cache_manifest.json", cache_layout)
    atomic_jsonl(output_root / "branches/rollout_manifest.jsonl", rollout_rows)
    atomic_jsonl(output_root / "branches/horizon_state_manifest.jsonl", branch_rows)
    atomic_csv(output_root / "branches/action_trace_validation.csv", action_rows)
    atomic_csv(output_root / "branches/state_hash_parity.csv", parity_rows)
    feature_rows = []
    for row in rows:
        for horizon in row["eligible_horizons"]:
            feature_rows.append({
                "contract_sha256": contract["contract_sha256"],
                "state_id": row["state_id"],
                "uid": row["uid"],
                "image_group_id": row["image_group_id"],
                "dataset": row["dataset"],
                "source_regime": row["source_regime"],
                "dense_correct": row["dense_correct"],
                "dense_wrong": row["dense_wrong"],
                "layer": row["layer"],
                "trigger_layer": row["trigger_layer"],
                "trigger_relative_depth": row["trigger_relative_depth"],
                "cohort": row["cohort"],
                "h_r": row["h_r"],
                "horizon": horizon,
                "common_h8": row["common_h8"],
                "horizon_row_index": row["horizon_indices"][str(horizon)]["row_index"],
                "visual_offset": row["horizon_indices"][str(horizon)]["visual_offset"],
                "visual_tokens": row["visual_tokens"],
                "text_tokens": 1,
                "cache_manifest": "features/cache_manifest.json",
            })
    atomic_jsonl(output_root / "features/pooled_horizon_features.jsonl", feature_rows)
    atomic_jsonl(output_root / "features/delta_horizon_features.jsonl", feature_rows)
    atomic_jsonl(output_root / "features/token_horizon_features.jsonl", feature_rows)
    atomic_json(output_root / "work/extraction_complete.json", {
        "contract_sha256": contract["contract_sha256"],
        "states": len(rows),
        "horizon_state_pairs": len(feature_rows),
        "branch_records": len(branch_rows),
        "cache_manifest_sha256": file_sha256(output_root / "features/cache_manifest.json"),
        "complete": True,
    })
    print(json.dumps({"complete": True, "states": len(rows), "horizon_state_pairs": len(feature_rows), "branch_records": len(branch_rows)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "smoke", "extract", "finalize-extraction"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config)
    elif args.command == "smoke":
        smoke(args.config, args.device_index)
    elif args.command == "extract":
        extraction_worker(args.config, args.rank, args.device_index, resume=args.resume)
    else:
        finalize_extraction(args.config)


if __name__ == "__main__":
    main()
