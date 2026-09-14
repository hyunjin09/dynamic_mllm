#!/usr/bin/env python3
"""Audit frozen Stage-2 expert-state fit versus closed-loop state drift."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from binary_policy.executor import capture_four_action_route  # noqa: E402
from binary_policy.executor.four_action import capture_online_four_action_route  # noqa: E402
from binary_policy.executor.inputs import build_binary_inputs  # noqa: E402
from dense_failure_stage1.runtime import build_dense_inputs, configure_dense_determinism  # noqa: E402
from dense_failure_stage2.closed_loop_trajectory_set import (  # noqa: E402
    build_trajectory_sets,
    compact_router_state,
)
from dense_failure_stage2.teacher_forced_free_run import (  # noqa: E402
    action_diagnostics,
    classify_bottleneck,
    first_non_full_offset,
    prefix_support_trace,
    score_cached_uid,
    select_reference_programs,
    support_survival,
)
from dense_failure_stage2.v1_router import ACTION_NAMES, SharedReadWriteRouter  # noqa: E402
from experiments.run_closed_loop_trajectory_set import (  # noqa: E402
    _atomic_bytes,
    _combined_state_hash,
    _git_state,
    _load_model,
    _load_uid_payload,
    _runtime_state,
    _state_tensor_hashes,
    _uid_slug,
    _verify_model_snapshot,
    atomic_csv,
    atomic_json,
    atomic_jsonl,
    canonical_hash,
    file_sha256,
    read_json,
    read_jsonl,
    resolve_path,
    tensor_sha256,
    train_generate,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/teacher_forced_free_run_audit_v1.json"
BOUND_CODE = (
    "configs/teacher_forced_free_run_audit_v1.json",
    "dense_failure_stage2/teacher_forced_free_run.py",
    "experiments/audit_teacher_forced_free_run.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/model.py",
    "dense_failure_stage1/lmms_scoring.py",
    "dense_failure_stage1/runtime.py",
    "dense_failure_stage2/closed_loop_trajectory_set.py",
    "dense_failure_stage2/v1_router.py",
    "experiments/run_closed_loop_trajectory_set.py",
    "experiments/run_stage2_v1_training_revised.py",
    "plans/teacher_forced_vs_free_run_divergence_audit_plan.md",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "teacher_forced_free_run_audit_config_v1":
        raise ValueError("unsupported teacher-forced/free-run audit config")
    if int(config.get("world_size", 0)) != 4:
        raise ValueError("the audit requires all four direct GPUs")
    expected = {
        "uids": 569,
        "dense_c_uids": 106,
        "dense_w_uids": 463,
        "programs": 4948,
        "unique_states": 35565,
        "route_state_occurrences": 69178,
        "internal_train_uids": 454,
        "internal_dev_uids": 115,
    }
    if {key: int(config["population"][key]) for key in expected} != expected:
        raise ValueError("audit population differs from the approved plan")
    return config


def _source_paths(config: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    parent = config["parent"]
    checkpoints = config["checkpoints"]
    return {
        "parent_frozen_protocol": (parent["frozen_protocol"], parent["frozen_protocol_sha256"]),
        "parent_artifact_manifest": (parent["artifact_manifest"], parent["artifact_manifest_sha256"]),
        "phase76_config": (parent["phase76_config"], parent["phase76_config_sha256"]),
        "trajectory_manifest": (parent["trajectory_manifest"], parent["trajectory_manifest_sha256"]),
        "uid_state_files": (parent["uid_state_files"], parent["uid_state_files_sha256"]),
        "state_manifest": (parent["state_manifest"], parent["state_manifest_sha256"]),
        "replay_completion": (parent["replay_completion"], parent["replay_completion_sha256"]),
        "internal_split": (parent["internal_split"], parent["internal_split_sha256"]),
        "full_checkpoint": (checkpoints["full_refit"]["path"], checkpoints["full_refit"]["sha256"]),
        "full_checkpoint_manifest": (
            checkpoints["full_refit"]["manifest"], checkpoints["full_refit"]["manifest_sha256"]
        ),
        "internal_checkpoint": (checkpoints["internal_dev"]["path"], checkpoints["internal_dev"]["sha256"]),
        "internal_selection": (
            checkpoints["internal_dev"]["selection"], checkpoints["internal_dev"]["selection_sha256"]
        ),
    }


def _validate_parent(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    for name, (path, expected) in _source_paths(config).items():
        observed = file_sha256(path)
        if observed != expected:
            raise RuntimeError(f"frozen source differs: {name}: {observed} != {expected}")
    parent_contract = read_json(config["parent"]["frozen_protocol"])
    if (
        parent_contract.get("contract_sha256") != config["parent"]["contract_sha256"]
        or canonical_hash(parent_contract) != parent_contract["contract_sha256"]
    ):
        raise RuntimeError("Phase-76 parent contract differs")
    for path, expected in parent_contract["bound_code_sha256"].items():
        if file_sha256(path) != expected:
            raise RuntimeError(f"Phase-76 bound code changed after its run: {path}")
    parent_root = resolve_path(config["parent"]["output_root"])
    for relative, expected in parent_contract["internal_manifest_sha256"].items():
        if file_sha256(parent_root / relative) != expected:
            raise RuntimeError(f"Phase-76 internal manifest changed after its run: {relative}")
    replay = read_json(config["parent"]["replay_completion"])
    if (
        replay.get("contract_sha256") != parent_contract["contract_sha256"]
        or int(replay.get("quarantined_programs", -1)) != 0
        or not all(replay.get(key) for key in (
            "all_exact_token_parity", "all_final_correct",
            "all_parent_trigger_state_exact", "all_compact_full_router_logits_exact"
        ))
    ):
        raise RuntimeError("Phase-76 replay completion is not valid")
    rows = read_jsonl(config["parent"]["trajectory_manifest"])
    grouped, census = build_trajectory_sets(rows, total_layers=28)
    expected = config["population"]
    comparison = {
        "uids": int(expected["uids"]),
        "programs": int(expected["programs"]),
        "dense_c_uids": int(expected["dense_c_uids"]),
        "dense_w_uids": int(expected["dense_w_uids"]),
        "route_state_occurrences": int(expected["route_state_occurrences"]),
        "unique_prefix_states": int(expected["unique_states"]),
    }
    if census != comparison:
        raise RuntimeError(f"Phase-76 corpus census differs: {census}")
    split = read_json(config["parent"]["internal_split"])
    uid_split = {str(uid): str(value) for uid, value in split["uid_split"].items()}
    if set(uid_split) != set(grouped) or Counter(uid_split.values()) != Counter({
        "train": int(expected["internal_train_uids"]),
        "dev": int(expected["internal_dev_uids"]),
    }):
        raise RuntimeError("frozen internal split differs")
    for view, details in config["checkpoints"].items():
        checkpoint = torch.load(resolve_path(details["path"]), map_location="cpu", weights_only=False)
        required_mode = "full" if view == "full_refit" else "internal"
        if checkpoint.get("contract_sha256") != parent_contract["contract_sha256"] or checkpoint.get("mode") != required_mode:
            raise RuntimeError(f"{view} checkpoint provenance differs")
    return parent_contract, split, grouped


def _schedule_rows(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    split: Mapping[str, Any],
    *,
    view: str,
    world_size: int,
) -> list[dict[str, Any]]:
    if view == "seen":
        uids = sorted(grouped)
    elif view == "heldout":
        uids = sorted(uid for uid in grouped if split["uid_split"][uid] == "dev")
    else:
        raise ValueError(f"unsupported view: {view}")
    return [
        {
            "uid": uid,
            "worker_rank": index % int(world_size),
            "dataset": grouped[uid][0]["dataset"],
            "dense_outcome": grouped[uid][0]["dense_outcome"],
            "trigger_layer": int(grouped[uid][0]["trigger_layer"]),
        }
        for index, uid in enumerate(uids)
    ]


def prepare(config_path: Path) -> None:
    config = load_config(config_path)
    parent_contract, split, grouped = _validate_parent(config)
    output_root = resolve_path(config["output_root"])
    if output_root.exists():
        raise RuntimeError("refusing to overwrite an existing audit output root")
    runtime = _runtime_state()
    if not runtime["cuda_available"] or int(runtime["cuda_device_count"]) != 4:
        raise RuntimeError("the approved audit requires four visible CUDA GPUs")
    phase76 = read_json(config["parent"]["phase76_config"])
    _verify_model_snapshot(
        resolve_path(phase76["model"]["snapshot_path"]),
        parent_contract["model_snapshot_sha256"],
    )
    output_root.mkdir(parents=True, exist_ok=False)
    for relative in (
        "contracts", "teacher_forced", "free_run", "release", "generalization",
        "metrics", "figures", "summaries", "work/teacher_forced/seen",
        "work/teacher_forced/heldout", "work/rollout/seen", "work/rollout/heldout", "smoke",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    schedules = {}
    for view in ("seen", "heldout"):
        schedule = _schedule_rows(
            grouped, split, view=view, world_size=int(config["world_size"])
        )
        path = output_root / f"work/{view}_schedule.jsonl"
        atomic_jsonl(path, schedule)
        schedules[view] = {"path": str(path.relative_to(output_root)), "sha256": file_sha256(path), "uids": len(schedule)}

    contract = {
        "schema_version": "teacher_forced_free_run_audit_contract_v1",
        "created_at": utc_now(),
        "static_config": config,
        "config_sha256": file_sha256(config_path),
        "parent_contract_sha256": parent_contract["contract_sha256"],
        "source_sha256": {name: expected for name, (_path, expected) in _source_paths(config).items()},
        "bound_code_sha256": {path: file_sha256(path) for path in BOUND_CODE},
        "schedule_manifest": schedules,
        "git": _git_state(),
        "runtime": runtime,
        "population": dict(config["population"]),
        "external_label_firewall": "No Phase-69/76 external evaluation row or label is an audit input.",
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_protocol.json", contract)
    checkpoint_contract = {
        "schema_version": "teacher_forced_free_run_checkpoint_contract_v1",
        "contract_sha256": contract["contract_sha256"],
        "parent_contract_sha256": parent_contract["contract_sha256"],
        "seen": config["checkpoints"]["full_refit"],
        "heldout": config["checkpoints"]["internal_dev"],
    }
    atomic_json(output_root / "contracts/checkpoint_contract.json", checkpoint_contract)
    atomic_json(output_root / "contracts/corpus_contract.json", {
        "schema_version": "teacher_forced_free_run_corpus_contract_v1",
        "contract_sha256": contract["contract_sha256"],
        "population": config["population"],
        "trajectory_manifest_sha256": config["parent"]["trajectory_manifest_sha256"],
        "uid_state_files_sha256": config["parent"]["uid_state_files_sha256"],
        "state_manifest_sha256": config["parent"]["state_manifest_sha256"],
        "replay_completion_sha256": config["parent"]["replay_completion_sha256"],
        "internal_split_sha256": config["parent"]["internal_split_sha256"],
    })
    _atomic_bytes(output_root / "contracts/internal_dev_availability.md", (
        "# Frozen internal-dev availability\n\n"
        f"- Available: **yes**\n- Frozen dev UIDs: **{schedules['heldout']['uids']}**\n"
        f"- Split SHA-256: `{config['parent']['internal_split_sha256']}`\n"
        f"- Checkpoint SHA-256: `{config['checkpoints']['internal_dev']['sha256']}`\n"
        "- The split is reused exactly; no post-result reconstruction is permitted.\n"
    ).encode())
    _atomic_bytes(output_root / "protocol.md", (
        "# Teacher-forced versus free-run divergence audit\n\n"
        f"- Contract: `{contract['contract_sha256']}`\n"
        "- Seen view: final full-refit checkpoint on all 569 training-side UIDs.\n"
        "- Held-out view: exact frozen internal-dev checkpoint on the frozen 115 image-group-disjoint dev UIDs.\n"
        "- Teacher forcing: score every exact cached route-state occurrence without executing the prediction.\n"
        "- R0/R1/R2: free from trigger; force before first corrective action; force through first corrective action.\n"
        "- No training, search, relabeling, threshold change, or external evaluation is part of this phase.\n"
    ).encode())
    print(json.dumps({"contract_sha256": contract["contract_sha256"], "schedules": schedules}, sort_keys=True))


def verify_contract(config_path: Path, *, verify_model: bool = False) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if contract.get("contract_sha256") != canonical_hash(contract):
        raise RuntimeError("audit contract hash differs")
    if contract.get("static_config") != config or contract.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError("audit config changed after freeze")
    for path, expected in contract["bound_code_sha256"].items():
        if file_sha256(path) != expected:
            raise RuntimeError(f"bound audit code changed after freeze: {path}")
    for name, (path, expected) in _source_paths(config).items():
        if file_sha256(path) != expected or contract["source_sha256"].get(name) != expected:
            raise RuntimeError(f"frozen audit input changed: {name}")
    if _git_state() != contract["git"]:
        raise RuntimeError("git commit/branch/worktree status differs from the audit contract")
    if _runtime_state() != contract["runtime"]:
        raise RuntimeError("runtime/environment differs from the audit contract")
    for view, details in contract["schedule_manifest"].items():
        path = output_root / details["path"]
        if file_sha256(path) != details["sha256"] or len(read_jsonl(path)) != int(details["uids"]):
            raise RuntimeError(f"{view} schedule differs")
    parent_contract = read_json(config["parent"]["frozen_protocol"])
    if parent_contract.get("contract_sha256") != contract["parent_contract_sha256"]:
        raise RuntimeError("parent contract differs")
    if verify_model:
        phase76 = read_json(config["parent"]["phase76_config"])
        _verify_model_snapshot(resolve_path(phase76["model"]["snapshot_path"]), parent_contract["model_snapshot_sha256"])
    return contract, parent_contract, output_root


def _router_for_checkpoint(
    phase76: Mapping[str, Any], checkpoint_path: str, expected_sha256: str,
    parent_contract_sha256: str, device: torch.device,
) -> SharedReadWriteRouter:
    if file_sha256(checkpoint_path) != expected_sha256:
        raise RuntimeError("router checkpoint bytes differ")
    settings = phase76["stage2"]
    router = SharedReadWriteRouter(
        hidden_size=int(phase76["model"]["hidden_size"]),
        router_size=int(settings["router_size"]),
        num_heads=int(settings["num_heads"]),
        dropout=float(settings["dropout"]),
    ).to(device=device, dtype=torch.float32)
    checkpoint = torch.load(resolve_path(checkpoint_path), map_location="cpu", weights_only=False)
    if checkpoint.get("contract_sha256") != parent_contract_sha256:
        raise RuntimeError("router checkpoint belongs to another Phase-76 contract")
    router.load_state_dict(checkpoint["state_dict"], strict=True)
    return router.eval()


def _uid_index(config: Mapping[str, Any], parent_contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(config["parent"]["uid_state_files"])
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        uid = str(row["uid"])
        if uid in index or row.get("contract_sha256") != parent_contract["contract_sha256"]:
            raise RuntimeError("UID state-file index has duplicate or invalid provenance")
        index[uid] = row
    if len(index) != int(config["population"]["uids"]):
        raise RuntimeError("UID state-file index is incomplete")
    return index


def _view_checkpoint(config: Mapping[str, Any], view: str) -> tuple[str, str]:
    if view == "seen":
        details = config["checkpoints"]["full_refit"]
    elif view == "heldout":
        details = config["checkpoints"]["internal_dev"]
    else:
        raise ValueError(f"unsupported view: {view}")
    return str(details["path"]), str(details["sha256"])


def teacher_worker(config_path: Path, *, view: str, rank: int, resume: bool) -> None:
    contract, parent_contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    if rank < 0 or rank >= int(config["world_size"]):
        raise ValueError("invalid worker rank")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    checkpoint_path, checkpoint_sha = _view_checkpoint(config, view)
    phase76 = read_json(config["parent"]["phase76_config"])
    router = _router_for_checkpoint(
        phase76, checkpoint_path, checkpoint_sha, parent_contract["contract_sha256"], device
    )
    schedule = [
        row for row in read_jsonl(output_root / f"work/{view}_schedule.jsonl")
        if int(row["worker_rank"]) == rank
    ]
    index = _uid_index(config, parent_contract)
    parent_root = resolve_path(config["parent"]["output_root"])
    rank_root = output_root / f"work/teacher_forced/{view}/rank{rank:02d}"
    completed = 0
    started = time.monotonic()
    for item in schedule:
        uid = str(item["uid"])
        path = rank_root / f"{_uid_slug(uid)}.json"
        if resume and path.exists():
            old = read_json(path)
            if (
                old.get("contract_sha256") == contract["contract_sha256"]
                and old.get("checkpoint_sha256") == checkpoint_sha
                and old.get("uid") == uid
                and old.get("state_file_sha256") == index[uid]["state_file_sha256"]
            ):
                completed += 1
                continue
        payload = _load_uid_payload(parent_contract, parent_root, index[uid])
        scored = score_cached_uid(
            router, payload, device=device,
            state_microbatch=int(config["teacher_forced"]["state_microbatch"]),
        )
        result = {
            "schema_version": "teacher_forced_uid_result_v1",
            "contract_sha256": contract["contract_sha256"],
            "checkpoint_sha256": checkpoint_sha,
            "view": view,
            "uid": uid,
            "dataset": payload["dataset"],
            "source_regime": payload["source_regime"],
            "dense_outcome": payload["dense_outcome"],
            "image_group_id": payload["image_group_id"],
            "trigger_layer": int(payload["trigger_layer"]),
            "state_file_sha256": index[uid]["state_file_sha256"],
            "unique_state_count": len(scored["state_logits"]),
            "program_count": len(scored["programs"]),
            "occurrences": scored["occurrences"],
            "programs": scored["programs"],
        }
        atomic_json(path, result)
        completed += 1
        del payload, scored, result
        if completed % 10 == 0 or completed == len(schedule):
            print(json.dumps({
                "view": view, "rank": rank, "completed": completed,
                "total": len(schedule), "elapsed_seconds": time.monotonic() - started,
            }), flush=True)
    atomic_json(rank_root / "complete.json", {
        "schema_version": "teacher_forced_rank_complete_v1",
        "contract_sha256": contract["contract_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "view": view,
        "rank": rank,
        "expected_uids": len(schedule),
        "completed_uids": completed,
        "completed_at": utc_now(),
    })


def _collect_results(
    contract: Mapping[str, Any], output_root: Path, *, kind: str, view: str
) -> list[dict[str, Any]]:
    config = contract["static_config"]
    checkpoint_path, checkpoint_sha = _view_checkpoint(config, view)
    del checkpoint_path
    schedule = read_jsonl(output_root / f"work/{view}_schedule.jsonl")
    expected = [str(row["uid"]) for row in schedule]
    results = []
    for rank in range(int(config["world_size"])):
        rank_root = output_root / f"work/{kind}/{view}/rank{rank:02d}"
        completion = read_json(rank_root / "complete.json")
        expected_rank = sum(int(row["worker_rank"]) == rank for row in schedule)
        if (
            completion.get("contract_sha256") != contract["contract_sha256"]
            or completion.get("checkpoint_sha256") != checkpoint_sha
            or completion.get("view") != view
            or int(completion.get("expected_uids", -1)) != expected_rank
            or int(completion.get("completed_uids", -1)) != expected_rank
        ):
            raise RuntimeError(f"{kind} {view} rank {rank} is incomplete")
        results.extend(
            read_json(path) for path in sorted(rank_root.glob("[0-9a-f]*.json"))
            if path.name != "complete.json"
        )
    observed = Counter(str(row["uid"]) for row in results)
    if observed != Counter(expected) or any(count != 1 for count in observed.values()):
        raise RuntimeError(f"{kind} {view} global UID completion differs")
    by_uid = {str(row["uid"]): row for row in results}
    return [by_uid[uid] for uid in expected]


def _mean(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _median(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _metric_row(scope: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target_probs = [float(row["target_probability"]) for row in rows]
    margins = [float(row["target_vs_full_margin"]) for row in rows]
    return {
        "scope": scope,
        "n": len(rows),
        "top1_correct": sum(bool(row["target_top1"]) for row in rows),
        "top1_recall": _rate(sum(bool(row["target_top1"]) for row in rows), len(rows)),
        "mean_target_probability": _mean(target_probs),
        "median_target_probability": _median(target_probs),
        "mean_target_vs_full_margin": _mean(margins),
        "median_target_vs_full_margin": _median(margins),
        "mean_target_rank": _mean([float(row["target_rank"]) for row in rows]),
    }


def _first_nonfull_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for result in results:
        if result["dense_outcome"] != "W":
            continue
        by_program: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in result["occurrences"]:
            by_program[str(row["program_id"])].append(row)
        for program_id, rows in by_program.items():
            current = min(
                (row for row in rows if row["target_action"] != "FULL"),
                key=lambda row: int(row["depth_after_trigger"]),
            )
            output.append(dict(current))
    return output


def aggregate_teacher(config_path: Path) -> None:
    contract, _parent, output_root = verify_contract(config_path)
    config = contract["static_config"]
    seen = _collect_results(contract, output_root, kind="teacher_forced", view="seen")
    heldout = _collect_results(contract, output_root, kind="teacher_forced", view="heldout")
    occurrences = [row for result in seen for row in result["occurrences"]]
    programs = [row for result in seen for row in result["programs"]]
    if (
        len(seen) != int(config["population"]["uids"])
        or len(occurrences) != int(config["population"]["route_state_occurrences"])
        or len(programs) != int(config["population"]["programs"])
        or sum(int(result["unique_state_count"]) for result in seen) != int(config["population"]["unique_states"])
    ):
        raise RuntimeError("seen teacher-forced census differs")
    checkpoint_sha = config["checkpoints"]["full_refit"]["sha256"]
    atomic_jsonl(output_root / "teacher_forced/all_route_state_logits.jsonl", (
        {"schema_version": "teacher_forced_route_state_logit_v1", "contract_sha256": contract["contract_sha256"],
         "checkpoint_sha256": checkpoint_sha, "view": "seen", **row}
        for row in occurrences
    ))

    action_rows = [_metric_row("all_actions", occurrences)]
    for action in ACTION_NAMES:
        action_rows.append(_metric_row(action, [row for row in occurrences if row["target_action"] == action]))
    action_rows.append(_metric_row("nonFULL", [row for row in occurrences if row["target_action"] != "FULL"]))
    atomic_csv(output_root / "teacher_forced/action_class_metrics.csv", action_rows)

    first_rows = _first_nonfull_rows(seen)
    first_metrics = [_metric_row("all_successful_routes", first_rows)]
    for layer in sorted({int(row["layer"]) for row in first_rows}):
        first_metrics.append(_metric_row(f"layer_{layer}", [row for row in first_rows if int(row["layer"]) == layer]))
    for provenance in config["provenance_groups"]:
        subset = [row for row in first_rows if provenance in row["provenance"]]
        if subset:
            first_metrics.append(_metric_row(f"provenance_{provenance}", subset))
    atomic_csv(output_root / "teacher_forced/first_nonfull_metrics.csv", first_metrics)

    responsibility = [row for row in programs if any(item["dense_outcome"] == "W" and item["uid"] == row["uid"] for item in seen)]
    selected_ids = select_reference_programs(responsibility)
    occurrence_index = {(row["uid"], row["program_id"], int(row["depth_after_trigger"])): row for row in occurrences}
    result_index = {str(row["uid"]): row for row in seen}
    selected_rows = []
    for program in programs:
        uid = str(program["uid"])
        if selected_ids.get(uid) != program["program_id"]:
            continue
        result = result_index[uid]
        offset = first_non_full_offset(program["actions"])
        selected_rows.append({
            "schema_version": "teacher_forced_selected_reference_route_v1",
            "contract_sha256": contract["contract_sha256"],
            "checkpoint_sha256": checkpoint_sha,
            "uid": uid,
            "dataset": result["dataset"],
            "source_regime": result["source_regime"],
            "image_group_id": result["image_group_id"],
            "trigger_layer": int(result["trigger_layer"]),
            "program_id": program["program_id"],
            "actions": program["actions"],
            "provenance": program["provenance"],
            "route_logp": program["route_logp"],
            "responsibility": program["responsibility"],
            "first_nonfull_offset": offset,
            "first_nonfull_layer": int(result["trigger_layer"]) + offset,
        })
    if len(selected_rows) != int(config["population"]["dense_w_uids"]):
        raise RuntimeError("reference-route selection does not cover every W UID")
    atomic_jsonl(output_root / "release/selected_reference_routes.jsonl", selected_rows)
    selected_keys = {(row["uid"], row["program_id"]) for row in selected_rows}
    selected_occurrences = [row for row in occurrences if (row["uid"], row["program_id"]) in selected_keys]
    selected_first = [row for row in first_rows if (row["uid"], row["program_id"]) in selected_keys]
    highest = [
        _metric_row("highest_responsibility_all", selected_occurrences),
        _metric_row("highest_responsibility_FULL", [row for row in selected_occurrences if row["target_action"] == "FULL"]),
        _metric_row("highest_responsibility_nonFULL", [row for row in selected_occurrences if row["target_action"] != "FULL"]),
        _metric_row("highest_responsibility_first_nonFULL", selected_first),
    ]
    atomic_csv(output_root / "teacher_forced/highest_responsibility_route_metrics.csv", highest)

    layer_rows = []
    for layer in range(28):
        current = [row for row in occurrences if int(row["layer"]) == layer]
        if current:
            layer_rows.append({"layer_or_band": str(layer), **{k: v for k, v in _metric_row("", current).items() if k != "scope"}})
    for name, low, high in (("early", 0, 8), ("middle", 9, 18), ("late", 19, 27)):
        current = [row for row in occurrences if low <= int(row["layer"]) <= high]
        layer_rows.append({"layer_or_band": name, **{k: v for k, v in _metric_row("", current).items() if k != "scope"}})
    atomic_csv(output_root / "teacher_forced/layer_breakdown.csv", layer_rows)
    provenance_rows = []
    for provenance in config["provenance_groups"]:
        current = [row for row in occurrences if provenance in row["provenance"]]
        if current:
            provenance_rows.append(_metric_row(provenance, current))
    atomic_csv(output_root / "teacher_forced/provenance_breakdown.csv", provenance_rows)

    heldout_occurrences = [row for result in heldout for row in result["occurrences"]]
    heldout_first = _first_nonfull_rows(heldout)
    heldout_metrics = [
        _metric_row("all_actions", heldout_occurrences),
        _metric_row("FULL", [row for row in heldout_occurrences if row["target_action"] == "FULL"]),
        _metric_row("nonFULL", [row for row in heldout_occurrences if row["target_action"] != "FULL"]),
        _metric_row("first_nonFULL", heldout_first),
    ]
    for action in ACTION_NAMES[1:]:
        heldout_metrics.append(_metric_row(action, [row for row in heldout_occurrences if row["target_action"] == action]))
    atomic_csv(output_root / "generalization/heldout_teacher_forced_metrics.csv", heldout_metrics)
    atomic_json(output_root / "work/teacher_aggregation.json", {
        "schema_version": "teacher_forced_aggregation_v1",
        "contract_sha256": contract["contract_sha256"],
        "seen_uids": len(seen), "heldout_uids": len(heldout),
        "seen_occurrences": len(occurrences), "seen_programs": len(programs),
        "seen_unique_states": sum(int(row["unique_state_count"]) for row in seen),
        "selected_reference_routes": len(selected_rows),
        "required_sha256": {
            relative: file_sha256(output_root / relative) for relative in (
                "teacher_forced/all_route_state_logits.jsonl",
                "teacher_forced/action_class_metrics.csv",
                "teacher_forced/first_nonfull_metrics.csv",
                "teacher_forced/highest_responsibility_route_metrics.csv",
                "teacher_forced/layer_breakdown.csv",
                "teacher_forced/provenance_breakdown.csv",
                "release/selected_reference_routes.jsonl",
                "generalization/heldout_teacher_forced_metrics.csv",
            )
        },
    })
    print(json.dumps({"seen_uids": len(seen), "occurrences": len(occurrences), "heldout_uids": len(heldout), "selected": len(selected_rows)}, sort_keys=True))


def _compact_state_hash(text: torch.Tensor, visual: torch.Tensor, meta) -> str:
    compact = compact_router_state(text, visual, meta.text_valid_mask, meta.visual_valid_mask)
    cached = {
        name: value.detach().cpu().to(torch.bfloat16).contiguous()
        if name in {"text_states", "visual_states"}
        else value.detach().cpu().bool().contiguous()
        for name, value in compact.items()
    }
    return _combined_state_hash(_state_tensor_hashes(cached))


@torch.inference_mode()
def _run_policy(
    *, processor, wrapped, router, sample: Mapping[str, Any], prepared,
    input_ids: torch.Tensor,
    trigger: int, condition: str, reference: Mapping[str, Any] | None,
    payload: Mapping[str, Any], contract_sha256: str,
) -> dict[str, Any]:
    if condition == "R0":
        forced_count = 0
    elif condition == "R1":
        if reference is None:
            raise ValueError("R1 requires a reference route")
        forced_count = int(reference["first_nonfull_offset"])
    elif condition == "R2":
        if reference is None:
            raise ValueError("R2 requires a reference route")
        forced_count = int(reference["first_nonfull_offset"]) + 1
    else:
        raise ValueError(f"unsupported release condition: {condition}")
    expert_actions = list(reference["actions"]) if reference is not None else ["FULL"] * (28 - trigger)
    program = None
    if reference is not None:
        program = next(
            item for item in payload["programs"] if item["program_id"] == reference["program_id"]
        )
    trace: list[dict[str, Any]] = []
    release_state_exact = None

    def selector(layer, text_state, visual_state, meta):
        nonlocal release_state_exact
        if layer < trigger:
            return "FULL"
        offset = int(layer) - trigger
        forced = offset < forced_count
        logits = router(
            text_state, visual_state,
            text_mask=meta.text_valid_mask, visual_mask=meta.visual_valid_mask,
        )[0]
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"non-finite router logits for {sample['uid']} L{layer}")
        probabilities = logits.float().softmax(dim=-1)
        action = expert_actions[offset] if forced else ACTION_NAMES[int(probabilities.argmax().item())]
        current_hash = _compact_state_hash(text_state, visual_state, meta)
        expected_state_hash = None
        expert_state_exact = None
        if program is not None and offset < len(program["state_ids"]):
            state_id = str(program["state_ids"][offset])
            expected_state_hash = str(payload["states"][state_id]["state_sha256"])
            if offset <= forced_count:
                expert_state_exact = current_hash == expected_state_hash
                if not expert_state_exact:
                    raise RuntimeError(
                        f"forced prefix did not reach exact expert state: {sample['uid']} {condition} L{layer}"
                    )
        if offset == forced_count:
            release_state_exact = expert_state_exact
        trace.append({
            "layer": int(layer), "depth_after_trigger": offset,
            "forced": forced, "action": action,
            "logits": {name: float(logits[index].item()) for index, name in enumerate(ACTION_NAMES)},
            "probabilities": {name: float(probabilities[index].item()) for index, name in enumerate(ACTION_NAMES)},
            "state_sha256": current_hash,
            "expected_expert_state_sha256": expected_state_hash,
            "expert_state_exact": expert_state_exact,
        })
        return action

    output = capture_online_four_action_route(
        wrapped, {}, selector, prepared_inputs=prepared, use_cache=True, native_full_rows=True
    )
    ids, text, score = train_generate(
        processor, wrapped, output, input_ids, sample
    )
    actions = [row["action"] for row in trace]
    return {
        "schema_version": "teacher_forced_release_rollout_v1",
        "contract_sha256": contract_sha256,
        "condition": condition,
        "forced_action_count": forced_count,
        "release_layer": trigger + forced_count if trigger + forced_count < 28 else None,
        "release_state_exact": release_state_exact,
        "actions": actions,
        "trace": trace,
        "generated_token_ids": ids,
        "generated_answer": text,
        "lmms_metric": score.metric_name,
        "lmms_score": float(score.raw_score),
        "correct": bool(score.correct),
        "any_non_full": any(action != "FULL" for action in actions),
        "first_nonfull_layer": next((trigger + i for i, action in enumerate(actions) if action != "FULL"), None),
    }


def rollout_worker(config_path: Path, *, view: str, rank: int, resume: bool) -> None:
    contract, parent_contract, output_root = verify_contract(config_path, verify_model=True)
    config = contract["static_config"]
    if rank < 0 or rank >= int(config["world_size"]):
        raise ValueError("invalid worker rank")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    phase76 = read_json(config["parent"]["phase76_config"])
    configure_dense_determinism(int(config["seed"]) + rank, phase76["backend_settings"])
    processor, _base, wrapped = _load_model(phase76, device)
    checkpoint_path, checkpoint_sha = _view_checkpoint(config, view)
    router = _router_for_checkpoint(
        phase76, checkpoint_path, checkpoint_sha, parent_contract["contract_sha256"], device
    )
    schedule = [
        row for row in read_jsonl(output_root / f"work/{view}_schedule.jsonl")
        if int(row["worker_rank"]) == rank
    ]
    manifest_rows = read_jsonl(config["parent"]["trajectory_manifest"])
    grouped, _ = build_trajectory_sets(manifest_rows, total_layers=28)
    index = _uid_index(config, parent_contract)
    parent_root = resolve_path(config["parent"]["output_root"])
    references = {
        str(row["uid"]): row for row in read_jsonl(output_root / "release/selected_reference_routes.jsonl")
    }
    rank_root = output_root / f"work/rollout/{view}/rank{rank:02d}"
    completed = 0
    started = time.monotonic()
    for item in schedule:
        uid = str(item["uid"])
        result_path = rank_root / f"{_uid_slug(uid)}.json"
        required_conditions = ["R0"] if item["dense_outcome"] == "C" or view == "heldout" else ["R0", "R1", "R2"]
        if resume and result_path.exists():
            old = read_json(result_path)
            if (
                old.get("contract_sha256") == contract["contract_sha256"]
                and old.get("checkpoint_sha256") == checkpoint_sha
                and old.get("uid") == uid
                and sorted(old.get("conditions", {})) == sorted(required_conditions)
                and old.get("state_file_sha256") == index[uid]["state_file_sha256"]
            ):
                completed += 1
                continue
        rows = grouped[uid]
        sample = dict(rows[0]["sample"])
        payload = _load_uid_payload(parent_contract, parent_root, index[uid])
        inputs, metadata = build_dense_inputs(processor, sample, device)
        prepared = build_binary_inputs(wrapped, inputs)
        baseline = capture_four_action_route(
            wrapped, {}, ["FULL"] * 28, prepared_inputs=prepared,
            use_cache=True, native_full_rows=True,
        )
        dense_ids, dense_text, dense_score = train_generate(
            processor, wrapped, baseline, inputs["input_ids"], sample
        )
        expected_dense = [int(value) for value in rows[0]["dense_generated_token_ids"]]
        if dense_ids != expected_dense or bool(dense_score.correct) != (rows[0]["dense_outcome"] == "C"):
            raise RuntimeError(f"dense parity failed before rollout: {uid}")
        reference = references.get(uid)
        if rows[0]["dense_outcome"] == "W" and reference is None:
            raise RuntimeError(f"W UID lacks frozen reference route: {uid}")
        conditions = {}
        for condition in required_conditions:
            conditions[condition] = _run_policy(
                processor=processor, wrapped=wrapped, router=router, sample=sample,
                prepared=prepared, input_ids=inputs["input_ids"],
                trigger=int(rows[0]["trigger_layer"]),
                condition=condition, reference=reference, payload=payload,
                contract_sha256=contract["contract_sha256"],
            )
        result = {
            "schema_version": "teacher_forced_free_run_uid_result_v1",
            "contract_sha256": contract["contract_sha256"],
            "checkpoint_sha256": checkpoint_sha,
            "view": view,
            "uid": uid,
            "dataset": rows[0]["dataset"],
            "source_regime": rows[0]["source_regime"],
            "dense_outcome": rows[0]["dense_outcome"],
            "image_group_id": rows[0]["image_group_id"],
            "trigger_layer": int(rows[0]["trigger_layer"]),
            "state_file_sha256": index[uid]["state_file_sha256"],
            "consumed_image_sha256": metadata["consumed_image_sha256"],
            "dense_generated_token_ids": dense_ids,
            "dense_generated_answer": dense_text,
            "dense_lmms_score": float(dense_score.raw_score),
            "dense_correct": bool(dense_score.correct),
            "reference_program_id": None if reference is None else reference["program_id"],
            "conditions": conditions,
        }
        atomic_json(result_path, result)
        completed += 1
        del result, conditions, baseline, prepared, inputs, payload
        if completed % 5 == 0 or completed == len(schedule):
            print(json.dumps({
                "view": view, "rank": rank, "completed": completed,
                "total": len(schedule), "elapsed_seconds": time.monotonic() - started,
            }), flush=True)
    atomic_json(rank_root / "complete.json", {
        "schema_version": "teacher_forced_rollout_rank_complete_v1",
        "contract_sha256": contract["contract_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "view": view, "rank": rank,
        "expected_uids": len(schedule), "completed_uids": completed,
        "completed_at": utc_now(),
    })


def smoke(config_path: Path, *, device_index: int) -> None:
    contract, parent_contract, output_root = verify_contract(config_path, verify_model=True)
    config = contract["static_config"]
    phase76 = read_json(config["parent"]["phase76_config"])
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    configure_dense_determinism(int(config["seed"]), phase76["backend_settings"])
    processor, _base, wrapped = _load_model(phase76, device)
    router = _router_for_checkpoint(
        phase76, config["checkpoints"]["full_refit"]["path"],
        config["checkpoints"]["full_refit"]["sha256"],
        parent_contract["contract_sha256"], device,
    )
    grouped, _ = build_trajectory_sets(
        read_jsonl(config["parent"]["trajectory_manifest"]), total_layers=28
    )
    references = {str(row["uid"]): row for row in read_jsonl(output_root / "release/selected_reference_routes.jsonl")}
    w_uid = next(uid for uid in sorted(grouped) if grouped[uid][0]["dense_outcome"] == "W")
    c_uid = next(uid for uid in sorted(grouped) if grouped[uid][0]["dense_outcome"] == "C")
    index = _uid_index(config, parent_contract)
    parent_root = resolve_path(config["parent"]["output_root"])
    results = []
    for uid in (w_uid, c_uid):
        row = grouped[uid][0]
        payload = _load_uid_payload(parent_contract, parent_root, index[uid])
        sample = dict(row["sample"])
        inputs, _metadata = build_dense_inputs(processor, sample, device)
        prepared = build_binary_inputs(wrapped, inputs)
        conditions = ["R0", "R1", "R2"] if row["dense_outcome"] == "W" else ["R0"]
        current = [
            _run_policy(
                processor=processor, wrapped=wrapped, router=router, sample=sample,
                prepared=prepared, input_ids=inputs["input_ids"],
                trigger=int(row["trigger_layer"]), condition=condition,
                reference=references.get(uid), payload=payload,
                contract_sha256=contract["contract_sha256"],
            )
            for condition in conditions
        ]
        results.append({"uid": uid, "dense_outcome": row["dense_outcome"], "conditions": current})
    passed = (
        len(results) == 2
        and all(condition["release_state_exact"] is not False for result in results for condition in result["conditions"])
        and all(len(condition["actions"]) == 28 - int(grouped[result["uid"]][0]["trigger_layer"])
                for result in results for condition in result["conditions"])
    )
    report = {
        "schema_version": "teacher_forced_free_run_smoke_v1",
        "contract_sha256": contract["contract_sha256"],
        "passed": passed, "uids": [w_uid, c_uid], "results": results,
    }
    atomic_json(output_root / "smoke/implementation_smoke.json", report)
    print(json.dumps({"passed": passed, "uids": [w_uid, c_uid]}, sort_keys=True))
    if not passed:
        raise RuntimeError("teacher-forced/free-run audit smoke failed")


def _free_metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    w = [row for row in results if row["dense_outcome"] == "W"]
    c = [row for row in results if row["dense_outcome"] == "C"]
    return {
        "uids": len(results),
        "dense_w_uids": len(w),
        "dense_c_uids": len(c),
        "w_success": sum(bool(row["conditions"]["R0"]["correct"]) for row in w),
        "w_success_rate": _rate(sum(bool(row["conditions"]["R0"]["correct"]) for row in w), len(w)),
        "c_preserved": sum(bool(row["conditions"]["R0"]["correct"]) for row in c),
        "c_preservation_rate": _rate(sum(bool(row["conditions"]["R0"]["correct"]) for row in c), len(c)),
    }


def _teacher_view_metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    occurrences = [row for result in results for row in result["occurrences"]]
    first = _first_nonfull_rows(results)
    nonfull = [row for row in occurrences if row["target_action"] != "FULL"]
    full = [row for row in occurrences if row["target_action"] == "FULL"]
    return {
        "full_recall": _rate(sum(bool(row["target_top1"]) for row in full), len(full)),
        "nonfull_recall": _rate(sum(bool(row["target_top1"]) for row in nonfull), len(nonfull)),
        "first_nonfull_recall": _rate(sum(bool(row["target_top1"]) for row in first), len(first)),
        "first_nonfull_mean_probability": _mean([float(row["target_probability"]) for row in first]),
        "first_nonfull_median_probability": _median([float(row["target_probability"]) for row in first]),
        "first_nonfull_mean_margin": _mean([float(row["target_vs_full_margin"]) for row in first]),
        "first_nonfull_median_margin": _median([float(row["target_vs_full_margin"]) for row in first]),
    }


def _write_figures(
    output_root: Path,
    *,
    first_rows: Sequence[Mapping[str, Any]],
    occurrences: Sequence[Mapping[str, Any]],
    survival_rows: Sequence[Mapping[str, Any]],
    off_rows: Sequence[Mapping[str, Any]],
    release_rows: Sequence[Mapping[str, Any]],
    seen_teacher: Mapping[str, Any],
    heldout_teacher: Mapping[str, Any],
    dense_c_rows: Sequence[Mapping[str, Any]],
) -> None:
    layers = sorted({int(row["layer"]) for row in first_rows})
    recalls = [
        np.mean([bool(row["target_top1"]) for row in first_rows if int(row["layer"]) == layer])
        for layer in layers
    ]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(layers, recalls, marker="o")
    axis.set(xlabel="First corrective layer", ylabel="Top-1 recall", ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(output_root / "figures/first_nonfull_recall_by_layer.png", dpi=160)
    plt.close(figure)

    nonfull = [row for row in occurrences if row["target_action"] != "FULL"]
    target = [float(row["target_probability"]) for row in nonfull]
    full = [float(row["probabilities"]["FULL"]) for row in nonfull]
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(full, target, s=4, alpha=0.15)
    axis.plot([0, 1], [0, 1], color="black", linewidth=1)
    axis.set(xlabel="P(FULL)", ylabel="P(target non-FULL)", xlim=(0, 1), ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(output_root / "figures/target_nonfull_vs_full_probability.png", dpi=160)
    plt.close(figure)

    trigger_survival = [row for row in survival_rows if row["alignment"] == "trigger"]
    first_survival = [row for row in survival_rows if row["alignment"] == "first_nonfull"]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot([row["relative_depth"] for row in trigger_survival], [row["survival"] for row in trigger_survival], label="From trigger")
    axis.plot([row["relative_depth"] for row in first_survival], [row["survival"] for row in first_survival], label="Relative to first non-FULL")
    axis.set(xlabel="Aligned layer offset", ylabel="Known-route support survival", ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "figures/prefix_support_survival.png", dpi=160)
    plt.close(figure)

    delays = [int(row["trigger_to_off_support_delay"]) for row in off_rows if row["off_support_layer"] != "NONE"]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(delays, bins=range(0, 30))
    axis.set(xlabel="Trigger-to-first-off-support delay", ylabel="Dense-W UIDs")
    figure.tight_layout()
    figure.savefig(output_root / "figures/first_off_support_distribution.png", dpi=160)
    plt.close(figure)

    labels = [row["condition"] for row in release_rows]
    values = [float(row["success_rate"]) for row in release_rows]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(labels, values)
    axis.set(ylabel="Final-correct rate", ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(output_root / "figures/r0_r1_r2_success.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(["Seen full-refit", "Held-out internal-dev"], [seen_teacher["first_nonfull_recall"], heldout_teacher["first_nonfull_recall"]])
    axis.set(ylabel="First non-FULL top-1 recall", ylim=(0, 1))
    axis.tick_params(axis="x", rotation=12)
    figure.tight_layout()
    figure.savefig(output_root / "figures/seen_vs_heldout_expert_recall.png", dpi=160)
    plt.close(figure)

    preservation = _rate(sum(bool(row["r0_correct"]) for row in dense_c_rows), len(dense_c_rows))
    allfull = _rate(sum(row["first_deviation_layer"] == "NONE" for row in dense_c_rows), len(dense_c_rows))
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(["Final preservation", "All-FULL stability"], [preservation, allfull])
    axis.set(ylim=(0, 1), ylabel="Fraction of Dense-C UIDs")
    figure.tight_layout()
    figure.savefig(output_root / "figures/dense_c_preservation_control.png", dpi=160)
    plt.close(figure)


def aggregate_rollout(config_path: Path) -> None:
    contract, _parent, output_root = verify_contract(config_path)
    config = contract["static_config"]
    teacher_gate = read_json(output_root / "work/teacher_aggregation.json")
    if teacher_gate.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("teacher-forced aggregation has not passed")
    seen = _collect_results(contract, output_root, kind="rollout", view="seen")
    heldout = _collect_results(contract, output_root, kind="rollout", view="heldout")
    if len(seen) != int(config["population"]["uids"]) or len(heldout) != int(config["population"]["internal_dev_uids"]):
        raise RuntimeError("rollout population differs")
    atomic_jsonl(output_root / "free_run/training_uid_rollouts.jsonl", seen)

    manifest_rows = read_jsonl(config["parent"]["trajectory_manifest"])
    grouped, _ = build_trajectory_sets(manifest_rows, total_layers=28)
    references = {str(row["uid"]): row for row in read_jsonl(output_root / "release/selected_reference_routes.jsonl")}
    w_rows = [row for row in seen if row["dense_outcome"] == "W"]
    c_rows = [row for row in seen if row["dense_outcome"] == "C"]
    if len(w_rows) != int(config["population"]["dense_w_uids"]) or len(c_rows) != int(config["population"]["dense_c_uids"]):
        raise RuntimeError("seen rollout C/W census differs")

    support_rows = []
    first_off_rows = []
    mass_rows = []
    traces = []
    relative_support: dict[int, list[bool]] = defaultdict(list)
    for result in w_rows:
        uid = str(result["uid"])
        r0 = result["conditions"]["R0"]
        programs = [
            {"program_id": row["program_id"], "actions": row["suffix_actions"]}
            for row in grouped[uid]
        ]
        probabilities = [row["probabilities"] for row in r0["trace"]]
        trace, first = prefix_support_trace(
            r0["actions"], programs, trigger_layer=int(result["trigger_layer"]),
            probabilities=probabilities,
        )
        traces.append(trace)
        first_expert = int(references[uid]["first_nonfull_layer"])
        for row in trace:
            relative_support[int(row["layer"]) - first_expert].append(bool(row["supported"]))
        relation = "none"
        if first is not None:
            relation = (
                "before" if int(first["layer"]) < first_expert
                else "at" if int(first["layer"]) == first_expert else "after"
            )
        support_rows.append({
            "schema_version": "teacher_forced_prefix_support_trace_v1",
            "contract_sha256": contract["contract_sha256"],
            "uid": uid, "dataset": result["dataset"],
            "trigger_layer": result["trigger_layer"],
            "first_expert_nonfull_layer": first_expert,
            "trace": trace, "first_off_support": first,
            "off_support_relative_to_first_expert_nonfull": relation,
        })
        off_row = {
            "uid": uid, "dataset": result["dataset"],
            "trigger_layer": int(result["trigger_layer"]),
            "first_expert_nonfull_layer": first_expert,
            "off_support_layer": "NONE" if first is None else int(first["layer"]),
            "trigger_to_off_support_delay": "NONE" if first is None else int(first["depth_after_trigger"]),
            "relation_to_first_expert_nonfull": relation,
            "chosen_action": "NONE" if first is None else first["chosen_action"],
            "supported_actions": "" if first is None else "|".join(first["supported_actions"]),
            "compatible_route_count_before": "" if first is None else int(first["compatible_route_count_before"]),
            "supported_probability_mass": "" if first is None else float(first["supported_probability_mass"]),
            "full_probability": "" if first is None else float(first["full_probability"]),
            "best_supported_action": "" if first is None else first["best_supported_action"],
            "best_supported_probability": "" if first is None else float(first["best_supported_probability"]),
            "chosen_vs_best_supported_logit_margin": "" if first is None else float(first["chosen_vs_best_supported_logit_margin"]),
        }
        first_off_rows.append(off_row)
        if first is not None:
            mass_rows.append({key: off_row[key] for key in (
                "uid", "dataset", "off_support_layer", "chosen_action", "supported_actions",
                "supported_probability_mass", "full_probability", "best_supported_action",
                "best_supported_probability", "chosen_vs_best_supported_logit_margin",
            )})
    atomic_jsonl(output_root / "free_run/support_tracking.jsonl", support_rows)
    atomic_csv(output_root / "free_run/first_off_support.csv", first_off_rows)
    atomic_csv(output_root / "metrics/supported_action_mass_at_divergence.csv", mass_rows)

    survival_rows = [
        {"alignment": "trigger", "relative_depth": row["depth_after_trigger"], **{key: row[key] for key in ("eligible_uids", "supported_uids", "survival")}}
        for row in support_survival(traces)
    ]
    for offset in sorted(relative_support):
        values = relative_support[offset]
        survival_rows.append({
            "alignment": "first_nonfull", "relative_depth": offset,
            "eligible_uids": len(values), "supported_uids": sum(values),
            "survival": sum(values) / len(values),
        })
    atomic_csv(output_root / "free_run/prefix_support_survival.csv", survival_rows)

    release_files = {
        "R0": "release/r0_free_from_trigger.jsonl",
        "R1": "release/r1_release_at_first_nonfull.jsonl",
        "R2": "release/r2_force_first_nonfull_then_release.jsonl",
    }
    release_summary = []
    for condition, relative in release_files.items():
        rows = [
            {
                "contract_sha256": contract["contract_sha256"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "uid": result["uid"], "dataset": result["dataset"],
                "trigger_layer": result["trigger_layer"],
                "reference_program_id": result["reference_program_id"],
                **result["conditions"][condition],
            }
            for result in w_rows
        ]
        atomic_jsonl(output_root / relative, rows)
        successes = sum(bool(row["correct"]) for row in rows)
        release_summary.append({
            "condition": condition, "uids": len(rows), "final_correct": successes,
            "success_rate": successes / len(rows),
            "any_non_full": sum(bool(row["any_non_full"]) for row in rows),
        })
    atomic_csv(output_root / "release/release_success_summary.csv", release_summary)

    dense_c_control = []
    for result in c_rows:
        r0 = result["conditions"]["R0"]
        full_rows = [row for row in r0["trace"]]
        dense_c_control.append({
            "uid": result["uid"], "dataset": result["dataset"],
            "trigger_layer": result["trigger_layer"],
            "full_top1_layers": sum(row["action"] == "FULL" for row in full_rows),
            "suffix_layers": len(full_rows),
            "mean_full_probability": _mean([float(row["probabilities"]["FULL"]) for row in full_rows]),
            "mean_full_vs_best_nonfull_margin": _mean([
                float(row["logits"]["FULL"]) - max(float(row["logits"][action]) for action in ACTION_NAMES[1:])
                for row in full_rows
            ]),
            "first_deviation_layer": next((int(row["layer"]) for row in full_rows if row["action"] != "FULL"), "NONE"),
            "r0_correct": bool(r0["correct"]),
            "regressed": not bool(r0["correct"]),
        })
    atomic_csv(output_root / "free_run/dense_c_control.csv", dense_c_control)

    seen_teacher_results = _collect_results(contract, output_root, kind="teacher_forced", view="seen")
    heldout_teacher_results = _collect_results(contract, output_root, kind="teacher_forced", view="heldout")
    seen_teacher = _teacher_view_metrics(seen_teacher_results)
    heldout_teacher = _teacher_view_metrics(heldout_teacher_results)
    seen_free = _free_metrics(seen)
    heldout_free = _free_metrics(heldout)
    heldout_support = []
    for result in heldout:
        if result["dense_outcome"] != "W":
            continue
        trace, first = prefix_support_trace(
            result["conditions"]["R0"]["actions"],
            [{"program_id": row["program_id"], "actions": row["suffix_actions"]} for row in grouped[result["uid"]]],
            trigger_layer=int(result["trigger_layer"]),
            probabilities=[row["probabilities"] for row in result["conditions"]["R0"]["trace"]],
        )
        heldout_support.append(first is not None)
    seen_off_rate = _rate(sum(row["off_support_layer"] != "NONE" for row in first_off_rows), len(first_off_rows))
    heldout_off_rate = _rate(sum(heldout_support), len(heldout_support))
    generalization_rows = [
        {"view": "full_refit_on_seen_uids", "uids": len(seen),
         "first_nonfull_recall": seen_teacher["first_nonfull_recall"],
         "nonfull_top1": seen_teacher["nonfull_recall"],
         "free_run_w_success": seen_free["w_success_rate"], "off_support_rate": seen_off_rate},
        {"view": "frozen_dev_checkpoint_on_heldout_uids", "uids": len(heldout),
         "first_nonfull_recall": heldout_teacher["first_nonfull_recall"],
         "nonfull_top1": heldout_teacher["nonfull_recall"],
         "free_run_w_success": heldout_free["w_success_rate"], "off_support_rate": heldout_off_rate},
    ]
    atomic_csv(output_root / "generalization/seen_vs_heldout_summary.csv", generalization_rows)
    atomic_csv(output_root / "generalization/heldout_free_run_metrics.csv", [{**heldout_free, "off_support_rate": heldout_off_rate}])

    nonfull_occurrences = [row for result in seen_teacher_results for row in result["occurrences"] if row["target_action"] != "FULL"]
    action_probability_rows = []
    for action in ACTION_NAMES:
        current = [row for result in seen_teacher_results for row in result["occurrences"] if row["target_action"] == action]
        action_probability_rows.append({
            "target_action": action, "n": len(current),
            "mean_target_probability": _mean([float(row["target_probability"]) for row in current]),
            "median_target_probability": _median([float(row["target_probability"]) for row in current]),
            "mean_full_probability": _mean([float(row["probabilities"]["FULL"]) for row in current]),
            "top1_recall": _rate(sum(bool(row["target_top1"]) for row in current), len(current)),
        })
    atomic_csv(output_root / "metrics/action_probability_summary.csv", action_probability_rows)

    release_by_name = {row["condition"]: row for row in release_summary}
    pre_fraction = sum(row["relation_to_first_expert_nonfull"] == "before" for row in first_off_rows) / len(first_off_rows)
    post_fraction = sum(row["relation_to_first_expert_nonfull"] in {"at", "after"} for row in first_off_rows) / len(first_off_rows)
    decision_inputs = {
        "seen_first_nonfull_recall": seen_teacher["first_nonfull_recall"],
        "heldout_first_nonfull_recall": heldout_teacher["first_nonfull_recall"],
        "r0_success": release_by_name["R0"]["success_rate"],
        "r1_success": release_by_name["R1"]["success_rate"],
        "r2_success": release_by_name["R2"]["success_rate"],
        "pre_intervention_off_support_fraction": pre_fraction,
        "post_intervention_off_support_fraction": post_fraction,
    }
    decision = classify_bottleneck(decision_inputs, config["decision_thresholds"])
    decision_rows = [
        {"metric": key, "value": value} for key, value in decision_inputs.items()
    ] + [{"metric": "dominant_case", "value": decision["case"]}]
    atomic_csv(output_root / "metrics/decision_metrics.csv", decision_rows)

    first_rows = _first_nonfull_rows(seen_teacher_results)
    occurrences = [row for result in seen_teacher_results for row in result["occurrences"]]
    _write_figures(
        output_root, first_rows=first_rows, occurrences=occurrences,
        survival_rows=survival_rows, off_rows=first_off_rows,
        release_rows=release_summary, seen_teacher=seen_teacher,
        heldout_teacher=heldout_teacher, dense_c_rows=dense_c_control,
    )

    off_delays = [int(row["trigger_to_off_support_delay"]) for row in first_off_rows if row["off_support_layer"] != "NONE"]
    supported_mass = [float(row["supported_probability_mass"]) for row in mass_rows]
    selected_keys = {(row["uid"], row["program_id"]) for row in references.values()}
    selected_first = [row for row in first_rows if (row["uid"], row["program_id"]) in selected_keys]
    selected_first_recall = _rate(sum(bool(row["target_top1"]) for row in selected_first), len(selected_first))
    action_recall = {
        action: _rate(
            sum(bool(row["target_top1"]) for row in occurrences if row["target_action"] == action),
            sum(row["target_action"] == action for row in occurrences),
        ) for action in ACTION_NAMES
    }
    summary = f"""# Teacher-forced versus free-run audit summary

Contract: `{contract['contract_sha256']}`

## Population and exact expert-state fit

- Audited **{len(seen)} UIDs**, **{config['population']['programs']} trajectories**, and **{len(occurrences)} route-state occurrences** in the seen/full-refit view.
- FULL top-1 accuracy: **{seen_teacher['full_recall']:.4f}**.
- Non-FULL top-1 accuracy: **{seen_teacher['nonfull_recall']:.4f}**.
- First-nonFULL top-1 recall: **{seen_teacher['first_nonfull_recall']:.4f}** across all successful routes.
- Highest-responsibility-route first-nonFULL recall: **{selected_first_recall:.4f}**.
- First-nonFULL target probability: mean **{seen_teacher['first_nonfull_mean_probability']:.4f}**, median **{seen_teacher['first_nonfull_median_probability']:.4f}**.
- First-nonFULL target-vs-FULL margin: mean **{seen_teacher['first_nonfull_mean_margin']:.4f}**, median **{seen_teacher['first_nonfull_median_margin']:.4f}**.
- Action recall — READ_ONLY **{action_recall['READ_ONLY']:.4f}**, WRITE_ONLY **{action_recall['WRITE_ONLY']:.4f}**, IGNORE **{action_recall['IGNORE']:.4f}**.

## Free rollout and known-route support

- Training Dense-W R0 rescue: **{seen_free['w_success']}/{seen_free['dense_w_uids']} ({seen_free['w_success_rate']:.4f})**.
- Training Dense-C R0 preservation: **{seen_free['c_preserved']}/{seen_free['dense_c_uids']} ({seen_free['c_preservation_rate']:.4f})**.
- Left known successful-route support: **{sum(row['off_support_layer'] != 'NONE' for row in first_off_rows)}/{len(first_off_rows)} ({seen_off_rate:.4f})**.
- First off-support delay: median **{_median(off_delays) if off_delays else 'not estimable'}** layers after trigger.
- Off-support relation to selected first expert intervention: before **{sum(row['relation_to_first_expert_nonfull'] == 'before' for row in first_off_rows)}**, at **{sum(row['relation_to_first_expert_nonfull'] == 'at' for row in first_off_rows)}**, after **{sum(row['relation_to_first_expert_nonfull'] == 'after' for row in first_off_rows)}**, no divergence **{sum(row['relation_to_first_expert_nonfull'] == 'none' for row in first_off_rows)}**.
- Supported-action probability mass at divergence: mean **{_mean(supported_mass) if supported_mass else 'not estimable'}**, median **{_median(supported_mass) if supported_mass else 'not estimable'}**.

## Release experiment

- R0 free from trigger: **{release_by_name['R0']['final_correct']}/{release_by_name['R0']['uids']} ({release_by_name['R0']['success_rate']:.4f})**.
- R1 force only before first non-FULL, release on its exact state: **{release_by_name['R1']['final_correct']}/{release_by_name['R1']['uids']} ({release_by_name['R1']['success_rate']:.4f})**.
- R2 force through the first non-FULL, then release: **{release_by_name['R2']['final_correct']}/{release_by_name['R2']['uids']} ({release_by_name['R2']['success_rate']:.4f})**.
- R1-R0: **{release_by_name['R1']['success_rate'] - release_by_name['R0']['success_rate']:+.4f}**; R2-R1: **{release_by_name['R2']['success_rate'] - release_by_name['R1']['success_rate']:+.4f}**.

## Held-out generalization and decision

- Frozen internal-dev checkpoint, 115 group-disjoint dev UIDs: first-nonFULL recall **{heldout_teacher['first_nonfull_recall']:.4f}**, non-FULL recall **{heldout_teacher['nonfull_recall']:.4f}**, W free-run success **{heldout_free['w_success_rate']:.4f}**, off-support rate **{heldout_off_rate:.4f}**.
- Seen-to-held-out first-nonFULL recall drop: **{seen_teacher['first_nonfull_recall'] - heldout_teacher['first_nonfull_recall']:+.4f}**.
- Prospectively frozen decision rule: **{decision['case']}**.

## Scope limits

Off-support means outside the observed successful route corpus, not provably invalid. The route corpus is not exhaustive, expert actions need not be unique, internal success does not guarantee external success, and this audit does not establish that DAgger or any proposed objective change will work. No external evaluation labels were read or used.
"""
    _atomic_bytes(output_root / "summaries/teacher_forced_free_run_audit_summary.md", summary.encode())

    recommendations = {
        "objective_action_learning": (
            "Run one bounded refit that retains the exact trajectory-marginal objective and adds one fixed-weight auxiliary cross-entropy term only at each route's first non-FULL expert state. "
            "This is the smallest test of whether sparse corrective-action dilution is causal. A rise in held-out first-nonFULL recall and R1 success supports the change; no rise rejects it. It must not be interpreted as evidence of external benchmark benefit without a separately authorized evaluation."
        ),
        "pre_intervention_exposure": (
            "Run one bounded on-policy prefix collection on the existing 569 training-side UIDs, relabel only the first off-support state with the existing successful-route/search oracle, and refit once. "
            "Improved R0 support survival and rescue supports pre-intervention exposure as causal; no improvement rejects the collection strategy. It does not show that known-route support is exhaustive."
        ),
        "post_intervention_exposure": (
            "Run one bounded post-first-intervention on-policy state collection on the existing 569 training-side UIDs and relabel only later off-support states with the existing oracle. "
            "Improved R2 continuation success supports post-intervention exposure; no improvement rejects the relabeling strategy. It does not establish external generalization."
        ),
        "generalization": (
            "Run one frozen-split representation diagnostic that adds only decoder-layer identity to the existing routed-state router and refits on the same internal train split. "
            "A held-out first-nonFULL recall gain supports missing depth context; no gain rejects that minimal representation hypothesis. It must not be treated as authorization for a broader architecture pivot."
        ),
        "mixed": (
            "Run one bounded refit with a fixed-weight first-nonFULL auxiliary cross-entropy term while retaining the current trajectory marginal, because the earliest measured failure is expert-state corrective-action selection. "
            "Improved held-out first-nonFULL recall and R1 success supports action-learning dilution; no improvement rejects it. Later exposure effects and external benefit remain unestablished."
        ),
    }
    recommendation = (
        "# Next Stage-2 recommendation\n\n"
        f"Dominant audit classification: **{decision['case']}**.\n\n"
        f"{recommendations[decision['case']]}\n\n"
        "This is a recommendation only. It was not executed in Phase 77.\n"
    )
    _atomic_bytes(output_root / "summaries/next_stage2_recommendation.md", recommendation.encode())
    atomic_json(output_root / "work/rollout_aggregation.json", {
        "schema_version": "teacher_forced_free_run_rollout_aggregation_v1",
        "contract_sha256": contract["contract_sha256"],
        "seen_uids": len(seen), "heldout_uids": len(heldout),
        "decision": decision, "decision_inputs": decision_inputs,
        "completed_at": utc_now(),
    })
    print(json.dumps({
        "seen": len(seen), "heldout": len(heldout),
        "r0": release_by_name["R0"]["success_rate"],
        "r1": release_by_name["R1"]["success_rate"],
        "r2": release_by_name["R2"]["success_rate"],
        "case": decision["case"],
    }, sort_keys=True))


REQUIRED_ARTIFACTS = (
    "protocol.md",
    "frozen_protocol.json",
    "contracts/checkpoint_contract.json",
    "contracts/corpus_contract.json",
    "contracts/internal_dev_availability.md",
    "teacher_forced/all_route_state_logits.jsonl",
    "teacher_forced/action_class_metrics.csv",
    "teacher_forced/first_nonfull_metrics.csv",
    "teacher_forced/highest_responsibility_route_metrics.csv",
    "teacher_forced/layer_breakdown.csv",
    "teacher_forced/provenance_breakdown.csv",
    "free_run/training_uid_rollouts.jsonl",
    "free_run/support_tracking.jsonl",
    "free_run/first_off_support.csv",
    "free_run/prefix_support_survival.csv",
    "free_run/dense_c_control.csv",
    "release/selected_reference_routes.jsonl",
    "release/r0_free_from_trigger.jsonl",
    "release/r1_release_at_first_nonfull.jsonl",
    "release/r2_force_first_nonfull_then_release.jsonl",
    "release/release_success_summary.csv",
    "generalization/seen_vs_heldout_summary.csv",
    "generalization/heldout_teacher_forced_metrics.csv",
    "generalization/heldout_free_run_metrics.csv",
    "metrics/decision_metrics.csv",
    "metrics/supported_action_mass_at_divergence.csv",
    "metrics/action_probability_summary.csv",
    "figures/first_nonfull_recall_by_layer.png",
    "figures/target_nonfull_vs_full_probability.png",
    "figures/prefix_support_survival.png",
    "figures/first_off_support_distribution.png",
    "figures/r0_r1_r2_success.png",
    "figures/seen_vs_heldout_expert_recall.png",
    "figures/dense_c_preservation_control.png",
    "summaries/teacher_forced_free_run_audit_summary.md",
    "summaries/next_stage2_recommendation.md",
)


def verify_artifacts(config_path: Path) -> None:
    contract, _parent, output_root = verify_contract(config_path)
    config = contract["static_config"]
    seen_teacher = _collect_results(contract, output_root, kind="teacher_forced", view="seen")
    heldout_teacher = _collect_results(contract, output_root, kind="teacher_forced", view="heldout")
    seen_rollout = _collect_results(contract, output_root, kind="rollout", view="seen")
    heldout_rollout = _collect_results(contract, output_root, kind="rollout", view="heldout")
    occurrence_count = sum(len(row["occurrences"]) for row in seen_teacher)
    program_count = sum(len(row["programs"]) for row in seen_teacher)
    unique_states = sum(int(row["unique_state_count"]) for row in seen_teacher)
    references = read_jsonl(output_root / "release/selected_reference_routes.jsonl")
    route_counts = {
        condition: len(read_jsonl(output_root / relative))
        for condition, relative in {
            "R0": "release/r0_free_from_trigger.jsonl",
            "R1": "release/r1_release_at_first_nonfull.jsonl",
            "R2": "release/r2_force_first_nonfull_then_release.jsonl",
        }.items()
    }
    checks = {
        "seen_teacher_uids": len(seen_teacher) == int(config["population"]["uids"]),
        "heldout_teacher_uids": len(heldout_teacher) == int(config["population"]["internal_dev_uids"]),
        "route_state_occurrences": occurrence_count == int(config["population"]["route_state_occurrences"]),
        "programs": program_count == int(config["population"]["programs"]),
        "unique_states": unique_states == int(config["population"]["unique_states"]),
        "reference_routes": len(references) == int(config["population"]["dense_w_uids"]),
        "seen_rollout_uids": len(seen_rollout) == int(config["population"]["uids"]),
        "heldout_rollout_uids": len(heldout_rollout) == int(config["population"]["internal_dev_uids"]),
        "release_counts": all(value == int(config["population"]["dense_w_uids"]) for value in route_counts.values()),
        "all_required_artifacts": all((output_root / relative).is_file() for relative in REQUIRED_ARTIFACTS),
        "all_release_states_exact": all(
            condition["release_state_exact"] is not False
            for result in seen_rollout for condition in result["conditions"].values()
        ),
        "all_image_hashes_bound": all(
            str(result["image_group_id"]).removeprefix("sha256:") == result["consumed_image_sha256"]
            for result in seen_rollout + heldout_rollout
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"artifact verification failed: {checks}")
    manifest = {
        "schema_version": "teacher_forced_free_run_artifact_manifest_v1",
        "contract_sha256": contract["contract_sha256"],
        "verified_at": utc_now(),
        "checks": checks,
        "counts": {
            "seen_teacher_uids": len(seen_teacher), "heldout_teacher_uids": len(heldout_teacher),
            "route_state_occurrences": occurrence_count, "programs": program_count,
            "unique_states": unique_states, "reference_routes": len(references),
            "seen_rollout_uids": len(seen_rollout), "heldout_rollout_uids": len(heldout_rollout),
            **{f"{key.lower()}_rows": value for key, value in route_counts.items()},
        },
        "files": {relative: file_sha256(output_root / relative) for relative in REQUIRED_ARTIFACTS},
    }
    atomic_json(output_root / "artifact_manifest.json", manifest)
    print(json.dumps({"passed": True, **manifest["counts"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    teacher = subparsers.add_parser("teacher-worker")
    teacher.add_argument("--view", choices=("seen", "heldout"), required=True)
    teacher.add_argument("--rank", type=int, required=True)
    teacher.add_argument("--resume", action="store_true")
    subparsers.add_parser("aggregate-teacher")
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--device", type=int, default=0)
    rollout = subparsers.add_parser("rollout-worker")
    rollout.add_argument("--view", choices=("seen", "heldout"), required=True)
    rollout.add_argument("--rank", type=int, required=True)
    rollout.add_argument("--resume", action="store_true")
    subparsers.add_parser("aggregate-rollout")
    subparsers.add_parser("verify-artifacts")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config)
    elif args.command == "teacher-worker":
        teacher_worker(args.config, view=args.view, rank=args.rank, resume=args.resume)
    elif args.command == "aggregate-teacher":
        aggregate_teacher(args.config)
    elif args.command == "smoke":
        smoke(args.config, device_index=args.device)
    elif args.command == "rollout-worker":
        rollout_worker(args.config, view=args.view, rank=args.rank, resume=args.resume)
    elif args.command == "aggregate-rollout":
        aggregate_rollout(args.config)
    elif args.command == "verify-artifacts":
        verify_artifacts(args.config)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
