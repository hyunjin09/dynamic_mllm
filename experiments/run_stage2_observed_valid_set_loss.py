#!/usr/bin/env python3
"""Matched Stage-2 Experiment C with exact observed-valid-set supervision."""

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
import platform
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.parallel import DistributedDataParallel  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from binary_policy.executor.four_action import (  # noqa: E402
    capture_four_action_route,
    capture_online_four_action_route,
)
from dense_failure_stage1.runtime import build_dense_inputs, configure_dense_determinism  # noqa: E402
from dense_failure_stage2.full_label_generation import verify_artifact_manifest  # noqa: E402
from dense_failure_stage2.mcts_diagnosis import intervention_index, paired_uid_bootstrap  # noqa: E402
from dense_failure_stage2.observed_valid_set import (  # noqa: E402
    build_exact_valid_sets,
    exact_state_id,
    observed_valid_metrics,
    observed_valid_set_loss,
    uid_paired_values,
    valid_action_mask,
)
from dense_failure_stage2.shared_union import OPERATING_POINTS  # noqa: E402
from dense_failure_stage2.v1_router import ACTION_NAMES, ACTION_TO_INDEX  # noqa: E402
from experiments.run_stage2_shared_union_training import (  # noqa: E402
    _atomic_bytes,
    _binary_to,
    _generate,
    _load_model,
    _nvidia_memory,
    _physical_device_index,
    _prepare_binary,
    _router,
    _stack_route_states,
    _verify_model_snapshot,
    append_jsonl,
    atomic_csv,
    atomic_json,
    atomic_jsonl,
    canonical_hash,
    command_output,
    file_sha256,
    read_json,
    read_jsonl,
    resolve_path,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/stage2_observed_valid_set_loss_v1.json"
ROUTE_SOURCES = ("preservation_full", "single", "mcts")
BOUND_CODE = (
    "configs/stage2_observed_valid_set_loss_v1.json",
    "dense_failure_stage2/observed_valid_set.py",
    "dense_failure_stage2/mcts_diagnosis.py",
    "dense_failure_stage2/shared_union.py",
    "dense_failure_stage2/v1_router.py",
    "experiments/run_stage2_observed_valid_set_loss.py",
    "experiments/run_stage2_shared_union_training.py",
    "experiments/run_stage2_v1_training_revised.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/model.py",
    "dense_failure_stage1/runtime.py",
    "dense_failure_stage1/lmms_scoring.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "stage2_observed_valid_set_loss_config_v1":
        raise ValueError("unsupported observed-valid-set config")
    if int(config["world_size"]) != 4:
        raise ValueError("Experiment C requires exactly four workers")
    if config["loss"] != "exact_observed_valid_set_logsumexp":
        raise ValueError("only the exact observed-valid-set loss is supported")
    if int(config["smoke"]["states_per_category"]) != 12:
        raise ValueError("the deliberate smoke must retain 12 states per category")
    return config


def _route_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources = config["sources"]
    rows = []
    for key, expected in (
        ("union_preservation", "preservation_full"),
        ("union_single", "single"),
        ("union_mcts", "mcts"),
    ):
        family = read_jsonl(resolve_path(sources[key]))
        if any(str(row["route_source"]) != expected for row in family):
            raise RuntimeError(f"route-source mismatch in {key}")
        rows.extend(family)
    if len(rows) != 2519 or len({row["route_id"] for row in rows}) != len(rows):
        raise RuntimeError("frozen union route population differs from Phase 66")
    return rows


def _distribution_rows(states: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    scopes = {"overall": list(states)}
    for source in ROUTE_SOURCES:
        scopes[source] = [row for row in states if source in row["route_sources"]]
    for source, rows in scopes.items():
        counts = Counter(int(row["valid_action_count"]) for row in rows)
        for size in range(1, 5):
            output.append(
                {
                    "breakdown": "valid_action_count",
                    "route_source": source,
                    "valid_action_count": size,
                    "observed_valid_actions": "",
                    "states": counts[size],
                }
            )
        set_counts = Counter("|".join(row["observed_valid_actions"]) for row in rows)
        for action_set, count in sorted(set_counts.items(), key=lambda item: (-item[1], item[0])):
            output.append(
                {
                    "breakdown": "action_set",
                    "route_source": source,
                    "valid_action_count": len(action_set.split("|")),
                    "observed_valid_actions": action_set,
                    "states": count,
                }
            )
    return output


def _choose_smoke_states(
    states: Sequence[Mapping[str, Any]],
    occurrences: Sequence[Mapping[str, Any]],
    routes: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    per_category: int,
    epochs: int,
    world_size: int,
) -> list[dict[str, Any]]:
    occurrence_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in occurrences:
        occurrence_by_state[str(row["state_id"])].append(dict(row))
    route_by_id = {str(row["route_id"]): row for row in routes}
    categories = (
        ("single_valid", lambda row: int(row["valid_action_count"]) == 1 and "preservation_full" not in row["route_sources"]),
        ("two_valid", lambda row: int(row["valid_action_count"]) == 2),
        ("three_valid", lambda row: int(row["valid_action_count"]) == 3),
        ("preservation_full", lambda row: "preservation_full" in row["route_sources"]),
        ("single_corrective", lambda row: "single" in row["route_sources"] and row["contains_nonFULL"] and not row["contains_FULL"]),
        ("mcts_corrective", lambda row: "mcts" in row["route_sources"] and row["contains_nonFULL"] and not row["contains_FULL"]),
    )
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for category, predicate in categories:
        candidates = [row for row in states if predicate(row) and row["state_id"] not in used]
        candidates.sort(key=lambda row: sha256(f"{seed}:{category}:{row['state_id']}".encode()).hexdigest())
        if len(candidates) < per_category:
            raise RuntimeError(f"insufficient deliberate smoke states for {category}")
        for state in candidates[:per_category]:
            preferred_source = category if category in ROUTE_SOURCES else None
            if category == "single_corrective":
                preferred_source = "single"
            elif category == "mcts_corrective":
                preferred_source = "mcts"
            refs = occurrence_by_state[str(state["state_id"])]
            if preferred_source:
                refs = [row for row in refs if row["route_source"] == preferred_source]
            ref = min(refs, key=lambda row: str(row["route_id"]))
            route = route_by_id[str(ref["route_id"])]
            selected.append(
                {
                    "category": category,
                    "state_id": state["state_id"],
                    "uid": state["uid"],
                    "layer": state["layer"],
                    "route_id": ref["route_id"],
                    "route_source": ref["route_source"],
                    "actions": route["actions"],
                    "target_action": ref["target_action"],
                    "target_action_index": ACTION_TO_INDEX[ref["target_action"]],
                    "observed_valid_actions": state["observed_valid_actions"],
                    "observed_valid_action_indices": state["observed_valid_action_indices"],
                }
            )
            used.add(str(state["state_id"]))
    if len(selected) != per_category * len(categories):
        raise RuntimeError("deliberate smoke state count differs")
    schedule = []
    for epoch in range(epochs):
        rows = [dict(row) for row in selected]
        random.Random((seed << 16) + epoch).shuffle(rows)
        for draw_index, row in enumerate(rows):
            schedule.append(
                {
                    "schema_version": "stage2_observed_valid_smoke_draw_v1",
                    "epoch": epoch,
                    "draw_index": draw_index,
                    "worker_rank": draw_index % world_size,
                    "global_update": epoch * (len(rows) // world_size) + draw_index // world_size + 1,
                    **row,
                }
            )
    return schedule


def prepare(config_path: Path) -> None:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    sources = {name: resolve_path(value) for name, value in config["sources"].items()}
    phase66 = read_json(sources["phase66_contract"])
    phase67 = read_json(sources["phase67_contract"])
    if canonical_hash(phase66) != phase66.get("contract_sha256"):
        raise RuntimeError("Phase-66 contract hash is invalid")
    if canonical_hash(phase67) != phase67.get("contract_sha256"):
        raise RuntimeError("Phase-67 contract hash is invalid")
    for key, contract in (("phase66_artifact_manifest", phase66), ("phase67_artifact_manifest", phase67)):
        manifest = read_json(sources[key])
        verify_artifact_manifest(sources[key].parent, manifest)
        if manifest["contract_sha256"] != contract["contract_sha256"]:
            raise RuntimeError(f"{key} contract differs")
    parent = phase66["static_config"]
    if int(config["seed"]) != int(parent["seed"]) or int(config["world_size"]) != int(parent["world_size"]):
        raise RuntimeError("Experiment C seed/world size differs from B")
    if parent["training"]["loss"] != "plain_four_way_cross_entropy":
        raise RuntimeError("Phase-66 B reference loss differs")

    routes = _route_rows(config)
    states, occurrences = build_exact_valid_sets(routes)
    if len(occurrences) != 34253 or len(states) != 21071:
        raise RuntimeError("exact-state population differs from the diagnosed union")
    state_by_id = {row["state_id"]: row for row in states}
    parent_schedule = read_jsonl(sources["experiment_B_schedule"])
    if len(parent_schedule) != int(parent["training"]["total_global_optimizer_updates"]) * 4:
        raise RuntimeError("Phase-66 B schedule length differs")
    full_schedule = []
    for row in parent_schedule:
        state_ids, valid_actions, valid_indices = [], [], []
        for layer, target in zip(row["selected_layers"], row["selected_actions"]):
            state_id = exact_state_id(str(row["uid"]), int(layer), row["actions"])
            state = state_by_id.get(state_id)
            if state is None or str(target) not in state["observed_valid_actions"]:
                raise RuntimeError(f"schedule state/target is not observed-valid: {row['route_id']}:L{layer}")
            state_ids.append(state_id)
            valid_actions.append(state["observed_valid_actions"])
            valid_indices.append(state["observed_valid_action_indices"])
        full_schedule.append(
            {
                **row,
                "experiment": "C",
                "schema_version": "stage2_observed_valid_full_draw_v1",
                "selected_state_ids": state_ids,
                "observed_valid_actions": valid_actions,
                "observed_valid_action_indices": valid_indices,
            }
        )
    smoke_schedule = _choose_smoke_states(
        states,
        occurrences,
        routes,
        seed=int(config["seed"]) + 7000,
        per_category=int(config["smoke"]["states_per_category"]),
        epochs=int(config["smoke"]["epochs"]),
        world_size=int(config["world_size"]),
    )

    output_root.mkdir(parents=True)
    atomic_jsonl(output_root / "valid_sets/exact_state_index.jsonl", states)
    atomic_jsonl(
        output_root / "valid_sets/observed_valid_actions.jsonl",
        [
            {
                "state_id": row["state_id"],
                "uid": row["uid"],
                "layer": row["layer"],
                "route_sources": row["route_sources"],
                "observed_valid_actions": row["observed_valid_actions"],
                "observed_valid_action_indices": row["observed_valid_action_indices"],
                "valid_action_count": row["valid_action_count"],
                "contains_FULL": row["contains_FULL"],
                "contains_nonFULL": row["contains_nonFULL"],
            }
            for row in states
        ],
    )
    atomic_csv(output_root / "valid_sets/valid_set_distribution.csv", _distribution_rows(states))
    atomic_jsonl(output_root / "work/full_schedule.jsonl", full_schedule)
    atomic_jsonl(output_root / "work/smoke_schedule.jsonl", smoke_schedule)
    atomic_jsonl(output_root / "work/train_samples.jsonl", read_jsonl(sources["train_samples"]))
    atomic_jsonl(output_root / "work/validation_manifest.jsonl", read_jsonl(sources["validation_manifest"]))
    atomic_json(
        output_root / "training/config.yaml",
        {
            "experiment": "C",
            "changed_variable": {"from": "plain_four_way_cross_entropy", "to": config["loss"]},
            "model": parent["model"],
            "router": parent["router"],
            "sampling": parent["sampling"],
            "optimizer_and_budget": parent["training"],
            "seed": parent["seed"],
        },
    )
    audit = {
        "schema_version": "stage2_observed_valid_set_provenance_audit_v1",
        "passed": True,
        "routes": len(routes),
        "route_state_occurrences": len(occurrences),
        "unique_exact_states": len(states),
        "hash_collisions": 0,
        "empty_valid_sets": 0,
        "valid_action_count": dict(sorted(Counter(row["valid_action_count"] for row in states).items())),
        "full_schedule_rows": len(full_schedule),
        "full_schedule_core_matches_parent": all(
            all(
                key in {"experiment", "schema_version"} or new[key] == value
                for key, value in old.items()
            )
            for new, old in zip(full_schedule, parent_schedule)
        ),
        "full_schedule_states": sum(len(row["selected_layers"]) for row in full_schedule),
        "smoke_schedule_rows": len(smoke_schedule),
        "smoke_categories": dict(Counter(row["category"] for row in smoke_schedule if row["epoch"] == 0)),
        "state_identity": "sha256(UID + layer + complete entering action prefix)",
        "approximate_state_matching": False,
    }
    if not audit["full_schedule_core_matches_parent"]:
        raise RuntimeError("Experiment C schedule differs from Phase-66 B")
    atomic_json(output_root / "valid_sets/provenance_audit.json", audit)

    source_hashes = {name: file_sha256(path) for name, path in sources.items()}
    bound_hashes = {relative: file_sha256(resolve_path(relative)) for relative in BOUND_CODE}
    internal = {
        str(path.relative_to(output_root)): file_sha256(path)
        for path in sorted((output_root / "valid_sets").rglob("*"))
        if path.is_file()
    }
    internal.update(
        {
            relative: file_sha256(output_root / relative)
            for relative in (
                "work/full_schedule.jsonl",
                "work/smoke_schedule.jsonl",
                "work/train_samples.jsonl",
                "work/validation_manifest.jsonl",
                "training/config.yaml",
            )
        }
    )
    contract: dict[str, Any] = {
        "schema_version": "stage2_observed_valid_set_loss_contract_v1",
        "created_at": utc_now(),
        "static_config": config,
        "phase66_contract_sha256": phase66["contract_sha256"],
        "phase67_contract_sha256": phase67["contract_sha256"],
        "parent_static_config": parent,
        "source_sha256": source_hashes,
        "bound_code_sha256": bound_hashes,
        "internal_manifest_sha256": internal,
        "model_snapshot_sha256": phase66["model_snapshot_sha256"],
        "matched_change": {
            "only": "training objective and exact observed-valid target construction",
            "from": parent["training"]["loss"],
            "to": config["loss"],
            "unchanged_schedule_sha256": source_hashes["experiment_B_schedule"],
        },
        "population": audit,
        "git": {
            "commit": command_output(("git", "rev-parse", "HEAD")),
            "branch": command_output(("git", "branch", "--show-current")),
            "worktree_status_at_freeze": command_output(("git", "status", "--short")),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "cuda_runtime": torch.version.cuda,
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_protocol.json", contract)
    protocol = f"""# Stage-2 observed-valid-set loss protocol

- Contract: `{contract['contract_sha256']}`
- Parent B contract: `{phase66['contract_sha256']}`; diagnosis contract: `{phase67['contract_sha256']}`.
- Exact identity: UID + layer + complete entering routed-action prefix; no approximate state matching.
- Population: {len(states):,} unique states from {len(occurrences):,} successful route-state occurrences.
- Changed variable: plain single-label CE -> stable exact observed-valid-set logsumexp loss.
- Frozen: model/router, seed, optimizer, 3,144 updates, C:W and Single:MCTS draws, route/state schedule, Stage-1 thresholds, Historical-800 validation, executor, and LMMS scoring.
- Checkpoint: final global update only, matching Experiment B.
- Stop: B-vs-C oracle evaluation, C P98/P95/P90 rollout, and one unexecuted next decision; no test/search/threshold/architecture change.
"""
    _atomic_bytes(output_root / "protocol.md", protocol.encode())
    print(json.dumps({"prepared": True, "contract_sha256": contract["contract_sha256"], **audit}, sort_keys=True))


def verify_contract(config_path: Path, *, verify_model: bool = False) -> tuple[dict[str, Any], Path]:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if canonical_hash(contract) != contract.get("contract_sha256"):
        raise RuntimeError("Experiment C frozen contract hash mismatch")
    if contract["static_config"] != config:
        raise RuntimeError("active config differs from frozen Experiment C contract")
    for name, expected in contract["source_sha256"].items():
        if file_sha256(resolve_path(config["sources"][name])) != expected:
            raise RuntimeError(f"source hash mismatch: {name}")
    for relative, expected in contract["bound_code_sha256"].items():
        if file_sha256(resolve_path(relative)) != expected:
            raise RuntimeError(f"bound-code hash mismatch: {relative}")
    for relative, expected in contract["internal_manifest_sha256"].items():
        if file_sha256(output_root / relative) != expected:
            raise RuntimeError(f"internal manifest hash mismatch: {relative}")
    if verify_model:
        _verify_model_snapshot(
            resolve_path(contract["parent_static_config"]["model"]["snapshot_path"]),
            contract["model_snapshot_sha256"],
        )
    return contract, output_root


def _router_state_hash(router: torch.nn.Module) -> str:
    digest = sha256()
    for name, tensor in sorted(router.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def implementation_smoke(config_path: Path, device_index: int) -> None:
    contract, output_root = verify_contract(config_path, verify_model=True)
    config = contract["static_config"]
    parent = contract["parent_static_config"]
    before = _nvidia_memory(device_index)
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    configure_dense_determinism(int(config["seed"]), parent["backend_settings"])
    torch.cuda.reset_peak_memory_stats(device)
    processor, base, wrapped = _load_model(parent, device)
    samples = {row["uid"]: row for row in read_jsonl(output_root / "work/train_samples.jsonl")}
    schedule = read_jsonl(output_root / "work/full_schedule.jsonl")
    candidates = [row for row in schedule if any(len(values) > 1 for values in row["observed_valid_actions"])]
    row = max(candidates, key=lambda item: int(samples[item["uid"]]["dense_output"]["prompt_token_count"]))
    sample = samples[row["uid"]]["sample"]
    inputs, _ = build_dense_inputs(processor, sample, device)
    meta = _prepare_binary(processor, wrapped, sample, device)
    dense = capture_four_action_route(
        wrapped, {}, ["FULL"] * 28, prepared_inputs=meta, use_cache=True, native_full_rows=True
    )
    dense_ids, _dense_text, _dense_score = _generate(
        processor, wrapped, dense, inputs["input_ids"], sample
    )
    dense_parity = dense_ids == samples[row["uid"]]["dense_output"]["generated_token_ids"]
    del dense
    routed = capture_four_action_route(
        wrapped, {}, row["actions"], prepared_inputs=meta, use_cache=False, native_full_rows=True
    )
    torch.manual_seed(int(config["seed"]))
    router = _router(parent, device).train()
    initial_hash = _router_state_hash(router)
    text, visual, text_mask, visual_mask = _stack_route_states(routed, row["selected_layers"], device)
    mask = valid_action_mask(row["observed_valid_action_indices"], device=device)
    logits = router(text, visual, text_mask=text_mask, visual_mask=visual_mask)
    loss = observed_valid_set_loss(logits, mask)
    loss.backward()

    fixed_logits = torch.tensor(
        [[1.2, -0.3, 2.1, 0.4], [-1.0, 0.2, 0.5, 1.7]], device=device
    )
    fixed_targets = torch.tensor([2, 3], dtype=torch.long, device=device)
    fixed_mask = valid_action_mask([[2], [3]], device=device)
    set_value = observed_valid_set_loss(fixed_logits, fixed_mask)
    ce_value = F.cross_entropy(fixed_logits.float(), fixed_targets)
    equivalence_error = abs(float(set_value.item()) - float(ce_value.item()))
    equivalence = {
        "schema_version": "stage2_single_valid_ce_equivalence_v1",
        "contract_sha256": contract["contract_sha256"],
        "set_loss": float(set_value.item()),
        "cross_entropy": float(ce_value.item()),
        "absolute_error": equivalence_error,
        "tolerance": 1e-7,
        "passed": equivalence_error <= 1e-7,
    }
    atomic_json(output_root / "smoke/single_valid_ce_equivalence.json", equivalence)
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device) / 2**20)
    peak_reserved = int(torch.cuda.max_memory_reserved(device) / 2**20)
    checks = {
        "native_dense_token_parity": dense_parity,
        "single_valid_equals_ce": equivalence["passed"],
        "finite_logits": bool(torch.isfinite(logits).all()),
        "finite_loss": bool(torch.isfinite(loss)),
        "router_gradients": all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in router.parameters()),
        "backbone_frozen": all(parameter.grad is None for parameter in base.parameters()),
        "valid_sets_nonempty": bool(mask.any(dim=-1).all()),
        "memory_headroom": before["free_mib"] - peak_reserved >= 2048,
    }
    report = {
        "schema_version": "stage2_observed_valid_implementation_smoke_v1",
        "contract_sha256": contract["contract_sha256"],
        "passed": all(checks.values()),
        "checks": checks,
        "physical_device_index": _physical_device_index(device_index),
        "uid": row["uid"],
        "route_id": row["route_id"],
        "selected_layers": row["selected_layers"],
        "valid_action_counts": [len(value) for value in row["observed_valid_actions"]],
        "loss": float(loss.item()),
        "initialization_sha256": initial_hash,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "conservative_headroom_mib": before["free_mib"] - peak_reserved,
    }
    atomic_json(output_root / "smoke/implementation_smoke.json", report)
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise RuntimeError("Experiment C implementation smoke failed")


def _training_settings(contract: Mapping[str, Any], mode: str) -> dict[str, Any]:
    if mode == "full":
        return dict(contract["parent_static_config"]["training"])
    parent_overfit = contract["parent_static_config"]["overfit"]
    return {
        "epochs": int(contract["static_config"]["smoke"]["epochs"]),
        "learning_rate": parent_overfit["learning_rate"],
        "weight_decay": parent_overfit["weight_decay"],
        "gradient_clip_norm": parent_overfit["gradient_clip_norm"],
    }


def train_worker(config_path: Path, mode: str) -> None:
    if mode not in {"overfit", "full"}:
        raise ValueError("training mode must be overfit or full")
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    parent = contract["parent_static_config"]
    if not read_json(output_root / "smoke/implementation_smoke.json").get("passed"):
        raise RuntimeError("implementation smoke is not passing")
    if mode == "full" and not read_json(output_root / "smoke/overfit_gate.json").get("passed"):
        raise RuntimeError("observed-valid overfit gate is not passing")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != int(config["world_size"]):
        raise RuntimeError("Experiment C training requires four ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    configure_dense_determinism(int(config["seed"]) + rank, parent["backend_settings"])
    dist.init_process_group(backend="nccl")
    processor, base, wrapped = _load_model(parent, device)
    torch.manual_seed(int(config["seed"]) + (0 if mode == "full" else 5000))
    router = _router(parent, device)
    initial_hash = _router_state_hash(router)
    gathered_hashes: list[str | None] = [None] * world_size
    dist.all_gather_object(gathered_hashes, initial_hash)
    if len(set(gathered_hashes)) != 1:
        raise RuntimeError("router initialization differs across ranks")
    ddp = DistributedDataParallel(router, device_ids=[local_rank], output_device=local_rank)
    settings = _training_settings(contract, mode)
    optimizer = torch.optim.AdamW(
        ddp.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    schedule_all = read_jsonl(output_root / f"work/{'smoke' if mode == 'overfit' else 'full'}_schedule.jsonl")
    schedule = [row for row in schedule_all if int(row["worker_rank"]) == rank]
    epochs = int(settings["epochs"])
    updates_per_rank_epoch = len(schedule) // epochs
    expected = 18 * epochs if mode == "overfit" else int(parent["training"]["total_global_optimizer_updates"])
    if len(schedule) != expected or len(schedule) % epochs:
        raise RuntimeError(f"rank {rank} schedule length differs: {len(schedule)}")
    samples = {row["uid"]: row["sample"] for row in read_jsonl(output_root / "work/train_samples.jsonl")}
    log_path = output_root / ("smoke/overfit_train_log.jsonl" if mode == "overfit" else "training/train_log.jsonl")
    if rank == 0 and log_path.exists():
        raise RuntimeError(f"refusing to overwrite training log: {log_path}")
    dist.barrier()
    cache: dict[str, Any] = {}
    epoch_rows = []
    started = time.monotonic()
    for epoch in range(epochs):
        ddp.train()
        totals = torch.zeros(10, dtype=torch.float64, device=device)
        confusion = torch.zeros(4, 4, dtype=torch.long, device=device)
        category_totals: dict[str, torch.Tensor] = defaultdict(
            lambda: torch.zeros(6, dtype=torch.float64, device=device)
        )
        rows = schedule[epoch * updates_per_rank_epoch : (epoch + 1) * updates_per_rank_epoch]
        for row in rows:
            uid = str(row["uid"])
            if uid not in cache:
                cache[uid] = _binary_to(_prepare_binary(processor, wrapped, samples[uid], device), "cpu")
            meta = _binary_to(cache[uid], device)
            with torch.inference_mode():
                routed = capture_four_action_route(
                    wrapped, {}, row["actions"], prepared_inputs=meta, use_cache=False, native_full_rows=True
                )
            layers = row["selected_layers"] if mode == "full" else [row["layer"]]
            valid_indices = row["observed_valid_action_indices"] if mode == "full" else [row["observed_valid_action_indices"]]
            target_indices = row["selected_action_indices"] if mode == "full" else [row["target_action_index"]]
            text, visual, text_mask, visual_mask = _stack_route_states(routed, layers, device)
            mask = valid_action_mask(valid_indices, device=device)
            targets = torch.tensor(target_indices, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            if not bool(torch.isfinite(text).all()) or not bool(torch.isfinite(visual).all()):
                raise RuntimeError(f"non-finite routed state for {uid}")
            logits = ddp(text, visual, text_mask=text_mask, visual_mask=visual_mask)
            loss = observed_valid_set_loss(logits, mask)
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in ddp.parameters()):
                raise RuntimeError(f"non-finite router gradient for {uid}")
            torch.nn.utils.clip_grad_norm_(
                ddp.parameters(), float(settings["gradient_clip_norm"]), error_if_nonfinite=True
            )
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in ddp.parameters()):
                raise RuntimeError(f"non-finite router parameter after {uid}")
            measured = observed_valid_metrics(logits, mask)
            predictions = logits.detach().argmax(dim=-1)
            totals[0] += float(loss.item())
            totals[1] += 1
            totals[2] += measured["rows"]
            totals[3] += measured["observed_valid_top1"]
            totals[4] += measured["valid_probability_mass_sum"]
            totals[5] += measured["non_full_valid_mass_sum"]
            totals[6] += measured["contains_non_full_rows"]
            preservation = str(row["route_source"]) == "preservation_full"
            if preservation:
                totals[7] += len(predictions)
                totals[8] += int((predictions == 0).sum().item())
            totals[9] += int((predictions != 0).sum().item())
            for target, prediction in zip(targets, predictions):
                confusion[int(target), int(prediction)] += 1
            if mode == "overfit":
                category = str(row["category"])
                category_totals[category] += torch.tensor(
                    [
                        measured["rows"],
                        measured["observed_valid_top1"],
                        measured["valid_probability_mass_sum"],
                        measured["non_full_valid_mass_sum"],
                        measured["contains_non_full_rows"],
                        int((predictions == 0).sum().item()),
                    ],
                    dtype=torch.float64,
                    device=device,
                )
            del routed, text, visual, text_mask, visual_mask, mask, targets, logits, loss
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(confusion, op=dist.ReduceOp.SUM)
        categories = sorted({row.get("category", "") for row in schedule_all if row.get("category")})
        for category in categories:
            dist.all_reduce(category_totals[category], op=dist.ReduceOp.SUM)
        target_counts = confusion.sum(dim=1)
        recall = {
            action: (float(confusion[index, index]) / int(target_counts[index]) if int(target_counts[index]) else None)
            for index, action in enumerate(ACTION_NAMES)
        }
        row_count = float(totals[2].item())
        category_metrics = {}
        for category in categories:
            values = category_totals[category]
            count = float(values[0].item())
            category_metrics[category] = {
                "rows": int(count),
                "observed_valid_top1_accuracy": float(values[1].item() / count),
                "valid_probability_mass": float(values[2].item() / count),
                "non_full_valid_mass": float(values[3].item() / count),
                "full_prediction_fraction": float(values[5].item() / count),
            }
        epoch_row = {
            "schema_version": "stage2_observed_valid_training_epoch_v1",
            "contract_sha256": contract["contract_sha256"],
            "mode": mode,
            "epoch": epoch + 1,
            "global_update": (epoch + 1) * updates_per_rank_epoch,
            "loss": float(totals[0].item() / totals[1].item()),
            "state_rows": int(row_count),
            "observed_valid_top1_accuracy": float(totals[3].item() / row_count),
            "valid_probability_mass": float(totals[4].item() / row_count),
            "non_full_valid_mass": float(totals[5].item() / row_count),
            "predicted_non_full_fraction": float(totals[9].item() / row_count),
            "preservation_full_recall": float(totals[8].item() / totals[7].item()) if totals[7] else None,
            "nominal_recall": recall,
            "nominal_confusion": confusion.cpu().tolist(),
            "category_metrics": category_metrics,
            "initialization_sha256": initial_hash,
            "elapsed_seconds": time.monotonic() - started,
        }
        epoch_rows.append(epoch_row)
        if rank == 0:
            append_jsonl(log_path, epoch_row)
            print(json.dumps(epoch_row, sort_keys=True), flush=True)

    if not all(parameter.grad is None for parameter in base.parameters()):
        raise RuntimeError("frozen Qwen unexpectedly acquired gradients")
    dist.barrier()
    if rank == 0:
        checkpoint_path = output_root / ("smoke/overfit_checkpoint.pt" if mode == "overfit" else "training/final_checkpoint.pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "stage2_observed_valid_checkpoint_v1",
            "contract_sha256": contract["contract_sha256"],
            "experiment": "C",
            "mode": mode,
            "global_update": expected,
            "initialization_sha256": initial_hash,
            "router_config": parent["router"],
            "state_dict": {key: value.detach().cpu() for key, value in ddp.module.state_dict().items()},
        }
        temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp.{os.getpid()}")
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint_path)
        checkpoint_hash = file_sha256(checkpoint_path)
        if mode == "overfit":
            first, final = epoch_rows[0], epoch_rows[-1]
            smoke_config = config["smoke"]
            corrective_categories = ("single_corrective", "mcts_corrective")
            first_corrective_mass = sum(first["category_metrics"][name]["non_full_valid_mass"] for name in corrective_categories) / 2
            final_corrective_mass = sum(final["category_metrics"][name]["non_full_valid_mass"] for name in corrective_categories) / 2
            reduction = (first["loss"] - final["loss"]) / first["loss"]
            checks = {
                "loss_reduction": reduction >= float(smoke_config["minimum_loss_reduction"]),
                "valid_mass_gain": final["valid_probability_mass"] - first["valid_probability_mass"] >= float(smoke_config["minimum_valid_mass_gain"]),
                "corrective_non_full_mass_gain": final_corrective_mass - first_corrective_mass >= float(smoke_config["minimum_corrective_non_full_mass_gain"]),
                "observed_valid_top1": final["observed_valid_top1_accuracy"] >= float(smoke_config["minimum_final_observed_valid_top1"]),
                "preservation_full_recall": final["preservation_full_recall"] >= float(smoke_config["minimum_final_preservation_full_recall"]),
                "finite": all(math.isfinite(float(row["loss"])) for row in epoch_rows),
            }
            gate = {
                "schema_version": "stage2_observed_valid_overfit_gate_v1",
                "contract_sha256": contract["contract_sha256"],
                "passed": all(checks.values()),
                "checks": checks,
                "first_loss": first["loss"],
                "final_loss": final["loss"],
                "loss_reduction": reduction,
                "first_valid_mass": first["valid_probability_mass"],
                "final_valid_mass": final["valid_probability_mass"],
                "first_corrective_non_full_mass": first_corrective_mass,
                "final_corrective_non_full_mass": final_corrective_mass,
                "final_metrics": final,
                "checkpoint_sha256": checkpoint_hash,
            }
            atomic_json(output_root / "smoke/overfit_gate.json", gate)
            atomic_csv(
                output_root / "smoke/overfit_metrics.csv",
                [
                    {
                        "epoch": row["epoch"],
                        "loss": row["loss"],
                        "observed_valid_top1_accuracy": row["observed_valid_top1_accuracy"],
                        "valid_probability_mass": row["valid_probability_mass"],
                        "non_full_valid_mass": row["non_full_valid_mass"],
                        "preservation_full_recall": row["preservation_full_recall"],
                    }
                    for row in epoch_rows
                ],
            )
            atomic_csv(
                output_root / "smoke/overfit_valid_mass.csv",
                [
                    {
                        "epoch": row["epoch"],
                        "category": category,
                        **metrics,
                    }
                    for row in epoch_rows
                    for category, metrics in row["category_metrics"].items()
                ],
            )
            print(json.dumps(gate, sort_keys=True), flush=True)
        else:
            manifest = {
                "schema_version": "stage2_observed_valid_checkpoint_manifest_v1",
                "contract_sha256": contract["contract_sha256"],
                "checkpoints": [{"path": "training/final_checkpoint.pt", "sha256": checkpoint_hash, "global_update": expected, "selection_eligible": True}],
                "epoch_checkpoints": [],
            }
            atomic_json(output_root / "training/checkpoint_manifest.json", manifest)
            atomic_json(
                output_root / "training/selected_checkpoint.json",
                {
                    "schema_version": "stage2_observed_valid_selected_checkpoint_v1",
                    "contract_sha256": contract["contract_sha256"],
                    "selection_rule": "final_global_update_only",
                    "path": "training/final_checkpoint.pt",
                    "sha256": checkpoint_hash,
                    "global_update": expected,
                    "initialization_sha256": initial_hash,
                },
            )
    dist.barrier()
    atomic_json(
        output_root / f"work/{mode}_rank{rank:02d}.complete.json",
        {
            "passed": True,
            "rank": rank,
            "mode": mode,
            "updates": len(schedule),
            "contract_sha256": contract["contract_sha256"],
            "completed_at": utc_now(),
        },
    )
    dist.destroy_process_group()


def _rank_for_uid(uid: str, world_size: int) -> int:
    return int(sha256(str(uid).encode()).hexdigest()[:16], 16) % int(world_size)


def _load_selected_router(
    contract: Mapping[str, Any], output_root: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    selected = read_json(output_root / "training/selected_checkpoint.json")
    checkpoint_path = output_root / selected["path"]
    if file_sha256(checkpoint_path) != selected["sha256"]:
        raise RuntimeError("Experiment C selected checkpoint hash mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("contract_sha256") != contract["contract_sha256"] or payload.get("experiment") != "C":
        raise RuntimeError("Experiment C checkpoint provenance mismatch")
    router = _router(contract["parent_static_config"], device).eval()
    router.load_state_dict(payload["state_dict"], strict=True)
    return router, selected


def _require_full_training(contract: Mapping[str, Any], output_root: Path) -> None:
    for rank in range(int(contract["static_config"]["world_size"])):
        path = output_root / f"work/full_rank{rank:02d}.complete.json"
        if not path.is_file():
            raise RuntimeError(f"missing full-training completion for rank {rank}")
        row = read_json(path)
        if not row.get("passed") or row.get("contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError(f"invalid full-training completion for rank {rank}")


def oracle_worker(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    parent = contract["parent_static_config"]
    _require_full_training(contract, output_root)
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != int(config["world_size"]):
        raise RuntimeError("oracle evaluation requires four workers")
    result_path = output_root / f"oracle_eval/work/rank{rank:02d}.jsonl"
    complete_path = output_root / f"oracle_eval/work/rank{rank:02d}.complete.json"
    if result_path.exists() or complete_path.exists():
        raise RuntimeError(f"refusing to overwrite oracle rank {rank}")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    configure_dense_determinism(int(config["seed"]) + 20000 + rank, parent["backend_settings"])
    processor, _base, wrapped = _load_model(parent, device)
    router, selected = _load_selected_router(contract, output_root, device)
    states = {row["state_id"]: row for row in read_jsonl(output_root / "valid_sets/observed_valid_actions.jsonl")}
    samples = {row["uid"]: row["sample"] for row in read_jsonl(output_root / "work/train_samples.jsonl")}
    routes = [row for row in _route_rows(config) if _rank_for_uid(str(row["uid"]), world_size) == rank]
    rows = []
    started = time.monotonic()
    for route_index, route in enumerate(routes, 1):
        uid = str(route["uid"])
        meta = _prepare_binary(processor, wrapped, samples[uid], device)
        output = capture_four_action_route(
            wrapped, {}, route["actions"], prepared_inputs=meta, use_cache=False, native_full_rows=True
        )
        layers = list(range(int(route["activation_layer"]), 28))
        text, visual, text_mask, visual_mask = _stack_route_states(output, layers, device)
        with torch.inference_mode():
            logits = router(text, visual, text_mask=text_mask, visual_mask=visual_mask).float()
            probabilities = logits.softmax(dim=-1)
        for offset, layer in enumerate(layers):
            state_id = exact_state_id(uid, layer, route["actions"])
            state = states[state_id]
            mask = valid_action_mask([state["observed_valid_action_indices"]], device=device)[0]
            vector = probabilities[offset]
            prediction_index = int(vector.argmax().item())
            target_action = str(route["actions"][layer])
            target_index = ACTION_TO_INDEX[target_action]
            valid_mass = float(vector[mask].sum().item())
            non_full_mask = mask.clone()
            non_full_mask[0] = False
            non_full_mass = float(vector[non_full_mask].sum().item())
            index = intervention_index(route["actions"], layer, route["activation_layer"])
            rows.append(
                {
                    "schema_version": "stage2_observed_valid_oracle_state_v1",
                    "contract_sha256": contract["contract_sha256"],
                    "checkpoint": "C",
                    "checkpoint_sha256": selected["sha256"],
                    "route_id": route["route_id"],
                    "route_source": route["route_source"],
                    "uid": uid,
                    "dataset": route["dataset"],
                    "source_regime": route["source_regime"],
                    "valid_operating_points": route["valid_operating_points"],
                    "activation_layer": route["activation_layer"],
                    "layer": layer,
                    "state_id": state_id,
                    "target_action": target_action,
                    "predicted_action": ACTION_NAMES[prediction_index],
                    "nominal_target_match": prediction_index == target_index,
                    "observed_valid_top1": bool(mask[prediction_index]),
                    "observed_valid_actions": state["observed_valid_actions"],
                    "valid_action_count": state["valid_action_count"],
                    "contains_FULL": state["contains_FULL"],
                    "contains_nonFULL": state["contains_nonFULL"],
                    "target_probability": float(vector[target_index].item()),
                    "valid_probability_mass": valid_mass,
                    "non_full_valid_mass": non_full_mass,
                    "full_probability": float(vector[0].item()),
                    "best_non_full_probability": float(vector[1:].max().item()),
                    "full_vs_best_non_full_margin": float(vector[0].item() - vector[1:].max().item()),
                    "probabilities": {name: float(vector[index].item()) for index, name in enumerate(ACTION_NAMES)},
                    "route_non_full_count": route["non_full_count"],
                    "intervention_index": index,
                }
            )
        del output, meta, text, visual, text_mask, visual_mask, logits, probabilities
        if route_index % 100 == 0:
            print(json.dumps({"rank": rank, "routes": route_index, "assigned": len(routes), "states": len(rows), "elapsed_seconds": time.monotonic() - started}), flush=True)
    atomic_jsonl(result_path, rows)
    completion = {
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        "rank": rank,
        "routes": len(routes),
        "records": len(rows),
        "checkpoint_sha256": selected["sha256"],
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(complete_path, completion)
    print(json.dumps(completion, sort_keys=True), flush=True)


def rollout_worker(config_path: Path, point: str) -> None:
    if point not in OPERATING_POINTS:
        raise ValueError("rollout point must be P98/P95/P90")
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    parent = contract["parent_static_config"]
    _require_full_training(contract, output_root)
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != int(config["world_size"]):
        raise RuntimeError("free rollout requires four workers")
    rollout_root = output_root / "free_rollout" / point
    result_path = rollout_root / f"work/rank{rank:02d}.jsonl"
    complete_path = rollout_root / f"work/rank{rank:02d}.complete.json"
    if result_path.exists() or complete_path.exists():
        raise RuntimeError(f"refusing to overwrite C/{point} rank {rank}")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    configure_dense_determinism(int(config["seed"]) + 30000 + rank, parent["backend_settings"])
    processor, _base, wrapped = _load_model(parent, device)
    router, selected = _load_selected_router(contract, output_root, device)
    validation = read_jsonl(output_root / "work/validation_manifest.jsonl")
    triggered = sorted(
        [row for row in validation if row["trigger_layers"][point] is not None],
        key=lambda row: str(row["uid"]),
    )
    assigned = [row for index, row in enumerate(triggered) if index % world_size == rank]
    started = time.monotonic()
    for count, row in enumerate(assigned, 1):
        sample = row["sample"]
        inputs, _ = build_dense_inputs(processor, sample, device)
        meta = _prepare_binary(processor, wrapped, sample, device)
        trigger = int(row["trigger_layers"][point])
        chosen = []

        def selector(layer, text_states, visual_states, current_meta):
            if layer < trigger:
                chosen.append({"layer": layer, "action": "FULL", "active": False})
                return "FULL"
            with torch.inference_mode():
                logits = router(
                    text_states.detach().clone(),
                    visual_states.detach().clone(),
                    text_mask=current_meta.text_valid_mask.detach().clone(),
                    visual_mask=current_meta.visual_valid_mask.detach().clone(),
                )
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError(f"non-finite rollout logits: {row['uid']}:L{layer}")
                probabilities = logits.float().softmax(dim=-1)[0]
            action_index = int(probabilities.argmax().item())
            action = ACTION_NAMES[action_index]
            chosen.append(
                {
                    "layer": layer,
                    "action": action,
                    "active": True,
                    "probabilities": {name: float(probabilities[index].item()) for index, name in enumerate(ACTION_NAMES)},
                }
            )
            return action

        output = capture_online_four_action_route(
            wrapped, {}, selector, prepared_inputs=meta, use_cache=True, native_full_rows=True
        )
        token_ids, text, score = _generate(processor, wrapped, output, inputs["input_ids"], sample)
        active = [item for item in chosen if item["active"]]
        non_full = [item["layer"] for item in active if item["action"] != "FULL"]
        result = {
            "schema_version": "stage2_observed_valid_rollout_row_v1",
            "contract_sha256": contract["contract_sha256"],
            "experiment": "C",
            "operating_point": point,
            "threshold": float(parent["operating_points"][point]),
            "checkpoint_sha256": selected["sha256"],
            "uid": row["uid"],
            "dataset": row["dataset"],
            "source_regime": "historical",
            "triggered": True,
            "trigger_layer": trigger,
            "dense_correct": bool(row["dense_correct"]),
            "dense_wrong": bool(row["dense_wrong"]),
            "dense_generated_answer": row["dense_output"]["generated_answer"],
            "dense_generated_token_ids": row["dense_output"]["generated_token_ids"],
            "routed_generated_answer": text,
            "routed_generated_token_ids": token_ids,
            "lmms_metric": score.metric_name,
            "lmms_score": score.raw_score,
            "routed_correct": bool(score.correct),
            "routed_wrong": not bool(score.correct),
            "transition": ("W" if row["dense_wrong"] else "C") + "→" + ("C" if score.correct else "W"),
            "actions": [item["action"] for item in chosen],
            "action_rows": chosen,
            "post_trigger_action_counts": dict(Counter(item["action"] for item in active)),
            "post_trigger_actions": len(active),
            "non_full_count": len(non_full),
            "any_non_full": bool(non_full),
            "first_non_full_layer": non_full[0] if non_full else None,
            "trigger_to_first_non_full_delay": non_full[0] - trigger if non_full else None,
            "full_fraction_after_trigger": sum(item["action"] == "FULL" for item in active) / len(active),
            "worker_rank": rank,
        }
        append_jsonl(result_path, result)
        if count % 10 == 0:
            print(json.dumps({"point": point, "rank": rank, "completed": count, "assigned": len(assigned), "elapsed_seconds": time.monotonic() - started}), flush=True)
        del output, meta, inputs
    completion = {
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        "operating_point": point,
        "checkpoint_sha256": selected["sha256"],
        "rank": rank,
        "expected": len(assigned),
        "completed": len(assigned),
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(complete_path, completion)
    print(json.dumps(completion, sort_keys=True), flush=True)


def _normalize_phase67_row(row: Mapping[str, Any]) -> dict[str, Any]:
    valid_actions = [str(action) for action in row["observed_successful_actions"]]
    probabilities = {str(key): float(value) for key, value in row["probabilities"].items()}
    prediction = str(row["predicted_action"])
    return {
        "schema_version": "stage2_observed_valid_oracle_state_v1",
        "contract_sha256": str(row["contract_sha256"]),
        "checkpoint": str(row["checkpoint"]),
        "checkpoint_sha256": str(row["checkpoint_sha256"]),
        "route_id": str(row["route_id"]),
        "route_source": str(row["route_source"]),
        "uid": str(row["uid"]),
        "dataset": str(row["dataset"]),
        "source_regime": str(row["source_regime"]),
        "valid_operating_points": row["valid_operating_points"],
        "activation_layer": int(row["activation_layer"]),
        "layer": int(row["layer"]),
        "state_id": str(row["state_key"]),
        "target_action": str(row["target_action"]),
        "predicted_action": prediction,
        "nominal_target_match": bool(row["prediction_matches_target"]),
        "observed_valid_top1": prediction in valid_actions,
        "observed_valid_actions": valid_actions,
        "valid_action_count": int(row["observed_successful_action_count"]),
        "contains_FULL": "FULL" in valid_actions,
        "contains_nonFULL": any(action != "FULL" for action in valid_actions),
        "target_probability": float(row["target_probability"]),
        "valid_probability_mass": float(row["observed_set_probability"]),
        "non_full_valid_mass": sum(probabilities[action] for action in valid_actions if action != "FULL"),
        "full_probability": float(row["full_probability"]),
        "best_non_full_probability": float(row["best_non_full_probability"]),
        "full_vs_best_non_full_margin": float(row["full_vs_best_non_full_margin"]),
        "probabilities": probabilities,
        "route_non_full_count": int(row["route_non_full_count"]),
        "intervention_index": row["intervention_index"],
    }


def _oracle_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"states": 0}
    corrective = [row for row in rows if row["target_action"] != "FULL"]
    return {
        "states": len(rows),
        "uids": len({row["uid"] for row in rows}),
        "nominal_top1_accuracy": sum(row["nominal_target_match"] for row in rows) / len(rows),
        "observed_valid_top1_accuracy": sum(row["observed_valid_top1"] for row in rows) / len(rows),
        "mean_valid_probability_mass": sum(float(row["valid_probability_mass"]) for row in rows) / len(rows),
        "mean_non_full_valid_mass": sum(float(row["non_full_valid_mass"]) for row in rows) / len(rows),
        "mean_full_probability": sum(float(row["full_probability"]) for row in rows) / len(rows),
        "mean_full_vs_best_non_full_margin": sum(float(row["full_vs_best_non_full_margin"]) for row in rows) / len(rows),
        "corrective_states": len(corrective),
        "nominal_non_full_recall": (
            sum(row["nominal_target_match"] for row in corrective) / len(corrective)
            if corrective
            else None
        ),
        "observed_valid_non_full_recall": (
            sum(row["observed_valid_top1"] and row["predicted_action"] != "FULL" for row in corrective)
            / len(corrective)
            if corrective
            else None
        ),
        "full_recall": (
            sum(row["predicted_action"] == "FULL" for row in rows if row["target_action"] == "FULL")
            / sum(row["target_action"] == "FULL" for row in rows)
            if any(row["target_action"] == "FULL" for row in rows)
            else None
        ),
    }


def _aggregate_rollout(
    contract: Mapping[str, Any], output_root: Path, point: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = contract["static_config"]
    validation = read_jsonl(output_root / "work/validation_manifest.jsonl")
    expected = [row for row in validation if row["trigger_layers"][point] is not None]
    worker_rows = []
    for rank in range(int(config["world_size"])):
        complete = read_json(output_root / f"free_rollout/{point}/work/rank{rank:02d}.complete.json")
        rows = read_jsonl(output_root / f"free_rollout/{point}/work/rank{rank:02d}.jsonl")
        if not complete.get("passed") or len(rows) != int(complete["expected"]):
            raise RuntimeError(f"C/{point} rank {rank} is incomplete")
        worker_rows.extend(rows)
    if len(worker_rows) != len(expected) or len({row["uid"] for row in worker_rows}) != len(worker_rows):
        raise RuntimeError(f"C/{point} triggered rollout coverage differs")
    routed = {row["uid"]: row for row in worker_rows}
    per_sample = []
    selected = read_json(output_root / "training/selected_checkpoint.json")
    for row in validation:
        uid = str(row["uid"])
        if uid in routed:
            per_sample.append(routed[uid])
            continue
        dense_correct = bool(row["dense_correct"])
        per_sample.append(
            {
                "schema_version": "stage2_observed_valid_rollout_row_v1",
                "contract_sha256": contract["contract_sha256"],
                "experiment": "C",
                "operating_point": point,
                "threshold": float(contract["parent_static_config"]["operating_points"][point]),
                "checkpoint_sha256": selected["sha256"],
                "uid": uid,
                "dataset": row["dataset"],
                "source_regime": "historical",
                "triggered": False,
                "trigger_layer": None,
                "dense_correct": dense_correct,
                "dense_wrong": not dense_correct,
                "dense_generated_answer": row["dense_output"]["generated_answer"],
                "dense_generated_token_ids": row["dense_output"]["generated_token_ids"],
                "routed_generated_answer": row["dense_output"]["generated_answer"],
                "routed_generated_token_ids": row["dense_output"]["generated_token_ids"],
                "lmms_metric": row["dense_output"]["lmms_eval_metric"],
                "lmms_score": row["dense_output"]["lmms_eval_per_sample_score"],
                "routed_correct": dense_correct,
                "routed_wrong": not dense_correct,
                "transition": "C→C" if dense_correct else "W→W",
                "actions": ["FULL"] * 28,
                "action_rows": [],
                "post_trigger_action_counts": {},
                "post_trigger_actions": 0,
                "non_full_count": 0,
                "any_non_full": False,
                "first_non_full_layer": None,
                "trigger_to_first_non_full_delay": None,
                "full_fraction_after_trigger": None,
                "worker_rank": None,
            }
        )
    per_sample.sort(key=lambda row: str(row["uid"]))
    if len(per_sample) != 800 or len({row["uid"] for row in per_sample}) != 800:
        raise RuntimeError(f"C/{point} final 800-row coverage differs")
    atomic_jsonl(output_root / f"free_rollout/{point}/per_sample_results.jsonl", per_sample)
    transitions = Counter(row["transition"] for row in per_sample)
    triggered = [row for row in per_sample if row["triggered"]]
    active_rows = [item for row in triggered for item in row["action_rows"] if item["active"]]
    active_counts = Counter(item["action"] for item in active_rows)
    non_full_rows = [row for row in triggered if row["any_non_full"]]
    dense_correct = sum(row["dense_correct"] for row in per_sample)
    routed_correct = sum(row["routed_correct"] for row in per_sample)
    metrics = {
        "schema_version": "stage2_observed_valid_rollout_metrics_v1",
        "contract_sha256": contract["contract_sha256"],
        "experiment": "C",
        "operating_point": point,
        "threshold": float(contract["parent_static_config"]["operating_points"][point]),
        "checkpoint_sha256": selected["sha256"],
        "validation_scope": "historical_frozen_val_800",
        "validation_samples": len(per_sample),
        "triggered_samples": len(triggered),
        "triggered_w": sum(row["dense_wrong"] for row in triggered),
        "triggered_c": sum(row["dense_correct"] for row in triggered),
        "dense_correct": dense_correct,
        "routed_correct": routed_correct,
        "dense_accuracy": dense_correct / len(per_sample),
        "routed_accuracy": routed_correct / len(per_sample),
        "delta_accuracy": (routed_correct - dense_correct) / len(per_sample),
        "transitions": {name: transitions[name] for name in ("W→C", "W→W", "C→C", "C→W")},
        "net_corrections": transitions["W→C"] - transitions["C→W"],
        "w_to_c_rescue_rate": transitions["W→C"] / 400,
        "c_to_c_preservation_rate": transitions["C→C"] / 400,
        "triggered_any_non_full_fraction": len(non_full_rows) / len(triggered),
        "post_trigger_action_counts": dict(active_counts),
        "post_trigger_action_distribution": {
            action: active_counts[action] / len(active_rows) for action in ACTION_NAMES
        },
        "mean_non_full_actions_per_triggered": sum(row["non_full_count"] for row in triggered) / len(triggered),
        "mean_first_non_full_layer": (
            sum(row["first_non_full_layer"] for row in non_full_rows) / len(non_full_rows)
            if non_full_rows
            else None
        ),
        "mean_trigger_to_first_non_full_delay": (
            sum(row["trigger_to_first_non_full_delay"] for row in non_full_rows) / len(non_full_rows)
            if non_full_rows
            else None
        ),
    }
    atomic_json(output_root / f"free_rollout/{point}/metrics.json", metrics)
    return metrics, per_sample


def _bar(path: Path, labels: Sequence[str], values: Sequence[float], ylabel: str) -> None:
    plt.figure(figsize=(max(6, len(labels) * 1.1), 4.2))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def aggregate(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    _require_full_training(contract, output_root)
    c_rows = []
    for rank in range(int(config["world_size"])):
        complete = read_json(output_root / f"oracle_eval/work/rank{rank:02d}.complete.json")
        rows = read_jsonl(output_root / f"oracle_eval/work/rank{rank:02d}.jsonl")
        if not complete.get("passed") or len(rows) != int(complete["records"]):
            raise RuntimeError(f"C oracle rank {rank} is incomplete")
        c_rows.extend(rows)
    if len(c_rows) != 34253 or len({(row["route_id"], row["layer"]) for row in c_rows}) != 34253:
        raise RuntimeError("C oracle-state coverage differs")
    phase67_rows = []
    for rank in range(4):
        phase67_rows.extend(read_jsonl(resolve_path(config["sources"][f"phase67_oracle_rank{rank}"])))
    old_rows = [_normalize_phase67_row(row) for row in phase67_rows]
    if len(old_rows) != 68506:
        raise RuntimeError("Phase-67 A/B oracle-state coverage differs")
    a_rows = [row for row in old_rows if row["checkpoint"] == "A"]
    b_rows = [row for row in old_rows if row["checkpoint"] == "B"]
    c_keys = {(row["route_id"], row["layer"]) for row in c_rows}
    if c_keys != {(row["route_id"], row["layer"]) for row in b_rows}:
        raise RuntimeError("B/C oracle-state keys differ")
    atomic_jsonl(output_root / "oracle_eval/per_state_results.jsonl", b_rows + c_rows)

    all_rows = {"A": a_rows, "B": b_rows, "C": c_rows}
    overall_rows = []
    for checkpoint in ("B", "C"):
        for source in ("overall", *ROUTE_SOURCES):
            subset = all_rows[checkpoint] if source == "overall" else [row for row in all_rows[checkpoint] if row["route_source"] == source]
            overall_rows.append({"checkpoint": checkpoint, "route_source": source, **_oracle_metrics(subset)})
    atomic_csv(output_root / "oracle_eval/overall_metrics.csv", overall_rows)

    validity_rows = []
    for checkpoint in ("B", "C"):
        for category, predicate in (
            ("single_valid", lambda row: int(row["valid_action_count"]) == 1),
            ("multi_valid", lambda row: int(row["valid_action_count"]) >= 2),
        ):
            subset = [row for row in all_rows[checkpoint] if predicate(row)]
            validity_rows.append({"checkpoint": checkpoint, "validity_category": category, **_oracle_metrics(subset)})
    atomic_csv(output_root / "oracle_eval/validity_count_breakdown.csv", validity_rows)

    intervention_rows = []
    for checkpoint in ("B", "C"):
        mcts = [row for row in all_rows[checkpoint] if row["route_source"] == "mcts"]
        for category, predicate in (
            ("first", lambda row: row["intervention_index"] == 1),
            ("later", lambda row: row["intervention_index"] is not None and int(row["intervention_index"]) >= 2),
        ):
            subset = [row for row in mcts if predicate(row)]
            intervention_rows.append({"checkpoint": checkpoint, "intervention_index": category, **_oracle_metrics(subset)})
    atomic_csv(output_root / "oracle_eval/intervention_index_breakdown.csv", intervention_rows)

    single_rows = {
        checkpoint: [row for row in values if row["route_source"] == "single" and row["target_action"] != "FULL"]
        for checkpoint, values in all_rows.items()
    }
    single_transfer = [{"checkpoint": checkpoint, **_oracle_metrics(single_rows[checkpoint])} for checkpoint in ("A", "B", "C")]
    atomic_csv(output_root / "oracle_eval/single_negative_transfer.csv", single_transfer)
    paired_source = b_rows + c_rows
    bootstrap_rows = []
    for metric in ("nominal_target_match", "observed_valid_top1", "valid_probability_mass"):
        eligible = [row for row in paired_source if row["route_source"] == "single" and row["target_action"] != "FULL"]
        pairs = uid_paired_values(eligible, metric, "B", "C")
        bootstrap_rows.append(
            {"metric": metric, **paired_uid_bootstrap(pairs, seed=int(config["seed"]) + len(metric), replicates=int(config["bootstrap_replicates"]))}
        )
    atomic_csv(output_root / "oracle_eval/paired_bootstrap.csv", bootstrap_rows)

    abc_oracle = []
    for checkpoint in ("A", "B", "C"):
        single = _oracle_metrics(single_rows[checkpoint])
        mcts = [row for row in all_rows[checkpoint] if row["route_source"] == "mcts"]
        first = _oracle_metrics([row for row in mcts if row["intervention_index"] == 1])
        later = _oracle_metrics([row for row in mcts if row["intervention_index"] is not None and int(row["intervention_index"]) >= 2])
        abc_oracle.append(
            {
                "experiment": checkpoint,
                "loss": "single-only CE" if checkpoint == "A" else ("single+MCTS CE" if checkpoint == "B" else "single+MCTS observed-valid-set"),
                "single_nominal_non_full_recall": single["nominal_non_full_recall"],
                "single_observed_valid_non_full_recall": single["observed_valid_non_full_recall"],
                "single_valid_probability_mass": single["mean_valid_probability_mass"],
                "mcts_nominal_non_full_recall": _oracle_metrics(mcts)["nominal_non_full_recall"],
                "mcts_observed_valid_non_full_recall": _oracle_metrics(mcts)["observed_valid_non_full_recall"],
                "mcts_valid_probability_mass": _oracle_metrics(mcts)["mean_valid_probability_mass"],
                "mcts_first_nominal_recall": first["nominal_non_full_recall"],
                "mcts_first_observed_valid_recall": first["observed_valid_non_full_recall"],
                "mcts_later_nominal_recall": later["nominal_non_full_recall"],
                "mcts_later_observed_valid_recall": later["observed_valid_non_full_recall"],
            }
        )
    atomic_csv(output_root / "comparison/A_B_C_oracle.csv", abc_oracle)

    c_rollouts = {}
    c_samples = {}
    for point in OPERATING_POINTS:
        c_rollouts[point], c_samples[point] = _aggregate_rollout(contract, output_root, point)
    threshold_rows = []
    behavior_rows = []
    abc_rollout = []
    for point in OPERATING_POINTS:
        metrics = c_rollouts[point]
        transitions = metrics["transitions"]
        threshold_rows.append(
            {
                "operating_point": point,
                "threshold": metrics["threshold"],
                "samples": 800,
                "triggered": metrics["triggered_samples"],
                "triggered_w": metrics["triggered_w"],
                "triggered_c": metrics["triggered_c"],
                "dense_accuracy": metrics["dense_accuracy"],
                "routed_accuracy": metrics["routed_accuracy"],
                "delta_accuracy": metrics["delta_accuracy"],
                "w_to_c": transitions["W→C"],
                "w_to_w": transitions["W→W"],
                "c_to_c": transitions["C→C"],
                "c_to_w": transitions["C→W"],
                "net_correction": metrics["net_corrections"],
                "rescue_rate": metrics["w_to_c_rescue_rate"],
                "preservation_rate": metrics["c_to_c_preservation_rate"],
            }
        )
        behavior_rows.append(
            {
                "operating_point": point,
                "triggered": metrics["triggered_samples"],
                "any_non_full_fraction": metrics["triggered_any_non_full_fraction"],
                "post_trigger_full_fraction": metrics["post_trigger_action_distribution"]["FULL"],
                "mean_non_full_actions": metrics["mean_non_full_actions_per_triggered"],
                "mean_first_non_full_layer": metrics["mean_first_non_full_layer"],
                "mean_trigger_to_first_non_full_delay": metrics["mean_trigger_to_first_non_full_delay"],
                **{f"{action.lower()}_count": metrics["post_trigger_action_counts"].get(action, 0) for action in ACTION_NAMES},
            }
        )
        for experiment in ("A", "B"):
            prior = read_jsonl(resolve_path(config["sources"][f"rollout_{experiment}_{point}"]))
            transitions_prior = Counter(row["transition"] for row in prior)
            triggered_prior = [row for row in prior if row["triggered"]]
            abc_rollout.append(
                {
                    "experiment": experiment,
                    "loss": "single-only CE" if experiment == "A" else "single+MCTS CE",
                    "operating_point": point,
                    "w_to_c": transitions_prior["W→C"],
                    "c_to_w": transitions_prior["C→W"],
                    "net_correction": transitions_prior["W→C"] - transitions_prior["C→W"],
                    "any_non_full_fraction": sum(row["any_non_full"] for row in triggered_prior) / len(triggered_prior),
                    "routed_accuracy": sum(row["routed_correct"] for row in prior) / len(prior),
                }
            )
        abc_rollout.append(
            {
                "experiment": "C",
                "loss": "single+MCTS observed-valid-set",
                "operating_point": point,
                "w_to_c": transitions["W→C"],
                "c_to_w": transitions["C→W"],
                "net_correction": metrics["net_corrections"],
                "any_non_full_fraction": metrics["triggered_any_non_full_fraction"],
                "routed_accuracy": metrics["routed_accuracy"],
            }
        )
    atomic_csv(output_root / "free_rollout/threshold_comparison.csv", threshold_rows)
    atomic_csv(output_root / "free_rollout/action_behavior.csv", behavior_rows)
    atomic_csv(output_root / "comparison/A_B_C_rollout.csv", abc_rollout)

    output_root.joinpath("figures").mkdir(parents=True, exist_ok=True)
    distribution = Counter(row["valid_action_count"] for row in read_jsonl(output_root / "valid_sets/exact_state_index.jsonl"))
    _bar(output_root / "figures/valid_set_size_distribution.png", [str(i) for i in range(1, 5)], [distribution[i] for i in range(1, 5)], "Unique exact states")
    _bar(output_root / "figures/oracle_nonfull_recall_ABC.png", ["A single", "B single", "C single", "B MCTS", "C MCTS"], [abc_oracle[0]["single_nominal_non_full_recall"], abc_oracle[1]["single_nominal_non_full_recall"], abc_oracle[2]["single_nominal_non_full_recall"], abc_oracle[1]["mcts_nominal_non_full_recall"], abc_oracle[2]["mcts_nominal_non_full_recall"]], "Nominal non-FULL recall")
    multi = {row["checkpoint"]: row for row in validity_rows if row["validity_category"] == "multi_valid"}
    _bar(output_root / "figures/multi_valid_accuracy_B_vs_C.png", ["B", "C"], [multi["B"]["observed_valid_top1_accuracy"], multi["C"]["observed_valid_top1_accuracy"]], "Multi-valid top-1 in observed set")
    _bar(output_root / "figures/single_negative_transfer_B_vs_C.png", ["A", "B", "C"], [row["observed_valid_non_full_recall"] for row in single_transfer], "Single corrective observed-valid recall")
    p90 = [row for row in abc_rollout if row["operating_point"] == "P90"]
    labels = [row["experiment"] for row in p90]
    positions = range(len(labels))
    plt.figure(figsize=(6, 4.2))
    plt.bar([x - 0.18 for x in positions], [row["w_to_c"] for row in p90], 0.36, label="W→C")
    plt.bar([x + 0.18 for x in positions], [row["c_to_w"] for row in p90], 0.36, label="C→W")
    plt.xticks(list(positions), labels)
    plt.ylabel("Historical validation samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "figures/p90_rescue_regression_ABC.png", dpi=160)
    plt.close()
    plt.figure(figsize=(6, 4.2))
    for checkpoint in ("B", "C"):
        values = [next(row for row in validity_rows if row["checkpoint"] == checkpoint and row["validity_category"] == category)["mean_valid_probability_mass"] for category in ("single_valid", "multi_valid")]
        plt.plot(["single-valid", "multi-valid"], values, marker="o", label=checkpoint)
    plt.ylabel("Observed-valid probability mass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "figures/valid_probability_mass.png", dpi=160)
    plt.close()

    b_all = {row["route_source"]: row for row in overall_rows if row["checkpoint"] == "B"}
    c_all = {row["route_source"]: row for row in overall_rows if row["checkpoint"] == "C"}
    b_single = next(row for row in single_transfer if row["checkpoint"] == "B")
    c_single = next(row for row in single_transfer if row["checkpoint"] == "C")
    b_first = next(row for row in intervention_rows if row["checkpoint"] == "B" and row["intervention_index"] == "first")
    c_first = next(row for row in intervention_rows if row["checkpoint"] == "C" and row["intervention_index"] == "first")
    b_later = next(row for row in intervention_rows if row["checkpoint"] == "B" and row["intervention_index"] == "later")
    c_later = next(row for row in intervention_rows if row["checkpoint"] == "C" and row["intervention_index"] == "later")
    p90_c = c_rollouts["P90"]
    thresholds = config["decision_thresholds"]
    oracle_mcts_gain = c_all["mcts"]["observed_valid_non_full_recall"] - b_all["mcts"]["observed_valid_non_full_recall"]
    mcts_mass_gain = c_all["mcts"]["mean_valid_probability_mass"] - b_all["mcts"]["mean_valid_probability_mass"]
    single_gain = c_single["observed_valid_non_full_recall"] - b_single["observed_valid_non_full_recall"]
    multi_gain = multi["C"]["observed_valid_top1_accuracy"] - multi["B"]["observed_valid_top1_accuracy"]
    single_valid = {row["checkpoint"]: row for row in validity_rows if row["validity_category"] == "single_valid"}
    single_valid_gain = single_valid["C"]["observed_valid_top1_accuracy"] - single_valid["B"]["observed_valid_top1_accuracy"]
    oracle_improved = oracle_mcts_gain >= float(thresholds["meaningful_oracle_recall_gain"]) or mcts_mass_gain >= float(thresholds["meaningful_valid_mass_gain"])
    single_recovered = single_gain >= float(thresholds["meaningful_oracle_recall_gain"])
    p90_positive = p90_c["net_corrections"] >= int(thresholds["positive_rollout_net_min"])
    regression_heavy = p90_c["transitions"]["C→W"] > p90_c["transitions"]["W→C"]
    if regression_heavy:
        decision_case = "E"
        recommendation = ("conservative action-selection calibration", "test one fixed post-router abstention margin without changing Stage-1 thresholds", "whether valid-set learning is useful but needs a safer intervention boundary")
    elif oracle_improved and single_recovered and p90_positive:
        decision_case = "A"
        recommendation = ("small partial-prefix on-policy collection", "collect only first-deviation states under the fixed C policy", "whether remaining free-rollout errors are exposure-driven after the objective repair")
    elif oracle_improved and not p90_positive:
        decision_case = "B"
        recommendation = ("small partial-prefix on-policy collection", "collect only first-deviation states under the fixed C policy", "whether exposure shift explains the oracle-to-rollout gap")
    elif single_recovered and oracle_mcts_gain < float(thresholds["meaningful_oracle_recall_gain"]):
        decision_case = "C"
        recommendation = ("bounded MCTS-state overfit diagnostic", "overfit an action-balanced exact-state MCTS cohort with the unchanged router", "whether remaining MCTS failure is optimization or representation limited")
    else:
        decision_case = "D"
        recommendation = ("bounded MCTS-state overfit diagnostic", "overfit an action-balanced exact-state MCTS cohort with the unchanged router", "whether the architecture can separate MCTS corrective actions when ambiguity is removed")

    summary = f"""# Stage-2 observed-valid-set loss result

## Frozen matched experiment

Experiment C used the exact Phase-66 B model, router, seed, optimizer, 3,144-update schedule, sampler, union routes, Stage-1 thresholds, Historical-800 validation set, executor, and LMMS evaluator. The only training change was one-hot CE to exact-prefix observed-valid-set logsumexp loss.

## Answers

1. Exact routed states with 1/2/3/4 observed-valid actions: **19,772 / 785 / 377 / 137**.
2. Overall observed-valid top-1 accuracy B→C: **{b_all['overall']['observed_valid_top1_accuracy']:.4f} → {c_all['overall']['observed_valid_top1_accuracy']:.4f}**.
3. Overall observed-valid probability mass B→C: **{b_all['overall']['mean_valid_probability_mass']:.4f} → {c_all['overall']['mean_valid_probability_mass']:.4f}**; MCTS gain {mcts_mass_gain:+.4f}.
4. MCTS first-intervention nominal / observed-valid recall B→C: **{b_first['nominal_non_full_recall']:.4f}/{b_first['observed_valid_non_full_recall']:.4f} → {c_first['nominal_non_full_recall']:.4f}/{c_first['observed_valid_non_full_recall']:.4f}**.
5. MCTS later-intervention nominal / observed-valid recall B→C: **{b_later['nominal_non_full_recall']:.4f}/{b_later['observed_valid_non_full_recall']:.4f} → {c_later['nominal_non_full_recall']:.4f}/{c_later['observed_valid_non_full_recall']:.4f}**.
6. Single corrective nominal / observed-valid recall B→C: **{b_single['nominal_non_full_recall']:.4f}/{b_single['observed_valid_non_full_recall']:.4f} → {c_single['nominal_non_full_recall']:.4f}/{c_single['observed_valid_non_full_recall']:.4f}**; A nominal reference is 0.1078.
7. Observed-valid top-1 gain is {multi_gain:+.4f} on multi-valid states versus {single_valid_gain:+.4f} on single-valid states.
8. MCTS FULL-minus-best-non-FULL margin B→C: **{b_all['mcts']['mean_full_vs_best_non_full_margin']:.4f} → {c_all['mcts']['mean_full_vs_best_non_full_margin']:.4f}**.
9. P90 C rollout: W→C/C→W/net = **{p90_c['transitions']['W→C']}/{p90_c['transitions']['C→W']}/{p90_c['net_corrections']:+d}**, any non-FULL {p90_c['triggered_any_non_full_fraction']:.2%}, routed accuracy {p90_c['routed_accuracy']:.6f}.
10. W→C recovery relative to B: **{p90_c['transitions']['W→C']} versus 0**.
11. C→W under C: **{p90_c['transitions']['C→W']}** versus B's 0.
12. Fixed decision case: **{decision_case}**. Oracle MCTS observed-valid recall gain is {oracle_mcts_gain:+.4f}; single corrective gain is {single_gain:+.4f}. This determines whether label ambiguity is strengthened as a causal bottleneck or only a descriptive correlate.
13. This experiment cannot establish unseen-source/test generalization, exhaustive action validity, globally optimal MCTS routes, or whether a different architecture/on-policy distribution would work.

## Scope

No test set, new search, Stage-1 change, threshold change, architecture change, layer embedding, Stage-1 latent, class weighting, focal loss, RL, or on-policy training was run.
"""
    _atomic_bytes(output_root / "summaries/observed_valid_set_loss_summary.md", summary.encode())
    decision = f"""# Next Stage-2 decision

## One next step

**{recommendation[0]}**

- Why it matters: Experiment C resolves the isolated objective hypothesis under fixed data and architecture; the remaining largest uncertainty should now be tested without combining changes.
- Remaining hypothesis: {recommendation[2]}.
- Fixed implementation: {recommendation[1]}.
- Positive result: the targeted oracle or free-rollout deficit improves while preservation remains acceptable.
- Negative result: this mechanism is insufficient under the current router and the next decision should revisit the remaining diagnosed limitation rather than repeat it.
- Cannot conclude: deployment benefit, held-out test performance, unseen-source generalization, or impossibility of sequential correction.

This recommendation is not executed in Phase 68.
"""
    _atomic_bytes(output_root / "summaries/next_stage2_decision.md", decision.encode())

    required = [
        "protocol.md", "valid_sets/exact_state_index.jsonl", "valid_sets/observed_valid_actions.jsonl",
        "valid_sets/valid_set_distribution.csv", "valid_sets/provenance_audit.json",
        "smoke/single_valid_ce_equivalence.json", "smoke/overfit_metrics.csv", "smoke/overfit_valid_mass.csv",
        "training/config.yaml", "training/train_log.jsonl", "training/checkpoint_manifest.json", "training/selected_checkpoint.json",
        "oracle_eval/per_state_results.jsonl", "oracle_eval/overall_metrics.csv", "oracle_eval/validity_count_breakdown.csv",
        "oracle_eval/intervention_index_breakdown.csv", "oracle_eval/single_negative_transfer.csv", "oracle_eval/paired_bootstrap.csv",
        "free_rollout/P98/per_sample_results.jsonl", "free_rollout/P98/metrics.json",
        "free_rollout/P95/per_sample_results.jsonl", "free_rollout/P95/metrics.json",
        "free_rollout/P90/per_sample_results.jsonl", "free_rollout/P90/metrics.json",
        "free_rollout/threshold_comparison.csv", "free_rollout/action_behavior.csv",
        "comparison/A_B_C_oracle.csv", "comparison/A_B_C_rollout.csv",
        "figures/valid_set_size_distribution.png", "figures/oracle_nonfull_recall_ABC.png",
        "figures/multi_valid_accuracy_B_vs_C.png", "figures/single_negative_transfer_B_vs_C.png",
        "figures/p90_rescue_regression_ABC.png", "figures/valid_probability_mass.png",
        "summaries/observed_valid_set_loss_summary.md", "summaries/next_stage2_decision.md",
    ]
    missing = [relative for relative in required if not (output_root / relative).is_file()]
    if missing:
        raise RuntimeError(f"required Experiment C artifacts missing: {missing}")
    files = {
        str(path.relative_to(output_root)): file_sha256(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    manifest = {
        "schema_version": "stage2_observed_valid_set_artifact_manifest_v1",
        "passed": True,
        "completed_at": utc_now(),
        "contract_sha256": contract["contract_sha256"],
        "decision_case": decision_case,
        "recommended_next_step": recommendation[0],
        "oracle_records_B_C": len(b_rows) + len(c_rows),
        "validation_records_per_point": 800,
        "files": files,
    }
    atomic_json(output_root / "artifact_manifest.json", manifest)
    verify_artifact_manifest(output_root, manifest)
    print(json.dumps({**manifest, "files": len(files)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "implementation-smoke", "train", "oracle-worker", "rollout", "aggregate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("overfit", "full"))
    parser.add_argument("--point", choices=OPERATING_POINTS)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    if args.command == "prepare":
        prepare(config_path)
    elif args.command == "implementation-smoke":
        implementation_smoke(config_path, args.device_index)
    elif args.command == "train":
        if not args.mode:
            parser.error("train requires --mode")
        train_worker(config_path, args.mode)
    elif args.command == "oracle-worker":
        oracle_worker(config_path)
    elif args.command == "rollout":
        if not args.point:
            parser.error("rollout requires --point")
        rollout_worker(config_path, args.point)
    else:
        aggregate(config_path)


if __name__ == "__main__":
    main()
