#!/usr/bin/env python3
"""Frozen Stage-2 treatment-selectivity representation diagnostic."""

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
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from binary_policy.executor.four_action import capture_four_action_route  # noqa: E402
from dense_failure_stage1.runtime import configure_dense_determinism  # noqa: E402
from dense_failure_stage2.full_label_generation import verify_artifact_manifest  # noqa: E402
from dense_failure_stage2.observed_valid_set import (  # noqa: E402
    build_exact_valid_sets,
    exact_state_id,
)
from dense_failure_stage2.treatment_selectivity import (  # noqa: E402
    LABEL_INTERVENE,
    LABEL_KEEP,
    LABEL_MIXED,
    LABEL_UNRESOLVED,
    assign_group_folds,
    binary_metrics,
    classify_observed_actions,
    exact_router_representations,
    fit_binary_probe,
    matched_evaluation_weights,
    predict_binary_probe,
    selective_operating_points,
    standardize_fold,
    training_weights,
    validate_extracted_state_rows,
)
from dense_failure_stage2.v1_router import ACTION_NAMES  # noqa: E402
from experiments.run_stage2_shared_union_training import (  # noqa: E402
    _binary_to,
    _load_model,
    _prepare_binary,
    _router,
    _stack_route_states,
    _verify_model_snapshot,
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


DEFAULT_CONFIG = PROJECT_ROOT / "configs/stage2_treatment_selectivity_separability_v1.json"
ROUTE_SOURCES = ("preservation_full", "single", "mcts")
PROBE_NAMES = ("M0_margin", "M1_logits", "R_read", "W_write", "RW_read_write")
BOUND_CODE = (
    "configs/stage2_treatment_selectivity_separability_v1.json",
    "dense_failure_stage2/treatment_selectivity.py",
    "dense_failure_stage2/observed_valid_set.py",
    "dense_failure_stage2/v1_router.py",
    "experiments/analyze_stage2_treatment_selectivity.py",
    "experiments/run_stage2_shared_union_training.py",
    "experiments/run_stage2_v1_training_revised.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/model.py",
    "dense_failure_stage1/runtime.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "stage2_treatment_selectivity_separability_config_v1":
        raise ValueError("unsupported treatment-selectivity config")
    if int(config["world_size"]) != 4 or int(config["folds"]) != 5:
        raise ValueError("the frozen diagnostic requires four extraction workers and five folds")
    coverages = [float(value) for value in config["support"]["coverage_fractions"]]
    if coverages != [0.05, 0.10, 0.20]:
        raise ValueError("coverage operating points differ from the plan")
    return config


def _layer_bin(layer: int) -> str:
    if int(layer) <= 8:
        return "early_0_8"
    if int(layer) <= 18:
        return "middle_9_18"
    return "late_19_27"


def _route_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for key, expected in (
        ("union_preservation", "preservation_full"),
        ("union_single", "single"),
        ("union_mcts", "mcts"),
    ):
        rows = read_jsonl(resolve_path(config["sources"][key]))
        if any(str(row["route_source"]) != expected for row in rows):
            raise RuntimeError(f"route source mismatch: {key}")
        output.extend(rows)
    if len(output) != 2519 or len({str(row["route_id"]) for row in output}) != 2519:
        raise RuntimeError("frozen successful-route population differs from Phase 66")
    return output


def _state_summary_rows(states: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    dimensions: list[tuple[str, str, list[Mapping[str, Any]]]] = [("overall", "all", list(states))]
    for field in ("dataset", "source_regime", "layer", "representative_trigger_depth_bin"):
        for value in sorted({str(row[field]) for row in states}):
            dimensions.append((field, value, [row for row in states if str(row[field]) == value]))
    for route_source in ROUTE_SOURCES:
        dimensions.append(
            (
                "route_source_membership",
                route_source,
                [row for row in states if route_source in row["route_sources"]],
            )
        )
    output = []
    for dimension, value, rows in dimensions:
        counts = Counter(str(row["state_label"]) for row in rows)
        for label in (LABEL_KEEP, LABEL_INTERVENE, LABEL_MIXED, LABEL_UNRESOLVED):
            output.append(
                {
                    "breakdown": dimension,
                    "value": value,
                    "label": label,
                    "states": counts[label],
                    "unique_uids": len({str(row["uid"]) for row in rows if row["state_label"] == label}),
                }
            )
    return output


def _assign_extraction_workers(
    exact_states: Sequence[Mapping[str, Any]], routes: Sequence[Mapping[str, Any]], world_size: int
) -> list[list[dict[str, Any]]]:
    route_by_id = {str(row["route_id"]): dict(row) for row in routes}
    states_by_route: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    uid_state_counts = Counter(str(row["uid"]) for row in exact_states)
    for state in exact_states:
        route_id = str(state["representative_route_id"])
        if route_id not in route_by_id:
            raise RuntimeError(f"representative route is missing: {route_id}")
        states_by_route[route_id].append(state)
    rank_loads = [0] * world_size
    uid_rank: dict[str, int] = {}
    for uid, count in sorted(uid_state_counts.items(), key=lambda item: (-item[1], item[0])):
        rank = min(range(world_size), key=lambda value: (rank_loads[value], value))
        uid_rank[uid] = rank
        rank_loads[rank] += count
    schedules: list[list[dict[str, Any]]] = [[] for _ in range(world_size)]
    for route_id, selected in sorted(states_by_route.items()):
        route = route_by_id[route_id]
        uid = str(route["uid"])
        ordered = sorted(selected, key=lambda row: int(row["layer"]))
        layers = [int(row["layer"]) for row in ordered]
        state_ids = [str(row["state_id"]) for row in ordered]
        for layer, state_id in zip(layers, state_ids, strict=True):
            if exact_state_id(uid, layer, route["actions"]) != state_id:
                raise RuntimeError(f"representative route does not reconstruct exact state {state_id}")
        schedules[uid_rank[uid]].append(
            {
                "schema_version": "stage2_treatment_selectivity_extraction_route_v1",
                "route_id": route_id,
                "route_source": str(route["route_source"]),
                "uid": uid,
                "dataset": str(route["dataset"]),
                "source_regime": str(route["source_regime"]),
                "activation_layer": int(route["activation_layer"]),
                "actions": list(route["actions"]),
                "selected_layers": layers,
                "selected_state_ids": state_ids,
                "expected_states": len(state_ids),
                "worker_rank": uid_rank[uid],
            }
        )
    if [sum(row["expected_states"] for row in schedule) for schedule in schedules] != rank_loads:
        raise RuntimeError("worker state loads do not match schedules")
    return schedules


def prepare(config_path: Path) -> None:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing nonempty output root: {output_root}")
    sources = {name: resolve_path(value) for name, value in config["sources"].items()}
    phase66 = read_json(sources["phase66_contract"])
    phase68 = read_json(sources["phase68_contract"])
    if canonical_hash(phase66) != phase66.get("contract_sha256"):
        raise RuntimeError("Phase-66 contract hash is invalid")
    if canonical_hash(phase68) != phase68.get("contract_sha256"):
        raise RuntimeError("Phase-68 contract hash is invalid")
    for name, contract in (
        ("phase66_artifact_manifest", phase66),
        ("phase68_artifact_manifest", phase68),
    ):
        manifest = read_json(sources[name])
        verify_artifact_manifest(sources[name].parent, manifest)
        if str(manifest["contract_sha256"]) != str(contract["contract_sha256"]):
            raise RuntimeError(f"{name} is bound to a different contract")

    parent = phase66["static_config"]
    _verify_model_snapshot(
        resolve_path(parent["model"]["snapshot_path"]), phase66["model_snapshot_sha256"]
    )
    selection = read_json(sources["experiment_a_checkpoint_selection"])
    checkpoint_sha = file_sha256(sources["experiment_a_checkpoint"])
    if (
        selection.get("experiment") != "A"
        or selection.get("selection_rule") != "final_global_update_only"
        or str(selection.get("sha256")) != checkpoint_sha
    ):
        raise RuntimeError("Experiment-A checkpoint selection is invalid")
    checkpoint = torch.load(sources["experiment_a_checkpoint"], map_location="cpu", weights_only=False)
    if (
        checkpoint.get("contract_sha256") != phase66["contract_sha256"]
        or checkpoint.get("experiment") != "A"
        or int(checkpoint.get("global_update", -1)) != 3144
    ):
        raise RuntimeError("Experiment-A checkpoint provenance differs")

    routes = _route_rows(config)
    rebuilt_states, occurrences = build_exact_valid_sets(routes)
    source_states = read_jsonl(sources["exact_state_index"])
    if len(rebuilt_states) != 21071 or len(occurrences) != 34253 or rebuilt_states != source_states:
        raise RuntimeError("exact-state index does not reproduce from frozen successful routes")
    samples = {str(row["uid"]): row for row in read_jsonl(sources["train_samples"])}
    if len(samples) != 569:
        raise RuntimeError("frozen union sample population differs")

    exact_states: list[dict[str, Any]] = []
    for state in source_states:
        uid = str(state["uid"])
        sample_row = samples.get(uid)
        if sample_row is None:
            raise RuntimeError(f"exact state has no sample payload: {uid}")
        sample = sample_row["sample"]
        label = classify_observed_actions(state["observed_valid_actions"])
        route_signature = "+".join(str(value) for value in state["route_sources"])
        layer = int(state["layer"])
        trigger_layer = int(state["representative_activation_layer"])
        row = {
            "schema_version": "stage2_treatment_selectivity_exact_state_v1",
            "state_id": str(state["state_id"]),
            "uid": uid,
            "dataset": str(sample_row["dataset"]),
            "source_regime": str(sample_row["source_regime"]),
            "image_group_id": str(sample["image_group_id"]),
            "image_content_sha256": str(sample["image_content_sha256"]),
            "layer": layer,
            "layer_bin": _layer_bin(layer),
            "representative_trigger_layer": trigger_layer,
            "representative_trigger_depth_bin": _layer_bin(trigger_layer),
            "prefix_actions": list(state["prefix_actions"]),
            "prefix_sha256": sha256("|".join(state["prefix_actions"]).encode()).hexdigest(),
            "observed_valid_actions": list(state["observed_valid_actions"]),
            "observed_valid_action_indices": list(state["observed_valid_action_indices"]),
            "state_label": label,
            "binary_target": 0 if label == LABEL_KEEP else 1 if label == LABEL_INTERVENE else None,
            "route_sources": list(state["route_sources"]),
            "route_source_signature": route_signature,
            "route_ids": list(state["route_ids"]),
            "representative_route_id": str(state["representative_route_id"]),
            "match_cell": "|".join(
                (
                    str(sample_row["dataset"]),
                    str(sample_row["source_regime"]),
                    _layer_bin(layer),
                    route_signature,
                )
            ),
        }
        exact_states.append(row)
    exact_states.sort(key=lambda row: str(row["state_id"]))
    clean = [row for row in exact_states if row["state_label"] in {LABEL_KEEP, LABEL_INTERVENE}]
    mixed = [row for row in exact_states if row["state_label"] == LABEL_MIXED]
    unresolved = [row for row in exact_states if row["state_label"] == LABEL_UNRESOLVED]
    counts = Counter(str(row["state_label"]) for row in exact_states)
    if counts != Counter({LABEL_KEEP: 18438, LABEL_INTERVENE: 1657, LABEL_MIXED: 976}):
        raise RuntimeError(f"clean state-label population differs: {counts}")
    if unresolved:
        raise RuntimeError("unexpected unresolved states in successful-route index")

    clean_folds = assign_group_folds(clean, folds=int(config["folds"]), seed=int(config["seed"]))
    fold_by_state = {str(row["state_id"]): int(row["fold"]) for row in clean_folds}
    fold_by_group: dict[str, int] = {}
    for row in clean_folds:
        group = str(row["image_group_id"])
        if group in fold_by_group and fold_by_group[group] != int(row["fold"]):
            raise RuntimeError("image group crosses clean folds")
        fold_by_group[group] = int(row["fold"])
    fold_loads = Counter(fold_by_group.values())
    for state in mixed:
        group = str(state["image_group_id"])
        if group not in fold_by_group:
            fold = min(range(int(config["folds"])), key=lambda value: (fold_loads[value], value))
            fold_by_group[group] = fold
            fold_loads[fold] += 1
        fold_by_state[str(state["state_id"])] = fold_by_group[group]
    fold_manifest = [
        {
            "schema_version": "stage2_treatment_selectivity_fold_v1",
            "state_id": str(row["state_id"]),
            "uid": str(row["uid"]),
            "image_group_id": str(row["image_group_id"]),
            "state_label": str(row["state_label"]),
            "fit_eligible": row["state_label"] in {LABEL_KEEP, LABEL_INTERVENE},
            "fold": fold_by_state[str(row["state_id"])],
        }
        for row in exact_states
    ]
    for group in {str(row["image_group_id"]) for row in fold_manifest}:
        if len({int(row["fold"]) for row in fold_manifest if str(row["image_group_id"]) == group}) != 1:
            raise RuntimeError("image group crosses final folds")

    schedules = _assign_extraction_workers(exact_states, routes, int(config["world_size"]))
    output_root.mkdir(parents=True)
    for directory in (
        "dataset",
        "features/work",
        "metrics",
        "figures",
        "summaries",
        "smoke",
        "work",
        "probes/margin_baseline",
        "probes/logits_linear",
        "probes/read_linear",
        "probes/write_linear",
        "probes/read_write_linear",
        "probes/read_write_mlp_optional",
    ):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output_root / "dataset/exact_state_manifest.jsonl", exact_states)
    atomic_jsonl(output_root / "dataset/clean_binary_states.jsonl", clean)
    atomic_jsonl(output_root / "dataset/mixed_states.jsonl", mixed)
    atomic_jsonl(output_root / "dataset/fold_manifest.jsonl", fold_manifest)
    atomic_csv(output_root / "dataset/dataset_summary.csv", _state_summary_rows(exact_states))
    uid_rows = []
    for uid in sorted({str(row["uid"]) for row in exact_states}):
        rows = [row for row in exact_states if str(row["uid"]) == uid]
        labels = Counter(str(row["state_label"]) for row in rows)
        uid_rows.append(
            {
                "uid": uid,
                "dataset": rows[0]["dataset"],
                "source_regime": rows[0]["source_regime"],
                "image_group_id": rows[0]["image_group_id"],
                "states": len(rows),
                "KEEP_REQUIRED": labels[LABEL_KEEP],
                "INTERVENE_REQUIRED": labels[LABEL_INTERVENE],
                "MIXED": labels[LABEL_MIXED],
            }
        )
    atomic_csv(output_root / "dataset/uid_state_counts.csv", uid_rows)
    for rank, schedule in enumerate(schedules):
        atomic_jsonl(output_root / f"work/extraction_rank{rank:02d}.jsonl", schedule)

    feature_schema = """# Frozen representation schema

- Exact state timing: immediately before the indexed decoder layer under the complete routed prefix.
- `z_R`: 256-wide FP32 output of Experiment-A `read_attention` at its single text-query position.
- `z_W`: 256-wide FP32 output of Experiment-A `write_attention` at its learned write-query position.
- `action_logits`: four FP32 logits in `[FULL, READ_ONLY, WRITE_ONLY, IGNORE]` order from the unchanged Experiment-A action head over `[z_R; z_W]`.
- `nonfull_full_margin`: `max(action_logits[1:]) - action_logits[0]`.
- No layer, dataset, source, Stage-1, visual-count, label, or route-result feature is concatenated to the learned probe inputs.
- Branch extraction mirrors the frozen router forward and is accepted only after exact-logit parity.
"""
    _atomic_text(output_root / "features/feature_schema.md", feature_schema)

    source_hashes = {name: file_sha256(path) for name, path in sources.items()}
    bound_hashes = {relative: file_sha256(resolve_path(relative)) for relative in BOUND_CODE}
    internal_paths = [
        "dataset/exact_state_manifest.jsonl",
        "dataset/clean_binary_states.jsonl",
        "dataset/mixed_states.jsonl",
        "dataset/dataset_summary.csv",
        "dataset/uid_state_counts.csv",
        "dataset/fold_manifest.jsonl",
        "features/feature_schema.md",
        *[f"work/extraction_rank{rank:02d}.jsonl" for rank in range(int(config["world_size"]))],
    ]
    internal_hashes = {relative: file_sha256(output_root / relative) for relative in internal_paths}
    contract: dict[str, Any] = {
        "schema_version": "stage2_treatment_selectivity_separability_contract_v1",
        "created_at": utc_now(),
        "static_config": config,
        "phase66_contract_sha256": phase66["contract_sha256"],
        "phase68_contract_sha256": phase68["contract_sha256"],
        "experiment_a_checkpoint_sha256": checkpoint_sha,
        "model_snapshot_sha256": phase66["model_snapshot_sha256"],
        "source_sha256": source_hashes,
        "bound_code_sha256": bound_hashes,
        "internal_manifest_sha256": internal_hashes,
        "population": {
            "successful_routes": len(routes),
            "route_state_occurrences": len(occurrences),
            "unique_exact_states": len(exact_states),
            "unique_uids": len({str(row["uid"]) for row in exact_states}),
            "clean_binary_states": len(clean),
            "KEEP_REQUIRED": counts[LABEL_KEEP],
            "INTERVENE_REQUIRED": counts[LABEL_INTERVENE],
            "MIXED": counts[LABEL_MIXED],
            "UNRESOLVED": counts[LABEL_UNRESOLVED],
            "extraction_states_per_rank": [
                sum(int(row["expected_states"]) for row in schedule) for schedule in schedules
            ],
            "extraction_routes_per_rank": [len(schedule) for schedule in schedules],
        },
        "frozen_components": {
            "model": parent["model"],
            "stage1_operating_points": parent["operating_points"],
            "router": parent["router"],
            "router_checkpoint_selection": selection,
            "state_identity": "sha256(UID + layer + complete entering action prefix)",
            "labels": {
                "KEEP_REQUIRED": "observed successful action set equals {FULL}",
                "INTERVENE_REQUIRED": "FULL absent and at least one non-FULL action observed successful",
                "MIXED": "FULL and at least one non-FULL action observed successful",
                "search_miss_as_KEEP": False,
            },
        },
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
            "execution": "direct_four_gpu_exact_route_replay",
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_protocol.json", contract)
    atomic_json(
        output_root / "features/representation_manifest.json",
        {
            "schema_version": "stage2_treatment_selectivity_representation_manifest_v1",
            "contract_sha256": contract["contract_sha256"],
            "status": "pending_extraction",
            "expected_states": len(exact_states),
        },
    )
    _atomic_text(
        output_root / "protocol.md",
        f"""# Stage-2 treatment-selectivity separability protocol

- Contract: `{contract['contract_sha256']}`
- Frozen checkpoint: Experiment A final update 3,144, `{checkpoint_sha}`.
- Population: {len(exact_states):,} unique exact prefix-states from {len(occurrences):,} successful route-state occurrences and 569 UIDs.
- Labels: {counts[LABEL_KEEP]:,} KEEP_REQUIRED, {counts[LABEL_INTERVENE]:,} INTERVENE_REQUIRED, {counts[LABEL_MIXED]:,} MIXED; no search miss is labeled KEEP.
- Evaluation: five UID/image-group-disjoint folds; UID- and class-balanced training, unbalanced held-out metrics, fold-local normalization.
- Inputs: scalar margin, four logits, z_R, z_W, and [z_R;z_W]. No layer/dataset/source/Stage-1 inputs.
- Matched sensitivity: reweight held-out KEEP/INTERVENE within supported dataset/source/layer-bin/route-source-signature cells.
- Scope: diagnostic probes only. No deployment head, router retraining, search, Stage-1 change, or external evaluation.
""",
    )
    print(json.dumps({"prepared": True, "contract_sha256": contract["contract_sha256"], **contract["population"]}, sort_keys=True))


def verify_contract(config_path: Path, *, verify_model: bool = False) -> tuple[dict[str, Any], Path]:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if canonical_hash(contract) != contract.get("contract_sha256"):
        raise RuntimeError("treatment-selectivity frozen contract hash mismatch")
    if contract["static_config"] != config:
        raise RuntimeError("active config differs from frozen contract")
    for name, expected in contract["source_sha256"].items():
        if file_sha256(resolve_path(config["sources"][name])) != str(expected):
            raise RuntimeError(f"source hash mismatch: {name}")
    for relative, expected in contract["bound_code_sha256"].items():
        if file_sha256(resolve_path(relative)) != str(expected):
            raise RuntimeError(f"bound code hash mismatch: {relative}")
    for relative, expected in contract["internal_manifest_sha256"].items():
        if file_sha256(output_root / relative) != str(expected):
            raise RuntimeError(f"internal input hash mismatch: {relative}")
    if verify_model:
        parent = read_json(resolve_path(config["sources"]["phase66_contract"]))["static_config"]
        _verify_model_snapshot(
            resolve_path(parent["model"]["snapshot_path"]), contract["model_snapshot_sha256"]
        )
    return contract, output_root


def _load_frozen_router(contract: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    config = contract["static_config"]
    parent = read_json(resolve_path(config["sources"]["phase66_contract"]))["static_config"]
    router = _router(parent, device).eval()
    checkpoint_path = resolve_path(config["sources"]["experiment_a_checkpoint"])
    if file_sha256(checkpoint_path) != contract["experiment_a_checkpoint_sha256"]:
        raise RuntimeError("Experiment-A checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    router.load_state_dict(checkpoint["state_dict"], strict=True)
    return router


def smoke(config_path: Path, device_index: int) -> None:
    contract, output_root = verify_contract(config_path, verify_model=True)
    device = torch.device(f"cuda:{int(device_index)}")
    torch.cuda.set_device(device)
    parent = read_json(resolve_path(contract["static_config"]["sources"]["phase66_contract"]))[
        "static_config"
    ]
    configure_dense_determinism(int(contract["static_config"]["seed"]), parent["backend_settings"])
    processor, base, wrapped = _load_model(parent, device)
    router = _load_frozen_router(contract, device)
    samples = {
        str(row["uid"]): row["sample"]
        for row in read_jsonl(output_root.parent / "shared_union_training/work/train_samples.jsonl")
    }
    schedules = []
    for rank in range(4):
        schedules.extend(read_jsonl(output_root / f"work/extraction_rank{rank:02d}.jsonl")[:1])
    checks = []
    with torch.inference_mode():
        for row in schedules:
            meta = _prepare_binary(processor, wrapped, samples[str(row["uid"])], device)
            routed = capture_four_action_route(
                wrapped,
                {},
                row["actions"],
                prepared_inputs=meta,
                use_cache=False,
                native_full_rows=True,
            )
            text, visual, text_mask, visual_mask = _stack_route_states(
                routed, row["selected_layers"], device
            )
            expected = router(text, visual, text_mask=text_mask, visual_mask=visual_mask)
            read, write, logits = exact_router_representations(
                router, text, visual, text_mask=text_mask, visual_mask=visual_mask
            )
            state_ids = [
                exact_state_id(str(row["uid"]), int(layer), row["actions"])
                for layer in row["selected_layers"]
            ]
            checks.append(
                {
                    "uid": row["uid"],
                    "route_id": row["route_id"],
                    "states": len(state_ids),
                    "state_identity_exact": state_ids == row["selected_state_ids"],
                    "logit_parity_exact": bool(torch.equal(expected, logits)),
                    "finite": bool(torch.isfinite(read).all() and torch.isfinite(write).all() and torch.isfinite(logits).all()),
                    "read_shape": list(read.shape),
                    "write_shape": list(write.shape),
                    "logit_shape": list(logits.shape),
                }
            )
            del routed, text, visual, text_mask, visual_mask, expected, read, write, logits, meta
    passed = all(
        row["state_identity_exact"]
        and row["logit_parity_exact"]
        and row["finite"]
        and row["read_shape"][1] == 256
        and row["write_shape"][1] == 256
        and row["logit_shape"][1] == 4
        for row in checks
    )
    report = {
        "schema_version": "stage2_treatment_selectivity_extraction_smoke_v1",
        "contract_sha256": contract["contract_sha256"],
        "passed": passed,
        "device_index": int(device_index),
        "routes": len(checks),
        "states": sum(int(row["states"]) for row in checks),
        "checks": checks,
        "backbone_gradients_all_none": all(parameter.grad is None for parameter in base.parameters()),
    }
    report["passed"] = bool(report["passed"] and report["backbone_gradients_all_none"])
    atomic_json(output_root / "smoke/extraction_smoke.json", report)
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise RuntimeError("representation extraction smoke failed")


def extract_worker(config_path: Path, rank: int) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    rank = int(rank)
    if rank < 0 or rank >= int(contract["static_config"]["world_size"]):
        raise ValueError("worker rank is outside frozen world size")
    smoke_report = read_json(output_root / "smoke/extraction_smoke.json")
    if not smoke_report.get("passed") or smoke_report.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("passing extraction smoke is required")
    shard_path = output_root / f"features/work/rank{rank:02d}.pt"
    index_path = output_root / f"features/work/rank{rank:02d}.jsonl"
    complete_path = output_root / f"features/work/rank{rank:02d}.complete.json"
    if shard_path.exists() or index_path.exists() or complete_path.exists():
        raise RuntimeError(f"refusing to overwrite extraction worker {rank}")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    parent = read_json(resolve_path(contract["static_config"]["sources"]["phase66_contract"]))[
        "static_config"
    ]
    configure_dense_determinism(int(contract["static_config"]["seed"]) + rank, parent["backend_settings"])
    processor, _base, wrapped = _load_model(parent, device)
    router = _load_frozen_router(contract, device)
    samples = {
        str(row["uid"]): row["sample"]
        for row in read_jsonl(output_root.parent / "shared_union_training/work/train_samples.jsonl")
    }
    schedule = read_jsonl(output_root / f"work/extraction_rank{rank:02d}.jsonl")
    expected_states = sum(int(row["expected_states"]) for row in schedule)
    cached_inputs: dict[str, Any] = {}
    state_ids: list[str] = []
    read_rows: list[torch.Tensor] = []
    write_rows: list[torch.Tensor] = []
    logit_rows: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    index_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    with torch.inference_mode():
        for route_index, row in enumerate(schedule, 1):
            uid = str(row["uid"])
            if uid not in cached_inputs:
                cached_inputs[uid] = _binary_to(
                    _prepare_binary(processor, wrapped, samples[uid], device), "cpu"
                )
            meta = _binary_to(cached_inputs[uid], device)
            routed = capture_four_action_route(
                wrapped,
                {},
                row["actions"],
                prepared_inputs=meta,
                use_cache=False,
                native_full_rows=True,
            )
            text, visual, text_mask, visual_mask = _stack_route_states(
                routed, row["selected_layers"], device
            )
            expected = router(text, visual, text_mask=text_mask, visual_mask=visual_mask)
            read, write, logits = exact_router_representations(
                router, text, visual, text_mask=text_mask, visual_mask=visual_mask
            )
            if not torch.equal(expected, logits):
                raise RuntimeError(f"exact branch/logit parity failed: {row['route_id']}")
            if not bool(torch.isfinite(read).all() and torch.isfinite(write).all() and torch.isfinite(logits).all()):
                raise RuntimeError(f"non-finite representation: {row['route_id']}")
            margin = logits[:, 1:].max(dim=1).values - logits[:, 0]
            start = len(state_ids)
            state_ids.extend(map(str, row["selected_state_ids"]))
            read_rows.append(read.detach().cpu().float())
            write_rows.append(write.detach().cpu().float())
            logit_rows.append(logits.detach().cpu().float())
            margins.append(margin.detach().cpu().float())
            for offset, (state_id, layer) in enumerate(
                zip(row["selected_state_ids"], row["selected_layers"], strict=True)
            ):
                index_rows.append(
                    {
                        "schema_version": "stage2_treatment_selectivity_feature_index_v1",
                        "contract_sha256": contract["contract_sha256"],
                        "checkpoint_sha256": contract["experiment_a_checkpoint_sha256"],
                        "state_id": str(state_id),
                        "uid": uid,
                        "layer": int(layer),
                        "representative_route_id": str(row["route_id"]),
                        "worker_rank": rank,
                        "tensor_row": start + offset,
                    }
                )
            del routed, text, visual, text_mask, visual_mask, expected, read, write, logits, margin, meta
            if route_index % 50 == 0 or route_index == len(schedule):
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "routes_complete": route_index,
                            "routes_total": len(schedule),
                            "states_complete": len(state_ids),
                            "states_total": expected_states,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    validate_extracted_state_rows(
        [state_id for row in schedule for state_id in row["selected_state_ids"]], index_rows
    )
    payload = {
        "schema_version": "stage2_treatment_selectivity_feature_shard_v1",
        "contract_sha256": contract["contract_sha256"],
        "checkpoint_sha256": contract["experiment_a_checkpoint_sha256"],
        "feature_schema_sha256": file_sha256(output_root / "features/feature_schema.md"),
        "worker_rank": rank,
        "state_ids": state_ids,
        "z_R": torch.cat(read_rows, dim=0),
        "z_W": torch.cat(write_rows, dim=0),
        "action_logits": torch.cat(logit_rows, dim=0),
        "nonfull_full_margin": torch.cat(margins, dim=0),
    }
    _atomic_torch(shard_path, payload)
    atomic_jsonl(index_path, index_rows)
    complete = {
        "schema_version": "stage2_treatment_selectivity_extraction_complete_v1",
        "contract_sha256": contract["contract_sha256"],
        "checkpoint_sha256": contract["experiment_a_checkpoint_sha256"],
        "passed": True,
        "worker_rank": rank,
        "routes": len(schedule),
        "states": len(state_ids),
        "unique_uids": len({str(row["uid"]) for row in schedule}),
        "feature_shard": str(shard_path.relative_to(output_root)),
        "feature_shard_sha256": file_sha256(shard_path),
        "index": str(index_path.relative_to(output_root)),
        "index_sha256": file_sha256(index_path),
        "elapsed_seconds": time.monotonic() - started,
        "completed_at": utc_now(),
    }
    atomic_json(complete_path, complete)
    print(json.dumps(complete, sort_keys=True), flush=True)


def aggregate(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    expected_rows = read_jsonl(output_root / "dataset/exact_state_manifest.jsonl")
    expected_ids = [str(row["state_id"]) for row in expected_rows]
    all_index: list[dict[str, Any]] = []
    shards = []
    for rank in range(int(contract["static_config"]["world_size"])):
        complete = read_json(output_root / f"features/work/rank{rank:02d}.complete.json")
        if (
            not complete.get("passed")
            or complete.get("contract_sha256") != contract["contract_sha256"]
            or complete.get("checkpoint_sha256") != contract["experiment_a_checkpoint_sha256"]
        ):
            raise RuntimeError(f"extraction rank {rank} is incomplete or incompatible")
        shard_path = output_root / complete["feature_shard"]
        index_path = output_root / complete["index"]
        if file_sha256(shard_path) != complete["feature_shard_sha256"] or file_sha256(index_path) != complete["index_sha256"]:
            raise RuntimeError(f"extraction rank {rank} hash mismatch")
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        index_rows = read_jsonl(index_path)
        if (
            payload.get("contract_sha256") != contract["contract_sha256"]
            or payload.get("checkpoint_sha256") != contract["experiment_a_checkpoint_sha256"]
            or payload.get("feature_schema_sha256") != file_sha256(output_root / "features/feature_schema.md")
            or len(payload["state_ids"]) != len(index_rows)
            or any(payload[name].shape[0] != len(index_rows) for name in ("z_R", "z_W", "action_logits", "nonfull_full_margin"))
        ):
            raise RuntimeError(f"extraction rank {rank} payload is incompatible")
        final_shard = output_root / f"features/shard_{rank:02d}.pt"
        if final_shard.exists():
            raise RuntimeError(f"refusing to overwrite final feature shard {rank}")
        _atomic_torch(final_shard, payload)
        for row in index_rows:
            row["feature_shard"] = str(final_shard.relative_to(output_root))
        all_index.extend(index_rows)
        shards.append(
            {
                "worker_rank": rank,
                "path": str(final_shard.relative_to(output_root)),
                "sha256": file_sha256(final_shard),
                "states": len(index_rows),
            }
        )
    audit = validate_extracted_state_rows(expected_ids, all_index)
    atomic_jsonl(output_root / "features/feature_index.jsonl", all_index)
    manifest = {
        "schema_version": "stage2_treatment_selectivity_representation_manifest_v1",
        "contract_sha256": contract["contract_sha256"],
        "checkpoint_sha256": contract["experiment_a_checkpoint_sha256"],
        "feature_schema_sha256": file_sha256(output_root / "features/feature_schema.md"),
        "status": "complete",
        "passed": True,
        "states": audit["states"],
        "duplicates": audit["duplicates"],
        "missing": audit["missing"],
        "feature_index_sha256": file_sha256(output_root / "features/feature_index.jsonl"),
        "shards": shards,
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "features/representation_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


def _load_feature_map(
    contract: Mapping[str, Any], output_root: Path
) -> dict[str, dict[str, np.ndarray | float]]:
    manifest = read_json(output_root / "features/representation_manifest.json")
    if (
        not manifest.get("passed")
        or manifest.get("status") != "complete"
        or manifest.get("contract_sha256") != contract["contract_sha256"]
        or manifest.get("checkpoint_sha256") != contract["experiment_a_checkpoint_sha256"]
        or manifest.get("feature_schema_sha256") != file_sha256(output_root / "features/feature_schema.md")
        or manifest.get("feature_index_sha256") != file_sha256(output_root / "features/feature_index.jsonl")
    ):
        raise RuntimeError("representation manifest is incomplete or incompatible")
    index_rows = read_jsonl(output_root / "features/feature_index.jsonl")
    by_shard: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in index_rows:
        by_shard[str(row["feature_shard"])].append(row)
    output: dict[str, dict[str, np.ndarray | float]] = {}
    for shard in manifest["shards"]:
        relative = str(shard["path"])
        path = output_root / relative
        if file_sha256(path) != str(shard["sha256"]):
            raise RuntimeError(f"final feature shard hash mismatch: {relative}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows = by_shard[relative]
        if len(rows) != int(shard["states"]):
            raise RuntimeError(f"feature-index count mismatch: {relative}")
        for row in rows:
            tensor_row = int(row["tensor_row"])
            state_id = str(row["state_id"])
            if state_id in output or str(payload["state_ids"][tensor_row]) != state_id:
                raise RuntimeError(f"feature state/index mismatch: {state_id}")
            output[state_id] = {
                "z_R": payload["z_R"][tensor_row].numpy().astype(np.float32, copy=False),
                "z_W": payload["z_W"][tensor_row].numpy().astype(np.float32, copy=False),
                "action_logits": payload["action_logits"][tensor_row]
                .numpy()
                .astype(np.float32, copy=False),
                "nonfull_full_margin": float(payload["nonfull_full_margin"][tensor_row]),
            }
    validate_extracted_state_rows(
        [str(row["state_id"]) for row in read_jsonl(output_root / "dataset/exact_state_manifest.jsonl")],
        [{"state_id": state_id} for state_id in output],
    )
    return output


def _matrix(
    rows: Sequence[Mapping[str, Any]],
    feature_map: Mapping[str, Mapping[str, np.ndarray | float]],
    probe: str,
) -> np.ndarray:
    if probe == "M1_logits":
        return np.stack([feature_map[str(row["state_id"])]["action_logits"] for row in rows])
    if probe == "R_read":
        return np.stack([feature_map[str(row["state_id"])]["z_R"] for row in rows])
    if probe == "W_write":
        return np.stack([feature_map[str(row["state_id"])]["z_W"] for row in rows])
    if probe in {"RW_read_write", "RW_mlp"}:
        return np.stack(
            [
                np.concatenate(
                    (
                        np.asarray(feature_map[str(row["state_id"])]["z_R"]),
                        np.asarray(feature_map[str(row["state_id"])]["z_W"]),
                    )
                )
                for row in rows
            ]
        )
    raise ValueError(f"unsupported learned probe: {probe}")


def _nuisance_matrix(
    clean: Sequence[Mapping[str, Any]], mixed: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    fields = ("dataset", "source_regime", "layer_bin", "route_source_signature")
    categories = {
        field: sorted({str(row[field]) for row in [*clean, *mixed]}) for field in fields
    }
    names = [f"{field}={value}" for field in fields for value in categories[field]]

    def encode(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        matrix = np.zeros((len(rows), len(names)), dtype=np.float32)
        offset = 0
        for field in fields:
            lookup = {value: index for index, value in enumerate(categories[field])}
            for row_index, row in enumerate(rows):
                matrix[row_index, offset + lookup[str(row[field])]] = 1.0
            offset += len(categories[field])
        return matrix

    return encode(clean), encode(mixed), names


PROBE_DIRS = {
    "M0_margin": "margin_baseline",
    "M1_logits": "logits_linear",
    "R_read": "read_linear",
    "W_write": "write_linear",
    "RW_read_write": "read_write_linear",
    "RW_mlp": "read_write_mlp_optional",
    "NUISANCE": "../metrics",
}


def _fold_metric_row(
    probe: str,
    fold: int,
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    values = binary_metrics(labels, scores)
    return {
        "probe": probe,
        "fold": fold,
        "states": len(rows),
        "uids": len({str(row["uid"]) for row in rows}),
        "KEEP_REQUIRED": int((labels == 0).sum()),
        "INTERVENE_REQUIRED": int((labels == 1).sum()),
        **values,
    }


def _fit_oof(
    *,
    contract: Mapping[str, Any],
    output_root: Path,
    probe: str,
    clean: Sequence[Mapping[str, Any]],
    mixed: Sequence[Mapping[str, Any]],
    clean_features: np.ndarray,
    mixed_features: np.ndarray,
    device: torch.device,
    kind: str = "linear",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    config = contract["static_config"]
    settings = config["probe"]
    labels = np.asarray([int(row["binary_target"]) for row in clean], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in clean], dtype=np.int64)
    mixed_folds = np.asarray([int(row["fold"]) for row in mixed], dtype=np.int64)
    predictions = np.full(len(clean), np.nan, dtype=np.float64)
    mixed_predictions = np.full(len(mixed), np.nan, dtype=np.float64)
    fold_metrics: list[dict[str, Any]] = []
    model_root = output_root / "probes" / PROBE_DIRS[probe]
    for fold in range(int(config["folds"])):
        train_indices = np.flatnonzero(folds != fold)
        test_indices = np.flatnonzero(folds == fold)
        mixed_indices = np.flatnonzero(mixed_folds == fold)
        train_rows = [clean[int(index)] for index in train_indices]
        weights = training_weights(train_rows)
        train_x, test_x, normalization = standardize_fold(
            clean_features[train_indices], clean_features[test_indices], weights
        )
        if len(mixed_indices):
            mean = np.asarray(normalization["mean"], dtype=np.float64)
            std = np.asarray(normalization["std"], dtype=np.float64)
            mixed_x = ((mixed_features[mixed_indices] - mean) / std).astype(np.float32)
        else:
            mixed_x = np.empty((0, clean_features.shape[1]), dtype=np.float32)
        fitted = fit_binary_probe(
            train_x,
            labels[train_indices],
            weights,
            kind=kind,
            seed=int(config["seed"]) + 1000 * fold + (17 if kind == "mlp" else 0),
            epochs=int(settings[f"{kind}_epochs"]),
            learning_rate=float(settings[f"{kind}_learning_rate"]),
            weight_decay=float(settings[f"{kind}_weight_decay"]),
            device=device,
            hidden_size=int(settings["mlp_hidden_size"]) if kind == "mlp" else None,
        )
        predictions[test_indices] = predict_binary_probe(fitted, test_x, device=device)
        if len(mixed_indices):
            mixed_predictions[mixed_indices] = predict_binary_probe(fitted, mixed_x, device=device)
        atomic_json(
            model_root / f"fold_{fold}.json",
            {
                "schema_version": "stage2_treatment_selectivity_probe_fold_v1",
                "contract_sha256": contract["contract_sha256"],
                "probe": probe,
                "fold": fold,
                "train_states": len(train_indices),
                "heldout_states": len(test_indices),
                "train_uids": len({str(clean[int(index)]["uid"]) for index in train_indices}),
                "heldout_uids": len({str(clean[int(index)]["uid"]) for index in test_indices}),
                "normalization": normalization,
                "fitted": fitted,
            },
        )
        fold_rows = [clean[int(index)] for index in test_indices]
        fold_metrics.append(
            _fold_metric_row(
                probe, fold, fold_rows, labels[test_indices], predictions[test_indices]
            )
        )
    if not np.isfinite(predictions).all() or (len(mixed) and not np.isfinite(mixed_predictions).all()):
        raise RuntimeError(f"incomplete OOF predictions for {probe}")
    oof_rows = [
        {
            "schema_version": "stage2_treatment_selectivity_oof_prediction_v1",
            "contract_sha256": contract["contract_sha256"],
            "probe": probe,
            "state_id": str(row["state_id"]),
            "uid": str(row["uid"]),
            "fold": int(row["fold"]),
            "state_label": str(row["state_label"]),
            "binary_target": int(row["binary_target"]),
            "intervene_score": float(predictions[index]),
        }
        for index, row in enumerate(clean)
    ]
    oof_rows.extend(
        {
            "schema_version": "stage2_treatment_selectivity_oof_prediction_v1",
            "contract_sha256": contract["contract_sha256"],
            "probe": probe,
            "state_id": str(row["state_id"]),
            "uid": str(row["uid"]),
            "fold": int(row["fold"]),
            "state_label": LABEL_MIXED,
            "binary_target": None,
            "intervene_score": float(mixed_predictions[index]),
        }
        for index, row in enumerate(mixed)
    )
    atomic_jsonl(model_root / "oof_predictions.jsonl", oof_rows)
    return predictions, mixed_predictions, fold_metrics


def _supported_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    indices: Sequence[int],
    *,
    minimum: int,
) -> dict[str, Any]:
    subset = np.asarray(list(indices), dtype=np.int64)
    subset_labels = labels[subset]
    support = {
        "states": len(subset),
        "KEEP_REQUIRED": int((subset_labels == 0).sum()),
        "INTERVENE_REQUIRED": int((subset_labels == 1).sum()),
    }
    if min(support["KEEP_REQUIRED"], support["INTERVENE_REQUIRED"]) < minimum:
        return {**support, "supported": False, "auroc": None, "auprc": None}
    return {**support, "supported": True, **binary_metrics(subset_labels, scores[subset])}


def _breakdown_rows(
    clean: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    scopes: Sequence[tuple[str, str, Sequence[int]]],
    *,
    minimum: int,
) -> list[dict[str, Any]]:
    output = []
    for scope_type, scope, indices in scopes:
        for probe, scores in predictions.items():
            output.append(
                {
                    "scope_type": scope_type,
                    "scope": scope,
                    "probe": probe,
                    **_supported_metrics(labels, scores, indices, minimum=minimum),
                }
            )
    return output


def _roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    return np.r_[0.0, fp / fp[-1]], np.r_[0.0, tp / tp[-1]]


def _pr_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y == 1)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / tp[-1]
    return np.r_[0.0, recall], np.r_[1.0, precision]


def _save_figures(
    output_root: Path,
    clean: Sequence[Mapping[str, Any]],
    mixed: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    mixed_predictions: Mapping[str, np.ndarray],
    matched_rows: Sequence[Mapping[str, Any]],
) -> None:
    display = {
        "M0_margin": "Margin",
        "M1_logits": "4 logits",
        "R_read": "READ",
        "W_write": "WRITE",
        "RW_read_write": "READ+WRITE",
        "RW_mlp": "READ+WRITE MLP",
    }
    plt.figure(figsize=(7, 6))
    for probe, scores in predictions.items():
        if probe == "NUISANCE":
            continue
        fpr, tpr = _roc_curve(labels, scores)
        plt.plot(fpr, tpr, label=f"{display[probe]} ({binary_metrics(labels, scores)['auroc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_root / "figures/probe_roc_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 6))
    for probe, scores in predictions.items():
        if probe == "NUISANCE":
            continue
        recall, precision = _pr_curve(labels, scores)
        plt.plot(recall, precision, label=f"{display[probe]} ({binary_metrics(labels, scores)['auprc']:.3f})")
    plt.axhline(labels.mean(), color="k", linestyle="--", linewidth=1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_root / "figures/probe_pr_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    coverages = np.linspace(0.01, 0.30, 30)
    for probe in ("M0_margin", "M1_logits", "RW_read_write"):
        scores = predictions[probe]
        precisions = []
        for coverage in coverages:
            row = selective_operating_points(
                labels, scores, coverage_fractions=[float(coverage)], precision_targets=[]
            )[0]
            precisions.append(row["precision"])
        plt.plot(coverages, precisions, label=display[probe])
    plt.xlabel("Selected intervention coverage")
    plt.ylabel("INTERVENE precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "figures/high_precision_tradeoff.png", dpi=180)
    plt.close()

    scopes = sorted({(str(row["source_regime"]), str(row["dataset"])) for row in clean})
    values = []
    names = []
    for source, dataset in scopes:
        indices = [index for index, row in enumerate(clean) if row["source_regime"] == source and row["dataset"] == dataset]
        metric = _supported_metrics(labels, predictions["RW_read_write"], indices, minimum=10)
        names.append(f"{source[:4]}-{dataset}")
        values.append(np.nan if metric["auroc"] is None else metric["auroc"])
    plt.figure(figsize=(9, 5))
    plt.bar(names, values)
    plt.axhline(0.5, color="k", linestyle="--", linewidth=1)
    plt.ylim(0, 1)
    plt.ylabel("READ+WRITE AUROC")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_root / "figures/dataset_source_auroc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for probe in ("M0_margin", "RW_read_write"):
        layers, values = [], []
        for layer in range(28):
            indices = [index for index, row in enumerate(clean) if int(row["layer"]) == layer]
            metric = _supported_metrics(labels, predictions[probe], indices, minimum=10)
            if metric["supported"]:
                layers.append(layer)
                values.append(metric["auroc"])
        plt.plot(layers, values, marker="o", markersize=3, label=display[probe])
    plt.axhline(0.5, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "figures/layerwise_separability.png", dpi=180)
    plt.close()

    probes = ["R_read", "W_write", "RW_read_write"]
    plt.figure(figsize=(6, 5))
    plt.bar([display[p] for p in probes], [binary_metrics(labels, predictions[p])["auroc"] for p in probes])
    plt.axhline(0.5, color="k", linestyle="--", linewidth=1)
    plt.ylim(0, 1)
    plt.ylabel("OOF AUROC")
    plt.tight_layout()
    plt.savefig(output_root / "figures/read_vs_write_separability.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for probe in ("M0_margin", "M1_logits", "RW_read_write"):
        plt.hist(mixed_predictions[probe], bins=30, alpha=0.45, density=True, label=display[probe])
    plt.xlabel("OOF intervention score on MIXED states")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "figures/mixed_state_score_distribution.png", dpi=180)
    plt.close()

    selected = [row for row in matched_rows if row["probe"] in {"M0_margin", "RW_read_write"}]
    plt.figure(figsize=(7, 5))
    positions = np.arange(len(selected))
    width = 0.35
    plt.bar(positions - width / 2, [row["natural_auroc"] for row in selected], width, label="Natural")
    plt.bar(positions + width / 2, [row["matched_auroc"] for row in selected], width, label="Matched")
    plt.xticks(positions, [display[row["probe"]] for row in selected])
    plt.axhline(0.5, color="k", linestyle="--", linewidth=1)
    plt.ylim(0, 1)
    plt.ylabel("AUROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "figures/matched_vs_unmatched.png", dpi=180)
    plt.close()


def analyze(config_path: Path, device_index: int) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    device = torch.device(f"cuda:{int(device_index)}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config = contract["static_config"]
    clean = read_jsonl(output_root / "dataset/clean_binary_states.jsonl")
    mixed = read_jsonl(output_root / "dataset/mixed_states.jsonl")
    folds = {str(row["state_id"]): int(row["fold"]) for row in read_jsonl(output_root / "dataset/fold_manifest.jsonl")}
    for row in [*clean, *mixed]:
        row["fold"] = folds[str(row["state_id"])]
    labels = np.asarray([int(row["binary_target"]) for row in clean], dtype=np.int64)
    feature_map = _load_feature_map(contract, output_root)
    predictions: dict[str, np.ndarray] = {
        "M0_margin": np.asarray(
            [float(feature_map[str(row["state_id"])]["nonfull_full_margin"]) for row in clean],
            dtype=np.float64,
        )
    }
    mixed_predictions: dict[str, np.ndarray] = {
        "M0_margin": np.asarray(
            [float(feature_map[str(row["state_id"])]["nonfull_full_margin"]) for row in mixed],
            dtype=np.float64,
        )
    }
    fold_metrics = []
    for fold in range(int(config["folds"])):
        indices = [index for index, row in enumerate(clean) if int(row["fold"]) == fold]
        fold_metrics.append(
            _fold_metric_row(
                "M0_margin",
                fold,
                [clean[index] for index in indices],
                labels[indices],
                predictions["M0_margin"][indices],
            )
        )
    atomic_jsonl(
        output_root / "probes/margin_baseline/oof_predictions.jsonl",
        [
            {
                "schema_version": "stage2_treatment_selectivity_oof_prediction_v1",
                "contract_sha256": contract["contract_sha256"],
                "probe": "M0_margin",
                "state_id": str(row["state_id"]),
                "uid": str(row["uid"]),
                "fold": int(row["fold"]),
                "state_label": str(row["state_label"]),
                "binary_target": row["binary_target"],
                "intervene_score": float(
                    predictions["M0_margin"][index]
                    if row["state_label"] != LABEL_MIXED
                    else mixed_predictions["M0_margin"][index - len(clean)]
                ),
            }
            for index, row in enumerate([*clean, *mixed])
        ],
    )
    for probe in ("M1_logits", "R_read", "W_write", "RW_read_write"):
        prediction, mixed_prediction, rows = _fit_oof(
            contract=contract,
            output_root=output_root,
            probe=probe,
            clean=clean,
            mixed=mixed,
            clean_features=_matrix(clean, feature_map, probe),
            mixed_features=_matrix(mixed, feature_map, probe),
            device=device,
        )
        predictions[probe] = prediction
        mixed_predictions[probe] = mixed_prediction
        fold_metrics.extend(rows)
        print(json.dumps({"probe": probe, **binary_metrics(labels, prediction)}, sort_keys=True), flush=True)

    rw_metrics = binary_metrics(labels, predictions["RW_read_write"])
    thresholds = config["probe"]
    run_mlp = float(thresholds["weak_nonrandom_auroc_min"]) <= rw_metrics["auroc"] < float(
        thresholds["strong_auroc_min"]
    )
    if run_mlp:
        prediction, mixed_prediction, rows = _fit_oof(
            contract=contract,
            output_root=output_root,
            probe="RW_mlp",
            clean=clean,
            mixed=mixed,
            clean_features=_matrix(clean, feature_map, "RW_mlp"),
            mixed_features=_matrix(mixed, feature_map, "RW_mlp"),
            device=device,
            kind="mlp",
        )
        predictions["RW_mlp"] = prediction
        mixed_predictions["RW_mlp"] = mixed_prediction
        fold_metrics.extend(rows)
        print(json.dumps({"probe": "RW_mlp", **binary_metrics(labels, prediction)}, sort_keys=True), flush=True)
    else:
        atomic_json(
            output_root / "probes/read_write_mlp_optional/not_run.json",
            {
                "contract_sha256": contract["contract_sha256"],
                "reason": "RW-linear did not fall in prospectively frozen weak-but-nonrandom AUROC interval",
                "rw_linear_auroc": rw_metrics["auroc"],
                "interval": [
                    float(thresholds["weak_nonrandom_auroc_min"]),
                    float(thresholds["strong_auroc_min"]),
                ],
            },
        )

    nuisance_clean, nuisance_mixed, nuisance_names = _nuisance_matrix(clean, mixed)
    nuisance_prediction, nuisance_mixed_prediction, nuisance_fold_rows = _fit_oof(
        contract=contract,
        output_root=output_root,
        probe="NUISANCE",
        clean=clean,
        mixed=mixed,
        clean_features=nuisance_clean,
        mixed_features=nuisance_mixed,
        device=device,
    )
    predictions["NUISANCE"] = nuisance_prediction
    mixed_predictions["NUISANCE"] = nuisance_mixed_prediction
    fold_metrics.extend(nuisance_fold_rows)
    nuisance_metrics = binary_metrics(labels, nuisance_prediction)
    atomic_csv(
        output_root / "metrics/nuisance_controls.csv",
        [
            {
                "probe": "dataset+source+layer_bin+route_source",
                "input_columns": ";".join(nuisance_names),
                "states": len(clean),
                **nuisance_metrics,
            }
        ],
    )

    probe_summary = []
    operating_rows = []
    for probe, scores in predictions.items():
        if probe == "NUISANCE":
            continue
        metric = binary_metrics(labels, scores)
        ops = selective_operating_points(
            labels,
            scores,
            coverage_fractions=config["support"]["coverage_fractions"],
            precision_targets=config["support"]["precision_targets"],
        )
        operating_rows.extend({"probe": probe, **row} for row in ops)
        by_name = {str(row["operating_point"]): row for row in ops}
        probe_summary.append(
            {
                "probe": probe,
                "states": len(clean),
                "uids": len({str(row["uid"]) for row in clean}),
                **metric,
                "precision_at_5pct": by_name["top_5pct"]["precision"],
                "precision_at_10pct": by_name["top_10pct"]["precision"],
                "precision_at_20pct": by_name["top_20pct"]["precision"],
                "recall_at_90pct_precision": by_name["recall_at_90pct_precision"]["recall"],
                "recall_at_95pct_precision": by_name["recall_at_95pct_precision"]["recall"],
            }
        )
    atomic_csv(output_root / "metrics/probe_summary.csv", probe_summary)
    atomic_csv(output_root / "metrics/fold_metrics.csv", fold_metrics)
    atomic_csv(output_root / "metrics/precision_recall_operating_points.csv", operating_rows)

    minimum = int(config["support"]["minimum_class_rows_for_breakdown"])
    dataset_scopes = []
    for source in sorted({str(row["source_regime"]) for row in clean}):
        for dataset in sorted({str(row["dataset"]) for row in clean}):
            indices = [
                index
                for index, row in enumerate(clean)
                if row["source_regime"] == source and row["dataset"] == dataset
            ]
            dataset_scopes.append(("dataset_source", f"{source}:{dataset}", indices))
    atomic_csv(
        output_root / "metrics/dataset_source_breakdown.csv",
        _breakdown_rows(clean, labels, predictions, dataset_scopes, minimum=minimum),
    )
    layer_scopes = []
    for layer_bin in ("early_0_8", "middle_9_18", "late_19_27"):
        layer_scopes.append(
            (
                "layer_bin",
                layer_bin,
                [index for index, row in enumerate(clean) if row["layer_bin"] == layer_bin],
            )
        )
    for layer in range(28):
        layer_scopes.append(
            ("layer", str(layer), [index for index, row in enumerate(clean) if int(row["layer"]) == layer])
        )
    atomic_csv(
        output_root / "metrics/layer_breakdown.csv",
        _breakdown_rows(clean, labels, predictions, layer_scopes, minimum=minimum),
    )
    route_scopes = [
        (
            "route_source_membership",
            route_source,
            [index for index, row in enumerate(clean) if route_source in row["route_sources"]],
        )
        for route_source in ROUTE_SOURCES
    ]
    atomic_csv(
        output_root / "metrics/route_source_breakdown.csv",
        _breakdown_rows(clean, labels, predictions, route_scopes, minimum=minimum),
    )

    matched_weights, matched_support = matched_evaluation_weights(clean)
    matched_indices = np.flatnonzero(matched_weights > 0)
    matched_rows = []
    for probe in ("M0_margin", "RW_read_write"):
        natural = binary_metrics(labels, predictions[probe])
        matched = binary_metrics(labels, predictions[probe], matched_weights)
        weights = matched_weights[matched_indices]
        matched_rows.append(
            {
                "probe": probe,
                "natural_auroc": natural["auroc"],
                "natural_auprc": natural["auprc"],
                "matched_auroc": matched["auroc"],
                "matched_auprc": matched["auprc"],
                "auroc_drop": natural["auroc"] - matched["auroc"],
                "supported_cells": matched_support["supported_cells"],
                "raw_supported_states": matched_support["supported_states"],
                "excluded_states": matched_support["excluded_states"],
                "effective_state_count_kish": float(weights.sum() ** 2 / np.square(weights).sum()),
                "effective_uid_count": len({str(clean[int(index)]["uid"]) for index in matched_indices}),
            }
        )
    atomic_csv(output_root / "metrics/matched_sensitivity.csv", matched_rows)

    mixed_rows = []
    for probe, scores in mixed_predictions.items():
        if probe == "NUISANCE":
            continue
        for scope, indices in [
            ("overall", list(range(len(mixed)))),
            *[
                (
                    route_source,
                    [index for index, row in enumerate(mixed) if route_source in row["route_sources"]],
                )
                for route_source in ROUTE_SOURCES
            ],
        ]:
            values = scores[np.asarray(indices, dtype=np.int64)]
            mixed_rows.append(
                {
                    "probe": probe,
                    "scope": scope,
                    "states": len(values),
                    "mean": float(np.mean(values)) if len(values) else None,
                    "median": float(np.median(values)) if len(values) else None,
                    "q10": float(np.quantile(values, 0.10)) if len(values) else None,
                    "q25": float(np.quantile(values, 0.25)) if len(values) else None,
                    "q75": float(np.quantile(values, 0.75)) if len(values) else None,
                    "q90": float(np.quantile(values, 0.90)) if len(values) else None,
                }
            )
    atomic_csv(output_root / "metrics/mixed_state_analysis.csv", mixed_rows)

    _save_figures(
        output_root,
        clean,
        mixed,
        labels,
        predictions,
        mixed_predictions,
        matched_rows,
    )
    atomic_json(
        output_root / "work/analysis_complete.json",
        {
            "schema_version": "stage2_treatment_selectivity_analysis_complete_v1",
            "contract_sha256": contract["contract_sha256"],
            "passed": True,
            "probes": list(predictions),
            "optional_mlp_run": run_mlp,
            "completed_at": utc_now(),
        },
    )
    print(json.dumps({"analysis_complete": True, "optional_mlp_run": run_mlp, "probe_summary": probe_summary}, sort_keys=True))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float:
    return float(value)


def finalize(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    analysis_complete = read_json(output_root / "work/analysis_complete.json")
    if not analysis_complete.get("passed") or analysis_complete.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("probe analysis is incomplete")
    summary_rows = {row["probe"]: row for row in _csv_rows(output_root / "metrics/probe_summary.csv")}
    matched_rows = {row["probe"]: row for row in _csv_rows(output_root / "metrics/matched_sensitivity.csv")}
    nuisance = _csv_rows(output_root / "metrics/nuisance_controls.csv")[0]
    route_rows = _csv_rows(output_root / "metrics/route_source_breakdown.csv")
    dataset_rows = _csv_rows(output_root / "metrics/dataset_source_breakdown.csv")
    mixed_rows = _csv_rows(output_root / "metrics/mixed_state_analysis.csv")
    population = contract["population"]
    settings = contract["static_config"]["probe"]
    m0 = summary_rows["M0_margin"]
    m1 = summary_rows["M1_logits"]
    read = summary_rows["R_read"]
    write = summary_rows["W_write"]
    rw = summary_rows["RW_read_write"]
    matched_rw = matched_rows["RW_read_write"]
    rw_auroc = _float(rw["auroc"])
    m0_auroc = _float(m0["auroc"])
    m1_auroc = _float(m1["auroc"])
    read_auroc = _float(read["auroc"])
    write_auroc = _float(write["auroc"])
    matched_auroc = _float(matched_rw["matched_auroc"])
    nuisance_auroc = _float(nuisance["auroc"])
    gain = float(settings["clear_gain_min"])
    strong = float(settings["strong_auroc_min"])
    matched_collapsed = (
        _float(matched_rw["auroc_drop"]) >= float(settings["matched_collapse_drop_min"])
        or matched_auroc <= float(settings["matched_weak_auroc_max"])
    )
    natural_high = rw_auroc >= strong
    if natural_high and nuisance_auroc >= float(settings["nuisance_high_auroc_min"]) and matched_collapsed:
        decision_case = "E"
        interpretation = "Natural RW separability is substantially explained by dataset/source/layer/route priors."
        recommendation = "Do not add a treatment gate; next run one representation/training-state diversity diagnostic on nuisance-matched support."
    elif (
        natural_high
        and rw_auroc - m0_auroc >= gain
        and _float(rw["precision_at_5pct"]) >= float(settings["high_precision_at_5pct_min"])
        and matched_auroc > float(settings["matched_weak_auroc_max"])
    ):
        decision_case = "A"
        interpretation = "Frozen READ/WRITE representations contain treatment-selectivity signal that the scalar margin does not use well."
        recommendation = "Train a minimal KEEP-vs-INTERVENE head on frozen z_R/z_W, invoking the existing action selector only after INTERVENE."
    elif m1_auroc >= strong and m1_auroc - m0_auroc >= gain and rw_auroc - m1_auroc < gain:
        decision_case = "B"
        interpretation = "Selectivity is already present in the four-logit pattern; the scalar margin is the deficient readout."
        recommendation = "Train a minimal learned abstention readout over the frozen four-way logits."
    elif max(read_auroc, write_auroc) >= strong and abs(read_auroc - write_auroc) >= gain:
        decision_case = "C"
        branch = "READ" if read_auroc > write_auroc else "WRITE"
        interpretation = f"Treatment selectivity is concentrated in the {branch} branch under this diagnostic."
        recommendation = f"Run one branch-focused, nuisance-matched representation diagnostic centered on {branch}, without removing the other branch."
    else:
        decision_case = "D"
        interpretation = "The frozen Stage-2 representations do not meet the prospective generalizable/high-precision evidence gates."
        recommendation = "Do not add another head; diagnose representation and training-state diversity next."

    route_rw = [row for row in route_rows if row["probe"] == "RW_read_write"]
    supported_route = [row for row in route_rw if row["supported"] == "True"]
    route_description = ", ".join(
        f"{row['scope']} AUROC {float(row['auroc']):.3f} (I={row['INTERVENE_REQUIRED']}, K={row['KEEP_REQUIRED']})"
        for row in supported_route
    )
    dataset_rw = [row for row in dataset_rows if row["probe"] == "RW_read_write"]
    supported_dataset = [row for row in dataset_rw if row["supported"] == "True"]
    dataset_description = ", ".join(
        f"{row['scope']} {float(row['auroc']):.3f}" for row in supported_dataset
    )
    mixed_rw = next(
        row for row in mixed_rows if row["probe"] == "RW_read_write" and row["scope"] == "overall"
    )
    optional = "RW_mlp" in summary_rows
    summary = f"""# Stage-2 treatment-selectivity separability summary

## Decision

- Case **{decision_case}**.
- {interpretation}
- Contract: `{contract['contract_sha256']}`.
- All numbers below are out-of-fold under five UID/image-group-disjoint folds.

## Answers to the plan questions

1. **Unique UIDs:** {population['unique_uids']}.
2. **Exact routed states:** {population['unique_exact_states']:,} unique exact prefix-states from {population['route_state_occurrences']:,} route-state occurrences.
3. **KEEP_REQUIRED:** {population['KEEP_REQUIRED']:,}.
4. **INTERVENE_REQUIRED:** {population['INTERVENE_REQUIRED']:,}.
5. **MIXED:** {population['MIXED']:,}; excluded from fitting and scored OOF only.
6. **Scalar margin:** AUROC {m0_auroc:.4f}, AUPRC {_float(m0['auprc']):.4f}; this is {'above' if m0_auroc >= 0.55 else 'not meaningfully above'} the frozen weak-signal reference.
7. **Four logits versus margin:** M1 AUROC {m1_auroc:.4f} ({m1_auroc - m0_auroc:+.4f} versus M0).
8. **z_R:** AUROC {read_auroc:.4f}, AUPRC {_float(read['auprc']):.4f}.
9. **z_W:** AUROC {write_auroc:.4f}, AUPRC {_float(write['auprc']):.4f}.
10. **[z_R;z_W]:** AUROC {rw_auroc:.4f}, AUPRC {_float(rw['auprc']):.4f}; optional MLP {'was run' if optional else 'was not triggered'}.
11. **High-precision subset:** RW precision at top 5/10/20% coverage is {_float(rw['precision_at_5pct']):.3f}/{_float(rw['precision_at_10pct']):.3f}/{_float(rw['precision_at_20pct']):.3f}; recall at 90/95% precision is {_float(rw['recall_at_90pct_precision']):.3f}/{_float(rw['recall_at_95pct_precision']):.3f}.
12. **Group-disjoint survival:** yes in evaluation design; RW OOF AUROC is {rw_auroc:.4f}, so signal is {'useful by the frozen criterion' if rw_auroc >= 0.55 else 'weak'}.
13. **Matched survival:** RW matched AUROC/AUPRC is {matched_auroc:.4f}/{_float(matched_rw['matched_auprc']):.4f}, versus natural {rw_auroc:.4f}/{_float(rw['auprc']):.4f}; AUROC change {_float(matched_rw['auroc_drop']):+.4f} drop. Support is {matched_rw['raw_supported_states']} states and {matched_rw['effective_uid_count']} UIDs (Kish state ESS {_float(matched_rw['effective_state_count_kish']):.1f}).
14. **Route consistency:** {route_description or 'insufficient two-class route-source support'}.
15. **Learned KEEP-vs-INTERVENE head justified:** {'yes, as a separately authorized next experiment' if decision_case == 'A' else 'no under the prospective decision rule'}.
16. **Not justified:** This does not show that routing cannot work, that READ/WRITE control is invalid, that unobserved actions are truly harmful, or that richer/diversified representations cannot help. KEEP_REQUIRED means uniquely observed FULL among replay-valid successful routes, not exhaustive proof against every unsearched intervention.

## Additional controls

- Nuisance-only AUROC: {nuisance_auroc:.4f} using dataset, source regime, layer bin, and route-source signature only.
- RW dataset/source AUROCs where supported: {dataset_description or 'none'}.
- MIXED-state RW score median {float(mixed_rw['median']):.4f} (q25–q75 {float(mixed_rw['q25']):.4f}–{float(mixed_rw['q75']):.4f}); this is descriptive, not a binary target.
"""
    _atomic_text(output_root / "summaries/treatment_selectivity_summary.md", summary)
    recommendation_text = f"""# Next Stage-2 recommendation

## Recommended next action

**{recommendation}**

- Decision case: {decision_case}.
- Why it matters: {interpretation}
- Hypothesis tested next: whether the identified frozen signal can generalize to selective correction without repeating the external regression pattern.
- Positive result: intervention precision remains high while W→C exceeds C→W on untouched validation evidence.
- Negative result: the diagnostic separability does not translate into safe sequential deployment, or the current representation/coverage remains insufficient.
- Do not overinterpret: the present probes are diagnostic and trained on observed replay-valid actions; they are not a deployed gate and do not establish causal action benefit.

This recommendation is not executed in Phase 72 and requires separate authorization.
"""
    _atomic_text(output_root / "summaries/next_stage2_recommendation.md", recommendation_text)

    required = [
        "protocol.md",
        "dataset/exact_state_manifest.jsonl",
        "dataset/clean_binary_states.jsonl",
        "dataset/mixed_states.jsonl",
        "dataset/dataset_summary.csv",
        "dataset/uid_state_counts.csv",
        "dataset/fold_manifest.jsonl",
        "features/representation_manifest.json",
        "features/feature_schema.md",
        "features/feature_index.jsonl",
        "metrics/probe_summary.csv",
        "metrics/fold_metrics.csv",
        "metrics/precision_recall_operating_points.csv",
        "metrics/dataset_source_breakdown.csv",
        "metrics/layer_breakdown.csv",
        "metrics/route_source_breakdown.csv",
        "metrics/matched_sensitivity.csv",
        "metrics/nuisance_controls.csv",
        "metrics/mixed_state_analysis.csv",
        "figures/probe_roc_comparison.png",
        "figures/probe_pr_comparison.png",
        "figures/high_precision_tradeoff.png",
        "figures/dataset_source_auroc.png",
        "figures/layerwise_separability.png",
        "figures/read_vs_write_separability.png",
        "figures/mixed_state_score_distribution.png",
        "figures/matched_vs_unmatched.png",
        "summaries/treatment_selectivity_summary.md",
        "summaries/next_stage2_recommendation.md",
    ]
    missing = [relative for relative in required if not (output_root / relative).is_file()]
    if missing:
        raise RuntimeError(f"required output files are missing: {missing}")
    files = {
        str(path.relative_to(output_root)): file_sha256(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    artifact_manifest = {
        "schema_version": "stage2_treatment_selectivity_artifact_manifest_v1",
        "contract_sha256": contract["contract_sha256"],
        "passed": True,
        "decision_case": decision_case,
        "files": files,
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "artifact_manifest.json", artifact_manifest)
    verify_artifact_manifest(output_root, artifact_manifest)
    print(
        json.dumps(
            {
                "finalized": True,
                "contract_sha256": contract["contract_sha256"],
                "decision_case": decision_case,
                "recommendation": recommendation,
                "rw_auroc": rw_auroc,
                "rw_matched_auroc": matched_auroc,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--device", type=int, default=0)
    worker_parser = subparsers.add_parser("extract-worker")
    worker_parser.add_argument("--rank", type=int, required=True)
    subparsers.add_parser("aggregate")
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--device", type=int, default=0)
    subparsers.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config)
    elif args.command == "smoke":
        smoke(args.config, args.device)
    elif args.command == "extract-worker":
        extract_worker(args.config, args.rank)
    elif args.command == "aggregate":
        aggregate(args.config)
    elif args.command == "analyze":
        analyze(args.config, args.device)
    elif args.command == "finalize":
        finalize(args.config)


if __name__ == "__main__":
    main()
