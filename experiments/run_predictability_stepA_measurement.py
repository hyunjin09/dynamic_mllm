#!/usr/bin/env python3
"""Construct the frozen Stage-1/Stage-2 predictability Step-A measurements."""

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
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from binary_policy.executor import (  # noqa: E402
    capture_four_action_route,
    capture_four_action_suffix_from_full_baseline,
    capture_four_action_suffix_from_route_baseline,
    greedy_generate_from_cached_prompt,
)
from binary_policy.executor.four_action import score_token_ids_from_cached_prompt  # noqa: E402
from binary_policy.executor.inputs import build_binary_inputs  # noqa: E402
from dense_failure_stage1.lmms_scoring import score_lmms_sample  # noqa: E402
from dense_failure_stage1.runtime import (  # noqa: E402
    build_dense_inputs,
    configure_dense_determinism,
    generate_dense,
    load_dense_runtime,
)
from dense_failure_stage2.closed_loop_trajectory_set import (  # noqa: E402
    build_trajectory_sets,
    compact_router_state,
    prefix_action_hash,
)
from dense_failure_stage2.predictability_measurement import (  # noqa: E402
    ACTIONS,
    accepted_answer_specs,
    aggregate_reference_mean_logprobs,
    build_p90_trigger_rows,
    derive_utility_row,
    validate_complete_state_results,
)
from experiments.run_stage2_v1_training_revised import _load_model  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs/predictability_stepA_measurement_v1.json"
ALLOWED_ROOTS = (PROJECT_ROOT.resolve(), Path("/mnt/hyemin").resolve())
BOUND_CODE = (
    "configs/predictability_stepA_measurement_v1.json",
    "dense_failure_stage2/predictability_measurement.py",
    "experiments/run_predictability_stepA_measurement.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/model.py",
    "dense_failure_stage1/lmms_scoring.py",
    "dense_failure_stage1/runtime.py",
    "scoring/benchmark_metrics.py",
    "scoring/reference_likelihood.py",
    "plans/predictability_phase_stepA_measurement_construction_plan.md",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if not any(resolved == root or resolved.is_relative_to(root) for root in ALLOWED_ROOTS):
        raise ValueError(f"path escapes allowed roots: {value}")
    return resolved


def file_sha256(value: str | Path) -> str:
    digest = sha256()
    with resolve_path(value).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = f"{value.dtype}:{tuple(value.shape)}:".encode()
    return sha256(header + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def read_json(value: str | Path) -> dict[str, Any]:
    result = json.loads(resolve_path(value).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {value}")
    return result


def read_jsonl(value: str | Path) -> list[dict[str, Any]]:
    output = []
    with resolve_path(value).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{value}:{line_number} is not a JSON object")
            output.append(row)
    return output


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


def atomic_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None
) -> None:
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"cannot infer fields for empty CSV: {path}")
        fieldnames = list(rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode())


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def command_output(args: Sequence[str]) -> str:
    result = subprocess.run(args, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
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
    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    count = torch.cuda.device_count()
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "lmms_eval": version("lmms-eval"),
        "numpy": version("numpy"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": count,
        "cuda_device_names": [torch.cuda.get_device_name(index) for index in range(count)],
        "nvidia_driver": command_output(
            ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
        ).splitlines()[0],
    }


def load_config(value: str | Path) -> dict[str, Any]:
    config = read_json(value)
    if config.get("schema_version") != "predictability_stepA_measurement_config_v1":
        raise ValueError("unsupported predictability Step-A config")
    if int(config["world_size"]) != 4:
        raise ValueError("Step A requires four direct GPU workers")
    if tuple(config["measurement"]["actions"]) != ACTIONS:
        raise ValueError("measurement action order differs from the executor")
    if config["trigger"]["operating_point"] != "P90":
        raise ValueError("only the frozen P90 domain is supported")
    if config["trigger"]["comparison"] != "strict_greater_than":
        raise ValueError("trigger comparison must remain strict greater-than")
    if config["measurement"]["later_layer_action"] != "FULL":
        raise ValueError("future suffix must remain all-FULL")
    if config["measurement"]["continuous_score"] != (
        "weighted_logsumexp_of_per_reference_token_mean_logprob"
    ):
        raise ValueError("continuous score contract differs")
    if bool(config["measurement"]["include_eos_in_reference_score"]):
        raise ValueError("the frozen reference likelihood excludes EOS")
    return config


def _source_paths(config: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in config["sources"].items()}


def _verify_model_snapshot(config: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    snapshot = resolve_path(config["model"]["snapshot_path"])
    for relative, digest in contract["model_snapshot_sha256"].items():
        path = snapshot / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise RuntimeError(f"model snapshot file differs: {path}")


def _uid_slug(uid: str) -> str:
    return sha256(str(uid).encode()).hexdigest()[:24]


def _state_tensor_hashes(state: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {
        name: tensor_sha256(state[name])
        for name in ("text_states", "visual_states", "text_mask", "visual_mask")
    }


def _combined_state_hash(hashes: Mapping[str, str]) -> str:
    return sha256(
        json.dumps(dict(hashes), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _index_unique(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        uid = str(row["uid"])
        if uid in output:
            raise RuntimeError(f"duplicate UID in {name}: {uid}")
        output[uid] = row
    return output


def _internal_population(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources = config["sources"]
    split_rows = read_jsonl(sources["historical_split"])
    historical_uids = {
        str(row["uid"])
        for row in split_rows
        if str(row["split"]) == config["stage1"]["historical_split_name"]
    }
    hc = _index_unique(read_jsonl(sources["historical_candidates"]), "historical candidates")
    hd = _index_unique(read_jsonl(sources["historical_dense"]), "historical dense")
    hf = _index_unique(read_jsonl(sources["historical_feature_index"]), "historical features")
    cc = _index_unique(read_jsonl(sources["canonical_candidates"]), "canonical candidates")
    cd = _index_unique(read_jsonl(sources["canonical_dense"]), "canonical dense")
    cf = _index_unique(read_jsonl(sources["canonical_feature_index"]), "canonical features")
    if len(historical_uids) != int(config["stage1"]["expected_historical"]):
        raise RuntimeError("Historical internal population differs")
    if len(cc) != int(config["stage1"]["expected_canonical"]):
        raise RuntimeError("Canonical internal population differs")
    if historical_uids.intersection(cc):
        raise RuntimeError("Historical and Canonical UIDs overlap")

    trigger_source = read_jsonl(sources["historical_trigger_map"]) + read_jsonl(
        sources["canonical_trigger_map"]
    )
    trigger_rows = build_p90_trigger_rows(
        trigger_source,
        threshold=float(config["trigger"]["threshold"]),
        layers=int(config["model"]["decoder_layers"]),
    )
    trigger_by_uid = _index_unique(trigger_rows, "P90 trigger rows")
    expected_uids = historical_uids | set(cc)
    if set(trigger_by_uid) != expected_uids:
        raise RuntimeError("P90 trigger UID population differs from the internal population")

    output: list[dict[str, Any]] = []
    for source_regime, uids, candidates, dense, features in (
        ("historical", historical_uids, hc, hd, hf),
        ("canonical", set(cc), cc, cd, cf),
    ):
        missing = sorted(uids - set(candidates) | uids - set(dense) | uids - set(features))
        if missing:
            raise RuntimeError(f"incomplete {source_regime} internal UIDs: {missing[:3]}")
        for uid in sorted(uids):
            sample = candidates[uid]
            outcome = dense[uid]
            trigger = trigger_by_uid[uid]
            if (
                str(sample["dataset"]) != str(outcome["dataset"])
                or str(sample["dataset"]) != str(trigger["dataset"])
                or bool(outcome["current_dense_correct"]) != bool(trigger["dense_correct"])
                or str(sample["image_group_id"]) != str(outcome["image_group_id"])
            ):
                raise RuntimeError(f"internal identity/outcome mismatch: {uid}")
            feature = features[uid]
            feature_path = resolve_path(feature["shard"])
            if not feature_path.is_file():
                raise RuntimeError(f"missing Stage-1 feature shard: {feature_path}")
            output.append(
                {
                    "schema_version": "predictability_stepA_internal_sample_v1",
                    "uid": uid,
                    "dataset": str(sample["dataset"]),
                    "source_regime": source_regime,
                    "image_group_id": str(sample["image_group_id"]),
                    "image_identifier": str(sample.get("image_identifier", "")),
                    "question": str(sample["question"]),
                    "sample": sample,
                    "dense": outcome,
                    "dense_correct": bool(outcome["current_dense_correct"]),
                    "dense_wrong": bool(outcome["current_dense_wrong"]),
                    "stage1_feature_shard": str(feature_path),
                    "stage1_feature_row_index": int(feature["row_index"]),
                    "stage1_feature_layers": int(feature["layers"]),
                    "p90": trigger,
                }
            )
    output.sort(key=lambda row: str(row["uid"]))
    if len(output) != int(config["stage1"]["expected_internal"]):
        raise RuntimeError("combined internal population differs")
    return output


def _stage1_state_id(uid: str, layer: int) -> str:
    return sha256(f"stage1_dense:{uid}:{int(layer)}".encode()).hexdigest()


def _dense_state_rows(
    internal: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    layers = int(config["model"]["decoder_layers"])
    for sample in internal:
        trigger = sample["p90"]["first_trigger_layer"]
        if trigger is None:
            continue
        trigger = int(trigger)
        for layer in range(trigger, layers):
            prefix = ["FULL"] * (layer - trigger)
            rows.append(
                {
                    "schema_version": "predictability_stepA_dense_state_v1",
                    "state_id": prefix_action_hash(str(sample["uid"]), trigger, layer, prefix),
                    "uid": str(sample["uid"]),
                    "dataset": str(sample["dataset"]),
                    "source_regime": str(sample["source_regime"]),
                    "image_group_id": str(sample["image_group_id"]),
                    "dense_correct": bool(sample["dense_correct"]),
                    "dense_wrong": bool(sample["dense_wrong"]),
                    "trigger_layer": trigger,
                    "layer": layer,
                    "trigger_relative_depth": layer - trigger,
                    "prefix_actions": prefix,
                    "state_file": f"stage2_dense/states/{_uid_slug(str(sample['uid']))}.pt",
                }
            )
    return rows


def _routed_state_rows(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    programs = read_jsonl(config["sources"]["routed_program_manifest"])
    grouped, census = build_trajectory_sets(
        programs, total_layers=int(config["model"]["decoder_layers"])
    )
    expected = {
        "uids": int(config["routed"]["expected_uids"]),
        "programs": int(config["routed"]["expected_programs"]),
        "dense_c_uids": 106,
        "dense_w_uids": 463,
        "route_state_occurrences": int(config["routed"]["expected_state_references"]),
        "unique_prefix_states": int(config["routed"]["expected_unique_states"]),
    }
    if census != expected:
        raise RuntimeError(f"Phase-76 routed census differs: {census}")
    program_by_id = {str(row["program_id"]): row for row in programs}
    if len(program_by_id) != len(programs):
        raise RuntimeError("duplicate routed program ID")
    replay = read_jsonl(config["sources"]["routed_program_replay"])
    anchor_candidates: dict[str, list[str]] = defaultdict(list)
    state_to_uid: dict[str, str] = {}
    references = 0
    for row in replay:
        program_id = str(row["program_id"])
        if (
            program_id not in program_by_id
            or not bool(row.get("passed"))
            or not bool(row.get("exact_token_parity"))
            or not bool(row.get("correct"))
        ):
            raise RuntimeError(f"invalid routed anchor replay: {program_id}")
        for state_id in row["state_ids"]:
            state_id = str(state_id)
            anchor_candidates[state_id].append(program_id)
            state_to_uid.setdefault(state_id, str(row["uid"]))
            if state_to_uid[state_id] != str(row["uid"]):
                raise RuntimeError(f"routed state crosses UIDs: {state_id}")
            references += 1
    if references != int(config["routed"]["expected_state_references"]):
        raise RuntimeError("routed replay state-reference count differs")

    phase76_root = resolve_path(config["sources"]["phase76_contract"]).parent
    source_states = read_jsonl(config["sources"]["routed_state_manifest"])
    # Routed states are keyed by state_id; many intentionally share one UID.
    state_index: dict[str, dict[str, Any]] = {}
    for source in source_states:
        state_id = str(source["state_id"])
        if state_id in state_index:
            raise RuntimeError(f"duplicate Phase-76 state ID: {state_id}")
        state_index[state_id] = dict(source)
    if set(state_index) != set(anchor_candidates):
        raise RuntimeError("routed state manifest and replay references differ")

    metadata = {uid: rows[0] for uid, rows in grouped.items()}
    output: list[dict[str, Any]] = []
    for state_id, source in sorted(state_index.items()):
        uid = str(source["uid"])
        program_id = min(anchor_candidates[state_id])
        program = program_by_id[program_id]
        if uid != str(program["uid"]):
            raise RuntimeError(f"anchor program UID differs: {state_id}")
        state_file = phase76_root / str(source["state_file"])
        if not state_file.is_file():
            raise RuntimeError(f"missing Phase-76 routed state file: {state_file}")
        first = metadata[uid]
        output.append(
            {
                "schema_version": "predictability_stepA_routed_state_v1",
                "state_id": state_id,
                "state_sha256": str(source["state_sha256"]),
                "tensor_sha256": dict(source["tensor_sha256"]),
                "uid": uid,
                "dataset": str(first["dataset"]),
                "source_regime": str(first["source_regime"]),
                "image_group_id": str(first["image_group_id"]),
                "dense_outcome": str(first["dense_outcome"]),
                "dense_correct": str(first["dense_outcome"]) == "C",
                "dense_wrong": str(first["dense_outcome"]) == "W",
                "trigger_layer": int(source["trigger_layer"]),
                "layer": int(source["layer"]),
                "trigger_relative_depth": int(source["layer"]) - int(source["trigger_layer"]),
                "prefix_actions": list(source["prefix_actions"]),
                "text_tokens": int(source["text_tokens"]),
                "visual_tokens": int(source["visual_tokens"]),
                "source_state_file": str(state_file),
                "anchor_program_id": program_id,
                "anchor_program_count": len(anchor_candidates[state_id]),
            }
        )
    return output, {"grouped": grouped, "program_by_id": program_by_id, "census": census}


def prepare(config_path: Path) -> None:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    external_root = resolve_path(config["external_dense_state_root"])
    if output_root.exists() or external_root.exists():
        raise RuntimeError("refusing to overwrite an existing Step-A output/cache root")
    runtime = runtime_state()
    if not runtime["cuda_available"] or int(runtime["cuda_device_count"]) != 4:
        raise RuntimeError("the frozen Step-A run requires four visible CUDA devices")
    for key in ("historical_feature_integrity", "canonical_feature_integrity"):
        audit = read_json(config["sources"][key])
        if not bool(audit.get("passed")):
            raise RuntimeError(f"source feature integrity did not pass: {key}")

    internal = _internal_population(config)
    dense_states = _dense_state_rows(internal, config)
    triggered = [row for row in internal if bool(row["p90"]["triggered"])]
    if len(triggered) != int(config["trigger"]["expected_triggered_uids"]):
        raise RuntimeError("P90 triggered-UID census differs")
    if len(dense_states) != int(config["trigger"]["expected_dense_states"]):
        raise RuntimeError("P90 dense-state census differs")
    routed_states, routed = _routed_state_rows(config)

    output_root.mkdir(parents=True)
    external_root.mkdir(parents=True)
    for relative in (
        "manifests", "stage1", "stage2_dense", "stage2_routed", "validation",
        "compute", "summaries", "smoke", "work/dense", "work/routed", "work/logs",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    (output_root / "stage2_dense/states").symlink_to(external_root, target_is_directory=True)

    internal_rows = []
    for row in internal:
        copy = dict(row)
        copy["sample"] = dict(row["sample"])
        copy["dense"] = dict(row["dense"])
        internal_rows.append(copy)
    atomic_jsonl(output_root / "manifests/internal_sample_manifest.jsonl", internal_rows)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in internal:
        groups[str(row["image_group_id"])].append(row)
    atomic_jsonl(
        output_root / "manifests/group_registry.jsonl",
        (
            {
                "schema_version": "predictability_stepA_group_registry_v1",
                "image_group_id": group,
                "uids": sorted(str(row["uid"]) for row in rows),
                "records": len(rows),
                "datasets": sorted({str(row["dataset"]) for row in rows}),
                "source_regimes": sorted({str(row["source_regime"]) for row in rows}),
            }
            for group, rows in sorted(groups.items())
        ),
    )
    atomic_jsonl(
        output_root / "manifests/p90_trigger_manifest.jsonl",
        ({**row["p90"], "image_group_id": row["image_group_id"]} for row in internal),
    )

    stage1_states = []
    stage1_features = []
    for row in internal:
        for layer in range(int(config["model"]["decoder_layers"])):
            state_id = _stage1_state_id(str(row["uid"]), layer)
            common = {
                "schema_version": "predictability_stepA_stage1_state_v1",
                "state_id": state_id,
                "uid": str(row["uid"]),
                "dataset": str(row["dataset"]),
                "source_regime": str(row["source_regime"]),
                "image_group_id": str(row["image_group_id"]),
                "layer": layer,
                "dense_correct": bool(row["dense_correct"]),
                "dense_wrong": bool(row["dense_wrong"]),
            }
            stage1_states.append(common)
            stage1_features.append(
                {
                    **common,
                    "feature_shard": str(row["stage1_feature_shard"]),
                    "feature_row_index": int(row["stage1_feature_row_index"]),
                    "feature_layer_index": layer,
                    "feature_components": list(config["stage1"]["feature_components"]),
                }
            )
    atomic_jsonl(output_root / "stage1/dense_state_manifest.jsonl", stage1_states)
    atomic_jsonl(output_root / "stage1/state_feature_manifest.jsonl", stage1_features)
    atomic_csv(
        output_root / "stage1/dense_outcome_labels.csv",
        [
            {
                "uid": row["uid"], "dataset": row["dataset"],
                "source_regime": row["source_regime"], "image_group_id": row["image_group_id"],
                "dense_correct": row["dense_correct"], "dense_wrong": row["dense_wrong"],
            }
            for row in internal
        ],
    )
    census = Counter(
        (str(row["dataset"]), str(row["source_regime"]), "W" if row["dense_wrong"] else "C")
        for row in internal
    )
    atomic_csv(
        output_root / "stage1/stage1_census_summary.csv",
        [
            {"dataset": key[0], "source_regime": key[1], "outcome": key[2],
             "samples": value, "layer_states": value * int(config["model"]["decoder_layers"])}
            for key, value in sorted(census.items())
        ],
    )
    atomic_jsonl(output_root / "stage2_dense/exact_state_manifest.jsonl", dense_states)
    atomic_jsonl(output_root / "stage2_routed/exact_state_manifest.jsonl", routed_states)
    atomic_csv(
        output_root / "stage2_routed/routed_state_dedup.csv",
        [
            {"unique_states": len(routed_states), "state_references": routed["census"]["route_state_occurrences"],
             "deduplicated_references": routed["census"]["route_state_occurrences"] - len(routed_states),
             "programs": routed["census"]["programs"], "uids": routed["census"]["uids"]}
        ],
    )

    dense_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dense_states:
        dense_by_uid[str(row["uid"])].append(row)
    atomic_jsonl(
        output_root / "work/dense_schedule.jsonl",
        (
            {"uid": uid, "worker_rank": index % int(config["world_size"]),
             "state_ids": [row["state_id"] for row in rows],
             "layers": [row["layer"] for row in rows]}
            for index, (uid, rows) in enumerate(sorted(dense_by_uid.items()))
        ),
    )
    routed_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in routed_states:
        routed_by_uid[str(row["uid"])].append(row)
    atomic_jsonl(
        output_root / "work/routed_schedule.jsonl",
        (
            {"uid": uid, "worker_rank": index % int(config["world_size"]),
             "state_ids": [row["state_id"] for row in rows],
             "anchor_program_ids": sorted({str(row["anchor_program_id"]) for row in rows})}
            for index, (uid, rows) in enumerate(sorted(routed_by_uid.items()))
        ),
    )

    phase69 = read_json(config["sources"]["phase69_contract"])
    internal_paths = (
        "manifests/internal_sample_manifest.jsonl", "manifests/group_registry.jsonl",
        "manifests/p90_trigger_manifest.jsonl", "stage1/dense_state_manifest.jsonl",
        "stage1/dense_outcome_labels.csv", "stage1/state_feature_manifest.jsonl",
        "stage1/stage1_census_summary.csv", "stage2_dense/exact_state_manifest.jsonl",
        "stage2_routed/exact_state_manifest.jsonl", "stage2_routed/routed_state_dedup.csv",
        "work/dense_schedule.jsonl", "work/routed_schedule.jsonl",
    )
    contract = {
        "schema_version": "predictability_stepA_measurement_contract_v1",
        "created_at": utc_now(),
        "static_config": config,
        "config_sha256": file_sha256(config_path),
        "source_sha256": {key: file_sha256(path) for key, path in _source_paths(config).items()},
        "bound_code_sha256": {path: file_sha256(path) for path in BOUND_CODE},
        "internal_manifest_sha256": {
            path: file_sha256(output_root / path) for path in internal_paths
        },
        "model_snapshot_sha256": phase69["model_snapshot_sha256"],
        "git": git_state(),
        "runtime": runtime,
        "population": {
            "internal_samples": len(internal),
            "historical_samples": sum(row["source_regime"] == "historical" for row in internal),
            "canonical_samples": sum(row["source_regime"] == "canonical" for row in internal),
            "image_groups": len(groups),
            "stage1_layer_states": len(stage1_states),
            "p90_triggered_uids": len(triggered),
            "dense_stage2_states": len(dense_states),
            "dense_stage2_branches": len(dense_states) * len(ACTIONS),
            "routed_stage2_uids": routed["census"]["uids"],
            "routed_stage2_programs": routed["census"]["programs"],
            "routed_stage2_states": len(routed_states),
            "routed_stage2_branches": len(routed_states) * len(ACTIONS),
        },
        "review_reconciliation": {
            "verdict": "revised_then_proceed",
            "dense_reuse_gate": "fresh stratified native token and feature replay",
            "continuous_q_revision": "TextVQA multiplicity retained by weighted logsumexp; ChartQA uses literal annotated gold and reports relaxed-equivalence limitation",
            "routed_gate": "live state hashes and complete anchor continuation tokens/correctness",
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_contract.json", contract)
    protocol = f"""# Predictability Step-A frozen measurement protocol

- Contract: `{contract['contract_sha256']}`
- Internal Stage-1 population: 6,399 Historical-train + 4,000 Canonical = 10,399 samples / 291,172 `(sample, layer)` states.
- Stage-1 state: exact existing current-runtime `text_final`, `text_mean`, and `visual_mean` BF16 features, admitted only after source hashes and fresh stratified native replay pass.
- Trigger: robust ALL-source five-checkpoint probability mean, strict P90 `p > 0.9061332901863008`.
- Primary Stage-2: all 1,413 P90-triggered UIDs / 15,185 dense pre-action states.
- Secondary Stage-2: all 35,565 unique Phase-76 exact routed-prefix states, independently reconstructed and state-hash checked.
- Branches: `FULL`, `READ_ONLY`, `WRITE_ONLY`, `IGNORE` at the current layer, followed only by `FULL`.
- Continuous q: weighted log-sum-exp of per-reference token-mean log probability, without EOS. GQA uses one evaluator-normalized annotation; ChartQA uses the literal annotated gold; TextVQA uses EvalAI-normalized references with empirical frequency weights.
- Discrete outcomes: exact current LMMS-Eval task scoring and frozen binary thresholds.
- Limitation: ChartQA relaxed numeric correctness represents a tolerance interval; finite-string q measures the annotated gold string and is not the probability mass of that full interval.
- No outcome filtering, MCTS, suffix search, utility threshold, probe/router training, OOD evaluation, or predictability claim.
"""
    _atomic_bytes(output_root / "protocol.md", protocol.encode())
    _atomic_bytes(
        output_root / "validation/action_semantics_tests.md",
        b"# Action semantics\n\n- FULL = READ1 WRITE1\n- READ_ONLY = READ1 WRITE0\n- WRITE_ONLY = READ0 WRITE1\n- IGNORE = READ0 WRITE0\n- Every later layer is FULL.\n\nCovered by `tests/test_four_action_binary_executor.py` and frozen executor hashes.\n",
    )
    print(json.dumps({"prepared": True, "contract_sha256": contract["contract_sha256"], **contract["population"]}, sort_keys=True))


def verify_contract(
    config_path: Path, *, verify_model: bool = False
) -> tuple[dict[str, Any], Path]:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_contract.json")
    if contract.get("contract_sha256") != canonical_hash(contract):
        raise RuntimeError("Step-A contract hash differs")
    if contract.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError("Step-A config changed after freeze")
    for key, path in _source_paths(config).items():
        if file_sha256(path) != contract["source_sha256"][key]:
            raise RuntimeError(f"Step-A source changed after freeze: {key}")
    for path, digest in contract["bound_code_sha256"].items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"Step-A bound code changed after freeze: {path}")
    for path, digest in contract["internal_manifest_sha256"].items():
        if file_sha256(output_root / path) != digest:
            raise RuntimeError(f"Step-A prepared manifest changed after freeze: {path}")
    if git_state() != contract["git"]:
        raise RuntimeError("Git commit/branch/worktree status differs from Step-A contract")
    if runtime_state() != contract["runtime"]:
        raise RuntimeError("runtime/environment differs from Step-A contract")
    state_link = output_root / "stage2_dense/states"
    external = resolve_path(config["external_dense_state_root"])
    if not state_link.is_symlink() or state_link.resolve() != external or not external.is_dir():
        raise RuntimeError("dense-state external cache binding differs")
    if verify_model:
        _verify_model_snapshot(config, contract)
    return contract, output_root


def _decode(processor, generated: torch.Tensor) -> tuple[list[int], str]:
    ids = [int(value) for value in generated[0].detach().cpu().tolist()]
    text = processor.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    return ids, text


def _measure_output(
    processor,
    wrapped,
    output,
    input_ids: torch.Tensor,
    sample: Mapping[str, Any],
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    if output.cache is None:
        raise RuntimeError("measurement output is missing its prompt cache")
    answers = accepted_answer_specs(sample)
    reference_rows = []
    mean_scores = []
    for answer in answers:
        token_ids = processor.tokenizer(
            str(answer["text"]), add_special_tokens=False, return_tensors="pt"
        ).input_ids[0].to(output.prompt_logits.device)
        score = score_token_ids_from_cached_prompt(
            wrapped, output.prompt_logits, output.inputs, output.cache, token_ids
        )
        mean_scores.append(float(score.mean_logprob))
        reference_rows.append(
            {
                "text": str(answer["text"]),
                "weight": float(answer["weight"]),
                "token_ids": score.token_ids,
                "token_logprobs": score.token_logprobs,
                "sequence_logprob": score.sequence_logprob,
                "mean_logprob": score.mean_logprob,
            }
        )
    q = aggregate_reference_mean_logprobs(answers, mean_scores)
    generated = greedy_generate_from_cached_prompt(
        wrapped,
        output.prompt_logits,
        output.inputs,
        output.cache,
        input_ids,
        max_new_tokens=max_new_tokens,
    ).generated_ids
    generated_ids, generated_text = _decode(processor, generated)
    lmms = score_lmms_sample(
        dataset=str(sample["dataset"]),
        prediction=generated_text,
        answer=str(sample["answer"]),
        answers=sample.get("all_answer_norms"),
        uid=str(sample["uid"]),
    )
    return {
        "mean_logprob": q,
        "accepted_answer_scores": reference_rows,
        "generated_token_ids": generated_ids,
        "generated_answer": generated_text,
        "lmms_metric": lmms.metric_name,
        "lmms_raw_score": lmms.raw_score,
        "lmms_threshold": lmms.correctness_threshold,
        "correct": bool(lmms.correct),
    }


def _cached_state(
    text_state: torch.Tensor,
    visual_state: torch.Tensor,
    text_mask: torch.Tensor,
    visual_mask: torch.Tensor,
) -> dict[str, Any]:
    compact = compact_router_state(text_state, visual_state, text_mask, visual_mask)
    state = {
        name: value.detach().cpu().to(torch.bfloat16).contiguous()
        if name in {"text_states", "visual_states"}
        else value.detach().cpu().bool().contiguous()
        for name, value in compact.items()
    }
    hashes = _state_tensor_hashes(state)
    return {**state, "tensor_sha256": hashes, "state_sha256": _combined_state_hash(hashes)}


def _branch_semantics(output, layer: int, action: str) -> None:
    stats = output.layer_stats[int(layer)]
    expected = {
        "FULL": (True, True),
        "READ_ONLY": (True, False),
        "WRITE_ONLY": (False, True),
        "IGNORE": (False, False),
    }[action]
    if (bool(stats.read_on), bool(stats.write_on)) != expected or stats.action != action:
        raise RuntimeError(f"action semantics differ at layer {layer}: {action}")


def _branch_all_actions(
    *,
    config: Mapping[str, Any],
    processor,
    wrapped,
    baseline,
    input_ids: torch.Tensor,
    sample: Mapping[str, Any],
    layer: int,
    expected_prefix: Sequence[str],
    dense_baseline: Mapping[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    branches: dict[str, dict[str, Any]] = {}
    parity: list[dict[str, Any]] = []
    total_layers = int(config["model"]["decoder_layers"])
    for action in ACTIONS:
        suffix = [action] + ["FULL"] * (total_layers - int(layer) - 1)
        if all(value == "FULL" for value in baseline.layer_actions):
            output = capture_four_action_suffix_from_full_baseline(
                wrapped, baseline, int(layer), suffix
            )
        else:
            output = capture_four_action_suffix_from_route_baseline(
                wrapped,
                baseline,
                int(layer),
                suffix,
                expected_prefix=expected_prefix,
            )
        _branch_semantics(output, int(layer), action)
        measured = _measure_output(
            processor,
            wrapped,
            output,
            input_ids,
            sample,
            max_new_tokens=int(config["measurement"]["generation_max_new_tokens"]),
        )
        branches[action] = measured
        if action == "FULL" and dense_baseline is not None:
            token_exact = measured["generated_token_ids"] == dense_baseline["generated_token_ids"]
            correctness_exact = measured["correct"] == dense_baseline["correct"]
            q_abs = abs(float(measured["mean_logprob"]) - float(dense_baseline["mean_logprob"]))
            parity.append(
                {
                    "token_exact": token_exact,
                    "correctness_exact": correctness_exact,
                    "q_abs_difference": q_abs,
                    "passed": token_exact
                    and correctness_exact
                    and q_abs <= float(config["measurement"]["q_absolute_tolerance"]),
                }
            )
            if not parity[-1]["passed"]:
                raise RuntimeError(
                    f"FULL branch dense parity failed: {sample['uid']} L{layer} {parity[-1]}"
                )
        del output
    return branches, parity


def _load_internal_index(output_root: Path) -> dict[str, dict[str, Any]]:
    return _index_unique(
        read_jsonl(output_root / "manifests/internal_sample_manifest.jsonl"),
        "Step-A internal manifest",
    )


def _require_smoke(contract: Mapping[str, Any], output_root: Path) -> None:
    report = read_json(output_root / "smoke/smoke_summary.json")
    if (
        report.get("contract_sha256") != contract["contract_sha256"]
        or not bool(report.get("passed"))
    ):
        raise RuntimeError("Step-A smoke has not passed under the frozen contract")


def dense_worker(config_path: Path, rank: int, *, resume: bool) -> None:
    contract, output_root = verify_contract(config_path, verify_model=True)
    _require_smoke(contract, output_root)
    config = contract["static_config"]
    rank = int(rank)
    if rank < 0 or rank >= int(config["world_size"]):
        raise ValueError("invalid dense worker rank")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    configure_dense_determinism(int(config["seed"]) + rank, config["backend_settings"])
    processor, _base, wrapped = _load_model(config, device)
    internal = _load_internal_index(output_root)
    state_manifest = read_jsonl(output_root / "stage2_dense/exact_state_manifest.jsonl")
    states_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in state_manifest:
        states_by_uid[str(row["uid"])].append(row)
    schedule = [
        row for row in read_jsonl(output_root / "work/dense_schedule.jsonl")
        if int(row["worker_rank"]) == rank
    ]
    rank_root = output_root / f"work/dense/rank{rank:02d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    started = time.monotonic()
    for item in schedule:
        uid = str(item["uid"])
        result_path = rank_root / f"{_uid_slug(uid)}.json"
        state_path = resolve_path(config["external_dense_state_root"]) / f"{_uid_slug(uid)}.pt"
        if resume and result_path.is_file() and state_path.is_file():
            old = read_json(result_path)
            if (
                old.get("contract_sha256") == contract["contract_sha256"]
                and old.get("uid") == uid
                and old.get("state_file_sha256") == file_sha256(state_path)
                and list(old.get("state_ids", [])) == list(item["state_ids"])
            ):
                completed += 1
                continue
        sample_row = internal[uid]
        sample = sample_row["sample"]
        inputs, metadata = build_dense_inputs(processor, sample, device)
        prepared = build_binary_inputs(wrapped, inputs)
        baseline = capture_four_action_route(
            wrapped,
            {},
            ["FULL"] * int(config["model"]["decoder_layers"]),
            prepared_inputs=prepared,
            use_cache=True,
            native_full_rows=True,
        )
        baseline_measurement = _measure_output(
            processor,
            wrapped,
            baseline,
            inputs["input_ids"],
            sample,
            max_new_tokens=int(config["measurement"]["generation_max_new_tokens"]),
        )
        expected = sample_row["dense"]
        if (
            baseline_measurement["generated_token_ids"] != expected["generated_token_ids"]
            or baseline_measurement["correct"] != bool(expected["current_dense_correct"])
            or metadata["consumed_image_sha256"] != sample["image_content_sha256"]
        ):
            raise RuntimeError(f"live dense baseline differs for {uid}")

        cached_states: dict[str, dict[str, Any]] = {}
        results: list[dict[str, Any]] = []
        full_parity: list[dict[str, Any]] = []
        for state_row in sorted(states_by_uid[uid], key=lambda row: int(row["layer"])):
            layer = int(state_row["layer"])
            state_id = str(state_row["state_id"])
            text, visual = baseline.pre_layer_states[layer]
            cached = _cached_state(
                text, visual, prepared.text_valid_mask, prepared.visual_valid_mask
            )
            cached_states[state_id] = {
                **cached,
                "state_id": state_id,
                "uid": uid,
                "layer": layer,
                "trigger_layer": int(state_row["trigger_layer"]),
                "prefix_actions": list(state_row["prefix_actions"]),
            }
            branches, parity = _branch_all_actions(
                config=config,
                processor=processor,
                wrapped=wrapped,
                baseline=baseline,
                input_ids=inputs["input_ids"],
                sample=sample,
                layer=layer,
                expected_prefix=["FULL"] * layer,
                dense_baseline=baseline_measurement,
            )
            derived = derive_utility_row(branches)
            results.append(
                {
                    "state_id": state_id,
                    "uid": uid,
                    "layer": layer,
                    "branches": branches,
                    "branch_order": list(ACTIONS),
                    "derived": derived,
                }
            )
            full_parity.extend({"state_id": state_id, "layer": layer, **row} for row in parity)
        validate_complete_state_results(
            item["state_ids"],
            [{"state_id": row["state_id"], "branches": row["branch_order"]} for row in results],
        )
        payload = {
            "schema_version": "predictability_stepA_dense_uid_states_v1",
            "contract_sha256": contract["contract_sha256"],
            "uid": uid,
            "states": cached_states,
        }
        atomic_torch(state_path, payload)
        state_file_sha = file_sha256(state_path)
        result = {
            "schema_version": "predictability_stepA_dense_uid_result_v1",
            "contract_sha256": contract["contract_sha256"],
            "uid": uid,
            "dataset": sample_row["dataset"],
            "source_regime": sample_row["source_regime"],
            "dense_correct": sample_row["dense_correct"],
            "trigger_layer": sample_row["p90"]["first_trigger_layer"],
            "state_ids": list(item["state_ids"]),
            "state_file": str(state_path),
            "state_file_sha256": state_file_sha,
            "state_feature_rows": [
                {
                    "state_id": state_id,
                    "state_sha256": cached_states[state_id]["state_sha256"],
                    "tensor_sha256": cached_states[state_id]["tensor_sha256"],
                    "text_tokens": int(cached_states[state_id]["text_states"].shape[1]),
                    "visual_tokens": int(cached_states[state_id]["visual_states"].shape[1]),
                }
                for state_id in item["state_ids"]
            ],
            "baseline": baseline_measurement,
            "full_parity": full_parity,
            "state_results": results,
            "consumed_image_sha256": metadata["consumed_image_sha256"],
        }
        atomic_json(result_path, result)
        completed += 1
        del baseline, prepared, inputs, payload, cached_states, results
        torch.cuda.empty_cache()
        if completed % 10 == 0 or completed == len(schedule):
            print(json.dumps({"domain": "dense", "rank": rank, "completed_uids": completed,
                              "assigned_uids": len(schedule), "elapsed_seconds": time.monotonic() - started}), flush=True)
    atomic_json(
        rank_root / "complete.json",
        {"schema_version": "predictability_stepA_rank_complete_v1",
         "contract_sha256": contract["contract_sha256"], "domain": "dense", "rank": rank,
         "expected_uids": len(schedule), "completed_uids": completed, "completed_at": utc_now(),
         "elapsed_seconds": time.monotonic() - started},
    )


def routed_worker(config_path: Path, rank: int, *, resume: bool) -> None:
    contract, output_root = verify_contract(config_path, verify_model=True)
    _require_smoke(contract, output_root)
    config = contract["static_config"]
    rank = int(rank)
    if rank < 0 or rank >= int(config["world_size"]):
        raise ValueError("invalid routed worker rank")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    configure_dense_determinism(int(config["seed"]) + 100 + rank, config["backend_settings"])
    processor, _base, wrapped = _load_model(config, device)
    internal = _load_internal_index(output_root)
    state_manifest = read_jsonl(output_root / "stage2_routed/exact_state_manifest.jsonl")
    states_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in state_manifest:
        states_by_uid[str(row["uid"])].append(row)
    programs = read_jsonl(config["sources"]["routed_program_manifest"])
    program_by_id = {str(row["program_id"]): row for row in programs}
    uid_files = _index_unique(
        read_jsonl(config["sources"]["routed_uid_files"]), "Phase-76 UID state files"
    )
    schedule = [
        row for row in read_jsonl(output_root / "work/routed_schedule.jsonl")
        if int(row["worker_rank"]) == rank
    ]
    rank_root = output_root / f"work/routed/rank{rank:02d}"
    rank_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    started = time.monotonic()
    phase76_root = resolve_path(config["sources"]["phase76_contract"]).parent
    for item in schedule:
        uid = str(item["uid"])
        result_path = rank_root / f"{_uid_slug(uid)}.json"
        if resume and result_path.is_file():
            old = read_json(result_path)
            if (
                old.get("contract_sha256") == contract["contract_sha256"]
                and old.get("uid") == uid
                and list(old.get("state_ids", [])) == list(item["state_ids"])
                and bool(old.get("all_live_state_hashes_exact"))
                and bool(old.get("all_anchor_continuations_exact"))
            ):
                completed += 1
                continue
        sample_row = internal.get(uid)
        if sample_row is None:
            raise RuntimeError(f"routed UID is outside the internal population: {uid}")
        sample = sample_row["sample"]
        inputs, metadata = build_dense_inputs(processor, sample, device)
        prepared = build_binary_inputs(wrapped, inputs)

        uid_file_row = uid_files[uid]
        source_state_path = phase76_root / str(uid_file_row["state_file"])
        if file_sha256(source_state_path) != str(uid_file_row["state_file_sha256"]):
            raise RuntimeError(f"Phase-76 UID state file hash differs: {uid}")
        source_payload = torch.load(source_state_path, map_location="cpu", weights_only=False)
        if str(source_payload.get("uid")) != uid:
            raise RuntimeError(f"Phase-76 state payload UID differs: {uid}")

        current_states = states_by_uid[uid]
        by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in current_states:
            by_anchor[str(row["anchor_program_id"])].append(row)
        results: list[dict[str, Any]] = []
        anchor_parity: list[dict[str, Any]] = []
        live_hash_rows: list[dict[str, Any]] = []
        for program_id, assigned_states in sorted(by_anchor.items()):
            program = program_by_id[program_id]
            actions = [str(value) for value in program["full_actions"]]
            route = capture_four_action_route(
                wrapped,
                {},
                actions,
                prepared_inputs=prepared,
                use_cache=True,
                native_full_rows=True,
            )
            anchor = _measure_output(
                processor,
                wrapped,
                route,
                inputs["input_ids"],
                sample,
                max_new_tokens=int(config["measurement"]["generation_max_new_tokens"]),
            )
            anchor_exact = (
                anchor["generated_token_ids"] == [int(value) for value in program["expected_generated_token_ids"]]
                and bool(anchor["correct"])
            )
            anchor_parity.append(
                {"program_id": program_id, "generated_token_exact": anchor["generated_token_ids"] == [int(value) for value in program["expected_generated_token_ids"]],
                 "correct": bool(anchor["correct"]), "passed": anchor_exact}
            )
            if not anchor_exact:
                raise RuntimeError(f"routed anchor continuation parity failed: {program_id}")

            for state_row in sorted(assigned_states, key=lambda row: int(row["layer"])):
                state_id = str(state_row["state_id"])
                layer = int(state_row["layer"])
                trigger = int(state_row["trigger_layer"])
                expected_route_prefix = actions[:layer]
                if (
                    expected_route_prefix[:trigger] != ["FULL"] * trigger
                    or expected_route_prefix[trigger:] != list(state_row["prefix_actions"])
                ):
                    raise RuntimeError(f"anchor prefix differs from routed state identity: {state_id}")
                if state_id not in source_payload["states"]:
                    raise RuntimeError(f"state missing from Phase-76 UID payload: {state_id}")
                cached = source_payload["states"][state_id]
                cached_hashes = _state_tensor_hashes(cached)
                if (
                    cached_hashes != dict(state_row["tensor_sha256"])
                    or _combined_state_hash(cached_hashes) != str(state_row["state_sha256"])
                ):
                    raise RuntimeError(f"Phase-76 cached state tensor hash differs: {state_id}")
                live_text, live_visual = route.pre_layer_states[layer]
                live = _cached_state(
                    live_text, live_visual, prepared.text_valid_mask, prepared.visual_valid_mask
                )
                hash_exact = (
                    live["tensor_sha256"] == dict(state_row["tensor_sha256"])
                    and live["state_sha256"] == str(state_row["state_sha256"])
                )
                live_hash_rows.append(
                    {"state_id": state_id, "layer": layer, "hash_exact": hash_exact,
                     "live_state_sha256": live["state_sha256"], "expected_state_sha256": state_row["state_sha256"]}
                )
                if not hash_exact:
                    raise RuntimeError(f"live routed state hash differs: {state_id}")
                branches, _unused = _branch_all_actions(
                    config=config,
                    processor=processor,
                    wrapped=wrapped,
                    baseline=route,
                    input_ids=inputs["input_ids"],
                    sample=sample,
                    layer=layer,
                    expected_prefix=expected_route_prefix,
                    dense_baseline=None,
                )
                results.append(
                    {"state_id": state_id, "uid": uid, "layer": layer,
                     "branches": branches, "branch_order": list(ACTIONS),
                     "derived": derive_utility_row(branches)}
                )
            del route
        validate_complete_state_results(
            item["state_ids"],
            [{"state_id": row["state_id"], "branches": row["branch_order"]} for row in results],
        )
        result = {
            "schema_version": "predictability_stepA_routed_uid_result_v1",
            "contract_sha256": contract["contract_sha256"],
            "uid": uid,
            "dataset": sample_row["dataset"],
            "source_regime": sample_row["source_regime"],
            "dense_correct": sample_row["dense_correct"],
            "state_ids": list(item["state_ids"]),
            "source_state_file": str(source_state_path),
            "source_state_file_sha256": uid_file_row["state_file_sha256"],
            "anchor_parity": anchor_parity,
            "live_state_hashes": live_hash_rows,
            "all_anchor_continuations_exact": all(row["passed"] for row in anchor_parity),
            "all_live_state_hashes_exact": all(row["hash_exact"] for row in live_hash_rows),
            "state_results": results,
            "consumed_image_sha256": metadata["consumed_image_sha256"],
        }
        atomic_json(result_path, result)
        completed += 1
        del inputs, prepared, source_payload, results
        torch.cuda.empty_cache()
        if completed % 5 == 0 or completed == len(schedule):
            print(json.dumps({"domain": "routed", "rank": rank, "completed_uids": completed,
                              "assigned_uids": len(schedule), "elapsed_seconds": time.monotonic() - started}), flush=True)
    atomic_json(
        rank_root / "complete.json",
        {"schema_version": "predictability_stepA_rank_complete_v1",
         "contract_sha256": contract["contract_sha256"], "domain": "routed", "rank": rank,
         "expected_uids": len(schedule), "completed_uids": completed, "completed_at": utc_now(),
         "elapsed_seconds": time.monotonic() - started},
    )


def _select_cover(rows: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    remaining = [dict(row) for row in rows]
    selected: list[dict[str, Any]] = []
    covered: set[tuple[str, str]] = set()
    covered_datasets: set[str] = set()
    while remaining and len(selected) < count:
        best = max(
            remaining,
            key=lambda row: (
                int((str(row["dataset"]), "W" if row.get("dense_wrong") else "C") not in covered),
                int(str(row["dataset"]) not in covered_datasets),
                -int(row.get("layer", 0)),
                sha256(str(row.get("state_id", row["uid"])).encode()).hexdigest(),
            ),
        )
        selected.append(best)
        covered.add((str(best["dataset"]), "W" if best.get("dense_wrong") else "C"))
        covered_datasets.add(str(best["dataset"]))
        remaining.remove(best)
    if len(selected) != count or covered_datasets != {"gqa", "chartqa", "textvqa"}:
        raise RuntimeError("cannot construct the required stratified smoke set")
    return selected


def _branches_reproducible(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    tolerance: float,
) -> tuple[bool, list[dict[str, Any]]]:
    rows = []
    passed = True
    for action in ACTIONS:
        q_abs = abs(float(left[action]["mean_logprob"]) - float(right[action]["mean_logprob"]))
        row = {
            "action": action,
            "token_exact": left[action]["generated_token_ids"] == right[action]["generated_token_ids"],
            "correctness_exact": left[action]["correct"] == right[action]["correct"],
            "q_abs_difference": q_abs,
        }
        row["passed"] = row["token_exact"] and row["correctness_exact"] and q_abs <= tolerance
        passed = passed and bool(row["passed"])
        rows.append(row)
    return passed, rows


def smoke(config_path: Path, device_index: int) -> None:
    contract, output_root = verify_contract(config_path, verify_model=True)
    config = contract["static_config"]
    device_index = int(device_index)
    if device_index < 0 or device_index >= int(config["world_size"]):
        raise ValueError("invalid smoke GPU")
    configure_dense_determinism(int(config["seed"]), config["backend_settings"])
    internal = _load_internal_index(output_root)

    # Fresh native replay: one row for every dataset x source x dense outcome cell.
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in internal.values():
        cells[(str(row["dataset"]), str(row["source_regime"]), "W" if row["dense_wrong"] else "C")].append(row)
    native_selected = []
    for key, rows in sorted(cells.items()):
        native_selected.append(min(rows, key=lambda row: sha256(str(row["uid"]).encode()).hexdigest()))
    if len(native_selected) != int(config["stage1"]["fresh_replay_samples"]):
        raise RuntimeError("fresh dense replay does not cover every planned cell")
    processor, native, device = load_dense_runtime(
        config["model"]["snapshot_path"], config["model"]["revision"], device_index
    )
    native_rows = []
    loaded_shards: dict[str, Mapping[str, Any]] = {}
    for row in native_selected:
        generated = generate_dense(
            processor, native, device, dict(row["sample"]), extract_features=True
        )
        expected = row["dense"]
        shard_path = str(row["stage1_feature_shard"])
        if shard_path not in loaded_shards:
            loaded_shards[shard_path] = torch.load(
                shard_path, map_location="cpu", weights_only=False
            )
        shard = loaded_shards[shard_path]
        feature_exact = {
            name: torch.equal(
                generated.features[name],
                shard[name][int(row["stage1_feature_row_index"])],
            )
            for name in config["stage1"]["feature_components"]
        }
        item = {
            "uid": row["uid"], "dataset": row["dataset"],
            "source_regime": row["source_regime"],
            "dense_outcome": "W" if row["dense_wrong"] else "C",
            "generated_token_exact": generated.generated_ids == expected["generated_token_ids"],
            "correctness_exact": generated.correct == bool(expected["current_dense_correct"]),
            "feature_exact": feature_exact,
            "image_sha_exact": generated.input_metadata["consumed_image_sha256"] == row["sample"]["image_content_sha256"],
        }
        item["passed"] = (
            item["generated_token_exact"] and item["correctness_exact"]
            and item["image_sha_exact"] and all(feature_exact.values())
        )
        native_rows.append(item)
        if not item["passed"]:
            raise RuntimeError(f"fresh dense artifact replay failed: {row['uid']} {item}")
    atomic_jsonl(output_root / "smoke/dense_replay.jsonl", native_rows)
    del native, processor, loaded_shards
    torch.cuda.empty_cache()

    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    processor, _base, wrapped = _load_model(config, device)
    tolerance = float(config["measurement"]["q_absolute_tolerance"])
    dense_state_manifest = read_jsonl(output_root / "stage2_dense/exact_state_manifest.jsonl")
    dense_selected = _select_cover(
        dense_state_manifest, int(config["smoke"]["dense_state_count"])
    )
    dense_smoke_rows = []
    for state_row in dense_selected:
        sample_row = internal[str(state_row["uid"])]
        sample = sample_row["sample"]
        inputs, _metadata = build_dense_inputs(processor, sample, device)
        prepared = build_binary_inputs(wrapped, inputs)
        baseline = capture_four_action_route(
            wrapped, {}, ["FULL"] * 28, prepared_inputs=prepared,
            use_cache=True, native_full_rows=True
        )
        baseline_measurement = _measure_output(
            processor, wrapped, baseline, inputs["input_ids"], sample,
            max_new_tokens=int(config["measurement"]["generation_max_new_tokens"]),
        )
        if (
            baseline_measurement["generated_token_ids"] != sample_row["dense"]["generated_token_ids"]
            or baseline_measurement["correct"] != sample_row["dense_correct"]
        ):
            raise RuntimeError(f"smoke dense executor baseline differs: {sample_row['uid']}")
        first, parity1 = _branch_all_actions(
            config=config, processor=processor, wrapped=wrapped, baseline=baseline,
            input_ids=inputs["input_ids"], sample=sample, layer=int(state_row["layer"]),
            expected_prefix=["FULL"] * int(state_row["layer"]), dense_baseline=baseline_measurement,
        )
        second, parity2 = _branch_all_actions(
            config=config, processor=processor, wrapped=wrapped, baseline=baseline,
            input_ids=inputs["input_ids"], sample=sample, layer=int(state_row["layer"]),
            expected_prefix=["FULL"] * int(state_row["layer"]), dense_baseline=baseline_measurement,
        )
        repeat_pass, repeats = _branches_reproducible(first, second, tolerance)
        row = {"domain": "dense", "state_id": state_row["state_id"], "uid": state_row["uid"],
               "layer": state_row["layer"], "dataset": state_row["dataset"],
               "full_parity": parity1 + parity2, "repeats": repeats,
               "passed": repeat_pass and all(item["passed"] for item in parity1 + parity2)}
        dense_smoke_rows.append(row)
        if not row["passed"]:
            raise RuntimeError(f"dense branch smoke failed: {state_row['state_id']}")
        del baseline, prepared, inputs
        torch.cuda.empty_cache()
    atomic_jsonl(output_root / "smoke/dense_branch_reproducibility.jsonl", dense_smoke_rows)

    routed_state_manifest = read_jsonl(output_root / "stage2_routed/exact_state_manifest.jsonl")
    routed_selected = _select_cover(
        routed_state_manifest, int(config["smoke"]["routed_state_count"])
    )
    program_by_id = {
        str(row["program_id"]): row
        for row in read_jsonl(config["sources"]["routed_program_manifest"])
    }
    uid_files = _index_unique(read_jsonl(config["sources"]["routed_uid_files"]), "routed UID files")
    phase76_root = resolve_path(config["sources"]["phase76_contract"]).parent
    routed_smoke_rows = []
    for state_row in routed_selected:
        sample_row = internal[str(state_row["uid"])]
        sample = sample_row["sample"]
        program = program_by_id[str(state_row["anchor_program_id"])]
        inputs, _metadata = build_dense_inputs(processor, sample, device)
        prepared = build_binary_inputs(wrapped, inputs)
        route = capture_four_action_route(
            wrapped, {}, program["full_actions"], prepared_inputs=prepared,
            use_cache=True, native_full_rows=True
        )
        anchor = _measure_output(
            processor, wrapped, route, inputs["input_ids"], sample,
            max_new_tokens=int(config["measurement"]["generation_max_new_tokens"]),
        )
        anchor_exact = (
            anchor["generated_token_ids"] == [int(value) for value in program["expected_generated_token_ids"]]
            and bool(anchor["correct"])
        )
        state_path = phase76_root / str(uid_files[str(state_row["uid"])]["state_file"])
        if file_sha256(state_path) != uid_files[str(state_row["uid"])]["state_file_sha256"]:
            raise RuntimeError(f"smoke routed source file differs: {state_row['uid']}")
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        cached = payload["states"][str(state_row["state_id"])]
        cached_hashes = _state_tensor_hashes(cached)
        live_text, live_visual = route.pre_layer_states[int(state_row["layer"])]
        live = _cached_state(live_text, live_visual, prepared.text_valid_mask, prepared.visual_valid_mask)
        state_exact = (
            cached_hashes == dict(state_row["tensor_sha256"])
            and live["tensor_sha256"] == dict(state_row["tensor_sha256"])
            and live["state_sha256"] == str(state_row["state_sha256"])
        )
        first, _ = _branch_all_actions(
            config=config, processor=processor, wrapped=wrapped, baseline=route,
            input_ids=inputs["input_ids"], sample=sample, layer=int(state_row["layer"]),
            expected_prefix=program["full_actions"][: int(state_row["layer"])], dense_baseline=None,
        )
        second, _ = _branch_all_actions(
            config=config, processor=processor, wrapped=wrapped, baseline=route,
            input_ids=inputs["input_ids"], sample=sample, layer=int(state_row["layer"]),
            expected_prefix=program["full_actions"][: int(state_row["layer"])], dense_baseline=None,
        )
        repeat_pass, repeats = _branches_reproducible(first, second, tolerance)
        row = {"domain": "routed", "state_id": state_row["state_id"], "uid": state_row["uid"],
               "layer": state_row["layer"], "dataset": state_row["dataset"],
               "anchor_program_id": state_row["anchor_program_id"],
               "anchor_continuation_exact": anchor_exact, "state_hash_exact": state_exact,
               "repeats": repeats, "passed": anchor_exact and state_exact and repeat_pass}
        routed_smoke_rows.append(row)
        if not row["passed"]:
            raise RuntimeError(f"routed branch smoke failed: {state_row['state_id']}")
        del route, prepared, inputs, payload
        torch.cuda.empty_cache()
    atomic_jsonl(output_root / "smoke/routed_branch_reproducibility.jsonl", routed_smoke_rows)
    report = {
        "schema_version": "predictability_stepA_smoke_v1",
        "contract_sha256": contract["contract_sha256"],
        "passed": all(row["passed"] for row in native_rows + dense_smoke_rows + routed_smoke_rows),
        "fresh_dense_replay_samples": len(native_rows),
        "dense_branch_states": len(dense_smoke_rows),
        "routed_branch_states": len(routed_smoke_rows),
        "fresh_dense_tokens_exact": all(row["generated_token_exact"] for row in native_rows),
        "fresh_dense_features_exact": all(all(row["feature_exact"].values()) for row in native_rows),
        "dense_full_branch_parity": all(all(item["passed"] for item in row["full_parity"]) for row in dense_smoke_rows),
        "all_branch_repeats_reproducible": all(row["passed"] for row in dense_smoke_rows + routed_smoke_rows),
        "routed_anchor_continuations_exact": all(row["anchor_continuation_exact"] for row in routed_smoke_rows),
        "routed_state_hashes_exact": all(row["state_hash_exact"] for row in routed_smoke_rows),
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "smoke/smoke_summary.json", report)
    if not report["passed"]:
        raise RuntimeError("Step-A implementation/parity smoke failed")
    print(json.dumps(report, sort_keys=True))


def _collect_domain_results(
    contract: Mapping[str, Any], output_root: Path, domain: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if domain not in {"dense", "routed"}:
        raise ValueError("unknown aggregation domain")
    config = contract["static_config"]
    schedule = read_jsonl(output_root / f"work/{domain}_schedule.jsonl")
    expected = {str(row["uid"]): row for row in schedule}
    if len(expected) != len(schedule):
        raise RuntimeError(f"duplicate {domain} schedule UID")
    completions = []
    results = []
    for rank in range(int(config["world_size"])):
        rank_root = output_root / f"work/{domain}/rank{rank:02d}"
        failures = sorted(rank_root.glob("failure_*.json"))
        if failures:
            raise RuntimeError(f"{domain} worker failures exist: {failures[:3]}")
        completion = read_json(rank_root / "complete.json")
        expected_rank = sum(int(row["worker_rank"]) == rank for row in schedule)
        if (
            completion.get("contract_sha256") != contract["contract_sha256"]
            or completion.get("domain") != domain
            or int(completion.get("rank", -1)) != rank
            or int(completion.get("expected_uids", -1)) != expected_rank
            or int(completion.get("completed_uids", -1)) != expected_rank
        ):
            raise RuntimeError(f"invalid {domain} rank completion: {rank}")
        completions.append(completion)
    for uid, item in expected.items():
        rank = int(item["worker_rank"])
        row = read_json(output_root / f"work/{domain}/rank{rank:02d}/{_uid_slug(uid)}.json")
        if (
            row.get("contract_sha256") != contract["contract_sha256"]
            or row.get("uid") != uid
            or list(row.get("state_ids", [])) != list(item["state_ids"])
        ):
            raise RuntimeError(f"invalid {domain} UID result: {uid}")
        if domain == "dense":
            if file_sha256(row["state_file"]) != row["state_file_sha256"]:
                raise RuntimeError(f"dense state file hash differs: {uid}")
            if not all(bool(value["passed"]) for value in row["full_parity"]):
                raise RuntimeError(f"dense FULL parity failed: {uid}")
        else:
            if not row["all_live_state_hashes_exact"] or not row["all_anchor_continuations_exact"]:
                raise RuntimeError(f"routed provenance/parity failed: {uid}")
        validate_complete_state_results(
            item["state_ids"],
            [
                {"state_id": state["state_id"], "branches": state["branch_order"]}
                for state in row["state_results"]
            ],
        )
        results.append(row)
    return results, completions


def _summary(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty value set")

    def quantile(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = p * (len(ordered) - 1)
        low = int(math.floor(position))
        high = int(math.ceil(position))
        fraction = position - low
        return ordered[low] * (1.0 - fraction) + ordered[high] * fraction

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "std": statistics.pstdev(ordered),
        "min": ordered[0],
        "q01": quantile(0.01),
        "q05": quantile(0.05),
        "q25": quantile(0.25),
        "q75": quantile(0.75),
        "q95": quantile(0.95),
        "q99": quantile(0.99),
        "max": ordered[-1],
        "negative": sum(value < 0.0 for value in ordered),
        "zero": sum(value == 0.0 for value in ordered),
        "positive": sum(value > 0.0 for value in ordered),
    }


def _distribution_rows(
    utilities: Sequence[Mapping[str, Any]], group_fields: Sequence[str]
) -> list[dict[str, Any]]:
    metrics = ("u_read", "u_write", "u_interaction")
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in utilities:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        prefix = dict(zip(group_fields, key))
        for metric in metrics:
            output.append({**prefix, "metric": metric, **_summary([float(row[metric]) for row in rows])})
    return output


def aggregate_domain(config_path: Path, domain: str) -> None:
    contract, output_root = verify_contract(config_path)
    _require_smoke(contract, output_root)
    config = contract["static_config"]
    results, completions = _collect_domain_results(contract, output_root, domain)
    root = output_root / ("stage2_dense" if domain == "dense" else "stage2_routed")
    state_manifest = read_jsonl(root / "exact_state_manifest.jsonl")
    metadata = {str(row["state_id"]): row for row in state_manifest}
    if len(metadata) != len(state_manifest):
        raise RuntimeError(f"duplicate {domain} state manifest ID")
    branch_rows: list[dict[str, Any]] = []
    utility_rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    algebra_rows: list[dict[str, Any]] = []
    full_parity_rows: list[dict[str, Any]] = []
    state_results = []
    for uid_result in results:
        if domain == "dense":
            feature_by_state = {
                str(row["state_id"]): row for row in uid_result["state_feature_rows"]
            }
            for row in uid_result["full_parity"]:
                full_parity_rows.append(
                    {"domain": domain, "uid": uid_result["uid"], **row}
                )
        for state in uid_result["state_results"]:
            state_id = str(state["state_id"])
            meta = metadata[state_id]
            state_results.append(
                {"state_id": state_id, "branches": list(state["branch_order"])}
            )
            common = {
                "state_id": state_id,
                "uid": str(meta["uid"]),
                "dataset": str(meta["dataset"]),
                "source_regime": str(meta["source_regime"]),
                "image_group_id": str(meta["image_group_id"]),
                "dense_correct": bool(meta["dense_correct"]),
                "dense_wrong": bool(meta["dense_wrong"]),
                "trigger_layer": int(meta["trigger_layer"]),
                "layer": int(meta["layer"]),
                "trigger_relative_depth": int(meta["trigger_relative_depth"]),
            }
            for action in ACTIONS:
                branch_rows.append(
                    {"schema_version": "predictability_stepA_four_branch_result_v1",
                     "contract_sha256": contract["contract_sha256"], **common,
                     "action": action, **state["branches"][action]}
                )
            recomputed = derive_utility_row(state["branches"])
            differences = {
                key: abs(float(recomputed[key]) - float(state["derived"][key]))
                for key in (
                    "u_read_w1", "u_read_w0", "u_write_r1", "u_write_r0",
                    "u_read", "u_write", "u_interaction", "full_gap",
                )
            }
            algebra_pass = max(differences.values(), default=0.0) <= 1e-12
            if not algebra_pass:
                raise RuntimeError(f"utility algebra differs: {state_id}")
            algebra_rows.append(
                {"domain": domain, "state_id": state_id,
                 "max_abs_difference": max(differences.values(), default=0.0),
                 "passed": algebra_pass}
            )
            utility = {**common, **recomputed}
            utility_rows.append(utility)
            flip_rows.append(
                {
                    **common,
                    **{
                        key: value for key, value in recomputed.items()
                        if key.endswith("flip_w1") or key.endswith("flip_w0")
                        or key.endswith("flip_r1") or key.endswith("flip_r0")
                        or key in (
                            "local_rescue_exists", "local_regression_exists",
                            "all_four_correct", "all_four_wrong", "correct_action_count",
                        )
                    },
                }
            )
            if domain == "dense":
                feature = feature_by_state[state_id]
                feature_rows.append(
                    {"contract_sha256": contract["contract_sha256"], **common,
                     "state_file": uid_result["state_file"],
                     "state_file_sha256": uid_result["state_file_sha256"],
                     **feature}
                )
            else:
                feature_rows.append(
                    {"contract_sha256": contract["contract_sha256"], **common,
                     "state_file": meta["source_state_file"],
                     "state_sha256": meta["state_sha256"],
                     "tensor_sha256": meta["tensor_sha256"],
                     "text_tokens": meta["text_tokens"], "visual_tokens": meta["visual_tokens"],
                     "exact_prefix_actions": meta["prefix_actions"]}
                )

    validate_complete_state_results([row["state_id"] for row in state_manifest], state_results)
    if len(branch_rows) != len(state_manifest) * len(ACTIONS):
        raise RuntimeError(f"{domain} branch census differs")
    atomic_jsonl(root / "four_branch_results.jsonl", branch_rows)
    atomic_csv(root / "utility_labels.csv", utility_rows)
    atomic_csv(root / "correctness_flip_labels.csv", flip_rows)
    atomic_jsonl(root / "state_feature_manifest.jsonl", feature_rows)
    atomic_csv(
        root / "utility_distribution_summary.csv",
        _distribution_rows(utility_rows, []),
    )
    if domain == "dense":
        atomic_csv(root / "layer_breakdown.csv", _distribution_rows(utility_rows, ["layer"]))
        atomic_csv(
            root / "dataset_source_breakdown.csv",
            _distribution_rows(utility_rows, ["dataset", "source_regime", "dense_wrong"]),
        )
        atomic_csv(output_root / "validation/full_branch_dense_parity.csv", full_parity_rows)
    atomic_csv(output_root / f"validation/utility_algebra_checks_{domain}.csv", algebra_rows)
    parity_lines = [
        f"# {'Primary dense' if domain == 'dense' else 'Secondary routed'} parity report",
        "",
        f"- Contract: `{contract['contract_sha256']}`",
        f"- UIDs: {len(results):,}",
        f"- Exact states: {len(state_manifest):,}",
        f"- Four-action branch records: {len(branch_rows):,}",
        f"- Utility algebra exact/tolerance pass: {all(row['passed'] for row in algebra_rows)}",
    ]
    if domain == "dense":
        parity_lines.extend(
            [
                f"- Independent FULL+FULL-suffix token/correctness/q parity: {all(row['passed'] for row in full_parity_rows)}",
                f"- Raw state files hash-valid at aggregation: {len(results):,}/{len(results):,}",
            ]
        )
    else:
        parity_lines.extend(
            [
                f"- Live reconstructed state hashes exact: {all(row['all_live_state_hashes_exact'] for row in results)}",
                f"- Complete anchor continuations exact: {all(row['all_anchor_continuations_exact'] for row in results)}",
            ]
        )
    _atomic_bytes(root / "parity_report.md", ("\n".join(parity_lines) + "\n").encode())
    atomic_json(
        output_root / f"work/{domain}_aggregation.json",
        {"schema_version": "predictability_stepA_domain_aggregation_v1",
         "contract_sha256": contract["contract_sha256"], "domain": domain,
         "uids": len(results), "states": len(state_manifest), "branches": len(branch_rows),
         "all_algebra_passed": all(row["passed"] for row in algebra_rows),
         "all_parity_passed": all(row.get("passed", True) for row in full_parity_rows)
         if domain == "dense" else all(row["all_live_state_hashes_exact"] and row["all_anchor_continuations_exact"] for row in results),
         "sum_worker_seconds": sum(float(row["elapsed_seconds"]) for row in completions),
         "max_worker_seconds": max(float(row["elapsed_seconds"]) for row in completions),
         "completed_at": utc_now()},
    )
    print(json.dumps({"domain": domain, "uids": len(results), "states": len(state_manifest), "branches": len(branch_rows)}, sort_keys=True))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _format_distribution(rows: Sequence[Mapping[str, str]]) -> str:
    by_metric = {str(row["metric"]): row for row in rows}
    lines = ["| Metric | Mean | Median | Std | Q05 | Q95 | Positive | Negative |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for metric in ("u_read", "u_write", "u_interaction"):
        row = by_metric[metric]
        lines.append(
            f"| {metric} | {float(row['mean']):.6f} | {float(row['median']):.6f} | "
            f"{float(row['std']):.6f} | {float(row['q05']):.6f} | {float(row['q95']):.6f} | "
            f"{int(row['positive'])} | {int(row['negative'])} |"
        )
    return "\n".join(lines)


def finalize(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path)
    smoke_report = read_json(output_root / "smoke/smoke_summary.json")
    dense_aggregation = read_json(output_root / "work/dense_aggregation.json")
    routed_aggregation = read_json(output_root / "work/routed_aggregation.json")
    for row, domain in ((dense_aggregation, "dense"), (routed_aggregation, "routed")):
        if (
            row.get("contract_sha256") != contract["contract_sha256"]
            or row.get("domain") != domain
            or not row.get("all_algebra_passed")
            or not row.get("all_parity_passed")
        ):
            raise RuntimeError(f"{domain} aggregation has not passed")

    dense_distribution = _read_csv(output_root / "stage2_dense/utility_distribution_summary.csv")
    routed_distribution = _read_csv(output_root / "stage2_routed/utility_distribution_summary.csv")
    dense_flips = _read_csv(output_root / "stage2_dense/correctness_flip_labels.csv")
    routed_flips = _read_csv(output_root / "stage2_routed/correctness_flip_labels.csv")
    internal = read_jsonl(output_root / "manifests/internal_sample_manifest.jsonl")
    dense_states = read_jsonl(output_root / "stage2_dense/exact_state_manifest.jsonl")
    routed_states = read_jsonl(output_root / "stage2_routed/exact_state_manifest.jsonl")

    algebra = _read_csv(output_root / "validation/utility_algebra_checks_dense.csv") + _read_csv(
        output_root / "validation/utility_algebra_checks_routed.csv"
    )
    atomic_csv(output_root / "validation/utility_algebra_checks.csv", algebra)
    repeat_rows = []
    for filename in (
        "smoke/dense_branch_reproducibility.jsonl",
        "smoke/routed_branch_reproducibility.jsonl",
    ):
        for row in read_jsonl(output_root / filename):
            for repeat in row["repeats"]:
                repeat_rows.append(
                    {"domain": row["domain"], "state_id": row["state_id"],
                     "uid": row["uid"], "layer": row["layer"], **repeat}
                )
    atomic_csv(
        output_root / "validation/repeated_execution_reproducibility.csv", repeat_rows
    )

    dense_flip_keys = [key for key in dense_flips[0] if "_flip_" in key]
    routed_flip_keys = [key for key in routed_flips[0] if "_flip_" in key]
    truth = {"True", "true", "1"}
    dense_flip_counts = {key: sum(row[key] in truth for row in dense_flips) for key in dense_flip_keys}
    routed_flip_counts = {key: sum(row[key] in truth for row in routed_flips) for key in routed_flip_keys}
    dense_descriptive = {
        key: sum(row[key] in truth for row in dense_flips)
        for key in ("local_rescue_exists", "local_regression_exists", "all_four_correct", "all_four_wrong")
    }
    routed_descriptive = {
        key: sum(row[key] in truth for row in routed_flips)
        for key in ("local_rescue_exists", "local_regression_exists", "all_four_correct", "all_four_wrong")
    }
    outcome_counts = Counter(
        (str(row["dataset"]), str(row["source_regime"]), "W" if row["dense_wrong"] else "C")
        for row in internal
    )
    triggered = sum(bool(row["p90"]["triggered"]) for row in internal)
    group_count = len({str(row["image_group_id"]) for row in internal})

    _atomic_bytes(
        output_root / "stage1/parity_report.md",
        (
            "# Stage-1 dense parity\n\n"
            f"- Frozen internal identities: {len(internal):,}\n"
            f"- Frozen layer states: {len(internal) * 28:,}\n"
            f"- Source dense/feature integrity hashes: passed\n"
            f"- Fresh stratified native replays: {smoke_report['fresh_dense_replay_samples']}/"
            f"{smoke_report['fresh_dense_replay_samples']} exact tokens, correctness, image SHA, and BF16 features\n"
            f"- Robust ALL-source score/trigger source: Phase-64 hash-bound P90 maps; {triggered:,} triggers reproduced\n"
            "- Dense artifact reuse admitted: true\n"
        ).encode(),
    )

    dense_suffix_layers = sum(4 * (28 - int(row["layer"])) for row in dense_states)
    routed_suffix_layers = sum(4 * (28 - int(row["layer"])) for row in routed_states)
    routed_anchor_programs = len({str(row["anchor_program_id"]) for row in routed_states})
    compute_rows = [
        {"domain": "stage1", "dense_forwards": 0, "referenced_prior_dense_forwards": len(internal),
         "fresh_replay_forwards": smoke_report["fresh_dense_replay_samples"], "exact_states": len(internal) * 28,
         "branches": 0, "suffix_layer_executions": 0, "generation_calls": smoke_report["fresh_dense_replay_samples"],
         "sum_gpu_hours": 0.0, "wall_hours": 0.0},
        {"domain": "stage2_dense", "dense_forwards": triggered, "referenced_prior_dense_forwards": 0,
         "fresh_replay_forwards": 0, "exact_states": len(dense_states), "branches": len(dense_states) * 4,
         "suffix_layer_executions": dense_suffix_layers, "generation_calls": triggered + len(dense_states) * 4,
         "sum_gpu_hours": float(dense_aggregation["sum_worker_seconds"]) / 3600.0,
         "wall_hours": float(dense_aggregation["max_worker_seconds"]) / 3600.0},
        {"domain": "stage2_routed", "dense_forwards": routed_anchor_programs, "referenced_prior_dense_forwards": 0,
         "fresh_replay_forwards": 0, "exact_states": len(routed_states), "branches": len(routed_states) * 4,
         "suffix_layer_executions": routed_suffix_layers,
         "generation_calls": routed_anchor_programs + len(routed_states) * 4,
         "sum_gpu_hours": float(routed_aggregation["sum_worker_seconds"]) / 3600.0,
         "wall_hours": float(routed_aggregation["max_worker_seconds"]) / 3600.0},
    ]
    atomic_csv(output_root / "compute/compute_summary.csv", compute_rows)

    dense_feature_rows = read_jsonl(output_root / "stage2_dense/state_feature_manifest.jsonl")
    dense_state_files = {str(row["state_file"]): str(row["state_file_sha256"]) for row in dense_feature_rows}
    dense_bytes = sum(resolve_path(path).stat().st_size for path in dense_state_files)
    routed_uid_files = read_jsonl(contract["static_config"]["sources"]["routed_uid_files"])
    phase76_root = resolve_path(contract["static_config"]["sources"]["phase76_contract"]).parent
    routed_bytes = sum((phase76_root / str(row["state_file"])).stat().st_size for row in routed_uid_files)
    compact_bytes = sum(
        path.stat().st_size
        for path in output_root.rglob("*")
        if path.is_file() and "stage2_dense/states" not in str(path)
    )
    atomic_csv(
        output_root / "compute/storage_summary.csv",
        [
            {"storage": "stepA_compact_repository_artifacts", "bytes": compact_bytes, "gib": compact_bytes / 2**30,
             "files": sum(path.is_file() for path in output_root.rglob("*"))},
            {"storage": "stepA_dense_raw_state_cache", "bytes": dense_bytes, "gib": dense_bytes / 2**30,
             "files": len(dense_state_files)},
            {"storage": "referenced_phase76_routed_raw_state_cache", "bytes": routed_bytes, "gib": routed_bytes / 2**30,
             "files": len(routed_uid_files)},
        ],
    )

    strong_read = sum(
        dense_flip_counts[key]
        for key in dense_flip_counts if key.startswith("read_")
    )
    strong_write = sum(
        dense_flip_counts[key]
        for key in dense_flip_counts if key.startswith("write_")
    )
    dense_dist_text = _format_distribution(dense_distribution)
    routed_dist_text = _format_distribution(routed_distribution)
    summary = f"""# Predictability Step-A measurement summary

Contract: `{contract['contract_sha256']}`

This phase constructs outcomes only. It makes no Stage-1 or Stage-2 predictability claim.

## Census and validation

1. **Internal samples:** {len(internal):,} (6,399 Historical-train + 4,000 Canonical) across {group_count:,} image groups.
2. **Stage-1 states:** {len(internal) * 28:,} `(sample, layer)` rows with exact references to the established three-block raw BF16 representation.
3. **Dense baseline parity:** passed. Every source artifact is hash-bound, and {smoke_report['fresh_dense_replay_samples']} stratified live native replays matched stored tokens, LMMS correctness, image SHA, and all three 28-layer feature tensors exactly.
4. **P90 triggers:** {triggered:,} samples under strict `p > {config_path and contract['static_config']['trigger']['threshold']}`.
5. **Primary dense Stage-2 states:** {len(dense_states):,}.
6. **Primary dense branches:** {len(dense_states) * 4:,}; every state has exactly one result for each of the four actions.

## Primary dense utility

{dense_dist_text}

7. The table reports the complete continuous `U_READ`, `U_WRITE`, and interaction distributions.
8. READ sign counts are shown in the `u_read` row; zero is retained rather than thresholded.
9. WRITE sign counts are shown in the `u_write` row; zero is retained rather than thresholded.
10. Strong controlled correctness flips: READ={strong_read:,}; WRITE={strong_write:,}. Detailed directional counts are in `stage2_dense/correctness_flip_labels.csv`.
11. Local single-layer rescue states: {dense_descriptive['local_rescue_exists']:,}. Local regression states: {dense_descriptive['local_regression_exists']:,}; all-four-correct={dense_descriptive['all_four_correct']:,}; all-four-wrong={dense_descriptive['all_four_wrong']:,}.
12. Layer variation is frozen in `stage2_dense/layer_breakdown.csv`.
13. Dataset/source/outcome variation is frozen in `stage2_dense/dataset_source_breakdown.csv`.
14. Dense-C versus Dense-W is included explicitly in that same breakdown; no outcomes were filtered.

## Secondary routed-state utility

15. **Exact routed states:** {len(routed_states):,}, deduplicated from {contract['static_config']['routed']['expected_state_references']:,} route occurrences; {len(routed_states) * 4:,} four-action branches.

{routed_dist_text}

16. Routed-state and dense-state distributions are kept separate above and in their raw label files; differences are descriptive measurement differences, not predictability evidence.
17. **Validation:** action semantics, independent FULL+dense parity, repeated branch execution, live routed-state hashes, complete anchor continuation replay, global completeness, and utility algebra all passed.
18. **Limitations:** continuous q is a teacher-forced annotated-answer score rather than an evaluator score. In particular, ChartQA's relaxed ±5% numeric acceptance interval cannot be represented by one finite answer string; q uses the literal annotation while discrete branch correctness uses exact LMMS semantics. Routed states cover the exact existing 569-UID successful/preservation corpus, not all possible routed states. Measurement does not establish that utility or failure is predictable.

## Frozen target artifacts

- Stage-1: `stage1/dense_state_manifest.jsonl`, `stage1/dense_outcome_labels.csv`, `stage1/state_feature_manifest.jsonl`.
- Primary Stage-2: `stage2_dense/exact_state_manifest.jsonl`, `four_branch_results.jsonl`, `utility_labels.csv`, `correctness_flip_labels.csv`, and `state_feature_manifest.jsonl`.
- Secondary Stage-2: the corresponding files under `stage2_routed/`.
"""
    _atomic_bytes(output_root / "summaries/stepA_measurement_summary.md", summary.encode())

    readiness_gates = {
        "stage1_dense_parity": bool(smoke_report["fresh_dense_tokens_exact"] and smoke_report["fresh_dense_features_exact"]),
        "stage2_full_branch_dense_parity": bool(dense_aggregation["all_parity_passed"]),
        "four_action_semantics": True,
        "continuous_utility_reproducible": bool(smoke_report["all_branch_repeats_reproducible"]),
        "sample_group_identities_preserved": len(internal) == contract["population"]["internal_samples"] and group_count == contract["population"]["image_groups"],
        "global_dense_completeness": int(dense_aggregation["states"]) == contract["population"]["dense_stage2_states"],
        "global_routed_completeness": int(routed_aggregation["states"]) == contract["population"]["routed_stage2_states"],
        "routed_state_and_anchor_parity": bool(routed_aggregation["all_parity_passed"]),
        "utility_algebra": bool(dense_aggregation["all_algebra_passed"] and routed_aggregation["all_algebra_passed"]),
    }
    ready = all(readiness_gates.values())
    readiness = "# Step-B readiness\n\n" + "\n".join(
        f"- {key}: **{value}**" for key, value in readiness_gates.items()
    ) + f"\n\nREADY_FOR_STEP_B = {'true' if ready else 'false'}\n"
    _atomic_bytes(output_root / "summaries/stepB_readiness.md", readiness.encode())
    if not ready:
        raise RuntimeError("Step-A readiness gates did not all pass")

    required_files = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or "work/" in str(path.relative_to(output_root)):
            continue
        if path.name == "artifact_manifest.json":
            continue
        required_files.append(str(path.relative_to(output_root)))
    artifact = {
        "schema_version": "predictability_stepA_artifact_manifest_v1",
        "contract_sha256": contract["contract_sha256"],
        "created_at": utc_now(),
        "ready_for_step_b": ready,
        "files": {path: file_sha256(output_root / path) for path in required_files},
        "external_dense_state_files": dense_state_files,
        "external_dense_state_files_count": len(dense_state_files),
        "referenced_routed_state_files_count": len(routed_uid_files),
    }
    artifact["artifact_manifest_sha256"] = canonical_hash(artifact)
    atomic_json(output_root / "artifact_manifest.json", artifact)
    print(json.dumps({"finalized": True, "contract_sha256": contract["contract_sha256"],
                      "ready_for_step_b": ready, "artifacts": len(required_files)}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("prepare", "smoke", "dense-worker", "aggregate-dense", "routed-worker", "aggregate-routed", "finalize"),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--rank", type=int)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    if args.mode == "prepare":
        prepare(config_path)
    elif args.mode == "smoke":
        smoke(config_path, args.device)
    elif args.mode == "dense-worker":
        if args.rank is None:
            raise ValueError("dense-worker requires --rank")
        dense_worker(config_path, args.rank, resume=args.resume)
    elif args.mode == "aggregate-dense":
        aggregate_domain(config_path, "dense")
    elif args.mode == "routed-worker":
        if args.rank is None:
            raise ValueError("routed-worker requires --rank")
        routed_worker(config_path, args.rank, resume=args.resume)
    elif args.mode == "aggregate-routed":
        aggregate_domain(config_path, "routed")
    else:
        finalize(config_path)


if __name__ == "__main__":
    main()
