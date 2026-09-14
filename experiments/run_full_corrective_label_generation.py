#!/usr/bin/env python3
"""Generate the frozen full Phase-56 corrective-label corpus and stop before training."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from binary_policy.executor import (  # noqa: E402
    capture_four_action_route,
    capture_four_action_suffix_from_full_baseline,
    capture_full_baseline,
)
from binary_policy.executor.inputs import build_binary_inputs  # noqa: E402
from dense_failure_stage1.runtime import (  # noqa: E402
    build_dense_inputs,
    configure_dense_determinism,
)
from dense_failure_stage2.corrective_search import (  # noqa: E402
    ACTIONS,
    all_single_routes,
    run_sequential_mcts,
    trigger_depth_bin,
    validate_complete_results,
)
from dense_failure_stage2.full_label_generation import (  # noqa: E402
    FIXABILITY_CLASSES,
    assign_workers,
    build_corpus_rows,
    cap_pilot_record,
    choose_preferred_route,
    file_sha256,
    validate_population_completion,
    verify_artifact_manifest,
)
from experiments.run_trigger_conditioned_corrective_search_pilot import (  # noqa: E402
    _atomic_bytes,
    _generate_output,
    _load_model,
    _pool_route_states,
    _runtime_metadata,
    _safe_uid,
    atomic_csv,
    atomic_json,
    atomic_jsonl,
    atomic_torch,
    canonical_hash,
    command_output,
    read_json,
    read_jsonl,
    resolve_path,
    utc_now,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/full_corrective_label_generation_v1.json"
DATASETS = ("gqa", "chartqa", "textvqa")
DEPTH_BINS = ("L0", "L1-8", "L9-18", "L19-27")
PHASE55_ROOT = PROJECT_ROOT / "analysis/dense_failure_stage2/corrective_search_pilot"
BOUND_CODE_PATHS = (
    "configs/full_corrective_label_generation_v1.json",
    "dense_failure_stage2/full_label_generation.py",
    "dense_failure_stage2/corrective_search.py",
    "dense_failure_stage1/lmms_scoring.py",
    "dense_failure_stage1/runtime.py",
    "tools/research_analysis/dense_failure_stage1.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/model.py",
    "experiments/run_trigger_conditioned_corrective_search_pilot.py",
    "experiments/run_full_corrective_label_generation.py",
)


def load_static(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "full_corrective_label_generation_config_v1":
        raise ValueError("unsupported full corrective-label config")
    if int(config["world_size"]) != 4 or tuple(config["mcts"]["actions"]) != ACTIONS:
        raise ValueError("world size/action order differs from the authorized plan")
    if int(config["mcts"]["maximum_iterations"]) != 200:
        raise ValueError("MCTS cap must remain exactly 200")
    if config["mcts"]["rollout_cardinalities"] != [2, 3, 4]:
        raise ValueError("rollout semantics differ from the provenance-matched pilot")
    if int(config["mcts"]["extra_iterations_after_first_success"]) != 25:
        raise ValueError("post-success MCTS budget differs from the pilot")
    if int(config["mcts"]["retained_successful_routes"]) != 8:
        raise ValueError("successful-route retention cap must remain 8")
    if int(config["execution"]["pilot_import_cap"]) != 200:
        raise ValueError("pilot import cap must match the full search cap")
    return config


def _sample(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: candidate[key]
        for key in (
            "uid",
            "sample_id",
            "dataset",
            "prompt",
            "question",
            "answer",
            "all_answer_norms",
            "local_image_path",
            "image_content_sha256",
            "image_group_id",
            "max_new_tokens",
        )
    }


def _join_populations(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = config["sources"]
    candidates = {row["uid"]: row for row in read_jsonl(resolve_path(sources["candidate_manifest"]))}
    dense = {row["uid"]: row for row in read_jsonl(resolve_path(sources["dense_outputs"]))}

    def join(path_key: str, *, wrong: bool) -> list[dict[str, Any]]:
        triggers = read_jsonl(resolve_path(sources[path_key]))
        expected = 1881 if wrong else 39
        if len(triggers) != expected or len({str(row["uid"]) for row in triggers}) != expected:
            raise RuntimeError(f"frozen source population differs from {expected}: {path_key}")
        output = []
        for trigger in triggers:
            uid = str(trigger["uid"])
            if uid not in candidates or uid not in dense:
                raise RuntimeError(f"trigger UID is absent from current dense population: {uid}")
            candidate, dense_row = candidates[uid], dense[uid]
            current_wrong = bool(dense_row.get("current_dense_wrong"))
            if (
                trigger.get("split") != "train"
                or current_wrong != wrong
                or bool(trigger.get("dense_wrong")) != wrong
                or str(trigger["dataset"]) != str(candidate["dataset"])
                or str(trigger["group_id"]) != str(candidate["image_group_id"])
                or str(dense_row["image_content_sha256"]) != str(candidate["image_content_sha256"])
            ):
                raise RuntimeError(f"Phase-54/current-dense join mismatch: {uid}")
            layer = int(trigger["first_trigger_layer"])
            output.append(
                {
                    **trigger,
                    "task_type": (
                        "triggered_wrong" if wrong else "triggered_correct_preservation"
                    ),
                    "trigger_depth_bin": trigger_depth_bin(layer),
                    "sample": _sample(candidate),
                    "dense_output": dense_row,
                    "estimated_terminal_evaluations": (
                        3 * (28 - layer) + int(config["mcts"]["maximum_iterations"])
                        if wrong
                        else 2
                    ),
                }
            )
        return sorted(output, key=lambda row: str(row["uid"]))

    wrong_rows = join("triggered_wrong_manifest", wrong=True)
    correct_rows = join("triggered_correct_manifest", wrong=False)
    if {row["uid"] for row in wrong_rows} & {row["uid"] for row in correct_rows}:
        raise RuntimeError("triggered W/C populations overlap")
    return wrong_rows, correct_rows


def _phase55_samples() -> list[dict[str, Any]]:
    paths = sorted((PHASE55_ROOT / "work/pilot").glob("rank*/samples/*.json"))
    rows = [read_json(path) for path in paths]
    if len(rows) != 120 or len({str(row["uid"]) for row in rows}) != 120:
        raise RuntimeError("Phase-55 raw pilot records are incomplete")
    return sorted(rows, key=lambda row: str(row["uid"]))


def _feature_schema(config: Mapping[str, Any]) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "schema_version": "full_stage2_routed_training_state_schema_v1",
        "state_timing": "pre_layer_entering_chosen_action",
        "trajectory_semantics": "actual_replay_valid_four_action_trajectory",
        "layers": list(range(28)),
        "hidden_size": int(config["features"]["hidden_size"]),
        "tensor_dtype": config["features"]["dtype"],
        "features": {
            "text_final": "last literal user instruction/query token in compact text stream",
            "text_mean": "mean of literal user instruction/question token rows",
            "visual_mean": "mean of valid visual-token rows",
        },
        "route_sources": ["preservation_full", "single", "mcts"],
        "label": "chosen action at the same layer on an exact-replay LMMS-correct route",
        "forbidden_inputs": ["dataset_id", "ground_truth", "W_to_C", "historical_bucket"],
    }
    schema["feature_schema_sha256"] = canonical_hash(
        schema, excluded=("feature_schema_sha256",)
    )
    return schema


def _pilot_import_selection(config: Mapping[str, Any], pilot_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cap = int(config["execution"]["pilot_import_cap"])
    retain = int(config["mcts"]["retained_successful_routes"])
    selections = []
    counts = Counter()
    eligible_mcts_routes = 0
    for sample in pilot_rows:
        capped = cap_pilot_record(sample, cap=cap, retain_successes=retain)
        counts[capped["outcome_class"]] += 1
        eligible_mcts_routes += sum(
            str(route.get("search_stage")) == "mcts"
            for route in capped["retained_successful_routes"]
        )
        feature_file = sample.get("feature_file")
        feature_hash = sample.get("feature_file_sha256")
        if capped["eligible_route_keys"]:
            if not feature_file:
                raise RuntimeError(f"cap-eligible pilot record lacks a feature shard: {sample['uid']}")
            path = PHASE55_ROOT / str(feature_file)
            if not path.is_file() or file_sha256(path) != feature_hash:
                raise RuntimeError(f"pilot feature shard is missing or changed: {sample['uid']}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            available = Counter(str(row["route_key"]) for row in payload["records"])
            horizon = 28 - int(sample["trigger_layer"])
            for key in capped["eligible_route_keys"]:
                if available[key] != horizon:
                    raise RuntimeError(
                        f"pilot route/state row mismatch for {sample['uid']} {key}: "
                        f"{available[key]} != {horizon}"
                    )
        selections.append(
            {
                "uid": sample["uid"],
                "dataset": sample["dataset"],
                "trigger_layer": sample["trigger_layer"],
                "trigger_depth_bin": sample["trigger_depth_bin"],
                "cap": cap,
                "outcome_class": capped["outcome_class"],
                "first_success_iteration": capped["first_success_iteration"],
                "mcts_iterations": capped["mcts_iterations"],
                "mcts_unique_terminal_routes": capped["mcts_unique_terminal_routes"],
                "eligible_route_keys": capped["eligible_route_keys"],
                "excluded_post_cap_route_keys": capped["excluded_post_cap_route_keys"],
                "source_feature_file": feature_file,
                "source_feature_file_sha256": feature_hash,
                "source_contract_sha256": sample["contract_sha256"],
            }
        )
    expected = Counter(
        {"SINGLE_FIXABLE": 35, "MCTS_ONLY_FIXABLE": 21, "UNRESOLVED": 64}
    )
    if counts != expected or eligible_mcts_routes != 42:
        raise RuntimeError(
            f"cap-200 pilot transcript audit changed: counts={counts}, "
            f"retained_mcts_routes={eligible_mcts_routes}"
        )
    return sorted(selections, key=lambda row: str(row["uid"]))


def _select_smoke(rows: Sequence[Mapping[str, Any]], *, count: int, seed: int, task: str) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            sha256(f"phase56-smoke:{seed}:{task}:{row['uid']}".encode()).hexdigest(),
            str(row["uid"]),
        ),
    )
    if len(ordered) < count:
        raise RuntimeError(f"insufficient smoke candidates for {task}")
    return [dict(row) for row in ordered[:count]]


def prepare(config_path: Path) -> None:
    config = load_static(config_path)
    output_root = resolve_path(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")

    phase55_manifest = read_json(resolve_path(config["sources"]["phase55_artifact_manifest"]))
    verify_artifact_manifest(PHASE55_ROOT, phase55_manifest)
    wrong_rows, correct_rows = _join_populations(config)
    pilot_rows = _phase55_samples()
    pilot_uids = {str(row["uid"]) for row in pilot_rows}
    wrong_uids = {str(row["uid"]) for row in wrong_rows}
    if len(pilot_uids) != 120 or not pilot_uids <= wrong_uids:
        raise RuntimeError("Phase-55 pilot is not an exact subset of the frozen 1,881 W rows")
    remaining_wrong = [row for row in wrong_rows if str(row["uid"]) not in pilot_uids]
    if len(remaining_wrong) != 1761:
        raise RuntimeError("fresh triggered-W workload must contain exactly 1,761 rows")
    full_work = assign_workers(
        [*remaining_wrong, *correct_rows], world_size=int(config["world_size"])
    )
    import_selection = _pilot_import_selection(config, pilot_rows)

    smoke_wrong = _select_smoke(
        remaining_wrong,
        count=int(config["smoke"]["wrong_records"]),
        seed=int(config["seed"]),
        task="wrong",
    )
    smoke_correct = _select_smoke(
        correct_rows,
        count=int(config["smoke"]["preservation_records"]),
        seed=int(config["seed"]),
        task="preservation",
    )
    smoke = []
    for rank, (wrong, correct) in enumerate(zip(smoke_wrong, smoke_correct, strict=True)):
        smoke.extend(({**wrong, "worker_rank": rank}, {**correct, "worker_rank": rank}))
    smoke.sort(key=lambda row: str(row["uid"]))

    schema = _feature_schema(config)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output_root / "manifests/full_triggered_wrong_manifest.jsonl", wrong_rows)
    atomic_jsonl(output_root / "work/manifests/remaining_wrong.jsonl", remaining_wrong)
    atomic_jsonl(output_root / "work/manifests/full_work_manifest.jsonl", full_work)
    atomic_jsonl(output_root / "work/manifests/smoke_manifest.jsonl", smoke)
    atomic_jsonl(output_root / "work/manifests/pilot_import_selection.jsonl", import_selection)
    atomic_json(output_root / "states/feature_schema.json", schema)

    source_hashes = {
        name: file_sha256(resolve_path(path)) for name, path in config["sources"].items()
    }
    bound_hashes = {path: file_sha256(resolve_path(path)) for path in BOUND_CODE_PATHS}
    phase55_contract = read_json(resolve_path(config["sources"]["phase55_contract"]))
    snapshot = resolve_path(config["model"]["snapshot_path"])
    model_files = sorted(path for path in snapshot.iterdir() if path.is_file())
    model_hashes = {path.name: file_sha256(path) for path in model_files}
    if model_hashes != phase55_contract["model_snapshot_sha256"]:
        raise RuntimeError("live model snapshot differs from the replay-verified Phase-55 model")

    internal = {
        relative: file_sha256(output_root / relative)
        for relative in (
            "manifests/full_triggered_wrong_manifest.jsonl",
            "work/manifests/remaining_wrong.jsonl",
            "work/manifests/full_work_manifest.jsonl",
            "work/manifests/smoke_manifest.jsonl",
            "work/manifests/pilot_import_selection.jsonl",
            "states/feature_schema.json",
        )
    }
    contract: dict[str, Any] = {
        "schema_version": "full_corrective_label_generation_contract_v1",
        "static_config": config,
        "git": {
            "commit": command_output(("git", "rev-parse", "HEAD")),
            "branch": command_output(("git", "branch", "--show-current")),
            "worktree_status_at_freeze": command_output(("git", "status", "--short")),
        },
        "runtime": _runtime_metadata(),
        "source_sha256": source_hashes,
        "bound_code_sha256": bound_hashes,
        "internal_manifest_sha256": internal,
        "model_snapshot_sha256": model_hashes,
        "phase54_contract_sha256": read_json(
            resolve_path(config["sources"]["phase54_contract"])
        )["contract_sha256"],
        "phase55_contract_sha256": phase55_contract["contract_sha256"],
        "phase55_artifact_manifest_sha256": source_hashes["phase55_artifact_manifest"],
        "feature_schema_sha256": schema["feature_schema_sha256"],
        "population": {
            "triggered_wrong": 1881,
            "pilot_imported": 120,
            "fresh_wrong": 1761,
            "triggered_correct_preservation": 39,
        },
        "pilot_cap200_audit": {
            "single_fixable": 35,
            "mcts_only_fixable": 21,
            "unresolved": 64,
            "retained_mcts_routes": 42,
            "excluded_post_cap_rescue_uid": "gqa:gqa_gh_09367372",
            "source_tensor_import": "exact selected rows copied into new contract-bound shards",
        },
        "review_reconciliation": {
            "reviewer_verdict": "stable",
            "selected": "cap-200 transcript-equivalent import of all 120 pilot rows",
            "alternative_rejected": "rerun all 120 because every canonical cap-200 route has a replay-valid shard",
            "guard": "no route or tensor first discovered after iteration 200 may enter the corpus",
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_protocol.json", contract)
    protocol = f"""# Full corrective-label generation protocol

- Contract SHA-256: `{contract['contract_sha256']}`
- Frozen populations: 1,881 Stage-1-triggered Dense-W and 39 Stage-1-triggered Dense-C train rows.
- Import: 120 Phase-55 pilot W rows are transcript-equivalently reinterpreted at cap 200; the one rescue first found at iteration 294 is excluded, yielding 35 SINGLE_FIXABLE, 21 MCTS_ONLY_FIXABLE, and 64 UNRESOLVED imported rows.
- Fresh workload: 1,761 W searches and 39 C preservation replays, assigned deterministically across four direct GPUs.
- Dense prefix: exact native all-FULL execution through the state entering the frozen trigger layer. No layer before the trigger may change.
- W search: exhaustive singles from trigger through layer 27; MCTS only when no single succeeds; deterministic UCB1 2/3/4-cardinality rollout, binary current LMMS correctness, cap 200, 25 post-success iterations within the cap, retain at most 8 correct MCTS routes.
- C preservation: no search; retain and replay the known-safe all-FULL suffix.
- Every retained route is exact-token/correctness replay valid. Every saved feature is the actual pre-layer routed state for its route and is hash-bound to this contract/model/code/schema/source.
- Corpus A (`preservation_full`), B (`single`), and C (`mcts`) remain separate. Route multiplicity does not decide future sampling weight.
- This phase ends after search, replay, corpus construction, audit, figures, and artifact verification. It does not train Stage 2 or modify Stage 1.
"""
    _atomic_bytes(output_root / "protocol.md", protocol.encode())
    atomic_json(
        output_root / "preparation_audit.json",
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "triggered_wrong": len(wrong_rows),
            "pilot_imported": len(pilot_rows),
            "fresh_wrong": len(remaining_wrong),
            "triggered_correct_preservation": len(correct_rows),
            "smoke_records": len(smoke),
            "pilot_cap200_counts": dict(Counter(row["outcome_class"] for row in import_selection)),
            "prepared_at": utc_now(),
        },
    )
    print(json.dumps({"passed": True, "contract_sha256": contract["contract_sha256"], "fresh_work": len(full_work)}))


def load_contract(
    config_path: Path, *, verify_model_snapshot: bool = False
) -> tuple[dict[str, Any], Path]:
    config = load_static(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if canonical_hash(contract) != contract.get("contract_sha256") or contract["static_config"] != config:
        raise RuntimeError("frozen contract/config mismatch")
    if command_output(("git", "rev-parse", "HEAD")) != contract["git"]["commit"]:
        raise RuntimeError("git commit differs from the frozen contract")
    if command_output(("git", "branch", "--show-current")) != contract["git"]["branch"]:
        raise RuntimeError("git branch differs from the frozen contract")
    if command_output(("git", "status", "--short")) != contract["git"]["worktree_status_at_freeze"]:
        raise RuntimeError("worktree status differs from the frozen contract")
    if _runtime_metadata() != contract["runtime"]:
        raise RuntimeError("runtime/environment differs from the frozen contract")
    for name, expected in contract["source_sha256"].items():
        if file_sha256(resolve_path(config["sources"][name])) != expected:
            raise RuntimeError(f"source hash mismatch: {name}")
    for relative, expected in contract["bound_code_sha256"].items():
        if file_sha256(resolve_path(relative)) != expected:
            raise RuntimeError(f"bound-code hash mismatch: {relative}")
    for relative, expected in contract["internal_manifest_sha256"].items():
        if file_sha256(output_root / relative) != expected:
            raise RuntimeError(f"internal manifest hash mismatch: {relative}")
    schema = read_json(output_root / "states/feature_schema.json")
    if (
        schema.get("feature_schema_sha256") != contract["feature_schema_sha256"]
        or canonical_hash(schema, excluded=("feature_schema_sha256",))
        != contract["feature_schema_sha256"]
    ):
        raise RuntimeError("feature schema hash mismatch")
    if verify_model_snapshot:
        snapshot = resolve_path(config["model"]["snapshot_path"])
        if sorted(path.name for path in snapshot.iterdir() if path.is_file()) != sorted(
            contract["model_snapshot_sha256"]
        ):
            raise RuntimeError("model snapshot inventory mismatch")
        for name, expected in contract["model_snapshot_sha256"].items():
            if file_sha256(snapshot / name) != expected:
                raise RuntimeError(f"model snapshot hash mismatch: {name}")
    return contract, output_root


def _route_id(uid: str, route_key: str) -> str:
    return sha256(f"{uid}\0{route_key}".encode()).hexdigest()[:24]


def _route_counts(actions: Sequence[str]) -> dict[str, Any]:
    changed = [layer for layer, action in enumerate(actions) if action != "FULL"]
    counts = Counter(actions)
    return {
        "changed_layers": changed,
        "changed_actions": [actions[layer] for layer in changed],
        "non_full_count": len(changed),
        "read_only_count": counts["READ_ONLY"],
        "write_only_count": counts["WRITE_ONLY"],
        "ignore_count": counts["IGNORE"],
        "full_count": counts["FULL"],
        "first_non_full_layer": min(changed) if changed else None,
        "last_non_full_layer": max(changed) if changed else None,
    }


def _route_manifest_row(
    *,
    route: Mapping[str, Any],
    sample: Mapping[str, Any],
    route_source: str,
    feature_file: str,
    feature_hash: str,
    feature_rows: int,
    contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    actions = list(route["actions"])
    uid = str(sample["uid"])
    key = str(route["route_key"])
    counts = _route_counts(actions)
    discovery = route.get("discovery_iteration", route.get("discovery_order"))
    if "trigger_layer" in sample:
        trigger_layer = int(sample["trigger_layer"])
    elif "first_trigger_layer" in sample:
        trigger_layer = int(sample["first_trigger_layer"])
    else:
        raise ValueError("route-manifest sample has no frozen trigger layer")
    return {
        "schema_version": "full_stage2_successful_route_manifest_v1",
        "contract_sha256": contract["contract_sha256"],
        "uid": uid,
        "dataset": sample["dataset"],
        "trigger_layer": trigger_layer,
        "trigger_depth_bin": sample["trigger_depth_bin"],
        "route_id": _route_id(uid, key),
        "route_source": route_source,
        "route_key": key,
        "actions": actions,
        **counts,
        "discovery_iteration": discovery,
        "intervention_layer": counts["changed_layers"][0] if route_source == "single" else None,
        "intervention_action": counts["changed_actions"][0] if route_source == "single" else None,
        "generated_token_ids": route["generated_ids"],
        "generated_answer": route["generated_answer"],
        "lmms_metric": route["lmms_metric"],
        "lmms_score": route["lmms_score"],
        "final_lmms_correct": bool(route["correct"]),
        "feature_file": feature_file,
        "feature_file_sha256": feature_hash,
        "feature_rows": feature_rows,
        "replay_token_parity": True,
        **dict(provenance),
    }


def _state_relative(route_source: str, uid: str, *, smoke: bool = False, rank: int = 0) -> Path:
    key = _safe_uid(uid)
    if smoke:
        return Path(f"work/smoke/rank{rank:02d}/features/{key}.pt")
    directory = {
        "single": "single_route_states",
        "mcts": "mcts_route_states",
        "preservation_full": "preservation_full_states",
    }[route_source]
    return Path(f"states/{directory}/{key}.pt")


def _validate_saved_shard(path: Path, payload: Mapping[str, Any]) -> None:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    for key in (
        "contract_sha256",
        "model_revision",
        "code_commit",
        "feature_schema_sha256",
        "source_manifest_sha256",
        "split_run_id",
        "uid",
        "route_source",
    ):
        if saved.get(key) != payload.get(key):
            raise RuntimeError(f"saved state-shard provenance mismatch: {key}")
    records = saved.get("records", [])
    if any(int(row["tensor_row"]) != index for index, row in enumerate(records)):
        raise RuntimeError("state-shard tensor rows are not contiguous")
    expected = (len(records), 3584)
    if any(saved.get(key).shape != expected for key in ("text_final", "text_mean", "visual_mean")):
        raise RuntimeError("saved state-shard tensor shape mismatch")


def _save_native_route_states(
    *,
    routes: Sequence[Mapping[str, Any]],
    route_source: str,
    manifest_row: Mapping[str, Any],
    baseline: Any,
    wrapped: Any,
    processor: Any,
    base: Any,
    inputs: Mapping[str, Any],
    contract: Mapping[str, Any],
    output_root: Path,
    smoke: bool,
    rank: int,
) -> tuple[str | None, str | None, list[dict[str, Any]], list[dict[str, Any]]]:
    if not routes:
        return None, None, [], []
    start = int(manifest_row["first_trigger_layer"])
    tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    index_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    tensor_row = 0
    replayed: list[tuple[Mapping[str, Any], dict[str, Any], int]] = []
    for route in routes:
        actions = tuple(route["actions"])
        replay = capture_four_action_suffix_from_full_baseline(
            wrapped, baseline, start, actions[start:]
        )
        replay_state = _generate_output(
            processor, wrapped, replay, inputs, manifest_row["sample"]
        )
        parity = (
            replay_state["generated_ids"] == route["generated_ids"]
            and replay_state["generated_answer"] == route["generated_answer"]
            and replay_state["lmms_score"] == route["lmms_score"]
            and replay_state["correct"] == route["correct"]
        )
        if not parity or not replay_state["correct"]:
            raise RuntimeError(f"successful {route_source} route replay failed")
        pooled, metadata = _pool_route_states(replay, processor, base, inputs, start)
        route_id = _route_id(str(manifest_row["uid"]), str(route["route_key"]))
        for key, values in pooled.items():
            tensors[key].append(values)
        for offset, item in enumerate(metadata):
            index_rows.append(
                {
                    "schema_version": "full_stage2_routed_training_state_index_v1",
                    "contract_sha256": contract["contract_sha256"],
                    "uid": manifest_row["uid"],
                    "dataset": manifest_row["dataset"],
                    "trigger_layer": start,
                    "trigger_depth_bin": manifest_row["trigger_depth_bin"],
                    "route_id": route_id,
                    "route_key": route["route_key"],
                    "route_source": route_source,
                    "route_non_full_count": int(route["non_full_count"]),
                    "layer": int(item["layer"]),
                    "chosen_action": item["action"],
                    "tensor_row": tensor_row + offset,
                    "final_lmms_correct": True,
                    "replay_token_parity": True,
                    "state_origin": "fresh_exact_replay",
                }
            )
        replayed.append((route, replay_state, len(metadata)))
        tensor_row += len(metadata)

    packed = {key: torch.cat(values, dim=0) for key, values in tensors.items()}
    expected = (len(index_rows), int(contract["static_config"]["features"]["hidden_size"]))
    if set(packed) != {"text_final", "text_mean", "visual_mean"} or any(
        value.shape != expected for value in packed.values()
    ):
        raise RuntimeError("native routed-state tensor schema mismatch")
    relative = _state_relative(
        route_source, str(manifest_row["uid"]), smoke=smoke, rank=rank
    )
    source_manifest_key = (
        "work/manifests/smoke_manifest.jsonl"
        if smoke
        else "work/manifests/full_work_manifest.jsonl"
    )
    payload = {
        "schema_version": "full_stage2_routed_training_state_shard_v1",
        "contract_sha256": contract["contract_sha256"],
        "model_revision": contract["static_config"]["model"]["revision"],
        "code_commit": contract["git"]["commit"],
        "feature_schema_sha256": contract["feature_schema_sha256"],
        "source_manifest_sha256": contract["internal_manifest_sha256"][source_manifest_key],
        "split_run_id": contract["static_config"]["run_id"],
        "uid": manifest_row["uid"],
        "route_source": route_source,
        "state_origin": "fresh_exact_replay",
        "records": index_rows,
        **packed,
    }
    path = output_root / relative
    atomic_torch(path, payload)
    _validate_saved_shard(path, payload)
    shard_hash = file_sha256(path)
    for route, _replay_state, rows in replayed:
        route_rows.append(
            _route_manifest_row(
                route=route,
                sample=manifest_row,
                route_source=route_source,
                feature_file=str(relative),
                feature_hash=shard_hash,
                feature_rows=rows,
                contract=contract,
                provenance={"route_origin": "fresh_search_or_preservation"},
            )
        )
    for row in index_rows:
        row.update(
            {
                "feature_file": str(relative),
                "feature_file_sha256": shard_hash,
                "feature_schema_sha256": contract["feature_schema_sha256"],
                "source_manifest_sha256": payload["source_manifest_sha256"],
                "split_run_id": payload["split_run_id"],
            }
        )
    return str(relative), shard_hash, index_rows, route_rows


def import_pilot(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    completion = output_root / "work/imported/complete.json"
    if completion.exists():
        raise RuntimeError("pilot import is already complete")
    phase55_manifest = read_json(
        resolve_path(contract["static_config"]["sources"]["phase55_artifact_manifest"])
    )
    verify_artifact_manifest(PHASE55_ROOT, phase55_manifest)
    source_samples = {str(row["uid"]): row for row in _phase55_samples()}
    selection = read_jsonl(output_root / "work/manifests/pilot_import_selection.jsonl")
    output_rows = []
    for selected in selection:
        uid = str(selected["uid"])
        source = source_samples[uid]
        capped = cap_pilot_record(
            source,
            cap=int(contract["static_config"]["mcts"]["maximum_iterations"]),
            retain_successes=int(contract["static_config"]["mcts"]["retained_successful_routes"]),
        )
        if (
            capped["outcome_class"] != selected["outcome_class"]
            or capped["eligible_route_keys"] != selected["eligible_route_keys"]
        ):
            raise RuntimeError(f"pilot import selection changed after freeze: {uid}")

        routes = [dict(route) for route in capped["retained_successful_routes"]]
        route_source = None
        if capped["outcome_class"] == "SINGLE_FIXABLE":
            route_source = "single"
            order = {
                str(route["route_key"]): index
                for index, route in enumerate(source["single_executions"][1:], 1)
            }
            for route in routes:
                route["discovery_order"] = order[str(route["route_key"])]
        elif capped["outcome_class"] == "MCTS_ONLY_FIXABLE":
            route_source = "mcts"

        feature_file = None
        feature_hash = None
        index_rows: list[dict[str, Any]] = []
        route_rows: list[dict[str, Any]] = []
        if routes:
            source_path = PHASE55_ROOT / str(source["feature_file"])
            if file_sha256(source_path) != source["feature_file_sha256"]:
                raise RuntimeError(f"pilot feature shard changed during import: {uid}")
            old = torch.load(source_path, map_location="cpu", weights_only=False)
            old_records = old["records"]
            selected_indices: list[int] = []
            new_records = []
            tensor_row = 0
            for route in routes:
                key = str(route["route_key"])
                matching = [
                    (index, row)
                    for index, row in enumerate(old_records)
                    if str(row["route_key"]) == key
                ]
                matching.sort(key=lambda item: int(item[1]["layer"]))
                if len(matching) != 28 - int(source["trigger_layer"]):
                    raise RuntimeError(f"pilot tensor subset is incomplete: {uid}/{key}")
                route_id = _route_id(uid, key)
                for index, row in matching:
                    selected_indices.append(index)
                    new_records.append(
                        {
                            "schema_version": "full_stage2_routed_training_state_index_v1",
                            "contract_sha256": contract["contract_sha256"],
                            "uid": uid,
                            "dataset": source["dataset"],
                            "trigger_layer": source["trigger_layer"],
                            "trigger_depth_bin": source["trigger_depth_bin"],
                            "route_id": route_id,
                            "route_key": key,
                            "route_source": route_source,
                            "route_non_full_count": int(route["non_full_count"]),
                            "layer": int(row["layer"]),
                            "chosen_action": row["chosen_action"],
                            "tensor_row": tensor_row,
                            "final_lmms_correct": True,
                            "replay_token_parity": True,
                            "state_origin": "phase55_exact_tensor_subset_import",
                            "source_phase55_contract_sha256": source["contract_sha256"],
                            "source_phase55_feature_file_sha256": source["feature_file_sha256"],
                            "source_phase55_tensor_row": int(row["tensor_row"]),
                        }
                    )
                    tensor_row += 1
            index_tensor = torch.tensor(selected_indices, dtype=torch.long)
            packed = {
                key: old[key].index_select(0, index_tensor).clone()
                for key in ("text_final", "text_mean", "visual_mean")
            }
            relative = _state_relative(route_source, uid)
            payload = {
                "schema_version": "full_stage2_routed_training_state_shard_v1",
                "contract_sha256": contract["contract_sha256"],
                "model_revision": contract["static_config"]["model"]["revision"],
                "code_commit": contract["git"]["commit"],
                "feature_schema_sha256": contract["feature_schema_sha256"],
                "source_manifest_sha256": contract["internal_manifest_sha256"][
                    "work/manifests/pilot_import_selection.jsonl"
                ],
                "split_run_id": contract["static_config"]["run_id"],
                "uid": uid,
                "route_source": route_source,
                "state_origin": "phase55_exact_tensor_subset_import",
                "source_phase55_contract_sha256": source["contract_sha256"],
                "source_phase55_feature_file": source["feature_file"],
                "source_phase55_feature_file_sha256": source["feature_file_sha256"],
                "records": new_records,
                **packed,
            }
            path = output_root / relative
            atomic_torch(path, payload)
            _validate_saved_shard(path, payload)
            feature_file, feature_hash = str(relative), file_sha256(path)
            for row in new_records:
                row.update(
                    {
                        "feature_file": feature_file,
                        "feature_file_sha256": feature_hash,
                        "feature_schema_sha256": contract["feature_schema_sha256"],
                        "source_manifest_sha256": payload["source_manifest_sha256"],
                        "split_run_id": payload["split_run_id"],
                    }
                )
            index_rows = new_records
            by_key = Counter(str(row["route_key"]) for row in new_records)
            for route in routes:
                route_rows.append(
                    _route_manifest_row(
                        route=route,
                        sample=source,
                        route_source=route_source,
                        feature_file=feature_file,
                        feature_hash=feature_hash,
                        feature_rows=by_key[str(route["route_key"])],
                        contract=contract,
                        provenance={
                            "route_origin": "phase55_cap200_transcript_import",
                            "source_phase55_contract_sha256": source["contract_sha256"],
                            "source_phase55_feature_file_sha256": source["feature_file_sha256"],
                        },
                    )
                )

        mcts = source.get("mcts_result")
        capped_search = (
            []
            if mcts is None
            else [
                row
                for row in mcts["search_rows"]
                if int(row["iteration"])
                <= int(contract["static_config"]["mcts"]["maximum_iterations"])
            ]
        )
        result = {
            "schema_version": "full_corrective_label_sample_result_v1",
            "contract_sha256": contract["contract_sha256"],
            "uid": uid,
            "dataset": source["dataset"],
            "image_group_id": source["image_group_id"],
            "trigger_layer": source["trigger_layer"],
            "trigger_depth_bin": source["trigger_depth_bin"],
            "task_type": "triggered_wrong",
            "outcome_class": capped["outcome_class"],
            "result_origin": "phase55_cap200_transcript_import",
            "source_phase55_contract_sha256": source["contract_sha256"],
            "single_routes_evaluated": len(source["single_executions"]) - 1,
            "successful_single_routes": len(source["successful_single_routes"]),
            "mcts_iterations": capped["mcts_iterations"],
            "mcts_unique_terminal_routes": capped["mcts_unique_terminal_routes"],
            "mcts_first_success_iteration": capped["first_success_iteration"],
            "physical_terminal_evaluations": (
                len(source["single_executions"]) - 1
                + sum(not bool(row.get("terminal_cache_hit")) for row in capped_search)
            ),
            "retained_successful_routes": route_rows,
            "feature_file": feature_file,
            "feature_file_sha256": feature_hash,
            "feature_index": index_rows,
            "source_elapsed_seconds": source["elapsed_seconds"],
            "fresh_elapsed_seconds": 0.0,
            "passed": True,
        }
        path = output_root / f"work/imported/samples/{_safe_uid(uid)}.json"
        atomic_json(path, result)
        output_rows.append(result)

    validate_complete_results([row["uid"] for row in selection], output_rows)
    counts = Counter(row["outcome_class"] for row in output_rows)
    if counts != Counter({"SINGLE_FIXABLE": 35, "MCTS_ONLY_FIXABLE": 21, "UNRESOLVED": 64}):
        raise RuntimeError(f"imported cap-200 classes changed: {counts}")
    atomic_json(
        completion,
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "records": len(output_rows),
            "outcome_counts": dict(counts),
            "completed_at": utc_now(),
        },
    )
    print(json.dumps({"passed": True, "records": len(output_rows), "outcome_counts": dict(counts)}))


def _worker_root(output_root: Path, mode: str, rank: int) -> Path:
    return output_root / f"work/{mode}/rank{rank:02d}"


def _collect_mode(output_root: Path, mode: str, contract_sha256: str) -> list[dict[str, Any]]:
    rows = []
    for rank in range(4):
        root = _worker_root(output_root, mode, rank)
        complete = read_json(root / "complete.json")
        if not complete.get("passed") or complete.get("contract_sha256") != contract_sha256:
            raise RuntimeError(f"invalid {mode} completion for rank {rank}")
        failures = list(root.glob("failure_*.json"))
        if failures:
            raise RuntimeError(f"{mode} rank {rank} contains failure records")
        rows.extend(read_json(path) for path in sorted((root / "samples").glob("*.json")))
    if len({str(row["uid"]) for row in rows}) != len(rows):
        raise RuntimeError(f"{mode} outputs contain duplicate UIDs")
    return sorted(rows, key=lambda row: str(row["uid"]))


def worker(
    config_path: Path, *, mode: str, rank: int, world_size: int, resume: bool
) -> None:
    worker_started = time.monotonic()
    contract, output_root = load_contract(config_path, verify_model_snapshot=True)
    config = contract["static_config"]
    if mode not in {"smoke", "full"}:
        raise ValueError("worker mode must be smoke or full")
    if world_size != 4 or rank not in range(world_size) or torch.cuda.device_count() != 4:
        raise RuntimeError("full corrective-label workers require four visible GPUs/processes")
    if mode == "full":
        if not read_json(output_root / "work/smoke/completion.json").get("passed"):
            raise RuntimeError("full execution requires a passing smoke")
        if not read_json(output_root / "work/imported/complete.json").get("passed"):
            raise RuntimeError("full execution requires a complete cap-200 pilot import")
    manifest_path = output_root / (
        "work/manifests/smoke_manifest.jsonl"
        if mode == "smoke"
        else "work/manifests/full_work_manifest.jsonl"
    )
    rows = [row for row in read_jsonl(manifest_path) if int(row["worker_rank"]) == rank]
    worker_root = _worker_root(output_root, mode, rank)
    complete_path = worker_root / "complete.json"
    if complete_path.exists():
        raise RuntimeError(f"worker already complete: {complete_path}")
    sample_dir = worker_root / "samples"
    existing = list(sample_dir.glob("*.json")) if sample_dir.exists() else []
    if existing and not resume:
        raise FileExistsError(f"worker outputs exist; use --resume: rank {rank}")
    completed: dict[str, dict[str, Any]] = {}
    for path in existing:
        row = read_json(path)
        if row.get("contract_sha256") != contract["contract_sha256"] or not row.get("passed"):
            raise RuntimeError(f"incompatible resume record: {path}")
        uid = str(row["uid"])
        if uid in completed:
            raise RuntimeError(f"duplicate resume UID: {uid}")
        feature_file = row.get("feature_file")
        if feature_file:
            feature_path = output_root / str(feature_file)
            if not feature_path.is_file() or file_sha256(feature_path) != row["feature_file_sha256"]:
                raise RuntimeError(f"resume feature shard is missing or incompatible: {uid}")
        completed[uid] = row

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    configure_dense_determinism(
        int(config["seed"]) + rank,
        read_json(resolve_path(config["sources"]["dense_config"]))["backend_settings"],
    )
    processor, base, wrapped = _load_model(config, device)
    failures = 0
    for index, manifest_row in enumerate(rows):
        uid = str(manifest_row["uid"])
        if uid in completed:
            continue
        sample_path = sample_dir / f"{_safe_uid(uid)}.json"
        started = time.monotonic()
        try:
            sample = manifest_row["sample"]
            inputs, input_metadata = build_dense_inputs(processor, sample, device)
            if input_metadata["consumed_image_sha256"] != sample["image_content_sha256"]:
                raise RuntimeError("consumed image hash differs from the frozen manifest")
            prepared = build_binary_inputs(wrapped, inputs)
            baseline = capture_full_baseline(
                wrapped,
                inputs,
                prepared_inputs=prepared,
                use_cache=True,
                native_causal=bool(config["execution"]["native_full_row_dispatch"]),
            )
            baseline_state = _generate_output(processor, wrapped, baseline, inputs, sample)
            expected = manifest_row["dense_output"]
            expected_correct = not bool(manifest_row["dense_wrong"])
            baseline_checks = {
                "generated_ids_match_current_dense": baseline_state["generated_ids"]
                == expected["generated_token_ids"],
                "answer_match_current_dense": baseline_state["generated_answer"]
                == expected["generated_answer"],
                "lmms_score_match_current_dense": baseline_state["lmms_score"]
                == float(expected["lmms_eval_per_sample_score"]),
                "correctness_match_current_dense": baseline_state["correct"]
                == bool(expected["current_dense_correct"]),
                "prompt_sha256_match_current_dense": input_metadata["literal_prompt_sha256"]
                == expected["literal_prompt_sha256"],
                "image_sha256_match_manifest": input_metadata["consumed_image_sha256"]
                == sample["image_content_sha256"],
                "expected_task_correctness": baseline_state["correct"] == expected_correct,
            }
            if not all(baseline_checks.values()):
                raise RuntimeError(f"native FULL/current-dense parity failed: {baseline_checks}")

            start = int(manifest_row["first_trigger_layer"])
            full_actions = ["FULL"] * 28
            full_key = "|".join(full_actions)
            control = {
                "schema_version": "full_corrective_search_route_v1",
                "contract_sha256": contract["contract_sha256"],
                "uid": uid,
                "dataset": sample["dataset"],
                "trigger_layer": start,
                "trigger_depth_bin": manifest_row["trigger_depth_bin"],
                "search_stage": "all_full_control",
                "actions": full_actions,
                "route_key": full_key,
                **_route_counts(full_actions),
                **baseline_state,
                "physical_terminal_evaluation": False,
            }
            route_cache: dict[str, dict[str, Any]] = {full_key: control}

            def evaluate_actions(actions: Sequence[str], search_stage: str) -> tuple[dict[str, Any], bool]:
                route_key = "|".join(actions)
                if route_key in route_cache:
                    return route_cache[route_key], False
                route_started = time.monotonic()
                output = capture_four_action_suffix_from_full_baseline(
                    wrapped, baseline, start, tuple(actions[start:])
                )
                state = _generate_output(processor, wrapped, output, inputs, sample)
                row = {
                    "schema_version": "full_corrective_search_route_v1",
                    "contract_sha256": contract["contract_sha256"],
                    "uid": uid,
                    "dataset": sample["dataset"],
                    "trigger_layer": start,
                    "trigger_depth_bin": manifest_row["trigger_depth_bin"],
                    "search_stage": search_stage,
                    "actions": list(actions),
                    "route_key": route_key,
                    **_route_counts(actions),
                    **state,
                    "elapsed_seconds": time.monotonic() - route_started,
                    "prompt_decoder_layers": 28 - start,
                    "physical_terminal_evaluation": True,
                }
                route_cache[route_key] = row
                return row, True

            suffix_complete_parity = None
            mcts_result = None
            successful_routes: list[dict[str, Any]] = []
            single_routes_evaluated = 0
            successful_single_count = 0
            route_source: str | None = None
            if manifest_row["task_type"] == "triggered_correct_preservation":
                preservation, _ = evaluate_actions(full_actions, "preservation_full")
                preservation = {
                    **preservation,
                    "search_stage": "preservation_full",
                    "discovery_order": 0,
                }
                successful_routes = [preservation]
                route_source = "preservation_full"
                outcome_class = "PRESERVATION_FULL"
            else:
                singles = all_single_routes(start_layer=start)
                single_executions = [control]
                for candidate_index, candidate in enumerate(singles, 1):
                    route, _ = evaluate_actions(candidate["actions"], "single")
                    route["discovery_order"] = candidate_index
                    single_executions.append(route)
                    if route["correct"]:
                        successful_routes.append(route)
                single_routes_evaluated = len(singles)
                successful_single_count = len(successful_routes)
                if successful_routes:
                    outcome_class = "SINGLE_FIXABLE"
                    route_source = "single"
                    successful_routes.sort(key=lambda row: (int(row["discovery_order"]), row["route_key"]))
                else:
                    mcts_result = run_sequential_mcts(
                        uid=uid,
                        start_layer=start,
                        seed=int(config["seed"]),
                        max_iterations=int(config["mcts"]["maximum_iterations"]),
                        extra_iterations_after_success=int(
                            config["mcts"]["extra_iterations_after_first_success"]
                        ),
                        evaluate=lambda actions: evaluate_actions(actions, "mcts")[0]["correct"],
                        exploration_constant=float(config["mcts"]["exploration_constant"]),
                        rollout_cardinalities=config["mcts"]["rollout_cardinalities"],
                        retain_successes=int(config["mcts"]["retained_successful_routes"]),
                    )
                    first_by_key: dict[str, int] = {}
                    for search_row in mcts_result["search_rows"]:
                        if search_row["reward"]:
                            first_by_key.setdefault(
                                str(search_row["route_key"]), int(search_row["iteration"])
                            )
                    if mcts_result["first_success_iteration"] is not None:
                        outcome_class = "MCTS_ONLY_FIXABLE"
                        route_source = "mcts"
                        successful_routes = []
                        for retained in mcts_result["successful_routes"]:
                            route = route_cache[str(retained["route_key"])]
                            route["discovery_iteration"] = first_by_key[str(route["route_key"])]
                            successful_routes.append(route)
                    else:
                        outcome_class = "UNRESOLVED"

            parity_route = successful_routes[0] if successful_routes else next(
                row for key, row in route_cache.items() if key != full_key
            )
            if mode == "smoke":
                complete = capture_four_action_route(
                    wrapped,
                    inputs,
                    parity_route["actions"],
                    prepared_inputs=prepared,
                    use_cache=True,
                    native_full_rows=bool(config["execution"]["native_full_row_dispatch"]),
                )
                complete_state = _generate_output(processor, wrapped, complete, inputs, sample)
                suffix_complete_parity = {
                    "generated_ids_equal": complete_state["generated_ids"]
                    == parity_route["generated_ids"],
                    "answer_equal": complete_state["generated_answer"]
                    == parity_route["generated_answer"],
                    "lmms_score_equal": complete_state["lmms_score"] == parity_route["lmms_score"],
                    "correctness_equal": complete_state["correct"] == parity_route["correct"],
                }
                if not all(suffix_complete_parity.values()):
                    raise RuntimeError(f"cached-suffix/complete-route parity failed: {suffix_complete_parity}")

            feature_file, feature_hash, feature_index, route_rows = _save_native_route_states(
                routes=successful_routes,
                route_source=route_source or "single",
                manifest_row=manifest_row,
                baseline=baseline,
                wrapped=wrapped,
                processor=processor,
                base=base,
                inputs=inputs,
                contract=contract,
                output_root=output_root,
                smoke=mode == "smoke",
                rank=rank,
            )
            final = {
                "schema_version": "full_corrective_label_sample_result_v1",
                "contract_sha256": contract["contract_sha256"],
                "uid": uid,
                "dataset": sample["dataset"],
                "image_group_id": sample["image_group_id"],
                "trigger_layer": start,
                "trigger_depth_bin": manifest_row["trigger_depth_bin"],
                "worker_rank": rank,
                "mode": mode,
                "task_type": manifest_row["task_type"],
                "outcome_class": outcome_class,
                "result_origin": "fresh_execution",
                "baseline_state": baseline_state,
                "baseline_checks": baseline_checks,
                "suffix_complete_route_parity": suffix_complete_parity,
                "single_routes_evaluated": single_routes_evaluated,
                "successful_single_routes": successful_single_count,
                "mcts_iterations": 0 if mcts_result is None else mcts_result["iterations"],
                "mcts_unique_terminal_routes": 0
                if mcts_result is None
                else mcts_result["unique_terminal_routes"],
                "mcts_first_success_iteration": None
                if mcts_result is None
                else mcts_result["first_success_iteration"],
                "physical_terminal_evaluations": sum(
                    bool(row.get("physical_terminal_evaluation")) for row in route_cache.values()
                ),
                "retained_successful_routes": route_rows,
                "feature_file": feature_file,
                "feature_file_sha256": feature_hash,
                "feature_index": feature_index,
                "source_elapsed_seconds": None,
                "fresh_elapsed_seconds": time.monotonic() - started,
                "passed": True,
            }
            atomic_json(sample_path, final)
            completed[uid] = final
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "rank": rank,
                        "sample": index + 1,
                        "total": len(rows),
                        "uid": uid,
                        "class": outcome_class,
                        "physical_routes": final["physical_terminal_evaluations"],
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            failures += 1
            atomic_json(
                worker_root / f"failure_{_safe_uid(uid)}.json",
                {
                    "contract_sha256": contract["contract_sha256"],
                    "uid": uid,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "worker_rank": rank,
                    "created_at": utc_now(),
                },
            )
            break
        finally:
            torch.cuda.empty_cache()
    if failures:
        raise RuntimeError(f"worker {rank} failed")
    expected_uids = [str(row["uid"]) for row in rows]
    final_rows = list(completed.values())
    validate_complete_results(expected_uids, final_rows)
    atomic_json(
        complete_path,
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "mode": mode,
            "rank": rank,
            "records": len(final_rows),
            "worker_elapsed_seconds": time.monotonic() - worker_started,
            "completed_at": utc_now(),
        },
    )


def finalize_smoke(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    samples = _collect_mode(output_root, "smoke", contract["contract_sha256"])
    manifest = read_jsonl(output_root / "work/manifests/smoke_manifest.jsonl")
    validate_complete_results([row["uid"] for row in manifest], samples)
    if len(samples) != 8 or Counter(row["task_type"] for row in samples) != Counter(
        {"triggered_wrong": 4, "triggered_correct_preservation": 4}
    ):
        raise RuntimeError("smoke population differs from the frozen 4W+4C design")
    if not all(all(row["baseline_checks"].values()) for row in samples):
        raise RuntimeError("smoke dense-prefix parity failed")
    if not all(
        row["suffix_complete_route_parity"]
        and all(row["suffix_complete_route_parity"].values())
        for row in samples
    ):
        raise RuntimeError("smoke cached-suffix/full-route parity failed")
    feature_rows = 0
    for row in samples:
        if not row.get("feature_file"):
            if row["outcome_class"] != "UNRESOLVED":
                raise RuntimeError("fixable smoke sample lacks routed states")
            continue
        path = output_root / row["feature_file"]
        if file_sha256(path) != row["feature_file_sha256"]:
            raise RuntimeError("smoke feature shard hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _validate_saved_shard(path, payload)
        if any(not item["replay_token_parity"] for item in payload["records"]):
            raise RuntimeError("smoke route replay flag failed")
        feature_rows += len(payload["records"])
    atomic_json(
        output_root / "work/smoke/completion.json",
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "samples": len(samples),
            "wrong_samples": 4,
            "preservation_samples": 4,
            "feature_rows": feature_rows,
            "completed_at": utc_now(),
        },
    )
    report = f"""# Full corrective-label smoke report

- Contract: `{contract['contract_sha256']}`
- Frozen implementation smoke: 8/8 records (4 triggered-W full searches and 4 triggered-C preservation replays) completed across four ranks.
- Native all-FULL/current-dense token, answer, prompt, image SHA-256, LMMS score, and correctness parity: **PASS**.
- Cached-prefix suffix versus complete four-action route parity: **PASS**.
- Exact successful-route replay and contract-bound routed-state capture: **PASS** ({feature_rows} state rows).
- Unique UID coverage, four-rank completion, and no failure records: **PASS**.
- Smoke rows are implementation evidence only and are excluded from final corpus counts.
"""
    _atomic_bytes(output_root / "work/smoke/smoke_report.md", report.encode())
    print(json.dumps({"passed": True, "samples": len(samples), "feature_rows": feature_rows}))


def _rate(count: int, total: int) -> float:
    return count / total if total else float("nan")


def _plot_composition(path: Path, labels: Sequence[str], values: Sequence[float], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4.8))
    colors = ["#4C78A8", "#F58518", "#A0A0A0"][: len(values)]
    bars = axis.bar(labels, values, color=colors)
    axis.set_ylabel("Fraction")
    axis.set_ylim(0, max(1.0, max(values, default=0.0) * 1.15))
    axis.set_title(title)
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_stacked_fixability(
    path: Path, labels: Sequence[str], single: Sequence[float], mcts: Sequence[float], title: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(x, single, label="Single", color="#4C78A8")
    axis.bar(x, mcts, bottom=single, label="MCTS only", color="#F58518")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Fraction of triggered Dense-W")
    axis.set_title(title)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _result_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "uid": row["uid"],
        "dataset": row["dataset"],
        "image_group_id": row["image_group_id"],
        "trigger_layer": row["trigger_layer"],
        "trigger_depth_bin": row["trigger_depth_bin"],
        "fixability_type": row["outcome_class"],
        "result_origin": row["result_origin"],
        "single_routes_evaluated": row["single_routes_evaluated"],
        "successful_single_routes": row["successful_single_routes"],
        "mcts_iterations": row["mcts_iterations"],
        "mcts_unique_terminal_routes": row["mcts_unique_terminal_routes"],
        "mcts_first_success_iteration": row["mcts_first_success_iteration"],
        "physical_terminal_evaluations": row["physical_terminal_evaluations"],
        "retained_successful_routes": len(row["retained_successful_routes"]),
        "passed": row["passed"],
    }


def aggregate(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    if not read_json(output_root / "work/smoke/completion.json").get("passed"):
        raise RuntimeError("aggregation requires a passing smoke")
    if not read_json(output_root / "work/imported/complete.json").get("passed"):
        raise RuntimeError("aggregation requires a complete pilot import")
    fresh = _collect_mode(output_root, "full", contract["contract_sha256"])
    fresh_manifest = read_jsonl(output_root / "work/manifests/full_work_manifest.jsonl")
    validate_complete_results([row["uid"] for row in fresh_manifest], fresh)
    if len(fresh) != 1800:
        raise RuntimeError("fresh completion must contain 1,761 W + 39 C rows")
    imported = [
        read_json(path)
        for path in sorted((output_root / "work/imported/samples").glob("*.json"))
    ]
    import_manifest = read_jsonl(output_root / "work/manifests/pilot_import_selection.jsonl")
    validate_complete_results([row["uid"] for row in import_manifest], imported)
    if len(imported) != 120:
        raise RuntimeError("imported completion must contain 120 pilot rows")
    all_results = sorted([*fresh, *imported], key=lambda row: str(row["uid"]))
    wrong_manifest = read_jsonl(output_root / "manifests/full_triggered_wrong_manifest.jsonl")
    correct_source = read_jsonl(
        resolve_path(contract["static_config"]["sources"]["triggered_correct_manifest"])
    )
    validate_population_completion(
        [row["uid"] for row in wrong_manifest],
        [row["uid"] for row in correct_source],
        all_results,
    )
    wrong_results = [row for row in all_results if row["task_type"] == "triggered_wrong"]
    correct_results = [
        row for row in all_results if row["task_type"] == "triggered_correct_preservation"
    ]
    if len(wrong_results) != 1881 or len(correct_results) != 39:
        raise RuntimeError("final W/C result counts differ from the frozen populations")
    class_counts = Counter(row["outcome_class"] for row in wrong_results)
    if set(class_counts) - set(FIXABILITY_CLASSES):
        raise RuntimeError(f"unsupported final fixability class: {class_counts}")
    if sum(class_counts.values()) != 1881:
        raise RuntimeError("every triggered-W row must have exactly one final class")

    route_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    verified_shards: set[str] = set()
    for result in all_results:
        expected_source = {
            "SINGLE_FIXABLE": "single",
            "MCTS_ONLY_FIXABLE": "mcts",
            "UNRESOLVED": None,
            "PRESERVATION_FULL": "preservation_full",
        }[result["outcome_class"]]
        routes = result["retained_successful_routes"]
        if expected_source is None and routes:
            raise RuntimeError(f"unresolved sample has retained routes: {result['uid']}")
        if expected_source is not None and (
            not routes or any(row["route_source"] != expected_source for row in routes)
        ):
            raise RuntimeError(f"route-source/class mismatch: {result['uid']}")
        if any(
            not row["final_lmms_correct"] or not row["replay_token_parity"] for row in routes
        ):
            raise RuntimeError(f"non-correct or non-replayed route entered final labels: {result['uid']}")
        route_rows.extend(routes)
        feature_file = result.get("feature_file")
        if not feature_file:
            if routes:
                raise RuntimeError(f"retained routes lack a state shard: {result['uid']}")
            continue
        feature_path = output_root / str(feature_file)
        if file_sha256(feature_path) != result["feature_file_sha256"]:
            raise RuntimeError(f"final feature shard hash mismatch: {result['uid']}")
        payload = torch.load(feature_path, map_location="cpu", weights_only=False)
        _validate_saved_shard(feature_path, payload)
        if payload["contract_sha256"] != contract["contract_sha256"]:
            raise RuntimeError(f"feature shard contract mismatch: {result['uid']}")
        verified_shards.add(str(feature_file))
        state_rows.extend(result["feature_index"])

    if len({str(row["route_id"]) for row in route_rows}) != len(route_rows):
        raise RuntimeError("route IDs are not globally unique")
    expected_state_rows = sum(int(row["feature_rows"]) for row in route_rows)
    if len(state_rows) != expected_state_rows:
        raise RuntimeError("route-manifest/state-index row counts differ")
    route_by_id = {str(row["route_id"]): row for row in route_rows}
    observed_actions: dict[tuple[str, int, tuple[str, ...]], set[str]] = defaultdict(set)
    for row in state_rows:
        route = route_by_id[str(row["route_id"])]
        layer = int(row["layer"])
        state_key = (str(row["uid"]), layer, tuple(route["actions"][:layer]))
        observed_actions[state_key].add(str(row["chosen_action"]))
    for row in state_rows:
        route = route_by_id[str(row["route_id"])]
        layer = int(row["layer"])
        state_key = (str(row["uid"]), layer, tuple(route["actions"][:layer]))
        row["observed_successful_actions"] = sorted(observed_actions[state_key])

    single_results = [row for row in wrong_results if row["outcome_class"] == "SINGLE_FIXABLE"]
    mcts_results = [row for row in wrong_results if row["outcome_class"] == "MCTS_ONLY_FIXABLE"]
    unresolved_results = [row for row in wrong_results if row["outcome_class"] == "UNRESOLVED"]
    atomic_jsonl(
        output_root / "manifests/single_fixable.jsonl",
        [_result_summary(row) for row in single_results],
    )
    atomic_jsonl(
        output_root / "manifests/mcts_only_fixable.jsonl",
        [_result_summary(row) for row in mcts_results],
    )
    atomic_jsonl(
        output_root / "manifests/unresolved.jsonl",
        [_result_summary(row) for row in unresolved_results],
    )
    preservation_rows = []
    for result in correct_results:
        route = result["retained_successful_routes"][0]
        preservation_rows.append(
            {
                **_result_summary(result),
                "route_source": "preservation_full",
                "dense_correct": True,
                "final_correct": True,
                "route_id": route["route_id"],
                "actions": route["actions"],
                "feature_file": route["feature_file"],
                "feature_file_sha256": route["feature_file_sha256"],
            }
        )
    atomic_jsonl(
        output_root / "manifests/triggered_correct_preservation.jsonl",
        preservation_rows,
    )

    single_routes = sorted(
        [row for row in route_rows if row["route_source"] == "single"],
        key=lambda row: (str(row["uid"]), str(row["route_id"])),
    )
    mcts_routes = sorted(
        [row for row in route_rows if row["route_source"] == "mcts"],
        key=lambda row: (str(row["uid"]), str(row["route_id"])),
    )
    preservation_routes = sorted(
        [row for row in route_rows if row["route_source"] == "preservation_full"],
        key=lambda row: str(row["uid"]),
    )
    atomic_jsonl(output_root / "routes/successful_single_routes.jsonl", single_routes)
    atomic_jsonl(output_root / "routes/successful_mcts_routes.jsonl", mcts_routes)
    preferred = []
    for result in [*single_results, *mcts_results]:
        chosen = choose_preferred_route(result["retained_successful_routes"])
        if chosen is None:
            raise RuntimeError(f"fixable sample has no preferred route candidates: {result['uid']}")
        preferred.append({**chosen, "preferred_for_diagnostics": True})
    preferred.sort(key=lambda row: str(row["uid"]))
    atomic_jsonl(output_root / "routes/preferred_routes.jsonl", preferred)
    atomic_jsonl(output_root / "states/state_index.jsonl", state_rows)

    corpora = build_corpus_rows(route_rows)
    corpus_paths = {
        "A": "stage2_corpora/corpus_A_preservation.jsonl",
        "B": "stage2_corpora/corpus_B_single_corrective.jsonl",
        "C": "stage2_corpora/corpus_C_mcts_corrective.jsonl",
    }
    for key, relative in corpus_paths.items():
        atomic_jsonl(output_root / relative, corpora[key])
    source_to_key = {"preservation_full": "A", "single": "B", "mcts": "C"}
    state_counts = Counter(source_to_key[str(row["route_source"])] for row in state_rows)
    corpus_manifest = {
        "schema_version": "full_stage2_corpus_manifest_v1",
        "contract_sha256": contract["contract_sha256"],
        "feature_schema_sha256": contract["feature_schema_sha256"],
        "sampling_policy": "not_selected; future training must choose sample-balanced or route-balanced explicitly",
        "corpora": {
            key: {
                "path": relative,
                "sha256": file_sha256(output_root / relative),
                "route_records": len(corpora[key]),
                "unique_samples": len({str(row["uid"]) for row in corpora[key]}),
                "training_state_records": state_counts[key],
                "route_source": {"A": "preservation_full", "B": "single", "C": "mcts"}[key],
            }
            for key, relative in corpus_paths.items()
        },
    }
    atomic_json(output_root / "stage2_corpora/corpus_manifest.json", corpus_manifest)

    single_count = class_counts["SINGLE_FIXABLE"]
    mcts_count = class_counts["MCTS_ONLY_FIXABLE"]
    unresolved_count = class_counts["UNRESOLVED"]
    overall_rows = [
        {
            "population": "triggered_dense_wrong",
            "records": 1881,
            "single_fixable": single_count,
            "single_fixable_rate": _rate(single_count, 1881),
            "mcts_only_fixable": mcts_count,
            "mcts_only_fixable_rate": _rate(mcts_count, 1881),
            "total_bounded_fixable": single_count + mcts_count,
            "total_bounded_fixable_rate": _rate(single_count + mcts_count, 1881),
            "unresolved": unresolved_count,
            "unresolved_rate": _rate(unresolved_count, 1881),
            "pilot_cap200_total_fixable_rate": 56 / 120,
            "full_minus_pilot_rate": _rate(single_count + mcts_count, 1881) - 56 / 120,
        }
    ]
    atomic_csv(output_root / "metrics/overall_fixability.csv", overall_rows)

    preferred_by_uid = {str(row["uid"]): row for row in preferred}

    def breakdown(field: str, values: Sequence[str]) -> list[dict[str, Any]]:
        output = []
        for value in values:
            cell = [row for row in wrong_results if str(row[field]) == value]
            counts = Counter(row["outcome_class"] for row in cell)
            route_complexities = [
                int(preferred_by_uid[str(row["uid"])]["non_full_count"])
                for row in cell
                if str(row["uid"]) in preferred_by_uid
            ]
            actions = Counter(
                action
                for row in cell
                if str(row["uid"]) in preferred_by_uid
                for action in preferred_by_uid[str(row["uid"])]["actions"]
                if action != "FULL"
            )
            output.append(
                {
                    field: value,
                    "records": len(cell),
                    "single_fixable": counts["SINGLE_FIXABLE"],
                    "single_fixable_rate": _rate(counts["SINGLE_FIXABLE"], len(cell)),
                    "mcts_only_fixable": counts["MCTS_ONLY_FIXABLE"],
                    "mcts_only_fixable_rate": _rate(counts["MCTS_ONLY_FIXABLE"], len(cell)),
                    "total_bounded_fixable": counts["SINGLE_FIXABLE"] + counts["MCTS_ONLY_FIXABLE"],
                    "total_bounded_fixable_rate": _rate(
                        counts["SINGLE_FIXABLE"] + counts["MCTS_ONLY_FIXABLE"], len(cell)
                    ),
                    "unresolved": counts["UNRESOLVED"],
                    "unresolved_rate": _rate(counts["UNRESOLVED"], len(cell)),
                    "preferred_route_median_non_full": (
                        float(np.median(route_complexities)) if route_complexities else None
                    ),
                    "preferred_read_only": actions["READ_ONLY"],
                    "preferred_write_only": actions["WRITE_ONLY"],
                    "preferred_ignore": actions["IGNORE"],
                }
            )
        return output

    dataset_rows = breakdown("dataset", DATASETS)
    depth_rows = breakdown("trigger_depth_bin", DEPTH_BINS)
    atomic_csv(output_root / "metrics/dataset_breakdown.csv", dataset_rows)
    atomic_csv(output_root / "metrics/trigger_depth_breakdown.csv", depth_rows)

    preferred_complexities = [int(row["non_full_count"]) for row in preferred]
    mcts_complexities = [int(row["non_full_count"]) for row in preferred if row["route_source"] == "mcts"]
    complexity_rows = []
    for scope, values in (("all_fixable", preferred_complexities), ("mcts_only", mcts_complexities)):
        for label, predicate in (
            ("1", lambda value: value == 1),
            ("2", lambda value: value == 2),
            ("3", lambda value: value == 3),
            ("4+", lambda value: value >= 4),
        ):
            count = sum(predicate(value) for value in values)
            complexity_rows.append(
                {
                    "scope": scope,
                    "complexity_bin": label,
                    "samples": count,
                    "fraction": _rate(count, len(values)),
                    "median": float(np.median(values)) if values else None,
                    "q25": float(np.quantile(values, 0.25)) if values else None,
                    "q75": float(np.quantile(values, 0.75)) if values else None,
                    "p95": float(np.quantile(values, 0.95)) if values else None,
                }
            )
    atomic_csv(output_root / "metrics/route_complexity.csv", complexity_rows)

    action_layer_rows = []
    for source in ("preservation_full", "single", "mcts"):
        source_routes = [row for row in route_rows if row["route_source"] == source]
        for layer in range(28):
            for action in ACTIONS:
                count = sum(
                    int(row["trigger_layer"]) <= layer and row["actions"][layer] == action
                    for row in source_routes
                )
                action_layer_rows.append(
                    {
                        "route_source": source,
                        "layer": layer,
                        "action": action,
                        "count": count,
                        "eligible_routes": sum(int(row["trigger_layer"]) <= layer for row in source_routes),
                    }
                )
    atomic_csv(output_root / "metrics/action_by_layer.csv", action_layer_rows)
    action_distribution_rows = []
    for source in ("preservation_full", "single", "mcts"):
        source_rows = [row for row in action_layer_rows if row["route_source"] == source]
        total = sum(int(row["count"]) for row in source_rows)
        for action in ACTIONS:
            count = sum(int(row["count"]) for row in source_rows if row["action"] == action)
            action_distribution_rows.append(
                {
                    "route_source": source,
                    "action": action,
                    "count": count,
                    "fraction_of_suffix_actions": _rate(count, total),
                }
            )
    atomic_csv(output_root / "metrics/action_distribution.csv", action_distribution_rows)

    worker_completions = [
        read_json(_worker_root(output_root, "full", rank) / "complete.json")
        for rank in range(4)
    ]
    compute_rows = [
        {"metric": "triggered_wrong_records", "value": 1881, "unit": "samples"},
        {"metric": "pilot_records_imported", "value": 120, "unit": "samples"},
        {"metric": "fresh_wrong_records", "value": 1761, "unit": "samples"},
        {"metric": "fresh_preservation_records", "value": 39, "unit": "samples"},
        {"metric": "single_routes_evaluated", "value": sum(row["single_routes_evaluated"] for row in wrong_results), "unit": "routes"},
        {"metric": "mcts_iterations", "value": sum(row["mcts_iterations"] for row in wrong_results), "unit": "iterations"},
        {"metric": "physical_terminal_evaluations", "value": sum(row["physical_terminal_evaluations"] for row in wrong_results), "unit": "routes"},
        {"metric": "successful_single_routes_retained", "value": len(single_routes), "unit": "routes"},
        {"metric": "successful_mcts_routes_retained", "value": len(mcts_routes), "unit": "routes"},
        {"metric": "preservation_routes_retained", "value": len(preservation_routes), "unit": "routes"},
        {"metric": "routed_training_state_records", "value": len(state_rows), "unit": "states"},
        {"metric": "fresh_sample_gpu_seconds", "value": sum(float(row["fresh_elapsed_seconds"]) for row in fresh), "unit": "gpu_seconds"},
        {"metric": "fresh_sample_gpu_hours", "value": sum(float(row["fresh_elapsed_seconds"]) for row in fresh) / 3600, "unit": "gpu_hours"},
        {"metric": "max_worker_wall_seconds", "value": max(float(row.get("worker_elapsed_seconds", 0)) for row in worker_completions), "unit": "seconds"},
        {"metric": "phase55_source_elapsed_seconds_not_charged_to_phase56", "value": sum(float(row["source_elapsed_seconds"]) for row in imported), "unit": "historical_gpu_seconds"},
    ]
    atomic_csv(output_root / "metrics/compute_summary.csv", compute_rows)

    _plot_composition(
        output_root / "figures/fixability_composition.png",
        ["Single", "MCTS only", "Unresolved"],
        [_rate(single_count, 1881), _rate(mcts_count, 1881), _rate(unresolved_count, 1881)],
        "Full triggered-W bounded fixability",
    )
    all_complexity = [
        next(row for row in complexity_rows if row["scope"] == "all_fixable" and row["complexity_bin"] == label)["fraction"]
        for label in ("1", "2", "3", "4+")
    ]
    _plot_composition(
        output_root / "figures/route_complexity_distribution.png",
        ["1", "2", "3", "4+"],
        all_complexity,
        "Preferred successful-route complexity",
    )
    single_layer_counts = Counter(int(row["intervention_layer"]) for row in single_routes)
    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(range(28), [single_layer_counts[layer] for layer in range(28)], color="#4C78A8")
    axis.set_xlabel("Decoder layer")
    axis.set_ylabel("Successful single routes")
    axis.set_title("Single corrective intervention layer")
    fig.tight_layout()
    fig.savefig(output_root / "figures/single_intervention_layer_distribution.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 5))
    for action, color in (("READ_ONLY", "#4C78A8"), ("WRITE_ONLY", "#F58518"), ("IGNORE", "#54A24B")):
        axis.plot(
            range(28),
            [sum(row["actions"][layer] == action and int(row["trigger_layer"]) <= layer for row in [*single_routes, *mcts_routes]) for layer in range(28)],
            label=action,
            color=color,
        )
    axis.set_xlabel("Decoder layer")
    axis.set_ylabel("Successful-route action count")
    axis.set_title("Corrective actions by layer")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_root / "figures/action_by_layer.png", dpi=180)
    plt.close(fig)
    _plot_stacked_fixability(
        output_root / "figures/dataset_fixability.png",
        [row["dataset"] for row in dataset_rows],
        [row["single_fixable_rate"] for row in dataset_rows],
        [row["mcts_only_fixable_rate"] for row in dataset_rows],
        "Bounded fixability by dataset",
    )
    _plot_stacked_fixability(
        output_root / "figures/trigger_depth_fixability.png",
        [row["trigger_depth_bin"] for row in depth_rows],
        [row["single_fixable_rate"] for row in depth_rows],
        [row["mcts_only_fixable_rate"] for row in depth_rows],
        "Bounded fixability by trigger depth",
    )

    median_complexity = float(np.median(preferred_complexities)) if preferred_complexities else float("nan")
    q25 = float(np.quantile(preferred_complexities, 0.25)) if preferred_complexities else float("nan")
    q75 = float(np.quantile(preferred_complexities, 0.75)) if preferred_complexities else float("nan")
    p95 = float(np.quantile(preferred_complexities, 0.95)) if preferred_complexities else float("nan")
    dataset_rates = [float(row["total_bounded_fixable_rate"]) for row in dataset_rows]
    depth_rates = [float(row["total_bounded_fixable_rate"]) for row in depth_rows]
    pilot_delta = _rate(single_count + mcts_count, 1881) - 56 / 120
    decision = f"""# Full corrective-label generation decision summary

Contract: `{contract['contract_sha256']}`. The final corpus covers exactly the frozen 1,881 triggered Dense-W and 39 triggered Dense-C train samples. `UNRESOLVED` means no rescue was found under this bounded contract, not proven unfixability.

1. **SINGLE_FIXABLE:** {single_count}/1,881 = **{_rate(single_count, 1881):.4f}**.
2. **Additional MCTS_ONLY_FIXABLE:** {mcts_count}/1,881 = **{_rate(mcts_count, 1881):.4f}**.
3. **Final bounded correctability:** {single_count + mcts_count}/1,881 = **{_rate(single_count + mcts_count, 1881):.4f}**; {unresolved_count}/1,881 remain unresolved.
4. **Pilot comparison:** cap-200 pilot projection was 56/120 = 0.4667; the full-cohort rate differs by **{pilot_delta:+.4f}**.
5. **Successful single routes:** **{len(single_routes)}** exact-replay routes were retained across {single_count} samples.
6. **Successful MCTS routes:** **{len(mcts_routes)}** exact-replay routes were retained across {mcts_count} samples.
7. **Route complexity:** preferred successful routes have median **{median_complexity:.2f}** non-FULL actions (IQR {q25:.2f}–{q75:.2f}, 95th percentile {p95:.2f}); exact all-fixable and MCTS-only bins are in `metrics/route_complexity.csv`.
8. **Single intervention location/action:** exact layer×action counts are in `metrics/action_by_layer.csv`; the dedicated distribution is `figures/single_intervention_layer_distribution.png`.
9. **Dataset variation:** total correctability spans **{max(dataset_rates) - min(dataset_rates):.4f}** across GQA/ChartQA/TextVQA; exact simple/MCTS/unresolved counts and actions are in `metrics/dataset_breakdown.csv`.
10. **Trigger-depth variation:** total correctability spans **{max(depth_rates) - min(depth_rates):.4f}** across L0/L1-8/L9-18/L19-27; exact results are in `metrics/trigger_depth_breakdown.csv`.
11. **Training states:** Corpus A has **{state_counts['A']}**, Corpus B **{state_counts['B']}**, and Corpus C **{state_counts['C']}** pre-layer routed-state records.
12. **Stage-2 V1 support:** **{'yes' if preservation_routes and single_routes else 'no'}**—Corpus A contains {len(preservation_routes)} preservation routes and Corpus B contains {len(single_routes)} simple corrective routes, kept explicitly separate.
13. **Stage-2 V2 support:** **{'yes' if mcts_count > 0 else 'not established'}**—Corpus C adds {mcts_count} MCTS-only samples and {len(mcts_routes)} retained trajectories, but whether they improve a learned policy remains untested.
14. **Replay/provenance integrity:** **PASS**. All {len(route_rows)} routes are current-LMMS-correct and exact-token replay valid; all {len(state_rows)} states are in {len(verified_shards)} contract/model/code/schema/source-bound shards. The 120-row pilot import uses only canonical cap-200 routes; the iteration-294 rescue and its tensor rows were excluded.

## Corpus boundary and stop

- Corpus A is only `preservation_full`; Corpus B only `single`; Corpus C only `mcts`.
- Multiple successful actions sharing the same routed prefix are retained as `observed_successful_actions` in `states/state_index.jsonl`; this is not a claim that all valid actions were found.
- Route multiplicity has not been converted into future sample weight.
- No Stage-2 model was trained, Stage 1 was not changed, validation/test were not searched, and no external/downstream evaluation ran.
"""
    _atomic_bytes(output_root / "decision_summary.md", decision.encode())

    required = [
        "protocol.md",
        "manifests/full_triggered_wrong_manifest.jsonl",
        "manifests/single_fixable.jsonl",
        "manifests/mcts_only_fixable.jsonl",
        "manifests/unresolved.jsonl",
        "manifests/triggered_correct_preservation.jsonl",
        "routes/successful_single_routes.jsonl",
        "routes/successful_mcts_routes.jsonl",
        "routes/preferred_routes.jsonl",
        "states/feature_schema.json",
        "states/state_index.jsonl",
        "metrics/overall_fixability.csv",
        "metrics/dataset_breakdown.csv",
        "metrics/trigger_depth_breakdown.csv",
        "metrics/route_complexity.csv",
        "metrics/action_by_layer.csv",
        "metrics/action_distribution.csv",
        "metrics/compute_summary.csv",
        "stage2_corpora/corpus_A_preservation.jsonl",
        "stage2_corpora/corpus_B_single_corrective.jsonl",
        "stage2_corpora/corpus_C_mcts_corrective.jsonl",
        "stage2_corpora/corpus_manifest.json",
        "figures/fixability_composition.png",
        "figures/route_complexity_distribution.png",
        "figures/single_intervention_layer_distribution.png",
        "figures/action_by_layer.png",
        "figures/dataset_fixability.png",
        "figures/trigger_depth_fixability.png",
        "decision_summary.md",
    ]
    missing = [relative for relative in required if not (output_root / relative).is_file()]
    if missing:
        raise RuntimeError(f"required full-label artifacts are missing: {missing}")
    artifact_paths = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file()
        and "work" not in path.relative_to(output_root).parts
        and path.name != "artifact_manifest.json"
    )
    artifact_manifest = {
        "schema_version": "full_corrective_label_artifact_manifest_v1",
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        "triggered_wrong_records": 1881,
        "triggered_correct_preservation_records": 39,
        "outcome_counts": dict(class_counts),
        "successful_single_routes": len(single_routes),
        "successful_mcts_routes": len(mcts_routes),
        "preservation_routes": len(preservation_routes),
        "routed_training_state_records": len(state_rows),
        "verified_state_shards": len(verified_shards),
        "files": {str(path.relative_to(output_root)): file_sha256(path) for path in artifact_paths},
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "artifact_manifest.json", artifact_manifest)
    verify_artifact_manifest(output_root, artifact_manifest)
    print(
        json.dumps(
            {
                "passed": True,
                "contract_sha256": contract["contract_sha256"],
                "outcome_counts": dict(class_counts),
                "routes": len(route_rows),
                "states": len(state_rows),
            }
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare", "import-pilot", "worker", "finalize-smoke", "aggregate"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("smoke", "full"))
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.config)
    elif args.command == "import-pilot":
        import_pilot(args.config)
    elif args.command == "worker":
        if args.mode is None or args.rank is None:
            raise SystemExit("worker requires --mode and --rank")
        worker(
            args.config,
            mode=args.mode,
            rank=args.rank,
            world_size=args.world_size,
            resume=args.resume,
        )
    elif args.command == "finalize-smoke":
        finalize_smoke(args.config)
    else:
        aggregate(args.config)


if __name__ == "__main__":
    main()
