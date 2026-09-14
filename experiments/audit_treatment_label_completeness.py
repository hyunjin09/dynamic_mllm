#!/usr/bin/env python3
"""Run the frozen large-scale Stage-2 treatment-label completeness audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from binary_policy.executor import (  # noqa: E402
    capture_four_action_route,
    capture_four_action_suffix_from_route_baseline,
)
from binary_policy.executor.inputs import build_binary_inputs  # noqa: E402
from dense_failure_stage1.runtime import build_dense_inputs, configure_dense_determinism  # noqa: E402
from dense_failure_stage2.corrective_search import run_sequential_mcts  # noqa: E402
from dense_failure_stage2.full_label_generation import file_sha256, verify_artifact_manifest  # noqa: E402
from dense_failure_stage2.treatment_label_completeness import (  # noqa: E402
    ACTIONS,
    AUDITED_INTERVENE,
    AUDITED_KEEP,
    AUDITED_MIXED,
    classify_audited_actions,
    deterministic_state_sample,
    mcts_budget_saturation,
    unobserved_actions,
)
from dense_failure_stage2.treatment_selectivity import (  # noqa: E402
    assign_group_folds,
    binary_metrics,
    exact_router_representations,
    fit_binary_probe,
    predict_binary_probe,
    selective_operating_points,
    standardize_fold,
    training_weights,
)
from experiments.analyze_stage2_treatment_selectivity import (  # noqa: E402
    _load_feature_map,
    _load_frozen_router,
)
from experiments.run_robust_gate_corrective_search import _runtime_metadata  # noqa: E402
from experiments.run_trigger_conditioned_corrective_search_pilot import (  # noqa: E402
    _generate_output,
    _load_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/treatment_label_completeness_audit_v1.json"
OUTPUT_SUBDIRS = (
    "sampling", "search", "labels", "metrics", "probe_recheck", "figures", "summaries",
    "work/smoke", "work/full",
)
BOUND_CODE = (
    "configs/treatment_label_completeness_audit_v1.json",
    "dense_failure_stage2/treatment_label_completeness.py",
    "dense_failure_stage2/corrective_search.py",
    "dense_failure_stage2/treatment_selectivity.py",
    "dense_failure_stage2/v1_router.py",
    "dense_failure_stage1/lmms_scoring.py",
    "dense_failure_stage1/runtime.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/model.py",
    "experiments/analyze_stage2_treatment_selectivity.py",
    "experiments/run_stage2_shared_union_training.py",
    "experiments/run_trigger_conditioned_corrective_search_pilot.py",
    "experiments/audit_treatment_label_completeness.py",
)
PROBE_NAMES = ("M0_margin", "M1_logits", "R_read", "W_write", "RW_read_write")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    roots = (PROJECT_ROOT.resolve(), Path("/mnt/hyemin").resolve())
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise ValueError(f"path escapes allowed roots: {value}")
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def canonical_hash(value: Mapping[str, Any], *, excluded: Sequence[str] = ("contract_sha256",)) -> str:
    payload = {key: item for key, item in value.items() if key not in excluded}
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def command_output(command: Sequence[str]) -> str:
    return subprocess.run(
        list(command), cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    _atomic_bytes(path, payload.encode())


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        _atomic_bytes(path, b"")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode())


def _atomic_text(path: Path, text: str) -> None:
    _atomic_bytes(path, text.encode())


def _safe_id(value: str) -> str:
    return sha256(value.encode()).hexdigest()[:24]


def _sample_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    sample = row["sample"]
    return {
        key: sample[key]
        for key in (
            "uid", "sample_id", "dataset", "prompt", "question", "answer",
            "all_answer_norms", "local_image_path", "image_content_sha256",
            "image_group_id", "max_new_tokens",
        )
    }


def _select_smoke(rows: Sequence[Mapping[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["unobserved_actions"] and int(row["layer"]) < 27]
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        cells[(str(row["old_label"]), str(row["dataset"]))].append(row)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cell in sorted(cells):
        candidates = sorted(
            cells[cell],
            key=lambda row: (
                sha256(f"{seed}:smoke:{row['state_id']}".encode()).hexdigest(),
                str(row["state_id"]),
            ),
        )
        if candidates and len(selected) < count:
            selected.append(dict(candidates[0]))
            seen.add(str(candidates[0]["state_id"]))
    remaining = sorted(
        [row for row in eligible if str(row["state_id"]) not in seen],
        key=lambda row: (
            sha256(f"{seed}:smoke-fill:{row['state_id']}".encode()).hexdigest(),
            str(row["state_id"]),
        ),
    )
    selected.extend(dict(row) for row in remaining[: count - len(selected)])
    if len(selected) != count:
        raise RuntimeError("insufficient smoke states")
    return sorted(selected, key=lambda row: str(row["state_id"]))


def _assign_state_workers(rows: Sequence[Mapping[str, Any]], world_size: int) -> list[dict[str, Any]]:
    by_uid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_uid[str(row["uid"])].append(row)
    uid_costs = sorted(
        ((sum(int(row["estimated_max_terminal_evaluations"]) for row in group), uid) for uid, group in by_uid.items()),
        key=lambda item: (-item[0], item[1]),
    )
    loads = [0] * world_size
    ranks: dict[str, int] = {}
    for cost, uid in uid_costs:
        rank = min(range(world_size), key=lambda item: (loads[item], item))
        ranks[uid] = rank
        loads[rank] += cost
    return [
        {**row, "worker_rank": ranks[str(row["uid"])]}
        for row in sorted(rows, key=lambda item: str(item["state_id"]))
    ]


def _state_cost(row: Mapping[str, Any]) -> int:
    layer = int(row["layer"])
    per_branch = 1 + 3 * max(27 - layer, 0) + (200 if layer < 27 else 0) + 1
    return 1 + len(row["unobserved_actions"]) * per_branch


def prepare(config_path: Path) -> None:
    config = read_json(resolve_path(config_path))
    output_root = resolve_path(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    for relative in OUTPUT_SUBDIRS:
        (output_root / relative).mkdir(parents=True, exist_ok=True)

    sources = {name: resolve_path(value) for name, value in config["sources"].items()}
    phase72 = read_json(sources["phase72_contract"])
    phase65 = read_json(sources["phase65_contract"])
    phase66 = read_json(sources["phase66_contract"])
    for name, contract in (("Phase 72", phase72), ("Phase 65", phase65), ("Phase 66", phase66)):
        if canonical_hash(contract) != contract.get("contract_sha256"):
            raise RuntimeError(f"{name} frozen contract hash is invalid")
    for artifact_name, contract in (
        ("phase72_artifact_manifest", phase72),
        ("phase65_artifact_manifest", phase65),
        ("phase66_artifact_manifest", phase66),
    ):
        artifact_path = sources[artifact_name]
        artifact = read_json(artifact_path)
        verify_artifact_manifest(artifact_path.parent, artifact)
        if artifact.get("contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError(f"{artifact_name} contract differs")
    if phase72["population"] != {
        **phase72["population"],
        "unique_exact_states": 21071,
        "KEEP_REQUIRED": 18438,
        "INTERVENE_REQUIRED": 1657,
        "MIXED": 976,
    }:
        raise RuntimeError("Phase-72 population differs from the approved census")

    exact_rows = read_jsonl(sources["exact_state_manifest"])
    state_index = {str(row["state_id"]): row for row in read_jsonl(sources["phase68_exact_state_index"])}
    feature_index = {str(row["state_id"]): row for row in read_jsonl(sources["phase72_feature_index"])}
    route_store = {str(row["route_id"]): row for row in read_jsonl(sources["global_route_store"])}
    train_samples = {str(row["uid"]): row for row in read_jsonl(sources["train_samples"])}
    if len(exact_rows) != 21071 or len(state_index) != 21071 or len(feature_index) != 21071:
        raise RuntimeError("exact-state source coverage differs from 21,071")
    if len(route_store) != 2519 or len(train_samples) != 569:
        raise RuntimeError("route/sample source coverage differs from Phase 72")

    representation = read_json(sources["phase72_representation_manifest"])
    shard_hash = {str(row["path"]): str(row["sha256"]) for row in representation["shards"]}
    labels = Counter(str(row["state_label"]) for row in exact_rows)
    if labels != Counter({"KEEP_REQUIRED": 18438, "INTERVENE_REQUIRED": 1657, "MIXED": 976}):
        raise RuntimeError(f"exact-state label census differs: {labels}")
    selected = deterministic_state_sample(
        exact_rows,
        targets=config["sampling"]["targets"],
        seed=int(config["seed"]),
        uid_cap=int(config["sampling"]["uid_cap"]),
    )
    states_by_representative_route: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in exact_rows:
        states_by_representative_route[str(row["representative_route_id"])].append(row)
    for route_id in states_by_representative_route:
        states_by_representative_route[route_id].sort(
            key=lambda item: (int(item["layer"]), str(item["state_id"]))
        )

    enriched = []
    for row in selected:
        state_id = str(row["state_id"])
        uid = str(row["uid"])
        if state_id not in state_index or state_id not in feature_index:
            raise RuntimeError(f"selected state lacks exact/feature index: {state_id}")
        source_state = state_index[state_id]
        feature = feature_index[state_id]
        route_id = str(row["representative_route_id"])
        if route_id not in route_store or uid not in train_samples:
            raise RuntimeError(f"selected state lacks route/sample provenance: {state_id}")
        route = route_store[route_id]
        actions = list(source_state["representative_actions"])
        layer = int(row["layer"])
        if (
            list(row["prefix_actions"]) != actions[:layer]
            or str(route["uid"]) != uid
            or list(route["actions"]) != actions
            or actions[layer] not in row["observed_valid_actions"]
            or str(feature["representative_route_id"]) != route_id
        ):
            raise RuntimeError(f"selected exact-state provenance differs: {state_id}")
        relative_shard = str(feature["feature_shard"])
        missing = list(unobserved_actions(row["observed_valid_actions"]))
        item = {
            "schema_version": "treatment_label_completeness_frozen_state_v1",
            "state_id": state_id,
            "uid": uid,
            "dataset": row["dataset"],
            "source_regime": row["source_regime"],
            "image_group_id": row["image_group_id"],
            "image_content_sha256": row["image_content_sha256"],
            "layer": layer,
            "layer_bin": row["layer_bin"],
            "prefix_actions": list(row["prefix_actions"]),
            "prefix_sha256": row["prefix_sha256"],
            "representative_actions": actions,
            "representative_route_id": route_id,
            "representative_expected_generated_token_ids": list(route["generated_token_ids"]),
            "representative_expected_generated_answer": route["generated_answer"],
            "representative_expected_lmms_metric": route["lmms_metric"],
            "representative_expected_lmms_score": float(route["lmms_score"]),
            "route_ids": list(row["route_ids"]),
            "route_sources": list(row["route_sources"]),
            "route_source_signature": row["route_source_signature"],
            "old_label": row["state_label"],
            "old_observed_valid_actions": list(row["observed_valid_actions"]),
            "unobserved_actions": missing,
            "phase72_feature_reference": {
                "shard": relative_shard,
                "shard_sha256": shard_hash[relative_shard],
                "tensor_row": int(feature["tensor_row"]),
                "feature_schema_sha256": representation["feature_schema_sha256"],
                "contract_sha256": phase72["contract_sha256"],
            },
            "phase72_extraction_batch": {
                "representative_route_id": route_id,
                "layers": [
                    int(item["layer"]) for item in states_by_representative_route[route_id]
                ],
                "state_ids": [
                    str(item["state_id"]) for item in states_by_representative_route[route_id]
                ],
            },
            "sample": _sample_payload(train_samples[uid]),
        }
        item["estimated_max_terminal_evaluations"] = _state_cost(item)
        enriched.append(item)

    enriched = _assign_state_workers(enriched, int(config["world_size"]))
    smoke = _assign_state_workers(
        _select_smoke(enriched, int(config["smoke"]["states"]), int(config["seed"])),
        int(config["world_size"]),
    )
    atomic_jsonl(output_root / "sampling/frozen_1200_state_manifest.jsonl", enriched)
    atomic_jsonl(output_root / "work/full_manifest.jsonl", enriched)
    atomic_jsonl(output_root / "work/smoke_manifest.jsonl", smoke)
    atomic_csv(
        output_root / "sampling/full_state_census.csv",
        [
            {
                "state_id": row["state_id"], "uid": row["uid"], "dataset": row["dataset"],
                "source_regime": row["source_regime"], "layer": row["layer"],
                "layer_bin": row["layer_bin"], "route_source_signature": row["route_source_signature"],
                "old_label": row["state_label"], "old_action_count": len(row["observed_valid_actions"]),
            }
            for row in exact_rows
        ],
    )
    strata = []
    for label in config["sampling"]["targets"]:
        for field in config["sampling"]["dimensions"]:
            full_counts = Counter(str(row[field]) for row in exact_rows if row["state_label"] == label)
            selected_counts = Counter(str(row[field]) for row in enriched if row["old_label"] == label)
            for value in sorted(full_counts):
                strata.append(
                    {
                        "old_label": label, "dimension": field, "value": value,
                        "census_states": full_counts[value], "selected_states": selected_counts[value],
                        "census_fraction": full_counts[value] / labels[label],
                        "selected_fraction": selected_counts[value] / int(config["sampling"]["targets"][label]),
                    }
                )
    atomic_csv(output_root / "sampling/sampling_strata.csv", strata)
    uid_counts = Counter(str(row["uid"]) for row in enriched)
    atomic_csv(
        output_root / "sampling/uid_coverage.csv",
        [
            {
                "uid": uid, "states": count,
                "image_group_id": next(row["image_group_id"] for row in enriched if row["uid"] == uid),
                "worker_rank": next(row["worker_rank"] for row in enriched if row["uid"] == uid),
            }
            for uid, count in sorted(uid_counts.items())
        ],
    )

    internal = {
        relative: file_sha256(output_root / relative)
        for relative in (
            "sampling/frozen_1200_state_manifest.jsonl", "sampling/full_state_census.csv",
            "sampling/sampling_strata.csv", "sampling/uid_coverage.csv",
            "work/full_manifest.jsonl", "work/smoke_manifest.jsonl",
        )
    }
    source_hashes = {name: file_sha256(path) for name, path in sources.items()}
    bound_hashes = {path: file_sha256(resolve_path(path)) for path in BOUND_CODE}
    runtime = _runtime_metadata()
    contract: dict[str, Any] = {
        "schema_version": "treatment_label_completeness_audit_contract_v1",
        "created_at": utc_now(),
        "static_config": config,
        "parent_contracts": {
            "phase72": phase72["contract_sha256"],
            "phase65": phase65["contract_sha256"],
            "phase66": phase66["contract_sha256"],
        },
        "source_sha256": source_hashes,
        "bound_code_sha256": bound_hashes,
        "internal_manifest_sha256": internal,
        "model_snapshot_sha256": phase72["model_snapshot_sha256"],
        "runtime": runtime,
        "git": {
            "commit": command_output(("git", "rev-parse", "HEAD")),
            "branch": command_output(("git", "branch", "--show-current")),
            "worktree_status_at_freeze": command_output(("git", "status", "--short")),
        },
        "population": {
            "census_states": len(exact_rows),
            "selected_states": len(enriched),
            "selected_uids": len(uid_counts),
            "selected_image_groups": len({str(row["image_group_id"]) for row in enriched}),
            "label_counts": dict(Counter(str(row["old_label"]) for row in enriched)),
            "max_states_per_uid": max(uid_counts.values()),
            "unobserved_action_branches": sum(len(row["unobserved_actions"]) for row in enriched),
            "estimated_max_terminal_evaluations": sum(int(row["estimated_max_terminal_evaluations"]) for row in enriched),
            "worker_estimated_max_terminal_evaluations": {
                str(rank): sum(
                    int(row["estimated_max_terminal_evaluations"])
                    for row in enriched if int(row["worker_rank"]) == rank
                )
                for rank in range(int(config["world_size"]))
            },
        },
        "prospective_interpretation": {
            "large_invalidation": "point estimate >=0.20 and UID-bootstrap 95% CI lower bound >=0.10",
            "mostly_stable": "both KEEP and INTERVENE UID-bootstrap 95% CI upper bounds <=0.10",
            "material_probe_improvement": "RW AUROC gain >=0.05 over Phase 72 and audited RW AUROC >=0.60",
            "mcts_saturated": "at least 10 MCTS discoveries and <=10% first discovered in iterations 151-200",
            "mcts_unsaturated": "at least 10 MCTS discoveries and >10% first discovered in iterations 151-200",
            "mcts_inconclusive": "fewer than 10 MCTS discoveries",
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_protocol.json", contract)
    _atomic_text(
        output_root / "protocol.md",
        f"""# Large-scale treatment-label completeness audit protocol

- Contract: `{contract['contract_sha256']}`
- Frozen census: 21,071 exact routed prefix states from 569 UIDs.
- Frozen audit sample: 1,200 states ({dict(Counter(row['old_label'] for row in enriched))}); maximum four states per UID.
- Sampling inputs: old label, dataset, source regime, layer bin, route-source signature, UID/group identity, and deterministic hash only.
- Search per unobserved first action: forced action + FULL suffix; exhaustive one later non-FULL; then sequential MCTS capped at 200 from the actual post-action state.
- Every discovery and one existing successful continuation require exact replay. Errors or parity failures are quarantined.
- Labels remain bounded-search `AUDITED_KEEP`, `AUDITED_INTERVENE`, or `AUDITED_MIXED`; no structural necessity claim is permitted.
- Prospective interpretation: {json.dumps(contract['prospective_interpretation'], sort_keys=True)}
- Probe recheck: frozen Phase-72 M0/M1/R/W/RW families, five UID/image-group-disjoint folds, audited MIXED excluded.
- Scope stops after audit, replay, label revision, probe recheck, and recommendation. No Stage-2 retraining or external evaluation.
""",
    )
    print(json.dumps({"prepared": True, "contract_sha256": contract["contract_sha256"], **contract["population"]}, sort_keys=True))


def load_contract(config_path: Path, *, verify_model: bool = False) -> tuple[dict[str, Any], Path]:
    config = read_json(resolve_path(config_path))
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if canonical_hash(contract) != contract.get("contract_sha256"):
        raise RuntimeError("frozen completeness-audit contract hash is invalid")
    if contract["static_config"] != config:
        raise RuntimeError("active config differs from frozen completeness-audit contract")
    if command_output(("git", "rev-parse", "HEAD")) != contract["git"]["commit"]:
        raise RuntimeError("git commit differs from frozen completeness-audit contract")
    if command_output(("git", "branch", "--show-current")) != contract["git"]["branch"]:
        raise RuntimeError("git branch differs from frozen completeness-audit contract")
    if command_output(("git", "status", "--short")) != contract["git"]["worktree_status_at_freeze"]:
        raise RuntimeError("worktree status differs from frozen completeness-audit contract")
    if _runtime_metadata() != contract["runtime"]:
        raise RuntimeError("runtime differs from frozen completeness-audit contract")
    for name, expected in contract["source_sha256"].items():
        if file_sha256(resolve_path(config["sources"][name])) != str(expected):
            raise RuntimeError(f"source hash mismatch: {name}")
    for relative, expected in contract["bound_code_sha256"].items():
        if file_sha256(resolve_path(relative)) != str(expected):
            raise RuntimeError(f"bound-code hash mismatch: {relative}")
    for relative, expected in contract["internal_manifest_sha256"].items():
        if file_sha256(output_root / relative) != str(expected):
            raise RuntimeError(f"internal-manifest hash mismatch: {relative}")
    if verify_model:
        snapshot = resolve_path(config["model"]["snapshot_path"])
        actual = sorted(path.name for path in snapshot.iterdir() if path.is_file())
        if actual != sorted(contract["model_snapshot_sha256"]):
            raise RuntimeError("model snapshot inventory differs from frozen contract")
        for name, expected in contract["model_snapshot_sha256"].items():
            if file_sha256(snapshot / name) != str(expected):
                raise RuntimeError(f"model snapshot file hash mismatch: {name}")
    return contract, output_root


def _phase72_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return read_json(resolve_path(config["sources"]["phase72_contract"]))


def _load_selected_feature_map(
    contract: Mapping[str, Any], manifest: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, np.ndarray | float]]:
    config = contract["static_config"]
    phase72 = _phase72_contract(config)
    root = resolve_path(config["sources"]["phase72_contract"]).parent
    all_features = _load_feature_map(phase72, root)
    selected = {str(row["state_id"]): all_features[str(row["state_id"])] for row in manifest}
    if len(selected) != len(manifest):
        raise RuntimeError("selected Phase-72 feature map is incomplete")
    return selected


def _exact_state_representation_check(
    *,
    baseline: Any,
    layer: int,
    router: torch.nn.Module,
    expected: Mapping[str, np.ndarray | float],
    extraction_batch: Mapping[str, Any],
) -> dict[str, Any]:
    layers = [int(value) for value in extraction_batch["layers"]]
    state_ids = list(map(str, extraction_batch["state_ids"]))
    if len(layers) != len(state_ids) or int(layer) not in layers:
        raise RuntimeError("Phase-72 extraction batch metadata is invalid")
    batch_index = layers.index(int(layer))
    text = torch.cat([baseline.pre_layer_states[value][0] for value in layers], dim=0)
    visual = torch.cat([baseline.pre_layer_states[value][1] for value in layers], dim=0)
    text_mask = baseline.inputs.text_valid_mask.expand(len(layers), -1)
    visual_mask = baseline.inputs.visual_valid_mask.expand(len(layers), -1)
    normal_logits = router(text, visual, text_mask=text_mask, visual_mask=visual_mask)
    read, write, logits = exact_router_representations(
        router, text, visual, text_mask=text_mask, visual_mask=visual_mask
    )
    read = read[batch_index : batch_index + 1]
    write = write[batch_index : batch_index + 1]
    logits = logits[batch_index : batch_index + 1]
    normal_selected = normal_logits[batch_index : batch_index + 1]
    expected_read = torch.from_numpy(np.asarray(expected["z_R"], dtype=np.float32)).to(read.device)[None]
    expected_write = torch.from_numpy(np.asarray(expected["z_W"], dtype=np.float32)).to(write.device)[None]
    expected_logits = torch.from_numpy(np.asarray(expected["action_logits"], dtype=np.float32)).to(logits.device)[None]
    checks = {
        "router_internal_logit_parity": bool(torch.equal(normal_selected, logits)),
        "phase72_z_R_exact": bool(torch.equal(read.float(), expected_read)),
        "phase72_z_W_exact": bool(torch.equal(write.float(), expected_write)),
        "phase72_action_logits_exact": bool(torch.equal(logits.float(), expected_logits)),
        "finite": bool(torch.isfinite(read).all() and torch.isfinite(write).all() and torch.isfinite(logits).all()),
        "phase72_extraction_batch_size": len(layers),
        "phase72_extraction_batch_index": batch_index,
    }
    checks["passed"] = all(
        checks[key]
        for key in (
            "router_internal_logit_parity",
            "phase72_z_R_exact",
            "phase72_z_W_exact",
            "phase72_action_logits_exact",
            "finite",
        )
    )
    return checks


def _generated_record(generated: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "generated_token_ids": list(generated["generated_ids"]),
        "generated_answer": str(generated["generated_answer"]),
        "lmms_metric": str(generated["lmms_metric"]),
        "lmms_score": float(generated["lmms_score"]),
        "correct": bool(generated["correct"]),
    }


def _same_generation(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        list(left["generated_token_ids"]) == list(right["generated_token_ids"])
        and str(left["generated_answer"]) == str(right["generated_answer"])
        and str(left["lmms_metric"]) == str(right["lmms_metric"])
        and float(left["lmms_score"]) == float(right["lmms_score"])
        and bool(left["correct"]) == bool(right["correct"])
    )


def _route_row(
    *,
    actions: Sequence[str],
    generated: Mapping[str, Any],
    stage: str,
    first_action: str,
    later_layer: int | None = None,
    later_action: str | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    return {
        "actions": list(actions),
        "route_key": "|".join(actions),
        "search_stage": stage,
        "first_action": first_action,
        "later_layer": later_layer,
        "later_action": later_action,
        **_generated_record(generated),
        "elapsed_seconds": elapsed_seconds,
    }


def _run_state_audit(
    *,
    row: Mapping[str, Any],
    sample_inputs: Mapping[str, Any],
    prepared: Any,
    wrapped: Any,
    processor: Any,
    router: torch.nn.Module,
    expected_feature: Mapping[str, np.ndarray | float],
    config: Mapping[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    state_id = str(row["state_id"])
    layer = int(row["layer"])
    sample = row["sample"]
    representative_actions = list(row["representative_actions"])
    prefix = tuple(row["prefix_actions"])
    started = time.monotonic()
    representative = capture_four_action_route(
        wrapped,
        sample_inputs,
        representative_actions,
        prepared_inputs=prepared,
        use_cache=True,
        native_full_rows=bool(config["search"]["native_full_row_dispatch"]),
    )
    representative_generated = _generated_record(
        _generate_output(processor, wrapped, representative, sample_inputs, sample)
    )
    expected_representative = {
        "generated_token_ids": row["representative_expected_generated_token_ids"],
        "generated_answer": row["representative_expected_generated_answer"],
        "lmms_metric": row["representative_expected_lmms_metric"],
        "lmms_score": row["representative_expected_lmms_score"],
        "correct": True,
    }
    existing_replay = {
        "kind": "existing_success",
        "state_id": state_id,
        "route_id": row["representative_route_id"],
        "action": representative_actions[layer],
        "exact_token_parity": _same_generation(representative_generated, expected_representative),
        **representative_generated,
    }
    if not existing_replay["exact_token_parity"] or not existing_replay["correct"]:
        raise RuntimeError(f"existing successful continuation replay failed: {state_id}")
    if tuple(representative.layer_actions[:layer]) != prefix:
        raise RuntimeError(f"representative route prefix differs: {state_id}")
    representation_check = _exact_state_representation_check(
        baseline=representative,
        layer=layer,
        router=router,
        expected=expected_feature,
        extraction_batch=row["phase72_extraction_batch"],
    )
    if not representation_check["passed"]:
        raise RuntimeError(f"Phase-72 exact-state representation parity failed: {state_id}")

    direct_rows: list[dict[str, Any]] = []
    single_rows: list[dict[str, Any]] = []
    mcts_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = [existing_replay]
    branch_rows: list[dict[str, Any]] = []
    terminal_evaluations = 1
    audit_actions = list(row["unobserved_actions"])
    for first_action in audit_actions:
        branch_started = time.monotonic()
        forced_actions = [*prefix, first_action, *(["FULL"] * (27 - layer))]
        forced = capture_four_action_suffix_from_route_baseline(
            wrapped,
            representative,
            layer,
            forced_actions[layer:],
            expected_prefix=prefix,
        )
        direct_generated = _generate_output(processor, wrapped, forced, sample_inputs, sample)
        direct = _route_row(
            actions=forced_actions,
            generated=direct_generated,
            stage="direct_suffix",
            first_action=first_action,
            elapsed_seconds=time.monotonic() - branch_started,
        )
        direct_rows.append(direct)
        terminal_evaluations += 1
        success = direct if direct["correct"] else None
        first_success_stage = "direct_suffix" if success else None
        first_success_iteration = None
        if (success is None or smoke) and layer < 27:
            later_pairs = [
                (later_layer, later_action)
                for later_layer in range(layer + 1, 28)
                for later_action in ACTIONS[1:]
            ]
            if smoke:
                later_pairs = later_pairs[:1]
            for later_layer, later_action in later_pairs:
                route_actions = list(forced_actions)
                route_actions[later_layer] = later_action
                route_started = time.monotonic()
                output = capture_four_action_suffix_from_route_baseline(
                    wrapped,
                    forced,
                    later_layer,
                    route_actions[later_layer:],
                    expected_prefix=tuple(route_actions[:later_layer]),
                )
                generated = _generate_output(processor, wrapped, output, sample_inputs, sample)
                attempted = _route_row(
                    actions=route_actions,
                    generated=generated,
                    stage="single_later",
                    first_action=first_action,
                    later_layer=later_layer,
                    later_action=later_action,
                    elapsed_seconds=time.monotonic() - route_started,
                )
                single_rows.append(attempted)
                terminal_evaluations += 1
                del output
                if attempted["correct"] and success is None:
                    success = attempted
                    first_success_stage = "single_later"
                    break

        mcts_result = None
        if (success is None or smoke) and layer < 27:
            route_cache: dict[str, dict[str, Any]] = {}

            def evaluate_mcts(actions: Sequence[str]) -> bool:
                nonlocal terminal_evaluations
                full_actions = [*prefix, first_action, *list(actions[layer + 1 :])]
                key = "|".join(full_actions)
                if key not in route_cache:
                    route_started = time.monotonic()
                    output = capture_four_action_suffix_from_route_baseline(
                        wrapped,
                        forced,
                        layer + 1,
                        full_actions[layer + 1 :],
                        expected_prefix=tuple(full_actions[: layer + 1]),
                    )
                    generated = _generate_output(processor, wrapped, output, sample_inputs, sample)
                    route_cache[key] = _route_row(
                        actions=full_actions,
                        generated=generated,
                        stage="mcts",
                        first_action=first_action,
                        elapsed_seconds=time.monotonic() - route_started,
                    )
                    terminal_evaluations += 1
                    del output
                return bool(route_cache[key]["correct"])

            maximum = (
                int(config["smoke"]["mcts_iterations_per_branch"])
                if smoke
                else int(config["search"]["mcts_maximum_iterations"])
            )
            mcts_result = run_sequential_mcts(
                uid=f"{state_id}:{first_action}",
                start_layer=layer + 1,
                seed=int(config["seed"]),
                max_iterations=maximum,
                extra_iterations_after_success=(
                    maximum if smoke else int(config["search"]["mcts_extra_iterations_after_first_success"])
                ),
                evaluate=evaluate_mcts,
                exploration_constant=float(config["search"]["mcts_exploration_constant"]),
                rollout_cardinalities=config["search"]["mcts_rollout_cardinalities"],
                retain_successes=int(config["search"]["mcts_retained_successful_routes"]),
            )
            compact_search = []
            for search_row in mcts_result["search_rows"]:
                full_actions = [*prefix, first_action, *list(search_row["actions"][layer + 1 :])]
                compact_search.append(
                    {
                        "iteration": int(search_row["iteration"]),
                        "route_key": "|".join(full_actions),
                        "reward": int(search_row["reward"]),
                        "terminal_cache_hit": bool(search_row["terminal_cache_hit"]),
                        "non_full_count": sum(action != "FULL" for action in full_actions),
                    }
                )
            mcts_row = {
                "state_id": state_id,
                "uid": row["uid"],
                "first_action": first_action,
                "root_layer": layer + 1,
                "iterations": int(mcts_result["iterations"]),
                "unique_terminal_routes": int(mcts_result["unique_terminal_routes"]),
                "first_success_iteration": mcts_result["first_success_iteration"],
                "search_rows": compact_search,
            }
            mcts_rows.append(mcts_row)
            if success is None and mcts_result["successful_routes"]:
                successful = mcts_result["successful_routes"][0]
                full_actions = [*prefix, first_action, *list(successful["actions"][layer + 1 :])]
                success = route_cache["|".join(full_actions)]
                first_success_stage = "mcts"
                first_success_iteration = int(mcts_result["first_success_iteration"])

        replay_valid = None
        if success is not None:
            replay_started = time.monotonic()
            success_actions = list(success["actions"])
            replay = capture_four_action_suffix_from_route_baseline(
                wrapped,
                representative,
                layer,
                success_actions[layer:],
                expected_prefix=prefix,
            )
            replay_generated = _generated_record(
                _generate_output(processor, wrapped, replay, sample_inputs, sample)
            )
            terminal_evaluations += 1
            replay_valid = bool(replay_generated["correct"] and _same_generation(success, replay_generated))
            replay_rows.append(
                {
                    "kind": "new_discovery",
                    "state_id": state_id,
                    "first_action": first_action,
                    "search_stage": first_success_stage,
                    "first_success_iteration": first_success_iteration,
                    "route_key": success["route_key"],
                    "exact_token_parity": replay_valid,
                    "elapsed_seconds": time.monotonic() - replay_started,
                    **replay_generated,
                }
            )
            del replay
            if not replay_valid:
                raise RuntimeError(f"new action discovery replay failed: {state_id}/{first_action}")
        branch_rows.append(
            {
                "first_action": first_action,
                "searched": True,
                "success": success is not None,
                "replay_valid": replay_valid,
                "discovery_stage": first_success_stage,
                "first_success_iteration": first_success_iteration,
                "successful_route_key": None if success is None else success["route_key"],
                "elapsed_seconds": time.monotonic() - branch_started,
            }
        )
        del forced

    if smoke:
        complete = capture_four_action_route(
            wrapped,
            sample_inputs,
            direct_rows[0]["actions"],
            prepared_inputs=prepared,
            use_cache=True,
            native_full_rows=bool(config["search"]["native_full_row_dispatch"]),
        )
        complete_generated = _generated_record(
            _generate_output(processor, wrapped, complete, sample_inputs, sample)
        )
        full_route_parity = _same_generation(direct_rows[0], complete_generated)
        if not full_route_parity:
            raise RuntimeError(f"cached branch/full route parity failed: {state_id}")
        del complete
    else:
        full_route_parity = None
    del representative
    return {
        "schema_version": "treatment_label_completeness_state_audit_v1",
        "state_id": state_id,
        "uid": row["uid"],
        "dataset": row["dataset"],
        "source_regime": row["source_regime"],
        "image_group_id": row["image_group_id"],
        "layer": layer,
        "layer_bin": row["layer_bin"],
        "route_sources": row["route_sources"],
        "route_source_signature": row["route_source_signature"],
        "old_label": row["old_label"],
        "old_observed_valid_actions": row["old_observed_valid_actions"],
        "searched_unobserved_actions": audit_actions,
        "existing_success_replay": existing_replay,
        "exact_state_representation_check": representation_check,
        "direct_suffix_results": direct_rows,
        "single_later_results": single_rows,
        "mcts_results": mcts_rows,
        "replay_validation": replay_rows,
        "action_branches": branch_rows,
        "full_route_parity": full_route_parity,
        "terminal_evaluations": terminal_evaluations,
        "elapsed_seconds": time.monotonic() - started,
        "quarantined": False,
        "passed": True,
    }


def worker(
    config_path: Path,
    *,
    mode: str,
    rank: int,
    world_size: int,
    resume: bool,
) -> None:
    started = time.monotonic()
    contract, output_root = load_contract(config_path, verify_model=True)
    config = contract["static_config"]
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    if world_size != int(config["world_size"]) or rank not in range(world_size):
        raise RuntimeError("worker rank/world size differs from frozen contract")
    if torch.cuda.device_count() != world_size:
        raise RuntimeError("all four visible GPUs are required")
    if mode == "full":
        smoke_completion = read_json(output_root / "work/smoke/completion.json")
        if not smoke_completion.get("passed") or smoke_completion.get("contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError("full audit requires a passing contract-bound smoke")
    manifest = read_jsonl(output_root / f"work/{mode}_manifest.jsonl")
    rows = [row for row in manifest if int(row["worker_rank"]) == rank]
    worker_root = output_root / f"work/{mode}/rank{rank:02d}"
    complete_path = worker_root / "complete.json"
    if complete_path.exists():
        raise RuntimeError(f"worker already complete: {complete_path}")
    sample_dir = worker_root / "states"
    existing = sorted(sample_dir.glob("*.json")) if sample_dir.exists() else []
    if existing and not resume:
        raise RuntimeError(f"partial output requires --resume: rank {rank}")
    completed: dict[str, dict[str, Any]] = {}
    expected_ids = {str(row["state_id"]) for row in rows}
    for path in existing:
        saved = read_json(path)
        state_id = str(saved["state_id"])
        if (
            saved.get("contract_sha256") != contract["contract_sha256"]
            or state_id not in expected_ids
            or state_id in completed
            or not (saved.get("passed") or saved.get("quarantined"))
        ):
            raise RuntimeError(f"incompatible resume state: {path}")
        completed[state_id] = saved

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    phase66 = read_json(resolve_path(config["sources"]["phase66_contract"]))
    configure_dense_determinism(int(config["seed"]) + rank, phase66["static_config"]["backend_settings"])
    processor, base, wrapped = _load_model(config, device)
    phase72 = _phase72_contract(config)
    router = _load_frozen_router(phase72, device)
    feature_map = _load_selected_feature_map(contract, manifest)

    by_uid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_uid[str(row["uid"])].append(row)
    completed_count = len(completed)
    quarantined_count = sum(bool(row.get("quarantined")) for row in completed.values())
    for uid in sorted(by_uid):
        pending = [row for row in by_uid[uid] if str(row["state_id"]) not in completed]
        if not pending:
            continue
        sample = pending[0]["sample"]
        inputs, input_metadata = build_dense_inputs(processor, sample, device)
        if input_metadata["consumed_image_sha256"] != sample["image_content_sha256"]:
            raise RuntimeError(f"consumed image hash differs for {uid}")
        prepared = build_binary_inputs(wrapped, inputs)
        for row in sorted(pending, key=lambda item: str(item["state_id"])):
            state_id = str(row["state_id"])
            path = sample_dir / f"{_safe_id(state_id)}.json"
            try:
                result = _run_state_audit(
                    row=row,
                    sample_inputs=inputs,
                    prepared=prepared,
                    wrapped=wrapped,
                    processor=processor,
                    router=router,
                    expected_feature=feature_map[state_id],
                    config=config,
                    smoke=mode == "smoke",
                )
                result.update(
                    {
                        "contract_sha256": contract["contract_sha256"],
                        "worker_rank": rank,
                        "mode": mode,
                        "consumed_image_sha256": input_metadata["consumed_image_sha256"],
                        "completed_at": utc_now(),
                    }
                )
            except Exception as exc:
                result = {
                    "schema_version": "treatment_label_completeness_state_audit_v1",
                    "contract_sha256": contract["contract_sha256"],
                    "state_id": state_id,
                    "uid": uid,
                    "dataset": row["dataset"],
                    "source_regime": row["source_regime"],
                    "layer": row["layer"],
                    "old_label": row["old_label"],
                    "worker_rank": rank,
                    "mode": mode,
                    "passed": False,
                    "quarantined": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "completed_at": utc_now(),
                }
                quarantined_count += 1
            atomic_json(path, result)
            completed[state_id] = result
            completed_count += 1
            print(
                json.dumps(
                    {
                        "mode": mode, "rank": rank, "state_id": state_id,
                        "completed": completed_count, "total": len(rows),
                        "quarantined": quarantined_count,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del inputs, prepared
        torch.cuda.empty_cache()
    actual_ids = set(completed)
    if actual_ids != expected_ids:
        raise RuntimeError(f"worker state coverage differs: missing={len(expected_ids - actual_ids)} extra={len(actual_ids - expected_ids)}")
    atomic_json(
        complete_path,
        {
            "schema_version": "treatment_label_completeness_worker_complete_v1",
            "contract_sha256": contract["contract_sha256"],
            "mode": mode,
            "worker_rank": rank,
            "passed": True,
            "states": len(rows),
            "quarantined_states": quarantined_count,
            "elapsed_seconds": time.monotonic() - started,
            "completed_at": utc_now(),
        },
    )
    print(json.dumps(read_json(complete_path), sort_keys=True), flush=True)


def _collect_workers(output_root: Path, mode: str, contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for rank in range(int(contract["static_config"]["world_size"])):
        complete = read_json(output_root / f"work/{mode}/rank{rank:02d}/complete.json")
        if not complete.get("passed") or complete.get("contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError(f"worker completion invalid: {mode}/{rank}")
        paths = sorted((output_root / f"work/{mode}/rank{rank:02d}/states").glob("*.json"))
        if len(paths) != int(complete["states"]):
            raise RuntimeError(f"worker output count differs: {mode}/{rank}")
        rows.extend(read_json(path) for path in paths)
    expected = read_jsonl(output_root / f"work/{mode}_manifest.jsonl")
    expected_ids = Counter(str(row["state_id"]) for row in expected)
    actual_ids = Counter(str(row["state_id"]) for row in rows)
    if expected_ids != actual_ids:
        raise RuntimeError(f"global {mode} state coverage differs")
    if any(row.get("contract_sha256") != contract["contract_sha256"] for row in rows):
        raise RuntimeError(f"global {mode} contains incompatible records")
    return sorted(rows, key=lambda row: str(row["state_id"]))


def finalize_smoke(config_path: Path) -> None:
    contract, output_root = load_contract(config_path, verify_model=False)
    rows = _collect_workers(output_root, "smoke", contract)
    failures = [row for row in rows if not row.get("passed") or row.get("quarantined")]
    checks = {
        "states": len(rows),
        "quarantined": len(failures),
        "existing_success_replay_exact": all(
            row.get("existing_success_replay", {}).get("exact_token_parity") for row in rows if row.get("passed")
        ),
        "phase72_representation_exact": all(
            row.get("exact_state_representation_check", {}).get("passed") for row in rows if row.get("passed")
        ),
        "forced_action_direct_executed": all(row.get("direct_suffix_results") for row in rows if row.get("passed")),
        "single_later_executed": all(row.get("single_later_results") for row in rows if row.get("passed")),
        "mcts_root_after_forced_action_executed": all(row.get("mcts_results") for row in rows if row.get("passed")),
        "cached_branch_full_route_parity": all(row.get("full_route_parity") for row in rows if row.get("passed")),
        "all_four_forced_actions_executed": {
            str(branch["first_action"])
            for row in rows if row.get("passed")
            for branch in row.get("action_branches", [])
        }
        == set(ACTIONS),
    }
    passed = not failures and all(value for key, value in checks.items() if key not in {"states", "quarantined"})
    report = {
        "schema_version": "treatment_label_completeness_smoke_completion_v1",
        "contract_sha256": contract["contract_sha256"],
        "passed": passed,
        "checks": checks,
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "work/smoke/completion.json", report)
    atomic_jsonl(output_root / "search/smoke_results.jsonl", rows)
    print(json.dumps(report, sort_keys=True))
    if not passed:
        raise RuntimeError("treatment-label completeness smoke failed")


def _audit_label_row(row: Mapping[str, Any]) -> dict[str, Any]:
    old_actions = set(map(str, row["old_observed_valid_actions"]))
    discovered = {
        str(branch["first_action"])
        for branch in row["action_branches"]
        if bool(branch["success"]) and bool(branch["replay_valid"])
    }
    audited_actions = [action for action in ACTIONS if action in old_actions | discovered]
    audited_label = classify_audited_actions(audited_actions)
    old_label = str(row["old_label"])
    invalidated = (
        old_label == "KEEP_REQUIRED" and audited_label == AUDITED_MIXED
    ) or (
        old_label == "INTERVENE_REQUIRED" and audited_label == AUDITED_MIXED
    )
    return {
        "schema_version": "treatment_label_completeness_audited_label_v1",
        "contract_sha256": row["contract_sha256"],
        "state_id": row["state_id"],
        "uid": row["uid"],
        "dataset": row["dataset"],
        "source_regime": row["source_regime"],
        "image_group_id": row["image_group_id"],
        "layer": row["layer"],
        "layer_bin": row["layer_bin"],
        "route_sources": row["route_sources"],
        "route_source_signature": row["route_source_signature"],
        "old_label": old_label,
        "old_observed_valid_actions": [action for action in ACTIONS if action in old_actions],
        "new_replay_valid_actions": [action for action in ACTIONS if action in discovered],
        "audited_observed_valid_actions": audited_actions,
        "old_action_count": len(old_actions),
        "audited_action_count": len(audited_actions),
        "new_action_count": len(discovered),
        "any_new_action": bool(discovered),
        "invalidated_to_mixed": invalidated,
        "audited_label": audited_label,
        "bounded_search_statement": "no additional successful first action found under the frozen bounded audit" if not discovered else "at least one additional first action found and replay validated",
    }


def _rate_row(scope_type: str, scope: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keep = [row for row in rows if row["old_label"] == "KEEP_REQUIRED"]
    intervene = [row for row in rows if row["old_label"] == "INTERVENE_REQUIRED"]
    mixed = [row for row in rows if row["old_label"] == "MIXED"]
    return {
        "scope_type": scope_type,
        "scope": scope,
        "states": len(rows),
        "uids": len({str(row["uid"]) for row in rows}),
        "old_KEEP": len(keep),
        "old_INTERVENE": len(intervene),
        "old_MIXED": len(mixed),
        "KEEP_invalidated": sum(bool(row["invalidated_to_mixed"]) for row in keep),
        "KEEP_invalidation_rate": (
            sum(bool(row["invalidated_to_mixed"]) for row in keep) / len(keep) if keep else None
        ),
        "INTERVENE_invalidated": sum(bool(row["invalidated_to_mixed"]) for row in intervene),
        "INTERVENE_invalidation_rate": (
            sum(bool(row["invalidated_to_mixed"]) for row in intervene) / len(intervene)
            if intervene else None
        ),
        "states_with_new_action": sum(bool(row["any_new_action"]) for row in rows),
        "any_new_action_rate": (
            sum(bool(row["any_new_action"]) for row in rows) / len(rows) if rows else None
        ),
        "mean_old_action_count": float(np.mean([row["old_action_count"] for row in rows])) if rows else None,
        "mean_audited_action_count": float(np.mean([row["audited_action_count"] for row in rows])) if rows else None,
    }


def _uid_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, metric: str, replicates: int, seed: int
) -> dict[str, Any]:
    if metric == "KEEP_invalidation_rate":
        eligible = [row for row in rows if row["old_label"] == "KEEP_REQUIRED"]
        outcome = lambda row: bool(row["invalidated_to_mixed"])
    elif metric == "INTERVENE_invalidation_rate":
        eligible = [row for row in rows if row["old_label"] == "INTERVENE_REQUIRED"]
        outcome = lambda row: bool(row["invalidated_to_mixed"])
    elif metric == "any_new_action_rate":
        eligible = list(rows)
        outcome = lambda row: bool(row["any_new_action"])
    else:
        raise ValueError(f"unsupported bootstrap metric: {metric}")
    by_uid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_uid[str(row["uid"])].append(row)
    uids = sorted(by_uid)
    if not uids:
        return {"metric": metric, "states": 0, "uids": 0, "estimate": None, "ci95_low": None, "ci95_high": None}
    estimate = sum(outcome(row) for row in eligible) / len(eligible)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sampled = rng.choice(uids, size=len(uids), replace=True)
        sampled_rows = [row for uid in sampled for row in by_uid[str(uid)]]
        values.append(sum(outcome(row) for row in sampled_rows) / len(sampled_rows))
    return {
        "metric": metric,
        "states": len(eligible),
        "uids": len(uids),
        "estimate": estimate,
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "replicates": replicates,
        "resampling_unit": "uid",
    }


def _write_audit_figures(
    output_root: Path,
    labels: Sequence[Mapping[str, Any]],
    state_rows: Sequence[Mapping[str, Any]],
) -> None:
    def save_bar(path: str, names: Sequence[str], values: Sequence[float], title: str, ylabel: str) -> None:
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.bar(names, values, color="#4c78a8")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(output_root / f"figures/{path}", dpi=180)
        plt.close(fig)

    overall = _rate_row("overall", "overall", labels)
    save_bar(
        "label_transition_matrix.png",
        ["KEEP stable", "KEEP→MIXED", "INTERVENE stable", "INTERVENE→MIXED", "MIXED"],
        [
            overall["old_KEEP"] - overall["KEEP_invalidated"], overall["KEEP_invalidated"],
            overall["old_INTERVENE"] - overall["INTERVENE_invalidated"], overall["INTERVENE_invalidated"],
            overall["old_MIXED"],
        ],
        "Old to audited bounded-search labels",
        "states",
    )
    action_counts = Counter(
        action for row in labels for action in row["new_replay_valid_actions"]
    )
    save_bar(
        "new_action_discovery_by_type.png", ACTIONS,
        [action_counts[action] for action in ACTIONS], "New replay-valid first actions", "states",
    )
    stage_counts = Counter(
        branch["discovery_stage"]
        for row in state_rows for branch in row["action_branches"] if branch["success"]
    )
    save_bar(
        "discovery_stage_attribution.png", ["direct_suffix", "single_later", "mcts"],
        [stage_counts[name] for name in ("direct_suffix", "single_later", "mcts")],
        "Discovery stage attribution", "successful action branches",
    )
    datasets = sorted({str(row["dataset"]) for row in labels})
    save_bar(
        "invalidation_rate_by_dataset.png", datasets,
        [_rate_row("dataset", dataset, [row for row in labels if row["dataset"] == dataset])["any_new_action_rate"] for dataset in datasets],
        "New-action discovery rate by dataset", "rate",
    )
    sources = sorted({str(row["source_regime"]) for row in labels})
    save_bar(
        "invalidation_rate_by_source.png", sources,
        [_rate_row("source", source, [row for row in labels if row["source_regime"] == source])["any_new_action_rate"] for source in sources],
        "New-action discovery rate by source", "rate",
    )
    bins = ("early_0_8", "middle_9_18", "late_19_27")
    save_bar(
        "invalidation_rate_by_layer_bin.png", bins,
        [_rate_row("layer", value, [row for row in labels if row["layer_bin"] == value])["any_new_action_rate"] for value in bins],
        "New-action discovery rate by layer bin", "rate",
    )
    mcts_iterations = [
        int(branch["first_success_iteration"])
        for row in state_rows for branch in row["action_branches"]
        if branch["discovery_stage"] == "mcts"
    ]
    checkpoints = (50, 100, 150, 200)
    save_bar(
        "mcts_budget_saturation.png", [str(value) for value in checkpoints],
        [sum(iteration <= value for iteration in mcts_iterations) for value in checkpoints],
        "Cumulative MCTS discoveries", "successful action branches",
    )
    route_sources = ("preservation_full", "single", "mcts")
    save_bar(
        "route_source_incompleteness.png", route_sources,
        [
            _rate_row("route_source", source, [row for row in labels if source in row["route_sources"]])["any_new_action_rate"]
            for source in route_sources
        ],
        "New-action discovery by old route-source membership", "rate",
    )


def aggregate(config_path: Path) -> None:
    contract, output_root = load_contract(config_path, verify_model=False)
    config = contract["static_config"]
    state_rows = _collect_workers(output_root, "full", contract)
    quarantined = [row for row in state_rows if row.get("quarantined") or not row.get("passed")]
    clean_states = [row for row in state_rows if row.get("passed") and not row.get("quarantined")]
    labels = [_audit_label_row(row) for row in clean_states]
    if len({str(row["state_id"]) for row in labels}) != len(labels):
        raise RuntimeError("audited labels contain duplicate state IDs")

    atomic_jsonl(output_root / "search/per_state_action_audit.jsonl", state_rows)
    atomic_jsonl(
        output_root / "search/direct_suffix_results.jsonl",
        [
            {"state_id": row["state_id"], "uid": row["uid"], **item}
            for row in clean_states for item in row["direct_suffix_results"]
        ],
    )
    atomic_jsonl(
        output_root / "search/single_later_results.jsonl",
        [
            {"state_id": row["state_id"], "uid": row["uid"], **item}
            for row in clean_states for item in row["single_later_results"]
        ],
    )
    atomic_jsonl(
        output_root / "search/mcts_results.jsonl",
        [
            {"contract_sha256": contract["contract_sha256"], **item}
            for row in clean_states for item in row["mcts_results"]
        ],
    )
    atomic_jsonl(
        output_root / "search/replay_validation.jsonl",
        [
            {"state_id": row["state_id"], "uid": row["uid"], **item}
            for row in clean_states for item in row["replay_validation"]
        ],
    )
    atomic_jsonl(output_root / "search/quarantined_states.jsonl", quarantined)
    atomic_jsonl(output_root / "labels/old_vs_audited_action_sets.jsonl", labels)
    atomic_jsonl(
        output_root / "labels/audited_clean_labels.jsonl",
        [row for row in labels if row["audited_label"] != AUDITED_MIXED],
    )
    atomic_jsonl(
        output_root / "labels/audited_mixed_states.jsonl",
        [row for row in labels if row["audited_label"] == AUDITED_MIXED],
    )

    transitions = Counter((row["old_label"], row["audited_label"]) for row in labels)
    atomic_csv(
        output_root / "labels/label_transition_summary.csv",
        [
            {"old_label": old, "audited_label": audited, "states": count}
            for (old, audited), count in sorted(transitions.items())
        ],
    )
    cardinality = Counter((row["old_action_count"], row["audited_action_count"]) for row in labels)
    atomic_csv(
        output_root / "labels/action_expansion_summary.csv",
        [
            {"old_action_count": old, "audited_action_count": new, "states": count}
            for (old, new), count in sorted(cardinality.items())
        ],
    )

    overall = _rate_row("overall", "overall", labels)
    atomic_csv(output_root / "metrics/overall_completeness.csv", [overall])
    breakdown_specs = (
        ("dataset_breakdown.csv", "dataset", sorted({str(row["dataset"]) for row in labels})),
        ("source_breakdown.csv", "source_regime", sorted({str(row["source_regime"]) for row in labels})),
        ("layer_breakdown.csv", "layer_bin", ["early_0_8", "middle_9_18", "late_19_27"]),
    )
    for filename, field, values in breakdown_specs:
        atomic_csv(
            output_root / f"metrics/{filename}",
            [_rate_row(field, value, [row for row in labels if str(row[field]) == value]) for value in values],
        )
    route_sources = ("preservation_full", "single", "mcts")
    atomic_csv(
        output_root / "metrics/route_source_breakdown.csv",
        [_rate_row("route_source_membership", source, [row for row in labels if source in row["route_sources"]]) for source in route_sources],
    )
    action_discovery = []
    for action in ACTIONS:
        eligible = [row for row in labels if action not in row["old_observed_valid_actions"]]
        found = [row for row in eligible if action in row["new_replay_valid_actions"]]
        action_discovery.append(
            {
                "action": action, "eligible_states": len(eligible), "discovered_states": len(found),
                "discovery_rate": len(found) / len(eligible) if eligible else None,
            }
        )
    atomic_csv(output_root / "metrics/action_discovery_rates.csv", action_discovery)
    stage_counts = Counter(
        branch["discovery_stage"]
        for row in clean_states for branch in row["action_branches"] if branch["success"]
    )
    atomic_csv(
        output_root / "metrics/search_stage_attribution.csv",
        [
            {"search_stage": stage, "successful_action_branches": stage_counts[stage]}
            for stage in ("direct_suffix", "single_later", "mcts")
        ],
    )
    mcts_discovery = [
        int(branch["first_success_iteration"])
        for row in clean_states for branch in row["action_branches"]
        if branch["discovery_stage"] == "mcts"
    ]
    saturation = mcts_budget_saturation(mcts_discovery)
    budget_rows = [
        {
            "budget": budget,
            "cumulative_discoveries": sum(value <= budget for value in mcts_discovery),
            "total_mcts_discoveries": len(mcts_discovery),
            "fraction_of_mcts_discoveries": (
                sum(value <= budget for value in mcts_discovery) / len(mcts_discovery)
                if mcts_discovery else None
            ),
            **({f"saturation_{key}": value for key, value in saturation.items()} if budget == 200 else {}),
        }
        for budget in (50, 100, 150, 200)
    ]
    atomic_csv(output_root / "metrics/budget_saturation.csv", budget_rows)
    bootstrap = [
        _uid_bootstrap(
            labels,
            metric=metric,
            replicates=int(config["decision_thresholds"]["bootstrap_replicates"]),
            seed=int(config["seed"]) + index,
        )
        for index, metric in enumerate(
            ("KEEP_invalidation_rate", "INTERVENE_invalidation_rate", "any_new_action_rate")
        )
    ]
    atomic_csv(output_root / "metrics/uid_bootstrap_ci.csv", bootstrap)
    worker_complete = [
        read_json(output_root / f"work/full/rank{rank:02d}/complete.json")
        for rank in range(int(config["world_size"]))
    ]
    atomic_csv(
        output_root / "metrics/compute_summary.csv",
        [
            {
                "selected_states": len(state_rows),
                "clean_states": len(clean_states),
                "quarantined_states": len(quarantined),
                "terminal_evaluations": sum(int(row.get("terminal_evaluations", 0)) for row in clean_states),
                "worker_wall_hours_sum": sum(float(row["elapsed_seconds"]) for row in worker_complete) / 3600,
                "parallel_wall_hours_max": max(float(row["elapsed_seconds"]) for row in worker_complete) / 3600,
                "gpu_hours": sum(float(row["elapsed_seconds"]) for row in worker_complete) / 3600,
            }
        ],
    )
    _write_audit_figures(output_root, labels, clean_states)
    atomic_json(
        output_root / "work/aggregation_complete.json",
        {
            "schema_version": "treatment_label_completeness_aggregation_complete_v1",
            "contract_sha256": contract["contract_sha256"],
            "passed": True,
            "selected_states": len(state_rows),
            "clean_states": len(clean_states),
            "quarantined_states": len(quarantined),
            "overall": overall,
            "bootstrap": bootstrap,
            "mcts_saturation": saturation,
            "completed_at": utc_now(),
        },
    )
    print(json.dumps(read_json(output_root / "work/aggregation_complete.json"), sort_keys=True))


def _feature_matrix(
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
    if probe == "RW_read_write":
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
    raise ValueError(f"unsupported probe matrix: {probe}")


def _fit_probe_cohort(
    *,
    cohort: str,
    rows: list[dict[str, Any]],
    feature_map: Mapping[str, Mapping[str, np.ndarray | float]],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    folds = assign_group_folds(rows, folds=int(config["probe"]["folds"]), seed=int(config["seed"]))
    fold_by_state = {str(row["state_id"]): int(row["fold"]) for row in folds}
    for row in rows:
        row["fold"] = fold_by_state[str(row["state_id"])]
    labels = np.asarray([int(row["binary_target"]) for row in rows], dtype=np.int64)
    fold_values = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    predictions: dict[str, np.ndarray] = {
        "M0_margin": np.asarray(
            [float(feature_map[str(row["state_id"])]["nonfull_full_margin"]) for row in rows],
            dtype=np.float64,
        )
    }
    fold_metrics = []
    for probe in PROBE_NAMES[1:]:
        matrix = _feature_matrix(rows, feature_map, probe).astype(np.float32, copy=False)
        prediction = np.full(len(rows), np.nan, dtype=np.float64)
        for fold in range(int(config["probe"]["folds"])):
            train_indices = np.flatnonzero(fold_values != fold)
            test_indices = np.flatnonzero(fold_values == fold)
            train_rows = [rows[int(index)] for index in train_indices]
            weights = training_weights(train_rows)
            train_x, test_x, _normalization = standardize_fold(
                matrix[train_indices], matrix[test_indices], weights
            )
            fitted = fit_binary_probe(
                train_x,
                labels[train_indices],
                weights,
                kind="linear",
                seed=int(config["seed"]) + 1000 * fold,
                epochs=int(config["probe"]["linear_epochs"]),
                learning_rate=float(config["probe"]["linear_learning_rate"]),
                weight_decay=float(config["probe"]["linear_weight_decay"]),
                device=device,
            )
            prediction[test_indices] = predict_binary_probe(fitted, test_x, device=device)
            fold_metric = binary_metrics(labels[test_indices], prediction[test_indices])
            fold_metrics.append(
                {
                    "cohort": cohort, "probe": probe, "fold": fold,
                    "states": len(test_indices),
                    "uids": len({str(rows[int(index)]["uid"]) for index in test_indices}),
                    **fold_metric,
                }
            )
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"incomplete OOF predictions: {cohort}/{probe}")
        predictions[probe] = prediction

    summary = []
    high_precision = []
    for probe in PROBE_NAMES:
        score = predictions[probe]
        metrics = binary_metrics(labels, score)
        ops = selective_operating_points(
            labels,
            score,
            coverage_fractions=config["probe"]["coverage_fractions"],
            precision_targets=config["probe"]["precision_targets"],
        )
        by_name = {str(row["operating_point"]): row for row in ops}
        summary.append(
            {
                "cohort": cohort, "probe": probe, "states": len(rows),
                "uids": len({str(row["uid"]) for row in rows}),
                "KEEP": int((labels == 0).sum()), "INTERVENE": int((labels == 1).sum()),
                **metrics,
                "precision_at_5pct": by_name["top_5pct"]["precision"],
                "precision_at_10pct": by_name["top_10pct"]["precision"],
                "precision_at_20pct": by_name["top_20pct"]["precision"],
                "recall_at_90pct_precision": by_name["recall_at_90pct_precision"]["recall"],
                "recall_at_95pct_precision": by_name["recall_at_95pct_precision"]["recall"],
            }
        )
        high_precision.extend({"cohort": cohort, "probe": probe, **row} for row in ops)
    return summary, high_precision, [{"cohort": cohort, **row} for row in folds] + fold_metrics


def probe_recheck(config_path: Path, device_index: int) -> None:
    contract, output_root = load_contract(config_path, verify_model=False)
    aggregation = read_json(output_root / "work/aggregation_complete.json")
    if not aggregation.get("passed") or aggregation.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("probe recheck requires passing label aggregation")
    config = contract["static_config"]
    manifest = read_jsonl(output_root / "sampling/frozen_1200_state_manifest.jsonl")
    by_state = {str(row["state_id"]): row for row in manifest}
    audited = read_jsonl(output_root / "labels/old_vs_audited_action_sets.jsonl")
    audited_clean = []
    for label in audited:
        if label["audited_label"] == AUDITED_MIXED:
            continue
        source = by_state[str(label["state_id"])]
        audited_clean.append(
            {
                **label,
                "binary_target": 0 if label["audited_label"] == AUDITED_KEEP else 1,
                "match_cell": "|".join(
                    (str(label["dataset"]), str(label["source_regime"]), str(label["layer_bin"]), str(label["route_source_signature"]))
                ),
                "phase72_feature_reference": source["phase72_feature_reference"],
            }
        )
    old_clean = [
        {
            **row,
            "binary_target": 0 if row["old_label"] == "KEEP_REQUIRED" else 1,
            "match_cell": "|".join(
                (str(row["dataset"]), str(row["source_regime"]), str(row["layer_bin"]), str(row["route_source_signature"]))
            ),
        }
        for row in manifest
        if row["old_label"] in {"KEEP_REQUIRED", "INTERVENE_REQUIRED"}
    ]
    if not audited_clean or {int(row["binary_target"]) for row in audited_clean} != {0, 1}:
        raise RuntimeError("audited clean cohort lacks both binary classes")
    feature_map = _load_selected_feature_map(contract, manifest)
    device = torch.device(f"cuda:{int(device_index)}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    summaries = []
    high_precision = []
    fold_rows = []
    for cohort, rows in (("old_sampled_clean", old_clean), ("audited_clean", audited_clean)):
        summary, operating, folds = _fit_probe_cohort(
            cohort=cohort,
            rows=rows,
            feature_map=feature_map,
            config=config,
            device=device,
        )
        summaries.extend(summary)
        high_precision.extend(operating)
        fold_rows.extend(folds)
        print(json.dumps({"cohort": cohort, "probe_summary": summary}, sort_keys=True), flush=True)
    atomic_csv(output_root / "probe_recheck/probe_summary.csv", summaries)
    atomic_csv(output_root / "probe_recheck/high_precision_metrics.csv", high_precision)
    atomic_jsonl(
        output_root / "probe_recheck/fold_manifest.jsonl",
        [row for row in fold_rows if "state_id" in row],
    )
    atomic_csv(
        output_root / "probe_recheck/fold_metrics.csv",
        [row for row in fold_rows if "probe" in row],
    )
    phase72_rw = float(config["probe"]["phase72_rw_auroc"])
    old_rw = next(row for row in summaries if row["cohort"] == "old_sampled_clean" and row["probe"] == "RW_read_write")
    audited_rw = next(row for row in summaries if row["cohort"] == "audited_clean" and row["probe"] == "RW_read_write")
    comparison = [
        {
            "comparison": "Phase72_full_to_audited_clean",
            "reference_states": 20095,
            "reference_rw_auroc": phase72_rw,
            "audited_states": audited_rw["states"],
            "audited_rw_auroc": audited_rw["auroc"],
            "absolute_gain": audited_rw["auroc"] - phase72_rw,
        },
        {
            "comparison": "old_sampled_clean_to_audited_clean",
            "reference_states": old_rw["states"],
            "reference_rw_auroc": old_rw["auroc"],
            "audited_states": audited_rw["states"],
            "audited_rw_auroc": audited_rw["auroc"],
            "absolute_gain": audited_rw["auroc"] - old_rw["auroc"],
        },
    ]
    atomic_csv(output_root / "probe_recheck/old_vs_audited_probe_comparison.csv", comparison)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(PROBE_NAMES))
    width = 0.36
    old_values = [next(row["auroc"] for row in summaries if row["cohort"] == "old_sampled_clean" and row["probe"] == probe) for probe in PROBE_NAMES]
    audited_values = [next(row["auroc"] for row in summaries if row["cohort"] == "audited_clean" and row["probe"] == probe) for probe in PROBE_NAMES]
    axis.bar(x - width / 2, old_values, width, label="old sampled clean")
    axis.bar(x + width / 2, audited_values, width, label="audited clean")
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(x, PROBE_NAMES, rotation=20)
    axis.set_ylabel("OOF AUROC")
    axis.set_title("Old versus audited label probe recheck")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_root / "figures/precision_recall_comparison.png", dpi=180)
    plt.close(fig)

    cardinality = Counter(int(row["audited_action_count"]) for row in audited)
    fig, left = plt.subplots(figsize=(7, 4.5))
    keys = sorted(cardinality)
    left.bar([str(key) for key in keys], [cardinality[key] for key in keys], color="#72b7b2")
    left.set_xlabel("audited successful-action cardinality")
    left.set_ylabel("states")
    left.set_title(f"Audited action cardinality; RW AUROC={audited_rw['auroc']:.3f}")
    fig.tight_layout()
    fig.savefig(output_root / "figures/label_cardinality_vs_probe_performance.png", dpi=180)
    plt.close(fig)
    atomic_json(
        output_root / "work/probe_recheck_complete.json",
        {
            "schema_version": "treatment_label_completeness_probe_recheck_complete_v1",
            "contract_sha256": contract["contract_sha256"],
            "passed": True,
            "old_sampled_clean_states": len(old_clean),
            "audited_clean_states": len(audited_clean),
            "audited_rw_auroc": audited_rw["auroc"],
            "audited_rw_precision_at_5pct": audited_rw["precision_at_5pct"],
            "phase72_rw_auroc": phase72_rw,
            "rw_auroc_gain_vs_phase72": audited_rw["auroc"] - phase72_rw,
            "completed_at": utc_now(),
        },
    )


def _required_artifacts() -> list[str]:
    return [
        "protocol.md", "frozen_protocol.json",
        "sampling/full_state_census.csv", "sampling/frozen_1200_state_manifest.jsonl",
        "sampling/sampling_strata.csv", "sampling/uid_coverage.csv",
        "search/per_state_action_audit.jsonl", "search/direct_suffix_results.jsonl",
        "search/single_later_results.jsonl", "search/mcts_results.jsonl",
        "search/replay_validation.jsonl", "search/quarantined_states.jsonl",
        "labels/old_vs_audited_action_sets.jsonl", "labels/audited_clean_labels.jsonl",
        "labels/audited_mixed_states.jsonl", "labels/label_transition_summary.csv",
        "labels/action_expansion_summary.csv", "metrics/overall_completeness.csv",
        "metrics/uid_bootstrap_ci.csv", "metrics/dataset_breakdown.csv",
        "metrics/source_breakdown.csv", "metrics/layer_breakdown.csv",
        "metrics/route_source_breakdown.csv", "metrics/action_discovery_rates.csv",
        "metrics/search_stage_attribution.csv", "metrics/budget_saturation.csv",
        "metrics/compute_summary.csv", "probe_recheck/fold_manifest.jsonl",
        "probe_recheck/probe_summary.csv", "probe_recheck/high_precision_metrics.csv",
        "probe_recheck/old_vs_audited_probe_comparison.csv",
        "figures/label_transition_matrix.png", "figures/new_action_discovery_by_type.png",
        "figures/discovery_stage_attribution.png", "figures/invalidation_rate_by_dataset.png",
        "figures/invalidation_rate_by_source.png", "figures/invalidation_rate_by_layer_bin.png",
        "figures/mcts_budget_saturation.png", "figures/precision_recall_comparison.png",
        "figures/route_source_incompleteness.png",
        "figures/label_cardinality_vs_probe_performance.png",
        "summaries/treatment_label_completeness_summary.md",
        "summaries/next_stage2_recommendation.md",
    ]


def finalize(config_path: Path) -> None:
    contract, output_root = load_contract(config_path, verify_model=False)
    aggregation = read_json(output_root / "work/aggregation_complete.json")
    probe = read_json(output_root / "work/probe_recheck_complete.json")
    if not aggregation.get("passed") or not probe.get("passed"):
        raise RuntimeError("finalization requires passing aggregation and probe recheck")
    config = contract["static_config"]
    thresholds = config["decision_thresholds"]
    bootstrap = {row["metric"]: row for row in aggregation["bootstrap"]}
    keep_ci = bootstrap["KEEP_invalidation_rate"]
    intervene_ci = bootstrap["INTERVENE_invalidation_rate"]
    large = any(
        float(row["estimate"]) >= float(thresholds["large_invalidation_point_estimate_min"])
        and float(row["ci95_low"]) >= float(thresholds["large_invalidation_uid_bootstrap_ci_lower_min"])
        for row in (keep_ci, intervene_ci)
    )
    stable = all(
        float(row["ci95_high"]) <= float(thresholds["mostly_stable_uid_bootstrap_ci_upper_max"])
        for row in (keep_ci, intervene_ci)
    )
    material_probe = (
        float(probe["rw_auroc_gain_vs_phase72"]) >= float(thresholds["material_rw_auroc_gain_min"])
        and float(probe["audited_rw_auroc"]) >= float(thresholds["material_rw_auroc_floor"])
    )
    saturated = aggregation["mcts_saturation"]["saturated"]
    if saturated is False:
        case = "E"
        recommendation = "The MCTS discovery curve is unsaturated at 200; this audit cannot support a completeness conclusion. Preserve the bounded labels and do not retrain Stage 2 yet."
    elif large and material_probe:
        case = "A"
        recommendation = "Label incompleteness is large and audited-label probe separability materially improves. A separately authorized Stage-2 retrain on audited/expanded labels is defensible."
    elif large:
        case = "B"
        recommendation = "Label incompleteness is real and large, but corrected labels do not explain weak state separability. Do not assume broader search alone will repair Stage 2."
    elif material_probe:
        case = "D"
        recommendation = "Labels are not largely invalidated, yet the audited sample improves. Treat this as sampling/stratum sensitivity and validate it before any deployment-head decision."
    else:
        case = "C"
        recommendation = "Observed labels are not largely invalidated and the audited-label probe remains weak. Representation limitation is the stronger working explanation."
    saturation_qualifier = (
        "MCTS saturation was inconclusive because fewer than ten MCTS discoveries were observed."
        if saturated is None
        else f"MCTS saturation rule result: {saturated}."
    )
    overall = aggregation["overall"]
    summary = f"""# Treatment-label completeness audit summary

- Contract: `{contract['contract_sha256']}`
- Selected/clean/quarantined states: {aggregation['selected_states']:,} / {aggregation['clean_states']:,} / {aggregation['quarantined_states']:,}.
- Old KEEP invalidated to audited MIXED: {overall['KEEP_invalidated']}/{overall['old_KEEP']} ({overall['KEEP_invalidation_rate']:.4f}); UID-bootstrap 95% CI [{keep_ci['ci95_low']:.4f}, {keep_ci['ci95_high']:.4f}].
- Old INTERVENE invalidated to audited MIXED: {overall['INTERVENE_invalidated']}/{overall['old_INTERVENE']} ({overall['INTERVENE_invalidation_rate']:.4f}); UID-bootstrap 95% CI [{intervene_ci['ci95_low']:.4f}, {intervene_ci['ci95_high']:.4f}].
- Any new replay-valid action: {overall['states_with_new_action']}/{overall['states']} ({overall['any_new_action_rate']:.4f}).
- Audited clean-label RW OOF AUROC: {probe['audited_rw_auroc']:.4f}; gain versus Phase-72 0.5620: {probe['rw_auroc_gain_vs_phase72']:+.4f}; P@5%: {probe['audited_rw_precision_at_5pct']:.4f}.
- {saturation_qualifier}
- Prospective decision case: **{case}**. Large invalidation={large}; mostly stable={stable}; material RW improvement={material_probe}.

These remain bounded-search action sets, not proofs of treatment necessity or impossibility. No deployment-external full-evaluation outcome was used for sampling, search, labels, thresholds, or probes.
"""
    _atomic_text(output_root / "summaries/treatment_label_completeness_summary.md", summary)
    _atomic_text(
        output_root / "summaries/next_stage2_recommendation.md",
        f"""# Next Stage-2 recommendation

Prospective case: **{case}**.

{recommendation}

{saturation_qualifier}

This phase does not authorize Stage-2 retraining, Stage-1 threshold changes, router replacement, or external evaluation.
""",
    )
    missing = [relative for relative in _required_artifacts() if not (output_root / relative).is_file()]
    if missing:
        raise RuntimeError(f"required audit artifacts are missing: {missing}")
    files = {relative: file_sha256(output_root / relative) for relative in _required_artifacts()}
    artifact = {
        "schema_version": "treatment_label_completeness_artifact_manifest_v1",
        "contract_sha256": contract["contract_sha256"],
        "passed": True,
        "decision_case": case,
        "files": files,
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "artifact_manifest.json", artifact)
    print(json.dumps({"finalized": True, "decision_case": case, "contract_sha256": contract["contract_sha256"], **aggregation["overall"], **probe}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "worker", "finalize-smoke", "aggregate", "probe", "finalize"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("smoke", "full"))
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.config)
    elif args.command == "worker":
        if args.mode is None or args.rank is None:
            raise SystemExit("worker requires --mode and --rank")
        worker(args.config, mode=args.mode, rank=args.rank, world_size=args.world_size, resume=args.resume)
    elif args.command == "finalize-smoke":
        finalize_smoke(args.config)
    elif args.command == "aggregate":
        aggregate(args.config)
    elif args.command == "probe":
        probe_recheck(args.config, args.device)
    else:
        finalize(args.config)


if __name__ == "__main__":
    main()
