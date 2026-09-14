#!/usr/bin/env python3
"""Run shared corrective search for the frozen P98/P95/P90 Stage-1 gates."""

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
)
from dense_failure_stage2.full_label_generation import (  # noqa: E402
    assign_workers,
    file_sha256,
)
from dense_failure_stage2.robust_gate_search import (  # noqa: E402
    PAIR_OUTCOMES,
    POINTS,
    build_missing_union_rows,
    classify_pair_outcome,
    next_mcts_root,
    route_structural_points,
    state_slice,
    trigger_order_status,
)
from experiments.run_full_corrective_label_generation import (  # noqa: E402
    _feature_schema as phase56_feature_schema,
    _route_counts,
)
from experiments.run_trigger_conditioned_corrective_search_pilot import (  # noqa: E402
    _generate_output,
    _load_model,
    _pool_route_states,
    atomic_torch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/robust_gate_corrective_search_v1.json"
DATASETS = ("gqa", "chartqa", "textvqa")
SOURCES = ("historical", "canonical")
DEPTH_BINS = ("L0-L8", "L9-L18", "L19-L27")
BOUND_CODE = (
    "configs/robust_gate_corrective_search_v1.json",
    "dense_failure_stage2/robust_gate_search.py",
    "dense_failure_stage2/corrective_search.py",
    "dense_failure_stage2/full_label_generation.py",
    "dense_failure_stage1/lmms_scoring.py",
    "dense_failure_stage1/runtime.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/model.py",
    "experiments/run_trigger_conditioned_corrective_search_pilot.py",
    "experiments/run_full_corrective_label_generation.py",
    "experiments/run_robust_gate_corrective_search.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    roots = (PROJECT_ROOT.resolve(), Path("/mnt/hyemin").resolve())
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise ValueError(f"path escapes allowed roots: {value}")
    return resolved


def canonical_hash(value: Mapping[str, Any], *, excluded: Sequence[str] = ("contract_sha256",)) -> str:
    payload = {key: item for key, item in value.items() if key not in excluded}
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode())


def write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite frozen artifact: {path}")
        return
    _atomic_bytes(path, payload)


def command_output(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result.stdout.strip()


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "lmms_eval": importlib.metadata.version("lmms-eval"),
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }


def load_static(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "robust_gate_corrective_search_config_v1":
        raise ValueError("unsupported robust-gate search config")
    if int(config["world_size"]) != 4 or tuple(config["operating_points"]) != POINTS:
        raise ValueError("world size or operating-point order differs")
    mcts = config["mcts"]
    if (
        tuple(mcts["actions"]) != ACTIONS
        or mcts["rollout_cardinalities"] != [2, 3, 4]
        or int(mcts["maximum_iterations"]) != 200
        or int(mcts["extra_iterations_after_first_success"]) != 25
        or int(mcts["retained_successful_routes"]) != 8
    ):
        raise ValueError("MCTS semantics differ from the frozen cap-200 search")
    if config["execution"]["threshold_validation"] != "exact_replay_and_token_parity":
        raise ValueError("threshold replay semantics differ")
    return config


def _artifact_files(manifest: Mapping[str, Any]) -> dict[str, str]:
    files = manifest.get("files")
    if isinstance(files, Mapping):
        return {str(key): str(value) for key, value in files.items()}
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, list):
        return {str(row["path"]): str(row["sha256"]) for row in artifacts}
    raise ValueError("unsupported artifact-manifest schema")


def _verify_manifest_file(root: Path, manifest: Mapping[str, Any], path: Path) -> None:
    if not bool(manifest.get("passed")):
        raise RuntimeError(f"source artifact manifest is not passing: {root}")
    relative = str(path.resolve().relative_to(root.resolve()))
    expected = _artifact_files(manifest).get(relative)
    if expected is None or not path.is_file() or file_sha256(path) != expected:
        raise RuntimeError(f"source artifact hash mismatch: {path}")


def _sample(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: candidate[key]
        for key in (
            "uid", "sample_id", "dataset", "prompt", "question", "answer",
            "all_answer_norms", "local_image_path", "image_content_sha256",
            "image_group_id", "max_new_tokens",
        )
    }


def _source_data(config: Mapping[str, Any], source: str) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = {
        str(row["uid"]): row
        for row in read_jsonl(resolve_path(config["sources"][f"{source}_candidates"]))
    }
    dense = {
        str(row["uid"]): row
        for row in read_jsonl(resolve_path(config["sources"][f"{source}_dense"]))
    }
    return candidates, dense


def _trigger_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in SOURCES:
        rows.extend(
            read_jsonl(resolve_path(config["sources"][f"phase64_{source}_trigger_map"]))
        )
    if len(rows) != 10399 * len(POINTS):
        raise RuntimeError("Phase-64 trigger-map coverage differs")
    keys = {(str(row["uid"]), str(row["threshold_name"])) for row in rows}
    if len(keys) != len(rows):
        raise RuntimeError("Phase-64 trigger map contains duplicate pairs")
    return rows


def _existing_route_catalog(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        manifest_path = resolve_path(config["sources"][f"{source}_route_manifest"])
        manifest = read_json(manifest_path)
        for kind in ("single", "mcts"):
            path = resolve_path(config["sources"][f"{source}_{kind}_routes"])
            _verify_manifest_file(manifest_path.parent, manifest, path)
            for row in read_jsonl(path):
                route_id = str(row["route_id"])
                if route_id in output:
                    raise RuntimeError(f"duplicate existing route ID: {route_id}")
                if not bool(row["final_lmms_correct"]) or not bool(row["replay_token_parity"]):
                    raise RuntimeError(f"invalid existing route: {route_id}")
                output[route_id] = {**row, "source_regime": source, "existing_kind": kind}
    return output


def _existing_validity(
    config: Mapping[str, Any], catalog: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[tuple[str, str], list[str]], dict[str, set[str]], list[dict[str, Any]]]:
    replay_rows = read_jsonl(resolve_path(config["sources"]["phase64_route_replays"]))
    by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_route: dict[str, set[str]] = defaultdict(set)
    for row in replay_rows:
        if row["status"] != "REPLAY_COMPATIBLE_CORRECT" or not bool(
            row["exact_stored_output_parity"]
        ):
            raise RuntimeError("Phase-64 accepted replay is not exact/correct")
        route_id = str(row["route_id"])
        if route_id not in catalog:
            raise RuntimeError(f"Phase-64 replay route is absent upstream: {route_id}")
        key = (str(row["uid"]), str(row["operating_point"]))
        by_pair[key].append(route_id)
        by_route[route_id].add(str(row["operating_point"]))
    for key in by_pair:
        by_pair[key] = sorted(set(by_pair[key]))
    if len(replay_rows) != 2618 or len(by_route) != 1517:
        raise RuntimeError("Phase-64 replay population differs")
    return by_pair, by_route, replay_rows


def _depth_bin(layer: int) -> str:
    if 0 <= int(layer) <= 8:
        return "L0-L8"
    if 9 <= int(layer) <= 18:
        return "L9-L18"
    if 19 <= int(layer) <= 27:
        return "L19-L27"
    raise ValueError("trigger layer must lie in 0..27")


def _select_smoke(
    rows: Sequence[Mapping[str, Any]], *, count: int, task: str, seed: int
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row["task_type"] == task
        and (task != "triggered_wrong" or bool(row["search_required"]))
    ]
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        cells[(str(row["source_regime"]), str(row["dataset"]))].append(row)
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cell in sorted(cells):
        candidates = sorted(
            cells[cell],
            key=lambda row: (
                sha256(f"phase65-smoke:{seed}:{task}:{row['uid']}".encode()).hexdigest(),
                str(row["uid"]),
            ),
        )
        if candidates and len(chosen) < count:
            chosen.append(dict(candidates[0]))
            seen.add(str(candidates[0]["uid"]))
    remaining = sorted(
        [row for row in eligible if str(row["uid"]) not in seen],
        key=lambda row: (
            sha256(f"phase65-smoke-fill:{seed}:{task}:{row['uid']}".encode()).hexdigest(),
            str(row["uid"]),
        ),
    )
    chosen.extend(dict(row) for row in remaining[: count - len(chosen)])
    if len(chosen) != count:
        raise RuntimeError(f"insufficient smoke population for {task}")
    return chosen


def _build_work(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    trigger_rows = _trigger_rows(config)
    compatibility = read_jsonl(resolve_path(config["sources"]["phase64_pair_compatibility"]))
    union = build_missing_union_rows(trigger_rows, compatibility)
    if len(union) != 1104:
        raise RuntimeError(f"missing-search union differs: {len(union)}")
    union_by_uid = {str(row["uid"]): row for row in union}
    point_counts = Counter(
        point
        for row in union
        for point in POINTS
        if bool(row[f"needs_search_{point}"])
    )
    if point_counts != Counter({"P98": 277, "P95": 581, "P90": 1048}):
        raise RuntimeError(f"missing-pair counts differ: {point_counts}")

    catalog = _existing_route_catalog(config)
    existing_by_pair, valid_points_by_route, phase64_replays = _existing_validity(
        config, catalog
    )
    for row in union:
        uid = str(row["uid"])
        row["existing_replay_compatible_routes"] = {
            point: list(existing_by_pair.get((uid, point), [])) for point in POINTS
        }
    trigger_by_uid: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in trigger_rows:
        trigger_by_uid[str(row["uid"])][str(row["threshold_name"])] = row
    pair_by_key = {
        (str(row["uid"]), str(row["operating_point"])): row
        for row in compatibility
    }
    if len(pair_by_key) != len(compatibility):
        raise RuntimeError("Phase-64 pair compatibility contains duplicates")

    source_data = {source: _source_data(config, source) for source in SOURCES}
    work: list[dict[str, Any]] = []
    wrong_pair_rows: list[dict[str, Any]] = []
    wrong_uids = sorted({str(row["uid"]) for row in compatibility})
    if len(wrong_uids) != 1307:
        raise RuntimeError(f"triggered-W UID union differs: {len(wrong_uids)}")
    for uid in wrong_uids:
        by_point = trigger_by_uid[uid]
        metadata = next(iter(by_point.values()))
        source = str(metadata["source_regime"])
        candidates, dense = source_data[source]
        if uid not in candidates or uid not in dense:
            raise RuntimeError(f"triggered-W row lacks candidate/dense input: {uid}")
        candidate, dense_row = candidates[uid], dense[uid]
        if bool(dense_row["current_dense_correct"]) or not bool(dense_row["current_dense_wrong"]):
            raise RuntimeError(f"triggered-W dense outcome differs: {uid}")
        if candidate["image_content_sha256"] != dense_row["image_content_sha256"]:
            raise RuntimeError(f"triggered-W image provenance differs: {uid}")
        triggers = {
            point: (
                int(by_point[point]["first_trigger_layer"])
                if point in by_point and bool(by_point[point]["triggered"])
                else None
            )
            for point in POINTS
        }
        order = trigger_order_status(triggers)
        existing_ids = {
            point: existing_by_pair.get((uid, point), []) for point in POINTS
        }
        unique_existing = sorted({route_id for ids in existing_ids.values() for route_id in ids})
        existing_routes = []
        for route_id in unique_existing:
            route = catalog[route_id]
            valid_points = sorted(valid_points_by_route[route_id], key=POINTS.index)
            if str(route["uid"]) != uid:
                raise RuntimeError(f"existing route/UID mismatch: {route_id}")
            existing_routes.append(
                {
                    "route_id": route_id,
                    "route_type": str(route["existing_kind"]),
                    "route_origin": f"existing_{route['existing_kind']}",
                    "actions": list(route["actions"]),
                    "route_key": str(route["route_key"]),
                    "expected_generated_token_ids": list(route["generated_token_ids"]),
                    "expected_generated_answer": str(route["generated_answer"]),
                    "expected_lmms_metric": str(route["lmms_metric"]),
                    "expected_lmms_score": float(route["lmms_score"]),
                    "valid_points": valid_points,
                    "source_contract_sha256": str(route["contract_sha256"]),
                }
            )
        search_required = uid in union_by_uid
        needs = {
            point: bool(union_by_uid.get(uid, {}).get(f"needs_search_{point}", False))
            for point in POINTS
        }
        missing_layers = [int(triggers[point]) for point in POINTS if needs[point]]
        search_root = min(missing_layers) if missing_layers else None
        existing_capture_cost = sum(
            28 - min(int(triggers[point]) for point in route["valid_points"])
            for route in existing_routes
        )
        estimated = existing_capture_cost
        if search_required:
            estimated += 3 * (28 - int(search_root))
            estimated += 200 * len(set(missing_layers))
        for point in POINTS:
            if triggers[point] is None:
                continue
            pair = pair_by_key[(uid, point)]
            route_ids = existing_ids[point]
            single_ids = [rid for rid in route_ids if catalog[rid]["existing_kind"] == "single"]
            mcts_ids = [rid for rid in route_ids if catalog[rid]["existing_kind"] == "mcts"]
            if (
                len(single_ids) != int(pair["replay_compatible_single_routes"])
                or len(mcts_ids) != int(pair["replay_compatible_mcts_routes"])
                or bool(pair["requires_new_search"]) != needs[point]
            ):
                raise RuntimeError(f"existing pair route counts differ: {uid}/{point}")
            wrong_pair_rows.append(
                {
                    "uid": uid,
                    "dataset": str(metadata["dataset"]),
                    "source_regime": source,
                    "operating_point": point,
                    "trigger_layer": triggers[point],
                    "trigger_depth_bin": _depth_bin(int(triggers[point])),
                    "needs_search": needs[point],
                    "existing_single_route_ids": single_ids,
                    "existing_mcts_route_ids": mcts_ids,
                }
            )
        work.append(
            {
                "uid": uid,
                "dataset": str(metadata["dataset"]),
                "source_regime": source,
                "task_type": "triggered_wrong",
                "search_required": search_required,
                "search_root": search_root,
                "triggers": triggers,
                "needs_search": needs,
                "trigger_order": order,
                "existing_route_ids": existing_ids,
                "existing_routes": existing_routes,
                "sample": _sample(candidate),
                "dense_output": dense_row,
                "estimated_terminal_evaluations": estimated,
            }
        )

    correct_by_uid: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in trigger_rows:
        if bool(row["dense_correct"]) and bool(row["triggered"]):
            correct_by_uid[str(row["uid"])][str(row["threshold_name"])] = row
    if len(correct_by_uid) != 106:
        raise RuntimeError(f"triggered-C UID union differs: {len(correct_by_uid)}")
    preservation_rows = []
    for uid in sorted(correct_by_uid):
        by_point = correct_by_uid[uid]
        metadata = next(iter(by_point.values()))
        source = str(metadata["source_regime"])
        candidates, dense = source_data[source]
        candidate, dense_row = candidates[uid], dense[uid]
        if not bool(dense_row["current_dense_correct"]) or bool(dense_row["current_dense_wrong"]):
            raise RuntimeError(f"triggered-C dense outcome differs: {uid}")
        triggers = {
            point: (
                int(by_point[point]["first_trigger_layer"]) if point in by_point else None
            )
            for point in POINTS
        }
        capture_root = min(layer for layer in triggers.values() if layer is not None)
        preservation_rows.append(
            {
                "uid": uid,
                "dataset": str(metadata["dataset"]),
                "source_regime": source,
                "triggers": triggers,
            }
        )
        work.append(
            {
                "uid": uid,
                "dataset": str(metadata["dataset"]),
                "source_regime": source,
                "task_type": "triggered_correct_preservation",
                "search_required": False,
                "search_root": None,
                "triggers": triggers,
                "needs_search": {point: False for point in POINTS},
                "trigger_order": trigger_order_status(triggers),
                "existing_route_ids": {point: [] for point in POINTS},
                "existing_routes": [],
                "sample": _sample(candidate),
                "dense_output": dense_row,
                "estimated_terminal_evaluations": 2 * (28 - capture_root),
            }
        )
    if len(work) != 1413 or len(wrong_pair_rows) != 2378:
        raise RuntimeError("full robust-gate work population differs")
    metadata = {
        "catalog": catalog,
        "phase64_replays": phase64_replays,
        "missing_pair_counts": dict(point_counts),
        "trigger_order_violations": sum(
            not bool(row["trigger_order"]["monotonic"])
            for row in work
            if row["task_type"] == "triggered_wrong"
        ),
    }
    return work, union, wrong_pair_rows, {"preservation": preservation_rows, **metadata}


def prepare(config_path: Path) -> None:
    config = load_static(config_path)
    output_root = resolve_path(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    sources = {name: resolve_path(path) for name, path in config["sources"].items()}
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source files: {missing}")

    phase64_manifest = read_json(sources["phase64_artifact_manifest"])
    for name in (
        "phase64_contract", "phase64_historical_trigger_map",
        "phase64_canonical_trigger_map", "phase64_pair_compatibility",
        "phase64_route_replays", "phase64_named_operating_points",
    ):
        _verify_manifest_file(sources["phase64_artifact_manifest"].parent, phase64_manifest, sources[name])
    phase64 = read_json(sources["phase64_contract"])
    if canonical_hash(phase64) != phase64.get("contract_sha256"):
        raise RuntimeError("Phase-64 contract is invalid")
    if phase64_manifest.get("contract_sha256") != phase64["contract_sha256"]:
        raise RuntimeError("Phase-64 artifact/contract binding differs")

    dense_configs = {
        source: read_json(sources[f"{source}_dense_config"]) for source in SOURCES
    }
    if dense_configs["historical"]["backend_settings"] != dense_configs["canonical"]["backend_settings"]:
        raise RuntimeError("historical/canonical dense backend settings differ")
    for source, dense_config in dense_configs.items():
        if (
            dense_config["model_name"] != config["model"]["name"]
            or dense_config["model_revision"] != config["model"]["revision"]
            or dense_config["dtype"] != config["model"]["dtype"]
            or dense_config["attention_implementation"]
            != config["model"]["attention_implementation"]
        ):
            raise RuntimeError(f"{source} dense execution contract differs")

    named_points = {
        row["operating_point"]: row
        for row in read_csv(sources["phase64_named_operating_points"])
        if row.get("retained") == "True"
    }
    if tuple(point for point in POINTS if point in named_points) != POINTS:
        raise RuntimeError("Phase-64 retained operating points differ")
    operating_point_thresholds = {
        point: float(named_points[point]["threshold"]) for point in POINTS
    }

    work, union, wrong_pairs, metadata = _build_work(config)
    if metadata["trigger_order_violations"]:
        raise RuntimeError("unexpected trigger ordering found; inspect before execution")
    assigned = assign_workers(work, world_size=int(config["world_size"]))
    smoke_wrong = _select_smoke(
        work,
        count=int(config["smoke"]["wrong_records"]),
        task="triggered_wrong",
        seed=int(config["seed"]),
    )
    smoke_correct = _select_smoke(
        work,
        count=int(config["smoke"]["preservation_records"]),
        task="triggered_correct_preservation",
        seed=int(config["seed"]),
    )
    smoke = []
    for index, row in enumerate([*smoke_wrong, *smoke_correct]):
        smoke.append({**row, "worker_rank": index % int(config["world_size"])})
    smoke.sort(key=lambda row: str(row["uid"]))

    schema = phase56_feature_schema(config)
    old_schema = read_json(sources["phase56_feature_schema"])
    if schema != old_schema:
        raise RuntimeError("Stage-2 routed-state feature schema differs from Phase 56")
    atomic_jsonl(output_root / "manifests/missing_search_union.jsonl", union)
    atomic_jsonl(output_root / "work/manifests/triggered_wrong_pairs.jsonl", wrong_pairs)
    atomic_jsonl(output_root / "work/manifests/preservation_population.jsonl", metadata["preservation"])
    atomic_jsonl(output_root / "work/manifests/full_work_manifest.jsonl", assigned)
    atomic_jsonl(output_root / "work/manifests/smoke_manifest.jsonl", smoke)
    atomic_json(output_root / "states/feature_schema.json", schema)

    protocol = f"""# Robust Stage-1 missing corrective-search protocol

- Frozen parent: Phase-64 contract `{phase64['contract_sha256']}`.
- Operating points: P98/P95/P90 with exact frozen trigger maps and strict score comparison.
- Search population: 1,104 unique Dense-W UIDs, 1,906 missing threshold pairs.
- Route discovery: exhaustive single interventions once from each UID's earliest needed trigger.
- MCTS: deterministic existing tree/action/reward semantics, at most 200 iterations per actually launched unresolved trigger root, up to eight successful routes retained per root.
- Validation: structural eligibility followed by exact threshold-specific current Qwen/LMMS replay; token parity and correctness are required for corpus admission.
- State provenance: each unique route is captured once from its earliest exact-valid threshold trigger; every threshold view binds an explicit contiguous layer slice beginning at that threshold's trigger.
- Existing route handling: Phase-64 replay-compatible routes are retained and their states are recaptured under the robust trigger contract without new search.
- Preservation: all 106 triggered-C UIDs receive the FULL suffix, with threshold-specific state slices.
- Stop boundary: build P98/P95/P90 corpora and audits only; no Stage-2 training, Stage-1 change, test deployment, or external evaluation.
"""
    write_once(output_root / "protocol.md", protocol.encode())
    internal = {
        relative: file_sha256(output_root / relative)
        for relative in (
            "protocol.md", "manifests/missing_search_union.jsonl",
            "work/manifests/triggered_wrong_pairs.jsonl",
            "work/manifests/preservation_population.jsonl",
            "work/manifests/full_work_manifest.jsonl",
            "work/manifests/smoke_manifest.jsonl", "states/feature_schema.json",
        )
    }
    model_root = resolve_path(config["model"]["snapshot_path"])
    model_hashes = {
        path.name: file_sha256(path)
        for path in sorted(model_root.iterdir())
        if path.is_file()
    }
    if model_hashes != phase64["model_snapshot_sha256"]:
        raise RuntimeError("model snapshot differs from Phase-64 replay contract")
    contract: dict[str, Any] = {
        "schema_version": "robust_gate_corrective_search_contract_v1",
        "static_config": config,
        "git": {
            "commit": command_output(("git", "rev-parse", "HEAD")),
            "branch": command_output(("git", "branch", "--show-current")),
            "worktree_status_at_freeze": command_output(("git", "status", "--short")),
        },
        "runtime": _runtime_metadata(),
        "dense_backend_settings": dense_configs["historical"]["backend_settings"],
        "source_sha256": {name: file_sha256(path) for name, path in sources.items()},
        "bound_code_sha256": {path: file_sha256(resolve_path(path)) for path in BOUND_CODE},
        "internal_manifest_sha256": internal,
        "model_snapshot_sha256": model_hashes,
        "parent_phase64_contract_sha256": phase64["contract_sha256"],
        "operating_point_thresholds": operating_point_thresholds,
        "trigger_comparison": phase64["static_config"]["trigger"]["comparison"],
        "stage1_head_scoring": phase64["static_config"]["trigger"]["head_scoring"],
        "feature_schema_sha256": schema["feature_schema_sha256"],
        "population": {
            "missing_search_uids": 1104,
            "missing_pairs": 1906,
            "triggered_wrong_uids": 1307,
            "triggered_wrong_pairs": 2378,
            "triggered_correct_uids": 106,
            "existing_unique_routes": 1517,
            "full_work_uids": 1413,
            "smoke_uids": len(smoke),
        },
        "prospective_work": {
            "naive_pair_single_routes": 52800,
            "shared_uid_single_routes": 34023,
            "maximum_distinct_mcts_roots": 1867,
        },
        "created_at": utc_now(),
    }
    contract["contract_sha256"] = canonical_hash(contract)
    write_once(
        output_root / "frozen_protocol.json",
        (json.dumps(contract, indent=2, sort_keys=True) + "\n").encode(),
    )
    loads = Counter()
    for row in assigned:
        loads[int(row["worker_rank"])] += int(row["estimated_terminal_evaluations"])
    print(
        json.dumps(
            {
                "prepared": True,
                "contract_sha256": contract["contract_sha256"],
                "population": contract["population"],
                "estimated_worker_loads": dict(loads),
            },
            sort_keys=True,
        )
    )


def load_contract(
    config_path: Path, *, verify_model: bool = False
) -> tuple[dict[str, Any], Path]:
    config = load_static(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if canonical_hash(contract) != contract.get("contract_sha256"):
        raise RuntimeError("frozen search contract is invalid")
    if contract["static_config"] != config:
        raise RuntimeError("static config differs from frozen search contract")
    if command_output(("git", "rev-parse", "HEAD")) != contract["git"]["commit"]:
        raise RuntimeError("git commit differs from frozen search contract")
    if _runtime_metadata() != contract["runtime"]:
        raise RuntimeError("runtime differs from frozen search contract")
    for name, expected in contract["source_sha256"].items():
        if file_sha256(resolve_path(config["sources"][name])) != expected:
            raise RuntimeError(f"source changed after freeze: {name}")
    for path, expected in contract["bound_code_sha256"].items():
        if file_sha256(resolve_path(path)) != expected:
            raise RuntimeError(f"bound code changed after freeze: {path}")
    for relative, expected in contract["internal_manifest_sha256"].items():
        if file_sha256(output_root / relative) != expected:
            raise RuntimeError(f"internal manifest changed after freeze: {relative}")
    if verify_model:
        root = resolve_path(config["model"]["snapshot_path"])
        actual = {
            path.name: file_sha256(path)
            for path in sorted(root.iterdir())
            if path.is_file()
        }
        if actual != contract["model_snapshot_sha256"]:
            raise RuntimeError("model snapshot changed after freeze")
    return contract, output_root


def _safe_uid(uid: str) -> str:
    return sha256(uid.encode()).hexdigest()[:24]


def _new_route_id(contract_sha256: str, uid: str, route_origin: str, route_key: str) -> str:
    digest = sha256(
        f"{contract_sha256}:{uid}:{route_origin}:{route_key}".encode()
    ).hexdigest()[:24]
    return f"phase65:{route_origin}:{digest}"


def _worker_root(output_root: Path, mode: str, rank: int) -> Path:
    return output_root / f"work/{mode}/rank{rank:02d}"


def _state_relative(mode: str, rank: int, uid: str) -> Path:
    if mode == "smoke":
        return Path(f"work/smoke/rank{rank:02d}/states/{_safe_uid(uid)}.pt")
    return Path(f"states/shards/{_safe_uid(uid)}.pt")


def _state_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    return (
        list(actual["generated_ids"]) == list(expected["generated_token_ids"])
        and str(actual["generated_answer"]) == str(expected["generated_answer"])
        and str(actual["lmms_metric"]) == str(expected["lmms_metric"])
        and float(actual["lmms_score"]) == float(expected["lmms_score"])
        and bool(actual["correct"])
    )


def _route_state_payload(
    route: Mapping[str, Any], generated: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "generated_token_ids": list(generated["generated_ids"]),
        "generated_answer": str(generated["generated_answer"]),
        "lmms_metric": str(generated["lmms_metric"]),
        "lmms_score": float(generated["lmms_score"]),
        "correct": bool(generated["correct"]),
    }


def _validate_new_route(
    *,
    route: dict[str, Any],
    triggers: Mapping[str, int | None],
    baseline: Any,
    wrapped: Any,
    processor: Any,
    inputs: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    actions = list(route["actions"])
    structural = route_structural_points(triggers, actions)
    expected = {
        "generated_token_ids": route["generated_token_ids"],
        "generated_answer": route["generated_answer"],
        "lmms_metric": route["lmms_metric"],
        "lmms_score": route["lmms_score"],
    }
    validations = []
    valid = []
    replay_count = 0
    for point in POINTS:
        trigger = triggers.get(point)
        if trigger is None:
            validations.append(
                {
                    "route_id": route["route_id"],
                    "uid": route["uid"],
                    "operating_point": point,
                    "trigger_layer": None,
                    "status": "NOT_TRIGGERED",
                    "exact_token_parity": None,
                    "lmms_correct": None,
                }
            )
            continue
        if not structural[point]:
            validations.append(
                {
                    "route_id": route["route_id"],
                    "uid": route["uid"],
                    "operating_point": point,
                    "trigger_layer": int(trigger),
                    "status": "STRUCTURALLY_INCOMPATIBLE",
                    "exact_token_parity": None,
                    "lmms_correct": None,
                }
            )
            continue
        output = capture_four_action_suffix_from_full_baseline(
            wrapped, baseline, int(trigger), tuple(actions[int(trigger):])
        )
        observed = _generate_output(processor, wrapped, output, inputs, sample)
        replay_count += 1
        parity = (
            list(observed["generated_ids"]) == list(expected["generated_token_ids"])
            and str(observed["generated_answer"]) == str(expected["generated_answer"])
            and str(observed["lmms_metric"]) == str(expected["lmms_metric"])
            and float(observed["lmms_score"]) == float(expected["lmms_score"])
        )
        accepted = bool(observed["correct"]) and parity
        status = "REPLAY_COMPATIBLE_CORRECT" if accepted else (
            "REPLAY_INCOMPATIBLE_WRONG" if not bool(observed["correct"]) else "REPLAY_PARITY_ERROR"
        )
        validations.append(
            {
                "route_id": route["route_id"],
                "uid": route["uid"],
                "operating_point": point,
                "trigger_layer": int(trigger),
                "status": status,
                "exact_token_parity": parity,
                "lmms_correct": bool(observed["correct"]),
                "generated_token_ids": list(observed["generated_ids"]),
                "generated_answer": str(observed["generated_answer"]),
                "lmms_metric": str(observed["lmms_metric"]),
                "lmms_score": float(observed["lmms_score"]),
            }
        )
        if accepted:
            valid.append(point)
    discovery_root = int(route["search_root"])
    root_points = [
        point for point in POINTS if triggers.get(point) == discovery_root
    ]
    if root_points and not any(point in valid for point in root_points):
        raise RuntimeError(
            f"successful discovery route failed exact replay at its search root: {route['route_id']}"
        )
    route["valid_points"] = valid
    route["structural_points"] = [point for point in POINTS if structural.get(point)]
    route["validation_rows"] = validations
    return route, validations, replay_count


def _validate_saved_state_shard(
    path: Path, contract: Mapping[str, Any], *, mode: str
) -> None:
    if mode not in {"smoke", "full"}:
        raise ValueError("state-shard mode must be smoke or full")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in (
        ("contract_sha256", contract["contract_sha256"]),
        ("model_revision", contract["static_config"]["model"]["revision"]),
        ("code_commit", contract["git"]["commit"]),
        ("feature_schema_sha256", contract["feature_schema_sha256"]),
        ("split_run_id", contract["static_config"]["run_id"]),
    ):
        if saved.get(key) != expected:
            raise RuntimeError(f"state shard provenance mismatch: {key}")
    manifest_key = (
        "work/manifests/smoke_manifest.jsonl"
        if mode == "smoke"
        else "work/manifests/full_work_manifest.jsonl"
    )
    if saved.get("source_manifest_sha256") != contract["internal_manifest_sha256"][manifest_key]:
        raise RuntimeError("state shard provenance mismatch: source_manifest_sha256")
    records = saved.get("records")
    if not isinstance(records, list) or any(
        int(row["tensor_row"]) != index for index, row in enumerate(records)
    ):
        raise RuntimeError("state shard index is not contiguous")
    shape = (len(records), int(contract["static_config"]["features"]["hidden_size"]))
    if any(saved.get(key).shape != shape for key in ("text_final", "text_mean", "visual_mean")):
        raise RuntimeError("state shard tensor shapes differ")
    if any(saved[key].dtype != torch.bfloat16 for key in ("text_final", "text_mean", "visual_mean")):
        raise RuntimeError("state shard tensor dtypes differ")
    if any(
        row.get("contract_sha256") != contract["contract_sha256"]
        or str(row.get("uid")) != str(saved.get("uid"))
        or int(row.get("layer", -1)) not in range(28)
        or row.get("chosen_action") not in ACTIONS
        for row in records
    ):
        raise RuntimeError("state shard record provenance/schema differs")


def _capture_route_states(
    *,
    routes: Sequence[Mapping[str, Any]],
    manifest_row: Mapping[str, Any],
    baseline: Any,
    wrapped: Any,
    processor: Any,
    base: Any,
    inputs: Mapping[str, Any],
    contract: Mapping[str, Any],
    output_root: Path,
    mode: str,
    rank: int,
) -> tuple[str | None, str | None, list[dict[str, Any]], list[dict[str, Any]], int]:
    if not routes:
        return None, None, [], [], 0
    tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    tensor_row = 0
    state_replays = 0
    triggers = manifest_row["triggers"]
    for source in routes:
        route = dict(source)
        valid_points = [point for point in POINTS if point in route["valid_points"]]
        if not valid_points:
            continue
        capture_root = min(int(triggers[point]) for point in valid_points)
        actions = list(route["actions"])
        output = capture_four_action_suffix_from_full_baseline(
            wrapped, baseline, capture_root, tuple(actions[capture_root:])
        )
        observed = _generate_output(
            processor, wrapped, output, inputs, manifest_row["sample"]
        )
        expected = {
            "generated_token_ids": route["generated_token_ids"],
            "generated_answer": route["generated_answer"],
            "lmms_metric": route["lmms_metric"],
            "lmms_score": route["lmms_score"],
        }
        if not _state_matches(expected, observed):
            raise RuntimeError(f"state-capture route replay failed: {route['route_id']}")
        pooled, metadata = _pool_route_states(output, processor, base, inputs, capture_root)
        rows = len(metadata)
        if rows != 28 - capture_root:
            raise RuntimeError(f"state-capture horizon differs: {route['route_id']}")
        start = tensor_row
        stop = tensor_row + rows
        for key, values in pooled.items():
            tensors[key].append(values)
        for offset, item in enumerate(metadata):
            records.append(
                {
                    "schema_version": "robust_gate_routed_state_index_v1",
                    "contract_sha256": contract["contract_sha256"],
                    "uid": manifest_row["uid"],
                    "dataset": manifest_row["dataset"],
                    "source_regime": manifest_row["source_regime"],
                    "route_id": route["route_id"],
                    "route_type": route["route_type"],
                    "route_origin": route["route_origin"],
                    "capture_root": capture_root,
                    "layer": int(item["layer"]),
                    "chosen_action": str(item["action"]),
                    "tensor_row": start + offset,
                    "valid_operating_points": valid_points,
                    "final_lmms_correct": True,
                    "replay_token_parity": True,
                }
            )
        counts = _route_counts(actions)
        route_rows.append(
            {
                **route,
                **counts,
                "capture_root": capture_root,
                "tensor_start": start,
                "tensor_stop": stop,
                "state_rows": rows,
                "state_capture_token_parity": True,
                "final_lmms_correct": True,
            }
        )
        tensor_row = stop
        state_replays += 1
    if not route_rows:
        return None, None, [], [], state_replays
    packed = {key: torch.cat(values, dim=0) for key, values in tensors.items()}
    expected_shape = (
        len(records), int(contract["static_config"]["features"]["hidden_size"])
    )
    if set(packed) != {"text_final", "text_mean", "visual_mean"} or any(
        value.shape != expected_shape for value in packed.values()
    ):
        raise RuntimeError("routed-state packed tensor schema differs")
    relative = _state_relative(mode, rank, str(manifest_row["uid"]))
    manifest_key = (
        "work/manifests/smoke_manifest.jsonl"
        if mode == "smoke"
        else "work/manifests/full_work_manifest.jsonl"
    )
    payload = {
        "schema_version": "robust_gate_routed_state_shard_v1",
        "contract_sha256": contract["contract_sha256"],
        "model_revision": contract["static_config"]["model"]["revision"],
        "code_commit": contract["git"]["commit"],
        "feature_schema_sha256": contract["feature_schema_sha256"],
        "source_manifest_sha256": contract["internal_manifest_sha256"][manifest_key],
        "split_run_id": contract["static_config"]["run_id"],
        "uid": manifest_row["uid"],
        "records": records,
        **packed,
    }
    path = output_root / relative
    atomic_torch(path, payload)
    _validate_saved_state_shard(path, contract, mode=mode)
    digest = file_sha256(path)
    for row in records:
        row.update(
            {
                "feature_file": str(relative),
                "feature_file_sha256": digest,
                "feature_schema_sha256": contract["feature_schema_sha256"],
                "source_manifest_sha256": payload["source_manifest_sha256"],
                "split_run_id": payload["split_run_id"],
            }
        )
    for route in route_rows:
        route.update(
            {
                "feature_file": str(relative),
                "feature_file_sha256": digest,
                "feature_schema_sha256": contract["feature_schema_sha256"],
                "source_manifest_sha256": payload["source_manifest_sha256"],
                "split_run_id": payload["split_run_id"],
            }
        )
    return str(relative), digest, records, route_rows, state_replays


def _new_route(
    *,
    contract: Mapping[str, Any],
    uid: str,
    dataset: str,
    source_regime: str,
    origin: str,
    route_type: str,
    search_root: int,
    actions: Sequence[str],
    state: Mapping[str, Any],
    discovery: Mapping[str, Any],
) -> dict[str, Any]:
    route_key = "|".join(actions)
    return {
        "schema_version": "robust_gate_global_route_v1",
        "contract_sha256": contract["contract_sha256"],
        "uid": uid,
        "dataset": dataset,
        "source_regime": source_regime,
        "route_id": _new_route_id(contract["contract_sha256"], uid, origin, route_key),
        "route_key": route_key,
        "route_type": route_type,
        "route_origin": origin,
        "search_root": int(search_root),
        "search_roots": [int(search_root)],
        "actions": list(actions),
        **_route_counts(actions),
        "generated_token_ids": list(state["generated_token_ids"]),
        "generated_answer": str(state["generated_answer"]),
        "lmms_metric": str(state["lmms_metric"]),
        "lmms_score": float(state["lmms_score"]),
        "correct": bool(state["correct"]),
        **dict(discovery),
    }


def _existing_route(
    route: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    uid: str,
    dataset: str,
    source_regime: str,
) -> dict[str, Any]:
    actions = list(route["actions"])
    return {
        "schema_version": "robust_gate_global_route_v1",
        "contract_sha256": contract["contract_sha256"],
        "uid": uid,
        "dataset": dataset,
        "source_regime": source_regime,
        "route_id": str(route["route_id"]),
        "route_key": str(route["route_key"]),
        "route_type": str(route["route_type"]),
        "route_origin": str(route["route_origin"]),
        "search_root": None,
        "search_roots": [],
        "actions": actions,
        **_route_counts(actions),
        "generated_token_ids": list(route["expected_generated_token_ids"]),
        "generated_answer": str(route["expected_generated_answer"]),
        "lmms_metric": str(route["expected_lmms_metric"]),
        "lmms_score": float(route["expected_lmms_score"]),
        "correct": True,
        "valid_points": list(route["valid_points"]),
        "structural_points": list(route["valid_points"]),
        "validation_rows": [],
        "source_contract_sha256": str(route["source_contract_sha256"]),
    }


def _preservation_route(
    manifest_row: Mapping[str, Any], contract: Mapping[str, Any], baseline_state: Mapping[str, Any]
) -> dict[str, Any]:
    uid = str(manifest_row["uid"])
    actions = ["FULL"] * 28
    key = "|".join(actions)
    route_id = _new_route_id(contract["contract_sha256"], uid, "preservation_full", key)
    valid = [point for point in POINTS if manifest_row["triggers"].get(point) is not None]
    return {
        "schema_version": "robust_gate_global_route_v1",
        "contract_sha256": contract["contract_sha256"],
        "uid": uid,
        "dataset": manifest_row["dataset"],
        "source_regime": manifest_row["source_regime"],
        "route_id": route_id,
        "route_key": key,
        "route_type": "preservation_full",
        "route_origin": "preservation_full",
        "search_root": None,
        "search_roots": [],
        "actions": actions,
        **_route_counts(actions),
        **_route_state_payload({}, baseline_state),
        "valid_points": valid,
        "structural_points": valid,
        "validation_rows": [
            {
                "route_id": route_id,
                "uid": uid,
                "operating_point": point,
                "trigger_layer": int(manifest_row["triggers"][point]),
                "status": "PRESERVATION_FULL_CORRECT",
                "exact_token_parity": True,
                "lmms_correct": True,
            }
            for point in valid
        ],
    }


def worker(
    config_path: Path,
    *,
    mode: str,
    rank: int,
    world_size: int,
    resume: bool,
) -> None:
    worker_started = time.monotonic()
    contract, output_root = load_contract(config_path, verify_model=True)
    config = contract["static_config"]
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    if world_size != 4 or rank not in range(4) or torch.cuda.device_count() != 4:
        raise RuntimeError("Phase-65 workers require four visible GPUs/processes")
    if mode == "full":
        smoke = read_json(output_root / "work/smoke/completion.json")
        if not smoke.get("passed") or smoke["contract_sha256"] != contract["contract_sha256"]:
            raise RuntimeError("full search requires a passing smoke")
    manifest_path = output_root / (
        "work/manifests/smoke_manifest.jsonl"
        if mode == "smoke"
        else "work/manifests/full_work_manifest.jsonl"
    )
    rows = [
        row for row in read_jsonl(manifest_path) if int(row["worker_rank"]) == rank
    ]
    root = _worker_root(output_root, mode, rank)
    complete_path = root / "complete.json"
    if complete_path.exists():
        raise RuntimeError(f"worker is already complete: {complete_path}")
    sample_dir = root / "samples"
    existing = list(sample_dir.glob("*.json")) if sample_dir.exists() else []
    if existing and not resume:
        raise RuntimeError(f"partial worker output requires --resume: {rank}")
    completed: dict[str, dict[str, Any]] = {}
    for path in existing:
        row = read_json(path)
        if row.get("contract_sha256") != contract["contract_sha256"] or not row.get("passed"):
            raise RuntimeError(f"incompatible resume result: {path}")
        feature_file = row.get("feature_file")
        if feature_file:
            feature_path = output_root / str(feature_file)
            if not feature_path.is_file() or file_sha256(feature_path) != row["feature_file_sha256"]:
                raise RuntimeError(f"resume state shard is invalid: {row['uid']}")
            _validate_saved_state_shard(feature_path, contract, mode=mode)
        completed[str(row["uid"])] = row

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    configure_dense_determinism(
        int(config["seed"]) + rank, contract["dense_backend_settings"]
    )
    processor, base, wrapped = _load_model(config, device)
    for index, manifest_row in enumerate(rows):
        uid = str(manifest_row["uid"])
        if uid in completed:
            continue
        started = time.monotonic()
        sample_path = sample_dir / f"{_safe_uid(uid)}.json"
        try:
            sample = manifest_row["sample"]
            inputs, input_metadata = build_dense_inputs(processor, sample, device)
            if input_metadata["consumed_image_sha256"] != sample["image_content_sha256"]:
                raise RuntimeError("consumed image hash differs from frozen manifest")
            prepared = build_binary_inputs(wrapped, inputs)
            baseline = capture_full_baseline(
                wrapped,
                inputs,
                prepared_inputs=prepared,
                use_cache=True,
                native_causal=bool(config["execution"]["native_full_row_dispatch"]),
            )
            baseline_state = _generate_output(processor, wrapped, baseline, inputs, sample)
            dense = manifest_row["dense_output"]
            baseline_checks = {
                "generated_ids_match": list(baseline_state["generated_ids"])
                == list(dense["generated_token_ids"]),
                "answer_match": str(baseline_state["generated_answer"])
                == str(dense["generated_answer"]),
                "lmms_score_match": float(baseline_state["lmms_score"])
                == float(dense["lmms_eval_per_sample_score"]),
                "correctness_match": bool(baseline_state["correct"])
                == bool(dense["current_dense_correct"]),
                "prompt_sha_match": input_metadata["literal_prompt_sha256"]
                == dense["literal_prompt_sha256"],
                "image_sha_match": input_metadata["consumed_image_sha256"]
                == sample["image_content_sha256"],
            }
            if not all(baseline_checks.values()):
                raise RuntimeError(f"dense baseline parity failed: {baseline_checks}")
            expected_correct = manifest_row["task_type"] == "triggered_correct_preservation"
            if bool(baseline_state["correct"]) != expected_correct:
                raise RuntimeError("dense outcome differs from task type")

            route_cache: dict[str, dict[str, Any]] = {}
            physical_terminal_evaluations = 0

            def evaluate_actions(
                actions: Sequence[str], search_root: int, search_stage: str
            ) -> tuple[dict[str, Any], bool]:
                nonlocal physical_terminal_evaluations
                key = "|".join(actions)
                if key in route_cache:
                    return route_cache[key], False
                output = capture_four_action_suffix_from_full_baseline(
                    wrapped, baseline, int(search_root), tuple(actions[int(search_root):])
                )
                observed = _generate_output(processor, wrapped, output, inputs, sample)
                row = {
                    "actions": list(actions),
                    "route_key": key,
                    "search_root": int(search_root),
                    "search_stage": search_stage,
                    **_route_state_payload({}, observed),
                }
                route_cache[key] = row
                physical_terminal_evaluations += 1
                return row, True

            retained: dict[str, dict[str, Any]] = {}
            single_attempts: list[dict[str, Any]] = []
            validation_rows: list[dict[str, Any]] = []
            mcts_launches: list[dict[str, Any]] = []
            threshold_replays = 0

            for source_route in manifest_row["existing_routes"]:
                route = _existing_route(
                    source_route,
                    contract,
                    uid=uid,
                    dataset=str(manifest_row["dataset"]),
                    source_regime=str(manifest_row["source_regime"]),
                )
                retained[route["route_id"]] = route

            new_single_by_point: dict[str, set[str]] = defaultdict(set)
            new_mcts_by_point: dict[str, set[str]] = defaultdict(set)
            if bool(manifest_row["search_required"]):
                search_root = int(manifest_row["search_root"])
                for candidate_index, candidate in enumerate(
                    all_single_routes(start_layer=search_root), 1
                ):
                    observed, physical = evaluate_actions(
                        candidate["actions"], search_root, "new_single"
                    )
                    single_attempts.append(
                        {
                            "candidate_index": candidate_index,
                            "route_key": candidate["route_key"],
                            "intervention_layer": candidate["changed_layers"][0],
                            "intervention_action": candidate["changed_actions"][0],
                            "correct": bool(observed["correct"]),
                            "physical_terminal_evaluation": physical,
                            "generated_answer": observed["generated_answer"],
                            "lmms_score": observed["lmms_score"],
                        }
                    )
                    if not bool(observed["correct"]):
                        continue
                    route = _new_route(
                        contract=contract,
                        uid=uid,
                        dataset=manifest_row["dataset"],
                        source_regime=manifest_row["source_regime"],
                        origin="new_single",
                        route_type="single",
                        search_root=search_root,
                        actions=candidate["actions"],
                        state=observed,
                        discovery={"discovery_order": candidate_index},
                    )
                    route, validations, replayed = _validate_new_route(
                        route=route,
                        triggers=manifest_row["triggers"],
                        baseline=baseline,
                        wrapped=wrapped,
                        processor=processor,
                        inputs=inputs,
                        sample=sample,
                    )
                    threshold_replays += replayed
                    validation_rows.extend(validations)
                    retained[route["route_id"]] = route
                    for point in route["valid_points"]:
                        new_single_by_point[point].add(route["route_id"])

                remaining = {
                    point
                    for point in POINTS
                    if bool(manifest_row["needs_search"].get(point))
                    and not new_single_by_point[point]
                }
                attempted_roots: set[int] = set()
                mcts_routes_by_key: dict[str, dict[str, Any]] = {}
                while remaining:
                    trigger_values = {
                        point: int(manifest_row["triggers"][point]) for point in remaining
                    }
                    root_layer = next_mcts_root(
                        remaining, trigger_values, attempted_roots=attempted_roots
                    )
                    if root_layer is None:
                        break
                    attempted_roots.add(root_layer)
                    mcts_result = run_sequential_mcts(
                        uid=uid,
                        start_layer=root_layer,
                        seed=int(config["seed"]),
                        max_iterations=int(config["mcts"]["maximum_iterations"]),
                        extra_iterations_after_success=int(
                            config["mcts"]["extra_iterations_after_first_success"]
                        ),
                        evaluate=lambda actions, root_layer=root_layer: evaluate_actions(
                            actions, root_layer, "new_mcts"
                        )[0]["correct"],
                        exploration_constant=float(config["mcts"]["exploration_constant"]),
                        rollout_cardinalities=config["mcts"]["rollout_cardinalities"],
                        retain_successes=int(config["mcts"]["retained_successful_routes"]),
                    )
                    first_by_key: dict[str, int] = {}
                    for search_row in mcts_result["search_rows"]:
                        if int(search_row["reward"]) == 1:
                            first_by_key.setdefault(
                                str(search_row["route_key"]), int(search_row["iteration"])
                            )
                    launch = {
                        "uid": uid,
                        "dataset": manifest_row["dataset"],
                        "source_regime": manifest_row["source_regime"],
                        "search_root": root_layer,
                        "points_unresolved_at_launch": sorted(remaining, key=POINTS.index),
                        "iterations": int(mcts_result["iterations"]),
                        "first_success_iteration": mcts_result["first_success_iteration"],
                        "unique_terminal_routes": int(mcts_result["unique_terminal_routes"]),
                        "retained_successes": len(mcts_result["successful_routes"]),
                        "search_rows": mcts_result["search_rows"],
                    }
                    mcts_launches.append(launch)
                    for successful in mcts_result["successful_routes"]:
                        key = str(successful["route_key"])
                        observed = route_cache[key]
                        route_id = _new_route_id(
                            contract["contract_sha256"], uid, "new_mcts", key
                        )
                        if route_id in retained:
                            route = retained[route_id]
                            route["search_roots"] = sorted(
                                set(route["search_roots"]) | {root_layer}
                            )
                            route.setdefault("discovery_iterations_by_root", {})[
                                str(root_layer)
                            ] = first_by_key[key]
                            continue
                        route = _new_route(
                            contract=contract,
                            uid=uid,
                            dataset=manifest_row["dataset"],
                            source_regime=manifest_row["source_regime"],
                            origin="new_mcts",
                            route_type="mcts",
                            search_root=root_layer,
                            actions=successful["actions"],
                            state=observed,
                            discovery={
                                "discovery_iteration": first_by_key[key],
                                "discovery_iterations_by_root": {
                                    str(root_layer): first_by_key[key]
                                },
                            },
                        )
                        route, validations, replayed = _validate_new_route(
                            route=route,
                            triggers=manifest_row["triggers"],
                            baseline=baseline,
                            wrapped=wrapped,
                            processor=processor,
                            inputs=inputs,
                            sample=sample,
                        )
                        threshold_replays += replayed
                        validation_rows.extend(validations)
                        retained[route["route_id"]] = route
                        mcts_routes_by_key[key] = route
                        for point in route["valid_points"]:
                            new_mcts_by_point[point].add(route["route_id"])
                    remaining = {
                        point
                        for point in remaining
                        if not new_mcts_by_point[point]
                    }

            if manifest_row["task_type"] == "triggered_correct_preservation":
                route = _preservation_route(manifest_row, contract, baseline_state)
                retained[route["route_id"]] = route
                validation_rows.extend(route["validation_rows"])

            feature_file, feature_hash, feature_index, route_rows, state_replays = (
                _capture_route_states(
                    routes=list(retained.values()),
                    manifest_row=manifest_row,
                    baseline=baseline,
                    wrapped=wrapped,
                    processor=processor,
                    base=base,
                    inputs=inputs,
                    contract=contract,
                    output_root=output_root,
                    mode=mode,
                    rank=rank,
                )
            )
            result = {
                "schema_version": "robust_gate_corrective_search_sample_v1",
                "contract_sha256": contract["contract_sha256"],
                "uid": uid,
                "dataset": manifest_row["dataset"],
                "source_regime": manifest_row["source_regime"],
                "task_type": manifest_row["task_type"],
                "search_required": bool(manifest_row["search_required"]),
                "search_root": manifest_row["search_root"],
                "triggers": manifest_row["triggers"],
                "needs_search": manifest_row["needs_search"],
                "existing_route_ids": manifest_row["existing_route_ids"],
                "baseline_checks": baseline_checks,
                "single_attempts": single_attempts,
                "successful_new_single_routes": sum(
                    route["route_origin"] == "new_single" for route in route_rows
                ),
                "mcts_launches": mcts_launches,
                "successful_new_mcts_routes": sum(
                    route["route_origin"] == "new_mcts" for route in route_rows
                ),
                "new_single_route_ids_by_point": {
                    point: sorted(new_single_by_point[point]) for point in POINTS
                },
                "new_mcts_route_ids_by_point": {
                    point: sorted(new_mcts_by_point[point]) for point in POINTS
                },
                "threshold_validation_rows": validation_rows,
                "retained_routes": route_rows,
                "feature_file": feature_file,
                "feature_file_sha256": feature_hash,
                "feature_index": feature_index,
                "physical_terminal_evaluations": physical_terminal_evaluations,
                "threshold_replays": threshold_replays,
                "state_capture_replays": state_replays,
                "elapsed_seconds": time.monotonic() - started,
                "passed": True,
            }
            atomic_json(sample_path, result)
            completed[uid] = result
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "rank": rank,
                        "sample": index + 1,
                        "total": len(rows),
                        "uid": uid,
                        "single_successes": result["successful_new_single_routes"],
                        "mcts_roots": len(mcts_launches),
                        "mcts_successes": result["successful_new_mcts_routes"],
                        "routes": len(route_rows),
                        "seconds": round(result["elapsed_seconds"], 2),
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            atomic_json(
                root / f"failure_{_safe_uid(uid)}.json",
                {
                    "contract_sha256": contract["contract_sha256"],
                    "uid": uid,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "created_at": utc_now(),
                },
            )
            raise
        finally:
            torch.cuda.empty_cache()
    expected = {str(row["uid"]) for row in rows}
    if set(completed) != expected:
        raise RuntimeError(f"worker UID coverage differs: rank {rank}")
    atomic_json(
        complete_path,
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "mode": mode,
            "rank": rank,
            "uids": len(completed),
            "search_uids": sum(bool(row["search_required"]) for row in completed.values()),
            "worker_elapsed_seconds": time.monotonic() - worker_started,
            "completed_at": utc_now(),
        },
    )


def _collect_mode(
    output_root: Path, mode: str, contract_sha256: str
) -> list[dict[str, Any]]:
    rows = []
    for rank in range(4):
        root = _worker_root(output_root, mode, rank)
        complete = read_json(root / "complete.json")
        if not complete.get("passed") or complete["contract_sha256"] != contract_sha256:
            raise RuntimeError(f"incomplete {mode} rank: {rank}")
        failures = list(root.glob("failure_*.json"))
        if failures:
            raise RuntimeError(f"{mode} rank {rank} contains failure artifacts")
        rows.extend(read_json(path) for path in sorted((root / "samples").glob("*.json")))
    if len(rows) != len({str(row["uid"]) for row in rows}):
        raise RuntimeError(f"{mode} outputs contain duplicate UIDs")
    return sorted(rows, key=lambda row: str(row["uid"]))


def finalize_smoke(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    expected = read_jsonl(output_root / "work/manifests/smoke_manifest.jsonl")
    rows = _collect_mode(output_root, "smoke", contract["contract_sha256"])
    if {row["uid"] for row in rows} != {row["uid"] for row in expected} or len(rows) != 12:
        raise RuntimeError("smoke UID coverage differs")
    if Counter(row["task_type"] for row in rows) != Counter(
        {"triggered_wrong": 8, "triggered_correct_preservation": 4}
    ):
        raise RuntimeError("smoke task composition differs")
    if any(not all(row["baseline_checks"].values()) for row in rows):
        raise RuntimeError("smoke dense parity failed")
    if any(
        validation["status"] in {"REPLAY_PARITY_ERROR"}
        for row in rows
        for validation in row["threshold_validation_rows"]
    ):
        raise RuntimeError("smoke threshold replay parity failed")
    route_count = sum(len(row["retained_routes"]) for row in rows)
    state_count = sum(len(row["feature_index"]) for row in rows)
    if route_count < 4 or state_count < 4 * 1:
        raise RuntimeError("smoke did not exercise routed state capture")
    for row in rows:
        if row["retained_routes"]:
            path = output_root / row["feature_file"]
            if file_sha256(path) != row["feature_file_sha256"]:
                raise RuntimeError(f"smoke state hash mismatch: {row['uid']}")
            _validate_saved_state_shard(path, contract, mode="smoke")
            for route in row["retained_routes"]:
                for point in route["valid_points"]:
                    start, stop = state_slice(
                        capture_root=int(route["capture_root"]),
                        threshold_trigger=int(row["triggers"][point]),
                        tensor_start=int(route["tensor_start"]),
                    )
                    if stop - start != 28 - int(row["triggers"][point]):
                        raise RuntimeError("smoke threshold state slice differs")
    completion = {
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        "uids": len(rows),
        "routes": route_count,
        "state_rows": state_count,
        "search_uids": sum(row["search_required"] for row in rows),
        "mcts_roots": sum(len(row["mcts_launches"]) for row in rows),
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "work/smoke/completion.json", completion)
    print(json.dumps(completion, sort_keys=True))


def _route_view(
    route: Mapping[str, Any],
    *,
    point: str,
    trigger_layer: int,
    pair_outcome: str | None,
) -> dict[str, Any]:
    start, stop = state_slice(
        capture_root=int(route["capture_root"]),
        threshold_trigger=int(trigger_layer),
        tensor_start=int(route["tensor_start"]),
    )
    if stop > int(route["tensor_stop"]) or stop - start != 28 - int(trigger_layer):
        raise RuntimeError(f"invalid threshold state view: {route['route_id']}/{point}")
    return {
        "schema_version": "robust_gate_threshold_corpus_route_v1",
        "contract_sha256": route["contract_sha256"],
        "uid": route["uid"],
        "dataset": route["dataset"],
        "source_regime": route["source_regime"],
        "operating_point": point,
        "trigger_layer": int(trigger_layer),
        "route_id": route["route_id"],
        "route_type": route["route_type"],
        "route_origin": route["route_origin"],
        "pair_outcome": pair_outcome,
        "actions": route["actions"],
        "feature_file": route["feature_file"],
        "feature_file_sha256": route["feature_file_sha256"],
        "feature_schema_sha256": route["feature_schema_sha256"],
        "state_tensor_start": start,
        "state_tensor_stop": stop,
        "state_rows": stop - start,
        "capture_root": route["capture_root"],
        "final_lmms_correct": True,
        "exact_replay_valid": True,
    }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _collect_full_results(
    contract: Mapping[str, Any], output_root: Path
) -> list[dict[str, Any]]:
    expected = read_jsonl(output_root / "work/manifests/full_work_manifest.jsonl")
    rows = _collect_mode(output_root, "full", contract["contract_sha256"])
    if {row["uid"] for row in rows} != {row["uid"] for row in expected}:
        raise RuntimeError("full result UID coverage differs")
    if len(rows) != int(contract["population"]["full_work_uids"]):
        raise RuntimeError("full result count differs")
    return rows


def _plots(
    output_root: Path,
    coverage: Sequence[Mapping[str, Any]],
    dataset_rows: Sequence[Mapping[str, Any]],
    depth_rows: Sequence[Mapping[str, Any]],
    efficiency: Mapping[str, Any],
) -> None:
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    points = list(POINTS)
    known = [int(next(row for row in coverage if row["operating_point"] == point)["known_corrective_supervision"]) for point in points]
    unresolved = [int(next(row for row in coverage if row["operating_point"] == point)["unresolved"]) for point in points]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(points, known, label="known corrective")
    ax.bar(points, unresolved, bottom=known, label="unresolved")
    ax.set(ylabel="Triggered-W bases", title="Corrective coverage by operating point")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "corrective_coverage_by_threshold.png", dpi=180); plt.close(fig)

    singles = [int(next(row for row in coverage if row["operating_point"] == point)["single_resolved"]) for point in points]
    mcts = [int(next(row for row in coverage if row["operating_point"] == point)["mcts_only_resolved"]) for point in points]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(points, singles, label="single")
    ax.bar(points, mcts, bottom=singles, label="MCTS")
    ax.set(ylabel="Triggered-W outcome bases", title="Single-resolved vs MCTS-only")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "single_vs_mcts_by_threshold.png", dpi=180); plt.close(fig)

    labels = [f"{row['source_regime']}\n{row['dataset']}\n{row['operating_point']}" for row in dataset_rows]
    values = [float(row["known_corrective_coverage"]) for row in dataset_rows]
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set(ylabel="Coverage among triggered W", title="Dataset/source corrective coverage", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(figures / "dataset_source_corrective_coverage.png", dpi=180); plt.close(fig)

    depth_labels = [f"{row['operating_point']}\n{row['trigger_depth_bin']}" for row in depth_rows]
    depth_values = [float(row["total_bounded_correctability"]) for row in depth_rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(depth_labels)), depth_values)
    ax.set_xticks(range(len(depth_labels)), depth_labels, rotation=45, ha="right")
    ax.set(ylabel="Bounded correctability", title="Trigger-depth correctability", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(figures / "trigger_depth_correctability.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        ["Naive pair singles", "Shared UID singles", "Post-single pairs", "Actual MCTS roots"],
        [
            int(efficiency["naive_pair_single_terminal_evaluations"]),
            int(efficiency["actual_shared_single_terminal_evaluations"]),
            int(efficiency["post_single_unresolved_pairs"]),
            int(efficiency["actual_mcts_search_roots"]),
        ],
    )
    ax.set(ylabel="Count", title="Cross-threshold search reuse")
    ax.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(figures / "search_reuse_savings.png", dpi=180); plt.close(fig)


def aggregate(config_path: Path) -> None:
    contract, output_root = load_contract(config_path, verify_model=True)
    smoke = read_json(output_root / "work/smoke/completion.json")
    if not smoke.get("passed") or smoke["contract_sha256"] != contract["contract_sha256"]:
        raise RuntimeError("aggregation requires a passing smoke")
    results = _collect_full_results(contract, output_root)
    wrong_results = [row for row in results if row["task_type"] == "triggered_wrong"]
    correct_results = [
        row for row in results if row["task_type"] == "triggered_correct_preservation"
    ]
    if len(wrong_results) != 1307 or len(correct_results) != 106:
        raise RuntimeError("W/C full result composition differs")
    if sum(row["search_required"] for row in wrong_results) != 1104:
        raise RuntimeError("executed search population differs")

    routes: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    new_validations: list[dict[str, Any]] = []
    verified_shards = set()
    for result in results:
        route_rows = result["retained_routes"]
        if route_rows:
            path = output_root / str(result["feature_file"])
            if file_sha256(path) != result["feature_file_sha256"]:
                raise RuntimeError(f"full state shard hash mismatch: {result['uid']}")
            _validate_saved_state_shard(path, contract, mode="full")
            verified_shards.add(str(result["feature_file"]))
        elif result["feature_file"] is not None or result["feature_index"]:
            raise RuntimeError(f"empty route result has state artifacts: {result['uid']}")
        routes.extend(route_rows)
        states.extend(result["feature_index"])
        new_validations.extend(result["threshold_validation_rows"])
    route_ids = [str(row["route_id"]) for row in routes]
    if len(route_ids) != len(set(route_ids)):
        raise RuntimeError("global route store contains duplicate route IDs")
    route_by_id = {str(row["route_id"]): row for row in routes}
    if len(states) != sum(int(row["state_rows"]) for row in routes):
        raise RuntimeError("global route/state row coverage differs")
    state_counts = Counter(str(row["route_id"]) for row in states)
    if any(state_counts[route_id] != int(route["state_rows"]) for route_id, route in route_by_id.items()):
        raise RuntimeError("per-route state coverage differs")

    phase64_replays = read_jsonl(
        resolve_path(contract["static_config"]["sources"]["phase64_route_replays"])
    )
    imported_validations = [
        {
            **row,
            "route_origin": f"existing_{row['route_source']}",
            "exact_token_parity": bool(row["exact_stored_output_parity"]),
            "lmms_correct": row["status"] == "REPLAY_COMPATIBLE_CORRECT",
            "validation_origin": "phase64_exact_replay_import",
        }
        for row in phase64_replays
    ]
    for row in new_validations:
        row["route_origin"] = route_by_id[str(row["route_id"])]["route_origin"]
        row["validation_origin"] = "phase65_exact_replay"
    replay_rows = sorted(
        [*imported_validations, *new_validations],
        key=lambda row: (str(row["uid"]), str(row["route_id"]), str(row["operating_point"])),
    )
    accepted = {
        (str(row["route_id"]), str(row["operating_point"]))
        for row in replay_rows
        if row["status"] in {"REPLAY_COMPATIBLE_CORRECT", "PRESERVATION_FULL_CORRECT"}
        and bool(row["exact_token_parity"])
        and bool(row["lmms_correct"])
    }
    for route in routes:
        for point in route["valid_points"]:
            if (str(route["route_id"]), point) not in accepted:
                raise RuntimeError(f"route valid flag lacks exact replay: {route['route_id']}/{point}")
        route["valid_for_P98"] = "P98" in route["valid_points"]
        route["valid_for_P95"] = "P95" in route["valid_points"]
        route["valid_for_P90"] = "P90" in route["valid_points"]
        route.pop("validation_rows", None)
    routes.sort(key=lambda row: (str(row["uid"]), str(row["route_id"])))
    states.sort(key=lambda row: (str(row["uid"]), str(row["route_id"]), int(row["layer"])))
    atomic_jsonl(output_root / "routes/global_route_store.jsonl", routes)
    atomic_jsonl(output_root / "routes/replay_results.jsonl", replay_rows)
    atomic_jsonl(output_root / "states/routed_state_manifest.jsonl", states)

    single_search_rows = []
    mcts_launches = []
    for result in wrong_results:
        if result["search_required"]:
            single_search_rows.append(
                {
                    "uid": result["uid"],
                    "dataset": result["dataset"],
                    "source_regime": result["source_regime"],
                    "search_root": result["search_root"],
                    "routes_evaluated": len(result["single_attempts"]),
                    "successful_routes": result["successful_new_single_routes"],
                    "successful_route_ids_by_point": result["new_single_route_ids_by_point"],
                    "physical_terminal_evaluations": sum(
                        bool(row["physical_terminal_evaluation"])
                        for row in result["single_attempts"]
                    ),
                }
            )
        for launch in result["mcts_launches"]:
            mcts_launches.append({**launch, "contract_sha256": contract["contract_sha256"]})
    new_single_routes = [row for row in routes if row["route_origin"] == "new_single"]
    new_mcts_routes = [row for row in routes if row["route_origin"] == "new_mcts"]
    new_single_ids = {row["route_id"] for row in new_single_routes}
    new_mcts_ids = {row["route_id"] for row in new_mcts_routes}
    single_validations = [row for row in replay_rows if row["route_id"] in new_single_ids]
    mcts_validations = [row for row in replay_rows if row["route_id"] in new_mcts_ids]
    atomic_jsonl(output_root / "single_search/per_uid_results.jsonl", single_search_rows)
    atomic_jsonl(output_root / "single_search/successful_routes.jsonl", new_single_routes)
    atomic_jsonl(output_root / "single_search/threshold_compatibility.jsonl", single_validations)
    atomic_jsonl(output_root / "mcts/launched_searches.jsonl", mcts_launches)
    atomic_jsonl(output_root / "mcts/successful_routes.jsonl", new_mcts_routes)
    atomic_jsonl(output_root / "mcts/threshold_compatibility.jsonl", mcts_validations)
    root_summary = [
        {
            "uid": row["uid"],
            "dataset": row["dataset"],
            "source_regime": row["source_regime"],
            "search_root": row["search_root"],
            "points_unresolved_at_launch": "|".join(row["points_unresolved_at_launch"]),
            "iterations": row["iterations"],
            "first_success_iteration": row["first_success_iteration"],
            "unique_terminal_routes": row["unique_terminal_routes"],
            "retained_successes": row["retained_successes"],
        }
        for row in mcts_launches
    ]
    if root_summary:
        atomic_csv(output_root / "mcts/per_root_summary.csv", root_summary)
    else:
        raise RuntimeError("full search launched no MCTS roots")

    pair_source = read_jsonl(output_root / "work/manifests/triggered_wrong_pairs.jsonl")
    pair_rows = []
    routes_by_uid = defaultdict(list)
    for route in routes:
        routes_by_uid[str(route["uid"])].append(route)
    for pair in pair_source:
        uid, point = str(pair["uid"]), str(pair["operating_point"])
        available = [route for route in routes_by_uid[uid] if point in route["valid_points"]]
        existing_single = [route for route in available if route["route_origin"] == "existing_single"]
        existing_mcts = [route for route in available if route["route_origin"] == "existing_mcts"]
        new_single = [route for route in available if route["route_origin"] == "new_single"]
        new_mcts = [route for route in available if route["route_origin"] == "new_mcts"]
        outcome = classify_pair_outcome(
            existing_single=bool(existing_single),
            existing_mcts=bool(existing_mcts),
            new_single=bool(new_single),
            new_mcts=bool(new_mcts),
        )
        if outcome not in PAIR_OUTCOMES:
            raise RuntimeError("unsupported pair outcome")
        pair_rows.append(
            {
                **pair,
                "outcome_class": outcome,
                "existing_single_route_ids": [row["route_id"] for row in existing_single],
                "existing_mcts_route_ids": [row["route_id"] for row in existing_mcts],
                "new_single_route_ids": [row["route_id"] for row in new_single],
                "new_mcts_route_ids": [row["route_id"] for row in new_mcts],
                "known_corrective_supervision": outcome != "UNRESOLVED_AT_BUDGET",
            }
        )
    if len(pair_rows) != 2378 or len({(row["uid"], row["operating_point"]) for row in pair_rows}) != 2378:
        raise RuntimeError("final pair outcome coverage differs")
    remaining = [
        row
        for row in pair_rows
        if row["needs_search"] and row["outcome_class"] == "UNRESOLVED_AT_BUDGET"
    ]
    atomic_jsonl(output_root / "manifests/remaining_unresolved_pairs.jsonl", remaining)
    atomic_jsonl(output_root / "work/final_pair_outcomes.jsonl", pair_rows)

    corpus_stats = []
    corpus_routes_by_point: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for point in POINTS:
        point_pairs = {str(row["uid"]): row for row in pair_rows if row["operating_point"] == point}
        preservation = []
        singles = []
        mcts = []
        for route in routes:
            if point not in route["valid_points"]:
                continue
            trigger = next(
                int(result["triggers"][point])
                for result in results
                if result["uid"] == route["uid"] and result["triggers"].get(point) is not None
            )
            outcome = point_pairs.get(str(route["uid"]), {}).get("outcome_class")
            view = _route_view(route, point=point, trigger_layer=trigger, pair_outcome=outcome)
            if route["route_type"] == "preservation_full":
                preservation.append(view)
            elif route["route_type"] == "single":
                singles.append(view)
            elif route["route_type"] == "mcts":
                mcts.append(view)
            else:
                raise RuntimeError(f"unsupported route type: {route['route_type']}")
        unresolved = [
            {
                "schema_version": "robust_gate_threshold_unresolved_v1",
                "contract_sha256": contract["contract_sha256"],
                **row,
                "interpretation": "bounded-search unresolved; not provably unfixable",
            }
            for row in pair_rows
            if row["operating_point"] == point
            and row["outcome_class"] == "UNRESOLVED_AT_BUDGET"
        ]
        root = output_root / f"threshold_corpora/{point}"
        atomic_jsonl(root / "preservation.jsonl", preservation)
        atomic_jsonl(root / "single.jsonl", singles)
        atomic_jsonl(root / "mcts.jsonl", mcts)
        atomic_jsonl(root / "unresolved.jsonl", unresolved)
        corpus_routes_by_point[point] = {
            "preservation": preservation, "single": singles, "mcts": mcts
        }
        corpus_stats.append(
            {
                "operating_point": point,
                "preservation_bases": len({row["uid"] for row in preservation}),
                "single_corpus_w_bases": len({row["uid"] for row in singles}),
                "mcts_corpus_w_bases": len({row["uid"] for row in mcts}),
                "single_mcts_corpus_overlap_bases": len(
                    {row["uid"] for row in singles} & {row["uid"] for row in mcts}
                ),
                "unresolved_w_bases": len(unresolved),
                "retained_routes": len(preservation) + len(singles) + len(mcts),
                "routed_state_rows": sum(row["state_rows"] for row in [*preservation, *singles, *mcts]),
            }
        )

    coverage = []
    for point in POINTS:
        subset = [row for row in pair_rows if row["operating_point"] == point]
        counts = Counter(row["outcome_class"] for row in subset)
        known = len(subset) - counts["UNRESOLVED_AT_BUDGET"]
        single_supervised = len(
            {
                row["uid"]
                for row in corpus_routes_by_point[point]["single"]
            }
        )
        mcts_supervised = len(
            {
                row["uid"]
                for row in corpus_routes_by_point[point]["mcts"]
            }
        )
        coverage.append(
            {
                "operating_point": point,
                "triggered_wrong": len(subset),
                "existing_single_reused": counts["EXISTING_SINGLE_REUSED"],
                "existing_mcts_reused": counts["EXISTING_MCTS_REUSED"],
                "existing_wrong_covered": counts["EXISTING_SINGLE_REUSED"] + counts["EXISTING_MCTS_REUSED"],
                "new_single_fixable": counts["NEW_SINGLE_FIXABLE"],
                "new_mcts_fixable": counts["NEW_MCTS_FIXABLE"],
                "single_resolved": counts["EXISTING_SINGLE_REUSED"] + counts["NEW_SINGLE_FIXABLE"],
                "mcts_only_resolved": counts["EXISTING_MCTS_REUSED"] + counts["NEW_MCTS_FIXABLE"],
                "single_corpus_bases": single_supervised,
                "mcts_corpus_bases": mcts_supervised,
                "known_corrective_supervision": known,
                "unresolved": counts["UNRESOLVED_AT_BUDGET"],
                "conditional_corrective_coverage": _rate(known, len(subset)),
                "population_known_correctable_lower_bound": _rate(known, 4071),
            }
        )
    expected_triggered_wrong = {"P98": 344, "P95": 727, "P90": 1307}
    expected_preservation = {"P98": 7, "P95": 30, "P90": 106}
    if {
        row["operating_point"]: row["triggered_wrong"] for row in coverage
    } != expected_triggered_wrong:
        raise RuntimeError("threshold triggered-W population differs")
    if {
        row["operating_point"]: row["preservation_bases"] for row in corpus_stats
    } != expected_preservation:
        raise RuntimeError("threshold preservation population differs")
    atomic_csv(output_root / "metrics/threshold_coverage.csv", coverage)

    dataset_rows = []
    for point in POINTS:
        for source in SOURCES:
            for dataset in DATASETS:
                subset = [
                    row for row in pair_rows
                    if row["operating_point"] == point
                    and row["source_regime"] == source
                    and row["dataset"] == dataset
                ]
                counts = Counter(row["outcome_class"] for row in subset)
                known = len(subset) - counts["UNRESOLVED_AT_BUDGET"]
                dataset_rows.append(
                    {
                        "operating_point": point,
                        "source_regime": source,
                        "dataset": dataset,
                        "triggered_wrong": len(subset),
                        "single_fixable": counts["EXISTING_SINGLE_REUSED"] + counts["NEW_SINGLE_FIXABLE"],
                        "mcts_fixable": counts["EXISTING_MCTS_REUSED"] + counts["NEW_MCTS_FIXABLE"],
                        "unresolved": counts["UNRESOLVED_AT_BUDGET"],
                        "known_corrective_coverage": _rate(known, len(subset)),
                    }
                )
    atomic_csv(output_root / "metrics/dataset_source_breakdown.csv", dataset_rows)

    depth_rows = []
    for point in POINTS:
        for depth in DEPTH_BINS:
            subset = [
                row for row in pair_rows
                if row["operating_point"] == point and row["trigger_depth_bin"] == depth
            ]
            searched = [row for row in subset if row["needs_search"]]
            counts = Counter(row["outcome_class"] for row in subset)
            new_counts = Counter(row["outcome_class"] for row in searched)
            known = len(subset) - counts["UNRESOLVED_AT_BUDGET"]
            depth_rows.append(
                {
                    "operating_point": point,
                    "trigger_depth_bin": depth,
                    "triggered_wrong": len(subset),
                    "new_search_count": len(searched),
                    "new_single_fixable": new_counts["NEW_SINGLE_FIXABLE"],
                    "new_single_fixable_rate": _rate(new_counts["NEW_SINGLE_FIXABLE"], len(searched)),
                    "new_mcts_fixable": new_counts["NEW_MCTS_FIXABLE"],
                    "new_mcts_only_rate": _rate(new_counts["NEW_MCTS_FIXABLE"], len(searched)),
                    "total_known_corrective": known,
                    "total_bounded_correctability": _rate(known, len(subset)),
                    "unresolved": counts["UNRESOLVED_AT_BUDGET"],
                }
            )
    atomic_csv(output_root / "metrics/trigger_depth_breakdown.csv", depth_rows)

    searched_results = [row for row in wrong_results if row["search_required"]]
    actual_single = sum(len(row["single_attempts"]) for row in searched_results)
    post_single_unresolved_pairs = sum(
        bool(row["needs_search"].get(point))
        and not bool(row["new_single_route_ids_by_point"][point])
        for row in searched_results
        for point in POINTS
    )
    actual_mcts_roots = len(mcts_launches)
    mcts_iterations = sum(int(row["iterations"]) for row in mcts_launches)
    efficiency = {
        "naive_pair_level_search_count": 1906,
        "actual_single_search_uid_count": len(searched_results),
        "single_searches_avoided_by_uid_sharing": 1906 - len(searched_results),
        "naive_pair_single_terminal_evaluations": 52800,
        "actual_shared_single_terminal_evaluations": actual_single,
        "single_terminal_evaluations_saved": 52800 - actual_single,
        "post_single_unresolved_pairs": post_single_unresolved_pairs,
        "actual_mcts_search_roots": actual_mcts_roots,
        "mcts_searches_avoided_by_cross_threshold_reuse": post_single_unresolved_pairs - actual_mcts_roots,
        "naive_mcts_iteration_cap_units": post_single_unresolved_pairs * 200,
        "actual_mcts_iterations": mcts_iterations,
        "iteration_cap_units_saved_estimate": post_single_unresolved_pairs * 200 - mcts_iterations,
        "threshold_route_replays": sum(row["threshold_replays"] for row in results) + len(phase64_replays),
        "state_capture_replays": sum(row["state_capture_replays"] for row in results),
    }
    atomic_csv(output_root / "metrics/search_reuse_efficiency.csv", [efficiency])
    compute_rows = [
        {"metric": "full_work_uids", "value": len(results), "unit": "uids"},
        {"metric": "searched_unique_w_uids", "value": len(searched_results), "unit": "uids"},
        {"metric": "single_terminal_routes", "value": actual_single, "unit": "routes"},
        {"metric": "mcts_search_roots", "value": actual_mcts_roots, "unit": "roots"},
        {"metric": "mcts_iterations", "value": mcts_iterations, "unit": "iterations"},
        {"metric": "physical_search_terminal_evaluations", "value": sum(row["physical_terminal_evaluations"] for row in results), "unit": "routes"},
        {"metric": "threshold_route_replays", "value": efficiency["threshold_route_replays"], "unit": "replays"},
        {"metric": "state_capture_replays", "value": efficiency["state_capture_replays"], "unit": "replays"},
        {"metric": "sample_gpu_seconds", "value": sum(float(row["elapsed_seconds"]) for row in results), "unit": "gpu_seconds"},
        {"metric": "sample_gpu_hours", "value": sum(float(row["elapsed_seconds"]) for row in results) / 3600.0, "unit": "gpu_hours"},
    ]
    for rank in range(4):
        completion = read_json(output_root / f"work/full/rank{rank:02d}/complete.json")
        compute_rows.append(
            {"metric": f"rank_{rank}_worker_wall_seconds", "value": completion["worker_elapsed_seconds"], "unit": "seconds"}
        )
    atomic_csv(output_root / "metrics/compute_summary.csv", compute_rows)
    overall = [
        {
            "searched_unique_w_uids": len(searched_results),
            "single_searches_launched": len(searched_results),
            "successful_new_single_routes": len(new_single_routes),
            "mcts_roots_launched": actual_mcts_roots,
            "successful_new_mcts_routes": len(new_mcts_routes),
            "existing_routes_recaptured": sum(row["route_origin"].startswith("existing_") for row in routes),
            "preservation_routes": sum(row["route_origin"] == "preservation_full" for row in routes),
            "global_routes": len(routes),
            "routed_state_rows": len(states),
            "unresolved_missing_pairs": len(remaining),
        }
    ]
    atomic_csv(output_root / "metrics/overall_search_summary.csv", overall)
    _plots(output_root, coverage, dataset_rows, depth_rows, efficiency)
    _write_summaries(
        output_root=output_root,
        contract=contract,
        coverage=coverage,
        corpus_stats=corpus_stats,
        dataset_rows=dataset_rows,
        depth_rows=depth_rows,
        efficiency=efficiency,
        overall=overall[0],
        routes=routes,
    )
    run_summary = {
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        "searched_unique_w_uids": len(searched_results),
        "pair_outcome_counts": dict(Counter(row["outcome_class"] for row in pair_rows)),
        "route_origin_counts": dict(Counter(row["route_origin"] for row in routes)),
        "global_routes": len(routes),
        "routed_state_rows": len(states),
        "state_shards": len(verified_shards),
        "mcts_roots": actual_mcts_roots,
        "mcts_iterations": mcts_iterations,
        "threshold_coverage": {row["operating_point"]: row for row in coverage},
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "run_summary.json", run_summary)
    print(json.dumps(run_summary, sort_keys=True))


def _write_summaries(
    *,
    output_root: Path,
    contract: Mapping[str, Any],
    coverage: Sequence[Mapping[str, Any]],
    corpus_stats: Sequence[Mapping[str, Any]],
    dataset_rows: Sequence[Mapping[str, Any]],
    depth_rows: Sequence[Mapping[str, Any]],
    efficiency: Mapping[str, Any],
    overall: Mapping[str, Any],
    routes: Sequence[Mapping[str, Any]],
) -> None:
    by_point = {str(row["operating_point"]): row for row in coverage}
    corpus_by_point = {str(row["operating_point"]): row for row in corpus_stats}
    supported_dataset = [row for row in dataset_rows if int(row["triggered_wrong"]) > 0]
    supported_depth = [row for row in depth_rows if int(row["triggered_wrong"]) > 0]
    dataset_min = min(supported_dataset, key=lambda row: float(row["known_corrective_coverage"]))
    dataset_max = max(supported_dataset, key=lambda row: float(row["known_corrective_coverage"]))
    depth_min = min(supported_depth, key=lambda row: float(row["total_bounded_correctability"]))
    depth_max = max(supported_depth, key=lambda row: float(row["total_bounded_correctability"]))

    coverage_lines = [
        "| Point | Triggered W | Existing covered | New single | New MCTS | Known corrective | Unresolved | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in POINTS:
        row = by_point[point]
        coverage_lines.append(
            f"| {point} | {row['triggered_wrong']} | {row['existing_wrong_covered']} | "
            f"{row['new_single_fixable']} | {row['new_mcts_fixable']} | "
            f"{row['known_corrective_supervision']} | {row['unresolved']} | "
            f"{float(row['conditional_corrective_coverage']):.4f} |"
        )
    summary = f"""# Robust-gate corrective-search summary

Contract: `{contract['contract_sha256']}`. Thresholds are frozen at """ + ", ".join(
        f"{point}={contract['operating_point_thresholds'][point]:.16g}" for point in POINTS
    ) + f""" with `{contract['trigger_comparison']}` comparison semantics.

{chr(10).join(coverage_lines)}

1. **Unique W UIDs searched:** {overall['searched_unique_w_uids']:,}.
2. **Single searches launched:** {overall['single_searches_launched']:,}, one exhaustive search per missing-union UID.
3. **MCTS roots launched:** {overall['mcts_roots_launched']:,} at cap 200.
4. **Cross-threshold reuse:** single search avoided {efficiency['single_searches_avoided_by_uid_sharing']:,} duplicate pair-level launches and {efficiency['single_terminal_evaluations_saved']:,} terminal evaluations. MCTS sharing avoided {efficiency['mcts_searches_avoided_by_cross_threshold_reuse']:,} pair-root launches after single resolution.
5. **P98 known corrective:** {by_point['P98']['known_corrective_supervision']:,}/{by_point['P98']['triggered_wrong']:,}.
6. **P95 known corrective:** {by_point['P95']['known_corrective_supervision']:,}/{by_point['P95']['triggered_wrong']:,}.
7. **P90 known corrective:** {by_point['P90']['known_corrective_supervision']:,}/{by_point['P90']['triggered_wrong']:,}.
8. **Unresolved:** P98={by_point['P98']['unresolved']:,}, P95={by_point['P95']['unresolved']:,}, P90={by_point['P90']['unresolved']:,}. These are unresolved at the frozen budget, not proven unfixable.
9. **New single contribution:** P98={by_point['P98']['new_single_fixable']:,}, P95={by_point['P95']['new_single_fixable']:,}, P90={by_point['P90']['new_single_fixable']:,} pair-level outcomes; {overall['successful_new_single_routes']:,} successful new single routes were retained globally.
10. **New MCTS contribution:** P98={by_point['P98']['new_mcts_fixable']:,}, P95={by_point['P95']['new_mcts_fixable']:,}, P90={by_point['P90']['new_mcts_fixable']:,} additional pair-level outcomes; {overall['successful_new_mcts_routes']:,} successful new MCTS routes were retained globally.
11. **Dataset/source variation:** observed known-corrective coverage spans {float(dataset_min['known_corrective_coverage']):.4f} ({dataset_min['source_regime']} {dataset_min['dataset']} {dataset_min['operating_point']}) to {float(dataset_max['known_corrective_coverage']):.4f} ({dataset_max['source_regime']} {dataset_max['dataset']} {dataset_max['operating_point']}); cells and support are in `metrics/dataset_source_breakdown.csv`.
12. **Trigger-depth variation:** bounded correctability spans {float(depth_min['total_bounded_correctability']):.4f} ({depth_min['operating_point']} {depth_min['trigger_depth_bin']}) to {float(depth_max['total_bounded_correctability']):.4f} ({depth_max['operating_point']} {depth_max['trigger_depth_bin']}). This is descriptive and not a causal depth claim.
13. **Replay/provenance:** PASS. All {len(routes):,} retained routes have threshold-specific exact replay evidence, current LMMS correctness, and contract/model/code/schema/source-bound routed states.

No Stage-2 training, Stage-1 change, threshold change, test deployment, or external evaluation was performed.
"""
    _atomic_bytes(output_root / "summaries/robust_gate_corrective_search_summary.md", summary.encode())

    corrective_sets: dict[str, set[str]] = {}
    route_sets: dict[str, set[str]] = {}
    corpus_lines = [
        "| Point | Preservation C bases | Single-corpus W bases | MCTS-corpus W bases | Single/MCTS overlap | Unresolved W | Retained routes | Routed state rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in POINTS:
        row = corpus_by_point[point]
        single_rows = read_jsonl(output_root / f"threshold_corpora/{point}/single.jsonl")
        mcts_rows = read_jsonl(output_root / f"threshold_corpora/{point}/mcts.jsonl")
        corrective_sets[point] = {
            str(item["uid"]) for item in [*single_rows, *mcts_rows]
        }
        route_sets[point] = {
            str(item["route_id"]) for item in [*single_rows, *mcts_rows]
        }
        corpus_lines.append(
            f"| {point} | {row['preservation_bases']} | {row['single_corpus_w_bases']} | "
            f"{row['mcts_corpus_w_bases']} | {row['single_mcts_corpus_overlap_bases']} | "
            f"{row['unresolved_w_bases']} | {row['retained_routes']} | {row['routed_state_rows']} |"
        )
    triple_uids = len(set.intersection(*(corrective_sets[point] for point in POINTS)))
    triple_routes = len(set.intersection(*(route_sets[point] for point in POINTS)))
    overlap_lines = []
    for left, right in (("P98", "P95"), ("P98", "P90"), ("P95", "P90")):
        overlap_lines.append(
            f"- {left}/{right}: {len(corrective_sets[left] & corrective_sets[right]):,} corrective W bases; "
            f"{len(route_sets[left] & route_sets[right]):,} shared corrective route IDs."
        )
    corpus_summary = f"""# Threshold-specific Stage-2 corpus summary

Each threshold view starts at that threshold's own frozen trigger layer, even when the underlying route state shard was captured once from an earlier valid trigger.

{chr(10).join(corpus_lines)}

Cross-threshold overlap:

{chr(10).join(overlap_lines)}
- P98/P95/P90 intersection: {triple_uids:,} corrective W bases and {triple_routes:,} shared corrective route IDs.

The single and MCTS corpus columns count route-family support and may overlap on a base UID because all replay-valid routes are retained. Pair-level outcome classes remain mutually exclusive in `work/final_pair_outcomes.jsonl`.
"""
    _atomic_bytes(output_root / "summaries/threshold_corpus_summary.md", corpus_summary.encode())

    healthy = all(
        int(corpus_by_point[point]["preservation_bases"]) > 0
        and int(corpus_by_point[point]["single_corpus_w_bases"]) > 0
        for point in POINTS
    )
    recommendation = f"""# Next Stage-2 training recommendation

Threshold-specific corpus health: **{'PASS' if healthy else 'NOT ESTABLISHED'}**.

If separately authorized, use the same frozen Stage-2 V1 READ/WRITE router architecture, optimizer, sampling logic, loss, and validation evaluator in three independent runs:

1. P98 preservation + single corrective.
2. P95 preservation + single corrective.
3. P90 preservation + single corrective.

Keep MCTS corrective supervision as a matched second ablation after the preservation+single comparison. Select the downstream operating point using validation final accuracy, W-to-C, C-to-W, and C-to-C—not corpus size alone. No training was launched in this phase.
"""
    _atomic_bytes(output_root / "summaries/next_stage2_training_recommendation.md", recommendation.encode())


def _required_artifacts() -> list[str]:
    required = [
        "protocol.md",
        "frozen_protocol.json",
        "manifests/missing_search_union.jsonl",
        "manifests/remaining_unresolved_pairs.jsonl",
        "single_search/per_uid_results.jsonl",
        "single_search/successful_routes.jsonl",
        "single_search/threshold_compatibility.jsonl",
        "mcts/launched_searches.jsonl",
        "mcts/successful_routes.jsonl",
        "mcts/threshold_compatibility.jsonl",
        "mcts/per_root_summary.csv",
        "routes/global_route_store.jsonl",
        "routes/replay_results.jsonl",
        "states/feature_schema.json",
        "states/routed_state_manifest.jsonl",
        "metrics/overall_search_summary.csv",
        "metrics/threshold_coverage.csv",
        "metrics/dataset_source_breakdown.csv",
        "metrics/trigger_depth_breakdown.csv",
        "metrics/search_reuse_efficiency.csv",
        "metrics/compute_summary.csv",
        "figures/corrective_coverage_by_threshold.png",
        "figures/single_vs_mcts_by_threshold.png",
        "figures/dataset_source_corrective_coverage.png",
        "figures/trigger_depth_correctability.png",
        "figures/search_reuse_savings.png",
        "summaries/robust_gate_corrective_search_summary.md",
        "summaries/threshold_corpus_summary.md",
        "summaries/next_stage2_training_recommendation.md",
        "run_summary.json",
    ]
    for point in POINTS:
        required.extend(
            f"threshold_corpora/{point}/{name}.jsonl"
            for name in ("preservation", "single", "mcts", "unresolved")
        )
    return required


def verify_artifact_manifest(output_root: Path, manifest: Mapping[str, Any]) -> None:
    if not manifest.get("passed"):
        raise RuntimeError("artifact manifest is not passing")
    for relative, expected in manifest["files"].items():
        path = output_root / str(relative)
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"artifact hash mismatch: {relative}")


def finalize(config_path: Path) -> None:
    contract, output_root = load_contract(config_path, verify_model=True)
    run_summary = read_json(output_root / "run_summary.json")
    if not run_summary.get("passed") or run_summary["contract_sha256"] != contract["contract_sha256"]:
        raise RuntimeError("passing aggregate summary is required")
    required = _required_artifacts()
    missing = [relative for relative in required if not (output_root / relative).is_file()]
    if missing:
        raise RuntimeError(f"required robust-search artifacts are missing: {missing}")

    routes = read_jsonl(output_root / "routes/global_route_store.jsonl")
    states = read_jsonl(output_root / "states/routed_state_manifest.jsonl")
    if len(routes) != int(run_summary["global_routes"]) or len(states) != int(run_summary["routed_state_rows"]):
        raise RuntimeError("route/state aggregate counts differ")
    route_by_id = {str(row["route_id"]): row for row in routes}
    if len(route_by_id) != len(routes):
        raise RuntimeError("route IDs are not unique")
    state_files = sorted({str(row["feature_file"]) for row in states})
    if len(state_files) != int(run_summary["state_shards"]):
        raise RuntimeError("state-shard count differs")
    for relative in state_files:
        path = output_root / relative
        route_hashes = {
            str(row["feature_file_sha256"])
            for row in routes
            if str(row["feature_file"]) == relative
        }
        if len(route_hashes) != 1 or not path.is_file() or file_sha256(path) not in route_hashes:
            raise RuntimeError(f"route/state shard hash binding differs: {relative}")
        _validate_saved_state_shard(path, contract, mode="full")

    for point in POINTS:
        for kind in ("preservation", "single", "mcts"):
            for row in read_jsonl(output_root / f"threshold_corpora/{point}/{kind}.jsonl"):
                route = route_by_id.get(str(row["route_id"]))
                if route is None or point not in route["valid_points"]:
                    raise RuntimeError(f"threshold corpus route is invalid: {point}/{row['route_id']}")
                if (
                    row["contract_sha256"] != contract["contract_sha256"]
                    or int(row["state_tensor_stop"]) - int(row["state_tensor_start"])
                    != 28 - int(row["trigger_layer"])
                    or row["feature_file_sha256"] != route["feature_file_sha256"]
                ):
                    raise RuntimeError(f"threshold corpus provenance differs: {point}/{row['route_id']}")

    files = {relative: file_sha256(output_root / relative) for relative in required}
    files.update({relative: file_sha256(output_root / relative) for relative in state_files})
    artifact_manifest = {
        "schema_version": "robust_gate_corrective_search_artifact_manifest_v1",
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        "searched_unique_w_uids": run_summary["searched_unique_w_uids"],
        "pair_outcome_counts": run_summary["pair_outcome_counts"],
        "global_routes": len(routes),
        "routed_state_rows": len(states),
        "verified_state_shards": len(state_files),
        "files": dict(sorted(files.items())),
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "artifact_manifest.json", artifact_manifest)
    verify_artifact_manifest(output_root, artifact_manifest)
    print(json.dumps(artifact_manifest, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("prepare", "worker", "finalize-smoke", "aggregate", "finalize")
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
    elif args.command == "aggregate":
        aggregate(args.config)
    else:
        finalize(args.config)


if __name__ == "__main__":
    main()
