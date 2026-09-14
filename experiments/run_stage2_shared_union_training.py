#!/usr/bin/env python3
"""Run the frozen shared-union Stage-2 A/B training and threshold comparison."""

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
from dense_failure_stage2.shared_union import (  # noqa: E402
    OPERATING_POINTS,
    balanced_uid_draws,
    merge_threshold_routes,
    routes_by_uid,
    sample_mcts_layers,
    sample_preservation_layers,
    sample_single_layers,
    summarize_schedule,
    trigger_layer,
)
from dense_failure_stage2.v1_router import (  # noqa: E402
    ACTION_NAMES,
    ACTION_TO_INDEX,
    summarize_action_predictions,
)
from experiments.run_stage2_v1_training_revised import (  # noqa: E402
    _binary_to,
    _generate,
    _load_model,
    _nvidia_memory,
    _physical_device_index,
    _prepare_binary,
    _router,
    _stack_route_states,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/stage2_shared_union_threshold_comparison_v1.json"
EXPERIMENT_DIRS = {
    "A": "experiment_A_single",
    "B": "experiment_B_single_plus_mcts",
}
ROUTE_SOURCE_FILES = {
    "preservation_full": "union_preservation_manifest.jsonl",
    "single": "union_single_manifest.jsonl",
    "mcts": "union_mcts_manifest.jsonl",
}
BOUND_CODE = (
    "configs/stage2_shared_union_threshold_comparison_v1.json",
    "dense_failure_stage2/shared_union.py",
    "dense_failure_stage2/v1_router.py",
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


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    allowed = (PROJECT_ROOT.resolve(), Path("/mnt/hyemin").resolve())
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed):
        raise ValueError(f"path escapes allowed roots: {value}")
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number} is not an object")
            rows.append(row)
    return rows


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
    if fieldnames is None:
        if not rows:
            raise ValueError(f"cannot infer columns for empty CSV: {path}")
        fieldnames = list(rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode())


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def command_output(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result.stdout.strip()


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "stage2_shared_union_config_v1":
        raise ValueError("unsupported shared-union config")
    if int(config["world_size"]) != 4:
        raise ValueError("shared-union training requires four ranks")
    if tuple(config["router"]["action_order"]) != ACTION_NAMES:
        raise ValueError("router action order differs from executor")
    sampling, training = config["sampling"], config["training"]
    if int(sampling["w_draws_per_epoch"]) + int(sampling["c_draws_per_epoch"]) != int(
        sampling["global_draws_per_epoch"]
    ):
        raise ValueError("C/W draw totals are inconsistent")
    if int(sampling["global_draws_per_epoch"]) % int(config["world_size"]):
        raise ValueError("epoch draws are not divisible across ranks")
    if int(training["total_global_optimizer_updates"]) != int(training["epochs"]) * int(
        training["global_optimizer_updates_per_epoch"]
    ):
        raise ValueError("training update budget is inconsistent")
    if tuple(point for point in config["operating_points"] if point in OPERATING_POINTS) != OPERATING_POINTS:
        raise ValueError("operating-point order must be P98/P95/P90")
    if set(config["operating_points"]) != {*OPERATING_POINTS, "comparison"}:
        raise ValueError("operating-point config contains unsupported fields")
    return config


def _sample_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    sample = row.get("sample", row)
    required = (
        "uid",
        "dataset",
        "prompt",
        "answer",
        "local_image_path",
        "image_content_sha256",
        "max_new_tokens",
    )
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError(f"sample {row.get('uid')} lacks fields: {missing}")
    return {
        key: sample.get(key)
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


def _hash_rank(seed: int, value: str) -> str:
    return sha256(f"{seed}:{value}".encode()).hexdigest()


def _select_action_balanced_routes(
    route_rows: Sequence[Mapping[str, Any]],
    *,
    per_action: int,
    seed: int,
    excluded_uids: Sequence[str] = (),
) -> dict[str, dict[str, Any]]:
    by_uid = routes_by_uid(route_rows)
    selected: dict[str, dict[str, Any]] = {}
    excluded = {str(uid) for uid in excluded_uids}
    for action in ACTION_NAMES[1:]:
        candidates = []
        for uid, routes in by_uid.items():
            if uid in selected or uid in excluded:
                continue
            matching = [row for row in routes if action in row["actions"]]
            if matching:
                candidates.append((uid, matching))
        candidates.sort(key=lambda item: _hash_rank(seed, f"{action}:{item[0]}"))
        if len(candidates) < per_action:
            raise RuntimeError(f"cannot select {per_action} overfit routes for {action}")
        for uid, matching in candidates[:per_action]:
            selected[uid] = min(
                (dict(row) for row in matching),
                key=lambda row: _hash_rank(seed, str(row["route_id"])),
            )
    if len(selected) != 3 * per_action:
        raise RuntimeError("overfit route cohort is not action-balanced")
    return selected


def _schedule_rows(
    *,
    experiment: str,
    route_sets: Mapping[str, Sequence[Mapping[str, Any]]],
    preservation: Sequence[Mapping[str, Any]],
    epochs: int,
    family_draws: Mapping[str, int],
    c_draws: int,
    seed: int,
    world_size: int,
    name: str,
    fixed_routes: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grouped = {family: routes_by_uid(rows) for family, rows in route_sets.items()}
    c_by_uid = {str(row["uid"]): dict(row) for row in preservation}
    output: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        rng = random.Random((int(seed) << 16) + epoch)
        draws: list[tuple[str, str, dict[str, Any]]] = []
        if fixed_routes is not None:
            for uid in sorted(fixed_routes, key=lambda item: _hash_rank(seed + epoch, item)):
                route = dict(fixed_routes[uid])
                draws.append((str(route["route_source"]), uid, route))
        else:
            for family, count in family_draws.items():
                if int(count) == 0:
                    continue
                if family not in grouped or not grouped[family]:
                    raise RuntimeError(f"missing {family} route population")
                for uid in balanced_uid_draws(sorted(grouped[family]), int(count), rng):
                    draws.append((family, uid, dict(rng.choice(grouped[family][uid]))))
        for uid in balanced_uid_draws(sorted(c_by_uid), int(c_draws), rng):
            draws.append(("preservation_full", uid, dict(c_by_uid[uid])))
        rng.shuffle(draws)
        if len(draws) % int(world_size):
            raise RuntimeError(f"{name} epoch is not divisible across ranks")
        for draw_index, (family, uid, route) in enumerate(draws):
            local_rng = random.Random(
                int(_hash_rank(seed, f"{name}:{epoch}:{draw_index}:{uid}:{route['route_id']}")[:16], 16)
            )
            if family == "single":
                layers = sample_single_layers(route, local_rng)
            elif family == "mcts":
                layers = sample_mcts_layers(route, local_rng, state_cap=8)
            elif family == "preservation_full":
                layers = sample_preservation_layers(route, local_rng)
            else:
                raise ValueError(f"unsupported draw family: {family}")
            actions = [str(route["actions"][layer]) for layer in layers]
            output.append(
                {
                    "schema_version": "stage2_shared_union_draw_v1",
                    "experiment": experiment,
                    "schedule": name,
                    "epoch": epoch,
                    "draw_index": draw_index,
                    "worker_rank": draw_index % int(world_size),
                    "global_update": epoch * (len(draws) // int(world_size))
                    + draw_index // int(world_size)
                    + 1,
                    "draw_kind": "C" if family == "preservation_full" else "W",
                    "route_source": family,
                    "uid": uid,
                    "dataset": route["dataset"],
                    "source_regime": route["source_regime"],
                    "route_id": route["route_id"],
                    "activation_layer": int(route["activation_layer"]),
                    "valid_operating_points": route["valid_operating_points"],
                    "actions": list(route["actions"]),
                    "selected_layers": layers,
                    "selected_actions": actions,
                    "selected_action_indices": [ACTION_TO_INDEX[action] for action in actions],
                }
            )
    return output


def _read_threshold_routes(
    sources: Mapping[str, Path], route_type: str
) -> dict[str, list[dict[str, Any]]]:
    suffix = "preservation" if route_type == "preservation_full" else route_type
    return {
        point: read_jsonl(sources[f"{point.lower()}_{suffix}"])
        for point in OPERATING_POINTS
    }


def _verify_model_snapshot(snapshot: Path, expected: Mapping[str, str]) -> None:
    actual_names = sorted(path.name for path in snapshot.iterdir() if path.is_file())
    if actual_names != sorted(expected):
        raise RuntimeError("model snapshot inventory differs from Phase 65")
    for name, digest in expected.items():
        if file_sha256(snapshot / name) != str(digest):
            raise RuntimeError(f"model snapshot hash mismatch: {name}")


def prepare(config_path: Path) -> None:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    sources = {name: resolve_path(path) for name, path in config["sources"].items()}

    phase65 = read_json(sources["phase65_contract"])
    if canonical_hash(phase65) != phase65.get("contract_sha256"):
        raise RuntimeError("Phase-65 frozen contract hash is invalid")
    phase65_artifacts = read_json(sources["phase65_artifact_manifest"])
    verify_artifact_manifest(sources["phase65_artifact_manifest"].parent, phase65_artifacts)
    if phase65_artifacts["contract_sha256"] != phase65["contract_sha256"]:
        raise RuntimeError("Phase-65 artifact and contract IDs differ")

    configured_thresholds = {
        point: float(config["operating_points"][point]) for point in OPERATING_POINTS
    }
    if configured_thresholds != {
        point: float(phase65["operating_point_thresholds"][point])
        for point in OPERATING_POINTS
    }:
        raise RuntimeError("configured thresholds differ from Phase 65")
    robust_gate = read_json(sources["robust_stage1_gate"])
    if configured_thresholds["P98"] != float(robust_gate["global_threshold"]):
        raise RuntimeError("P98 differs from the frozen robust Stage-1 gate")
    if [str(item["sha256"]) for item in robust_gate["checkpoints"]] != [
        str(item["sha256"])
        for item in phase65["static_config"].get("stage1_checkpoints", robust_gate["checkpoints"])
    ]:
        # Phase 65 may not duplicate checkpoint rows. The source file hash below remains binding.
        if "stage1_checkpoints" in phase65["static_config"]:
            raise RuntimeError("Stage-1 checkpoint set differs from Phase 65")

    union_routes: dict[str, list[dict[str, Any]]] = {}
    expected_counts = {"preservation_full": 106, "single": 1688, "mcts": 725}
    expected_uid_counts = {"preservation_full": 106, "single": 270, "mcts": 216}
    expanded_counts: dict[str, int] = {}
    for route_type in ROUTE_SOURCE_FILES:
        threshold_rows = _read_threshold_routes(sources, route_type)
        expanded_counts[route_type] = sum(map(len, threshold_rows.values()))
        union_routes[route_type] = merge_threshold_routes(
            threshold_rows, expected_route_type=route_type
        )
        if len(union_routes[route_type]) != expected_counts[route_type]:
            raise RuntimeError(f"unexpected deduplicated {route_type} route count")
        if len({row["uid"] for row in union_routes[route_type]}) != expected_uid_counts[route_type]:
            raise RuntimeError(f"unexpected deduplicated {route_type} UID count")

    work_rows = read_jsonl(sources["phase65_work_manifest"])
    work_by_uid = {str(row["uid"]): row for row in work_rows}
    if len(work_by_uid) != len(work_rows):
        raise RuntimeError("Phase-65 work manifest contains duplicate UIDs")
    train_uids = {
        str(row["uid"]) for rows in union_routes.values() for row in rows
    }
    if not train_uids.issubset(work_by_uid):
        raise RuntimeError("union route sample metadata is incomplete")
    train_samples = [
        {
            "uid": uid,
            "dataset": work_by_uid[uid]["dataset"],
            "source_regime": work_by_uid[uid]["source_regime"],
            "sample": _sample_payload(work_by_uid[uid]),
            "dense_output": work_by_uid[uid]["dense_output"],
        }
        for uid in sorted(train_uids)
    ]

    score_rows = [
        row
        for row in read_jsonl(sources["historical_validation_scores"])
        if row.get("historical_split") == "val"
    ]
    scores = {str(row["uid"]): row for row in score_rows}
    validation_source = read_jsonl(sources["historical_validation_manifest"])
    if len(validation_source) != 800 or len(scores) != 800:
        raise RuntimeError("Historical validation is not the frozen 800-row population")
    if {str(row["uid"]) for row in validation_source} != set(scores):
        raise RuntimeError("Historical validation score/sample UIDs differ")
    validation = []
    for row in sorted(validation_source, key=lambda item: str(item["uid"])):
        uid = str(row["uid"])
        score = scores[uid]
        if bool(row["dense_wrong"]) != bool(score["current_dense_wrong"]):
            raise RuntimeError(f"validation dense-label mismatch: {uid}")
        triggers = {
            point: trigger_layer(score, configured_thresholds[point])
            for point in OPERATING_POINTS
        }
        validation.append(
            {
                "schema_version": "stage2_shared_union_validation_v1",
                "uid": uid,
                "dataset": str(row["dataset"]),
                "source_regime": "historical",
                "dense_wrong": bool(row["dense_wrong"]),
                "dense_correct": not bool(row["dense_wrong"]),
                "trigger_layers": triggers,
                "sample": row["sample"],
                "dense_output": row["dense_output"],
            }
        )

    train_groups = {work_by_uid[uid]["sample"]["image_group_id"] for uid in train_uids}
    train_hashes = {work_by_uid[uid]["sample"]["image_content_sha256"] for uid in train_uids}
    val_uids = {row["uid"] for row in validation}
    val_groups = {row["sample"]["image_group_id"] for row in validation}
    val_hashes = {row["sample"]["image_content_sha256"] for row in validation}
    overlap = {
        "uid_overlap": len(train_uids & val_uids),
        "image_group_overlap": len(train_groups & val_groups),
        "image_sha256_overlap": len(train_hashes & val_hashes),
    }
    if any(overlap.values()):
        raise RuntimeError(f"union/validation leakage detected: {overlap}")

    single_overfit_a = _select_action_balanced_routes(
        union_routes["single"], per_action=16, seed=int(config["seed"]) + 100
    )
    single_overfit_b = _select_action_balanced_routes(
        union_routes["single"], per_action=8, seed=int(config["seed"]) + 200
    )
    mcts_overfit_b = _select_action_balanced_routes(
        union_routes["mcts"],
        per_action=8,
        seed=int(config["seed"]) + 300,
        excluded_uids=list(single_overfit_b),
    )
    fixed_overfit = {
        "A": single_overfit_a,
        "B": {**single_overfit_b, **mcts_overfit_b},
    }
    if any(len(rows) != 48 for rows in fixed_overfit.values()):
        raise RuntimeError("overfit W cohort is not 48 unique UIDs")
    c_order = sorted(
        union_routes["preservation_full"],
        key=lambda row: _hash_rank(int(config["seed"]), f"C:{row['uid']}"),
    )
    overfit_c = c_order[:24]

    schedules: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for experiment in ("A", "B"):
        experiment_dir = EXPERIMENT_DIRS[experiment]
        family_key = f"experiment_{experiment}_w_family"
        family_draws = {
            family: int(count) for family, count in config["sampling"][family_key].items()
        }
        full = _schedule_rows(
            experiment=experiment,
            route_sets={"single": union_routes["single"], "mcts": union_routes["mcts"]},
            preservation=union_routes["preservation_full"],
            epochs=int(config["training"]["epochs"]),
            family_draws=family_draws,
            c_draws=int(config["sampling"]["c_draws_per_epoch"]),
            seed=int(config["seed"]),
            world_size=int(config["world_size"]),
            name="full",
        )
        overfit = _schedule_rows(
            experiment=experiment,
            route_sets={"single": union_routes["single"], "mcts": union_routes["mcts"]},
            preservation=overfit_c,
            epochs=int(config["overfit"]["epochs"]),
            family_draws={},
            c_draws=24,
            seed=int(config["seed"]) + 5000,
            world_size=int(config["world_size"]),
            name="overfit",
            fixed_routes=fixed_overfit[experiment],
        )
        expected_full_rows = int(config["training"]["total_global_optimizer_updates"]) * int(
            config["world_size"]
        )
        if len(full) != expected_full_rows:
            raise RuntimeError(f"Experiment {experiment} full update budget differs")
        if len(overfit) != int(config["overfit"]["epochs"]) * 72:
            raise RuntimeError(f"Experiment {experiment} overfit schedule differs")
        schedules[(experiment, "full")] = full
        schedules[(experiment, "overfit")] = overfit

    output_root.mkdir(parents=True)
    for route_type, filename in ROUTE_SOURCE_FILES.items():
        atomic_jsonl(output_root / "corpus" / filename, union_routes[route_type])
    atomic_jsonl(output_root / "work/train_samples.jsonl", train_samples)
    atomic_jsonl(output_root / "work/validation_manifest.jsonl", validation)
    for experiment in ("A", "B"):
        directory = output_root / EXPERIMENT_DIRS[experiment]
        atomic_jsonl(directory / "work/full_schedule.jsonl", schedules[(experiment, "full")])
        atomic_jsonl(directory / "work/overfit_schedule.jsonl", schedules[(experiment, "overfit")])
        atomic_json(
            directory / "config/experiment_config.json",
            {
                "experiment": experiment,
                "w_family_draws": config["sampling"][f"experiment_{experiment}_w_family"],
                "sampling": config["sampling"],
                "router": config["router"],
                "training": config["training"],
                "seed": config["seed"],
            },
        )

    route_dedup_rows = []
    for route_type in ROUTE_SOURCE_FILES:
        route_dedup_rows.append(
            {
                "route_source": route_type,
                "threshold_expanded_rows": expanded_counts[route_type],
                "unique_routes": len(union_routes[route_type]),
                "deduplicated_rows": expanded_counts[route_type] - len(union_routes[route_type]),
                "unique_uids": len({row["uid"] for row in union_routes[route_type]}),
            }
        )
    atomic_csv(output_root / "corpus/route_dedup_summary.csv", route_dedup_rows)
    validity_rows = []
    for route_type, rows in union_routes.items():
        for point in OPERATING_POINTS:
            subset = [row for row in rows if point in row["valid_operating_points"]]
            validity_rows.append(
                {
                    "operating_point": point,
                    "route_source": route_type,
                    "routes": len(subset),
                    "uids": len({row["uid"] for row in subset}),
                }
            )
    atomic_csv(output_root / "corpus/threshold_validity_summary.csv", validity_rows)
    loader_audit = {
        "schema_version": "stage2_shared_union_loader_audit_v1",
        "passed": True,
        "union_routes": {kind: len(rows) for kind, rows in union_routes.items()},
        "union_uids": {
            kind: len({row["uid"] for row in rows}) for kind, rows in union_routes.items()
        },
        "all_union_uids": len(train_uids),
        "all_union_image_groups": len(train_groups),
        "validation": {
            "records": len(validation),
            "source_regime": "historical",
            "canonical_status": "unavailable_without_stage2_training_leakage",
            **overlap,
            "triggered": {
                point: sum(row["trigger_layers"][point] is not None for row in validation)
                for point in OPERATING_POINTS
            },
        },
        "full_schedule": {
            experiment: summarize_schedule(schedules[(experiment, "full")])
            for experiment in ("A", "B")
        },
        "sampling_checks": {
            "route_deduplicated": True,
            "uid_balanced": True,
            "threshold_metadata_retained": True,
            "actual_route_actions_retained": True,
            "c_to_w_draws_each_epoch": "350:698",
            "single_random4": True,
            "mcts_state_cap": int(config["sampling"]["mcts_state_cap"]),
        },
    }
    atomic_json(output_root / "corpus/data_loader_audit.json", loader_audit)

    source_hashes = {name: file_sha256(path) for name, path in sources.items()}
    bound_hashes = {relative: file_sha256(resolve_path(relative)) for relative in BOUND_CODE}
    internal_relatives = [
        "corpus/union_preservation_manifest.jsonl",
        "corpus/union_single_manifest.jsonl",
        "corpus/union_mcts_manifest.jsonl",
        "corpus/route_dedup_summary.csv",
        "corpus/threshold_validity_summary.csv",
        "corpus/data_loader_audit.json",
        "work/train_samples.jsonl",
        "work/validation_manifest.jsonl",
    ]
    for experiment in ("A", "B"):
        directory = EXPERIMENT_DIRS[experiment]
        internal_relatives.extend(
            [
                f"{directory}/work/full_schedule.jsonl",
                f"{directory}/work/overfit_schedule.jsonl",
                f"{directory}/config/experiment_config.json",
            ]
        )
    internal_hashes = {
        relative: file_sha256(output_root / relative) for relative in internal_relatives
    }
    snapshot = resolve_path(config["model"]["snapshot_path"])
    model_hashes = phase65["model_snapshot_sha256"]
    _verify_model_snapshot(snapshot, model_hashes)
    contract: dict[str, Any] = {
        "schema_version": "stage2_shared_union_contract_v1",
        "created_at": utc_now(),
        "static_config": config,
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
            "execution": "direct_four_gpu",
        },
        "phase65_contract_sha256": phase65["contract_sha256"],
        "phase65_artifact_manifest_sha256": file_sha256(sources["phase65_artifact_manifest"]),
        "robust_stage1_gate_sha256": file_sha256(sources["robust_stage1_gate"]),
        "source_sha256": source_hashes,
        "bound_code_sha256": bound_hashes,
        "internal_manifest_sha256": internal_hashes,
        "model_snapshot_sha256": model_hashes,
        "population": loader_audit,
        "selection": {
            "checkpoint": "final global update only",
            "updates_per_experiment": int(config["training"]["total_global_optimizer_updates"]),
            "validation_population": "Historical frozen val 800",
            "validation_openings": {point: 1 for point in OPERATING_POINTS},
            "canonical_validation": "not available without Stage-2 training leakage",
        },
        "review_reconciliation": {
            "reviewer_verdict": "revise",
            "accepted": "Historical-800-only primary rollout with Canonical cells explicitly unavailable",
            "rejected": "Canonical OOF 4000 as Stage-2 validation because union training uses those UIDs",
            "claim_scope": "Historical validation only; no all-source deployment claim",
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_protocol.json", contract)
    protocol = f"""# Shared-union Stage-2 frozen protocol

- Contract: `{contract['contract_sha256']}`
- Phase-65 source contract: `{phase65['contract_sha256']}`
- Union A/B/C: 106 preservation UIDs, 270 single W UIDs / 1,688 routes, 216 MCTS W UIDs / 725 routes.
- Router: unchanged shared READ/WRITE V1 architecture; no layer, Stage-1, source, or dataset input.
- A: 698 single W + 350 C draws/epoch. B: 349 single + 349 MCTS W + 350 C draws/epoch.
- Both: 12 epochs, 3,144 updates, AdamW 3e-4, constant schedule, plain four-way CE, final-update checkpoint only.
- Rollout: same checkpoint at strict P98/P95/P90 over the zero-overlap frozen Historical validation 800.
- Canonical validation: unavailable; Canonical OOF 4,000 contains Stage-2 union-training UIDs and is not used as held-out evidence.
- Experiment B runs only if A has a passing implementation/overfit/training record and at least one threshold with W-to-C > 0 and net correction > 0.
- Stop: after the matched A/B (conditional) validation comparison; no test or Stage-1 change.
"""
    _atomic_bytes(output_root / "protocol.md", protocol.encode())
    print(
        json.dumps(
            {
                "prepared": True,
                "contract_sha256": contract["contract_sha256"],
                "union_uids": len(train_uids),
                "validation_overlap": overlap,
            },
            sort_keys=True,
        )
    )


def verify_contract(config_path: Path, *, verify_model: bool = False) -> tuple[dict[str, Any], Path]:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if canonical_hash(contract) != contract.get("contract_sha256"):
        raise RuntimeError("shared-union frozen contract hash mismatch")
    if contract["static_config"] != config:
        raise RuntimeError("active config differs from the frozen shared-union contract")
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
            resolve_path(config["model"]["snapshot_path"]), contract["model_snapshot_sha256"]
        )
    return contract, output_root


def _experiment_root(output_root: Path, experiment: str) -> Path:
    if experiment not in EXPERIMENT_DIRS:
        raise ValueError("experiment must be A or B")
    return output_root / EXPERIMENT_DIRS[experiment]


def implementation_smoke(config_path: Path, experiment: str, device_index: int) -> None:
    contract, output_root = verify_contract(config_path, verify_model=True)
    config = contract["static_config"]
    exp_root = _experiment_root(output_root, experiment)
    before = _nvidia_memory(device_index)
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    configure_dense_determinism(int(config["seed"]), config["backend_settings"])
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    processor, base, wrapped = _load_model(config, device)
    after_model = _nvidia_memory(device_index)
    samples = {row["uid"]: row for row in read_jsonl(output_root / "work/train_samples.jsonl")}
    schedule = read_jsonl(exp_root / "work/full_schedule.jsonl")
    first_epoch_w = [row for row in schedule if row["epoch"] == 0 and row["draw_kind"] == "W"]
    worst_uid = max(
        {row["uid"] for row in first_epoch_w},
        key=lambda uid: int(samples[uid]["dense_output"]["prompt_token_count"]),
    )
    row = next(item for item in first_epoch_w if item["uid"] == worst_uid)
    sample = samples[worst_uid]["sample"]
    inputs, _metadata = build_dense_inputs(processor, sample, device)
    meta = _prepare_binary(processor, wrapped, sample, device)
    dense_route = ["FULL"] * int(config["model"]["decoder_layers"])
    dense_output = capture_four_action_route(
        wrapped, {}, dense_route, prepared_inputs=meta, use_cache=True, native_full_rows=True
    )
    dense_ids, _dense_text, _dense_score = _generate(
        processor, wrapped, dense_output, inputs["input_ids"], sample
    )
    expected_ids = list(samples[worst_uid]["dense_output"]["generated_token_ids"])
    native_token_parity = dense_ids == expected_ids
    del dense_output

    routed = capture_four_action_route(
        wrapped,
        {},
        row["actions"],
        prepared_inputs=meta,
        use_cache=False,
        native_full_rows=True,
    )
    router = _router(config, device).train()
    text, visual, text_mask, visual_mask = _stack_route_states(
        routed, row["selected_layers"], device
    )
    targets = torch.tensor(row["selected_action_indices"], dtype=torch.long, device=device)
    logits = router(text, visual, text_mask=text_mask, visual_mask=visual_mask)
    loss = F.cross_entropy(logits, targets)
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise RuntimeError("implementation smoke produced non-finite output")
    loss.backward()
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device) / 2**20)
    peak_reserved = int(torch.cuda.max_memory_reserved(device) / 2**20)
    after_backward = _nvidia_memory(device_index)
    headroom = before["free_mib"] - peak_reserved
    router_gradients = all(parameter.grad is not None for parameter in router.parameters())
    backbone_frozen = all(parameter.grad is None for parameter in base.parameters())
    passed = all(
        (
            native_token_parity,
            tuple(logits.shape) == (len(row["selected_layers"]), 4),
            router_gradients,
            backbone_frozen,
            headroom >= 2048,
        )
    )
    report = {
        "schema_version": "stage2_shared_union_implementation_smoke_v1",
        "experiment": experiment,
        "passed": passed,
        "contract_sha256": contract["contract_sha256"],
        "physical_device_index": _physical_device_index(device_index),
        "worst_uid": worst_uid,
        "route_source": row["route_source"],
        "prompt_tokens": samples[worst_uid]["dense_output"]["prompt_token_count"],
        "visual_tokens": samples[worst_uid]["dense_output"]["visual_token_count"],
        "selected_layers": row["selected_layers"],
        "selected_actions": row["selected_actions"],
        "native_all_full_token_parity": native_token_parity,
        "logit_shape": list(logits.shape),
        "loss": float(loss.item()),
        "router_gradients_present": router_gradients,
        "backbone_gradients_all_none": backbone_frozen,
        "gpu_before": before,
        "gpu_after_model_load": after_model,
        "gpu_after_backward": after_backward,
        "process_peak_allocated_mib": peak_allocated,
        "process_peak_reserved_mib": peak_reserved,
        "conservative_headroom_mib": headroom,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(exp_root / "smoke/implementation_smoke.json", report)
    _atomic_bytes(
        exp_root / "smoke/implementation_smoke.md",
        f"""# Experiment {experiment} implementation smoke

- Passed: **{passed}**
- Native dense token parity: {native_token_parity}
- Route family: {row['route_source']}; selected actions: {row['selected_actions']}
- Router gradients present: {router_gradients}; Qwen gradients absent: {backbone_frozen}
- Peak allocated/reserved: {peak_allocated}/{peak_reserved} MiB; conservative headroom: {headroom} MiB
""".encode(),
    )
    print(json.dumps(report, sort_keys=True))
    if not passed:
        raise RuntimeError("implementation smoke failed")


def _metrics_from_confusion(confusion: torch.Tensor) -> dict[str, Any]:
    targets, predictions = [], []
    matrix = confusion.detach().cpu().long()
    for target in range(4):
        for prediction in range(4):
            count = int(matrix[target, prediction])
            targets.extend([target] * count)
            predictions.extend([prediction] * count)
    return summarize_action_predictions(targets=targets, predictions=predictions)


def train_worker(config_path: Path, experiment: str, mode: str) -> None:
    if mode not in {"overfit", "full"}:
        raise ValueError("training mode must be overfit or full")
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    exp_root = _experiment_root(output_root, experiment)
    implementation = read_json(exp_root / "smoke/implementation_smoke.json")
    if not implementation.get("passed"):
        raise RuntimeError("implementation smoke is not passing")
    if mode == "full" and not read_json(exp_root / "smoke/overfit_gate.json").get("passed"):
        raise RuntimeError("overfit gate is not passing")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != int(config["world_size"]):
        raise RuntimeError("training must use exactly four distributed ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    configure_dense_determinism(int(config["seed"]) + rank, config["backend_settings"])
    dist.init_process_group(backend="nccl")
    processor, base, wrapped = _load_model(config, device)
    torch.manual_seed(int(config["seed"]) + (0 if mode == "full" else 5000))
    router = _router(config, device)
    ddp = DistributedDataParallel(router, device_ids=[local_rank], output_device=local_rank)
    settings = config["training"] if mode == "full" else config["overfit"]
    optimizer = torch.optim.AdamW(
        ddp.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    schedule_all = read_jsonl(exp_root / f"work/{mode}_schedule.jsonl")
    schedule = [row for row in schedule_all if int(row["worker_rank"]) == rank]
    expected_updates = (
        int(config["training"]["total_global_optimizer_updates"])
        if mode == "full"
        else int(config["overfit"]["epochs"]) * 18
    )
    if len(schedule) != expected_updates:
        raise RuntimeError(f"rank {rank} has {len(schedule)} rather than {expected_updates} draws")
    samples = {
        row["uid"]: row["sample"] for row in read_jsonl(output_root / "work/train_samples.jsonl")
    }
    cache = {}
    log_path = exp_root / ("training/train_log.jsonl" if mode == "full" else "smoke/overfit_train_log.jsonl")
    if rank == 0 and log_path.exists():
        raise RuntimeError(f"refusing to overwrite training log: {log_path}")
    dist.barrier()
    epochs = int(settings["epochs"])
    updates_per_rank_epoch = len(schedule) // epochs
    epoch_rows_all = []
    started = time.monotonic()
    for epoch in range(epochs):
        ddp.train()
        local_loss = torch.zeros(2, dtype=torch.float64, device=device)
        local_confusion = torch.zeros(4, 4, dtype=torch.long, device=device)
        rows = schedule[epoch * updates_per_rank_epoch : (epoch + 1) * updates_per_rank_epoch]
        for row in rows:
            uid = str(row["uid"])
            if uid not in cache:
                cache[uid] = _binary_to(_prepare_binary(processor, wrapped, samples[uid], device), "cpu")
            meta = _binary_to(cache[uid], device)
            with torch.inference_mode():
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
            targets = torch.tensor(row["selected_action_indices"], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(text).all() or not torch.isfinite(visual).all():
                raise RuntimeError(f"non-finite routed state for {uid}")
            logits = ddp(text, visual, text_mask=text_mask, visual_mask=visual_mask)
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError(f"non-finite router forward/loss for {uid}")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in ddp.parameters()
            ):
                raise RuntimeError(f"non-finite router gradient for {uid}")
            torch.nn.utils.clip_grad_norm_(
                ddp.parameters(), float(settings["gradient_clip_norm"]), error_if_nonfinite=True
            )
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in ddp.parameters()):
                raise RuntimeError(f"non-finite router parameter after {uid}")
            predictions = logits.detach().argmax(dim=-1)
            local_loss[0] += float(loss.item())
            local_loss[1] += 1
            for target, prediction in zip(targets, predictions):
                local_confusion[int(target), int(prediction)] += 1
            del routed, text, visual, text_mask, visual_mask, targets, logits, loss

        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_confusion, op=dist.ReduceOp.SUM)
        metrics = _metrics_from_confusion(local_confusion)
        epoch_row = {
            "schema_version": "stage2_shared_union_training_epoch_v1",
            "contract_sha256": contract["contract_sha256"],
            "experiment": experiment,
            "mode": mode,
            "epoch": epoch + 1,
            "global_update": (epoch + 1) * updates_per_rank_epoch,
            "loss": float((local_loss[0] / local_loss[1]).item()),
            "accuracy": metrics["accuracy"],
            "non_full_recall": metrics["non_full_recall"],
            "recall": metrics["recall"],
            "predicted_distribution": metrics["predicted_distribution"],
            "target_distribution": metrics["target_distribution"],
            "confusion": local_confusion.cpu().tolist(),
            "elapsed_seconds": time.monotonic() - started,
        }
        epoch_rows_all.append(epoch_row)
        if rank == 0:
            append_jsonl(log_path, epoch_row)
            print(json.dumps(epoch_row, sort_keys=True), flush=True)

    if not all(parameter.grad is None for parameter in base.parameters()):
        raise RuntimeError("frozen Qwen unexpectedly acquired gradients")
    dist.barrier()
    if rank == 0:
        checkpoint = exp_root / (
            "training/final_checkpoint.pt" if mode == "full" else "smoke/overfit_checkpoint.pt"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": "stage2_shared_union_checkpoint_v1",
            "contract_sha256": contract["contract_sha256"],
            "experiment": experiment,
            "mode": mode,
            "global_update": expected_updates,
            "router_config": config["router"],
            "state_dict": {
                key: value.detach().cpu() for key, value in ddp.module.state_dict().items()
            },
        }
        temporary = checkpoint.with_name(f".{checkpoint.name}.tmp.{os.getpid()}")
        torch.save(state, temporary)
        os.replace(temporary, checkpoint)
        checkpoint_sha = file_sha256(checkpoint)
        final_metrics = epoch_rows_all[-1]
        if mode == "overfit":
            first_loss, final_loss = float(epoch_rows_all[0]["loss"]), float(final_metrics["loss"])
            reduction = (first_loss - final_loss) / first_loss
            supported = [
                action
                for action in ACTION_NAMES[1:]
                if final_metrics["target_distribution"][action] > 0
            ]
            gate_config = config["overfit"]
            predicted_non_full = 1.0 - final_metrics["predicted_distribution"]["FULL"]
            checks = {
                "loss_reduction": reduction >= float(gate_config["minimum_loss_reduction"]),
                "full_recall": final_metrics["recall"]["FULL"] >= float(gate_config["minimum_full_recall"]),
                "non_full_recall": final_metrics["non_full_recall"] >= float(gate_config["minimum_non_full_recall"]),
                "supported_action_recall": all(
                    final_metrics["recall"][action]
                    >= float(gate_config["minimum_supported_action_recall"])
                    for action in supported
                ),
                "predicted_non_full_fraction": predicted_non_full
                >= float(gate_config["minimum_predicted_non_full_fraction"]),
            }
            gate = {
                "schema_version": "stage2_shared_union_overfit_gate_v1",
                "experiment": experiment,
                "contract_sha256": contract["contract_sha256"],
                "passed": all(checks.values()),
                "checks": checks,
                "first_loss": first_loss,
                "final_loss": final_loss,
                "loss_reduction": reduction,
                "final_metrics": final_metrics,
                "checkpoint_sha256": checkpoint_sha,
            }
            atomic_json(exp_root / "smoke/overfit_gate.json", gate)
            metric_rows = []
            distribution_rows = []
            for item in epoch_rows_all:
                metric_rows.append(
                    {
                        "epoch": item["epoch"],
                        "loss": item["loss"],
                        "accuracy": item["accuracy"],
                        "full_recall": item["recall"]["FULL"],
                        "non_full_recall": item["non_full_recall"],
                        "read_only_recall": item["recall"]["READ_ONLY"],
                        "write_only_recall": item["recall"]["WRITE_ONLY"],
                        "ignore_recall": item["recall"]["IGNORE"],
                    }
                )
                distribution_rows.append({"epoch": item["epoch"], **item["predicted_distribution"]})
            atomic_csv(exp_root / "smoke/overfit_metrics.csv", metric_rows)
            atomic_csv(exp_root / "smoke/overfit_action_distribution.csv", distribution_rows)
            print(json.dumps(gate, sort_keys=True), flush=True)
        else:
            manifest = {
                "schema_version": "stage2_shared_union_checkpoint_manifest_v1",
                "experiment": experiment,
                "contract_sha256": contract["contract_sha256"],
                "checkpoints": [
                    {
                        "path": "training/final_checkpoint.pt",
                        "sha256": checkpoint_sha,
                        "global_update": expected_updates,
                        "selection_eligible": True,
                    }
                ],
                "epoch_checkpoints": [],
            }
            atomic_json(exp_root / "training/checkpoint_manifest.json", manifest)
            atomic_json(
                exp_root / "training/selected_checkpoint.json",
                {
                    "schema_version": "stage2_shared_union_selected_checkpoint_v1",
                    "experiment": experiment,
                    "contract_sha256": contract["contract_sha256"],
                    "selection_rule": "final_global_update_only",
                    "path": "training/final_checkpoint.pt",
                    "sha256": checkpoint_sha,
                    "global_update": expected_updates,
                },
            )
    dist.barrier()
    atomic_json(
        exp_root / f"work/{mode}_rank{rank:02d}.complete.json",
        {
            "passed": True,
            "experiment": experiment,
            "rank": rank,
            "updates": len(schedule),
            "contract_sha256": contract["contract_sha256"],
            "completed_at": utc_now(),
        },
    )
    dist.destroy_process_group()


def rollout_worker(config_path: Path, experiment: str, point: str) -> None:
    if point not in OPERATING_POINTS:
        raise ValueError("rollout point must be P98, P95, or P90")
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    exp_root = _experiment_root(output_root, experiment)
    selected = read_json(exp_root / "training/selected_checkpoint.json")
    checkpoint_path = exp_root / selected["path"]
    if file_sha256(checkpoint_path) != selected["sha256"]:
        raise RuntimeError("selected Stage-2 checkpoint hash mismatch")
    for rank_index in range(int(config["world_size"])):
        completion = exp_root / f"work/full_rank{rank_index:02d}.complete.json"
        if not completion.is_file() or not read_json(completion).get("passed"):
            raise RuntimeError("full training rank completion is incomplete")

    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != int(config["world_size"]):
        raise RuntimeError("free rollout must use exactly four ranks")
    rollout_root = exp_root / "rollout" / point
    worker_result = rollout_root / f"work/rank{rank:02d}.jsonl"
    worker_complete = rollout_root / f"work/rank{rank:02d}.complete.json"
    if worker_result.exists() or worker_complete.exists():
        raise RuntimeError(f"refusing to overwrite Experiment {experiment} {point} rank {rank}")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    configure_dense_determinism(
        int(config["seed"]) + 9000 + rank, config["backend_settings"]
    )
    processor, _base, wrapped = _load_model(config, device)
    router = _router(config, device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint["contract_sha256"] != contract["contract_sha256"]
        or checkpoint["experiment"] != experiment
    ):
        raise RuntimeError("checkpoint provenance differs from active rollout")
    router.load_state_dict(checkpoint["state_dict"], strict=True)

    validation = read_jsonl(output_root / "work/validation_manifest.jsonl")
    triggered = sorted(
        [row for row in validation if row["trigger_layers"][point] is not None],
        key=lambda row: str(row["uid"]),
    )
    assigned = [row for index, row in enumerate(triggered) if index % world_size == rank]
    started = time.monotonic()
    for count, row in enumerate(assigned, 1):
        sample = row["sample"]
        inputs, _metadata = build_dense_inputs(processor, sample, device)
        meta = _prepare_binary(processor, wrapped, sample, device)
        trigger = int(row["trigger_layers"][point])
        chosen: list[dict[str, Any]] = []

        def selector(layer_index, text_states, visual_states, current_meta):
            if layer_index < trigger:
                chosen.append({"layer": layer_index, "action": "FULL", "active": False})
                return "FULL"
            logits = router(
                text_states.detach().clone(),
                visual_states.detach().clone(),
                text_mask=current_meta.text_valid_mask.detach().clone(),
                visual_mask=current_meta.visual_valid_mask.detach().clone(),
            )
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"non-finite router logits for {row['uid']} at L{layer_index}")
            probabilities = logits.float().softmax(dim=-1)[0]
            action_index = int(probabilities.argmax().item())
            action = ACTION_NAMES[action_index]
            chosen.append(
                {
                    "layer": layer_index,
                    "action": action,
                    "active": True,
                    "probabilities": {
                        name: float(probabilities[index].item())
                        for index, name in enumerate(ACTION_NAMES)
                    },
                }
            )
            return action

        output = capture_online_four_action_route(
            wrapped,
            {},
            selector,
            prepared_inputs=meta,
            use_cache=True,
            native_full_rows=True,
        )
        token_ids, text, score = _generate(
            processor, wrapped, output, inputs["input_ids"], sample
        )
        active_actions = [item["action"] for item in chosen if item["active"]]
        non_full_layers = [
            int(item["layer"])
            for item in chosen
            if item["active"] and item["action"] != "FULL"
        ]
        result = {
            "schema_version": "stage2_shared_union_rollout_row_v1",
            "contract_sha256": contract["contract_sha256"],
            "experiment": experiment,
            "operating_point": point,
            "threshold": float(config["operating_points"][point]),
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
            "routed_correct": score.correct,
            "routed_wrong": not score.correct,
            "transition": ("W" if row["dense_wrong"] else "C")
            + "→"
            + ("C" if score.correct else "W"),
            "actions": [item["action"] for item in chosen],
            "action_rows": chosen,
            "post_trigger_action_counts": dict(Counter(active_actions)),
            "post_trigger_actions": len(active_actions),
            "non_full_count": len(non_full_layers),
            "any_non_full": bool(non_full_layers),
            "first_non_full_layer": non_full_layers[0] if non_full_layers else None,
            "trigger_to_first_non_full_delay": (
                non_full_layers[0] - trigger if non_full_layers else None
            ),
            "full_fraction_after_trigger": active_actions.count("FULL") / len(active_actions),
            "worker_rank": rank,
        }
        append_jsonl(worker_result, result)
        if count % 10 == 0:
            print(
                json.dumps(
                    {
                        "experiment": experiment,
                        "point": point,
                        "rank": rank,
                        "completed": count,
                        "assigned": len(assigned),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del output, meta, inputs
    atomic_json(
        worker_complete,
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "experiment": experiment,
            "operating_point": point,
            "checkpoint_sha256": selected["sha256"],
            "rank": rank,
            "expected": len(assigned),
            "completed": len(assigned),
            "elapsed_seconds": time.monotonic() - started,
            "completed_at": utc_now(),
        },
    )


def _aggregate_point(
    *,
    contract: Mapping[str, Any],
    output_root: Path,
    experiment: str,
    point: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = contract["static_config"]
    exp_root = _experiment_root(output_root, experiment)
    rollout_root = exp_root / "rollout" / point
    selected = read_json(exp_root / "training/selected_checkpoint.json")
    validation = read_jsonl(output_root / "work/validation_manifest.jsonl")
    expected = [row for row in validation if row["trigger_layers"][point] is not None]
    worker_rows = []
    for rank in range(int(config["world_size"])):
        completion_path = rollout_root / f"work/rank{rank:02d}.complete.json"
        if not completion_path.is_file():
            raise RuntimeError(f"Experiment {experiment} {point} rank {rank} is missing")
        completion = read_json(completion_path)
        rows = read_jsonl(rollout_root / f"work/rank{rank:02d}.jsonl")
        if not completion.get("passed") or len(rows) != int(completion["expected"]):
            raise RuntimeError(f"Experiment {experiment} {point} rank {rank} is partial")
        worker_rows.extend(rows)
    if Counter(row["uid"] for row in expected) != Counter(row["uid"] for row in worker_rows):
        raise RuntimeError(f"Experiment {experiment} {point} rollout coverage differs")
    routed = {row["uid"]: row for row in worker_rows}
    per_sample = []
    for row in validation:
        if row["trigger_layers"][point] is not None:
            per_sample.append(routed[row["uid"]])
        else:
            correct = bool(row["dense_correct"])
            per_sample.append(
                {
                    "schema_version": "stage2_shared_union_rollout_row_v1",
                    "contract_sha256": contract["contract_sha256"],
                    "experiment": experiment,
                    "operating_point": point,
                    "threshold": float(config["operating_points"][point]),
                    "checkpoint_sha256": selected["sha256"],
                    "uid": row["uid"],
                    "dataset": row["dataset"],
                    "source_regime": "historical",
                    "triggered": False,
                    "trigger_layer": None,
                    "dense_correct": correct,
                    "dense_wrong": not correct,
                    "dense_generated_answer": row["dense_output"]["generated_answer"],
                    "dense_generated_token_ids": row["dense_output"]["generated_token_ids"],
                    "routed_generated_answer": row["dense_output"]["generated_answer"],
                    "routed_generated_token_ids": row["dense_output"]["generated_token_ids"],
                    "lmms_metric": row["dense_output"]["lmms_eval_metric"],
                    "lmms_score": row["dense_output"]["lmms_eval_per_sample_score"],
                    "routed_correct": correct,
                    "routed_wrong": not correct,
                    "transition": "C→C" if correct else "W→W",
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
    atomic_jsonl(rollout_root / "per_sample_results.jsonl", per_sample)
    transitions = Counter(row["transition"] for row in per_sample)
    triggered_rows = [row for row in per_sample if row["triggered"]]
    post_counts = Counter()
    for row in triggered_rows:
        post_counts.update(row["post_trigger_action_counts"])
    total_post = sum(post_counts.values())
    delays = [
        row["trigger_to_first_non_full_delay"] for row in triggered_rows if row["any_non_full"]
    ]
    dense_correct = sum(row["dense_correct"] for row in per_sample)
    routed_correct = sum(row["routed_correct"] for row in per_sample)
    metrics = {
        "schema_version": "stage2_shared_union_rollout_metrics_v1",
        "contract_sha256": contract["contract_sha256"],
        "experiment": experiment,
        "operating_point": point,
        "threshold": float(config["operating_points"][point]),
        "checkpoint_sha256": selected["sha256"],
        "validation_scope": "historical_frozen_val_800",
        "validation_samples": len(per_sample),
        "triggered_samples": len(triggered_rows),
        "triggered_w": sum(row["triggered"] and row["dense_wrong"] for row in per_sample),
        "triggered_c": sum(row["triggered"] and row["dense_correct"] for row in per_sample),
        "dense_correct": dense_correct,
        "routed_correct": routed_correct,
        "dense_accuracy": dense_correct / len(per_sample),
        "routed_accuracy": routed_correct / len(per_sample),
        "delta_accuracy": (routed_correct - dense_correct) / len(per_sample),
        "transitions": {
            name: transitions[name] for name in ("W→C", "W→W", "C→C", "C→W")
        },
        "net_corrections": transitions["W→C"] - transitions["C→W"],
        "w_to_c_rescue_rate": transitions["W→C"]
        / (transitions["W→C"] + transitions["W→W"]),
        "c_to_c_preservation_rate": transitions["C→C"]
        / (transitions["C→C"] + transitions["C→W"]),
        "triggered_any_non_full_fraction": sum(row["any_non_full"] for row in triggered_rows)
        / len(triggered_rows),
        "mean_non_full_actions_per_triggered": sum(
            row["non_full_count"] for row in triggered_rows
        )
        / len(triggered_rows),
        "post_trigger_action_counts": dict(post_counts),
        "post_trigger_action_distribution": {
            action: post_counts[action] / total_post for action in ACTION_NAMES
        },
        "mean_first_non_full_layer": (
            sum(row["first_non_full_layer"] for row in triggered_rows if row["any_non_full"])
            / len(delays)
            if delays
            else None
        ),
        "mean_trigger_to_first_non_full_delay": sum(delays) / len(delays) if delays else None,
    }
    atomic_json(rollout_root / "overall_metrics.json", metrics)
    atomic_csv(
        rollout_root / "transition_counts.csv",
        [
            {"transition": name, "count": transitions[name]}
            for name in ("W→C", "W→W", "C→C", "C→W")
        ],
    )
    return metrics, per_sample


def finalize_experiment(config_path: Path, experiment: str) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    exp_root = _experiment_root(output_root, experiment)
    selected = read_json(exp_root / "training/selected_checkpoint.json")
    if file_sha256(exp_root / selected["path"]) != selected["sha256"]:
        raise RuntimeError("selected checkpoint changed before aggregation")
    point_metrics = {}
    point_rows = {}
    for point in OPERATING_POINTS:
        point_metrics[point], point_rows[point] = _aggregate_point(
            contract=contract,
            output_root=output_root,
            experiment=experiment,
            point=point,
        )

    threshold_rows = []
    behavior_rows = []
    breakdown_rows = []
    for point in OPERATING_POINTS:
        metrics = point_metrics[point]
        transitions = metrics["transitions"]
        threshold_rows.append(
            {
                "operating_point": point,
                "threshold": metrics["threshold"],
                "validation_scope": metrics["validation_scope"],
                "samples": metrics["validation_samples"],
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
                "mean_trigger_to_first_non_full_delay": metrics[
                    "mean_trigger_to_first_non_full_delay"
                ],
                **{
                    f"{action.lower()}_count": metrics["post_trigger_action_counts"].get(action, 0)
                    for action in ACTION_NAMES
                },
            }
        )
        for source in ("historical", "canonical"):
            for dataset in ("gqa", "chartqa", "textvqa"):
                if source == "canonical":
                    breakdown_rows.append(
                        {
                            "operating_point": point,
                            "source_regime": source,
                            "dataset": dataset,
                            "availability": "unavailable_stage2_training_leakage",
                            "samples": 0,
                            "triggered_w": "",
                            "triggered_c": "",
                            "w_to_c": "",
                            "c_to_w": "",
                            "dense_accuracy": "",
                            "routed_accuracy": "",
                            "delta_accuracy": "",
                        }
                    )
                    continue
                cell = [row for row in point_rows[point] if row["dataset"] == dataset]
                cell_transitions = Counter(row["transition"] for row in cell)
                dense_correct = sum(row["dense_correct"] for row in cell)
                routed_correct = sum(row["routed_correct"] for row in cell)
                breakdown_rows.append(
                    {
                        "operating_point": point,
                        "source_regime": source,
                        "dataset": dataset,
                        "availability": "heldout",
                        "samples": len(cell),
                        "triggered_w": sum(
                            row["triggered"] and row["dense_wrong"] for row in cell
                        ),
                        "triggered_c": sum(
                            row["triggered"] and row["dense_correct"] for row in cell
                        ),
                        "w_to_c": cell_transitions["W→C"],
                        "c_to_w": cell_transitions["C→W"],
                        "dense_accuracy": dense_correct / len(cell),
                        "routed_accuracy": routed_correct / len(cell),
                        "delta_accuracy": (routed_correct - dense_correct) / len(cell),
                    }
                )
    atomic_csv(exp_root / "metrics/threshold_comparison.csv", threshold_rows)
    atomic_csv(exp_root / "metrics/action_behavior.csv", behavior_rows)
    atomic_csv(exp_root / "metrics/dataset_source_breakdown.csv", breakdown_rows)

    train_log = read_jsonl(exp_root / "training/train_log.jsonl")
    if len(train_log) != int(config["training"]["epochs"]):
        raise RuntimeError("training log does not contain the frozen epoch count")
    final_train = train_log[-1]
    action_rows = [
        {
            "scope": "final_epoch_frozen_train_diagnostic",
            "action": action,
            "recall": final_train["recall"][action],
            "target_fraction": final_train["target_distribution"][action],
            "predicted_fraction": final_train["predicted_distribution"][action],
        }
        for action in ACTION_NAMES
    ]
    atomic_csv(exp_root / "teacher_forced/action_metrics.csv", action_rows)
    atomic_csv(
        exp_root / "teacher_forced/predicted_action_distribution.csv",
        [
            {"action": action, "fraction": final_train["predicted_distribution"][action]}
            for action in ACTION_NAMES
        ],
    )
    confusion_rows = []
    for target_index, target in enumerate(ACTION_NAMES):
        for prediction_index, prediction in enumerate(ACTION_NAMES):
            confusion_rows.append(
                {
                    "target": target,
                    "prediction": prediction,
                    "count": final_train["confusion"][target_index][prediction_index],
                }
            )
    atomic_csv(exp_root / "teacher_forced/confusion_matrix.csv", confusion_rows)
    overfit = read_json(exp_root / "smoke/overfit_gate.json")
    implementation = read_json(exp_root / "smoke/implementation_smoke.json")
    positive_points = [
        point
        for point in OPERATING_POINTS
        if point_metrics[point]["transitions"]["W→C"] > 0
        and point_metrics[point]["net_corrections"] > 0
    ]
    health = {
        "schema_version": "stage2_shared_union_experiment_health_v1",
        "experiment": experiment,
        "contract_sha256": contract["contract_sha256"],
        "implementation_smoke_passed": bool(implementation["passed"]),
        "overfit_gate_passed": bool(overfit["passed"]),
        "finite_training": all(
            float(row["loss"]) == float(row["loss"]) and abs(float(row["loss"])) < float("inf")
            for row in train_log
        ),
        "positive_net_operating_points": positive_points,
        "a_success": (
            experiment == "A"
            and bool(implementation["passed"])
            and bool(overfit["passed"])
            and bool(positive_points)
        ),
    }
    atomic_json(exp_root / "summaries/health_gate.json", health)
    best = sorted(
        threshold_rows,
        key=lambda row: (
            -float(row["routed_accuracy"]),
            int(row["c_to_w"]),
            -float(row["preservation_rate"]),
            OPERATING_POINTS.index(str(row["operating_point"])),
        ),
    )[0]
    union_a = read_jsonl(output_root / "corpus/union_preservation_manifest.jsonl")
    union_b = read_jsonl(output_root / "corpus/union_single_manifest.jsonl")
    union_c = read_jsonl(output_root / "corpus/union_mcts_manifest.jsonl")
    if experiment == "A":
        summary = f"""# Experiment A: union preservation + single

1. UNION_A contains **{len({row['uid'] for row in union_a})}** unique preservation bases.
2. UNION_B contains **{len({row['uid'] for row in union_b})}** unique single-fixable W bases and {len(union_b)} routes.
3. Implementation/overfit passed: **{implementation['passed']} / {overfit['passed']}**. Final train non-FULL recall: {final_train['non_full_recall']:.4f}.
4. P98/P95/P90 W-to-C: **{point_metrics['P98']['transitions']['W→C']} / {point_metrics['P95']['transitions']['W→C']} / {point_metrics['P90']['transitions']['W→C']}**.
5. P98/P95/P90 C-to-W: **{point_metrics['P98']['transitions']['C→W']} / {point_metrics['P95']['transitions']['C→W']} / {point_metrics['P90']['transitions']['C→W']}**.
6. Best Historical-validation net correction is **{best['operating_point']} = {best['net_correction']:+d}**.
7. P90 C preservation: **{point_metrics['P90']['c_to_c_preservation_rate']:.4f}**.
8. Dataset/source results are in `metrics/dataset_source_breakdown.csv`; Canonical cells are explicitly unavailable because the Canonical OOF population supplies Stage-2 union supervision.
9. Experiment A health gate for proceeding to B: **{health['a_success']}**.
"""
    else:
        summary = f"""# Experiment B: union preservation + single + MCTS

1. UNION_C adds **{len({row['uid'] for row in union_c})}** unique MCTS-fixable W bases and {len(union_c)} routes.
2. Final train non-FULL recall is **{final_train['non_full_recall']:.4f}**; threshold action behavior is in `metrics/action_behavior.csv`.
3. P98/P95/P90 W-to-C: **{point_metrics['P98']['transitions']['W→C']} / {point_metrics['P95']['transitions']['W→C']} / {point_metrics['P90']['transitions']['W→C']}**.
4. P98/P95/P90 C-to-W: **{point_metrics['P98']['transitions']['C→W']} / {point_metrics['P95']['transitions']['C→W']} / {point_metrics['P90']['transitions']['C→W']}**.
5. The threshold-dependent comparison is frozen in `metrics/threshold_comparison.csv`.
6. Dataset effects are in `metrics/dataset_source_breakdown.csv`; Canonical held-out evidence is unavailable.
7. Best Experiment-B Historical-validation point: **{best['operating_point']}**, routed accuracy **{best['routed_accuracy']:.6f}**.
"""
    filename = "experiment_A_summary.md" if experiment == "A" else "experiment_B_summary.md"
    _atomic_bytes(exp_root / "summaries" / filename, summary.encode())
    print(json.dumps({"experiment": experiment, "health": health, "best": best}, sort_keys=True))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _plot_global(
    output_root: Path,
    experiment_rows: Mapping[str, Sequence[Mapping[str, str]]],
    final_experiment: str,
) -> None:
    colors = {"A": "#4472C4", "B": "#ED7D31"}
    xs = list(range(len(OPERATING_POINTS)))
    plt.figure(figsize=(6.4, 4.2))
    for experiment, rows in experiment_rows.items():
        indexed = {row["operating_point"]: row for row in rows}
        plt.plot(
            xs,
            [float(indexed[point]["routed_accuracy"]) for point in OPERATING_POINTS],
            marker="o",
            label=f"Experiment {experiment}",
            color=colors[experiment],
        )
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Dense")
    plt.xticks(xs, OPERATING_POINTS)
    plt.ylabel("Historical validation accuracy")
    plt.xlabel("Stage-1 operating point")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "figures/threshold_accuracy_tradeoff.png", dpi=160)
    plt.close()

    final_rows = {row["operating_point"]: row for row in experiment_rows[final_experiment]}
    width = 0.35
    plt.figure(figsize=(6.4, 4.2))
    plt.bar(
        [x - width / 2 for x in xs],
        [int(final_rows[point]["w_to_c"]) for point in OPERATING_POINTS],
        width,
        label="W→C",
    )
    plt.bar(
        [x + width / 2 for x in xs],
        [int(final_rows[point]["c_to_w"]) for point in OPERATING_POINTS],
        width,
        label="C→W",
    )
    plt.xticks(xs, OPERATING_POINTS)
    plt.ylabel("Samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "figures/threshold_rescue_regression.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.4, 4.2))
    for experiment in experiment_rows:
        exp_root = _experiment_root(output_root, experiment)
        behavior = {row["operating_point"]: row for row in _read_csv(exp_root / "metrics/action_behavior.csv")}
        plt.plot(
            xs,
            [1.0 - float(behavior[point]["post_trigger_full_fraction"]) for point in OPERATING_POINTS],
            marker="o",
            label=f"Experiment {experiment}",
            color=colors[experiment],
        )
    plt.xticks(xs, OPERATING_POINTS)
    plt.ylabel("Post-trigger non-FULL fraction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "figures/threshold_action_behavior.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.4, 4.2))
    for experiment, rows in experiment_rows.items():
        indexed = {row["operating_point"]: row for row in rows}
        plt.plot(
            xs,
            [float(indexed[point]["delta_accuracy"]) for point in OPERATING_POINTS],
            marker="o",
            label=("Single" if experiment == "A" else "Single + MCTS"),
            color=colors[experiment],
        )
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(xs, OPERATING_POINTS)
    plt.ylabel("Accuracy change")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "figures/single_vs_mcts_accuracy.png", dpi=160)
    plt.close()

    breakdown = _read_csv(
        _experiment_root(output_root, final_experiment) / "metrics/dataset_source_breakdown.csv"
    )
    datasets = ("gqa", "chartqa", "textvqa")
    plt.figure(figsize=(7.2, 4.3))
    for index, point in enumerate(OPERATING_POINTS):
        values = []
        for dataset in datasets:
            row = next(
                item
                for item in breakdown
                if item["operating_point"] == point
                and item["source_regime"] == "historical"
                and item["dataset"] == dataset
            )
            values.append(float(row["delta_accuracy"]))
        offsets = [x + (index - 1) * 0.24 for x in range(len(datasets))]
        plt.bar(offsets, values, 0.24, label=point)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(range(len(datasets)), [name.upper() for name in datasets])
    plt.ylabel("Historical validation accuracy change")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "figures/dataset_source_accuracy_change.png", dpi=160)
    plt.close()


def finalize(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    a_root = _experiment_root(output_root, "A")
    a_health = read_json(a_root / "summaries/health_gate.json")
    experiment_rows: dict[str, list[dict[str, str]]] = {
        "A": _read_csv(a_root / "metrics/threshold_comparison.csv")
    }
    b_completed = (_experiment_root(output_root, "B") / "summaries/experiment_B_summary.md").is_file()
    if bool(a_health["a_success"]) and not b_completed:
        raise RuntimeError("Experiment A passed the B gate but Experiment B is incomplete")
    if b_completed:
        experiment_rows["B"] = _read_csv(
            _experiment_root(output_root, "B") / "metrics/threshold_comparison.csv"
        )
    final_experiment = "B" if b_completed else "A"

    a_index = {row["operating_point"]: row for row in experiment_rows["A"]}
    b_index = {
        row["operating_point"]: row for row in experiment_rows.get("B", [])
    }
    comparison_rows = []
    for point in OPERATING_POINTS:
        a = a_index[point]
        b = b_index.get(point)
        comparison_rows.append(
            {
                "operating_point": point,
                "single_only_delta_accuracy": a["delta_accuracy"],
                "single_plus_mcts_delta_accuracy": b["delta_accuracy"] if b else "",
                "delta_w_to_c": int(b["w_to_c"]) - int(a["w_to_c"]) if b else "",
                "delta_c_to_w": int(b["c_to_w"]) - int(a["c_to_w"]) if b else "",
                "experiment_B_status": "completed" if b else "not_run_A_gate_failed",
            }
        )
    atomic_csv(output_root / "comparison/A_vs_B.csv", comparison_rows)
    final_rows = [dict(row, selected_experiment=final_experiment) for row in experiment_rows[final_experiment]]
    atomic_csv(output_root / "comparison/threshold_final_comparison.csv", final_rows)
    conservative_rank = {point: index for index, point in enumerate(OPERATING_POINTS)}
    ranked = sorted(
        final_rows,
        key=lambda row: (
            -float(row["routed_accuracy"]),
            int(row["c_to_w"]),
            -float(row["preservation_rate"]),
            conservative_rank[row["operating_point"]],
        ),
    )
    selected = ranked[0]
    tied_accuracy = [
        row
        for row in ranked
        if abs(float(row["routed_accuracy"]) - float(selected["routed_accuracy"])) < 1e-12
    ]
    selection = {
        "schema_version": "stage2_shared_union_selected_operating_point_v1",
        "contract_sha256": contract["contract_sha256"],
        "selected_experiment": final_experiment,
        "experiment_B_completed": b_completed,
        "operating_point": selected["operating_point"],
        "threshold": float(selected["threshold"]),
        "validation_scope": "historical_frozen_val_800",
        "claim_scope": "Historical validation only",
        "selection_rule": "highest routed accuracy, fewer C-to-W, higher C-to-C, more conservative",
        "metrics": {
            key: selected[key]
            for key in (
                "routed_accuracy",
                "delta_accuracy",
                "w_to_c",
                "c_to_w",
                "c_to_c",
                "net_correction",
            )
        },
        "accuracy_tie_count_before_tiebreak": len(tied_accuracy),
        "canonical_validation_status": "unavailable_without_stage2_training_leakage",
    }
    atomic_json(output_root / "comparison/selected_operating_point.json", selection)
    _plot_global(output_root, experiment_rows, final_experiment)

    w_to_c = int(selected["w_to_c"])
    c_to_w = int(selected["c_to_w"])
    net = int(selected["net_correction"])
    if net > 0:
        decision = (
            "Decision A — Shared Stage-2 works; a Historical-validation operating point is identified"
            if len(tied_accuracy) == 1
            else "Decision B — Shared Stage-2 works but the Historical-validation threshold remains tied"
        )
    elif c_to_w > w_to_c:
        decision = "Decision D — Regression dominates"
    else:
        decision = "Decision C — Stage-2 still under-generalizes"
    summary = f"""# Shared Stage-2 decision

## Decision

**{decision}.**

- Final planned training condition: Experiment {final_experiment}.
- Selected operating point on the frozen Historical validation set: **{selected['operating_point']}**.
- Dense/routed accuracy: {float(selected['dense_accuracy']):.6f} / {float(selected['routed_accuracy']):.6f} ({float(selected['delta_accuracy']):+.6f}).
- W-to-C / C-to-W / net: {w_to_c} / {c_to_w} / {net:+d}.
- C-to-C preservation: {float(selected['preservation_rate']):.6f}.
- Experiment B completed: {b_completed}; Experiment-A B gate: {a_health['a_success']}.

## Scope limitation

This is a Historical-800 validation result, not an all-source deployment claim. The available Canonical 4,000 are Stage-1 OOF only and contribute Stage-2 union supervision, so they are not a held-out Stage-2 evaluation set. Their requested source cells are recorded as unavailable rather than leaked.

No test set, new threshold, Stage-1 change, new search, or architecture/loss change was run.
"""
    _atomic_bytes(output_root / "summaries/shared_stage2_decision.md", summary.encode())

    required = [
        "protocol.md",
        "frozen_protocol.json",
        "corpus/union_preservation_manifest.jsonl",
        "corpus/union_single_manifest.jsonl",
        "corpus/union_mcts_manifest.jsonl",
        "corpus/route_dedup_summary.csv",
        "corpus/threshold_validity_summary.csv",
        "experiment_A_single/metrics/threshold_comparison.csv",
        "experiment_A_single/metrics/dataset_source_breakdown.csv",
        "experiment_A_single/metrics/action_behavior.csv",
        "experiment_A_single/summaries/experiment_A_summary.md",
        "comparison/A_vs_B.csv",
        "comparison/threshold_final_comparison.csv",
        "comparison/selected_operating_point.json",
        "figures/threshold_accuracy_tradeoff.png",
        "figures/threshold_rescue_regression.png",
        "figures/threshold_action_behavior.png",
        "figures/single_vs_mcts_accuracy.png",
        "figures/dataset_source_accuracy_change.png",
        "summaries/shared_stage2_decision.md",
    ]
    if b_completed:
        required.extend(
            [
                "experiment_B_single_plus_mcts/metrics/threshold_comparison.csv",
                "experiment_B_single_plus_mcts/metrics/dataset_source_breakdown.csv",
                "experiment_B_single_plus_mcts/metrics/action_behavior.csv",
                "experiment_B_single_plus_mcts/summaries/experiment_B_summary.md",
            ]
        )
    missing = [relative for relative in required if not (output_root / relative).is_file()]
    if missing:
        raise RuntimeError(f"required final artifacts are missing: {missing}")
    files = {
        str(path.relative_to(output_root)): file_sha256(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    manifest = {
        "schema_version": "stage2_shared_union_artifact_manifest_v1",
        "passed": True,
        "completed_at": utc_now(),
        "contract_sha256": contract["contract_sha256"],
        "experiment_A_passed": True,
        "experiment_A_health_gate": bool(a_health["a_success"]),
        "experiment_B_completed": b_completed,
        "selected_experiment": final_experiment,
        "selected_operating_point": selected["operating_point"],
        "validation_scope": "historical_frozen_val_800",
        "files": files,
    }
    atomic_json(output_root / "artifact_manifest.json", manifest)
    verify_artifact_manifest(output_root, manifest)
    print(json.dumps(manifest | {"files": len(files)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "implementation-smoke",
            "train",
            "rollout",
            "finalize-experiment",
            "finalize",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment", choices=("A", "B"))
    parser.add_argument("--mode", choices=("overfit", "full"))
    parser.add_argument("--point", choices=OPERATING_POINTS)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    if args.command == "prepare":
        prepare(config_path)
    elif args.command == "implementation-smoke":
        if not args.experiment:
            parser.error("implementation-smoke requires --experiment")
        implementation_smoke(config_path, args.experiment, args.device_index)
    elif args.command == "train":
        if not args.experiment or not args.mode:
            parser.error("train requires --experiment and --mode")
        train_worker(config_path, args.experiment, args.mode)
    elif args.command == "rollout":
        if not args.experiment or not args.point:
            parser.error("rollout requires --experiment and --point")
        rollout_worker(config_path, args.experiment, args.point)
    elif args.command == "finalize-experiment":
        if not args.experiment:
            parser.error("finalize-experiment requires --experiment")
        finalize_experiment(config_path, args.experiment)
    else:
        finalize(config_path)


if __name__ == "__main__":
    main()
