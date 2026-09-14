#!/usr/bin/env python3
"""Calibrate a frozen global Stage-2 non-FULL abstention margin."""

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
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from binary_policy.executor.four_action import (  # noqa: E402
    capture_four_action_route,
    capture_online_four_action_route,
)
from binary_policy.executor.inputs import build_binary_inputs  # noqa: E402
from dense_failure_stage1.runtime import build_dense_inputs, configure_dense_determinism  # noqa: E402
from dense_failure_stage2.abstention_margin import (  # noqa: E402
    ACTION_NAMES,
    artifact_hash,
    assign_group_folds,
    build_margin_grid,
    choose_action_from_logits,
    select_margin,
    summarize_margin_rollout,
    validate_result_matrix,
)
from experiments.run_stage2_v1_training_revised import (  # noqa: E402
    _binary_to,
    _generate,
    _load_model,
    _prepare_binary,
    _router,
    _stack_route_states,
)
from experiments.run_full_benchmark_end_to_end_eval import (  # noqa: E402
    Runtime as ExternalRuntime,
    _build_inputs as external_build_inputs,
    _generate as external_generate,
    _verify_images as external_verify_images,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/stage2_abstention_margin_v1.json"
BOUND_CODE = (
    "configs/stage2_abstention_margin_v1.json",
    "dense_failure_stage2/abstention_margin.py",
    "experiments/run_stage2_abstention_margin.py",
    "dense_failure_stage2/v1_router.py",
    "experiments/run_stage2_v1_training_revised.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/model.py",
    "dense_failure_stage1/runtime.py",
    "dense_failure_stage1/lmms_scoring.py",
    "dense_failure_stage2/full_benchmark_eval.py",
    "experiments/run_full_benchmark_end_to_end_eval.py",
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
    roots = (PROJECT_ROOT.resolve(), Path("/mnt/hyemin").resolve())
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise ValueError(f"path escapes allowed roots: {value}")
    return resolved


def command_output(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result.stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{number}")
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


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"cannot infer columns for empty CSV: {path}")
    columns = list(fieldnames or rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "stage2_abstention_margin_config_v1":
        raise ValueError("unsupported abstention-margin config")
    if int(config["world_size"]) != 4:
        raise ValueError("abstention calibration requires four workers")
    if tuple(config["stage2"]["action_order"]) != ACTION_NAMES:
        raise ValueError("action order differs from the frozen router")
    if float(config["stage1_operating_point"]["threshold"]) != 0.9061332901863008:
        raise ValueError("Stage-1 P90 threshold differs")
    if config["stage1_operating_point"]["comparison"] != "strict_greater_than":
        raise ValueError("Stage-1 comparison differs")
    if config["stage2"]["selection_rule"] != (
        "best_nonfull_logit_minus_full_logit_strictly_greater_than_delta"
    ):
        raise ValueError("Stage-2 abstention rule differs")
    return config


def _source_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    return {name: resolve_path(value) for name, value in config["sources"].items()}


def _phase66_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return read_json(_source_paths(config)["phase66_config"])


def _validate_parent_contracts(
    config: Mapping[str, Any], sources: Mapping[str, Path], *, verify_model: bool
) -> None:
    phase66 = read_json(sources["phase66_contract"])
    phase69 = read_json(sources["phase69_contract"])
    if canonical_hash(phase66) != phase66.get("contract_sha256"):
        raise RuntimeError("Phase-66 contract hash is invalid")
    if canonical_hash(phase69) != phase69.get("contract_sha256"):
        raise RuntimeError("Phase-69 contract hash is invalid")
    if phase66["contract_sha256"] != config["stage2"]["parent_contract_sha256"]:
        raise RuntimeError("configured Stage-2 parent differs from Phase 66")
    if phase69["contract_sha256"] != config["external"]["parent_contract_sha256"]:
        raise RuntimeError("configured external parent differs from Phase 69")
    selected = read_json(sources["phase66_selected_checkpoint"])
    checkpoint_hash = file_sha256(sources["phase66_checkpoint"])
    if (
        selected["sha256"] != config["stage2"]["checkpoint_sha256"]
        or checkpoint_hash != selected["sha256"]
        or selected["contract_sha256"] != phase66["contract_sha256"]
        or selected["experiment"] != "A"
    ):
        raise RuntimeError("Experiment-A checkpoint identity differs")
    if verify_model:
        snapshot = resolve_path(phase66["static_config"]["model"]["snapshot_path"])
        actual_snapshot = {
            path.name: file_sha256(path)
            for path in sorted(snapshot.iterdir())
            if path.is_file()
        }
        if actual_snapshot != phase66["model_snapshot_sha256"]:
            raise RuntimeError("Qwen snapshot differs from the frozen Phase-66 model")


def _git_state() -> dict[str, str]:
    return {
        "commit": command_output(("git", "rev-parse", "HEAD")),
        "branch": command_output(("git", "branch", "--show-current")),
        "worktree_status_at_freeze": command_output(("git", "status", "--porcelain=v1")),
    }


def _runtime_state() -> dict[str, Any]:
    gpu_query = command_output(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        )
    ).splitlines()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "accelerate": package_version("accelerate"),
        "cuda_runtime": torch.version.cuda,
        "gpu_rows": gpu_query,
    }


def _validate_image(sample: Mapping[str, Any]) -> str:
    path = resolve_path(str(sample["local_image_path"]))
    expected = str(sample["image_content_sha256"])
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(f"missing image for {sample['uid']}: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise RuntimeError(f"image hash differs immediately before inference: {sample['uid']}")
    return observed


def prepare(config_path: Path) -> None:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    sources = _source_paths(config)
    _validate_parent_contracts(config, sources, verify_model=True)

    schedule_all = read_jsonl(sources["phase66_schedule"])
    final_epoch = int(config["grid"]["epoch_zero_based"])
    schedule = [row for row in schedule_all if int(row["epoch"]) == final_epoch]
    if len(schedule) != int(config["grid"]["expected_draws"]):
        raise RuntimeError("final training epoch draw count differs")
    if len({int(row["draw_index"]) for row in schedule}) != len(schedule):
        raise RuntimeError("final training epoch draw IDs are not unique")
    if Counter(row["draw_kind"] for row in schedule) != Counter({"W": 698, "C": 350}):
        raise RuntimeError("final training epoch sampler weighting differs")

    validation = read_jsonl(sources["phase66_validation_manifest"])
    expected = config["development"]
    if len(validation) != int(expected["expected_samples"]):
        raise RuntimeError("Historical validation population differs")
    if sum(bool(row["dense_correct"]) for row in validation) != int(expected["expected_dense_correct"]):
        raise RuntimeError("Historical dense-correct count differs")
    if sum(bool(row["dense_wrong"]) for row in validation) != int(expected["expected_dense_wrong"]):
        raise RuntimeError("Historical dense-wrong count differs")
    triggered = [
        row
        for row in validation
        if row["trigger_layers"][config["stage1_operating_point"]["name"]] is not None
    ]
    if len(triggered) != int(expected["expected_p90_triggered"]):
        raise RuntimeError("Historical P90 trigger count differs")

    fold_input = [
        {
            "uid": row["uid"],
            "image_group_id": row["sample"]["image_group_id"],
        }
        for row in validation
    ]
    folds = assign_group_folds(
        fold_input, folds=int(expected["folds"]), seed=int(config["seed"])
    )
    fold_rows = [
        {
            "schema_version": "stage2_abstention_fold_assignment_v1",
            "uid": row["uid"],
            "dataset": row["dataset"],
            "image_group_id": row["sample"]["image_group_id"],
            "fold": folds[row["uid"]],
        }
        for row in sorted(validation, key=lambda item: str(item["uid"]))
    ]

    output_root.mkdir(parents=True)
    atomic_jsonl(output_root / "work/training_margin_schedule.jsonl", schedule)
    atomic_jsonl(output_root / "work/triggered_validation.jsonl", triggered)
    atomic_jsonl(output_root / "crossfit/fold_assignments.jsonl", fold_rows)
    source_hashes = {name: file_sha256(path) for name, path in sources.items()}
    internal_hashes = {
        "work/training_margin_schedule.jsonl": file_sha256(
            output_root / "work/training_margin_schedule.jsonl"
        ),
        "work/triggered_validation.jsonl": file_sha256(
            output_root / "work/triggered_validation.jsonl"
        ),
        "crossfit/fold_assignments.jsonl": file_sha256(
            output_root / "crossfit/fold_assignments.jsonl"
        ),
    }
    contract = {
        "schema_version": "stage2_abstention_margin_protocol_v1",
        "created_at": utc_now(),
        "static_config": config,
        "git": _git_state(),
        "runtime": _runtime_state(),
        "source_sha256": source_hashes,
        "prepared_sha256": internal_hashes,
        "bound_code_sha256": {
            relative: file_sha256(resolve_path(relative)) for relative in BOUND_CODE
        },
        "parent_contracts": {
            "phase66": config["stage2"]["parent_contract_sha256"],
            "phase69": config["external"]["parent_contract_sha256"],
        },
        "population": {
            "training_margin_draws": len(schedule),
            "training_margin_states": sum(len(row["selected_layers"]) for row in schedule),
            "development_samples": len(validation),
            "development_triggered": len(triggered),
            "fold_counts": dict(Counter(str(row["fold"]) for row in fold_rows)),
        },
        "review_reconciliation": {
            "verdict": "stable",
            "accepted_design": "training-frozen grid, sequential Historical-800 sweep, group cross-fit, conditional locked external rerun",
            "recorded_caveat": "final-epoch sampler weighting may make the fixed quantile grid coarse for rollout-state confidence",
            "resolution": "retain the prespecified grid and do not use development outcomes to refine it",
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_protocol.json", contract)
    protocol = f"""# Stage-2 abstention-margin protocol

- Contract: `{contract['contract_sha256']}`
- Stage-1: robust ALL-source P90 `{config['stage1_operating_point']['threshold']}` with strict `>`.
- Stage-2: frozen Phase-66 Experiment A, checkpoint `{config['stage2']['checkpoint_sha256']}`.
- Rule: choose the best non-FULL action only when `best_nonfull_logit - full_logit > delta`; otherwise choose FULL.
- Grid source: the already-frozen final Experiment-A training epoch ({len(schedule)} sampler-weighted draws), never development outcomes.
- Development: frozen Historical-800, with actual sequential execution at every finite candidate margin and an exact dense `+inf` control.
- Selection: C→C preservation at least {expected['minimum_c_to_c_preservation']:.3%}; maximize Net, then fewer C→W, more W→C, fewer interventions, and larger delta. A positive margin must strictly improve Net over delta=0.
- Stability: five-fold image-group-disjoint cross-fit.
- External: one locked Phase-69 four-family rerun only if the selected development margin is positive; no external retuning.

The independent review found the design stable. Its grid-resolution caveat is retained as a limitation rather than addressed with outcome-dependent grid changes.
"""
    _atomic_bytes(output_root / "protocol.md", protocol.encode())
    print(json.dumps({"prepared": True, "contract_sha256": contract["contract_sha256"]}))


def verify_contract(config_path: Path, *, verify_model: bool = False) -> tuple[dict[str, Any], Path]:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if canonical_hash(contract) != contract.get("contract_sha256"):
        raise RuntimeError("abstention-margin contract hash is invalid")
    if contract["static_config"] != config:
        raise RuntimeError("active config differs from frozen contract")
    if contract["git"] != _git_state():
        raise RuntimeError("git identity or worktree differs from frozen contract")
    if contract["runtime"] != _runtime_state():
        raise RuntimeError("runtime differs from frozen contract")
    for relative, expected in contract["bound_code_sha256"].items():
        if file_sha256(resolve_path(relative)) != expected:
            raise RuntimeError(f"bound code differs: {relative}")
    for name, expected in contract["source_sha256"].items():
        if file_sha256(resolve_path(config["sources"][name])) != expected:
            raise RuntimeError(f"source differs: {name}")
    for relative, expected in contract["prepared_sha256"].items():
        if file_sha256(output_root / relative) != expected:
            raise RuntimeError(f"prepared input differs: {relative}")
    _validate_parent_contracts(config, _source_paths(config), verify_model=verify_model)
    return contract, output_root


def _load_stage2(
    config: Mapping[str, Any], device: torch.device
) -> tuple[Any, Any, torch.nn.Module, dict[str, Any]]:
    phase66 = _phase66_config(config)
    processor, _base, wrapped = _load_model(phase66, device)
    router = _router(phase66, device).eval()
    checkpoint = torch.load(
        _source_paths(config)["phase66_checkpoint"], map_location="cpu", weights_only=False
    )
    if (
        checkpoint.get("contract_sha256") != config["stage2"]["parent_contract_sha256"]
        or checkpoint.get("experiment") != "A"
    ):
        raise RuntimeError("loaded Stage-2 checkpoint provenance differs")
    router.load_state_dict(checkpoint["state_dict"], strict=True)
    return processor, wrapped, router, phase66


@torch.inference_mode()
def training_margin_worker(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != int(config["world_size"]):
        raise RuntimeError("training-margin collection requires exactly four workers")
    result_path = output_root / f"margin_grid/work/rank{rank:02d}.jsonl"
    complete_path = output_root / f"margin_grid/work/rank{rank:02d}.complete.json"
    if result_path.exists() or complete_path.exists():
        raise RuntimeError(f"refusing to overwrite training-margin rank {rank}")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    phase66 = _phase66_config(config)
    configure_dense_determinism(int(config["seed"]) + rank, phase66["backend_settings"])
    processor, wrapped, router, _ = _load_stage2(config, device)
    samples = {
        str(row["uid"]): row["sample"]
        for row in read_jsonl(_source_paths(config)["phase66_train_samples"])
    }
    schedule = [
        row
        for row in read_jsonl(output_root / "work/training_margin_schedule.jsonl")
        if int(row["worker_rank"]) == rank
    ]
    cache: dict[str, Any] = {}
    state_count = 0
    started = time.monotonic()
    for count, row in enumerate(schedule, 1):
        uid = str(row["uid"])
        sample = samples[uid]
        _validate_image(sample)
        if uid not in cache:
            cache[uid] = _binary_to(_prepare_binary(processor, wrapped, sample, device), "cpu")
        meta = _binary_to(cache[uid], device)
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
        logits = router(text, visual, text_mask=text_mask, visual_mask=visual_mask).float()
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError(f"non-finite training margin logits: {uid}")
        for position, layer in enumerate(row["selected_layers"]):
            values = logits[position].detach().cpu().tolist()
            decision = choose_action_from_logits(values, delta=0.0)
            append_jsonl(
                result_path,
                {
                    "schema_version": "stage2_training_margin_row_v1",
                    "contract_sha256": contract["contract_sha256"],
                    "checkpoint_sha256": config["stage2"]["checkpoint_sha256"],
                    "uid": uid,
                    "dataset": row["dataset"],
                    "source_regime": row["source_regime"],
                    "draw_index": int(row["draw_index"]),
                    "draw_kind": row["draw_kind"],
                    "layer": int(layer),
                    "target_action": row["selected_actions"][position],
                    "target_action_index": int(row["selected_action_indices"][position]),
                    "logits": values,
                    **decision,
                    "worker_rank": rank,
                },
            )
            state_count += 1
        if count % 25 == 0:
            print(
                json.dumps(
                    {
                        "phase": "training_margins",
                        "rank": rank,
                        "completed": count,
                        "assigned": len(schedule),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del meta, routed, text, visual, text_mask, visual_mask, logits
    atomic_json(
        complete_path,
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "checkpoint_sha256": config["stage2"]["checkpoint_sha256"],
            "rank": rank,
            "draws": len(schedule),
            "states": state_count,
            "elapsed_seconds": time.monotonic() - started,
            "completed_at": utc_now(),
        },
    )


def freeze_grid(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    schedule = read_jsonl(output_root / "work/training_margin_schedule.jsonl")
    expected_keys = Counter(
        (int(row["draw_index"]), int(layer))
        for row in schedule
        for layer in row["selected_layers"]
    )
    rows: list[dict[str, Any]] = []
    for rank in range(int(config["world_size"])):
        completion = read_json(output_root / f"margin_grid/work/rank{rank:02d}.complete.json")
        if not completion.get("passed") or completion["contract_sha256"] != contract["contract_sha256"]:
            raise RuntimeError(f"training-margin rank {rank} is incomplete or incompatible")
        rows.extend(read_jsonl(output_root / f"margin_grid/work/rank{rank:02d}.jsonl"))
    observed = Counter((int(row["draw_index"]), int(row["layer"])) for row in rows)
    if observed != expected_keys:
        raise RuntimeError("training-margin state coverage is incomplete or duplicated")
    if any(row["contract_sha256"] != contract["contract_sha256"] for row in rows):
        raise RuntimeError("training-margin row contract differs")
    margins = [float(row["raw_margin"]) for row in rows]
    positive = np.asarray([value for value in margins if value > 0], dtype=np.float64)
    grid = build_margin_grid(margins, quantiles=config["grid"]["quantiles"])
    distribution_rows = []
    for scope, values in (
        ("all", np.asarray(margins, dtype=np.float64)),
        ("positive", positive),
    ):
        distribution_rows.append(
            {
                "scope": scope,
                "count": len(values),
                "minimum": float(np.min(values)),
                "q10": float(np.quantile(values, 0.10)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.quantile(values, 0.50)),
                "q75": float(np.quantile(values, 0.75)),
                "q90": float(np.quantile(values, 0.90)),
                "maximum": float(np.max(values)),
                "mean": float(np.mean(values)),
            }
        )
    atomic_csv(output_root / "margin_grid/training_margin_distribution.csv", distribution_rows)
    candidate_rows = [
        {
            **row,
            "delta": "inf" if math.isinf(float(row["delta"])) else float(row["delta"]),
            "contract_sha256": contract["contract_sha256"],
            "grid_source": config["grid"]["source"],
        }
        for row in grid
    ]
    atomic_csv(output_root / "margin_grid/candidate_margins.csv", candidate_rows)
    atomic_jsonl(output_root / "margin_grid/training_margin_rows.jsonl", rows)
    binding = {
        "schema_version": "stage2_abstention_margin_grid_binding_v1",
        "contract_sha256": contract["contract_sha256"],
        "checkpoint_sha256": config["stage2"]["checkpoint_sha256"],
        "source_schedule_sha256": contract["prepared_sha256"]["work/training_margin_schedule.jsonl"],
        "training_margin_rows_sha256": file_sha256(
            output_root / "margin_grid/training_margin_rows.jsonl"
        ),
        "candidate_margins_sha256": file_sha256(
            output_root / "margin_grid/candidate_margins.csv"
        ),
        "positive_margin_count": len(positive),
        "candidate_count": len(candidate_rows),
        "frozen_before_development_outcomes": True,
        "created_at": utc_now(),
    }
    binding["binding_sha256"] = artifact_hash(binding, hash_field="binding_sha256")
    atomic_json(output_root / "margin_grid/grid_binding.json", binding)
    print(json.dumps(binding, sort_keys=True))


def _candidate_grid(output_root: Path, contract_sha256: str) -> list[dict[str, Any]]:
    binding = read_json(output_root / "margin_grid/grid_binding.json")
    if artifact_hash(binding, hash_field="binding_sha256") != binding.get("binding_sha256"):
        raise RuntimeError("margin-grid binding hash is invalid")
    if binding["contract_sha256"] != contract_sha256:
        raise RuntimeError("margin grid belongs to another contract")
    path = output_root / "margin_grid/candidate_margins.csv"
    if file_sha256(path) != binding["candidate_margins_sha256"]:
        raise RuntimeError("candidate-margin grid changed after freezing")
    rows = read_csv(path)
    return [
        {
            **row,
            "delta": math.inf if row["delta"] == "inf" else float(row["delta"]),
        }
        for row in rows
    ]


@torch.inference_mode()
def development_worker(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != int(config["world_size"]):
        raise RuntimeError("development rollout requires exactly four workers")
    result_path = output_root / f"development/work/rank{rank:02d}.jsonl"
    complete_path = output_root / f"development/work/rank{rank:02d}.complete.json"
    if result_path.exists() or complete_path.exists():
        raise RuntimeError(f"refusing to overwrite development rank {rank}")
    grid = [row for row in _candidate_grid(output_root, contract["contract_sha256"]) if not math.isinf(row["delta"])]
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    phase66 = _phase66_config(config)
    configure_dense_determinism(int(config["seed"]) + 1000 + rank, phase66["backend_settings"])
    processor, wrapped, router, _ = _load_stage2(config, device)
    triggered = sorted(
        read_jsonl(output_root / "work/triggered_validation.jsonl"),
        key=lambda row: str(row["uid"]),
    )
    assigned = [row for index, row in enumerate(triggered) if index % world_size == rank]
    started = time.monotonic()
    for count, row in enumerate(assigned, 1):
        sample = row["sample"]
        consumed_hash = _validate_image(sample)
        inputs, _metadata = build_dense_inputs(processor, sample, device)
        cached_meta = _binary_to(_prepare_binary(processor, wrapped, sample, device), "cpu")
        trigger = int(row["trigger_layers"][config["stage1_operating_point"]["name"]])
        for candidate in grid:
            delta = float(candidate["delta"])
            meta = _binary_to(cached_meta, device)
            action_rows: list[dict[str, Any]] = []

            def selector(layer_index, text_states, visual_states, current_meta):
                if int(layer_index) < trigger:
                    action_rows.append(
                        {"layer": int(layer_index), "action": "FULL", "active": False}
                    )
                    return "FULL"
                logits = router(
                    text_states.detach().clone(),
                    visual_states.detach().clone(),
                    text_mask=current_meta.text_valid_mask.detach().clone(),
                    visual_mask=current_meta.visual_valid_mask.detach().clone(),
                ).float()[0]
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError(f"non-finite Stage-2 logits: {row['uid']}")
                values = logits.detach().cpu().tolist()
                decision = choose_action_from_logits(values, delta=delta)
                probabilities = logits.softmax(dim=-1).detach().cpu().tolist()
                action_rows.append(
                    {
                        "layer": int(layer_index),
                        "active": True,
                        "logits": values,
                        "probabilities": {
                            name: float(probabilities[index])
                            for index, name in enumerate(ACTION_NAMES)
                        },
                        **decision,
                    }
                )
                return decision["action"]

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
            active = [item for item in action_rows if item["active"]]
            nonfull = [item for item in active if item["action"] != "FULL"]
            routed_correct = bool(score.correct)
            append_jsonl(
                result_path,
                {
                    "schema_version": "stage2_abstention_development_row_v1",
                    "contract_sha256": contract["contract_sha256"],
                    "grid_binding_sha256": read_json(
                        output_root / "margin_grid/grid_binding.json"
                    )["binding_sha256"],
                    "checkpoint_sha256": config["stage2"]["checkpoint_sha256"],
                    "margin_id": candidate["margin_id"],
                    "margin_label": candidate["label"],
                    "delta": delta,
                    "uid": row["uid"],
                    "dataset": row["dataset"],
                    "image_group_id": sample["image_group_id"],
                    "consumed_image_sha256": consumed_hash,
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
                    "routed_correct": routed_correct,
                    "routed_wrong": not routed_correct,
                    "transition": ("C" if row["dense_correct"] else "W")
                    + "→"
                    + ("C" if routed_correct else "W"),
                    "actions": [item["action"] for item in action_rows],
                    "action_rows": action_rows,
                    "post_trigger_action_counts": dict(
                        Counter(item["action"] for item in active)
                    ),
                    "post_trigger_actions": len(active),
                    "non_full_count": len(nonfull),
                    "any_non_full": bool(nonfull),
                    "first_non_full_layer": nonfull[0]["layer"] if nonfull else None,
                    "trigger_to_first_non_full_delay": (
                        nonfull[0]["layer"] - trigger if nonfull else None
                    ),
                    "worker_rank": rank,
                },
            )
            del meta, output
        if count % 5 == 0:
            print(
                json.dumps(
                    {
                        "phase": "development",
                        "rank": rank,
                        "completed_samples": count,
                        "assigned_samples": len(assigned),
                        "finite_margins": len(grid),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del inputs, cached_meta
    atomic_json(
        complete_path,
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "rank": rank,
            "samples": len(assigned),
            "finite_margins": len(grid),
            "records": len(assigned) * len(grid),
            "elapsed_seconds": time.monotonic() - started,
            "completed_at": utc_now(),
        },
    )


def _dense_control_row(
    source: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    contract_sha256: str,
    fold: int,
) -> dict[str, Any]:
    dense_correct = bool(source["dense_correct"])
    return {
        "schema_version": "stage2_abstention_development_row_v1",
        "contract_sha256": contract_sha256,
        "margin_id": candidate["margin_id"],
        "margin_label": candidate["label"],
        "delta": "inf" if math.isinf(float(candidate["delta"])) else float(candidate["delta"]),
        "uid": source["uid"],
        "dataset": source["dataset"],
        "image_group_id": source["sample"]["image_group_id"],
        "triggered": source["trigger_layers"]["P90"] is not None,
        "trigger_layer": source["trigger_layers"]["P90"],
        "dense_correct": dense_correct,
        "dense_wrong": not dense_correct,
        "dense_generated_answer": source["dense_output"]["generated_answer"],
        "dense_generated_token_ids": source["dense_output"]["generated_token_ids"],
        "routed_generated_answer": source["dense_output"]["generated_answer"],
        "routed_generated_token_ids": source["dense_output"]["generated_token_ids"],
        "lmms_metric": source["dense_output"]["lmms_eval_metric"],
        "lmms_score": source["dense_output"]["lmms_eval_per_sample_score"],
        "routed_correct": dense_correct,
        "routed_wrong": not dense_correct,
        "transition": "C→C" if dense_correct else "W→W",
        "actions": ["FULL"] * 28,
        "action_rows": [],
        "post_trigger_action_counts": (
            {"FULL": 28 - int(source["trigger_layers"]["P90"])}
            if source["trigger_layers"]["P90"] is not None
            else {}
        ),
        "post_trigger_actions": (
            28 - int(source["trigger_layers"]["P90"])
            if source["trigger_layers"]["P90"] is not None
            else 0
        ),
        "non_full_count": 0,
        "any_non_full": False,
        "first_non_full_layer": None,
        "trigger_to_first_non_full_delay": None,
        "fold": fold,
        "worker_rank": None,
    }


def _summary_row(candidate: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = summarize_margin_rollout(rows)
    return {
        "margin_id": candidate["margin_id"],
        "margin_label": candidate["label"],
        "delta": "inf" if math.isinf(float(candidate["delta"])) else float(candidate["delta"]),
        **summary,
    }


def _plot_development(output_root: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    finite = [row for row in summaries if str(row["delta"]) != "inf"]
    xs = [float(row["delta"]) for row in finite]
    specs = (
        ("net_vs_margin.png", "Net corrections", [row["net_corrections"] for row in finite]),
        ("preservation_vs_margin.png", "C→C preservation", [row["c_to_c_preservation_rate"] for row in finite]),
        ("nonfull_rate_vs_margin.png", "Intervention rate", [row["intervention_rate"] for row in finite]),
    )
    for filename, ylabel, ys in specs:
        plt.figure(figsize=(6.4, 4.2))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Abstention margin δ")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(output_root / "figures" / filename, dpi=160)
        plt.close()
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(xs, [row["w_to_c"] for row in finite], marker="o", label="W→C")
    plt.plot(xs, [row["c_to_w"] for row in finite], marker="o", label="C→W")
    plt.xlabel("Abstention margin δ")
    plt.ylabel("Samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "figures/rescue_regression_vs_margin.png", dpi=160)
    plt.close()


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def finalize_development(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    grid = _candidate_grid(output_root, contract["contract_sha256"])
    finite = [candidate for candidate in grid if not math.isinf(float(candidate["delta"]))]
    worker_rows: list[dict[str, Any]] = []
    for rank in range(int(config["world_size"])):
        completion = read_json(output_root / f"development/work/rank{rank:02d}.complete.json")
        if not completion.get("passed") or completion["contract_sha256"] != contract["contract_sha256"]:
            raise RuntimeError(f"development rank {rank} is incomplete or incompatible")
        worker_rows.extend(read_jsonl(output_root / f"development/work/rank{rank:02d}.jsonl"))
    triggered = read_jsonl(output_root / "work/triggered_validation.jsonl")
    validate_result_matrix(
        [row["uid"] for row in triggered],
        [row["margin_id"] for row in finite],
        worker_rows,
        contract_sha256=contract["contract_sha256"],
    )
    validation = read_jsonl(_source_paths(config)["phase66_validation_manifest"])
    folds = {
        row["uid"]: int(row["fold"])
        for row in read_jsonl(output_root / "crossfit/fold_assignments.jsonl")
    }
    triggered_map = {(row["uid"], row["margin_id"]): row for row in worker_rows}
    all_rows: list[dict[str, Any]] = []
    for candidate in grid:
        for source in validation:
            key = (source["uid"], candidate["margin_id"])
            if key in triggered_map:
                row = dict(triggered_map[key])
                row["fold"] = folds[source["uid"]]
            else:
                row = _dense_control_row(
                    source,
                    candidate,
                    contract_sha256=contract["contract_sha256"],
                    fold=folds[source["uid"]],
                )
            all_rows.append(row)
    validate_result_matrix(
        [row["uid"] for row in validation],
        [row["margin_id"] for row in grid],
        all_rows,
        contract_sha256=contract["contract_sha256"],
    )
    atomic_jsonl(output_root / "development/per_sample_results.jsonl", all_rows)

    original = {
        row["uid"]: row for row in read_jsonl(_source_paths(config)["phase66_delta0_results"])
    }
    zero = [row for row in all_rows if float(row["delta"]) == 0.0]
    mismatches = [
        row["uid"]
        for row in zero
        if row["routed_generated_token_ids"]
        != original[row["uid"]]["routed_generated_token_ids"]
        or bool(row["routed_correct"]) != bool(original[row["uid"]]["routed_correct"])
        or row["actions"] != original[row["uid"]]["actions"]
    ]
    if mismatches:
        raise RuntimeError(f"delta=0 rerun differs from Phase-66 baseline: {mismatches[:5]}")

    rows_by_margin = {
        candidate["margin_id"]: [
            row for row in all_rows if row["margin_id"] == candidate["margin_id"]
        ]
        for candidate in grid
    }
    summaries = [_summary_row(candidate, rows_by_margin[candidate["margin_id"]]) for candidate in grid]
    atomic_csv(output_root / "development/per_margin_rollout_summary.csv", summaries)
    transition_rows = []
    action_rows = []
    for candidate, summary in zip(grid, summaries):
        for name, key in (("W→C", "w_to_c"), ("W→W", "w_to_w"), ("C→C", "c_to_c"), ("C→W", "c_to_w")):
            transition_rows.append(
                {
                    "margin_id": candidate["margin_id"],
                    "margin_label": candidate["label"],
                    "delta": summary["delta"],
                    "transition": name,
                    "count": summary[key],
                }
            )
        cell = rows_by_margin[candidate["margin_id"]]
        active = Counter()
        for row in cell:
            active.update(row["post_trigger_action_counts"])
        total = sum(active.values())
        action_rows.append(
            {
                "margin_id": candidate["margin_id"],
                "margin_label": candidate["label"],
                "delta": summary["delta"],
                "triggered_samples": sum(bool(row["triggered"]) for row in cell),
                "intervened_samples": summary["intervened_samples"],
                "intervention_rate": summary["intervention_rate"],
                **{f"{action.lower()}_count": active[action] for action in ACTION_NAMES},
                **{
                    f"{action.lower()}_fraction": active[action] / total if total else None
                    for action in ACTION_NAMES
                },
            }
        )
    atomic_csv(output_root / "development/per_margin_transition_counts.csv", transition_rows)
    atomic_csv(output_root / "development/per_margin_action_behavior.csv", action_rows)

    minimum = float(config["development"]["minimum_c_to_c_preservation"])
    selected = select_margin(summaries, min_preservation=minimum)
    selected_candidate = next(row for row in grid if row["margin_id"] == selected["margin_id"])

    selected_by_fold = []
    heldout_metrics = []
    for heldout in range(int(config["development"]["folds"])):
        training_summaries = []
        for candidate in grid:
            cell = [
                row
                for row in rows_by_margin[candidate["margin_id"]]
                if int(row["fold"]) != heldout
            ]
            training_summaries.append(_summary_row(candidate, cell))
        fold_selected = select_margin(training_summaries, min_preservation=minimum)
        chosen_candidate = next(
            row for row in grid if row["margin_id"] == fold_selected["margin_id"]
        )
        heldout_cell = [
            row
            for row in rows_by_margin[chosen_candidate["margin_id"]]
            if int(row["fold"]) == heldout
        ]
        heldout_summary = _summary_row(chosen_candidate, heldout_cell)
        selected_by_fold.append(
            {
                "heldout_fold": heldout,
                "training_samples": 800 - len(heldout_cell),
                "selected_margin_id": chosen_candidate["margin_id"],
                "selected_margin_label": chosen_candidate["label"],
                "selected_delta": heldout_summary["delta"],
                "training_net_corrections": fold_selected["net_corrections"],
                "training_c_to_w": fold_selected["c_to_w"],
                "training_w_to_c": fold_selected["w_to_c"],
                "training_preservation": fold_selected["c_to_c_preservation_rate"],
            }
        )
        heldout_metrics.append({"heldout_fold": heldout, **heldout_summary})
    atomic_csv(output_root / "crossfit/selected_margin_by_fold.csv", selected_by_fold)
    atomic_csv(output_root / "crossfit/heldout_fold_metrics.csv", heldout_metrics)
    selected_counts = Counter(row["selected_margin_id"] for row in selected_by_fold)
    stability = {
        "schema_version": "stage2_abstention_crossfit_stability_v1",
        "contract_sha256": contract["contract_sha256"],
        "folds": int(config["development"]["folds"]),
        "selection_counts": dict(selected_counts),
        "all_folds_same_margin": len(selected_counts) == 1,
        "global_selected_margin_id": selected_candidate["margin_id"],
        "global_selected_in_folds": selected_counts[selected_candidate["margin_id"]],
        "pooled_heldout_net_corrections": sum(row["net_corrections"] for row in heldout_metrics),
        "pooled_heldout_w_to_c": sum(row["w_to_c"] for row in heldout_metrics),
        "pooled_heldout_c_to_w": sum(row["c_to_w"] for row in heldout_metrics),
    }
    atomic_json(output_root / "crossfit/stability_summary.json", stability)

    zero_rows = rows_by_margin[next(row["margin_id"] for row in grid if float(row["delta"]) == 0.0)]
    action_margins: dict[str, list[float]] = defaultdict(list)
    sample_margin_rows = []
    for row in zero_rows:
        active = [item for item in row["action_rows"] if item.get("active")]
        positive = [float(item["raw_margin"]) for item in active if float(item["raw_margin"]) > 0]
        if positive:
            sample_margin_rows.append(
                {
                    "uid": row["uid"],
                    "dataset": row["dataset"],
                    "transition": row["transition"],
                    "positive_margin_count": len(positive),
                    "maximum_positive_margin": max(positive),
                    "first_positive_margin": positive[0],
                }
            )
        for item in active:
            if item["action"] != "FULL":
                action_margins[item["action"]].append(float(item["raw_margin"]))
    transition_diag = []
    for name in ("W→C", "W→W", "C→C", "C→W"):
        cell = [row for row in sample_margin_rows if row["transition"] == name]
        first = _distribution([float(row["first_positive_margin"]) for row in cell])
        maximum = _distribution([float(row["maximum_positive_margin"]) for row in cell])
        transition_diag.append(
            {
                "transition": name,
                **{f"first_{key}": value for key, value in first.items()},
                **{f"maximum_{key}": value for key, value in maximum.items()},
            }
        )
    action_diag = [
        {"action": name, **_distribution(action_margins[name])}
        for name in ACTION_NAMES[1:]
    ]
    atomic_csv(output_root / "diagnostics/margin_by_transition.csv", transition_diag)
    atomic_csv(output_root / "diagnostics/margin_by_action.csv", action_diag)
    atomic_csv(
        output_root / "diagnostics/selectivity_summary.csv",
        sample_margin_rows,
        fieldnames=("uid", "dataset", "transition", "positive_margin_count", "maximum_positive_margin", "first_positive_margin"),
    )

    selection_payload = {
        "schema_version": "stage2_abstention_selected_margin_v1",
        "contract_sha256": contract["contract_sha256"],
        "grid_binding_sha256": read_json(output_root / "margin_grid/grid_binding.json")["binding_sha256"],
        "margin_id": selected_candidate["margin_id"],
        "label": selected_candidate["label"],
        "delta": float(selected_candidate["delta"]),
        "development_metrics": selected,
        "baseline_metrics": next(row for row in summaries if float(row["delta"]) == 0.0),
        "minimum_c_to_c_preservation": minimum,
        "positive_margin_supported": float(selected_candidate["delta"]) > 0 and math.isfinite(float(selected_candidate["delta"])),
        "selection_rule": config["development"]["tie_break"],
        "selected_at": utc_now(),
    }
    atomic_json(output_root / "selection/selected_margin.json", selection_payload)
    decision = (
        f"Selected positive margin **δ={selected_candidate['delta']:.9g}** because it strictly improved development Net over δ=0 while satisfying preservation. One locked external rerun is authorized by the plan."
        if selection_payload["positive_margin_supported"]
        else "Selected **δ=0**. No positive finite margin strictly improved development Net over δ=0 under the preservation constraint, so the plan stops before external re-evaluation."
    )
    _atomic_bytes(
        output_root / "selection/development_decision.md",
        ("# Development margin decision\n\n" + decision + "\n").encode(),
    )
    _plot_development(output_root, summaries)

    plt.figure(figsize=(6.4, 4.2))
    labels = [row["transition"] for row in transition_diag]
    medians = [
        row["maximum_median"] if row["maximum_median"] is not None else 0
        for row in transition_diag
    ]
    plt.bar(labels, medians)
    plt.ylabel("Median maximum executed non-FULL margin at δ=0")
    plt.tight_layout()
    plt.savefig(output_root / "figures/margin_by_transition.png", dpi=160)
    plt.close()
    plt.figure(figsize=(6.4, 4.2))
    plt.bar(
        [str(row["heldout_fold"]) for row in selected_by_fold],
        [float(row["selected_delta"]) for row in selected_by_fold],
    )
    plt.xlabel("Held-out fold")
    plt.ylabel("Margin selected on other four folds")
    plt.tight_layout()
    plt.savefig(output_root / "figures/crossfit_margin_stability.png", dpi=160)
    plt.close()
    print(json.dumps(selection_payload, sort_keys=True))


def prepare_external(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    selected_path = output_root / "selection/selected_margin.json"
    selected = read_json(selected_path)
    delta = float(selected["delta"])
    if not selected.get("positive_margin_supported") or not math.isfinite(delta) or delta <= 0:
        raise RuntimeError("external rerun is forbidden unless development selects a positive finite margin")
    external_root = output_root / "external/run_only_if_authorized"
    binding_path = external_root / "frozen_external_binding.json"
    if binding_path.exists():
        raise RuntimeError("external rerun is already frozen")
    phase69_rows = read_jsonl(_source_paths(config)["phase69_paired_results"])
    prepared = read_jsonl(_source_paths(config)["phase69_prepared_manifest"])
    if len(phase69_rows) != int(config["external"]["expected_samples"]):
        raise RuntimeError("Phase-69 paired population differs")
    if len(prepared) != len(phase69_rows):
        raise RuntimeError("Phase-69 prepared population differs")
    if {row["uid"] for row in phase69_rows} != {row["uid"] for row in prepared}:
        raise RuntimeError("Phase-69 paired/prepared UID populations differ")
    triggered = [row for row in phase69_rows if bool(row["triggered"])]
    binding = {
        "schema_version": "stage2_abstention_external_binding_v1",
        "contract_sha256": contract["contract_sha256"],
        "phase69_contract_sha256": config["external"]["parent_contract_sha256"],
        "selected_margin_sha256": file_sha256(selected_path),
        "margin_id": selected["margin_id"],
        "delta": delta,
        "checkpoint_sha256": config["stage2"]["checkpoint_sha256"],
        "phase69_paired_sha256": file_sha256(_source_paths(config)["phase69_paired_results"]),
        "phase69_prepared_sha256": file_sha256(_source_paths(config)["phase69_prepared_manifest"]),
        "population": len(phase69_rows),
        "triggered_population": len(triggered),
        "execution_scope": "rerun every frozen Phase-69 Stage-1-triggered UID; reuse exact dense/no-trigger rows",
        "equivalence_basis": "Stage-1 trigger is computed on the native dense pre-intervention trajectory and is frozen in Phase-69; no-trigger routed output is exactly dense",
        "no_external_retuning": True,
        "created_at": utc_now(),
    }
    binding["binding_sha256"] = artifact_hash(binding, hash_field="binding_sha256")
    atomic_json(binding_path, binding)
    atomic_jsonl(
        external_root / "work/triggered_manifest.jsonl",
        sorted(triggered, key=lambda row: str(row["uid"])),
    )
    print(json.dumps(binding, sort_keys=True))


def _verify_external_binding(
    output_root: Path, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = contract["static_config"]
    external_root = output_root / "external/run_only_if_authorized"
    binding = read_json(external_root / "frozen_external_binding.json")
    if artifact_hash(binding, hash_field="binding_sha256") != binding.get("binding_sha256"):
        raise RuntimeError("external binding hash is invalid")
    selected = read_json(output_root / "selection/selected_margin.json")
    if (
        binding["contract_sha256"] != contract["contract_sha256"]
        or file_sha256(output_root / "selection/selected_margin.json")
        != binding["selected_margin_sha256"]
        or selected["margin_id"] != binding["margin_id"]
        or float(selected["delta"]) != float(binding["delta"])
        or file_sha256(_source_paths(config)["phase69_paired_results"])
        != binding["phase69_paired_sha256"]
        or file_sha256(_source_paths(config)["phase69_prepared_manifest"])
        != binding["phase69_prepared_sha256"]
    ):
        raise RuntimeError("external rerun inputs differ from the locked binding")
    return binding, selected


@torch.inference_mode()
def external_worker(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    binding, selected = _verify_external_binding(output_root, contract)
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != int(config["world_size"]):
        raise RuntimeError("external rerun requires exactly four workers")
    result_path = output_root / f"external/run_only_if_authorized/work/rank{rank:02d}.jsonl"
    complete_path = output_root / f"external/run_only_if_authorized/work/rank{rank:02d}.complete.json"
    if result_path.exists() or complete_path.exists():
        raise RuntimeError(f"refusing to overwrite external rank {rank}")
    phase69_config = read_json(_source_paths(config)["phase69_config"])
    configure_dense_determinism(int(config["seed"]) + 2000 + rank, phase69_config["backend_settings"])
    runtime = ExternalRuntime(phase69_config, rank)
    paired = {
        row["uid"]: row for row in read_jsonl(_source_paths(config)["phase69_paired_results"])
    }
    prepared = {
        row["uid"]: row for row in read_jsonl(_source_paths(config)["phase69_prepared_manifest"])
    }
    triggered = sorted(
        read_jsonl(output_root / "external/run_only_if_authorized/work/triggered_manifest.jsonl"),
        key=lambda row: str(row["uid"]),
    )
    assigned = [row for index, row in enumerate(triggered) if index % world_size == rank]
    delta = float(selected["delta"])
    started = time.monotonic()
    for count, source in enumerate(assigned, 1):
        uid = str(source["uid"])
        row = prepared[uid]
        original = paired[uid]
        if not bool(original["triggered"]) or original["trigger_layer"] is None:
            raise RuntimeError(f"external triggered manifest contains no-trigger UID: {uid}")
        consumed = external_verify_images(row)
        inputs = external_build_inputs(runtime, row)
        prepared_inputs = build_binary_inputs(runtime.wrapped, inputs)
        trigger = int(original["trigger_layer"])
        action_rows: list[dict[str, Any]] = []

        def selector(layer_index, text_states, visual_states, meta):
            if int(layer_index) < trigger:
                action_rows.append(
                    {"layer": int(layer_index), "action": "FULL", "active": False}
                )
                return "FULL"
            logits = runtime.stage2(
                text_states.detach().clone(),
                visual_states.detach().clone(),
                text_mask=meta.text_valid_mask.detach().clone(),
                visual_mask=meta.visual_valid_mask.detach().clone(),
            ).float()[0]
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError(f"external Stage-2 logits are non-finite: {uid}")
            values = logits.detach().cpu().tolist()
            decision = choose_action_from_logits(values, delta=delta)
            probabilities = logits.softmax(dim=-1).detach().cpu().tolist()
            action_rows.append(
                {
                    "layer": int(layer_index),
                    "active": True,
                    "logits": values,
                    "probabilities": {
                        name: float(probabilities[index])
                        for index, name in enumerate(ACTION_NAMES)
                    },
                    **decision,
                }
            )
            return decision["action"]

        routed_output = capture_online_four_action_route(
            runtime.wrapped,
            {},
            selector,
            prepared_inputs=prepared_inputs,
            use_cache=True,
            native_full_rows=True,
        )
        actions = list(routed_output.layer_actions)
        if len(actions) != 28 or actions != [item["action"] for item in action_rows]:
            raise RuntimeError(f"external action trace is incomplete: {uid}")
        active = [item for item in action_rows if item["active"]]
        nonfull = [item for item in active if item["action"] != "FULL"]
        if nonfull:
            routed = external_generate(runtime, routed_output, inputs["input_ids"], row)
        else:
            routed = {
                "generated_token_ids": original["dense_generated_token_ids"],
                "generated_answer": original["dense_generated_answer"],
                "metric": original["metric_name"],
                "score": original["dense_score"],
                "threshold": original["correctness_threshold"],
                "correct": original["dense_correct"],
            }
        routed_correct = bool(routed["correct"])
        result = {
            **original,
            "schema_version": "stage2_abstention_external_row_v1",
            "contract_sha256": contract["contract_sha256"],
            "external_binding_sha256": binding["binding_sha256"],
            "parent_phase69_contract_sha256": config["external"]["parent_contract_sha256"],
            "margin_id": selected["margin_id"],
            "delta": delta,
            "consumed_image_sha256s": consumed,
            "routed_generated_token_ids": routed["generated_token_ids"],
            "routed_generated_answer": routed["generated_answer"],
            "routed_score": routed["score"],
            "routed_correct": routed_correct,
            "transition": ("C" if original["dense_correct"] else "W")
            + "→"
            + ("C" if routed_correct else "W"),
            "actions": actions,
            "action_rows": action_rows,
            "post_trigger_action_counts": dict(
                Counter(item["action"] for item in active)
            ),
            "post_trigger_actions": len(active),
            "non_full_count": len(nonfull),
            "any_non_full": bool(nonfull),
            "first_non_full_layer": nonfull[0]["layer"] if nonfull else None,
            "trigger_to_first_non_full_delay": (
                nonfull[0]["layer"] - trigger if nonfull else None
            ),
            "worker_rank": rank,
            "elapsed_seconds": time.monotonic() - started,
        }
        append_jsonl(result_path, result)
        if count % 25 == 0:
            print(
                json.dumps(
                    {
                        "phase": "external",
                        "rank": rank,
                        "completed": count,
                        "assigned": len(assigned),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del inputs, prepared_inputs, routed_output
    atomic_json(
        complete_path,
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "external_binding_sha256": binding["binding_sha256"],
            "rank": rank,
            "expected": len(assigned),
            "completed": len(assigned),
            "elapsed_seconds": time.monotonic() - started,
            "completed_at": utc_now(),
        },
    )


def _external_scope(rows: Sequence[Mapping[str, Any]], family: str) -> list[Mapping[str, Any]]:
    return list(rows) if family == "overall" else [row for row in rows if row["benchmark_family"] == family]


def finalize_external(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    binding, selected = _verify_external_binding(output_root, contract)
    external_root = output_root / "external/run_only_if_authorized"
    worker_rows: list[dict[str, Any]] = []
    for rank in range(int(config["world_size"])):
        completion = read_json(external_root / f"work/rank{rank:02d}.complete.json")
        if (
            not completion.get("passed")
            or completion["contract_sha256"] != contract["contract_sha256"]
            or completion["external_binding_sha256"] != binding["binding_sha256"]
        ):
            raise RuntimeError(f"external rank {rank} is incomplete or incompatible")
        worker_rows.extend(read_jsonl(external_root / f"work/rank{rank:02d}.jsonl"))
    original = read_jsonl(_source_paths(config)["phase69_paired_results"])
    expected_triggered = [row["uid"] for row in original if bool(row["triggered"])]
    observed = Counter(row["uid"] for row in worker_rows)
    if observed != Counter(expected_triggered):
        raise RuntimeError("external triggered rerun coverage is incomplete or duplicated")
    if any(
        row["contract_sha256"] != contract["contract_sha256"]
        or row["external_binding_sha256"] != binding["binding_sha256"]
        for row in worker_rows
    ):
        raise RuntimeError("external result provenance differs")
    rerun = {row["uid"]: row for row in worker_rows}
    selected_rows = []
    for row in original:
        if row["uid"] in rerun:
            selected_rows.append(rerun[row["uid"]])
            continue
        copied = dict(row)
        copied.update(
            {
                "schema_version": "stage2_abstention_external_row_v1",
                "contract_sha256": contract["contract_sha256"],
                "external_binding_sha256": binding["binding_sha256"],
                "parent_phase69_contract_sha256": config["external"]["parent_contract_sha256"],
                "margin_id": selected["margin_id"],
                "delta": float(selected["delta"]),
                "worker_rank": None,
                "reused_exact_phase69_no_trigger": True,
            }
        )
        selected_rows.append(copied)
    if len(selected_rows) != int(config["external"]["expected_samples"]) or len(
        {row["uid"] for row in selected_rows}
    ) != len(selected_rows):
        raise RuntimeError("external selected population is incomplete or duplicated")
    atomic_jsonl(external_root / "selected_margin_paired_results.jsonl", selected_rows)
    families = ("chartqa", "textvqa", "mmmu_pro", "pope", "overall")
    comparison = []
    for family in families:
        before = summarize_margin_rollout(_external_scope(original, family))
        after = summarize_margin_rollout(_external_scope(selected_rows, family))
        comparison.append(
            {
                "benchmark_family": family,
                "samples": after["samples"],
                "dense_accuracy": after["dense_accuracy"],
                "original_routed_accuracy": before["routed_accuracy"],
                "margin_routed_accuracy": after["routed_accuracy"],
                "original_w_to_c": before["w_to_c"],
                "original_c_to_w": before["c_to_w"],
                "original_net": before["net_corrections"],
                "margin_w_to_c": after["w_to_c"],
                "margin_c_to_w": after["c_to_w"],
                "margin_net": after["net_corrections"],
                "delta_net": after["net_corrections"] - before["net_corrections"],
                "margin_delta_accuracy": after["delta_accuracy"],
                "margin_c_to_c_preservation": after["c_to_c_preservation_rate"],
                "margin_intervention_rate": after["intervention_rate"],
                "margin_total_non_full_count": after["total_non_full_count"],
            }
        )
    atomic_csv(external_root / "external_comparison.csv", comparison)
    overall = comparison[-1]
    markdown = f"""# Locked external abstention-margin rerun

- Margin: **δ={float(selected['delta']):.9g}** (selected once on Historical-800; not retuned externally)
- Population: **{len(selected_rows):,}**, including all four frozen families.
- Re-executed Stage-1-triggered rows: **{len(worker_rows):,}**; reused exact Phase-69 no-trigger rows: **{len(selected_rows) - len(worker_rows):,}**.
- Original W→C/C→W/Net: **{overall['original_w_to_c']} / {overall['original_c_to_w']} / {int(overall['original_net']):+d}**.
- Margin W→C/C→W/Net: **{overall['margin_w_to_c']} / {overall['margin_c_to_w']} / {int(overall['margin_net']):+d}**.
- External Net change versus δ=0: **{int(overall['delta_net']):+d}**.
- This result is prospective with respect to margin selection; no alternative threshold was inspected or selected from external outcomes.
"""
    _atomic_bytes(external_root / "external_summary.md", markdown.encode())
    atomic_json(
        external_root / "external_summary.json",
        {
            "schema_version": "stage2_abstention_external_summary_v1",
            "contract_sha256": contract["contract_sha256"],
            "external_binding_sha256": binding["binding_sha256"],
            "margin_id": selected["margin_id"],
            "delta": float(selected["delta"]),
            "population": len(selected_rows),
            "rerun_triggered": len(worker_rows),
            "reused_no_trigger": len(selected_rows) - len(worker_rows),
            "comparison": comparison,
            "completed_at": utc_now(),
        },
    )
    print(json.dumps(overall, sort_keys=True))


def _artifact_manifest(output_root: Path, contract_sha256: str) -> dict[str, Any]:
    excluded = {"artifact_manifest.json"}
    files = sorted(
        path for path in output_root.rglob("*") if path.is_file() and path.relative_to(output_root).as_posix() not in excluded
    )
    return {
        "schema_version": "stage2_abstention_margin_artifact_manifest_v1",
        "contract_sha256": contract_sha256,
        "created_at": utc_now(),
        "files": [
            {
                "path": path.relative_to(output_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in files
        ],
    }


def finalize_reports(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    selected = read_json(output_root / "selection/selected_margin.json")
    summaries = read_csv(output_root / "development/per_margin_rollout_summary.csv")
    zero = next(row for row in summaries if float(row["delta"]) == 0.0)
    chosen = next(row for row in summaries if row["margin_id"] == selected["margin_id"])
    stability = read_json(output_root / "crossfit/stability_summary.json")
    transition_diag = {row["transition"]: row for row in read_csv(output_root / "diagnostics/margin_by_transition.csv")}
    positive = bool(selected["positive_margin_supported"])
    external_summary_path = output_root / "external/run_only_if_authorized/external_summary.json"
    external_summary = read_json(external_summary_path) if external_summary_path.exists() else None
    if positive and external_summary is None:
        raise RuntimeError("positive development margin requires the locked external rerun before final reporting")
    external_status = "completed_locked_rerun" if external_summary else "not_run_delta_zero_selected"
    external_line = ""
    if external_summary:
        external_overall = next(
            row for row in external_summary["comparison"] if row["benchmark_family"] == "overall"
        )
        external_line = f" External W→C/C→W/Net changed from **{external_overall['original_w_to_c']} / {external_overall['original_c_to_w']} / {int(external_overall['original_net']):+d}** to **{external_overall['margin_w_to_c']} / {external_overall['margin_c_to_w']} / {int(external_overall['margin_net']):+d}**."
    summary = f"""# Stage-2 abstention-margin calibration

1. The grid was frozen before development outcomes from positive raw margins on the final Experiment-A training epoch: `0`, q10/q25/q40/q50/q60/q70/q80/q90/q95 after deduplication, and `+inf`.
2. At δ=0, W→C/C→W/Net were **{zero['w_to_c']} / {zero['c_to_w']} / {int(zero['net_corrections']):+d}**. At the selected margin they were **{chosen['w_to_c']} / {chosen['c_to_w']} / {int(chosen['net_corrections']):+d}**.
3. C→W {'fell faster than W→C' if int(chosen['c_to_w']) - int(zero['c_to_w']) < int(chosen['w_to_c']) - int(zero['w_to_c']) else 'did not fall faster than W→C'} over the selected move; selected development C→C preservation was **{float(chosen['c_to_c_preservation_rate']):.4%}** against a {float(selected['minimum_c_to_c_preservation']):.4%} floor.
4. The best eligible development Net was **{int(chosen['net_corrections']):+d}** and the selected margin is **{selected['delta']:.9g}**.
5. Cross-fit chose the global margin in **{stability['global_selected_in_folds']}/{stability['folds']}** folds; all folds identical: **{stability['all_folds_same_margin']}**.
6. Intervention rate changed from **{float(zero['intervention_rate']):.4%}** to **{float(chosen['intervention_rate']):.4%}**, consistent with stronger abstention.
7. Median maximum executed non-FULL margins for W→C and C→W were **{transition_diag['W→C']['maximum_median']}** and **{transition_diag['C→W']['maximum_median']}** respectively. This is descriptive only.
8. Action-specific raw-margin distributions are in `diagnostics/margin_by_action.csv`; no action-conditioned rule was tuned. The largest executed-action count identifies whether one action dominates.
9. A nonzero global margin is **{'supported' if positive else 'not supported'}** on Historical-800.
10. External status: **{external_status}**.{external_line}
11. A negative result would reject this one-dimensional post-hoc confidence gate; it would not prove that Stage-2 correction is impossible or identify a unique representation/training defect.

The reviewer caveat remains: final-epoch sampler weighting can make this fixed grid coarse. Development outcomes were not used to add thresholds.
"""
    _atomic_bytes(output_root / "summaries/abstention_margin_summary.md", summary.encode())
    if positive and external_summary:
        external_overall = next(
            row for row in external_summary["comparison"] if row["benchmark_family"] == "overall"
        )
        if int(external_overall["margin_net"]) > int(external_overall["original_net"]):
            recommendation = "Freeze the margin as a method component; in a separately authorized action, decide whether correction coverage should be increased without retuning on this external set."
        else:
            recommendation = "Do not retune the margin on external outcomes; in a separately authorized diagnostic, investigate why development selectivity did not transfer."
    else:
        recommendation = "Move away from post-hoc global confidence gating and, in a separately authorized action, revisit the Stage-2 representation or training signal."
    _atomic_bytes(
        output_root / "summaries/next_improvement_recommendation.md",
        ("# Next improvement recommendation\n\n" + recommendation + "\n").encode(),
    )
    if not positive:
        marker = output_root / "external/run_only_if_authorized/NOT_RUN.md"
        _atomic_bytes(
            marker,
            b"# External rerun not launched\n\nDevelopment selected delta=0, so the plan forbids another external rerun.\n",
        )
    atomic_json(output_root / "artifact_manifest.json", _artifact_manifest(output_root, contract["contract_sha256"]))
    print(json.dumps({"selected": selected, "external_status": external_status}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "training-margins", "freeze-grid", "development", "finalize-development", "prepare-external", "external", "finalize-external", "finalize-reports"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.config)
    elif args.mode == "training-margins":
        training_margin_worker(args.config)
    elif args.mode == "freeze-grid":
        freeze_grid(args.config)
    elif args.mode == "development":
        development_worker(args.config)
    elif args.mode == "finalize-development":
        finalize_development(args.config)
    elif args.mode == "prepare-external":
        prepare_external(args.config)
    elif args.mode == "external":
        external_worker(args.config)
    elif args.mode == "finalize-external":
        finalize_external(args.config)
    else:
        finalize_reports(args.config)


if __name__ == "__main__":
    main()
