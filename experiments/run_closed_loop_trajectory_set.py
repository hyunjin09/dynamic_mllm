#!/usr/bin/env python3
"""Run the frozen closed-loop trajectory-set Stage-2 train/evaluation phase."""

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
import random
import subprocess
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
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel  # noqa: E402

from binary_policy.executor import (  # noqa: E402
    capture_four_action_route,
    capture_four_action_suffix_from_full_baseline,
)
from binary_policy.executor.four_action import capture_online_four_action_route  # noqa: E402
from binary_policy.executor.inputs import build_binary_inputs  # noqa: E402
from dense_failure_stage1.runtime import build_dense_inputs, configure_dense_determinism  # noqa: E402
from dense_failure_stage2.closed_loop_trajectory_set import (  # noqa: E402
    build_trajectory_sets,
    compact_router_state,
    compute_uid_trajectory_loss,
    hierarchical_uid_weights,
    prefix_action_hash,
    validate_state_references,
)
from dense_failure_stage2.full_benchmark_eval import paired_bootstrap  # noqa: E402
from dense_failure_stage2.v1_router import (  # noqa: E402
    ACTION_NAMES,
    ACTION_TO_INDEX,
    SharedReadWriteRouter,
)
from experiments.run_full_benchmark_end_to_end_eval import (  # noqa: E402
    Runtime as DenseEvalRuntime,
    _build_inputs as eval_build_inputs,
    _generate as eval_generate,
    _native_dense_generation,
    _stage1_scores,
    _verify_images,
)
from experiments.run_stage2_v1_training_revised import (  # noqa: E402
    _generate as train_generate,
    _load_model,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/closed_loop_trajectory_set_v1.json"
ALLOWED_ROOTS = (PROJECT_ROOT.resolve(), Path("/mnt/hyemin").resolve())
BOUND_CODE = (
    "configs/closed_loop_trajectory_set_v1.json",
    "dense_failure_stage2/closed_loop_trajectory_set.py",
    "experiments/run_closed_loop_trajectory_set.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/model.py",
    "dense_failure_stage1/lmms_scoring.py",
    "dense_failure_stage1/runtime.py",
    "dense_failure_stage2/full_benchmark_eval.py",
    "dense_failure_stage2/v1_router.py",
    "experiments/run_full_benchmark_end_to_end_eval.py",
    "experiments/run_stage2_v1_training_revised.py",
    "plans/closed_loop_trajectory_set_stage2_full_train_eval_plan.md",
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
    contiguous = tensor.detach().cpu().contiguous()
    header = f"{contiguous.dtype}:{tuple(contiguous.shape)}:".encode()
    return sha256(header + contiguous.view(torch.uint8).numpy().tobytes()).hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def read_json(value: str | Path) -> dict[str, Any]:
    data = json.loads(resolve_path(value).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {value}")
    return data


def read_jsonl(value: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with resolve_path(value).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{value}:{line_number} is not an object")
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


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames=None) -> None:
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"cannot infer columns for empty CSV: {path}")
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


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _command(*args: str) -> str:
    result = subprocess.run(
        args, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(f"command failed {args}: {result.stderr.strip()}")
    return result.stdout.strip()


def _git_state() -> dict[str, str]:
    return {
        "commit": _command("git", "rev-parse", "HEAD"),
        "branch": _command("git", "branch", "--show-current"),
        "worktree_status": _command("git", "status", "--short"),
    }


def _runtime_state() -> dict[str, Any]:
    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "lmms_eval": version("lmms-eval"),
        "numpy": version("numpy"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "nvidia_driver": _command(
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ).splitlines()[0],
    }


def load_config(value: str | Path) -> dict[str, Any]:
    config = read_json(value)
    if config.get("schema_version") != "closed_loop_trajectory_set_config_v1":
        raise ValueError("unsupported closed-loop trajectory-set config")
    if int(config.get("world_size", 0)) != 4:
        raise ValueError("the phase requires all four direct GPUs")
    if tuple(config["stage2"]["action_order"]) != ACTION_NAMES:
        raise ValueError("router action order differs from the frozen executor")
    if config["corpus"]["operating_point"] != "P90":
        raise ValueError("only the frozen P90 handoff is supported")
    if config["stage1"]["comparison"] != "strict_greater_than":
        raise ValueError("Stage-1 comparison semantics differ")
    if not config["training"]["uid_loss_unit"] or config["training"]["route_sampling"]:
        raise ValueError("training must use exact per-UID trajectory sets")
    if any(
        config["stage2"][key]
        for key in (
            "layer_embedding",
            "dataset_or_source_input",
            "stage1_score_input",
            "pretrigger_history_input",
        )
    ):
        raise ValueError("the primary representation contains a forbidden input")
    return config


def _router(config: Mapping[str, Any], device: torch.device) -> SharedReadWriteRouter:
    settings = config["stage2"]
    model = SharedReadWriteRouter(
        hidden_size=int(config["model"]["hidden_size"]),
        router_size=int(settings["router_size"]),
        num_heads=int(settings["num_heads"]),
        dropout=float(settings["dropout"]),
    ).to(device=device, dtype=torch.float32)
    checkpoint = torch.load(
        resolve_path(settings["checkpoint"]), map_location="cpu", weights_only=False
    )
    if checkpoint.get("contract_sha256") != settings["parent_contract_sha256"]:
        raise RuntimeError("Sequential-A checkpoint parent contract differs")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def _source_paths(config: Mapping[str, Any]) -> dict[str, str]:
    return {
        "phase74_contract": config["corpus"]["phase74_contract"],
        "phase74_artifact_manifest": config["corpus"]["phase74_artifact_manifest"],
        "program_manifest": config["corpus"]["program_manifest"],
        "program_replay": config["corpus"]["replay_validation"],
        "internal_split": config["corpus"]["internal_split"],
        "stage2a_checkpoint": config["stage2"]["checkpoint"],
        "evaluation_config": config["evaluation"]["base_config"],
        "phase69_contract": config["evaluation"]["phase69_contract"],
        "phase69_paired": config["evaluation"]["phase69_paired"],
        "phase74_eval_contract": config["evaluation"]["phase74_contract"],
        "phase74_paired": config["evaluation"]["phase74_paired"],
    }


def _verify_model_snapshot(snapshot: Path, expected: Mapping[str, str]) -> None:
    for relative, digest in expected.items():
        path = snapshot / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise RuntimeError(f"model snapshot file differs: {path}")


def prepare(config_path: Path) -> None:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    external_cache = resolve_path(config["external_state_cache_root"])
    if output_root.exists() or external_cache.exists():
        raise RuntimeError("refusing to overwrite an existing Phase-76 output/cache root")

    phase74 = read_json(config["corpus"]["phase74_contract"])
    phase74_artifacts = read_json(config["corpus"]["phase74_artifact_manifest"])
    phase69 = read_json(config["evaluation"]["phase69_contract"])
    if phase74.get("contract_sha256") != "0c1696ed895353f31ba40201633215f235bacc7175624dccd2ac8d7fdb597a49":
        raise RuntimeError("Phase-74 corpus contract differs")
    if phase69.get("contract_sha256") != "63379eeff80fd5b046cb980cdfccea7fab232e15eb5b14924a24b60393327e83":
        raise RuntimeError("Phase-69 evaluation contract differs")
    for key, expected in (
        ("program_manifest", config["corpus"]["program_manifest_sha256"]),
        ("program_replay", config["corpus"]["replay_validation_sha256"]),
        ("internal_split", config["corpus"]["internal_split_sha256"]),
    ):
        if file_sha256(_source_paths(config)[key]) != expected:
            raise RuntimeError(f"frozen corpus input differs: {key}")
    if phase74_artifacts["files"]["corpus/p90_program_corpus_manifest.jsonl"] != config["corpus"]["program_manifest_sha256"]:
        raise RuntimeError("Phase-74 artifact manifest does not bind the program corpus")
    if file_sha256(config["stage2"]["checkpoint"]) != config["stage2"]["checkpoint_sha256"]:
        raise RuntimeError("Sequential-A initialization checkpoint differs")
    _verify_model_snapshot(resolve_path(config["model"]["snapshot_path"]), phase69["model_snapshot_sha256"])

    programs = read_jsonl(config["corpus"]["program_manifest"])
    grouped, summary = build_trajectory_sets(
        programs, total_layers=int(config["model"]["decoder_layers"])
    )
    expected_summary = {
        "uids": int(config["corpus"]["expected_uids"]),
        "programs": int(config["corpus"]["expected_programs"]),
        "dense_c_uids": int(config["corpus"]["expected_dense_c_uids"]),
        "dense_w_uids": int(config["corpus"]["expected_dense_w_uids"]),
        "route_state_occurrences": int(config["corpus"]["expected_route_state_occurrences"]),
        "unique_prefix_states": int(config["corpus"]["expected_unique_prefix_states"]),
    }
    if summary != expected_summary:
        raise RuntimeError(f"trajectory corpus census differs: {summary}")
    replay = read_jsonl(config["corpus"]["replay_validation"])
    if (
        len(replay) != summary["programs"]
        or {row["program_id"] for row in replay} != {row["program_id"] for row in programs}
        or not all(bool(row.get("passed")) and bool(row.get("correct")) and bool(row.get("exact_token_parity")) for row in replay)
    ):
        raise RuntimeError("Phase-74 replay-valid program population differs")

    split = read_json(config["corpus"]["internal_split"])
    uid_split = {str(uid): str(value) for uid, value in split["uid_split"].items()}
    if set(uid_split) != set(grouped) or set(uid_split.values()) != {"train", "dev"}:
        raise RuntimeError("internal UID split differs from the full corpus")
    train_groups = {
        rows[0]["image_group_id"] for uid, rows in grouped.items() if uid_split[uid] == "train"
    }
    dev_groups = {
        rows[0]["image_group_id"] for uid, rows in grouped.items() if uid_split[uid] == "dev"
    }
    if train_groups.intersection(dev_groups):
        raise RuntimeError("internal development split leaks image groups")

    git = _git_state()
    runtime = _runtime_state()
    if not runtime["cuda_available"] or int(runtime["cuda_device_count"]) != 4:
        raise RuntimeError("the frozen run requires four available CUDA devices")
    output_root.mkdir(parents=True, exist_ok=False)
    external_cache.mkdir(parents=True, exist_ok=False)
    for relative in (
        "corpus", "objective", "splits", "training", "evaluation/chartqa",
        "evaluation/textvqa", "evaluation/mmmu_pro", "evaluation/pope", "metrics",
        "diagnostics", "figures", "summaries", "work/replay", "work/evaluation", "smoke",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    state_link = output_root / "corpus/routed_states"
    state_link.symlink_to(external_cache, target_is_directory=True)

    trajectory_rows = []
    for row in programs:
        copy = dict(row)
        copy.pop("program_weight", None)
        copy["schema_version"] = "closed_loop_trajectory_set_manifest_v1"
        trajectory_rows.append(copy)
    atomic_jsonl(output_root / "corpus/trajectory_set_manifest.jsonl", trajectory_rows)
    atomic_json(
        output_root / "corpus/uid_to_trajectories.json",
        {
            uid: {
                "program_ids": [row["program_id"] for row in rows],
                "trajectory_count": len(rows),
                "dataset": rows[0]["dataset"],
                "source_regime": rows[0]["source_regime"],
                "dense_outcome": rows[0]["dense_outcome"],
                "trigger_layer": rows[0]["trigger_layer"],
                "image_group_id": rows[0]["image_group_id"],
                "internal_split": uid_split[uid],
            }
            for uid, rows in grouped.items()
        },
    )
    atomic_jsonl(
        output_root / "corpus/trajectory_provenance.jsonl",
        (
            {
                "uid": row["uid"],
                "program_id": row["program_id"],
                "provenance": row["provenance"],
                "source_references": row["source_references"],
            }
            for row in trajectory_rows
        ),
    )
    provenance = Counter(source for row in programs for source in row["provenance"])
    atomic_csv(
        output_root / "corpus/corpus_summary.csv",
        [{**summary, "image_groups": len(train_groups | dev_groups), **{
            key: provenance[key] for key in (
                "preservation", "single", "original_mcts", "robust_search", "completeness_audit"
            )
        }}],
    )
    atomic_csv(
        output_root / "corpus/trajectory_count_per_uid.csv",
        [
            {
                "uid": uid,
                "dataset": rows[0]["dataset"],
                "source_regime": rows[0]["source_regime"],
                "dense_outcome": rows[0]["dense_outcome"],
                "trigger_layer": rows[0]["trigger_layer"],
                "trajectories": len(rows),
            }
            for uid, rows in grouped.items()
        ],
    )
    atomic_json(output_root / "splits/internal_dev_groups.json", split)
    atomic_json(
        output_root / "splits/full_refit_manifest.json",
        {"uids": sorted(grouped), "program_ids": sorted(row["program_id"] for row in programs), **summary},
    )
    replay_schedule = [
        {
            "uid": uid,
            "worker_rank": index % int(config["world_size"]),
            "program_ids": [row["program_id"] for row in rows],
            "program_count": len(rows),
            "sample": rows[0]["sample"],
        }
        for index, (uid, rows) in enumerate(grouped.items())
    ]
    atomic_jsonl(output_root / "work/replay_manifest.jsonl", replay_schedule)
    _atomic_bytes(
        output_root / "corpus/state_feature_schema.md",
        (
            "# Routed-state cache schema\n\n"
            "Each unique `(UID, layer, exact post-trigger action prefix)` state stores the final valid "
            "text/query hidden vector, the exact routed visual-state tensor and mask, action-prefix hash, "
            "and tensor hashes. The state is captured immediately before the current layer action. "
            "The visual tensor is not compacted because shape changes can alter attention logits by ulps.\n"
        ).encode(),
    )

    internal_paths = (
        "corpus/trajectory_set_manifest.jsonl",
        "corpus/uid_to_trajectories.json",
        "corpus/trajectory_provenance.jsonl",
        "corpus/corpus_summary.csv",
        "corpus/trajectory_count_per_uid.csv",
        "corpus/state_feature_schema.md",
        "splits/internal_dev_groups.json",
        "splits/full_refit_manifest.json",
        "work/replay_manifest.jsonl",
    )
    contract = {
        "schema_version": "closed_loop_trajectory_set_contract_v1",
        "created_at": utc_now(),
        "static_config": config,
        "config_sha256": file_sha256(config_path),
        "source_sha256": {key: file_sha256(path) for key, path in _source_paths(config).items()},
        "bound_code_sha256": {path: file_sha256(path) for path in BOUND_CODE},
        "internal_manifest_sha256": {
            path: file_sha256(output_root / path) for path in internal_paths
        },
        "parent_contracts": {
            "phase69": phase69["contract_sha256"],
            "phase74": phase74["contract_sha256"],
            "stage2a": config["stage2"]["parent_contract_sha256"],
        },
        "model_snapshot_sha256": phase69["model_snapshot_sha256"],
        "git": git,
        "runtime": runtime,
        "population": {**summary, "image_groups": len(train_groups | dev_groups)},
        "external_state_cache_root": str(external_cache),
        "review_reconciliation": {
            "verdict": "stable",
            "confidence": "medium",
            "required_gates": ["all_route_state_replay", "objective_numerical_parity", "objective_gradient_parity"],
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_protocol.json", contract)
    protocol = f"""# Closed-loop trajectory-set Stage-2 frozen protocol

- Contract: `{contract['contract_sha256']}`
- Population: 569 P90-triggered UIDs / 4,948 replay-valid complete programs.
- Dense-C: one all-FULL preservation route. Dense-W: every unique successful route.
- State: exact pre-action routed query + visual inputs, keyed by action-prefix hash.
- Objective: one length-normalized log-mean-exp successful-trajectory loss per UID.
- Router: frozen architecture initialized exactly from Sequential-A; READ/WRITE/shared head trainable, Qwen and Stage-1 frozen.
- Selection: image-group-disjoint internal dev selects only epoch count; full refit uses all 569 UIDs.
- Evaluation: exact Phase-69 19,960-row ChartQA/TextVQA/MMMU-Pro/POPE protocol with deterministic greedy closed-loop routing.
- No layer/source IDs, Stage-1 score/history input, route preference, sampling, beam, MCTS, or external-set tuning.
"""
    _atomic_bytes(output_root / "protocol.md", protocol.encode())
    print(json.dumps({"contract_sha256": contract["contract_sha256"], **contract["population"]}, sort_keys=True))


def verify_contract(config_path: Path, *, verify_model: bool = False) -> tuple[dict[str, Any], Path]:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if contract.get("contract_sha256") != canonical_hash(contract):
        raise RuntimeError("Phase-76 frozen contract hash differs")
    if contract.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError("Phase-76 config changed after freeze")
    for path, digest in contract["bound_code_sha256"].items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"Phase-76 bound code changed after freeze: {path}")
    for key, path in _source_paths(config).items():
        if file_sha256(path) != contract["source_sha256"][key]:
            raise RuntimeError(f"Phase-76 source changed after freeze: {key}")
    for path, digest in contract["internal_manifest_sha256"].items():
        if file_sha256(output_root / path) != digest:
            raise RuntimeError(f"Phase-76 prepared manifest changed after freeze: {path}")
    if _git_state() != contract["git"]:
        raise RuntimeError("Git commit/branch/worktree status differs from the frozen contract")
    if _runtime_state() != contract["runtime"]:
        raise RuntimeError("runtime/environment differs from the frozen contract")
    cache_root = resolve_path(config["external_state_cache_root"])
    state_link = output_root / "corpus/routed_states"
    if not state_link.is_symlink() or state_link.resolve() != cache_root or not cache_root.is_dir():
        raise RuntimeError("routed-state cache symlink differs from the frozen location")
    if verify_model:
        _verify_model_snapshot(resolve_path(config["model"]["snapshot_path"]), contract["model_snapshot_sha256"])
    return contract, output_root


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


def _load_parent_trigger_index(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    root = resolve_path(config["corpus"]["phase74_contract"]).parent
    rows = read_jsonl(root / "features/trigger_state_index.jsonl")
    if len(rows) != int(config["corpus"]["expected_uids"]):
        raise RuntimeError("Phase-74 trigger-state index count differs")
    return {str(row["uid"]): row for row in rows}


def _load_parent_trigger_state(
    config: Mapping[str, Any], index_row: Mapping[str, Any]
) -> dict[str, Any]:
    root = resolve_path(config["corpus"]["phase74_contract"]).parent
    path = root / str(index_row["state_file"])
    if file_sha256(path) != str(index_row["state_file_sha256"]):
        raise RuntimeError(f"Phase-74 trigger-state file hash differs: {index_row['uid']}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    for name, digest in index_row["tensor_sha256"].items():
        if tensor_sha256(state[name]) != digest:
            raise RuntimeError(f"Phase-74 trigger tensor differs: {index_row['uid']} {name}")
    return state


def _build_uid_cache(
    *,
    config: Mapping[str, Any],
    contract_sha256: str,
    processor,
    wrapped,
    parity_router: SharedReadWriteRouter,
    rows: Sequence[Mapping[str, Any]],
    parent_trigger_row: Mapping[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    uid = str(rows[0]["uid"])
    if any(str(row["uid"]) != uid for row in rows):
        raise ValueError("UID cache construction received mixed UIDs")
    trigger = int(rows[0]["trigger_layer"])
    sample = dict(rows[0]["sample"])
    device = next(parity_router.parameters()).device
    started = time.monotonic()
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
    dense_ids, dense_text, dense_score = train_generate(
        processor, wrapped, baseline, inputs["input_ids"], sample
    )
    expected_dense = [int(value) for value in rows[0]["dense_generated_token_ids"]]
    if dense_ids != expected_dense or bool(dense_score.correct) != (rows[0]["dense_outcome"] == "C"):
        raise RuntimeError(f"native dense parity failed for {uid}")

    parent = _load_parent_trigger_state(config, parent_trigger_row)
    live_text, live_visual = baseline.pre_layer_states[trigger]
    parent_exact = {
        "text_states": torch.equal(live_text.detach().cpu(), parent["text_states"]),
        "visual_states": torch.equal(live_visual.detach().cpu(), parent["visual_states"]),
        "text_mask": torch.equal(prepared.text_valid_mask.detach().cpu(), parent["text_mask"]),
        "visual_mask": torch.equal(prepared.visual_valid_mask.detach().cpu(), parent["visual_mask"]),
    }
    if int(parent["trigger_layer"]) != trigger or not all(parent_exact.values()):
        raise RuntimeError(f"P90 trigger-state parity failed for {uid}: {parent_exact}")

    states: dict[str, dict[str, Any]] = {}
    programs: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    compact_full_logit_parity = True
    parity_router.eval()
    for row in rows:
        suffix = list(row["suffix_actions"])
        routed = (
            baseline
            if all(action == "FULL" for action in suffix)
            else capture_four_action_suffix_from_full_baseline(
                wrapped, baseline, trigger, suffix
            )
        )
        state_ids: list[str] = []
        prefix: list[str] = []
        for offset, action in enumerate(suffix):
            layer = trigger + offset
            state_id = prefix_action_hash(uid, trigger, layer, prefix)
            text_state, visual_state = routed.pre_layer_states[layer]
            compact = compact_router_state(
                text_state,
                visual_state,
                prepared.text_valid_mask,
                prepared.visual_valid_mask,
            )
            cached = {
                name: value.detach().cpu().to(torch.bfloat16).contiguous()
                if name in {"text_states", "visual_states"}
                else value.detach().cpu().bool().contiguous()
                for name, value in compact.items()
            }
            hashes = _state_tensor_hashes(cached)
            state_hash = _combined_state_hash(hashes)
            if state_id in states:
                if states[state_id]["state_sha256"] != state_hash:
                    raise RuntimeError(f"shared prefix produced different state bytes: {uid} {state_id}")
            else:
                with torch.no_grad():
                    full_logits = parity_router(
                        text_state,
                        visual_state,
                        text_mask=prepared.text_valid_mask,
                        visual_mask=prepared.visual_valid_mask,
                    )
                    compact_logits = parity_router(
                        compact["text_states"],
                        compact["visual_states"],
                        text_mask=compact["text_mask"],
                        visual_mask=compact["visual_mask"],
                    )
                exact = torch.equal(full_logits, compact_logits)
                compact_full_logit_parity = compact_full_logit_parity and exact
                if not exact:
                    raise RuntimeError(f"compact/full router-logit parity failed: {uid} L{layer}")
                states[state_id] = {
                    **cached,
                    "state_id": state_id,
                    "state_sha256": state_hash,
                    "tensor_sha256": hashes,
                    "uid": uid,
                    "layer": layer,
                    "trigger_layer": trigger,
                    "prefix_actions": list(prefix),
                    "prefix_action_hash": state_id,
                }
                state_rows.append({
                    "schema_version": "closed_loop_routed_state_index_v1",
                    "contract_sha256": contract_sha256,
                    "uid": uid,
                    "state_id": state_id,
                    "state_sha256": state_hash,
                    "tensor_sha256": hashes,
                    "layer": layer,
                    "trigger_layer": trigger,
                    "prefix_actions": list(prefix),
                    "text_tokens": int(cached["text_states"].shape[1]),
                    "visual_tokens": int(cached["visual_states"].shape[1]),
                })
            state_ids.append(state_id)
            prefix.append(action)
        generated_ids, generated_text, score = train_generate(
            processor, wrapped, routed, inputs["input_ids"], sample
        )
        expected = [int(value) for value in row["expected_generated_token_ids"]]
        passed = generated_ids == expected and bool(score.correct)
        replay_row = {
            "schema_version": "closed_loop_trajectory_replay_v1",
            "contract_sha256": contract_sha256,
            "uid": uid,
            "program_id": str(row["program_id"]),
            "trigger_layer": trigger,
            "state_ids": state_ids,
            "action_indices": [ACTION_TO_INDEX[action] for action in suffix],
            "actions": suffix,
            "exact_token_parity": generated_ids == expected,
            "generated_token_ids": generated_ids,
            "generated_answer": generated_text,
            "lmms_metric": score.metric_name,
            "lmms_score": score.raw_score,
            "correct": bool(score.correct),
            "passed": passed,
        }
        if not passed:
            raise RuntimeError(f"exact routed replay failed: {row['program_id']}")
        replay_rows.append(replay_row)
        programs.append({
            "program_id": str(row["program_id"]),
            "state_ids": state_ids,
            "action_indices": [ACTION_TO_INDEX[action] for action in suffix],
            "actions": suffix,
            "provenance": list(row["provenance"]),
        })
        if routed is not baseline:
            del routed

    payload = {
        "schema_version": "closed_loop_uid_routed_state_cache_v1",
        "contract_sha256": contract_sha256,
        "uid": uid,
        "dataset": rows[0]["dataset"],
        "source_regime": rows[0]["source_regime"],
        "dense_outcome": rows[0]["dense_outcome"],
        "image_group_id": rows[0]["image_group_id"],
        "trigger_layer": trigger,
        "states": states,
        "programs": programs,
    }
    atomic_torch(state_path, payload)
    roundtrip = torch.load(state_path, map_location="cpu", weights_only=False)
    if (
        roundtrip.get("contract_sha256") != contract_sha256
        or roundtrip.get("uid") != uid
        or set(roundtrip["states"]) != set(states)
    ):
        raise RuntimeError(f"routed-state cache round trip failed: {uid}")
    for state_id, state in roundtrip["states"].items():
        if _state_tensor_hashes(state) != state["tensor_sha256"] or _combined_state_hash(state["tensor_sha256"]) != state["state_sha256"]:
            raise RuntimeError(f"routed-state tensor hash failed after save: {uid} {state_id}")
    relative_state_file = str(state_path.relative_to(resolve_path(config["output_root"])))
    for state_row in state_rows:
        state_row["state_file"] = relative_state_file
    result = {
        "schema_version": "closed_loop_uid_replay_result_v1",
        "contract_sha256": contract_sha256,
        "uid": uid,
        "trigger_layer": trigger,
        "dense_outcome": rows[0]["dense_outcome"],
        "state_file": relative_state_file,
        "state_file_sha256": file_sha256(state_path),
        "program_count": len(programs),
        "state_count": len(states),
        "state_rows": state_rows,
        "program_replays": replay_rows,
        "parent_trigger_state_exact": parent_exact,
        "compact_full_router_logits_exact": compact_full_logit_parity,
        "dense_generated_token_ids": dense_ids,
        "dense_generated_answer": dense_text,
        "dense_correct": bool(dense_score.correct),
        "consumed_image_sha256": metadata["consumed_image_sha256"],
        "elapsed_seconds": time.monotonic() - started,
    }
    del baseline, prepared, inputs, payload, roundtrip
    return result


def _smoke_uid_rows(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]], count: int
) -> list[str]:
    selected: list[str] = []
    covered_actions: set[str] = set()
    covered_outcomes: set[str] = set()
    candidates = sorted(grouped)
    while candidates and len(selected) < count:
        best = max(
            candidates,
            key=lambda uid: (
                len({action for row in grouped[uid] for action in row["suffix_actions"]} - covered_actions)
                + int(str(grouped[uid][0]["dense_outcome"]) not in covered_outcomes),
                sha256(uid.encode()).hexdigest(),
            ),
        )
        selected.append(best)
        covered_actions.update(action for row in grouped[best] for action in row["suffix_actions"])
        covered_outcomes.add(str(grouped[best][0]["dense_outcome"]))
        candidates.remove(best)
    if len(selected) != count or covered_actions != set(ACTION_NAMES) or covered_outcomes != {"C", "W"}:
        raise RuntimeError("cannot construct the required implementation-smoke cohort")
    return selected


def smoke(config_path: Path, device_index: int) -> None:
    contract, output_root = verify_contract(config_path, verify_model=True)
    config = contract["static_config"]
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    configure_dense_determinism(int(config["seed"]), config["backend_settings"])
    processor, base, wrapped = _load_model(config, device)
    router = _router(config, device)
    programs = read_jsonl(output_root / "corpus/trajectory_set_manifest.jsonl")
    grouped, _summary = build_trajectory_sets(programs, total_layers=28)
    selected = _smoke_uid_rows(grouped, int(config["smoke"]["training_uids"]))
    parent_index = _load_parent_trigger_index(config)
    results = []
    for uid in selected:
        path = output_root / f"corpus/routed_states/smoke/{_uid_slug(uid)}.pt"
        results.append(_build_uid_cache(
            config=config,
            contract_sha256=contract["contract_sha256"],
            processor=processor,
            wrapped=wrapped,
            parity_router=router,
            rows=grouped[uid],
            parent_trigger_row=parent_index[uid],
            state_path=path,
        ))

    training_router = _router(config, device).train()
    sample_payload = torch.load(
        output_root / results[0]["state_file"], map_location="cpu", weights_only=False
    )
    training_router.zero_grad(set_to_none=True)
    loss, objective_stats = compute_uid_trajectory_loss(
        training_router,
        sample_payload,
        device=device,
        state_microbatch=int(config["training"]["state_forward_microbatch"]),
    )
    loss.backward()
    gradients = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in training_router.parameters()
    )
    checkpoint_path = output_root / "smoke/checkpoint_roundtrip.pt"
    atomic_torch(checkpoint_path, training_router.state_dict())
    reloaded = _router(config, device)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    checkpoint_exact = all(
        torch.equal(value.detach().cpu(), reloaded.state_dict()[name].detach().cpu())
        for name, value in training_router.state_dict().items()
    )

    feedback_uid = selected[0]
    feedback_rows = grouped[feedback_uid]
    sample = feedback_rows[0]["sample"]
    trigger = int(feedback_rows[0]["trigger_layer"])
    inputs, _ = build_dense_inputs(processor, sample, device)
    prepared = build_binary_inputs(wrapped, inputs)
    trace: list[dict[str, Any]] = []
    reloaded.eval()

    def selector(layer, text_state, visual_state, current_meta):
        hashes = {
            "text_states": tensor_sha256(text_state),
            "visual_states": tensor_sha256(visual_state),
        }
        if layer < trigger:
            action = "FULL"
            logits = None
        else:
            current = reloaded(
                text_state,
                visual_state,
                text_mask=current_meta.text_valid_mask,
                visual_mask=current_meta.visual_valid_mask,
            )[0]
            action = ACTION_NAMES[int(current.argmax().item())]
            logits = [float(value) for value in current.detach().cpu().tolist()]
        trace.append({
            "layer": int(layer),
            "action": action,
            "state_sha256": _combined_state_hash(hashes),
            "logits": logits,
        })
        return action

    online = capture_online_four_action_route(
        wrapped, {}, selector, prepared_inputs=prepared, use_cache=True, native_full_rows=True
    )
    feedback_exact = True
    for item in trace:
        text_state, visual_state = online.pre_layer_states[item["layer"]]
        observed = _combined_state_hash({
            "text_states": tensor_sha256(text_state),
            "visual_states": tensor_sha256(visual_state),
        })
        feedback_exact = feedback_exact and observed == item["state_sha256"]
    _ids, _answer, _score = train_generate(
        processor, wrapped, online, inputs["input_ids"], sample
    )
    state_feedback_active = any(
        trace[index]["state_sha256"] != trace[index - 1]["state_sha256"]
        for index in range(1, len(trace))
    )
    passed = (
        len(results) == int(config["smoke"]["training_uids"])
        and all(all(row["parent_trigger_state_exact"].values()) for row in results)
        and all(row["compact_full_router_logits_exact"] for row in results)
        and all(all(item["passed"] for item in row["program_replays"]) for row in results)
        and gradients
        and checkpoint_exact
        and feedback_exact
        and state_feedback_active
        and all(parameter.grad is None for parameter in base.parameters())
    )
    report = {
        "schema_version": "closed_loop_trajectory_set_smoke_v1",
        "contract_sha256": contract["contract_sha256"],
        "passed": passed,
        "uids": selected,
        "uid_count": len(selected),
        "programs": sum(row["program_count"] for row in results),
        "states": sum(row["state_count"] for row in results),
        "all_route_replay_exact": all(all(item["passed"] for item in row["program_replays"]) for row in results),
        "all_compact_full_router_logits_exact": all(row["compact_full_router_logits_exact"] for row in results),
        "objective_loss": float(loss.detach().item()),
        "objective_stats": objective_stats,
        "all_router_gradients_finite": gradients,
        "backbone_gradients_all_none": all(parameter.grad is None for parameter in base.parameters()),
        "checkpoint_roundtrip_exact": checkpoint_exact,
        "state_feedback_trace_exact": feedback_exact,
        "state_feedback_active": state_feedback_active,
        "feedback_uid": feedback_uid,
        "feedback_trace": trace,
        "results": results,
    }
    atomic_json(output_root / "smoke/implementation_smoke.json", report)
    _atomic_bytes(
        output_root / "objective/objective_implementation.md",
        (
            "# Exact trajectory-set objective\n\n"
            "For each UID, every unique prefix state is forwarded once. Each route log-probability is the "
            "sum of its selected action log-probabilities. The loss is `-(logsumexp(route_logp)-log K)/T`. "
            "All K routes remain in one autograd graph; route sampling and hard responsibilities are disabled.\n"
        ).encode(),
    )
    atomic_json(output_root / "objective/numerical_stability_tests.json", {
        "contract_sha256": contract["contract_sha256"],
        "passed": bool(torch.isfinite(loss)),
        "actual_smoke_loss": float(loss.detach().item()),
        "long_suffix_unit_test": "tests/test_closed_loop_trajectory_set.py",
    })
    atomic_json(output_root / "objective/gradient_parity_tests.json", {
        "contract_sha256": contract["contract_sha256"],
        "passed": gradients,
        "brute_force_unit_test": "tests/test_closed_loop_trajectory_set.py",
        "all_router_gradients_finite": gradients,
    })
    _atomic_bytes(
        output_root / "objective/numerical_stability_tests.md",
        f"# Numerical stability tests\n\n- Passed: **{bool(torch.isfinite(loss))}**\n- Smoke loss: {float(loss.detach().item()):.8f}\n".encode(),
    )
    _atomic_bytes(
        output_root / "objective/gradient_parity_tests.md",
        f"# Gradient parity tests\n\n- Passed: **{gradients}**\n- Brute-force equality is covered by the focused unit test.\n".encode(),
    )
    print(json.dumps({key: report[key] for key in ("passed", "uid_count", "programs", "states")}, sort_keys=True))
    if not passed:
        raise RuntimeError("closed-loop trajectory-set implementation smoke failed")


def replay_worker(config_path: Path, rank: int, resume: bool) -> None:
    contract, output_root = verify_contract(config_path, verify_model=True)
    config = contract["static_config"]
    smoke_report = read_json(output_root / "smoke/implementation_smoke.json")
    if smoke_report.get("contract_sha256") != contract["contract_sha256"] or not smoke_report.get("passed"):
        raise RuntimeError("implementation smoke has not passed under this contract")
    rank = int(rank)
    if rank < 0 or rank >= int(config["world_size"]):
        raise ValueError("invalid replay rank")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    configure_dense_determinism(int(config["seed"]) + rank, config["backend_settings"])
    processor, _base, wrapped = _load_model(config, device)
    router = _router(config, device).eval()
    programs = read_jsonl(output_root / "corpus/trajectory_set_manifest.jsonl")
    grouped, _ = build_trajectory_sets(programs, total_layers=28)
    schedule = [
        row for row in read_jsonl(output_root / "work/replay_manifest.jsonl")
        if int(row["worker_rank"]) == rank
    ]
    parent_index = _load_parent_trigger_index(config)
    rank_root = output_root / f"work/replay/rank{rank:02d}"
    completed = 0
    started = time.monotonic()
    for item in schedule:
        uid = str(item["uid"])
        result_path = rank_root / f"{_uid_slug(uid)}.json"
        state_path = output_root / f"corpus/routed_states/full/{_uid_slug(uid)}.pt"
        if resume and result_path.exists() and state_path.exists():
            old = read_json(result_path)
            if (
                old.get("contract_sha256") == contract["contract_sha256"]
                and old.get("uid") == uid
                and file_sha256(state_path) == old.get("state_file_sha256")
            ):
                completed += 1
                continue
        try:
            result = _build_uid_cache(
                config=config,
                contract_sha256=contract["contract_sha256"],
                processor=processor,
                wrapped=wrapped,
                parity_router=router,
                rows=grouped[uid],
                parent_trigger_row=parent_index[uid],
                state_path=state_path,
            )
        except Exception as error:
            atomic_json(rank_root / f"failure_{_uid_slug(uid)}.json", {
                "contract_sha256": contract["contract_sha256"],
                "uid": uid,
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": utc_now(),
            })
            raise
        atomic_json(result_path, result)
        completed += 1
        if completed % 10 == 0 or completed == len(schedule):
            print(json.dumps({
                "rank": rank,
                "completed_uids": completed,
                "assigned_uids": len(schedule),
                "elapsed_seconds": time.monotonic() - started,
            }), flush=True)
    atomic_json(rank_root / "complete.json", {
        "schema_version": "closed_loop_replay_rank_complete_v1",
        "contract_sha256": contract["contract_sha256"],
        "rank": rank,
        "expected_uids": len(schedule),
        "completed_uids": completed,
        "completed_at": utc_now(),
    })


def aggregate_replay(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    schedule = read_jsonl(output_root / "work/replay_manifest.jsonl")
    expected_uids = [str(row["uid"]) for row in schedule]
    uid_results: list[dict[str, Any]] = []
    for rank in range(int(config["world_size"])):
        rank_root = output_root / f"work/replay/rank{rank:02d}"
        completion = read_json(rank_root / "complete.json")
        expected_rank = sum(int(row["worker_rank"]) == rank for row in schedule)
        if (
            completion.get("contract_sha256") != contract["contract_sha256"]
            or int(completion.get("expected_uids", -1)) != expected_rank
            or int(completion.get("completed_uids", -1)) != expected_rank
        ):
            raise RuntimeError(f"replay rank {rank} is incomplete")
        failures = list(rank_root.glob("failure_*.json"))
        if failures:
            raise RuntimeError(f"replay rank {rank} contains quarantined failures")
        uid_results.extend(
            read_json(path)
            for path in sorted(rank_root.glob("[0-9a-f]*.json"))
            if path.name != "complete.json"
        )
    observed = Counter(str(row["uid"]) for row in uid_results)
    if Counter(expected_uids) != observed or any(value != 1 for value in observed.values()):
        raise RuntimeError("global routed-state replay UID completion differs")
    by_uid = {str(row["uid"]): row for row in uid_results}
    uid_results = [by_uid[uid] for uid in expected_uids]
    replay_rows = [item for row in uid_results for item in row["program_replays"]]
    state_rows = [item for row in uid_results for item in row["state_rows"]]
    program_to_states = {str(row["program_id"]): list(row["state_ids"]) for row in replay_rows}
    coverage = validate_state_references(program_to_states, state_rows)
    if (
        len(replay_rows) != int(config["corpus"]["expected_programs"])
        or len(state_rows) != int(config["corpus"]["expected_unique_prefix_states"])
        or not all(row["passed"] for row in replay_rows)
        or not all(row["compact_full_router_logits_exact"] for row in uid_results)
    ):
        raise RuntimeError("full routed-state replay/cache gate failed")
    for row in uid_results:
        path = output_root / row["state_file"]
        if file_sha256(path) != row["state_file_sha256"]:
            raise RuntimeError(f"routed-state cache file changed: {row['uid']}")
    atomic_jsonl(output_root / "corpus/replay_validation.jsonl", replay_rows)
    atomic_jsonl(output_root / "corpus/routed_state_cache_manifest.jsonl", state_rows)
    atomic_jsonl(
        output_root / "corpus/uid_state_files.jsonl",
        (
            {
                "contract_sha256": contract["contract_sha256"],
                "uid": row["uid"],
                "state_file": row["state_file"],
                "state_file_sha256": row["state_file_sha256"],
                "program_count": row["program_count"],
                "state_count": row["state_count"],
            }
            for row in uid_results
        ),
    )
    atomic_json(output_root / "work/replay_completion.json", {
        "schema_version": "closed_loop_replay_completion_v1",
        "contract_sha256": contract["contract_sha256"],
        "uids": len(uid_results),
        "programs": len(replay_rows),
        "unique_states": len(state_rows),
        "state_references": coverage["state_references"],
        "all_exact_token_parity": True,
        "all_final_correct": True,
        "all_parent_trigger_state_exact": True,
        "all_compact_full_router_logits_exact": True,
        "quarantined_programs": 0,
        "replay_validation_sha256": file_sha256(output_root / "corpus/replay_validation.jsonl"),
        "state_manifest_sha256": file_sha256(output_root / "corpus/routed_state_cache_manifest.jsonl"),
        "completed_at": utc_now(),
    })
    print(json.dumps({"uids": len(uid_results), "programs": len(replay_rows), **coverage}, sort_keys=True))


def _verify_replay_completion(
    contract: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    completion = read_json(output_root / "work/replay_completion.json")
    expected = contract["population"]
    if (
        completion.get("contract_sha256") != contract["contract_sha256"]
        or int(completion.get("uids", -1)) != int(expected["uids"])
        or int(completion.get("programs", -1)) != int(expected["programs"])
        or int(completion.get("unique_states", -1)) != int(expected["unique_prefix_states"])
        or not completion.get("all_exact_token_parity")
        or not completion.get("all_final_correct")
        or not completion.get("all_parent_trigger_state_exact")
        or not completion.get("all_compact_full_router_logits_exact")
        or int(completion.get("quarantined_programs", -1)) != 0
    ):
        raise RuntimeError("the full routed-state replay/cache gate has not passed")
    if file_sha256(output_root / "corpus/replay_validation.jsonl") != completion["replay_validation_sha256"]:
        raise RuntimeError("replay validation changed after aggregation")
    if file_sha256(output_root / "corpus/routed_state_cache_manifest.jsonl") != completion["state_manifest_sha256"]:
        raise RuntimeError("state-cache manifest changed after aggregation")
    return completion


def _uid_file_index(
    contract: Mapping[str, Any], output_root: Path
) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(output_root / "corpus/uid_state_files.jsonl")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        uid = str(row["uid"])
        if uid in output or row.get("contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError("UID state-file index is duplicated or has wrong provenance")
        path = output_root / row["state_file"]
        if file_sha256(path) != row["state_file_sha256"]:
            raise RuntimeError(f"UID state cache changed: {uid}")
        output[uid] = row
    if len(output) != int(contract["population"]["uids"]):
        raise RuntimeError("UID state-file index is incomplete")
    return output


def _load_uid_payload(
    contract: Mapping[str, Any], output_root: Path, row: Mapping[str, Any]
) -> dict[str, Any]:
    path = output_root / str(row["state_file"])
    if file_sha256(path) != str(row["state_file_sha256"]):
        raise RuntimeError(f"UID state cache hash differs: {row['uid']}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("contract_sha256") != contract["contract_sha256"] or payload.get("uid") != row["uid"]:
        raise RuntimeError(f"UID state cache provenance differs: {row['uid']}")
    if len(payload["states"]) != int(row["state_count"]) or len(payload["programs"]) != int(row["program_count"]):
        raise RuntimeError(f"UID state cache cardinality differs: {row['uid']}")
    return payload


def _fixed_rank_uids(
    uids: Sequence[str], *, rank: int, world_size: int, seed: int, epoch: int
) -> tuple[list[str | None], int]:
    partitions = [[] for _ in range(world_size)]
    for uid in sorted(map(str, uids)):
        owner = int(sha256(uid.encode()).hexdigest(), 16) % world_size
        partitions[owner].append(uid)
    rng = random.Random((int(seed) << 12) + int(epoch))
    for index, values in enumerate(partitions):
        random.Random(rng.randrange(2**63) + index).shuffle(values)
    steps = max(map(len, partitions))
    current: list[str | None] = list(partitions[rank]) + [None] * (steps - len(partitions[rank]))
    random.Random((int(seed) << 16) + int(epoch) + rank).shuffle(current)
    return current, steps


def _allreduce_gradients(model: torch.nn.Module, world_size: int) -> None:
    for parameter in model.parameters():
        if parameter.grad is None:
            raise RuntimeError("router parameter is missing a gradient")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def _evaluate_uid_set(
    router: SharedReadWriteRouter,
    uids: Sequence[str],
    *,
    rank: int,
    world_size: int,
    contract: Mapping[str, Any],
    output_root: Path,
    file_index: Mapping[str, Mapping[str, Any]],
    uid_weights: Mapping[str, float],
    device: torch.device,
    state_microbatch: int,
) -> tuple[float, list[dict[str, Any]]]:
    router.eval()
    local_weighted_loss = 0.0
    local_weight = 0.0
    diagnostics = []
    with torch.no_grad():
        for uid in sorted(uids):
            if int(sha256(uid.encode()).hexdigest(), 16) % world_size != rank:
                continue
            payload = _load_uid_payload(contract, output_root, file_index[uid])
            loss, stats = compute_uid_trajectory_loss(
                router,
                payload,
                device=device,
                state_microbatch=state_microbatch,
            )
            weight = float(uid_weights[uid])
            local_weighted_loss += weight * float(loss.item())
            local_weight += weight
            diagnostics.append({
                "uid": uid,
                "dataset": payload["dataset"],
                "source_regime": payload["source_regime"],
                "dense_outcome": payload["dense_outcome"],
                "trigger_layer": payload["trigger_layer"],
                **stats,
            })
            del payload, loss
    totals = torch.tensor([local_weighted_loss, local_weight], dtype=torch.float64, device=device)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if float(totals[1]) <= 0:
        raise RuntimeError("development objective has zero weight")
    return float((totals[0] / totals[1]).item()), diagnostics


def _write_training_completion(
    root: Path,
    *,
    contract_sha256: str,
    rank: int,
    mode: str,
    epochs: int,
    uids: int,
) -> None:
    atomic_json(root / f"rank{rank:02d}.complete.json", {
        "schema_version": "closed_loop_training_rank_complete_v1",
        "contract_sha256": contract_sha256,
        "rank": rank,
        "mode": mode,
        "epochs": epochs,
        "uids": uids,
        "completed_at": utc_now(),
    })


def train(config_path: Path, mode: str) -> None:
    if mode not in {"internal", "full"}:
        raise ValueError("training mode must be internal or full")
    contract, output_root = verify_contract(config_path)
    _verify_replay_completion(contract, output_root)
    config = contract["static_config"]
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != int(config["world_size"]):
        raise RuntimeError("training must use exactly four distributed ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    configure_dense_determinism(int(config["seed"]) + rank, config["backend_settings"])
    dist.init_process_group(backend="nccl")
    split = read_json(output_root / "splits/internal_dev_groups.json")["uid_split"]
    all_uids = sorted(split)
    train_uids = [uid for uid in all_uids if split[uid] == "train"]
    dev_uids = [uid for uid in all_uids if split[uid] == "dev"]
    active_uids = train_uids if mode == "internal" else all_uids
    epochs = (
        int(config["training"]["maximum_epochs"])
        if mode == "internal"
        else int(read_json(output_root / "training/frozen_epoch_selection.json")["selected_epoch"])
    )
    root = output_root / f"work/training/{mode}"
    completion_path = root / f"rank{rank:02d}.complete.json"
    if completion_path.exists():
        raise RuntimeError(f"refusing to overwrite completed {mode} training rank {rank}")
    file_index = _uid_file_index(contract, output_root)
    active_programs = [
        row for row in read_jsonl(output_root / "corpus/trajectory_set_manifest.jsonl")
        if row["uid"] in set(active_uids)
    ]
    active_grouped, _ = build_trajectory_sets(active_programs, total_layers=28)
    active_weights = hierarchical_uid_weights(active_grouped)
    dev_programs = [
        row for row in read_jsonl(output_root / "corpus/trajectory_set_manifest.jsonl")
        if row["uid"] in set(dev_uids)
    ]
    dev_grouped, _ = build_trajectory_sets(dev_programs, total_layers=28)
    dev_weights = hierarchical_uid_weights(dev_grouped)
    router = _router(config, device)
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    microbatch = int(config["training"]["state_forward_microbatch"])
    payload_cache: dict[str, dict[str, Any]] = {}
    best_epoch = 0
    best_dev = float("inf")
    best_state = None
    train_log = output_root / (
        "training/internal_dev_train_log.jsonl" if mode == "internal" else "training/full_refit_train_log.jsonl"
    )
    if rank == 0 and train_log.exists():
        raise RuntimeError(f"refusing to overwrite training log: {train_log}")
    dist.barrier()
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        router.train()
        assignments, steps = _fixed_rank_uids(
            active_uids,
            rank=rank,
            world_size=world_size,
            seed=int(config["seed"]),
            epoch=epoch,
        )
        local = torch.zeros(5, dtype=torch.float64, device=device)
        responsibility_rows = []
        fallback_uid = next(uid for uid in assignments if uid is not None)
        for uid in assignments:
            actual_uid = fallback_uid if uid is None else uid
            if actual_uid not in payload_cache:
                payload_cache[actual_uid] = _load_uid_payload(
                    contract, output_root, file_index[actual_uid]
                )
            payload = payload_cache[actual_uid]
            optimizer.zero_grad(set_to_none=True)
            loss, stats = compute_uid_trajectory_loss(
                router,
                payload,
                device=device,
                state_microbatch=microbatch,
            )
            scale = 0.0 if uid is None else float(active_weights[actual_uid]) * len(active_uids)
            scaled = loss * scale
            scaled.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in router.parameters()
            ):
                raise RuntimeError(f"non-finite router gradient for {actual_uid}")
            _allreduce_gradients(router, world_size)
            torch.nn.utils.clip_grad_norm_(
                router.parameters(),
                float(config["training"]["gradient_clip_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in router.parameters()):
                raise RuntimeError(f"non-finite router parameter after {actual_uid}")
            if uid is not None:
                weight = float(active_weights[actual_uid])
                local[0] += weight * float(loss.detach().item())
                local[1] += weight
                local[2] += weight * float(stats["responsibility_entropy"])
                local[3] += weight * float(stats["top_responsibility"])
                local[4] += 1
                if epoch == epochs:
                    responsibility_rows.append({
                        "contract_sha256": contract["contract_sha256"],
                        "mode": mode,
                        "epoch": epoch,
                        "uid": actual_uid,
                        "dataset": payload["dataset"],
                        "source_regime": payload["source_regime"],
                        "dense_outcome": payload["dense_outcome"],
                        "trigger_layer": payload["trigger_layer"],
                        **stats,
                    })
            del loss, scaled
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        train_loss = float((local[0] / local[1]).item())
        train_entropy = float((local[2] / local[1]).item())
        train_top = float((local[3] / local[1]).item())
        if mode == "internal":
            dev_loss, _ = _evaluate_uid_set(
                router,
                dev_uids,
                rank=rank,
                world_size=world_size,
                contract=contract,
                output_root=output_root,
                file_index=file_index,
                uid_weights=dev_weights,
                device=device,
                state_microbatch=microbatch,
            )
        else:
            dev_loss = None
        epoch_row = {
            "schema_version": "closed_loop_trajectory_set_training_epoch_v1",
            "contract_sha256": contract["contract_sha256"],
            "mode": mode,
            "epoch": epoch,
            "epochs": epochs,
            "active_uids": len(active_uids),
            "optimizer_steps": steps,
            "hierarchical_uid_weighted_train_loss": train_loss,
            "hierarchical_uid_weighted_dev_loss": dev_loss,
            "responsibility_entropy": train_entropy,
            "top_responsibility_share": train_top,
            "elapsed_seconds": time.monotonic() - started,
        }
        if rank == 0:
            append_jsonl(train_log, epoch_row)
            print(json.dumps(epoch_row, sort_keys=True), flush=True)
            if mode == "internal" and float(dev_loss) < best_dev:
                best_dev = float(dev_loss)
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in router.state_dict().items()
                }
        if epoch == epochs:
            atomic_jsonl(root / f"rank{rank:02d}_responsibility.jsonl", responsibility_rows)

    if mode == "full":
        _full_eval_loss, responsibility_rows = _evaluate_uid_set(
            router,
            active_uids,
            rank=rank,
            world_size=world_size,
            contract=contract,
            output_root=output_root,
            file_index=file_index,
            uid_weights=active_weights,
            device=device,
            state_microbatch=microbatch,
        )
        atomic_jsonl(root / f"rank{rank:02d}_responsibility.jsonl", responsibility_rows)

    dist.barrier()
    if rank == 0:
        if mode == "internal":
            if best_state is None or best_epoch < 1:
                raise RuntimeError("internal development did not select an epoch")
            checkpoint = output_root / "training/internal_dev_selected_checkpoint.pt"
            atomic_torch(checkpoint, {
                "schema_version": "closed_loop_trajectory_set_checkpoint_v1",
                "contract_sha256": contract["contract_sha256"],
                "mode": mode,
                "selected_epoch": best_epoch,
                "state_dict": best_state,
            })
            selection = {
                "schema_version": "closed_loop_frozen_epoch_selection_v1",
                "contract_sha256": contract["contract_sha256"],
                "selection_rule": config["training"]["epoch_selection"],
                "selected_epoch": best_epoch,
                "selected_dev_loss": best_dev,
                "maximum_epochs": epochs,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "selected_at": utc_now(),
            }
            atomic_json(output_root / "training/frozen_epoch_selection.json", selection)
            _atomic_bytes(output_root / "training/internal_dev_summary.md", (
                "# Internal group-disjoint schedule selection\n\n"
                f"- Train/dev UIDs: {len(train_uids)}/{len(dev_uids)}\n"
                f"- Selected epoch: **{best_epoch}** of {epochs}\n"
                f"- Selected hierarchical UID-weighted dev loss: {best_dev:.8f}\n"
                "- Image-group overlap: **0**\n"
                "- External benchmarks were not used for selection.\n"
            ).encode())
        else:
            checkpoint = output_root / "training/full_refit_checkpoint.pt"
            atomic_torch(checkpoint, {
                "schema_version": "closed_loop_trajectory_set_checkpoint_v1",
                "contract_sha256": contract["contract_sha256"],
                "mode": mode,
                "selected_epoch": epochs,
                "state_dict": {key: value.detach().cpu() for key, value in router.state_dict().items()},
            })
            manifest = {
                "schema_version": "closed_loop_full_refit_checkpoint_manifest_v1",
                "contract_sha256": contract["contract_sha256"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "selected_epoch": epochs,
                "uids": len(active_uids),
                "programs": int(config["corpus"]["expected_programs"]),
                "completed_at": utc_now(),
            }
            atomic_json(output_root / "training/full_refit_checkpoint_manifest.json", manifest)
    _write_training_completion(
        root,
        contract_sha256=contract["contract_sha256"],
        rank=rank,
        mode=mode,
        epochs=epochs,
        uids=len(active_uids),
    )
    dist.barrier()
    if rank == 0 and mode == "full":
        rows = []
        for current_rank in range(world_size):
            rows.extend(read_jsonl(root / f"rank{current_rank:02d}_responsibility.jsonl"))
        if len(rows) != len(active_uids) or len({row["uid"] for row in rows}) != len(active_uids):
            raise RuntimeError("full-refit responsibility diagnostics are incomplete")
        atomic_csv(output_root / "training/responsibility_statistics.csv", [
            {
                "uid": row["uid"],
                "dataset": row["dataset"],
                "source_regime": row["source_regime"],
                "dense_outcome": row["dense_outcome"],
                "trigger_layer": row["trigger_layer"],
                "route_count": row["route_count"],
                "state_count": row["state_count"],
                "responsibility_entropy": row["responsibility_entropy"],
                "top_responsibility": row["top_responsibility"],
                "mean_route_logp": row["mean_route_logp"],
                "best_route_logp": row["best_route_logp"],
            }
            for row in rows
        ])
        likelihood = []
        responsibility = []
        for row in rows:
            for program_id, route_logp, share in zip(
                row["program_ids"], row["route_logps"], row["responsibilities"]
            ):
                likelihood.append({
                    "uid": row["uid"],
                    "program_id": program_id,
                    "dense_outcome": row["dense_outcome"],
                    "route_logp": route_logp,
                    "length_normalized_route_logp": route_logp / (28 - int(row["trigger_layer"])),
                })
                responsibility.append({
                    "uid": row["uid"],
                    "program_id": program_id,
                    "responsibility": share,
                    "dense_outcome": row["dense_outcome"],
                })
        atomic_csv(output_root / "diagnostics/internal_trajectory_likelihood.csv", likelihood)
        atomic_csv(output_root / "diagnostics/trajectory_responsibility.csv", responsibility)
        _atomic_bytes(output_root / "training/initialization_report.md", (
            "# Initialization report\n\n"
            "- Router architecture: existing SharedReadWriteRouter.\n"
            "- Initialization: exact strict load of the frozen Sequential-A checkpoint.\n"
            "- Trainable: READ branch, WRITE branch, shared action head.\n"
            "- Frozen/not loaded during cache training: Qwen backbone and Stage-1.\n"
        ).encode())
        atomic_json(output_root / "training/model_config.json", config["stage2"])
        router_parameters = sum(parameter.numel() for parameter in router.parameters())
        atomic_json(output_root / "training/parameter_count.json", {
            "trainable_router_parameters": router_parameters,
            "frozen_backbone_parameters_during_training": "not_loaded",
            "stage1_trainable_parameters": 0,
        })
    dist.barrier()
    dist.destroy_process_group()


class ClosedLoopEvalRuntime:
    def __init__(self, config: Mapping[str, Any], device_index: int):
        self.phase76_config = config
        self.eval_config = read_json(config["evaluation"]["base_config"])
        self.base = DenseEvalRuntime(self.eval_config, int(device_index))
        self.device = self.base.device
        self.router = _router(config, self.device)
        manifest = read_json(resolve_path(config["output_root"]) / "training/full_refit_checkpoint_manifest.json")
        checkpoint_path = resolve_path(manifest["checkpoint"])
        if file_sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
            raise RuntimeError("full-refit checkpoint hash differs")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("contract_sha256") != manifest["contract_sha256"]:
            raise RuntimeError("full-refit checkpoint contract differs")
        self.router.load_state_dict(checkpoint["state_dict"], strict=True)
        self.router.eval()
        self.checkpoint_sha256 = manifest["checkpoint_sha256"]


def prepare_evaluation(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    checkpoint = read_json(output_root / "training/full_refit_checkpoint_manifest.json")
    if checkpoint.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("full-refit checkpoint belongs to another contract")
    if file_sha256(checkpoint["checkpoint"]) != checkpoint["checkpoint_sha256"]:
        raise RuntimeError("full-refit checkpoint hash differs")
    for rank in range(int(config["world_size"])):
        completion = read_json(output_root / f"work/training/full/rank{rank:02d}.complete.json")
        if completion.get("contract_sha256") != contract["contract_sha256"] or completion.get("mode") != "full":
            raise RuntimeError(f"full-refit rank {rank} is incomplete")
    phase69 = read_json(config["evaluation"]["phase69_contract"])
    phase74 = read_json(config["evaluation"]["phase74_contract"])
    if phase69.get("contract_sha256") != contract["parent_contracts"]["phase69"]:
        raise RuntimeError("Phase-69 evaluation contract changed")
    if phase74.get("contract_sha256") != contract["parent_contracts"]["phase74"]:
        raise RuntimeError("Phase-74 evaluation contract changed")
    source_manifest = resolve_path(
        "analysis/dense_failure_stage2/full_benchmark_eval/manifests/all_full_manifest.jsonl"
    )
    expected_hash = phase69["prepared_manifest_sha256"]["manifests/all_full_manifest.jsonl"]
    if file_sha256(source_manifest) != expected_hash:
        raise RuntimeError("Phase-69 evaluation manifest differs")
    rows = read_jsonl(source_manifest)
    phase69_rows = read_jsonl(config["evaluation"]["phase69_paired"])
    phase74_rows = read_jsonl(config["evaluation"]["phase74_paired"])
    expected_total = int(config["evaluation"]["expected_total"])
    populations = [{str(row["uid"]) for row in current} for current in (rows, phase69_rows, phase74_rows)]
    if any(len(current) != expected_total for current in (rows, phase69_rows, phase74_rows)) or not (populations[0] == populations[1] == populations[2]):
        raise RuntimeError("evaluation baseline populations differ")
    schedule = []
    for index, row in enumerate(rows):
        copy = dict(row)
        copy["worker_rank"] = index % int(config["world_size"])
        schedule.append(copy)
    atomic_jsonl(output_root / "work/evaluation_manifest.jsonl", schedule)
    frozen = {
        "schema_version": "closed_loop_full_evaluation_input_v1",
        "contract_sha256": contract["contract_sha256"],
        "phase69_contract_sha256": phase69["contract_sha256"],
        "phase74_contract_sha256": phase74["contract_sha256"],
        "phase69_manifest_sha256": expected_hash,
        "phase69_paired_sha256": file_sha256(config["evaluation"]["phase69_paired"]),
        "phase74_paired_sha256": file_sha256(config["evaluation"]["phase74_paired"]),
        "evaluation_manifest_sha256": file_sha256(output_root / "work/evaluation_manifest.jsonl"),
        "full_refit_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "rows": len(rows),
        "prepared_at": utc_now(),
    }
    atomic_json(output_root / "work/evaluation_input.json", frozen)
    print(json.dumps(frozen, sort_keys=True))


@torch.inference_mode()
def _process_eval_row(
    runtime: ClosedLoopEvalRuntime,
    row: Mapping[str, Any],
    phase69_row: Mapping[str, Any],
    phase74_row: Mapping[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    started = time.monotonic()
    dense = _native_dense_generation(runtime.base, row, extract_features=True)
    features = dense.pop("features")
    scores, trigger = _stage1_scores(runtime.base, features)
    del features
    phase69_parity = {
        "dense_generated_token_ids": dense["generated_token_ids"] == phase69_row["dense_generated_token_ids"],
        "dense_score": float(dense["score"]) == float(phase69_row["dense_score"]),
        "dense_correct": bool(dense["correct"]) == bool(phase69_row["dense_correct"]),
        "stage1_scores": scores == [float(value) for value in phase69_row["stage1_scores"]],
        "trigger_layer": trigger == phase69_row["trigger_layer"],
    }
    phase74_parity = {
        "dense_generated_token_ids": dense["generated_token_ids"] == phase74_row["dense_generated_token_ids"],
        "dense_score": float(dense["score"]) == float(phase74_row["dense_score"]),
        "dense_correct": bool(dense["correct"]) == bool(phase74_row["dense_correct"]),
        "stage1_scores": scores == [float(value) for value in phase74_row["stage1_scores"]],
        "trigger_layer": trigger == phase74_row["trigger_layer"],
    }
    if not all(phase69_parity.values()) or not all(phase74_parity.values()):
        raise RuntimeError(f"Dense/Stage-1 baseline parity failed for {row['uid']}")

    trace: list[dict[str, Any]] = []
    if trigger is None:
        actions = ["FULL"] * 28
        routed = {key: dense[key] for key in (
            "generated_token_ids", "generated_answer", "metric", "score", "threshold", "correct"
        )}
        feedback_exact = True
    else:
        consumed = _verify_images(row)
        if consumed != dense["consumed_image_sha256s"]:
            raise RuntimeError("image bytes changed between dense and closed-loop inference")
        inputs = eval_build_inputs(runtime.base, row)
        prepared = build_binary_inputs(runtime.base.wrapped, inputs)

        def selector(layer_index, text_state, visual_state, current_meta):
            state_hash = _combined_state_hash({
                "text_states": tensor_sha256(text_state),
                "visual_states": tensor_sha256(visual_state),
            })
            if layer_index < int(trigger):
                action = "FULL"
                logits = None
                probabilities = None
            else:
                current_logits = runtime.router(
                    text_state,
                    visual_state,
                    text_mask=current_meta.text_valid_mask,
                    visual_mask=current_meta.visual_valid_mask,
                )[0]
                if not torch.isfinite(current_logits).all():
                    raise RuntimeError(f"non-finite closed-loop logits: {row['uid']} L{layer_index}")
                current_probabilities = current_logits.float().softmax(dim=-1)
                action = ACTION_NAMES[int(current_probabilities.argmax().item())]
                logits = [float(value) for value in current_logits.detach().cpu().tolist()]
                probabilities = {
                    name: float(current_probabilities[index].item())
                    for index, name in enumerate(ACTION_NAMES)
                }
            trace.append({
                "layer": int(layer_index),
                "active": layer_index >= int(trigger),
                "action": action,
                "state_sha256": state_hash,
                "logits": logits,
                "probabilities": probabilities,
            })
            return action

        output = capture_online_four_action_route(
            runtime.base.wrapped,
            {},
            selector,
            prepared_inputs=prepared,
            use_cache=True,
            native_full_rows=True,
        )
        feedback_exact = True
        for item in trace:
            text_state, visual_state = output.pre_layer_states[item["layer"]]
            observed = _combined_state_hash({
                "text_states": tensor_sha256(text_state),
                "visual_states": tensor_sha256(visual_state),
            })
            feedback_exact = feedback_exact and observed == item["state_sha256"]
        if not feedback_exact:
            raise RuntimeError(f"online selector did not receive actual routed states: {row['uid']}")
        final_hash = _combined_state_hash({
            "text_states": tensor_sha256(output.text_hidden_state),
            "visual_states": tensor_sha256(output.visual_hidden_state),
        })
        for index, item in enumerate(trace):
            item["previous_action"] = None if index == 0 else trace[index - 1]["action"]
            item["next_state_sha256"] = trace[index + 1]["state_sha256"] if index + 1 < len(trace) else final_hash
            item["next_layer_logits"] = trace[index + 1]["logits"] if index + 1 < len(trace) else None
        actions = [item["action"] for item in trace]
        active_actions = actions[int(trigger):]
        if all(action == "FULL" for action in active_actions):
            routed = {key: dense[key] for key in (
                "generated_token_ids", "generated_answer", "metric", "score", "threshold", "correct"
            )}
        else:
            routed = eval_generate(runtime.base, output, inputs["input_ids"], row)
        del output, prepared, inputs

    active = actions[int(trigger):] if trigger is not None else []
    nonfull = [index for index, action in enumerate(actions) if trigger is not None and index >= int(trigger) and action != "FULL"]
    switches = sum(left != right for left, right in zip(active, active[1:]))
    dense_correct = bool(dense["correct"])
    closed_correct = bool(routed["correct"])
    return {
        "schema_version": "closed_loop_full_eval_paired_row_v1",
        "contract_sha256": contract_sha256,
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "uid": str(row["uid"]),
        "sample_id": str(row["sample_id"]),
        "benchmark": str(row["benchmark"]),
        "benchmark_family": str(row["benchmark_family"]),
        "image_group_id": str(row["image_group_id"]),
        "metric_name": str(row["metric_name"]),
        "correctness_threshold": float(row["correctness_threshold"]),
        "answer": row["answer"],
        "all_answer_norms": row.get("all_answer_norms"),
        "dense_generated_answer": dense["generated_answer"],
        "dense_generated_token_ids": dense["generated_token_ids"],
        "dense_score": dense["score"],
        "dense_correct": dense_correct,
        "sequential_generated_answer": phase69_row["routed_generated_answer"],
        "sequential_generated_token_ids": phase69_row["routed_generated_token_ids"],
        "sequential_score": phase69_row["routed_score"],
        "sequential_correct": bool(phase69_row["routed_correct"]),
        "sequential_actions": phase69_row["actions"],
        "program_generated_answer": phase74_row["program_generated_answer"],
        "program_generated_token_ids": phase74_row["program_generated_token_ids"],
        "program_score": phase74_row["program_score"],
        "program_correct": bool(phase74_row["program_correct"]),
        "program_actions": phase74_row["program_actions"],
        "closed_loop_generated_answer": routed["generated_answer"],
        "closed_loop_generated_token_ids": routed["generated_token_ids"],
        "closed_loop_score": routed["score"],
        "closed_loop_correct": closed_correct,
        "closed_loop_transition": ("C" if dense_correct else "W") + "→" + ("C" if closed_correct else "W"),
        "sequential_transition": phase69_row["transition"],
        "program_transition": phase74_row["program_transition"],
        "stage1_scores": scores,
        "stage1_threshold": runtime.phase76_config["stage1"]["threshold"],
        "triggered": trigger is not None,
        "trigger_layer": trigger,
        "closed_loop_actions": actions,
        "closed_loop_suffix_actions": active,
        "action_rows": trace,
        "post_trigger_action_counts": dict(Counter(active)),
        "non_full_count": len(nonfull),
        "any_non_full": bool(nonfull),
        "first_non_full_layer": nonfull[0] if nonfull else None,
        "trigger_to_first_non_full_delay": nonfull[0] - int(trigger) if nonfull else None,
        "read_only_count": active.count("READ_ONLY"),
        "write_only_count": active.count("WRITE_ONLY"),
        "ignore_count": active.count("IGNORE"),
        "action_switches": switches,
        "phase69_reuse_parity": phase69_parity,
        "phase74_reuse_parity": phase74_parity,
        "state_feedback_exact": feedback_exact,
        "consumed_image_sha256s": dense["consumed_image_sha256s"],
        "prompt_tokens": dense["prompt_tokens"],
        "visual_tokens": dense["visual_tokens"],
        "elapsed_seconds": time.monotonic() - started,
    }


def evaluation_worker(config_path: Path, rank: int, resume: bool) -> None:
    contract, output_root = verify_contract(config_path, verify_model=True)
    config = contract["static_config"]
    frozen = read_json(output_root / "work/evaluation_input.json")
    if frozen.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("full evaluation input is not frozen")
    if file_sha256(output_root / "work/evaluation_manifest.jsonl") != frozen["evaluation_manifest_sha256"]:
        raise RuntimeError("full evaluation manifest changed after freeze")
    rank = int(rank)
    rows = [
        row for row in read_jsonl(output_root / "work/evaluation_manifest.jsonl")
        if int(row["worker_rank"]) == rank
    ]
    phase69 = {str(row["uid"]): row for row in read_jsonl(config["evaluation"]["phase69_paired"])}
    phase74 = {str(row["uid"]): row for row in read_jsonl(config["evaluation"]["phase74_paired"])}
    runtime = ClosedLoopEvalRuntime(config, rank)
    rank_root = output_root / f"work/evaluation/rank{rank:02d}"
    completed = 0
    chunk_size = 16
    started = time.monotonic()
    for part_index, start in enumerate(range(0, len(rows), chunk_size)):
        current = rows[start : start + chunk_size]
        path = rank_root / f"part_{part_index:05d}.jsonl"
        if resume and path.exists():
            old = read_jsonl(path)
            if (
                len(old) == len(current)
                and [row["uid"] for row in old] == [row["uid"] for row in current]
                and all(row.get("contract_sha256") == contract["contract_sha256"] for row in old)
                and all(row.get("checkpoint_sha256") == runtime.checkpoint_sha256 for row in old)
            ):
                completed += len(old)
                continue
        results = [
            _process_eval_row(
                runtime,
                row,
                phase69[str(row["uid"])],
                phase74[str(row["uid"])],
                contract["contract_sha256"],
            )
            for row in current
        ]
        atomic_jsonl(path, results)
        completed += len(results)
        if part_index % 10 == 0 or completed == len(rows):
            print(json.dumps({
                "rank": rank,
                "completed": completed,
                "total": len(rows),
                "elapsed_seconds": time.monotonic() - started,
            }), flush=True)
    atomic_json(rank_root / "complete.json", {
        "schema_version": "closed_loop_eval_rank_complete_v1",
        "contract_sha256": contract["contract_sha256"],
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "rank": rank,
        "expected_rows": len(rows),
        "completed_rows": completed,
        "completed_at": utc_now(),
    })


def _scopes(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    return {
        "overall": list(rows),
        **{
            family: [row for row in rows if row["benchmark_family"] == family]
            for family in ("chartqa", "textvqa", "mmmu_pro", "pope")
        },
    }


def _method_summary(rows: Sequence[Mapping[str, Any]], correct_key: str) -> dict[str, Any]:
    transitions = Counter(
        ("C" if row["dense_correct"] else "W")
        + "→"
        + ("C" if row[correct_key] else "W")
        for row in rows
    )
    n = len(rows)
    dense_correct = sum(bool(row["dense_correct"]) for row in rows)
    method_correct = sum(bool(row[correct_key]) for row in rows)
    return {
        "n": n,
        "dense_accuracy": dense_correct / n,
        "accuracy": method_correct / n,
        "w_to_c": transitions["W→C"],
        "c_to_w": transitions["C→W"],
        "c_to_c": transitions["C→C"],
        "w_to_w": transitions["W→W"],
        "net": transitions["W→C"] - transitions["C→W"],
    }


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _mean(values: Sequence[int | float]) -> float | None:
    return float(np.mean(values)) if values else None


def _write_figures(
    output_root: Path,
    scopes: Mapping[str, Sequence[Mapping[str, Any]]],
    responsibility_rows: Sequence[Mapping[str, Any]],
) -> None:
    families = ["chartqa", "textvqa", "mmmu_pro", "pope", "overall"]
    labels = ["ChartQA", "TextVQA", "MMMU-Pro", "POPE", "Overall"]
    methods = [
        ("Dense", "dense_correct"),
        ("Sequential-A", "sequential_correct"),
        ("Open-loop", "program_correct"),
        ("Closed-loop", "closed_loop_correct"),
    ]
    x = np.arange(len(families))
    width = 0.2
    figure, axis = plt.subplots(figsize=(11, 5))
    for index, (label, key) in enumerate(methods):
        values = [np.mean([bool(row[key]) for row in scopes[family]]) for family in families]
        axis.bar(x + (index - 1.5) * width, values, width, label=label)
    axis.set_xticks(x, labels, rotation=20)
    axis.set_ylabel("LMMS correctness")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "figures/benchmark_accuracy_comparison.png", dpi=160)
    plt.close(figure)

    w2c = [sum(row["closed_loop_transition"] == "W→C" for row in scopes[key]) for key in families]
    c2w = [sum(row["closed_loop_transition"] == "C→W" for row in scopes[key]) for key in families]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x - 0.18, w2c, 0.36, label="W→C")
    axis.bar(x + 0.18, c2w, 0.36, label="C→W")
    axis.set_xticks(x, labels, rotation=20)
    axis.set_ylabel("Samples")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "figures/rescue_regression_comparison.png", dpi=160)
    plt.close(figure)

    triggered = [sum(row["triggered"] for row in scopes[key]) for key in families]
    nonfull = [sum(row["any_non_full"] for row in scopes[key]) for key in families]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x - 0.18, triggered, 0.36, label="Stage-1 triggered")
    axis.bar(x + 0.18, nonfull, 0.36, label="Closed-loop non-FULL")
    axis.set_xticks(x, labels, rotation=20)
    axis.set_ylabel("Samples")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "figures/stage1_stage2_funnel.png", dpi=160)
    plt.close(figure)

    action_rows = [row for row in scopes["overall"] if row["triggered"]]
    action_counts = {
        family: Counter(
            action
            for row in scopes[family]
            if row["triggered"]
            for action in row["closed_loop_suffix_actions"]
        )
        for family in families[:-1]
    }
    figure, axis = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(4)
    for action in ACTION_NAMES:
        values = [action_counts[family][action] for family in families[:-1]]
        axis.bar(np.arange(4), values, bottom=bottom, label=action)
        bottom += values
    axis.set_xticks(np.arange(4), labels[:-1], rotation=20)
    axis.set_ylabel("Post-trigger actions")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "figures/action_usage_by_benchmark.png", dpi=160)
    plt.close(figure)

    delays = [row["trigger_to_first_non_full_delay"] for row in action_rows if row["trigger_to_first_non_full_delay"] is not None]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(delays, bins=range(0, 29))
    axis.set_xlabel("Trigger-to-first-intervention delay")
    axis.set_ylabel("Triggered samples")
    figure.tight_layout()
    figure.savefig(output_root / "figures/trigger_to_first_intervention.png", dpi=160)
    plt.close(figure)

    entropies = [float(row["responsibility_entropy"]) for row in responsibility_rows]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(entropies, bins=30)
    axis.set_xlabel("Trajectory responsibility entropy")
    axis.set_ylabel("Training UIDs")
    figure.tight_layout()
    figure.savefig(output_root / "figures/trajectory_responsibility_entropy.png", dpi=160)
    plt.close(figure)

    dense = _method_summary(scopes["overall"], "dense_correct")
    seq = _method_summary(scopes["overall"], "sequential_correct")
    program = _method_summary(scopes["overall"], "program_correct")
    closed = _method_summary(scopes["overall"], "closed_loop_correct")
    figure, axis = plt.subplots(figsize=(8, 4))
    names = ["Dense", "Sequential-A", "Open-loop", "Closed-loop"]
    nets = [dense["net"], seq["net"], program["net"], closed["net"]]
    axis.bar(names, nets)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_ylabel("Net = W→C - C→W")
    figure.tight_layout()
    figure.savefig(output_root / "figures/closed_loop_vs_open_loop.png", dpi=160)
    plt.close(figure)


def aggregate_evaluation(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    frozen = read_json(output_root / "work/evaluation_input.json")
    manifest = read_jsonl(output_root / "work/evaluation_manifest.jsonl")
    checkpoint_sha = frozen["full_refit_checkpoint_sha256"]
    rows: list[dict[str, Any]] = []
    for rank in range(int(config["world_size"])):
        rank_root = output_root / f"work/evaluation/rank{rank:02d}"
        completion = read_json(rank_root / "complete.json")
        expected = sum(int(row["worker_rank"]) == rank for row in manifest)
        if (
            completion.get("contract_sha256") != contract["contract_sha256"]
            or completion.get("checkpoint_sha256") != checkpoint_sha
            or int(completion.get("expected_rows", -1)) != expected
            or int(completion.get("completed_rows", -1)) != expected
        ):
            raise RuntimeError(f"evaluation rank {rank} is incomplete")
        for path in sorted(rank_root.glob("part_*.jsonl")):
            rows.extend(read_jsonl(path))
    observed = Counter(str(row["uid"]) for row in rows)
    expected_uids = [str(row["uid"]) for row in manifest]
    if Counter(expected_uids) != observed or len(rows) != int(config["evaluation"]["expected_total"]):
        raise RuntimeError("global full evaluation is missing or duplicating UIDs")
    by_uid = {str(row["uid"]): row for row in rows}
    rows = [by_uid[uid] for uid in expected_uids]
    if not all(
        all(row["phase69_reuse_parity"].values())
        and all(row["phase74_reuse_parity"].values())
        and row["state_feedback_exact"]
        and row["checkpoint_sha256"] == checkpoint_sha
        for row in rows
    ):
        raise RuntimeError("baseline parity or state-feedback sanity failed globally")
    scopes = _scopes(rows)
    for family in ("chartqa", "textvqa", "mmmu_pro", "pope"):
        atomic_jsonl(output_root / f"evaluation/{family}/paired_results.jsonl", scopes[family])
    atomic_jsonl(output_root / "evaluation/all_paired.jsonl", rows)

    method_keys = (
        ("Dense", "dense_correct"),
        ("Sequential-A", "sequential_correct"),
        ("Open-loop Program", "program_correct"),
        ("Closed-loop Trajectory-Set", "closed_loop_correct"),
    )
    summary_rows = []
    transition_rows = []
    bootstrap_rows = []
    for scope, current in scopes.items():
        summaries = {name: _method_summary(current, key) for name, key in method_keys}
        summary_rows.append({
            "scope": scope,
            "n": len(current),
            **{
                f"{name.lower().replace('-', '_').replace(' ', '_')}_accuracy": value["accuracy"]
                for name, value in summaries.items()
            },
            "closed_loop_w_to_c": summaries["Closed-loop Trajectory-Set"]["w_to_c"],
            "closed_loop_c_to_w": summaries["Closed-loop Trajectory-Set"]["c_to_w"],
            "closed_loop_net": summaries["Closed-loop Trajectory-Set"]["net"],
        })
        for name, result in summaries.items():
            transition_rows.append({
                "scope": scope,
                "method": name,
                **{key: result[key] for key in ("w_to_c", "c_to_w", "c_to_c", "w_to_w", "net")},
            })
        for comparison, baseline_key in (
            ("closed_loop_minus_dense", "dense_correct"),
            ("closed_loop_minus_open_loop_program", "program_correct"),
        ):
            paired = [
                {
                    "dense_correct": bool(row[baseline_key]),
                    "routed_correct": bool(row["closed_loop_correct"]),
                    "transition": ("C" if row[baseline_key] else "W")
                    + "→"
                    + ("C" if row["closed_loop_correct"] else "W"),
                }
                for row in current
            ]
            for item in paired_bootstrap(
                paired,
                draws=int(config["evaluation"]["paired_bootstrap_draws"]),
                seed=int(config["evaluation"]["paired_bootstrap_seed"]) + len(bootstrap_rows),
            ):
                bootstrap_rows.append({"scope": scope, "comparison": comparison, **item})
    atomic_csv(output_root / "metrics/full_benchmark_summary.csv", summary_rows)
    atomic_csv(output_root / "metrics/transition_counts.csv", transition_rows)
    atomic_csv(output_root / "metrics/paired_bootstrap.csv", bootstrap_rows)

    funnel_rows = []
    action_usage_rows = []
    intervention_rows = []
    preservation_rows = []
    treatment_rows = []
    for scope, current in scopes.items():
        dense_c = [row for row in current if row["dense_correct"]]
        dense_w = [row for row in current if not row["dense_correct"]]
        triggered_c = [row for row in dense_c if row["triggered"]]
        triggered_w = [row for row in dense_w if row["triggered"]]
        funnel_rows.append({
            "scope": scope,
            "n": len(current),
            "dense_c": len(dense_c),
            "dense_w": len(dense_w),
            "triggered_c": len(triggered_c),
            "triggered_w": len(triggered_w),
            "p_trigger_given_c": _safe_rate(len(triggered_c), len(dense_c)),
            "p_trigger_given_w": _safe_rate(len(triggered_w), len(dense_w)),
            "triggered_c_any_non_full": sum(row["any_non_full"] for row in triggered_c),
            "triggered_w_any_non_full": sum(row["any_non_full"] for row in triggered_w),
            "w_to_c": sum(row["closed_loop_transition"] == "W→C" for row in current),
            "c_to_w": sum(row["closed_loop_transition"] == "C→W" for row in current),
        })
        for outcome, subset in (
            ("Dense-C", triggered_c),
            ("Dense-W", triggered_w),
            ("W→C", [row for row in current if row["closed_loop_transition"] == "W→C"]),
            ("C→W", [row for row in current if row["closed_loop_transition"] == "C→W"]),
            ("C→C", [row for row in current if row["closed_loop_transition"] == "C→C" and row["triggered"]]),
            ("W→W", [row for row in current if row["closed_loop_transition"] == "W→W" and row["triggered"]]),
        ):
            counts = Counter(action for row in subset for action in row["closed_loop_suffix_actions"])
            for action in ACTION_NAMES:
                action_usage_rows.append({
                    "scope": scope, "outcome": outcome, "action": action, "count": counts[action]
                })
            intervention_rows.append({
                "scope": scope,
                "outcome": outcome,
                "n": len(subset),
                "any_non_full": sum(row["any_non_full"] for row in subset),
                "fraction_any_non_full": _safe_rate(sum(row["any_non_full"] for row in subset), len(subset)),
                "mean_non_full": _mean([row["non_full_count"] for row in subset]),
                "mean_first_non_full_layer": _mean([row["first_non_full_layer"] for row in subset if row["first_non_full_layer"] is not None]),
                "mean_trigger_to_first_intervention_delay": _mean([row["trigger_to_first_non_full_delay"] for row in subset if row["trigger_to_first_non_full_delay"] is not None]),
                "mean_action_switches": _mean([row["action_switches"] for row in subset]),
            })
        preservation_rows.append({
            "scope": scope,
            "triggered_dense_c": len(triggered_c),
            "triggered_c_any_non_full": sum(row["any_non_full"] for row in triggered_c),
            "c_to_w": sum(row["closed_loop_transition"] == "C→W" for row in triggered_c),
            "preservation_rate": _safe_rate(sum(row["closed_loop_transition"] == "C→C" for row in triggered_c), len(triggered_c)),
        })
        treatment_rows.append({
            "scope": scope,
            "triggered_dense_w": len(triggered_w),
            "triggered_w_any_non_full": sum(row["any_non_full"] for row in triggered_w),
            "w_to_c": sum(row["closed_loop_transition"] == "W→C" for row in triggered_w),
            "rescue_rate": _safe_rate(sum(row["closed_loop_transition"] == "W→C" for row in triggered_w), len(triggered_w)),
        })
    atomic_csv(output_root / "metrics/stage1_stage2_funnel.csv", funnel_rows)
    atomic_csv(output_root / "metrics/action_usage.csv", action_usage_rows)
    atomic_csv(output_root / "metrics/intervention_statistics.csv", intervention_rows)
    atomic_csv(output_root / "metrics/dense_c_preservation.csv", preservation_rows)
    atomic_csv(output_root / "metrics/triggered_w_treatment.csv", treatment_rows)

    trigger_rows = []
    for (family, layer), current in sorted(
        (key, [row for row in rows if row["benchmark_family"] == key[0] and row["trigger_layer"] == key[1]])
        for key in {(row["benchmark_family"], int(row["trigger_layer"])) for row in rows if row["triggered"]}
    ):
        trigger_rows.append({
            "benchmark_family": family,
            "trigger_layer": layer,
            "n": len(current),
            "any_non_full": sum(row["any_non_full"] for row in current),
            "w_to_c": sum(row["closed_loop_transition"] == "W→C" for row in current),
            "c_to_w": sum(row["closed_loop_transition"] == "C→W" for row in current),
        })
    atomic_csv(output_root / "metrics/trigger_layer_breakdown.csv", trigger_rows)
    benchmark_rows = []
    for benchmark in sorted({row["benchmark"] for row in rows}):
        current = [row for row in rows if row["benchmark"] == benchmark]
        values = {name: _method_summary(current, key) for name, key in method_keys}
        closed = values["Closed-loop Trajectory-Set"]
        benchmark_rows.append({
            "benchmark": benchmark,
            "n": len(current),
            "dense_accuracy": values["Dense"]["accuracy"],
            "sequential_accuracy": values["Sequential-A"]["accuracy"],
            "open_loop_program_accuracy": values["Open-loop Program"]["accuracy"],
            "closed_loop_accuracy": closed["accuracy"],
            "w_to_c": closed["w_to_c"],
            "c_to_w": closed["c_to_w"],
            "net": closed["net"],
        })
    atomic_csv(output_root / "metrics/benchmark_breakdown.csv", benchmark_rows)

    feedback_rows = []
    for row in rows:
        for item in row["action_rows"]:
            if item["active"]:
                feedback_rows.append({
                    "uid": row["uid"],
                    "benchmark_family": row["benchmark_family"],
                    "trigger_layer": row["trigger_layer"],
                    "layer": item["layer"],
                    "previous_action": item["previous_action"],
                    "action": item["action"],
                    "state_sha256": item["state_sha256"],
                    "next_state_sha256": item["next_state_sha256"],
                    "next_layer_logits": json.dumps(item["next_layer_logits"]),
                })
    atomic_csv(output_root / "diagnostics/state_feedback_analysis.csv", feedback_rows)
    responsibility_rows = list(
        csv.DictReader((output_root / "training/responsibility_statistics.csv").open())
    )
    _write_figures(output_root, scopes, responsibility_rows)

    overall = {name: _method_summary(rows, key) for name, key in method_keys}
    closed = overall["Closed-loop Trajectory-Set"]
    sequential = overall["Sequential-A"]
    program = overall["Open-loop Program"]
    family_values = {
        family: _method_summary(current, "closed_loop_correct")
        for family, current in scopes.items() if family != "overall"
    }
    best_probs = [
        math.exp(float(row["best_route_logp"]) / (28 - int(row["trigger_layer"])))
        for row in responsibility_rows
    ]
    median_best_probability = float(np.median(best_probs))
    internal_fit_strong = median_best_probability >= float(
        config["diagnostics"]["internal_fit_strong_threshold"]
    )
    mean_top_responsibility = float(np.mean([float(row["top_responsibility"]) for row in responsibility_rows]))
    if closed["net"] > 0:
        recommendation = "Keep the closed-loop trajectory-set formulation and run one separately authorized conservative refinement of its smallest observed bottleneck."
        bottleneck = "remaining treatment/selectivity behavior under a positive formulation"
    elif closed["w_to_c"] > program["w_to_c"] and closed["c_to_w"] >= closed["w_to_c"]:
        recommendation = "Run one separately authorized training-side conservative treatment/selectivity experiment while keeping the closed-loop trajectory-set formulation fixed."
        bottleneck = "preservation/selectivity"
    elif internal_fit_strong and closed["w_to_c"] <= max(
        int(config["diagnostics"]["external_rescue_near_prior_ceiling"]),
        program["w_to_c"],
    ):
        recommendation = "Run one separately authorized on-policy state-distribution diagnostic/relabeling experiment with the architecture and Stage-1 gate fixed."
        bottleneck = "on-policy state-distribution shift"
    else:
        recommendation = "Run one separately authorized minimal Stage-2 representation-enrichment experiment with the objective, Stage-1 gate, and evaluation fixed."
        bottleneck = "optimization or representation/generalization"
    summary_lines = [
        "# Closed-loop trajectory-set full evaluation summary",
        "",
        f"- Contract: `{contract['contract_sha256']}`",
        f"- Full corpus: **{contract['population']['uids']} UIDs / {contract['population']['programs']} replay-valid trajectories**.",
        f"- Dense-C/Dense-W UIDs: **{contract['population']['dense_c_uids']} / {contract['population']['dense_w_uids']}**.",
        f"- Routed-state cache: **{contract['population']['unique_prefix_states']} unique states / {contract['population']['route_state_occurrences']} route occurrences**, zero quarantines.",
        "- Exact marginal objective numerical/gradient tests: **passed**.",
        "- Full-corpus refit: **completed**.",
        f"- Full external evaluation: **{len(rows):,}/{config['evaluation']['expected_total']:,} rows**, exact Dense/Stage-1 baseline parity and exact state-feedback trace checks.",
        "",
        "| Method | Accuracy | W→C | C→W | Net |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("Dense", "Sequential-A", "Open-loop Program", "Closed-loop Trajectory-Set"):
        value = overall[name]
        summary_lines.append(
            f"| {name} | {value['accuracy']:.6f} | {value['w_to_c']} | {value['c_to_w']} | {value['net']} |"
        )
    overall_funnel = next(row for row in funnel_rows if row["scope"] == "overall")
    overall_intervention = next(
        row for row in intervention_rows if row["scope"] == "overall" and row["outcome"] == "Dense-W"
    )
    overall_c_intervention = next(
        row for row in intervention_rows if row["scope"] == "overall" and row["outcome"] == "Dense-C"
    )
    provenance = next(csv.DictReader((output_root / "corpus/corpus_summary.csv").open()))
    summary_lines.extend([
        "",
        "## Required answers",
        "",
        f"1. Training population: {contract['population']['uids']} UIDs and {contract['population']['programs']} trajectories.",
        f"2. Dense-C preservation / Dense-W corrective UIDs: {contract['population']['dense_c_uids']} / {contract['population']['dense_w_uids']}.",
        f"3. Provenance occurrences: preservation {provenance['preservation']}, single {provenance['single']}, original MCTS {provenance['original_mcts']}, robust search {provenance['robust_search']}, completeness audit {provenance['completeness_audit']}.",
        "4. Routed-state replay parity: passed for every retained route; quarantined routes: 0.",
        "5. Exact marginal objective numerical and brute-force gradient parity: passed.",
        "6. Full-corpus refit: completed from exact Sequential-A initialization.",
        "7. All four method accuracies are in the table above.",
        f"8. Closed-loop W→C/C→W/Net: {closed['w_to_c']}/{closed['c_to_w']}/{closed['net']} pooled; per benchmark is in `metrics/benchmark_breakdown.csv`.",
        f"9. Pooled Net positive: **{closed['net'] > 0}**.",
        f"10. Closed-loop accuracy >= Dense: **{closed['accuracy'] >= overall['Dense']['accuracy']}**.",
        f"11. W rescue improved over Sequential-A and Open-loop: **{closed['w_to_c'] > sequential['w_to_c'] and closed['w_to_c'] > program['w_to_c']}**.",
        f"12. C preservation improved over Sequential-A: **{closed['c_to_w'] < sequential['c_to_w']}**.",
        f"13. Triggered-W receiving non-FULL: {overall_funnel['triggered_w_any_non_full']}/{overall_funnel['triggered_w']} ({_safe_rate(overall_funnel['triggered_w_any_non_full'], overall_funnel['triggered_w']):.4f}).",
        f"14. Triggered-C receiving non-FULL: {overall_funnel['triggered_c_any_non_full']}/{overall_funnel['triggered_c']} ({_safe_rate(overall_funnel['triggered_c_any_non_full'], overall_funnel['triggered_c']):.4f}).",
        f"15. Mean trigger-to-first-intervention delay: W {overall_intervention['mean_trigger_to_first_intervention_delay']}; C {overall_c_intervention['mean_trigger_to_first_intervention_delay']}.",
        f"16. TextVQA rescues: {family_values['textvqa']['w_to_c']}.",
        f"17. MMMU-Pro rescues: {family_values['mmmu_pro']['w_to_c']}; improved beyond two: **{family_values['mmmu_pro']['w_to_c'] > 2}**.",
        f"18. POPE triggers: {next(row for row in funnel_rows if row['scope'] == 'pope')['triggered_c'] + next(row for row in funnel_rows if row['scope'] == 'pope')['triggered_w']}; Net {family_values['pope']['net']}.",
        f"19. Route responsibility mean top share: {mean_top_responsibility:.4f}; median best-route geometric action probability: {median_best_probability:.4f}.",
        f"20. Evidence supports this formulation as a deployment winner: **{closed['net'] > 0 and closed['accuracy'] >= overall['Dense']['accuracy']}**.",
        f"21. If unsuccessful, the next bottleneck is most consistent with: **{bottleneck}** under the fixed diagnostic rule.",
        "22. This result does not establish that dynamic READ/WRITE routing, MCTS, Stage-1, richer representations, or on-policy relabeling are generally ineffective.",
        "",
        "Paired bootstrap intervals are in `metrics/paired_bootstrap.csv`.",
    ])
    _atomic_bytes(
        output_root / "summaries/closed_loop_trajectory_set_full_eval_summary.md",
        ("\n".join(summary_lines) + "\n").encode(),
    )
    _atomic_bytes(
        output_root / "summaries/next_stage2_recommendation.md",
        (
            "# Next Stage-2 recommendation\n\n"
            + recommendation
            + "\n\nThis is one unexecuted recommendation. No follow-on experiment is authorized or run in Phase 76.\n"
        ).encode(),
    )

    required = [
        path
        for path in output_root.rglob("*")
        if path.is_file()
        and "work" not in path.relative_to(output_root).parts
        and "corpus/routed_states" not in str(path.relative_to(output_root))
        and path.name != "artifact_manifest.json"
    ]
    artifact = {
        "schema_version": "closed_loop_trajectory_set_artifact_manifest_v1",
        "contract_sha256": contract["contract_sha256"],
        "completed_at": utc_now(),
        "evaluation_rows": len(rows),
        "full_refit_checkpoint_sha256": checkpoint_sha,
        "files": {str(path.relative_to(output_root)): file_sha256(path) for path in sorted(required)},
        "routed_state_cache_bound_by": "corpus/uid_state_files.jsonl",
        "routed_state_files": int(contract["population"]["uids"]),
        "routed_states": int(contract["population"]["unique_prefix_states"]),
    }
    atomic_json(output_root / "artifact_manifest.json", artifact)
    print(json.dumps({
        "rows": len(rows),
        "dense_accuracy": overall["Dense"]["accuracy"],
        "sequential_accuracy": sequential["accuracy"],
        "program_accuracy": program["accuracy"],
        "closed_loop_accuracy": closed["accuracy"],
        "w_to_c": closed["w_to_c"],
        "c_to_w": closed["c_to_w"],
        "net": closed["net"],
        "recommended_bottleneck": bottleneck,
    }, sort_keys=True))


def verify_artifacts(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path)
    artifact = read_json(output_root / "artifact_manifest.json")
    if artifact.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("artifact manifest belongs to another contract")
    failures = []
    for relative, digest in artifact["files"].items():
        path = output_root / relative
        if not path.is_file() or file_sha256(path) != digest:
            failures.append(relative)
    files = _uid_file_index(contract, output_root)
    if len(files) != int(artifact["routed_state_files"]):
        failures.append("corpus/uid_state_files.jsonl")
    if failures:
        raise RuntimeError(f"artifact verification failed: {failures[:5]}")
    print(json.dumps({
        "passed": True,
        "declared_files": len(artifact["files"]),
        "routed_state_files": len(files),
        "routed_states": artifact["routed_states"],
        "evaluation_rows": artifact["evaluation_rows"],
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "prepare", "smoke", "replay-worker", "aggregate-replay", "train",
        "prepare-evaluation", "evaluation-worker", "aggregate-evaluation", "verify-artifacts",
    ))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mode", choices=("internal", "full"), default="internal")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config)
    elif args.command == "smoke":
        smoke(args.config, args.device)
    elif args.command == "replay-worker":
        replay_worker(args.config, args.rank, args.resume)
    elif args.command == "aggregate-replay":
        aggregate_replay(args.config)
    elif args.command == "train":
        train(args.config, args.mode)
    elif args.command == "prepare-evaluation":
        prepare_evaluation(args.config)
    elif args.command == "evaluation-worker":
        evaluation_worker(args.config, args.rank, args.resume)
    elif args.command == "aggregate-evaluation":
        aggregate_evaluation(args.config)
    else:
        verify_artifacts(args.config)


if __name__ == "__main__":
    main()
