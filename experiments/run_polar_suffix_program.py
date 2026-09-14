#!/usr/bin/env python3
"""Prepare, replay, train, and evaluate the Phase-74 suffix-program policy."""

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

import torch
from torch import nn

from binary_policy.executor import (
    capture_four_action_route,
    capture_four_action_suffix_from_full_baseline,
)
from binary_policy.executor.inputs import build_binary_inputs
from dense_failure_stage1.runtime import build_dense_inputs, configure_dense_determinism
from dense_failure_stage2.polar_suffix_program import (
    ACTION_NAMES,
    PolarSuffixProgramPredictor,
    assign_group_disjoint_dev,
    build_program_corpus,
    hierarchical_program_weights,
    initialize_from_stage2a,
    weighted_program_loss,
)
from experiments.run_stage2_v1_training_revised import _generate as _train_generate, _load_model


DEFAULT_CONFIG = PROJECT_ROOT / "configs/polar_suffix_program_v1.json"
ALLOWED_ROOTS = (PROJECT_ROOT.resolve(), Path("/mnt/hyemin").resolve())
BOUND_CODE = (
    "configs/polar_suffix_program_v1.json",
    "dense_failure_stage2/polar_suffix_program.py",
    "experiments/run_polar_suffix_program.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/model.py",
    "dense_failure_stage1/lmms_scoring.py",
    "dense_failure_stage1/runtime.py",
    "dense_failure_stage2/v1_router.py",
    "experiments/run_stage2_v1_training_revised.py",
    "experiments/run_full_benchmark_end_to_end_eval.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if not any(resolved == root or resolved.is_relative_to(root) for root in ALLOWED_ROOTS):
        raise ValueError(f"path escapes allowed roots: {value}")
    return resolved


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with resolve_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
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
    payload = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    _atomic_bytes(path, payload.encode())


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        _atomic_bytes(path, b"")
        return
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, buffer.getvalue().encode())


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_config(path: str | Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "polar_suffix_program_config_v1":
        raise ValueError("unexpected Phase-74 config schema")
    if int(config.get("world_size", 0)) != 4:
        raise ValueError("Phase-74 requires all four available GPUs")
    if list(config["program_model"]["actions"]) != list(ACTION_NAMES):
        raise ValueError("program action order differs from the four-action contract")
    if config["corpus"]["operating_point"] != "P90":
        raise ValueError("only the frozen P90 operating point is supported")
    if config["stage1"]["comparison"] != "strict_greater_than":
        raise ValueError("Stage-1 comparison semantics differ")
    return config


def _git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.run(
            args, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "worktree_status_at_freeze": run("git", "status", "--short"),
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
    }


def _program_model(config: Mapping[str, Any], device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    settings = config["program_model"]
    torch.manual_seed(int(config["seed"]))
    model = PolarSuffixProgramPredictor(
        hidden_size=int(config["model"]["hidden_size"]),
        router_size=int(settings["router_size"]),
        num_heads=int(settings["num_heads"]),
        decoder_layers=int(settings["decoder_blocks"]),
        feedforward_size=int(settings["feedforward_size"]),
        dropout=float(settings["dropout"]),
        total_layers=int(config["model"]["decoder_layers"]),
    ).to(device=device, dtype=torch.float32)
    checkpoint = torch.load(
        resolve_path(config["stage2a_initialization"]["checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("contract_sha256") != config["stage2a_initialization"][
        "parent_contract_sha256"
    ]:
        raise RuntimeError("Stage2-A checkpoint parent contract differs")
    report = initialize_from_stage2a(model, checkpoint["state_dict"])
    if not report["all_exact"]:
        raise RuntimeError("Stage2-A initialization was not exact")
    return model, report


def _source_paths(config: Mapping[str, Any]) -> dict[str, str]:
    corpus = config["corpus"]
    return {
        "p90_preservation": corpus["preservation"],
        "p90_single": corpus["single"],
        "p90_mcts": corpus["mcts"],
        "phase65_route_store": corpus["route_store"],
        "phase65_work_manifest": corpus["work_manifest"],
        "phase65_contract": corpus["phase65_contract"],
        "phase73_state_manifest": corpus["completeness_state_manifest"],
        "phase73_replay": corpus["completeness_replay"],
        "phase73_contract": corpus["phase73_contract"],
        "stage2a_checkpoint": config["stage2a_initialization"]["checkpoint"],
        "plan": "plans/polar_style_post_trigger_suffix_program_predictor_plan.md",
        "evaluation_config": config["evaluation"]["base_config"],
        "phase69_contract": config["evaluation"]["phase69_contract"],
        "phase69_paired": config["evaluation"]["phase69_paired"],
    }


def _verify_model_snapshot(snapshot: Path, expected: Mapping[str, str]) -> None:
    for relative, digest in expected.items():
        path = snapshot / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise RuntimeError(f"model snapshot file differs: {path}")


def prepare(config_path: Path) -> None:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    frozen_path = output_root / "frozen_protocol.json"
    if frozen_path.exists():
        raise RuntimeError("Phase-74 output is already frozen; verify or resume it instead")

    phase65 = read_json(config["corpus"]["phase65_contract"])
    phase73 = read_json(config["corpus"]["phase73_contract"])
    if phase65.get("contract_sha256") != "767284f1e8ed76d5d1c797562a3aaea4ea59183703d62b1521545cacc7c57ab4":
        raise RuntimeError("Phase-65 contract differs from the approved P90 source")
    if phase73.get("contract_sha256") != "3c371c2f4f8372b36eb5965a97344adc6d4f1a76ae906009a93c72e63bff08cc":
        raise RuntimeError("Phase-73 contract differs from the completeness source")
    if file_sha256(config["stage2a_initialization"]["checkpoint"]) != config[
        "stage2a_initialization"
    ]["checkpoint_sha256"]:
        raise RuntimeError("Stage2-A checkpoint hash differs")
    _verify_model_snapshot(
        resolve_path(config["model"]["snapshot_path"]), phase73["model_snapshot_sha256"]
    )

    base_routes = []
    for key in ("preservation", "single", "mcts"):
        base_routes.extend(read_jsonl(config["corpus"][key]))
    completeness = read_jsonl(config["corpus"]["completeness_replay"])
    work_rows = read_jsonl(config["corpus"]["work_manifest"])
    route_store = read_jsonl(config["corpus"]["route_store"])
    programs = build_program_corpus(base_routes, completeness, work_rows, route_store)
    if len(programs) != 4948 or len({row["uid"] for row in programs}) != 569:
        raise RuntimeError("eligible corpus differs from the audited 4,948-program/569-UID population")
    weights = hierarchical_program_weights(
        programs, minimum_cell_uids=int(config["training"]["minimum_cell_uids"])
    )
    split = assign_group_disjoint_dev(
        programs,
        dev_fraction=float(config["training"]["dev_fraction"]),
        seed=int(config["seed"]),
    )
    for row, weight in zip(programs, weights):
        row["program_weight"] = weight
        row["internal_split"] = split[row["uid"]]

    output_root.mkdir(parents=True, exist_ok=False)
    corpus_root = output_root / "corpus"
    splits_root = output_root / "splits"
    work_root = output_root / "work"
    manifest_path = corpus_root / "p90_program_corpus_manifest.jsonl"
    atomic_jsonl(manifest_path, programs)

    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in programs:
        by_uid[row["uid"]].append(row)
    uid_index = {
        uid: {
            "program_ids": [row["program_id"] for row in rows],
            "program_count": len(rows),
            "dataset": rows[0]["dataset"],
            "source_regime": rows[0]["source_regime"],
            "dense_outcome": rows[0]["dense_outcome"],
            "trigger_layer": rows[0]["trigger_layer"],
            "image_group_id": rows[0]["image_group_id"],
            "internal_split": rows[0]["internal_split"],
        }
        for uid, rows in sorted(by_uid.items())
    }
    atomic_json(corpus_root / "uid_program_index.json", uid_index)
    atomic_jsonl(
        corpus_root / "program_provenance.jsonl",
        ({key: row[key] for key in ("program_id", "uid", "provenance", "source_references")} for row in programs),
    )
    imported = [
        row for row in programs if "completeness_audit" in row["provenance"]
    ]
    atomic_jsonl(corpus_root / "completeness_audit_program_import.jsonl", imported)

    provenance_counts = Counter(source for row in programs for source in row["provenance"])
    atomic_csv(
        corpus_root / "corpus_summary.csv",
        [{
            "uids": len(by_uid),
            "image_groups": len({row["image_group_id"] for row in programs}),
            "programs": len(programs),
            "dense_c_uids": sum(rows[0]["dense_outcome"] == "C" for rows in by_uid.values()),
            "dense_w_uids": sum(rows[0]["dense_outcome"] == "W" for rows in by_uid.values()),
            **{source: provenance_counts[source] for source in (
                "preservation", "single", "original_mcts", "robust_search", "completeness_audit"
            )},
        }],
    )
    cell_counts = Counter(
        (row["dataset"], row["source_regime"], row["dense_outcome"]) for row in programs
    )
    atomic_csv(
        corpus_root / "source_outcome_counts.csv",
        [
            {"dataset": key[0], "source_regime": key[1], "dense_outcome": key[2], "programs": count}
            for key, count in sorted(cell_counts.items())
        ],
    )
    atomic_csv(
        corpus_root / "program_cardinality_per_uid.csv",
        [
            {"uid": uid, "programs": len(rows), "dense_outcome": rows[0]["dense_outcome"]}
            for uid, rows in sorted(by_uid.items())
        ],
    )
    dev_groups = sorted({row["image_group_id"] for row in programs if row["internal_split"] == "dev"})
    train_groups = {row["image_group_id"] for row in programs if row["internal_split"] == "train"}
    if train_groups.intersection(dev_groups):
        raise AssertionError("internal split contains image-group overlap")
    atomic_json(splits_root / "internal_dev_groups.json", {
        "seed": int(config["seed"]), "dev_fraction": config["training"]["dev_fraction"],
        "dev_groups": dev_groups, "uid_split": split, "image_group_overlap": 0,
    })
    atomic_json(splits_root / "full_refit_manifest.json", {
        "program_ids": [row["program_id"] for row in programs],
        "uids": sorted(by_uid),
        "program_count": len(programs),
        "uid_count": len(by_uid),
    })

    schedule = []
    for uid, rows in sorted(by_uid.items()):
        rank = int(sha256(uid.encode()).hexdigest(), 16) % int(config["world_size"])
        schedule.append({
            "uid": uid, "worker_rank": rank,
            "program_ids": [row["program_id"] for row in rows],
            "program_count": len(rows), "sample": rows[0]["sample"],
        })
    atomic_jsonl(work_root / "replay_manifest.jsonl", schedule)

    source_hashes = {name: file_sha256(path) for name, path in _source_paths(config).items()}
    code_hashes = {path: file_sha256(path) for path in BOUND_CODE}
    internal_paths = (
        "corpus/p90_program_corpus_manifest.jsonl",
        "corpus/uid_program_index.json",
        "splits/internal_dev_groups.json",
        "splits/full_refit_manifest.json",
        "work/replay_manifest.jsonl",
    )
    contract = {
        "schema_version": "polar_suffix_program_contract_v1",
        "created_at": utc_now(),
        "static_config": config,
        "config_sha256": file_sha256(config_path),
        "source_sha256": source_hashes,
        "bound_code_sha256": code_hashes,
        "internal_manifest_sha256": {
            path: file_sha256(output_root / path) for path in internal_paths
        },
        "parent_contracts": {
            "phase65": phase65["contract_sha256"],
            "phase73": phase73["contract_sha256"],
            "stage2a": config["stage2a_initialization"]["parent_contract_sha256"],
        },
        "model_snapshot_sha256": phase73["model_snapshot_sha256"],
        "git": _git_state(),
        "runtime": _runtime_state(),
        "population": {
            "uids": len(by_uid), "image_groups": len({row["image_group_id"] for row in programs}),
            "programs": len(programs), "dense_c_uids": 106, "dense_w_uids": 463,
            "dev_uids": sum(value == "dev" for value in split.values()),
            "train_uids": sum(value == "train" for value in split.values()),
            "provenance_counts": dict(provenance_counts),
        },
        "review_reconciliation": {
            "verdict": "revise_then_proceed",
            "all_uid_cache_to_live_parity_required": True,
            "length_normalized_program_nll": True,
        },
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(frozen_path, contract)
    atomic_json(output_root / "model/model_config.json", config["program_model"])
    atomic_json(output_root / "decoding/beam8_config.json", {
        "beam_width": 8, "score": "sum_action_log_probabilities", "primary": True,
    })
    protocol = f"""# Phase-74 frozen protocol

- Contract: `{contract['contract_sha256']}`
- Population: {len(by_uid)} P90-triggered UIDs / {len(programs)} unique programs.
- Dense-C: only all-FULL suffixes. Dense-W: every unique exact-replay-valid complete suffix.
- Objective: length-normalized teacher-forced program NLL, then dataset × source-regime × outcome / UID / 1-K weights.
- Trigger cache: raw BF16 all-FULL P90 token states and masks; exact cache/live branch, context, and initial-logit parity is mandatory for every UID.
- Decoder: 2 blocks, width 256, 4 heads, FFN 1024; beam-8 sum-log-probability primary.
- Internal dev selects only epoch count; the final checkpoint is reinitialized and refit on 100% of the corpus.
- Evaluation: exact Phase-69 19,960-row ChartQA/TextVQA/MMU-Pro/POPE protocol.
"""
    _atomic_bytes(output_root / "protocol.md", protocol.encode())
    print(json.dumps({"contract_sha256": contract["contract_sha256"], **contract["population"]}, sort_keys=True))


def verify_contract(config_path: Path, *, verify_model: bool = False) -> tuple[dict[str, Any], Path]:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if contract.get("contract_sha256") != canonical_hash(contract):
        raise RuntimeError("Phase-74 frozen contract hash differs")
    if contract.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError("Phase-74 config changed after freeze")
    for name, path in _source_paths(config).items():
        if file_sha256(path) != contract["source_sha256"][name]:
            raise RuntimeError(f"Phase-74 source changed after freeze: {name}")
    for path, digest in contract["bound_code_sha256"].items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"Phase-74 bound code changed after freeze: {path}")
    for path, digest in contract["internal_manifest_sha256"].items():
        if file_sha256(output_root / path) != digest:
            raise RuntimeError(f"Phase-74 internal manifest changed: {path}")
    if verify_model:
        _verify_model_snapshot(
            resolve_path(config["model"]["snapshot_path"]), contract["model_snapshot_sha256"]
        )
    return contract, output_root


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    payload = contiguous.view(torch.uint8).numpy().tobytes()
    header = f"{contiguous.dtype}:{tuple(contiguous.shape)}:".encode()
    return sha256(header + payload).hexdigest()


def _uid_slug(uid: str) -> str:
    return sha256(uid.encode()).hexdigest()[:24]


def _checkpoint_state(config: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    checkpoint = torch.load(
        resolve_path(config["stage2a_initialization"]["checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    return checkpoint["state_dict"]


def _initial_outputs(
    model: PolarSuffixProgramPredictor,
    state: Mapping[str, torch.Tensor],
    trigger: int,
    suffix_indices: Sequence[int],
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    text = state["text_states"].to(device)
    visual = state["visual_states"].to(device)
    text_mask = state["text_mask"].to(device)
    visual_mask = state["visual_mask"].to(device)
    z_read, z_write = model.compute_branches(text, visual, text_mask, visual_mask)
    context = model.compute_context(text, visual, text_mask, visual_mask)
    targets = torch.tensor([list(suffix_indices)], dtype=torch.long, device=device)
    logits = model.decode_teacher_forced(
        context, torch.tensor([int(trigger)], dtype=torch.long, device=device), targets
    )
    return {
        "z_read": z_read,
        "z_write": z_write,
        "context": context,
        "initial_decoder_logits": logits,
    }


def _replay_uid(
    *,
    config: Mapping[str, Any],
    contract_sha256: str,
    processor,
    wrapped,
    program_model: PolarSuffixProgramPredictor,
    rows: Sequence[Mapping[str, Any]],
    state_path: Path,
) -> dict[str, Any]:
    uid = str(rows[0]["uid"])
    if any(str(row["uid"]) != uid for row in rows):
        raise ValueError("replay batch mixes UIDs")
    trigger = int(rows[0]["trigger_layer"])
    sample = dict(rows[0]["sample"])
    started = time.monotonic()
    inputs, metadata = build_dense_inputs(processor, sample, next(program_model.parameters()).device)
    prepared = build_binary_inputs(wrapped, inputs)
    baseline = capture_four_action_route(
        wrapped,
        {},
        ["FULL"] * int(config["model"]["decoder_layers"]),
        prepared_inputs=prepared,
        use_cache=True,
        native_full_rows=True,
    )
    dense_ids, dense_text, dense_score = _train_generate(
        processor, wrapped, baseline, inputs["input_ids"], sample
    )
    expected_dense = [int(token) for token in rows[0]["dense_generated_token_ids"]]
    if dense_ids != expected_dense:
        raise RuntimeError(f"native all-FULL token parity failed for {uid}")
    if bool(dense_score.correct) != (str(rows[0]["dense_outcome"]) == "C"):
        raise RuntimeError(f"native all-FULL correctness differs for {uid}")

    live_text, live_visual = baseline.pre_layer_states[trigger]
    live_state = {
        "text_states": live_text.detach(),
        "visual_states": live_visual.detach(),
        "text_mask": prepared.text_valid_mask.detach(),
        "visual_mask": prepared.visual_valid_mask.detach(),
    }
    state = {
        "schema_version": "polar_trigger_state_v1",
        "contract_sha256": contract_sha256,
        "uid": uid,
        "trigger_layer": trigger,
        "text_states": live_text.detach().cpu().to(torch.bfloat16).contiguous(),
        "visual_states": live_visual.detach().cpu().to(torch.bfloat16).contiguous(),
        "text_mask": prepared.text_valid_mask.detach().cpu().bool().contiguous(),
        "visual_mask": prepared.visual_valid_mask.detach().cpu().bool().contiguous(),
    }
    state["tensor_sha256"] = {
        name: _tensor_sha256(state[name])
        for name in ("text_states", "visual_states", "text_mask", "visual_mask")
    }
    first_suffix = list(rows[0]["suffix_action_indices"])
    live_outputs = _initial_outputs(program_model, live_state, trigger, first_suffix)
    atomic_torch(state_path, state)
    cached = torch.load(state_path, map_location="cpu", weights_only=False)
    if cached.get("contract_sha256") != contract_sha256 or cached.get("uid") != uid:
        raise RuntimeError(f"cached trigger provenance failed for {uid}")
    tensor_exact = {
        name: torch.equal(live_state[name].cpu(), cached[name])
        and torch.equal(state[name], cached[name])
        and _tensor_sha256(cached[name]) == state["tensor_sha256"][name]
        for name in ("text_states", "visual_states", "text_mask", "visual_mask")
    }
    cached_outputs = _initial_outputs(program_model, cached, trigger, first_suffix)
    output_exact = {
        name: torch.equal(live_outputs[name], cached_outputs[name]) for name in live_outputs
    }
    if not all(tensor_exact.values()) or not all(output_exact.values()):
        raise RuntimeError(f"cache-to-live trigger representation parity failed for {uid}")

    replay_rows = []
    for row in rows:
        suffix = list(row["suffix_actions"])
        if all(action == "FULL" for action in suffix):
            generated_ids, generated_text, score = dense_ids, dense_text, dense_score
        else:
            routed = capture_four_action_suffix_from_full_baseline(
                wrapped, baseline, trigger, suffix
            )
            generated_ids, generated_text, score = _train_generate(
                processor, wrapped, routed, inputs["input_ids"], sample
            )
            del routed
        expected = [int(token) for token in row["expected_generated_token_ids"]]
        token_parity = generated_ids == expected
        passed = token_parity and bool(score.correct)
        replay_rows.append({
            "schema_version": "polar_program_replay_v1",
            "contract_sha256": contract_sha256,
            "program_id": str(row["program_id"]),
            "uid": uid,
            "trigger_layer": trigger,
            "generated_token_ids": generated_ids,
            "generated_answer": generated_text,
            "lmms_metric": score.metric_name,
            "lmms_score": score.raw_score,
            "correct": bool(score.correct),
            "exact_token_parity": token_parity,
            "passed": passed,
        })
        if not passed:
            raise RuntimeError(f"P90 exact program replay failed: {row['program_id']}")

    return {
        "schema_version": "polar_uid_replay_v1",
        "contract_sha256": contract_sha256,
        "uid": uid,
        "trigger_layer": trigger,
        "state_file": str(state_path.relative_to(resolve_path(config["output_root"]))),
        "state_file_sha256": file_sha256(state_path),
        "dense_generated_token_ids": dense_ids,
        "dense_generated_answer": dense_text,
        "dense_lmms_correct": bool(dense_score.correct),
        "consumed_image_sha256": metadata["consumed_image_sha256"],
        "tensor_roundtrip_exact": tensor_exact,
        "cache_live_output_exact": output_exact,
        "live_output_sha256": {name: _tensor_sha256(value) for name, value in live_outputs.items()},
        "program_count": len(replay_rows),
        "program_replays": replay_rows,
        "elapsed_seconds": time.monotonic() - started,
    }


def smoke(config_path: Path, device_index: int) -> None:
    contract, output_root = verify_contract(config_path, verify_model=True)
    config = contract["static_config"]
    device = torch.device(f"cuda:{int(device_index)}")
    torch.cuda.set_device(device)
    configure_dense_determinism(int(config["seed"]), config["backend_settings"])
    processor, base, wrapped = _load_model(config, device)
    program_model, initialization = _program_model(config, device)
    program_model.eval()
    programs = read_jsonl(output_root / "corpus/p90_program_corpus_manifest.jsonl")

    candidates = sorted(
        programs,
        key=lambda row: sha256(f"smoke:{config['seed']}:{row['program_id']}".encode()).hexdigest(),
    )
    selected = []
    covered_actions: set[str] = set()
    covered_outcomes: set[str] = set()
    used_uids: set[str] = set()
    target_count = int(config["smoke"]["training_uids"])
    while len(selected) < target_count:
        best = max(
            (row for row in candidates if row["uid"] not in used_uids),
            key=lambda row: (
                len(set(row["suffix_actions"]) - covered_actions)
                + int(row["dense_outcome"] not in covered_outcomes),
                sha256(str(row["program_id"]).encode()).hexdigest(),
            ),
        )
        selected.append(best)
        used_uids.add(best["uid"])
        covered_actions.update(best["suffix_actions"])
        covered_outcomes.add(best["dense_outcome"])

    smoke_root = output_root / "smoke"
    results = []
    for row in selected:
        result = _replay_uid(
            config=config,
            contract_sha256=contract["contract_sha256"],
            processor=processor,
            wrapped=wrapped,
            program_model=program_model,
            rows=[row],
            state_path=smoke_root / "states" / f"{_uid_slug(row['uid'])}.pt",
        )
        results.append(result)

    checkpoint_path = smoke_root / "checkpoint_roundtrip.pt"
    atomic_torch(checkpoint_path, program_model.state_dict())
    reloaded, _ = _program_model(config, device)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    checkpoint_exact = all(
        torch.equal(value.cpu(), reloaded.state_dict()[name].cpu())
        for name, value in program_model.state_dict().items()
    )
    passed = (
        initialization["all_exact"]
        and checkpoint_exact
        and covered_actions == set(ACTION_NAMES)
        and covered_outcomes == {"C", "W"}
        and all(
            all(row["tensor_roundtrip_exact"].values())
            and all(row["cache_live_output_exact"].values())
            and all(item["passed"] for item in row["program_replays"])
            for row in results
        )
    )
    report = {
        "schema_version": "polar_program_smoke_v1",
        "contract_sha256": contract["contract_sha256"],
        "passed": passed,
        "uids": len(results),
        "actions_covered": sorted(covered_actions),
        "outcomes_covered": sorted(covered_outcomes),
        "stage2a_initialization_exact": initialization["all_exact"],
        "checkpoint_roundtrip_exact": checkpoint_exact,
        "same_layer_read_semantics": "validated_by_existing_executor_contract_and_non_FULL_runtime_replay",
        "results": results,
    }
    atomic_json(smoke_root / "implementation_smoke.json", report)
    _atomic_bytes(
        smoke_root / "implementation_smoke.md",
        (
            "# Phase-74 implementation smoke\n\n"
            f"- Passed: **{passed}**\n"
            f"- UIDs: {len(results)}\n"
            f"- Actions covered: {', '.join(sorted(covered_actions))}\n"
            f"- Stage2-A initialization exact: {initialization['all_exact']}\n"
            f"- Trigger cache/live parity: {all(all(x['cache_live_output_exact'].values()) for x in results)}\n"
            f"- Checkpoint round trip exact: {checkpoint_exact}\n"
        ).encode(),
    )
    print(json.dumps({key: report[key] for key in ("passed", "uids", "actions_covered")}, sort_keys=True))
    if not passed:
        raise RuntimeError("Phase-74 implementation smoke failed")
    del reloaded, program_model, wrapped, base, processor
    torch.cuda.empty_cache()


def replay_worker(config_path: Path, rank: int, resume: bool) -> None:
    contract, output_root = verify_contract(config_path, verify_model=True)
    config = contract["static_config"]
    if not read_json(output_root / "smoke/implementation_smoke.json").get("passed"):
        raise RuntimeError("implementation smoke has not passed")
    rank = int(rank)
    if rank < 0 or rank >= int(config["world_size"]):
        raise ValueError("worker rank is outside the four-GPU world")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    configure_dense_determinism(int(config["seed"]) + rank, config["backend_settings"])
    processor, base, wrapped = _load_model(config, device)
    program_model, _ = _program_model(config, device)
    program_model.eval()
    programs = read_jsonl(output_root / "corpus/p90_program_corpus_manifest.jsonl")
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in programs:
        by_uid[row["uid"]].append(row)
    schedule = [
        row for row in read_jsonl(output_root / "work/replay_manifest.jsonl")
        if int(row["worker_rank"]) == rank
    ]
    rank_root = output_root / f"work/replay/rank{rank:02d}"
    completed = 0
    for index, item in enumerate(schedule, 1):
        uid = item["uid"]
        result_path = rank_root / f"{_uid_slug(uid)}.json"
        state_path = output_root / "features/trigger_states" / f"{_uid_slug(uid)}.pt"
        if resume and result_path.exists() and state_path.exists():
            old = read_json(result_path)
            if (
                old.get("contract_sha256") == contract["contract_sha256"]
                and int(old.get("program_count", -1)) == len(by_uid[uid])
                and old.get("state_file_sha256") == file_sha256(state_path)
            ):
                completed += 1
                continue
        result = _replay_uid(
            config=config,
            contract_sha256=contract["contract_sha256"],
            processor=processor,
            wrapped=wrapped,
            program_model=program_model,
            rows=by_uid[uid],
            state_path=state_path,
        )
        atomic_json(result_path, result)
        completed += 1
        if index % 10 == 0 or index == len(schedule):
            print(json.dumps({"rank": rank, "completed": completed, "total": len(schedule)}), flush=True)
    atomic_json(rank_root / "complete.json", {
        "schema_version": "polar_replay_rank_complete_v1",
        "contract_sha256": contract["contract_sha256"],
        "rank": rank, "expected_uids": len(schedule), "completed_uids": completed,
        "completed_at": utc_now(),
    })


def aggregate_replay(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    schedule = read_jsonl(output_root / "work/replay_manifest.jsonl")
    expected_uids = {row["uid"] for row in schedule}
    uid_results = []
    for item in schedule:
        rank = int(item["worker_rank"])
        path = output_root / f"work/replay/rank{rank:02d}/{_uid_slug(item['uid'])}.json"
        if not path.exists():
            raise RuntimeError(f"missing replay result for {item['uid']}")
        result = read_json(path)
        if result.get("contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError(f"replay contract differs for {item['uid']}")
        state_path = output_root / result["state_file"]
        if not state_path.exists() or file_sha256(state_path) != result["state_file_sha256"]:
            raise RuntimeError(f"trigger state file differs for {item['uid']}")
        if not all(result["tensor_roundtrip_exact"].values()) or not all(
            result["cache_live_output_exact"].values()
        ):
            raise RuntimeError(f"cache/live parity failed for {item['uid']}")
        uid_results.append(result)
    if len(uid_results) != len(expected_uids) or {row["uid"] for row in uid_results} != expected_uids:
        raise RuntimeError("global UID replay completion differs")

    replays = [item for row in uid_results for item in row["program_replays"]]
    expected_programs = {
        row["program_id"] for row in read_jsonl(output_root / "corpus/p90_program_corpus_manifest.jsonl")
    }
    observed = Counter(row["program_id"] for row in replays)
    if set(observed) != expected_programs or any(count != 1 for count in observed.values()):
        raise RuntimeError("global replay program completeness differs")
    if not all(row["passed"] for row in replays):
        raise RuntimeError("at least one program failed replay")
    atomic_jsonl(output_root / "corpus/replay_validation.jsonl", replays)
    feature_index = [
        {
            "uid": row["uid"], "trigger_layer": row["trigger_layer"],
            "state_file": row["state_file"], "state_file_sha256": row["state_file_sha256"],
            "contract_sha256": contract["contract_sha256"],
            "tensor_sha256": torch.load(
                output_root / row["state_file"], map_location="cpu", weights_only=False
            )["tensor_sha256"],
        }
        for row in uid_results
    ]
    atomic_jsonl(output_root / "features/trigger_state_index.jsonl", feature_index)
    atomic_json(output_root / "features/feature_schema.json", {
        "schema_version": "polar_trigger_state_schema_v1",
        "contract_sha256": contract["contract_sha256"],
        "state_timing": "pre_layer_entering_first_P90_trigger_action_after_all_FULL_prefix",
        "dtype": "bfloat16", "hidden_size": 3584,
        "values": ["all_text_token_states", "all_visual_token_states", "text_valid_mask", "visual_valid_mask"],
        "all_uid_cache_live_parity": True,
    })
    completion = {
        "schema_version": "polar_replay_completion_v1",
        "contract_sha256": contract["contract_sha256"],
        "uids": len(uid_results), "programs": len(replays),
        "all_exact_token_parity": all(row["exact_token_parity"] for row in replays),
        "all_correct": all(row["correct"] for row in replays),
        "all_uid_cache_live_parity": True,
        "replay_manifest_sha256": file_sha256(output_root / "corpus/replay_validation.jsonl"),
        "feature_index_sha256": file_sha256(output_root / "features/trigger_state_index.jsonl"),
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "work/replay_completion.json", completion)
    print(json.dumps(completion, sort_keys=True))


def _load_trigger_states(
    output_root: Path, contract_sha256: str
) -> dict[str, dict[str, torch.Tensor]]:
    output = {}
    for row in read_jsonl(output_root / "features/trigger_state_index.jsonl"):
        if row.get("contract_sha256") != contract_sha256:
            raise RuntimeError("trigger-state index contract differs")
        path = output_root / row["state_file"]
        if file_sha256(path) != row["state_file_sha256"]:
            raise RuntimeError(f"trigger-state file hash differs: {row['uid']}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("contract_sha256") != contract_sha256 or state.get("uid") != row["uid"]:
            raise RuntimeError(f"trigger-state provenance differs: {row['uid']}")
        for name, digest in row["tensor_sha256"].items():
            if _tensor_sha256(state[name]) != digest:
                raise RuntimeError(f"trigger-state tensor differs: {row['uid']} {name}")
        output[str(row["uid"])] = state
    return output


def _collate_programs(
    rows: Sequence[Mapping[str, Any]],
    states: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    max_text = max(int(states[row["uid"]]["text_states"].shape[1]) for row in rows)
    max_visual = max(int(states[row["uid"]]["visual_states"].shape[1]) for row in rows)
    max_suffix = max(len(row["suffix_action_indices"]) for row in rows)
    hidden = int(states[rows[0]["uid"]]["text_states"].shape[-1])
    text = torch.zeros(len(rows), max_text, hidden, dtype=torch.bfloat16, device=device)
    visual = torch.zeros(len(rows), max_visual, hidden, dtype=torch.bfloat16, device=device)
    text_mask = torch.zeros(len(rows), max_text, dtype=torch.bool, device=device)
    visual_mask = torch.zeros(len(rows), max_visual, dtype=torch.bool, device=device)
    targets = torch.full((len(rows), max_suffix), -100, dtype=torch.long, device=device)
    trigger = torch.empty(len(rows), dtype=torch.long, device=device)
    weights = torch.empty(len(rows), dtype=torch.float32, device=device)
    for index, row in enumerate(rows):
        state = states[str(row["uid"])]
        nt, nv = state["text_states"].shape[1], state["visual_states"].shape[1]
        ns = len(row["suffix_action_indices"])
        text[index, :nt] = state["text_states"][0].to(device)
        visual[index, :nv] = state["visual_states"][0].to(device)
        text_mask[index, :nt] = state["text_mask"][0].to(device)
        visual_mask[index, :nv] = state["visual_mask"][0].to(device)
        targets[index, :ns] = torch.tensor(row["suffix_action_indices"], device=device)
        trigger[index] = int(row["trigger_layer"])
        weights[index] = float(row["program_weight"])
    return {
        "text_states": text,
        "visual_states": visual,
        "text_mask": text_mask,
        "visual_mask": visual_mask,
        "targets": targets,
        "trigger_layers": trigger,
        "weights": weights,
    }


def _weighted_batches(
    rows: Sequence[Mapping[str, Any]], *, batch_size: int, seed: int
) -> list[list[Mapping[str, Any]]]:
    count = math.ceil(len(rows) / int(batch_size))
    batches: list[list[Mapping[str, Any]]] = [[] for _ in range(count)]
    masses = [0.0] * count
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["program_weight"]),
            sha256(f"{seed}:{row['program_id']}".encode()).hexdigest(),
        ),
    )
    for row in ranked:
        eligible = [index for index, batch in enumerate(batches) if len(batch) < int(batch_size)]
        selected = min(eligible, key=lambda index: (masses[index], index))
        batches[selected].append(row)
        masses[selected] += float(row["program_weight"])
    order = list(range(len(batches)))
    random.Random(int(seed)).shuffle(order)
    return [batches[index] for index in order]


def _assign_subset_weights(
    rows: Sequence[Mapping[str, Any]], minimum_cell_uids: int
) -> list[dict[str, Any]]:
    copied = [dict(row) for row in rows]
    weights = hierarchical_program_weights(copied, minimum_cell_uids=minimum_cell_uids)
    for row, weight in zip(copied, weights):
        row["program_weight"] = weight
    return copied


def _evaluate_program_nll(
    model: PolarSuffixProgramPredictor,
    rows: Sequence[Mapping[str, Any]],
    states: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            batch_rows = rows[start : start + int(batch_size)]
            batch = _collate_programs(batch_rows, states, device)
            context = model.compute_context(
                batch["text_states"], batch["visual_states"],
                batch["text_mask"], batch["visual_mask"],
            )
            logits = model.decode_teacher_forced(
                context, batch["trigger_layers"], batch["targets"]
            )
            _loss, per_program = weighted_program_loss(
                logits, batch["targets"], batch["weights"]
            )
            total += float((per_program * batch["weights"]).sum().item())
            total_weight += float(batch["weights"].sum().item())
    return total / total_weight


def _train_epoch(
    model: PolarSuffixProgramPredictor,
    optimizer: torch.optim.Optimizer,
    rows: Sequence[Mapping[str, Any]],
    states: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device,
    *,
    batch_size: int,
    seed: int,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    losses = []
    weight_masses = []
    for batch_rows in _weighted_batches(rows, batch_size=batch_size, seed=seed):
        batch = _collate_programs(batch_rows, states, device)
        optimizer.zero_grad(set_to_none=True)
        context = model.compute_context(
            batch["text_states"], batch["visual_states"],
            batch["text_mask"], batch["visual_mask"],
        )
        logits = model.decode_teacher_forced(
            context, batch["trigger_layers"], batch["targets"]
        )
        loss, _per_program = weighted_program_loss(logits, batch["targets"], batch["weights"])
        if not torch.isfinite(loss):
            raise RuntimeError("program training produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(gradient_clip_norm), error_if_nonfinite=True
        )
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise RuntimeError("program training produced a non-finite parameter")
        losses.append(float(loss.item()))
        weight_masses.append(float(batch["weights"].sum().item()))
    return {
        "batch_loss_mean": sum(losses) / len(losses),
        "batch_weight_mass_min": min(weight_masses),
        "batch_weight_mass_max": max(weight_masses),
    }


def _load_training_inputs(
    contract: Mapping[str, Any], output_root: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, torch.Tensor]]]:
    completion = read_json(output_root / "work/replay_completion.json")
    if (
        completion.get("contract_sha256") != contract["contract_sha256"]
        or not completion.get("all_uid_cache_live_parity")
        or not completion.get("all_exact_token_parity")
        or int(completion.get("programs", 0)) != 4948
    ):
        raise RuntimeError("full replay/cache gate has not passed")
    if file_sha256(output_root / "corpus/replay_validation.jsonl") != completion[
        "replay_manifest_sha256"
    ]:
        raise RuntimeError("replay validation changed after aggregation")
    programs = read_jsonl(output_root / "corpus/p90_program_corpus_manifest.jsonl")
    states = _load_trigger_states(output_root, contract["contract_sha256"])
    if {row["uid"] for row in programs} != set(states):
        raise RuntimeError("training program/state UID populations differ")
    return programs, states


def _program_distance(left: Sequence[int], right: Sequence[int]) -> int:
    if len(left) != len(right):
        raise ValueError("programs at one trigger must have equal lengths")
    return sum(int(a) != int(b) for a, b in zip(left, right))


def _write_internal_diagnostics(
    model: PolarSuffixProgramPredictor,
    rows: Sequence[Mapping[str, Any]],
    states: Mapping[str, Mapping[str, torch.Tensor]],
    output_root: Path,
    device: torch.device,
    beam_width: int,
) -> None:
    by_uid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_uid[row["uid"]].append(row)
    matches = []
    coherence = []
    for uid, uid_rows in sorted(by_uid.items()):
        state = states[uid]
        with torch.no_grad():
            context = model.compute_context(
                state["text_states"].to(device), state["visual_states"].to(device),
                state["text_mask"].to(device), state["visual_mask"].to(device),
            )
            beams = model.beam_decode(
                context, trigger_layer=int(uid_rows[0]["trigger_layer"]), beam_width=int(beam_width)
            )
        predicted = beams[0]["action_indices"]
        known = [list(row["suffix_action_indices"]) for row in uid_rows]
        distances = [_program_distance(predicted, target) for target in known]
        prefix = max(
            next((index for index, pair in enumerate(zip(predicted, target)) if pair[0] != pair[1]), len(predicted))
            for target in known
        )
        matches.append({
            "uid": uid, "dataset": uid_rows[0]["dataset"],
            "dense_outcome": uid_rows[0]["dense_outcome"], "known_programs": len(known),
            "exact_match_any": int(min(distances) == 0), "nearest_hamming_distance": min(distances),
            "maximum_prefix_match": prefix, "beam_top1_score": beams[0]["score"],
        })
        pairwise = [
            _program_distance(known[i], known[j])
            for i in range(len(known)) for j in range(i + 1, len(known))
        ]
        coherence.append({
            "uid": uid, "known_programs": len(known),
            "mean_pairwise_hamming": sum(pairwise) / len(pairwise) if pairwise else 0.0,
            "max_pairwise_hamming": max(pairwise) if pairwise else 0,
        })
    atomic_csv(output_root / "diagnostics/internal_program_match.csv", matches)
    atomic_csv(output_root / "diagnostics/nearest_valid_program_distance.csv", matches)
    atomic_csv(output_root / "diagnostics/multi_route_coherence.csv", coherence)


def train_internal_dev(config_path: Path, device_index: int) -> None:
    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    device = torch.device(f"cuda:{int(device_index)}")
    torch.cuda.set_device(device)
    configure_dense_determinism(int(config["seed"]), config["backend_settings"])
    programs, states = _load_training_inputs(contract, output_root)
    minimum = int(config["training"]["minimum_cell_uids"])
    train_rows = _assign_subset_weights(
        [row for row in programs if row["internal_split"] == "train"], minimum
    )
    dev_rows = _assign_subset_weights(
        [row for row in programs if row["internal_split"] == "dev"], minimum
    )
    model, initialization = _program_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    log_rows = []
    best_state = None
    for epoch in range(1, int(config["training"]["maximum_epochs"]) + 1):
        started = time.monotonic()
        train_step = _train_epoch(
            model, optimizer, train_rows, states, device,
            batch_size=int(config["training"]["batch_size"]),
            seed=int(config["seed"]) + epoch,
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
        )
        train_nll = _evaluate_program_nll(
            model, train_rows, states, device, int(config["training"]["batch_size"])
        )
        dev_nll = _evaluate_program_nll(
            model, dev_rows, states, device, int(config["training"]["batch_size"])
        )
        row = {
            "epoch": epoch, "train_weighted_program_nll": train_nll,
            "dev_weighted_program_nll": dev_nll, "elapsed_seconds": time.monotonic() - started,
            **train_step,
        }
        log_rows.append(row)
        atomic_jsonl(output_root / "training/internal_dev_train_log.jsonl", log_rows)
        if best_state is None or dev_nll < min(x["dev_weighted_program_nll"] for x in log_rows[:-1]):
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(json.dumps(row, sort_keys=True), flush=True)

    selected = min(log_rows, key=lambda row: (row["dev_weighted_program_nll"], row["epoch"]))
    if selected["epoch"] != next(
        row["epoch"] for row in log_rows if row["dev_weighted_program_nll"] == selected["dev_weighted_program_nll"]
    ):
        raise AssertionError("epoch tie did not select the earliest minimum")
    model.load_state_dict(best_state, strict=True)
    atomic_torch(output_root / "training/internal_dev_selected_checkpoint.pt", {
        "schema_version": "polar_internal_dev_checkpoint_v1",
        "contract_sha256": contract["contract_sha256"], "selected_epoch": selected["epoch"],
        "state_dict": best_state,
    })
    atomic_json(output_root / "training/frozen_epoch_selection.json", {
        "contract_sha256": contract["contract_sha256"],
        "criterion": config["training"]["epoch_selection"],
        "selected_epoch": selected["epoch"],
        "selected_dev_nll": selected["dev_weighted_program_nll"],
        "maximum_epochs": config["training"]["maximum_epochs"],
    })
    _write_internal_diagnostics(
        model, dev_rows, states, output_root, device, int(config["program_model"]["beam_width"])
    )
    parameter_count = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "qwen_trainable": 0, "stage1_trainable": 0,
    }
    atomic_json(output_root / "model/parameter_count.json", parameter_count)
    _atomic_bytes(output_root / "model/initialization_report.md", (
        "# Initialization report\n\n"
        f"- Stage2-A compatible tensors exact: **{initialization['all_exact']}**\n"
        f"- Copied tensors: {len(initialization['copied_tensors'])}\n"
        "- Randomly initialized: absolute/relative/action embeddings, two decoder blocks, output head.\n"
    ).encode())
    _atomic_bytes(output_root / "training/internal_dev_summary.md", (
        "# Internal development summary\n\n"
        f"- Train programs: {len(train_rows)}\n- Dev programs: {len(dev_rows)}\n"
        f"- Selected epoch: **{selected['epoch']}**\n"
        f"- Selected weighted dev NLL: {selected['dev_weighted_program_nll']:.8f}\n"
        "- This split was used only to freeze epoch count; no architecture or benchmark tuning occurred.\n"
    ).encode())


def train_full_refit(config_path: Path, device_index: int) -> None:
    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    selection = read_json(output_root / "training/frozen_epoch_selection.json")
    if selection.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("epoch selection contract differs")
    selected_epoch = int(selection["selected_epoch"])
    device = torch.device(f"cuda:{int(device_index)}")
    torch.cuda.set_device(device)
    configure_dense_determinism(int(config["seed"]), config["backend_settings"])
    programs, states = _load_training_inputs(contract, output_root)
    programs = _assign_subset_weights(
        programs, int(config["training"]["minimum_cell_uids"])
    )
    model, initialization = _program_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    log_rows = []
    for epoch in range(1, selected_epoch + 1):
        started = time.monotonic()
        train_step = _train_epoch(
            model, optimizer, programs, states, device,
            batch_size=int(config["training"]["batch_size"]),
            seed=int(config["seed"]) + epoch,
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
        )
        nll = _evaluate_program_nll(
            model, programs, states, device, int(config["training"]["batch_size"])
        )
        row = {"epoch": epoch, "full_weighted_program_nll": nll,
               "elapsed_seconds": time.monotonic() - started, **train_step}
        log_rows.append(row)
        atomic_jsonl(output_root / "training/full_refit_train_log.jsonl", log_rows)
        print(json.dumps(row, sort_keys=True), flush=True)
    checkpoint_path = output_root / "training/full_refit_checkpoint.pt"
    atomic_torch(checkpoint_path, {
        "schema_version": "polar_suffix_program_checkpoint_v1",
        "contract_sha256": contract["contract_sha256"],
        "selected_epoch": selected_epoch,
        "initialization_exact": initialization["all_exact"],
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    })
    manifest = {
        "schema_version": "polar_suffix_program_checkpoint_manifest_v1",
        "contract_sha256": contract["contract_sha256"],
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "selected_epoch": selected_epoch,
        "programs": len(programs), "uids": len({row["uid"] for row in programs}),
        "replay_manifest_sha256": read_json(output_root / "work/replay_completion.json")[
            "replay_manifest_sha256"
        ],
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "training/full_refit_checkpoint_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


class ProgramEvalRuntime:
    def __init__(self, config: Mapping[str, Any], device_index: int):
        from experiments.run_full_benchmark_end_to_end_eval import Runtime

        self.phase74_config = config
        self.eval_config = read_json(config["evaluation"]["base_config"])
        self.base_runtime = Runtime(self.eval_config, int(device_index))
        self.device = self.base_runtime.device
        self.program, self.initialization = _program_model(config, self.device)
        manifest = read_json(resolve_path(config["output_root"]) / "training/full_refit_checkpoint_manifest.json")
        checkpoint_path = resolve_path(manifest["checkpoint"])
        if file_sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
            raise RuntimeError("full-refit checkpoint hash differs")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("contract_sha256") != manifest["contract_sha256"]:
            raise RuntimeError("full-refit checkpoint contract differs")
        self.program.load_state_dict(checkpoint["state_dict"], strict=True)
        self.program.eval()


def prepare_evaluation(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    checkpoint = read_json(output_root / "training/full_refit_checkpoint_manifest.json")
    if checkpoint.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("full-refit checkpoint has not completed under this contract")
    if file_sha256(checkpoint["checkpoint"]) != checkpoint["checkpoint_sha256"]:
        raise RuntimeError("full-refit checkpoint hash differs")
    phase69 = read_json(config["evaluation"]["phase69_contract"])
    if phase69.get("contract_sha256") != "63379eeff80fd5b046cb980cdfccea7fab232e15eb5b14924a24b60393327e83":
        raise RuntimeError("Phase-69 evaluation contract differs")
    source_manifest = resolve_path("analysis/dense_failure_stage2/full_benchmark_eval/manifests/all_full_manifest.jsonl")
    expected_manifest_hash = phase69["prepared_manifest_sha256"]["manifests/all_full_manifest.jsonl"]
    if file_sha256(source_manifest) != expected_manifest_hash:
        raise RuntimeError("Phase-69 prepared evaluation manifest differs")
    rows = read_jsonl(source_manifest)
    if len(rows) != int(config["evaluation"]["expected_total"]):
        raise RuntimeError("full evaluation population count differs")
    paired = read_jsonl(config["evaluation"]["phase69_paired"])
    if len(paired) != len(rows) or {row["uid"] for row in paired} != {row["uid"] for row in rows}:
        raise RuntimeError("Phase-69 paired baseline population differs")
    schedule = []
    for index, row in enumerate(rows):
        copy = dict(row)
        copy["worker_rank"] = index % int(config["world_size"])
        schedule.append(copy)
    atomic_jsonl(output_root / "work/evaluation_manifest.jsonl", schedule)
    frozen = {
        "schema_version": "polar_full_evaluation_input_v1",
        "contract_sha256": contract["contract_sha256"],
        "phase69_contract_sha256": phase69["contract_sha256"],
        "phase69_manifest_sha256": expected_manifest_hash,
        "phase69_paired_sha256": file_sha256(config["evaluation"]["phase69_paired"]),
        "evaluation_manifest_sha256": file_sha256(output_root / "work/evaluation_manifest.jsonl"),
        "full_refit_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "rows": len(rows),
        "prepared_at": utc_now(),
    }
    atomic_json(output_root / "work/evaluation_input.json", frozen)
    _atomic_bytes(output_root / "decoding/decoding_contract.md", (
        "# Decoding contract\n\n"
        "At the first robust Stage-1 P90 trigger, the predictor consumes the exact all-FULL incoming trigger state once. "
        "It decodes the entire fixed-length suffix with beam width 8 and sum of action log probabilities before any suffix action executes. "
        "Greedy is logged for diagnostics only. No oracle, MCTS, reranking, or intervention penalty is used.\n"
    ).encode())
    print(json.dumps(frozen, sort_keys=True))


@torch.inference_mode()
def _process_eval_row(
    runtime: ProgramEvalRuntime,
    row: Mapping[str, Any],
    phase69_row: Mapping[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    from experiments.run_full_benchmark_end_to_end_eval import (
        _build_inputs as eval_build_inputs,
        _generate as eval_generate,
        _native_dense_generation,
        _stage1_scores,
        _verify_images,
    )

    started = time.monotonic()
    dense = _native_dense_generation(runtime.base_runtime, row, extract_features=True)
    features = dense.pop("features")
    scores, trigger = _stage1_scores(runtime.base_runtime, features)
    del features
    parity = {
        "dense_generated_token_ids": dense["generated_token_ids"]
        == phase69_row["dense_generated_token_ids"],
        "dense_score": float(dense["score"]) == float(phase69_row["dense_score"]),
        "dense_correct": bool(dense["correct"]) == bool(phase69_row["dense_correct"]),
        "stage1_scores": scores == [float(value) for value in phase69_row["stage1_scores"]],
        "trigger_layer": trigger == phase69_row["trigger_layer"],
    }
    if not all(parity.values()):
        raise RuntimeError(f"Phase-69 Dense/Stage1 reuse parity failed for {row['uid']}: {parity}")

    beam_rows: list[dict[str, Any]] = []
    greedy_suffix: list[int] = []
    if trigger is None:
        suffix = []
        actions = ["FULL"] * 28
        routed = {key: dense[key] for key in (
            "generated_token_ids", "generated_answer", "metric", "score", "threshold", "correct"
        )}
    else:
        consumed = _verify_images(row)
        if consumed != dense["consumed_image_sha256s"]:
            raise RuntimeError("image bytes changed between native and program inference")
        inputs = eval_build_inputs(runtime.base_runtime, row)
        prepared = build_binary_inputs(runtime.base_runtime.wrapped, inputs)
        baseline = capture_four_action_route(
            runtime.base_runtime.wrapped,
            {},
            ["FULL"] * 28,
            prepared_inputs=prepared,
            use_cache=True,
            native_full_rows=True,
        )
        text, visual = baseline.pre_layer_states[int(trigger)]
        context = runtime.program.compute_context(
            text, visual, prepared.text_valid_mask, prepared.visual_valid_mask
        )
        beam_rows = runtime.program.beam_decode(
            context,
            trigger_layer=int(trigger),
            beam_width=int(runtime.phase74_config["program_model"]["beam_width"]),
        )
        greedy_suffix = runtime.program.greedy_decode(context, trigger_layer=int(trigger))
        suffix = list(beam_rows[0]["actions"])
        actions = ["FULL"] * int(trigger) + suffix
        if all(action == "FULL" for action in suffix):
            routed = {key: dense[key] for key in (
                "generated_token_ids", "generated_answer", "metric", "score", "threshold", "correct"
            )}
        else:
            output = capture_four_action_suffix_from_full_baseline(
                runtime.base_runtime.wrapped, baseline, int(trigger), suffix
            )
            routed = eval_generate(
                runtime.base_runtime, output, inputs["input_ids"], row
            )
            del output
        del baseline, prepared, inputs
    nonfull = [index for index, action in enumerate(actions) if action != "FULL"]
    active = actions[int(trigger) :] if trigger is not None else []
    dense_correct = bool(dense["correct"])
    program_correct = bool(routed["correct"])
    return {
        "schema_version": "polar_full_eval_paired_row_v1",
        "contract_sha256": contract_sha256,
        "uid": str(row["uid"]), "sample_id": str(row["sample_id"]),
        "benchmark": str(row["benchmark"]), "benchmark_family": str(row["benchmark_family"]),
        "image_group_id": str(row["image_group_id"]),
        "metric_name": str(row["metric_name"]), "correctness_threshold": float(row["correctness_threshold"]),
        "answer": row["answer"], "all_answer_norms": row.get("all_answer_norms"),
        "dense_generated_answer": dense["generated_answer"],
        "dense_generated_token_ids": dense["generated_token_ids"],
        "dense_score": dense["score"], "dense_correct": dense_correct,
        "sequential_generated_answer": phase69_row["routed_generated_answer"],
        "sequential_generated_token_ids": phase69_row["routed_generated_token_ids"],
        "sequential_score": phase69_row["routed_score"],
        "sequential_correct": bool(phase69_row["routed_correct"]),
        "sequential_actions": phase69_row["actions"],
        "program_generated_answer": routed["generated_answer"],
        "program_generated_token_ids": routed["generated_token_ids"],
        "program_score": routed["score"], "program_correct": program_correct,
        "program_transition": ("C" if dense_correct else "W") + "→" + ("C" if program_correct else "W"),
        "sequential_transition": phase69_row["transition"],
        "stage1_scores": scores, "stage1_threshold": runtime.phase74_config["stage1"]["threshold"],
        "triggered": trigger is not None, "trigger_layer": trigger,
        "program_actions": actions, "program_suffix_actions": suffix,
        "greedy_suffix_action_indices": greedy_suffix,
        "greedy_suffix_actions": [ACTION_NAMES[index] for index in greedy_suffix],
        "beam_rows": beam_rows,
        "beam_top1_score": beam_rows[0]["score"] if beam_rows else None,
        "beam_top2_margin": (beam_rows[0]["score"] - beam_rows[1]["score"]) if len(beam_rows) > 1 else None,
        "post_trigger_action_counts": dict(Counter(active)),
        "non_full_count": len(nonfull), "any_non_full": bool(nonfull),
        "first_non_full_layer": nonfull[0] if nonfull else None,
        "trigger_to_first_non_full_delay": nonfull[0] - int(trigger) if nonfull and trigger is not None else None,
        "read_enabled_layers": sum(action in {"FULL", "READ_ONLY"} for action in actions),
        "write_enabled_layers": sum(action in {"FULL", "WRITE_ONLY"} for action in actions),
        "phase69_reuse_parity": parity,
        "consumed_image_sha256s": dense["consumed_image_sha256s"],
        "prompt_tokens": dense["prompt_tokens"], "visual_tokens": dense["visual_tokens"],
        "elapsed_seconds": time.monotonic() - started,
    }


def evaluation_worker(config_path: Path, rank: int, resume: bool) -> None:
    contract, output_root = verify_contract(config_path, verify_model=True)
    config = contract["static_config"]
    frozen = read_json(output_root / "work/evaluation_input.json")
    if frozen.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("evaluation input has not been frozen")
    if file_sha256(output_root / "work/evaluation_manifest.jsonl") != frozen[
        "evaluation_manifest_sha256"
    ]:
        raise RuntimeError("evaluation manifest changed after freeze")
    rank = int(rank)
    rows = [
        row for row in read_jsonl(output_root / "work/evaluation_manifest.jsonl")
        if int(row["worker_rank"]) == rank
    ]
    phase69 = {row["uid"]: row for row in read_jsonl(config["evaluation"]["phase69_paired"])}
    runtime = ProgramEvalRuntime(config, rank)
    rank_root = output_root / f"work/evaluation/rank{rank:02d}"
    chunk_size = 16
    completed = 0
    for part_index, start in enumerate(range(0, len(rows), chunk_size)):
        current = rows[start : start + chunk_size]
        path = rank_root / f"part_{part_index:05d}.jsonl"
        if resume and path.exists():
            old = read_jsonl(path)
            if (
                len(old) == len(current)
                and [row["uid"] for row in old] == [row["uid"] for row in current]
                and all(row.get("contract_sha256") == contract["contract_sha256"] for row in old)
            ):
                completed += len(old)
                continue
        results = []
        for row in current:
            results.append(_process_eval_row(
                runtime, row, phase69[row["uid"]], contract["contract_sha256"]
            ))
        atomic_jsonl(path, results)
        completed += len(results)
        if part_index % 10 == 0 or completed == len(rows):
            print(json.dumps({"rank": rank, "completed": completed, "total": len(rows)}), flush=True)
    atomic_json(rank_root / "complete.json", {
        "schema_version": "polar_eval_rank_complete_v1",
        "contract_sha256": contract["contract_sha256"], "rank": rank,
        "expected_rows": len(rows), "completed_rows": completed, "completed_at": utc_now(),
    })


def _scope_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    output = {"overall": list(rows)}
    for family in ("chartqa", "textvqa", "mmmu_pro", "pope"):
        output[family] = [row for row in rows if row["benchmark_family"] == family]
    return output


def _method_summary(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    correct_key = f"{method}_correct"
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
        "dense_correct": dense_correct,
        "method_correct": method_correct,
        "dense_accuracy": dense_correct / n,
        "method_accuracy": method_correct / n,
        "delta_accuracy": (method_correct - dense_correct) / n,
        "w_to_c": transitions["W→C"], "c_to_w": transitions["C→W"],
        "c_to_c": transitions["C→C"], "w_to_w": transitions["W→W"],
        "net": transitions["W→C"] - transitions["C→W"],
    }


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _segments(actions: Sequence[str]) -> int:
    count = 0
    active = False
    for action in actions:
        current = action != "FULL"
        if current and not active:
            count += 1
        active = current
    return count


def _write_figures(output_root: Path, scopes: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    families = ["chartqa", "textvqa", "mmmu_pro", "pope", "overall"]
    labels = ["ChartQA", "TextVQA", "MMMU-Pro", "POPE", "Overall"]
    dense = [sum(row["dense_correct"] for row in scopes[key]) / len(scopes[key]) for key in families]
    sequential = [sum(row["sequential_correct"] for row in scopes[key]) / len(scopes[key]) for key in families]
    program = [sum(row["program_correct"] for row in scopes[key]) / len(scopes[key]) for key in families]
    x = np.arange(len(families))
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x - 0.25, dense, 0.25, label="Dense")
    axis.bar(x, sequential, 0.25, label="Sequential-A")
    axis.bar(x + 0.25, program, 0.25, label="Program")
    axis.set_xticks(x, labels, rotation=20); axis.set_ylabel("LMMS correctness"); axis.legend()
    figure.tight_layout(); figure.savefig(output_root / "figures/benchmark_accuracy_comparison.png", dpi=160); plt.close(figure)

    w2c = [sum(row["program_transition"] == "W→C" for row in scopes[key]) for key in families]
    c2w = [sum(row["program_transition"] == "C→W" for row in scopes[key]) for key in families]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x - 0.18, w2c, 0.36, label="W→C"); axis.bar(x + 0.18, c2w, 0.36, label="C→W")
    axis.set_xticks(x, labels, rotation=20); axis.set_ylabel("Samples"); axis.legend()
    figure.tight_layout(); figure.savefig(output_root / "figures/rescue_regression_comparison.png", dpi=160); plt.close(figure)

    triggered = [sum(row["triggered"] for row in scopes[key]) for key in families]
    nonfull = [sum(row["any_non_full"] for row in scopes[key]) for key in families]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x - 0.18, triggered, 0.36, label="Stage-1 triggered"); axis.bar(x + 0.18, nonfull, 0.36, label="Program non-FULL")
    axis.set_xticks(x, labels, rotation=20); axis.set_ylabel("Samples"); axis.legend()
    figure.tight_layout(); figure.savefig(output_root / "figures/stage1_stage2_funnel.png", dpi=160); plt.close(figure)

    triggered_rows = [row for row in scopes["overall"] if row["triggered"]]
    nonfull_counts = [row["non_full_count"] for row in triggered_rows]
    delays = [row["trigger_to_first_non_full_delay"] for row in triggered_rows if row["trigger_to_first_non_full_delay"] is not None]
    figure, axis = plt.subplots(figsize=(7, 4)); axis.hist(nonfull_counts, bins=range(0, 30)); axis.set_xlabel("Non-FULL actions"); axis.set_ylabel("Triggered samples")
    figure.tight_layout(); figure.savefig(output_root / "figures/nonfull_program_length.png", dpi=160); plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4)); axis.hist(delays, bins=range(0, 29)); axis.set_xlabel("Trigger-to-first-intervention delay"); axis.set_ylabel("Samples")
    figure.tight_layout(); figure.savefig(output_root / "figures/trigger_to_first_intervention.png", dpi=160); plt.close(figure)

    action_counts = Counter(action for row in triggered_rows for action in row["program_suffix_actions"])
    figure, axis = plt.subplots(figsize=(7, 4)); axis.bar(ACTION_NAMES, [action_counts[action] for action in ACTION_NAMES]); axis.set_ylabel("Post-trigger actions")
    figure.tight_layout(); figure.savefig(output_root / "figures/action_usage_by_benchmark.png", dpi=160); plt.close(figure)
    internal = []
    path = output_root / "diagnostics/internal_program_match.csv"
    if path.exists():
        with path.open() as handle:
            internal = list(csv.DictReader(handle))
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist([int(row["nearest_hamming_distance"]) for row in internal], bins=range(0, 30))
    axis.set_xlabel("Nearest known-program Hamming distance"); axis.set_ylabel("Internal-dev UIDs")
    figure.tight_layout(); figure.savefig(output_root / "figures/program_coherence.png", dpi=160); plt.close(figure)


def aggregate_evaluation(config_path: Path) -> None:
    from dense_failure_stage2.full_benchmark_eval import paired_bootstrap

    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    frozen = read_json(output_root / "work/evaluation_input.json")
    manifest = read_jsonl(output_root / "work/evaluation_manifest.jsonl")
    rows = []
    for rank in range(int(config["world_size"])):
        completion = read_json(output_root / f"work/evaluation/rank{rank:02d}/complete.json")
        if (
            completion.get("contract_sha256") != contract["contract_sha256"]
            or int(completion["completed_rows"]) != int(completion["expected_rows"])
        ):
            raise RuntimeError(f"evaluation rank {rank} is incomplete")
        rank_root = output_root / f"work/evaluation/rank{rank:02d}"
        for path in sorted(rank_root.glob("part_*.jsonl")):
            rows.extend(read_jsonl(path))
    expected_uids = [row["uid"] for row in manifest]
    observed = Counter(row["uid"] for row in rows)
    if len(rows) != frozen["rows"] or set(observed) != set(expected_uids) or any(
        count != 1 for count in observed.values()
    ):
        raise RuntimeError("global full-evaluation completion differs")
    by_uid = {row["uid"]: row for row in rows}
    rows = [by_uid[uid] for uid in expected_uids]
    if not all(all(row["phase69_reuse_parity"].values()) for row in rows):
        raise RuntimeError("Sequential-A reuse parity was not exact")
    scopes = _scope_rows(rows)

    for family in ("chartqa", "textvqa", "mmmu_pro", "pope"):
        atomic_jsonl(output_root / f"evaluation/{family}/paired_results.jsonl", scopes[family])
    atomic_jsonl(output_root / "evaluation/all_paired.jsonl", rows)

    summaries = []
    transitions = []
    bootstraps = []
    for scope, current in scopes.items():
        seq = _method_summary(current, "sequential")
        program = _method_summary(current, "program")
        summaries.append({
            "scope": scope, "n": len(current), "dense_accuracy": program["dense_accuracy"],
            "sequential_accuracy": seq["method_accuracy"], "program_accuracy": program["method_accuracy"],
            "program_minus_dense": program["delta_accuracy"],
            "program_minus_sequential": program["method_accuracy"] - seq["method_accuracy"],
            "program_w_to_c": program["w_to_c"], "program_c_to_w": program["c_to_w"],
            "program_net": program["net"], "sequential_w_to_c": seq["w_to_c"],
            "sequential_c_to_w": seq["c_to_w"], "sequential_net": seq["net"],
        })
        for method, result in (("sequential", seq), ("program", program)):
            transitions.append({"scope": scope, "method": method, **{
                key: result[key] for key in ("w_to_c", "c_to_w", "c_to_c", "w_to_w", "net")
            }})
        bootstrap_rows = [
            {"dense_correct": row["dense_correct"], "routed_correct": row["program_correct"],
             "transition": row["program_transition"]}
            for row in current
        ]
        for item in paired_bootstrap(
            bootstrap_rows,
            draws=int(config["evaluation"]["paired_bootstrap_draws"]),
            seed=int(config["evaluation"]["paired_bootstrap_seed"]) + len(bootstraps),
        ):
            bootstraps.append({"scope": scope, **item})
    atomic_csv(output_root / "metrics/full_benchmark_summary.csv", summaries)
    atomic_csv(output_root / "metrics/transition_counts.csv", transitions)
    atomic_csv(output_root / "metrics/paired_bootstrap.csv", bootstraps)

    funnel = []
    action_usage = []
    program_stats = []
    dense_c_rows = []
    treatment_rows = []
    for scope, current in scopes.items():
        dense_c = [row for row in current if row["dense_correct"]]
        dense_w = [row for row in current if not row["dense_correct"]]
        triggered = [row for row in current if row["triggered"]]
        triggered_c = [row for row in dense_c if row["triggered"]]
        triggered_w = [row for row in dense_w if row["triggered"]]
        any_nonfull = [row for row in triggered if row["any_non_full"]]
        funnel.append({
            "scope": scope, "n": len(current), "dense_c": len(dense_c), "dense_w": len(dense_w),
            "triggered_c": len(triggered_c), "triggered_w": len(triggered_w),
            "p_trigger_given_c": _safe_rate(len(triggered_c), len(dense_c)),
            "p_trigger_given_w": _safe_rate(len(triggered_w), len(dense_w)),
            "predicted_all_full": len(triggered) - len(any_nonfull), "predicted_any_nonfull": len(any_nonfull),
            "triggered_w_any_nonfull": sum(row["any_non_full"] for row in triggered_w),
            "triggered_c_any_nonfull": sum(row["any_non_full"] for row in triggered_c),
            "w_to_c": sum(row["program_transition"] == "W→C" for row in current),
            "c_to_w": sum(row["program_transition"] == "C→W" for row in current),
        })
        counts = Counter(action for row in triggered for action in row["program_suffix_actions"])
        for action in ACTION_NAMES:
            action_usage.append({"scope": scope, "action": action, "count": counts[action]})
        nonfull_counts = [row["non_full_count"] for row in triggered]
        program_stats.append({
            "scope": scope, "triggered": len(triggered),
            "all_full_rate": _safe_rate(sum(not row["any_non_full"] for row in triggered), len(triggered)),
            "mean_non_full": sum(nonfull_counts) / len(nonfull_counts) if nonfull_counts else None,
            "median_non_full": sorted(nonfull_counts)[len(nonfull_counts)//2] if nonfull_counts else None,
            "mean_non_full_segments": (
                sum(_segments(row["program_suffix_actions"]) for row in triggered) / len(triggered)
                if triggered else None
            ),
            "mean_read_off_layers": (
                sum(28 - row["read_enabled_layers"] for row in triggered) / len(triggered)
                if triggered else None
            ),
            "mean_write_off_layers": (
                sum(28 - row["write_enabled_layers"] for row in triggered) / len(triggered)
                if triggered else None
            ),
        })
        nonfull_c = [row for row in triggered_c if row["any_non_full"]]
        dense_c_rows.append({
            "scope": scope, "triggered_dense_c": len(triggered_c),
            "all_full": sum(not row["any_non_full"] for row in triggered_c),
            "any_non_full": len(nonfull_c),
            "c_to_w": sum(row["program_transition"] == "C→W" for row in triggered_c),
            "c_to_w_given_non_full": _safe_rate(
                sum(row["program_transition"] == "C→W" for row in nonfull_c), len(nonfull_c)
            ),
            "c_to_c": sum(row["program_transition"] == "C→C" for row in triggered_c),
        })
        nonfull_w = [row for row in triggered_w if row["any_non_full"]]
        treatment_rows.append({
            "scope": scope, "triggered_dense_w": len(triggered_w),
            "all_full": sum(not row["any_non_full"] for row in triggered_w),
            "any_non_full": len(nonfull_w),
            "w_to_c": sum(row["program_transition"] == "W→C" for row in triggered_w),
            "w_to_c_given_non_full": _safe_rate(
                sum(row["program_transition"] == "W→C" for row in nonfull_w), len(nonfull_w)
            ),
        })
    atomic_csv(output_root / "metrics/stage1_stage2_funnel.csv", funnel)
    atomic_csv(output_root / "metrics/action_usage.csv", action_usage)
    atomic_csv(output_root / "metrics/program_statistics.csv", program_stats)
    atomic_csv(output_root / "metrics/dense_c_preservation.csv", dense_c_rows)
    atomic_csv(output_root / "metrics/triggered_w_treatment.csv", treatment_rows)

    trigger_rows = []
    for (family, layer), current in sorted(
        defaultdict(list, {
            key: [row for row in rows if row["benchmark_family"] == key[0] and row["trigger_layer"] == key[1]]
            for key in {(row["benchmark_family"], row["trigger_layer"]) for row in rows if row["triggered"]}
        }).items()
    ):
        trigger_rows.append({
            "benchmark_family": family, "trigger_layer": layer, "n": len(current),
            "w_to_c": sum(row["program_transition"] == "W→C" for row in current),
            "c_to_w": sum(row["program_transition"] == "C→W" for row in current),
            "any_non_full": sum(row["any_non_full"] for row in current),
        })
    atomic_csv(output_root / "metrics/trigger_layer_breakdown.csv", trigger_rows)
    benchmark_rows = []
    for benchmark in sorted({row["benchmark"] for row in rows}):
        current = [row for row in rows if row["benchmark"] == benchmark]
        seq, prog = _method_summary(current, "sequential"), _method_summary(current, "program")
        benchmark_rows.append({
            "benchmark": benchmark, "n": len(current), "dense_accuracy": prog["dense_accuracy"],
            "sequential_accuracy": seq["method_accuracy"], "program_accuracy": prog["method_accuracy"],
            "w_to_c": prog["w_to_c"], "c_to_w": prog["c_to_w"], "net": prog["net"],
        })
    atomic_csv(output_root / "metrics/benchmark_breakdown.csv", benchmark_rows)
    atomic_csv(output_root / "diagnostics/beam_statistics.csv", [
        {"uid": row["uid"], "benchmark_family": row["benchmark_family"],
         "trigger_layer": row["trigger_layer"], "beam_top1_score": row["beam_top1_score"],
         "beam_top2_margin": row["beam_top2_margin"],
         "greedy_equals_beam": row["greedy_suffix_actions"] == row["program_suffix_actions"]}
        for row in rows if row["triggered"]
    ])
    _write_figures(output_root, scopes)

    overall = next(row for row in summaries if row["scope"] == "overall")
    stronger = overall["program_net"] > 0 and overall["program_accuracy"] >= overall["dense_accuracy"]
    primary = overall["program_net"] > 0
    if stronger:
        outcome = "A: positive pooled Net with dense accuracy recovery/improvement"
        recommendation = "Retain complete suffix-program supervision as the leading Stage-2 formulation for the next separately authorized comparison."
    elif primary:
        outcome = "A-partial: positive pooled Net without full dense-accuracy recovery"
        recommendation = "Preserve the positive program result, but do not promote it over Dense until uncertainty and remaining regressions are addressed."
    elif overall["program_c_to_w"] < overall["sequential_c_to_w"]:
        outcome = "B: fewer regressions but insufficient rescues"
        recommendation = "Keep program supervision as a conservative formulation; treatment transfer remains the unresolved bottleneck."
    elif any(row["any_non_full"] for row in rows if row["triggered"]):
        outcome = "C: interventions occurred without positive rescue-regression balance"
        recommendation = "Do not deploy this checkpoint; complete-program supervision alone did not solve treatment generalization."
    else:
        outcome = "D: almost-always all-FULL behavior"
        recommendation = "Do not deploy this checkpoint; inspect W-program likelihood before any separately authorized redesign."
    corpus_summary = next(csv.DictReader((output_root / "corpus/corpus_summary.csv").open()))
    internal_rows = list(csv.DictReader((output_root / "diagnostics/internal_program_match.csv").open()))
    internal_match = _safe_rate(sum(int(row["exact_match_any"]) for row in internal_rows), len(internal_rows))
    pope = next(row for row in summaries if row["scope"] == "pope")
    mmmu = next(row for row in summaries if row["scope"] == "mmmu_pro")
    summary_md = f"""# POLAR suffix-program full evaluation summary

## Result

- Outcome: **{outcome}**
- Full training population: {corpus_summary['uids']} UIDs / {corpus_summary['programs']} programs.
- Provenance: preservation {corpus_summary['preservation']}; single {corpus_summary['single']}; original MCTS {corpus_summary['original_mcts']}; robust search {corpus_summary['robust_search']}; completeness audit {corpus_summary['completeness_audit']}.
- Full-corpus refit completed: **yes**.
- Full external evaluation completed: **{len(rows):,} / {config['evaluation']['expected_total']:,}** with exact Phase-69 Dense/Stage1 reuse parity.

| Scope | Dense | Sequential-A | Program | W→C | C→W | Net |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {row['scope']} | {row['dense_accuracy']:.6f} | {row['sequential_accuracy']:.6f} | {row['program_accuracy']:.6f} | {row['program_w_to_c']} | {row['program_c_to_w']} | {row['program_net']} |"
        for row in summaries
    ) + f"""

## Required answers

1. P90-triggered training UIDs: **{corpus_summary['uids']}**.
2. Unique complete suffix programs: **{corpus_summary['programs']}**.
3. Source counts are listed above; overlaps are preserved in `corpus/program_provenance.jsonl`.
4. Per-W-UID cardinality is frozen in `corpus/program_cardinality_per_uid.csv`.
5. Full-corpus training completed: **yes**, after group-disjoint epoch selection and exact reinitialization.
6. Dense / Sequential-A / Program accuracies are reported in the table.
7. W→C, C→W, and pooled Net are reported in the table.
8. Pooled Net positive: **{primary}**.
9. Program accuracy >= Dense: **{overall['program_accuracy'] >= overall['dense_accuracy']}**.
10. Dense-C preservation improved over Sequential-A: **{overall['program_c_to_w'] < overall['sequential_c_to_w']}** (Program C→W {overall['program_c_to_w']} vs Sequential-A {overall['sequential_c_to_w']}).
11. Treated-W success versus Sequential-A: Program W→C {overall['program_w_to_c']} vs Sequential-A {overall['sequential_w_to_c']}.
12. MMMU-Pro rescues: **{mmmu['program_w_to_c']}**.
13. POPE remained inactive when Stage-1 did not hand off: Stage-1/Stage-2 funnel is in `metrics/stage1_stage2_funnel.csv`; POPE program Net {pope['program_net']}.
14. All-FULL rate after trigger is in `metrics/program_statistics.csv`.
15. Non-FULL action counts are in `metrics/program_statistics.csv` and `metrics/action_usage.csv`.
16. Internal-dev exact match to any known program: **{internal_match if internal_match is not None else 'N/A'}**; nearest-program distances are diagnostic, while LMMS end-to-end correctness is primary.

Paired bootstrap intervals are in `metrics/paired_bootstrap.csv`. A positive point estimate is not interpreted as conclusive when its interval crosses zero.
"""
    _atomic_bytes(output_root / "summaries/polar_suffix_program_full_eval_summary.md", summary_md.encode())
    _atomic_bytes(output_root / "summaries/next_stage2_recommendation.md", (
        "# Next Stage-2 recommendation\n\n"
        f"{recommendation}\n\nNo follow-on experiment is authorized or executed in this phase.\n"
    ).encode())

    required = [
        path for path in output_root.rglob("*")
        if path.is_file()
        and "work" not in path.relative_to(output_root).parts
        and "features/trigger_states" not in str(path.relative_to(output_root))
        and path.name != "artifact_manifest.json"
    ]
    artifact = {
        "schema_version": "polar_suffix_program_artifact_manifest_v1",
        "contract_sha256": contract["contract_sha256"],
        "completed_at": utc_now(), "evaluation_rows": len(rows),
        "full_refit_checkpoint_sha256": read_json(
            output_root / "training/full_refit_checkpoint_manifest.json"
        )["checkpoint_sha256"],
        "files": {
            str(path.relative_to(output_root)): file_sha256(path) for path in sorted(required)
        },
        "trigger_state_index": "features/trigger_state_index.jsonl",
        "raw_trigger_states_bound_by_index": True,
    }
    atomic_json(output_root / "artifact_manifest.json", artifact)
    print(json.dumps({
        "evaluation_rows": len(rows), "dense_accuracy": overall["dense_accuracy"],
        "sequential_accuracy": overall["sequential_accuracy"],
        "program_accuracy": overall["program_accuracy"], "w_to_c": overall["program_w_to_c"],
        "c_to_w": overall["program_c_to_w"], "net": overall["program_net"],
        "outcome": outcome,
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "prepare", "smoke", "replay-worker", "aggregate-replay", "train-internal-dev",
        "train-full-refit", "prepare-evaluation", "evaluation-worker", "aggregate-evaluation",
    ))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
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
    elif args.command == "train-internal-dev":
        train_internal_dev(args.config, args.device)
    elif args.command == "train-full-refit":
        train_full_refit(args.config, args.device)
    elif args.command == "prepare-evaluation":
        prepare_evaluation(args.config)
    elif args.command == "evaluation-worker":
        evaluation_worker(args.config, args.rank, args.resume)
    elif args.command == "aggregate-evaluation":
        aggregate_evaluation(args.config)


if __name__ == "__main__":
    main()
