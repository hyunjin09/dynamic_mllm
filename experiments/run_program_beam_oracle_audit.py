#!/usr/bin/env python3
"""Audit every candidate from the frozen Phase-74 suffix-program beam."""

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
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from dense_failure_stage2.program_beam_oracle import (
    ACTION_NAMES,
    compare_top1_replay,
    freeze_beam_manifests,
    summarize_oracle_sample,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/program_beam_oracle_audit_v1.json"
ALLOWED_ROOTS = (PROJECT_ROOT.resolve(), Path("/mnt/hyemin").resolve())
BOUND_CODE = (
    "configs/program_beam_oracle_audit_v1.json",
    "dense_failure_stage2/program_beam_oracle.py",
    "experiments/run_program_beam_oracle_audit.py",
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
    rows: list[dict[str, Any]] = []
    with resolve_path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_bytes(
        path,
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows).encode(),
    )


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        _atomic_bytes(path, b"")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, buffer.getvalue().encode())


def atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode())


def _row_sha256(row: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _uid_slug(uid: str) -> str:
    return sha256(str(uid).encode()).hexdigest()[:24]


def _command_output(command: Sequence[str]) -> str:
    return subprocess.run(
        list(command), cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_state() -> dict[str, str]:
    return {
        "commit": _command_output(("git", "rev-parse", "HEAD")),
        "branch": _command_output(("git", "branch", "--show-current")),
        "worktree_status_at_freeze": _command_output(("git", "status", "--porcelain=v1")),
    }


def _runtime_state() -> dict[str, Any]:
    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "missing"

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "lmms_eval": version("lmms-eval"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }


def load_config(path: str | Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "program_beam_oracle_audit_config_v1":
        raise ValueError("unexpected beam-oracle config schema")
    if int(config["world_size"]) != 4:
        raise ValueError("beam-oracle audit requires four GPU workers")
    if list(config["beam"]["actions"]) != list(ACTION_NAMES):
        raise ValueError("beam action order differs")
    if int(config["beam"]["width"]) != 8:
        raise ValueError("beam width differs from Phase 74")
    if config["beam"]["score"] != "sum_action_log_probabilities":
        raise ValueError("beam score differs from Phase 74")
    if float(config["parity"]["beam_score_absolute_tolerance"]) != 1e-6:
        raise ValueError("score tolerance differs from reviewed protocol")
    return config


def _source_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    return {name: resolve_path(value) for name, value in config["sources"].items()}


def _verify_artifact_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    mismatches = []
    for relative, expected in manifest.get("files", {}).items():
        path = root / relative
        actual = file_sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches.append((relative, expected, actual))
    if mismatches:
        raise RuntimeError(f"Phase-74 artifact hash mismatch: {mismatches[:3]}")


def _verify_parent(config: Mapping[str, Any], *, verify_model: bool) -> dict[str, Any]:
    from experiments.run_polar_suffix_program import verify_contract as verify_phase74

    sources = _source_paths(config)
    parent, _ = verify_phase74(sources["phase74_config"], verify_model=verify_model)
    if parent["contract_sha256"] != config["frozen_parent"]["phase74_contract_sha256"]:
        raise RuntimeError("Phase-74 parent contract differs")
    if file_sha256(sources["phase74_contract"]) != file_sha256(
        resolve_path("analysis/dense_failure_stage2/polar_suffix_program/frozen_protocol.json")
    ):
        raise RuntimeError("Phase-74 configured contract path differs")
    manifest = read_json(sources["phase74_artifact_manifest"])
    if manifest.get("contract_sha256") != parent["contract_sha256"]:
        raise RuntimeError("Phase-74 artifact manifest contract differs")
    _verify_artifact_manifest(sources["phase74_artifact_manifest"].parent, manifest)
    checkpoint = read_json(sources["phase74_checkpoint_manifest"])
    if (
        checkpoint.get("contract_sha256") != parent["contract_sha256"]
        or checkpoint.get("checkpoint_sha256")
        != config["frozen_parent"]["phase74_checkpoint_sha256"]
        or file_sha256(sources["phase74_checkpoint"])
        != config["frozen_parent"]["phase74_checkpoint_sha256"]
    ):
        raise RuntimeError("Phase-74 checkpoint identity differs")
    return parent


def prepare(config_path: Path) -> None:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    parent = _verify_parent(config, verify_model=False)
    sources = _source_paths(config)
    parent_file = read_json(sources["phase74_contract"])
    if parent_file.get("contract_sha256") != parent["contract_sha256"]:
        raise RuntimeError("Phase-74 contract file differs from verified parent")
    evaluation_input = read_json(sources["phase74_evaluation_input"])
    if (
        evaluation_input.get("contract_sha256") != parent["contract_sha256"]
        or evaluation_input.get("phase69_contract_sha256")
        != config["frozen_parent"]["phase69_contract_sha256"]
    ):
        raise RuntimeError("Phase-74 evaluation input identity differs")
    paired_rows = read_jsonl(sources["phase74_paired"])
    evaluation_rows = read_jsonl(sources["phase74_evaluation_manifest"])
    phase69_rows = read_jsonl(sources["phase69_paired"])
    expected_external = int(config["population"]["external_rows"])
    if not (
        len(paired_rows) == len(evaluation_rows) == len(phase69_rows) == expected_external
    ):
        raise RuntimeError("external population count differs")
    paired_uids = [str(row["uid"]) for row in paired_rows]
    if (
        len(set(paired_uids)) != len(paired_uids)
        or set(paired_uids) != {str(row["uid"]) for row in evaluation_rows}
        or set(paired_uids) != {str(row["uid"]) for row in phase69_rows}
    ):
        raise RuntimeError("external UID populations differ")
    triggered = [row for row in paired_rows if bool(row.get("triggered"))]
    ranked, unique, structural = freeze_beam_manifests(
        triggered,
        beam_width=int(config["beam"]["width"]),
        total_layers=int(config["beam"]["total_layers"]),
    )
    expected_population = config["population"]
    checks = {
        "triggered": len(triggered),
        "triggered_dense_w": sum(not bool(row["dense_correct"]) for row in triggered),
        "triggered_dense_c": sum(bool(row["dense_correct"]) for row in triggered),
        "ranked_program_entries": len(ranked),
        "layer27_exhaustive_four": structural["late_trigger_exhaustive_four"],
    }
    for key, actual in checks.items():
        if actual != int(expected_population[key]):
            raise RuntimeError(
                f"frozen population differs for {key}: {actual} != {expected_population[key]}"
            )
    assigned_unique, worker_loads = _assign_uid_workers(unique, int(config["world_size"]))
    uid_rank = {str(row["uid"]): int(row["worker_rank"]) for row in assigned_unique}
    if len(uid_rank) != len(triggered):
        raise RuntimeError("worker assignment does not cover every triggered UID")
    ranked = [{**row, "worker_rank": uid_rank[str(row["uid"])]} for row in ranked]
    triggered_manifest = [
        {
            "uid": str(row["uid"]),
            "sample_id": str(row["sample_id"]),
            "benchmark": str(row["benchmark"]),
            "benchmark_family": str(row["benchmark_family"]),
            "image_group_id": str(row["image_group_id"]),
            "dense_correct": bool(row["dense_correct"]),
            "trigger_layer": int(row["trigger_layer"]),
            "available_beam_size": len(row["beam_rows"]),
            "worker_rank": uid_rank[str(row["uid"])],
            "phase74_row_sha256": _row_sha256(row),
        }
        for row in sorted(triggered, key=lambda item: str(item["uid"]))
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output_root / "candidates/triggered_sample_manifest.jsonl", triggered_manifest)
    atomic_jsonl(output_root / "candidates/beam8_ranked_programs.jsonl", ranked)
    atomic_jsonl(
        output_root / "candidates/unique_program_execution_manifest.jsonl", assigned_unique
    )
    internal_hashes = {
        relative: file_sha256(output_root / relative)
        for relative in (
            "candidates/triggered_sample_manifest.jsonl",
            "candidates/beam8_ranked_programs.jsonl",
            "candidates/unique_program_execution_manifest.jsonl",
        )
    }
    contract = {
        "schema_version": "program_beam_oracle_audit_contract_v1",
        "config_sha256": file_sha256(config_path),
        "source_sha256": {name: file_sha256(path) for name, path in sources.items()},
        "bound_code_sha256": {path: file_sha256(path) for path in BOUND_CODE},
        "internal_manifest_sha256": internal_hashes,
        "parent": {
            "phase74_contract_sha256": parent["contract_sha256"],
            "phase74_checkpoint_sha256": config["frozen_parent"][
                "phase74_checkpoint_sha256"
            ],
            "phase69_contract_sha256": evaluation_input["phase69_contract_sha256"],
        },
        "population": {
            **checks,
            "unique_programs": len(assigned_unique),
            "duplicate_ranked_entries": structural["duplicate_ranked_entries"],
            "worker_estimated_suffix_layer_loads": worker_loads,
        },
        "parity": dict(config["parity"]),
        "beam": dict(config["beam"]),
        "review_reconciliation": {
            "verdict": "revise_then_proceed",
            "separate_global_top1_gate": True,
            "score_tolerance_instead_of_float_equality": True,
            "layer27_exhaustive_four_support": True,
            "structural_audit_errors": 0,
        },
        "git": _git_state(),
        "runtime": _runtime_state(),
        "static_config": config,
        "created_at": utc_now(),
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_protocol.json", contract)
    atomic_json(
        output_root / "parity/manifest_hashes.json",
        {
            "contract_sha256": contract["contract_sha256"],
            "sources": contract["source_sha256"],
            "internal_manifests": internal_hashes,
        },
    )
    atomic_json(
        output_root / "parity/checkpoint_contract.json",
        {
            "contract_sha256": contract["contract_sha256"],
            **contract["parent"],
            "checkpoint_path": str(sources["phase74_checkpoint"].relative_to(PROJECT_ROOT)),
        },
    )
    atomic_text(
        output_root / "protocol.md",
        f"""# Frozen program beam-oracle audit protocol

- Contract: `{contract['contract_sha256']}`.
- Parent Phase-74 contract: `{parent['contract_sha256']}`.
- Frozen population: 19,960 external rows; 901 triggered (496 Dense-W, 405 Dense-C).
- Candidate set: {len(ranked):,} ranked entries / {len(assigned_unique):,} unique UID-program executions.
- Beam: width 8, sum of action log probabilities; layer-27 triggers use their exhaustive four-program support.
- Gate: all 901 top-1 paths must reproduce exact trigger layer, program, generated tokens, answer, LMMS score, and correctness; beam score tolerance is {config['parity']['beam_score_absolute_tolerance']}.
- Only after the global top-1 gate passes may frozen ranks 2-8 execute.
- External labels score already-frozen programs only. No search, reranking, tuning, retraining, or model modification is allowed.
""",
    )
    print(
        json.dumps(
            {"contract_sha256": contract["contract_sha256"], **contract["population"]},
            sort_keys=True,
        )
    )


def verify_contract(
    config_path: Path, *, verify_model: bool = False
) -> tuple[dict[str, Any], Path]:
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if contract.get("contract_sha256") != canonical_hash(contract):
        raise RuntimeError("Phase-75 frozen contract hash differs")
    if contract.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError("Phase-75 config changed after freeze")
    for name, path in _source_paths(config).items():
        if file_sha256(path) != contract["source_sha256"].get(name):
            raise RuntimeError(f"Phase-75 source changed after freeze: {name}")
    for path, expected in contract["bound_code_sha256"].items():
        if file_sha256(path) != expected:
            raise RuntimeError(f"Phase-75 bound code changed after freeze: {path}")
    for relative, expected in contract["internal_manifest_sha256"].items():
        if file_sha256(output_root / relative) != expected:
            raise RuntimeError(f"Phase-75 internal manifest changed: {relative}")
    parent = _verify_parent(config, verify_model=verify_model)
    if parent["contract_sha256"] != contract["parent"]["phase74_contract_sha256"]:
        raise RuntimeError("Phase-75 parent verification differs")
    return contract, output_root


def _load_source_maps(config: Mapping[str, Any]) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    sources = _source_paths(config)
    phase74 = {str(row["uid"]): row for row in read_jsonl(sources["phase74_paired"])}
    evaluation = {
        str(row["uid"]): row for row in read_jsonl(sources["phase74_evaluation_manifest"])
    }
    phase69 = {str(row["uid"]): row for row in read_jsonl(sources["phase69_paired"])}
    if not (set(phase74) == set(evaluation) == set(phase69)):
        raise RuntimeError("source UID maps differ")
    return phase74, evaluation, phase69


def _resume_json(path: Path, *, contract_sha256: str, uid: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    row = read_json(path)
    if row.get("contract_sha256") != contract_sha256 or str(row.get("uid")) != str(uid):
        raise RuntimeError(f"incompatible resume record: {path}")
    return row


def top1_worker(config_path: Path, rank: int, resume: bool) -> None:
    from experiments.run_polar_suffix_program import ProgramEvalRuntime, _process_eval_row

    contract, output_root = verify_contract(config_path, verify_model=True)
    config = contract["static_config"]
    rank = int(rank)
    if rank < 0 or rank >= int(config["world_size"]):
        raise ValueError("top-1 worker rank is outside the frozen world")
    manifest = [
        row
        for row in read_jsonl(output_root / "candidates/triggered_sample_manifest.jsonl")
        if int(row["worker_rank"]) == rank
    ]
    rank1_program = {
        str(row["uid"]): str(row["program_id"])
        for row in read_jsonl(output_root / "candidates/beam8_ranked_programs.jsonl")
        if int(row["rank"]) == 1
    }
    phase74, evaluation, phase69 = _load_source_maps(config)
    parent = read_json(_source_paths(config)["phase74_contract"])
    runtime = ProgramEvalRuntime(parent["static_config"], rank)
    worker_root = output_root / f"work/top1/rank{rank:02d}"
    completed = 0
    score_tolerance = float(config["parity"]["beam_score_absolute_tolerance"])
    for index, item in enumerate(manifest):
        uid = str(item["uid"])
        path = worker_root / "uids" / f"{_uid_slug(uid)}.json"
        old = _resume_json(path, contract_sha256=contract["contract_sha256"], uid=uid)
        if old is not None:
            if not resume:
                raise RuntimeError(f"existing top-1 output requires --resume: {path}")
            if old.get("phase74_row_sha256") != item["phase74_row_sha256"]:
                raise RuntimeError(f"top-1 resume source changed for {uid}")
            if not all(bool(value) for value in old["parity"].values()):
                raise RuntimeError(f"cannot resume mismatched top-1 replay for {uid}")
            completed += 1
            continue
        started = time.monotonic()
        expected = phase74[uid]
        if _row_sha256(expected) != item["phase74_row_sha256"]:
            raise RuntimeError(f"Phase-74 source row changed for {uid}")
        actual = _process_eval_row(
            runtime,
            evaluation[uid],
            phase69[uid],
            parent["contract_sha256"],
        )
        parity = compare_top1_replay(
            expected, actual, score_abs_tolerance=score_tolerance
        )
        result = {
            "schema_version": "program_beam_oracle_top1_parity_row_v1",
            "contract_sha256": contract["contract_sha256"],
            "phase74_contract_sha256": parent["contract_sha256"],
            "phase74_row_sha256": item["phase74_row_sha256"],
            "uid": uid,
            "worker_rank": rank,
            "benchmark": str(expected["benchmark"]),
            "benchmark_family": str(expected["benchmark_family"]),
            "dense_correct": bool(actual["dense_correct"]),
            "trigger_layer": int(actual["trigger_layer"]),
            "program_id": rank1_program[uid],
            "actions": list(actual["program_suffix_actions"]),
            "sequence_score": float(actual["beam_rows"][0]["score"]),
            "expected_sequence_score": float(expected["beam_rows"][0]["score"]),
            "sequence_score_absolute_error": abs(
                float(actual["beam_rows"][0]["score"])
                - float(expected["beam_rows"][0]["score"])
            ),
            "generated_token_ids": list(actual["program_generated_token_ids"]),
            "generated_answer": str(actual["program_generated_answer"]),
            "metric_name": str(expected["metric_name"]),
            "score": float(actual["program_score"]),
            "correctness_threshold": float(expected["correctness_threshold"]),
            "correct": bool(actual["program_correct"]),
            "program_correct": bool(actual["program_correct"]),
            "consumed_image_sha256s": list(actual["consumed_image_sha256s"]),
            "phase69_reuse_parity": dict(actual["phase69_reuse_parity"]),
            "parity": parity,
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(path, result)
        if not all(parity.values()):
            raise RuntimeError(f"top-1 parity failed for {uid}: {parity}")
        completed += 1
        if index % 20 == 0 or completed == len(manifest):
            print(
                json.dumps(
                    {"mode": "top1", "rank": rank, "completed": completed, "total": len(manifest)}
                ),
                flush=True,
            )
    atomic_json(
        worker_root / "complete.json",
        {
            "schema_version": "program_beam_oracle_top1_worker_complete_v1",
            "contract_sha256": contract["contract_sha256"],
            "worker_rank": rank,
            "expected_rows": len(manifest),
            "completed_rows": completed,
            "completed_at": utc_now(),
        },
    )


def _collect_top1_workers(
    contract: Mapping[str, Any], output_root: Path
) -> list[dict[str, Any]]:
    config = contract["static_config"]
    expected = read_jsonl(output_root / "candidates/triggered_sample_manifest.jsonl")
    rows: list[dict[str, Any]] = []
    for rank in range(int(config["world_size"])):
        completion = read_json(output_root / f"work/top1/rank{rank:02d}/complete.json")
        rank_expected = [row for row in expected if int(row["worker_rank"]) == rank]
        if (
            completion.get("contract_sha256") != contract["contract_sha256"]
            or int(completion.get("worker_rank", -1)) != rank
            or int(completion.get("expected_rows", -1)) != len(rank_expected)
            or int(completion.get("completed_rows", -1)) != len(rank_expected)
        ):
            raise RuntimeError(f"top-1 worker completion differs for rank {rank}")
        expected_paths = {
            output_root / f"work/top1/rank{rank:02d}/uids/{_uid_slug(item['uid'])}.json"
            for item in rank_expected
        }
        observed_paths = set(
            (output_root / f"work/top1/rank{rank:02d}/uids").glob("*.json")
        )
        if observed_paths != expected_paths:
            raise RuntimeError(f"top-1 worker file coverage differs for rank {rank}")
        for item in rank_expected:
            path = output_root / f"work/top1/rank{rank:02d}/uids/{_uid_slug(item['uid'])}.json"
            row = read_json(path)
            if row.get("contract_sha256") != contract["contract_sha256"]:
                raise RuntimeError(f"top-1 worker row contract differs: {item['uid']}")
            rows.append(row)
    return rows


def aggregate_top1(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path)
    config = contract["static_config"]
    expected = read_jsonl(output_root / "candidates/triggered_sample_manifest.jsonl")
    rows = _collect_top1_workers(contract, output_root)
    summary = _validate_parity_rows(
        expected,
        rows,
        expected_w_to_c=int(config["parity"]["expected_w_to_c"]),
        expected_c_to_w=int(config["parity"]["expected_c_to_w"]),
        expected_net=int(config["parity"]["expected_net"]),
    )
    ordered = sorted(rows, key=lambda row: str(row["uid"]))
    atomic_jsonl(output_root / "parity/top1_replay_rows.jsonl", ordered)
    max_score_error = max(float(row["sequence_score_absolute_error"]) for row in ordered)
    report = f"""# Top-1 parity report

- Audited triggered UIDs: **{summary['rows']} / {len(expected)}**.
- Trigger layer, top-1 suffix, generated token IDs, answer, LMMS score, and correctness: **exact for every UID**.
- Beam sequence-score absolute tolerance: **{config['parity']['beam_score_absolute_tolerance']}**.
- Maximum observed sequence-score absolute error: **{max_score_error}**.
- Reproduced W-to-C/C-to-W/net: **{summary['w_to_c']}/{summary['c_to_w']}/{summary['net']}**.
- Global parity gate: **PASS**.

Ranks 2-8 were not executed before this global gate passed.
"""
    atomic_text(output_root / "parity/top1_parity_report.md", report)
    gate = {
        "schema_version": "program_beam_oracle_top1_global_gate_v1",
        "contract_sha256": contract["contract_sha256"],
        **summary,
        "max_sequence_score_absolute_error": max_score_error,
        "top1_replay_rows_sha256": file_sha256(output_root / "parity/top1_replay_rows.jsonl"),
        "passed": True,
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "work/top1_global_gate.json", gate)
    print(json.dumps(gate, sort_keys=True))


def _verified_top1_gate(contract: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    gate = read_json(output_root / "work/top1_global_gate.json")
    if (
        gate.get("contract_sha256") != contract["contract_sha256"]
        or gate.get("passed") is not True
        or file_sha256(output_root / "parity/top1_replay_rows.jsonl")
        != gate.get("top1_replay_rows_sha256")
    ):
        raise RuntimeError("global top-1 parity gate is absent or invalid")
    return gate


def candidate_worker(config_path: Path, rank: int, resume: bool) -> None:
    from binary_policy.executor import (
        capture_four_action_route,
        capture_four_action_suffix_from_full_baseline,
    )
    from binary_policy.executor.inputs import build_binary_inputs
    from experiments.run_full_benchmark_end_to_end_eval import (
        _build_inputs as eval_build_inputs,
        _generate as eval_generate,
        _verify_images,
    )
    from experiments.run_polar_suffix_program import ProgramEvalRuntime

    contract, output_root = verify_contract(config_path, verify_model=True)
    config = contract["static_config"]
    _verified_top1_gate(contract, output_root)
    rank = int(rank)
    if rank < 0 or rank >= int(config["world_size"]):
        raise ValueError("candidate worker rank is outside the frozen world")
    unique = [
        row
        for row in read_jsonl(
            output_root / "candidates/unique_program_execution_manifest.jsonl"
        )
        if int(row["worker_rank"]) == rank and 1 not in [int(value) for value in row["source_ranks"]]
    ]
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique:
        by_uid[str(row["uid"])].append(row)
    phase74, evaluation, _ = _load_source_maps(config)
    parent = read_json(_source_paths(config)["phase74_contract"])
    runtime = ProgramEvalRuntime(parent["static_config"], rank)
    worker_root = output_root / f"work/candidates/rank{rank:02d}"
    completed_programs = 0
    for index, uid in enumerate(sorted(by_uid)):
        expected_rows = sorted(by_uid[uid], key=lambda row: min(row["source_ranks"]))
        path = worker_root / "uids" / f"{_uid_slug(uid)}.json"
        old = _resume_json(path, contract_sha256=contract["contract_sha256"], uid=uid)
        if old is not None:
            if not resume:
                raise RuntimeError(f"existing candidate output requires --resume: {path}")
            expected_ids = Counter(str(row["program_id"]) for row in expected_rows)
            observed_ids = Counter(str(row["program_id"]) for row in old.get("results", []))
            if expected_ids != observed_ids or any(count != 1 for count in observed_ids.values()):
                raise RuntimeError(f"candidate resume program coverage differs for {uid}")
            if list(old.get("consumed_image_sha256s", [])) != list(
                phase74[uid]["consumed_image_sha256s"]
            ):
                raise RuntimeError(f"candidate resume image provenance differs for {uid}")
            expected_by_id = {str(row["program_id"]): row for row in expected_rows}
            for result in old["results"]:
                expected = expected_by_id[str(result["program_id"])]
                if (
                    result.get("contract_sha256") != contract["contract_sha256"]
                    or list(result["actions"]) != list(expected["actions"])
                    or [int(value) for value in result["source_ranks"]]
                    != [int(value) for value in expected["source_ranks"]]
                    or float(result["sequence_score"]) != float(expected["sequence_score"])
                ):
                    raise RuntimeError(f"candidate resume provenance differs for {uid}")
            completed_programs += len(expected_rows)
            continue
        eval_row = evaluation[uid]
        source = phase74[uid]
        consumed = _verify_images(eval_row)
        if list(consumed) != list(source["consumed_image_sha256s"]):
            raise RuntimeError(f"candidate image bytes differ from Phase 74 for {uid}")
        inputs = eval_build_inputs(runtime.base_runtime, eval_row)
        prepared = build_binary_inputs(runtime.base_runtime.wrapped, inputs)
        baseline = capture_four_action_route(
            runtime.base_runtime.wrapped,
            {},
            ["FULL"] * int(config["beam"]["total_layers"]),
            prepared_inputs=prepared,
            use_cache=True,
            native_full_rows=True,
        )
        results = []
        for candidate in expected_rows:
            started = time.monotonic()
            output = capture_four_action_suffix_from_full_baseline(
                runtime.base_runtime.wrapped,
                baseline,
                int(candidate["trigger_layer"]),
                candidate["actions"],
            )
            generated = eval_generate(
                runtime.base_runtime, output, inputs["input_ids"], eval_row
            )
            if candidate["is_all_full"]:
                dense_parity = {
                    "generated_token_ids": list(generated["generated_token_ids"])
                    == list(source["dense_generated_token_ids"]),
                    "generated_answer": str(generated["generated_answer"])
                    == str(source["dense_generated_answer"]),
                    "score": float(generated["score"]) == float(source["dense_score"]),
                    "correct": bool(generated["correct"]) == bool(source["dense_correct"]),
                }
                if not all(dense_parity.values()):
                    raise RuntimeError(f"all-FULL candidate parity failed for {uid}: {dense_parity}")
            else:
                dense_parity = None
            results.append(
                {
                    "schema_version": "program_beam_oracle_candidate_result_v1",
                    "contract_sha256": contract["contract_sha256"],
                    "uid": uid,
                    "worker_rank": rank,
                    "program_id": str(candidate["program_id"]),
                    "source_ranks": [int(value) for value in candidate["source_ranks"]],
                    "actions": list(candidate["actions"]),
                    "sequence_score": float(candidate["sequence_score"]),
                    "generated_token_ids": list(generated["generated_token_ids"]),
                    "generated_answer": str(generated["generated_answer"]),
                    "metric_name": str(source["metric_name"]),
                    "score": float(generated["score"]),
                    "correctness_threshold": float(source["correctness_threshold"]),
                    "correct": bool(generated["correct"]),
                    "all_full_dense_parity": dense_parity,
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            del output
        atomic_json(
            path,
            {
                "schema_version": "program_beam_oracle_candidate_uid_results_v1",
                "contract_sha256": contract["contract_sha256"],
                "uid": uid,
                "worker_rank": rank,
                "consumed_image_sha256s": list(consumed),
                "results": results,
            },
        )
        completed_programs += len(results)
        del baseline, prepared, inputs
        if index % 10 == 0 or index + 1 == len(by_uid):
            print(
                json.dumps(
                    {
                        "mode": "candidates",
                        "rank": rank,
                        "completed_uids": index + 1,
                        "total_uids": len(by_uid),
                        "completed_programs": completed_programs,
                        "total_programs": len(unique),
                    }
                ),
                flush=True,
            )
    atomic_json(
        worker_root / "complete.json",
        {
            "schema_version": "program_beam_oracle_candidate_worker_complete_v1",
            "contract_sha256": contract["contract_sha256"],
            "worker_rank": rank,
            "expected_uids": len(by_uid),
            "expected_programs": len(unique),
            "completed_programs": completed_programs,
            "completed_at": utc_now(),
        },
    )


def _collect_candidate_workers(
    contract: Mapping[str, Any], output_root: Path
) -> list[dict[str, Any]]:
    config = contract["static_config"]
    unique = read_jsonl(output_root / "candidates/unique_program_execution_manifest.jsonl")
    remaining = [
        row for row in unique if 1 not in [int(value) for value in row["source_ranks"]]
    ]
    rows: list[dict[str, Any]] = []
    for rank in range(int(config["world_size"])):
        expected = [row for row in remaining if int(row["worker_rank"]) == rank]
        expected_uids = sorted({str(row["uid"]) for row in expected})
        completion = read_json(output_root / f"work/candidates/rank{rank:02d}/complete.json")
        if (
            completion.get("contract_sha256") != contract["contract_sha256"]
            or int(completion.get("worker_rank", -1)) != rank
            or int(completion.get("expected_uids", -1)) != len(expected_uids)
            or int(completion.get("expected_programs", -1)) != len(expected)
            or int(completion.get("completed_programs", -1)) != len(expected)
        ):
            raise RuntimeError(f"candidate worker completion differs for rank {rank}")
        expected_paths = {
            output_root / f"work/candidates/rank{rank:02d}/uids/{_uid_slug(uid)}.json"
            for uid in expected_uids
        }
        observed_paths = set(
            (output_root / f"work/candidates/rank{rank:02d}/uids").glob("*.json")
        )
        if observed_paths != expected_paths:
            raise RuntimeError(f"candidate worker file coverage differs for rank {rank}")
        for uid in expected_uids:
            payload = read_json(
                output_root / f"work/candidates/rank{rank:02d}/uids/{_uid_slug(uid)}.json"
            )
            if (
                payload.get("contract_sha256") != contract["contract_sha256"]
                or str(payload.get("uid")) != uid
            ):
                raise RuntimeError(f"candidate UID output contract differs for {uid}")
            rows.extend(payload["results"])
    return rows


def collect_candidates(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path)
    _verified_top1_gate(contract, output_root)
    unique = read_jsonl(output_root / "candidates/unique_program_execution_manifest.jsonl")
    unique_by_program = {str(row["program_id"]): row for row in unique}
    top1 = read_jsonl(output_root / "parity/top1_replay_rows.jsonl")
    remaining = _collect_candidate_workers(contract, output_root)
    rows = [
        {
            "schema_version": "program_beam_oracle_candidate_result_v1",
            "contract_sha256": contract["contract_sha256"],
            "uid": row["uid"],
            "worker_rank": row["worker_rank"],
            "program_id": row["program_id"],
            "source_ranks": [
                int(value) for value in unique_by_program[str(row["program_id"])]["source_ranks"]
            ],
            "actions": row["actions"],
            "sequence_score": row["sequence_score"],
            "generated_token_ids": row["generated_token_ids"],
            "generated_answer": row["generated_answer"],
            "metric_name": row["metric_name"],
            "score": row["score"],
            "correctness_threshold": row["correctness_threshold"],
            "correct": row["correct"],
            "all_full_dense_parity": None,
            "execution_source": "top1_global_parity_pass",
            "elapsed_seconds": row["elapsed_seconds"],
        }
        for row in top1
    ] + [
        {**row, "execution_source": "post_parity_candidate_pass"} for row in remaining
    ]
    _validate_execution_coverage(unique, rows)
    order = {str(row["program_id"]): index for index, row in enumerate(unique)}
    rows.sort(key=lambda row: order[str(row["program_id"])])
    atomic_jsonl(output_root / "candidates/candidate_execution_results.jsonl", rows)
    completion = {
        "schema_version": "program_beam_oracle_candidate_collection_v1",
        "contract_sha256": contract["contract_sha256"],
        "unique_programs": len(rows),
        "top1_programs": len(top1),
        "post_parity_programs": len(remaining),
        "candidate_results_sha256": file_sha256(
            output_root / "candidates/candidate_execution_results.jsonl"
        ),
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "work/candidate_collection_complete.json", completion)
    print(json.dumps(completion, sort_keys=True))


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _mean(values: Sequence[int | float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _quantile(values: Sequence[int | float | None], fraction: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * float(fraction)
    lower = int(position)
    upper = min(lower + 1, len(clean) - 1)
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def _scope_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    return {
        "overall": list(rows),
        "chartqa": [row for row in rows if row["benchmark_family"] == "chartqa"],
        "textvqa": [row for row in rows if row["benchmark_family"] == "textvqa"],
        "mmmu_pro": [row for row in rows if row["benchmark_family"] == "mmmu_pro"],
        "pope": [row for row in rows if row["benchmark_family"] == "pope"],
    }


def _candidate_for_rank(
    uid: str,
    rank: int,
    ranked_by_uid: Mapping[str, Sequence[Mapping[str, Any]]],
    result_by_program: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    candidate = next(row for row in ranked_by_uid[uid] if int(row["rank"]) == int(rank))
    return candidate, result_by_program[str(candidate["program_id"])]


def _write_figures(
    output_root: Path,
    rescue_rows: Sequence[Mapping[str, Any]],
    first_rank_rows: Sequence[Mapping[str, Any]],
    w_failure_rows: Sequence[Mapping[str, Any]],
    c_failure_rows: Sequence[Mapping[str, Any]],
    all_full_rows: Sequence[Mapping[str, Any]],
    diversity_rows: Sequence[Mapping[str, Any]],
    score_gap_rows: Sequence[Mapping[str, Any]],
    intensity_rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    scopes = ["chartqa", "textvqa", "mmmu_pro", "overall"]
    labels = ["ChartQA", "TextVQA", "MMMU-Pro", "Overall"]
    x = np.arange(len(scopes))
    figure, axis = plt.subplots(figsize=(9, 5))
    for offset, k in zip((-0.27, -0.09, 0.09, 0.27), (1, 2, 4, 8)):
        values = [
            next(row["rescued"] for row in rescue_rows if row["scope"] == scope and row["k"] == k)
            for scope in scopes
        ]
        axis.bar(x + offset, values, 0.18, label=f"@{k}")
    axis.set_xticks(x, labels); axis.set_ylabel("Dense-W with a correct beam candidate"); axis.legend()
    figure.tight_layout(); figure.savefig(figures / "rescue_at_k.png", dpi=160); plt.close(figure)

    overall_ranks = [row for row in first_rank_rows if row["scope"] == "overall" and row["rank"] != "NONE"]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar([int(row["rank"]) for row in overall_ranks], [int(row["count"]) for row in overall_ranks])
    axis.set_xlabel("First correct beam rank"); axis.set_ylabel("Triggered Dense-W samples")
    figure.tight_layout(); figure.savefig(figures / "first_correct_rank.png", dpi=160); plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    ranking = [next(row["count"] for row in w_failure_rows if row["scope"] == scope and row["category"] == "RANKING_FAILURE_W") for scope in scopes]
    generation = [next(row["count"] for row in w_failure_rows if row["scope"] == scope and row["category"] == "GENERATION_FAILURE_W") for scope in scopes]
    axis.bar(x - 0.18, ranking, 0.36, label="Ranking failure"); axis.bar(x + 0.18, generation, 0.36, label="Generation failure")
    axis.set_xticks(x, labels); axis.set_ylabel("Triggered Dense-W samples"); axis.legend()
    figure.tight_layout(); figure.savefig(figures / "w_ranking_vs_generation_failure.png", dpi=160); plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    c_rank = [next(row["count"] for row in c_failure_rows if row["scope"] == scope and row["category"] == "RANKING_FAILURE_C") for scope in scopes]
    c_gen = [next(row["count"] for row in c_failure_rows if row["scope"] == scope and row["category"] == "GENERATION_FAILURE_C") for scope in scopes]
    axis.bar(x - 0.18, c_rank, 0.36, label="Ranking failure"); axis.bar(x + 0.18, c_gen, 0.36, label="Generation failure")
    axis.set_xticks(x, labels); axis.set_ylabel("Triggered Dense-C samples"); axis.legend()
    figure.tight_layout(); figure.savefig(figures / "c_ranking_vs_generation_failure.png", dpi=160); plt.close(figure)

    ranks = [int(row["all_full_rank"]) for row in all_full_rows if row["all_full_rank"] is not None]
    figure, axis = plt.subplots(figsize=(7, 4)); axis.hist(ranks, bins=np.arange(0.5, 9.5, 1))
    axis.set_xlabel("All-FULL beam rank"); axis.set_ylabel("Triggered samples")
    figure.tight_layout(); figure.savefig(figures / "all_full_rank_distribution.png", dpi=160); plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    w_values = [row["mean_pairwise_hamming"] for row in diversity_rows if not row["dense_correct"]]
    c_values = [row["mean_pairwise_hamming"] for row in diversity_rows if row["dense_correct"]]
    axis.boxplot([w_values, c_values], tick_labels=["Dense-W", "Dense-C"], showfliers=False)
    axis.set_ylabel("Mean pairwise Hamming distance")
    figure.tight_layout(); figure.savefig(figures / "beam_diversity_w_vs_c.png", dpi=160); plt.close(figure)

    sample_gaps = [row["score_gap"] for row in score_gap_rows if row["row_kind"] == "sample"]
    figure, axis = plt.subplots(figsize=(7, 4)); axis.hist(sample_gaps, bins=30)
    axis.set_xlabel("Top-1 score minus best-correct score"); axis.set_ylabel("Ranking failures")
    figure.tight_layout(); figure.savefig(figures / "score_gap_correct_vs_top1.png", dpi=160); plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    categories = ["TOP1_WRONG", "BEST_CORRECT", "ALL_WRONG_BEAM"]
    values = [
        [row["non_full_count"] for row in intensity_rows if row["row_kind"] == "candidate" and row["category"] == category]
        for category in categories
    ]
    axis.boxplot(values, tick_labels=["Top-1 wrong", "Best correct", "All-wrong beam"], showfliers=False)
    axis.set_ylabel("Non-FULL actions")
    figure.tight_layout(); figure.savefig(figures / "nonfull_intensity_correct_vs_top1.png", dpi=160); plt.close(figure)


def aggregate(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path)
    _verified_top1_gate(contract, output_root)
    completion = read_json(output_root / "work/candidate_collection_complete.json")
    candidate_path = output_root / "candidates/candidate_execution_results.jsonl"
    if (
        completion.get("contract_sha256") != contract["contract_sha256"]
        or file_sha256(candidate_path) != completion.get("candidate_results_sha256")
    ):
        raise RuntimeError("candidate collection completion is invalid")
    source_rows = [
        row
        for row in read_jsonl(_source_paths(contract["static_config"])["phase74_paired"])
        if bool(row.get("triggered"))
    ]
    source_by_uid = {str(row["uid"]): row for row in source_rows}
    ranked = read_jsonl(output_root / "candidates/beam8_ranked_programs.jsonl")
    ranked_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        ranked_by_uid[str(row["uid"])].append(row)
    unique = read_jsonl(output_root / "candidates/unique_program_execution_manifest.jsonl")
    candidates = read_jsonl(candidate_path)
    _validate_execution_coverage(unique, candidates)
    result_by_program = {str(row["program_id"]): row for row in candidates}
    for row in unique:
        if not bool(row["is_all_full"]):
            continue
        source = source_by_uid[str(row["uid"])]
        result = result_by_program[str(row["program_id"])]
        parity = {
            "tokens": list(result["generated_token_ids"])
            == list(source["dense_generated_token_ids"]),
            "answer": str(result["generated_answer"]) == str(source["dense_generated_answer"]),
            "score": float(result["score"]) == float(source["dense_score"]),
            "correct": bool(result["correct"]) == bool(source["dense_correct"]),
        }
        if not all(parity.values()):
            raise RuntimeError(f"all-FULL oracle identity failed for {row['uid']}: {parity}")
    samples = [
        summarize_oracle_sample(
            source,
            ranked_by_uid[str(source["uid"])],
            result_by_program,
        )
        for source in sorted(source_rows, key=lambda item: str(item["uid"]))
    ]
    scopes = _scope_rows(samples)
    top1_stats: list[dict[str, Any]] = []
    rescue_rows: list[dict[str, Any]] = []
    first_rank_rows: list[dict[str, Any]] = []
    w_failure_rows: list[dict[str, Any]] = []
    c_failure_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    beam_summary: list[dict[str, Any]] = []
    for scope, current in scopes.items():
        for dense_correct, label in ((False, "W"), (True, "C")):
            outcome_rows = [row for row in current if bool(row["dense_correct"]) == dense_correct]
            source_subset = [source_by_uid[str(row["uid"])] for row in outcome_rows]
            all_full = [
                all(action == "FULL" for action in row["program_suffix_actions"])
                for row in source_subset
            ]
            nonfull_counts = [int(row["non_full_count"]) for row in source_subset]
            delays = [row["trigger_to_first_non_full_delay"] for row in source_subset]
            top1_stats.append(
                {
                    "scope": scope,
                    "dense_outcome": label,
                    "triggered": len(outcome_rows),
                    "top1_all_full": sum(all_full),
                    "top1_any_non_full": len(all_full) - sum(all_full),
                    "p_all_full": _safe_rate(sum(all_full), len(all_full)),
                    "p_any_non_full": _safe_rate(len(all_full) - sum(all_full), len(all_full)),
                    "mean_non_full": _mean(nonfull_counts),
                    "median_non_full": _quantile(nonfull_counts, 0.5),
                    "mean_first_non_full_delay": _mean(delays),
                    "median_first_non_full_delay": _quantile(delays, 0.5),
                }
            )
        dense_w = [row for row in current if not bool(row["dense_correct"])]
        dense_c = [row for row in current if bool(row["dense_correct"])]
        for k in (1, 2, 4, 8):
            rescued = sum(bool(row[f"rescue_at_{k}"]) for row in dense_w)
            top1 = sum(bool(row["rescue_at_1"]) for row in dense_w)
            rescue_rows.append(
                {
                    "scope": scope,
                    "k": k,
                    "triggered_w": len(dense_w),
                    "rescued": rescued,
                    "rate": _safe_rate(rescued, len(dense_w)),
                    "additional_vs_top1": rescued - top1,
                }
            )
        for rank in [1, 2, 3, 4, 5, 6, 7, 8, "NONE"]:
            count = sum(
                (row["first_correct_rank"] is None if rank == "NONE" else row["first_correct_rank"] == rank)
                for row in dense_w
            )
            first_rank_rows.append(
                {"scope": scope, "rank": rank, "count": count, "rate": _safe_rate(count, len(dense_w))}
            )
        for category in ("TOP1_SUCCESS_W", "RANKING_FAILURE_W", "GENERATION_FAILURE_W"):
            count = sum(row["w_failure_class"] == category for row in dense_w)
            w_failure_rows.append(
                {"scope": scope, "category": category, "count": count, "rate": _safe_rate(count, len(dense_w))}
            )
        for category in ("TOP1_SUCCESS_C", "RANKING_FAILURE_C", "GENERATION_FAILURE_C"):
            count = sum(row["c_failure_class"] == category for row in dense_c)
            c_failure_rows.append(
                {"scope": scope, "category": category, "count": count, "rate": _safe_rate(count, len(dense_c))}
            )
        w_at = {
            k: sum(bool(row[f"rescue_at_{k}"]) for row in dense_w) for k in (1, 2, 4, 8)
        }
        ranking_w = sum(row["w_failure_class"] == "RANKING_FAILURE_W" for row in dense_w)
        generation_w = sum(row["w_failure_class"] == "GENERATION_FAILURE_W" for row in dense_w)
        ranking_c = sum(row["c_failure_class"] == "RANKING_FAILURE_C" for row in dense_c)
        generation_c = sum(row["c_failure_class"] == "GENERATION_FAILURE_C" for row in dense_c)
        c_to_w = sum(not row["top1_correct"] for row in dense_c)
        all_full_c = sum(bool(row["all_full_in_beam"]) for row in dense_c)
        summary_row = {
            "scope": scope,
            "triggered_w": len(dense_w),
            "w_to_c_at_1": w_at[1],
            "w_to_c_at_2": w_at[2],
            "w_to_c_at_4": w_at[4],
            "w_to_c_at_8": w_at[8],
            "additional_w_rescues": w_at[8] - w_at[1],
            "ranking_failure_w": ranking_w,
            "generation_failure_w": generation_w,
            "triggered_c": len(dense_c),
            "c_to_w_at_1": c_to_w,
            "ranking_failure_c": ranking_c,
            "generation_failure_c": generation_c,
            "all_full_in_c_beam": all_full_c,
        }
        benchmark_rows.append(summary_row)
        beam_summary.append(
            {
                **summary_row,
                "mean_unique_programs_w": _mean([row["unique_programs"] for row in dense_w]),
                "mean_hamming_w": _mean([row["mean_pairwise_hamming"] for row in dense_w]),
                "mean_unique_programs_c": _mean([row["unique_programs"] for row in dense_c]),
                "mean_hamming_c": _mean([row["mean_pairwise_hamming"] for row in dense_c]),
            }
        )
    all_full_rows = [
        {
            "uid": row["uid"],
            "benchmark_family": row["benchmark_family"],
            "dense_correct": row["dense_correct"],
            "trigger_layer": row["trigger_layer"],
            "all_full_in_beam": row["all_full_in_beam"],
            "all_full_rank": row["all_full_rank"],
            "all_full_score": row["all_full_score"],
            "top1_correct": row["top1_correct"],
            "c_all_full_class": row["c_all_full_class"],
        }
        for row in samples
    ]
    diversity_rows = [
        {
            "uid": row["uid"],
            "benchmark_family": row["benchmark_family"],
            "dense_correct": row["dense_correct"],
            "w_failure_class": row["w_failure_class"],
            "c_failure_class": row["c_failure_class"],
            "available_beam_size": row["available_beam_size"],
            "unique_programs": row["unique_programs"],
            "mean_pairwise_hamming": row["mean_pairwise_hamming"],
            "distinct_first_actions": row["distinct_first_actions"],
            "distinct_non_full_positions": row["distinct_non_full_positions"],
        }
        for row in samples
    ]
    score_gap_rows: list[dict[str, Any]] = []
    for row in samples:
        if row["w_failure_class"] != "RANKING_FAILURE_W":
            continue
        uid = str(row["uid"])
        top_candidate, _ = _candidate_for_rank(uid, 1, ranked_by_uid, result_by_program)
        correct_candidates = [
            item for item in ranked_by_uid[uid] if result_by_program[str(item["program_id"])]["correct"]
        ]
        best = min(correct_candidates, key=lambda item: int(item["rank"]))
        score_gap_rows.append(
            {
                "row_kind": "sample",
                "scope": row["benchmark_family"],
                "uid": uid,
                "top1_score": top_candidate["sequence_score"],
                "best_correct_rank": best["rank"],
                "best_correct_score": best["sequence_score"],
                "score_gap": float(top_candidate["sequence_score"]) - float(best["sequence_score"]),
            }
        )
    for scope in ("overall", "chartqa", "textvqa", "mmmu_pro", "pope"):
        current = [
            row["score_gap"]
            for row in score_gap_rows
            if row["row_kind"] == "sample"
            and (scope == "overall" or row["scope"] == scope)
        ]
        score_gap_rows.append(
            {
                "row_kind": "summary",
                "scope": scope,
                "uid": None,
                "n": len(current),
                "median": _quantile(current, 0.5),
                "q25": _quantile(current, 0.25),
                "q75": _quantile(current, 0.75),
            }
        )
    intensity_candidates: list[dict[str, Any]] = []
    for row in samples:
        if row["dense_correct"]:
            continue
        uid = str(row["uid"])
        top, _ = _candidate_for_rank(uid, 1, ranked_by_uid, result_by_program)
        if not row["top1_correct"]:
            intensity_candidates.append({"category": "TOP1_WRONG", **top})
        if row["first_correct_rank"] is not None:
            correct, _ = _candidate_for_rank(
                uid, int(row["first_correct_rank"]), ranked_by_uid, result_by_program
            )
            intensity_candidates.append({"category": "BEST_CORRECT", **correct})
        if row["w_failure_class"] == "GENERATION_FAILURE_W":
            intensity_candidates.extend(
                {"category": "ALL_WRONG_BEAM", **candidate}
                for candidate in ranked_by_uid[uid]
            )
    intensity_rows: list[dict[str, Any]] = [
        {
            "row_kind": "candidate",
            "scope": row["benchmark_family"],
            "uid": row["uid"],
            "program_id": row["program_id"],
            "category": row["category"],
            "rank": row["rank"],
            "non_full_count": row["non_full_count"],
            "non_full_fraction": row["non_full_fraction"],
            "first_non_full_delay": row["first_non_full_delay"],
        }
        for row in intensity_candidates
    ]
    for scope in ("overall", "chartqa", "textvqa", "mmmu_pro", "pope"):
        for category in ("TOP1_WRONG", "BEST_CORRECT", "ALL_WRONG_BEAM"):
            current = [
                row
                for row in intensity_rows
                if row["row_kind"] == "candidate"
                and row["category"] == category
                and (scope == "overall" or row["scope"] == scope)
            ]
            intensity_rows.append(
                {
                    "row_kind": "summary",
                    "scope": scope,
                    "category": category,
                    "n": len(current),
                    "mean_non_full_count": _mean([row["non_full_count"] for row in current]),
                    "median_non_full_count": _quantile([row["non_full_count"] for row in current], 0.5),
                    "mean_first_non_full_delay": _mean([row["first_non_full_delay"] for row in current]),
                }
            )
    successful_w_candidates = []
    for row in samples:
        if row["dense_correct"]:
            continue
        for candidate in ranked_by_uid[str(row["uid"])]:
            if result_by_program[str(candidate["program_id"])]["correct"]:
                successful_w_candidates.append(candidate)
    action_pattern_rows: list[dict[str, Any]] = []
    for scope in ("overall", "chartqa", "textvqa", "mmmu_pro", "pope"):
        current = [
            row for row in successful_w_candidates if scope == "overall" or row["benchmark_family"] == scope
        ]
        first_actions = Counter(
            next((action for action in row["actions"] if action != "FULL"), "ALL_FULL")
            for row in current
        )
        action_totals = Counter(action for row in current for action in row["actions"] if action != "FULL")
        motifs = Counter(
            ">".join(row["actions"][: min(3, len(row["actions"]))]) for row in current
        )
        for value, count in sorted(first_actions.items()):
            action_pattern_rows.append({"scope": scope, "statistic": "first_non_full_action", "value": value, "count": count})
        for value, count in sorted(action_totals.items()):
            action_pattern_rows.append({"scope": scope, "statistic": "total_non_full_action", "value": value, "count": count})
        for value, count in motifs.most_common(10):
            action_pattern_rows.append({"scope": scope, "statistic": "top3_prefix_motif", "value": value, "count": count})
    atomic_csv(output_root / "metrics/top1_intervention_statistics.csv", top1_stats)
    atomic_csv(output_root / "metrics/beam_oracle_summary.csv", beam_summary)
    atomic_csv(output_root / "metrics/rescue_at_k.csv", rescue_rows)
    atomic_csv(output_root / "metrics/first_correct_rank.csv", first_rank_rows)
    atomic_csv(output_root / "metrics/w_failure_decomposition.csv", w_failure_rows)
    atomic_csv(output_root / "metrics/c_failure_decomposition.csv", c_failure_rows)
    atomic_csv(output_root / "metrics/score_gap_analysis.csv", score_gap_rows)
    atomic_csv(output_root / "metrics/all_full_inclusion.csv", all_full_rows)
    atomic_csv(output_root / "metrics/beam_diversity.csv", diversity_rows)
    atomic_csv(output_root / "metrics/nonfull_intensity.csv", intensity_rows)
    atomic_csv(output_root / "metrics/action_pattern_summary.csv", action_pattern_rows)
    atomic_csv(output_root / "metrics/benchmark_breakdown.csv", benchmark_rows)
    _write_figures(
        output_root,
        rescue_rows,
        first_rank_rows,
        w_failure_rows,
        c_failure_rows,
        all_full_rows,
        diversity_rows,
        score_gap_rows,
        intensity_rows,
    )
    overall = next(row for row in benchmark_rows if row["scope"] == "overall")
    w_rows = [row for row in samples if not row["dense_correct"]]
    c_rows = [row for row in samples if row["dense_correct"]]
    ranking_w = int(overall["ranking_failure_w"])
    generation_w = int(overall["generation_failure_w"])
    decision, recommendation = _decision_case(
        ranking_failures=ranking_w, generation_failures=generation_w
    )
    top1_w = next(
        row for row in top1_stats if row["scope"] == "overall" and row["dense_outcome"] == "W"
    )
    gaps = [row["score_gap"] for row in score_gap_rows if row["row_kind"] == "sample"]
    failed_w_diversity = [
        row["mean_pairwise_hamming"]
        for row in samples
        if row["w_failure_class"] == "GENERATION_FAILURE_W"
    ]
    top_wrong_intensity = [
        row["non_full_count"]
        for row in intensity_rows
        if row["row_kind"] == "candidate" and row["category"] == "TOP1_WRONG"
    ]
    correct_intensity = [
        row["non_full_count"]
        for row in intensity_rows
        if row["row_kind"] == "candidate" and row["category"] == "BEST_CORRECT"
    ]
    c_regressions = [row for row in c_rows if not row["top1_correct"]]
    c1 = sum(row["c_all_full_class"] == "C1_ALL_FULL_RANKED_BELOW_WRONG_TOP1" for row in c_regressions)
    c2 = sum(row["c_all_full_class"] == "C2_ALL_FULL_ABSENT" for row in c_regressions)
    text = next(row for row in benchmark_rows if row["scope"] == "textvqa")
    mmmu = next(row for row in benchmark_rows if row["scope"] == "mmmu_pro")
    overall_beam = next(row for row in beam_summary if row["scope"] == "overall")
    duplicate_rate = 1.0 - len(unique) / len(ranked)
    summary = f"""# Program predictor beam-oracle audit summary

## Result

- Main bottleneck: **{decision}**.
- Triggered population: **{len(samples)}** = {len(w_rows)} Dense-W + {len(c_rows)} Dense-C.
- Frozen unique candidates executed: **{len(candidates):,} / {len(unique):,}**.
- Mean unique programs per triggered beam: **{len(unique) / len(samples):.4f}**; duplicate-entry rate: **{duplicate_rate:.4%}**.
- Dense-W top-1 all-FULL / any-non-FULL: **{top1_w['top1_all_full']} / {top1_w['top1_any_non_full']}** ({top1_w['p_all_full']:.2%} / {top1_w['p_any_non_full']:.2%}).
- Dense-W mean unique programs / mean pairwise Hamming distance: **{overall_beam['mean_unique_programs_w']} / {overall_beam['mean_hamming_w']}**.
- W-to-C@1/@2/@4/@8: **{overall['w_to_c_at_1']} / {overall['w_to_c_at_2']} / {overall['w_to_c_at_4']} / {overall['w_to_c_at_8']}**.
- Additional rescues at ranks 2-8: **{overall['additional_w_rescues']}**.
- W top-1 success / ranking failure / generation failure: **{overall['w_to_c_at_1']} / {ranking_w} / {generation_w}**.

## Required answers

1. Stage-1-triggered samples audited: **{len(samples)}**.
2. Triggered Dense-W / Dense-C: **{len(w_rows)} / {len(c_rows)}**.
3. P(all-FULL | triggered W): **{top1_w['p_all_full']:.6f}**.
4. P(any non-FULL | triggered W): **{top1_w['p_any_non_full']:.6f}**.
5. W-to-C@1/@2/@4/@8: **{overall['w_to_c_at_1']}/{overall['w_to_c_at_2']}/{overall['w_to_c_at_4']}/{overall['w_to_c_at_8']}**.
6. Additional rank-2-8 rescues: **{overall['additional_w_rescues']}**.
7. Ranking failures among top-1 W failures: **{ranking_w}/{len(w_rows) - overall['w_to_c_at_1']} ({_safe_rate(ranking_w, len(w_rows) - overall['w_to_c_at_1']):.2%})**.
8. Generation failures among top-1 W failures: **{generation_w}/{len(w_rows) - overall['w_to_c_at_1']} ({_safe_rate(generation_w, len(w_rows) - overall['w_to_c_at_1']):.2%})**.
9. First-correct-rank counts are in `metrics/first_correct_rank.csv`.
10. Ranking-failure top1-minus-correct score gap median/IQR: **{_quantile(gaps, 0.5)} / [{_quantile(gaps, 0.25)}, {_quantile(gaps, 0.75)}]**.
11. Generation-failure W mean beam Hamming diversity: **{_mean(failed_w_diversity)}**.
12. Top-1-wrong versus best-correct median non-FULL actions: **{_quantile(top_wrong_intensity, 0.5)} versus {_quantile(correct_intensity, 0.5)}**.
13. All-FULL is present in **{sum(row['all_full_in_beam'] for row in c_rows)}/{len(c_rows)}** triggered-C beams.
14. The {len(c_regressions)} top-1 C-to-W regressions contain **{sum(row['c_failure_class'] == 'RANKING_FAILURE_C' for row in c_regressions)} ranking failures** and **{sum(row['c_failure_class'] == 'GENERATION_FAILURE_C' for row in c_regressions)} generation failures**; all-FULL C1/C2 is **{c1}/{c2}**.
15. TextVQA W-to-C@1/@8: **{text['w_to_c_at_1']}/{text['w_to_c_at_8']}**.
16. MMMU-Pro W-to-C@1/@8: **{mmmu['w_to_c_at_1']}/{mmmu['w_to_c_at_8']}**.
17. The quantitatively dominant W failure mode is **{decision}** ({ranking_w} ranking versus {generation_w} generation failures).
18. Exactly one next experiment is recommended: **{recommendation}**.
19. This oracle audit does not establish a deployable reranker, an inference-time oracle, or prospective benchmark improvement.

Layer-27 samples retain their denominator with exhaustive four-program support; no duplicate ranks were fabricated. External labels were used only after the complete candidate set was frozen.
"""
    atomic_text(output_root / "summaries/program_beam_oracle_audit_summary.md", summary)
    if decision == "ranking":
        positive = "more training-side correct programs rise to top-1 without losing preservation"
        negative = "beam-oracle capacity does not translate into top-1 selection"
    else:
        positive = "the enriched trigger state generates materially more valid corrective programs on held-out development data"
        negative = "candidate failure persists despite the minimal representation change"
    recommendation_text = f"""# Next Stage-2 recommendation

Recommend exactly one separately authorized experiment: **{recommendation}**.

- Why this is smallest: {generation_w} generation failures versus {ranking_w} ranking failures make the dominant unresolved limitation explicit without changing Stage 1, beam width, evaluator, or action semantics.
- Positive result: {positive}.
- Negative result: {negative}, supporting a stop or a separately reviewed strategic decision.
- Cannot conclude: beam-oracle correctness is not deployable, labels cannot select candidates at inference, and this audit does not validate any reranker or changed representation.

No follow-on experiment is authorized or executed in this phase.
"""
    atomic_text(output_root / "summaries/next_stage2_recommendation.md", recommendation_text)
    required = [
        path
        for path in output_root.rglob("*")
        if path.is_file()
        and "work" not in path.relative_to(output_root).parts
        and path.name != "artifact_manifest.json"
    ]
    artifact = {
        "schema_version": "program_beam_oracle_audit_artifact_manifest_v1",
        "contract_sha256": contract["contract_sha256"],
        "phase74_contract_sha256": contract["parent"]["phase74_contract_sha256"],
        "triggered_samples": len(samples),
        "unique_programs": len(candidates),
        "decision": decision,
        "files": {
            str(path.relative_to(output_root)): file_sha256(path) for path in sorted(required)
        },
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "artifact_manifest.json", artifact)
    atomic_json(
        output_root / "work/aggregation_complete.json",
        {
            "contract_sha256": contract["contract_sha256"],
            "triggered_samples": len(samples),
            "unique_programs": len(candidates),
            "w_to_c_at_1": overall["w_to_c_at_1"],
            "w_to_c_at_8": overall["w_to_c_at_8"],
            "ranking_failure_w": ranking_w,
            "generation_failure_w": generation_w,
            "decision": decision,
            "artifact_manifest_sha256": file_sha256(output_root / "artifact_manifest.json"),
            "completed_at": utc_now(),
        },
    )
    print(json.dumps(read_json(output_root / "work/aggregation_complete.json"), sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "top1-worker",
            "aggregate-top1",
            "candidate-worker",
            "collect-candidates",
            "aggregate",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config)
    elif args.command == "top1-worker":
        top1_worker(args.config, args.rank, args.resume)
    elif args.command == "aggregate-top1":
        aggregate_top1(args.config)
    elif args.command == "candidate-worker":
        candidate_worker(args.config, args.rank, args.resume)
    elif args.command == "collect-candidates":
        collect_candidates(args.config)
    elif args.command == "aggregate":
        aggregate(args.config)


def _assign_uid_workers(
    rows: Sequence[Mapping[str, Any]], world_size: int
) -> tuple[list[dict[str, Any]], list[int]]:
    """Greedily balance suffix-layer work while keeping each UID on one GPU."""
    if world_size < 1:
        raise ValueError("world_size must be positive")
    by_uid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_uid[str(row["uid"])].append(row)
    costs = sorted(
        (
            sum(len(item["actions"]) for item in items),
            uid,
        )
        for uid, items in by_uid.items()
    )
    loads = [0] * int(world_size)
    assignment: dict[str, int] = {}
    for cost, uid in reversed(costs):
        rank = min(range(int(world_size)), key=lambda item: (loads[item], item))
        assignment[uid] = rank
        loads[rank] += cost
    return [
        {**dict(row), "worker_rank": assignment[str(row["uid"])]}
        for row in rows
    ], loads


def _validate_execution_coverage(
    expected: Sequence[Mapping[str, Any]], observed: Sequence[Mapping[str, Any]]
) -> None:
    expected_ids = Counter(str(row["program_id"]) for row in expected)
    observed_ids = Counter(str(row["program_id"]) for row in observed)
    if expected_ids != observed_ids or any(count != 1 for count in observed_ids.values()):
        raise RuntimeError(
            "candidate execution coverage differs: "
            f"expected={len(expected_ids)} observed={len(observed_ids)} "
            f"duplicates={sum(count - 1 for count in observed_ids.values() if count > 1)}"
        )


def _decision_case(
    *, ranking_failures: int, generation_failures: int
) -> tuple[str, str]:
    """Choose the one next direction from the quantitatively dominant failure."""
    if ranking_failures > generation_failures:
        return "ranking", "training-side program preference/ranking objective"
    if generation_failures > ranking_failures:
        return (
            "generation/representation",
            "one minimal trigger-state representation enrichment experiment",
        )
    return (
        "mixed",
        "one minimal trigger-state representation enrichment experiment",
    )


def _validate_parity_rows(
    expected: Sequence[Mapping[str, Any]],
    observed: Sequence[Mapping[str, Any]],
    *,
    expected_w_to_c: int,
    expected_c_to_w: int,
    expected_net: int,
) -> dict[str, int]:
    expected_uids = Counter(str(row["uid"]) for row in expected)
    observed_uids = Counter(str(row["uid"]) for row in observed)
    if expected_uids != observed_uids or any(count != 1 for count in observed_uids.values()):
        raise RuntimeError("top-1 parity UID coverage differs")
    if not all(all(bool(value) for value in row["parity"].values()) for row in observed):
        raise RuntimeError("top-1 parity field mismatch")
    w_to_c = sum(
        not bool(row["dense_correct"]) and bool(row["program_correct"])
        for row in observed
    )
    c_to_w = sum(
        bool(row["dense_correct"]) and not bool(row["program_correct"])
        for row in observed
    )
    net = w_to_c - c_to_w
    if (w_to_c, c_to_w, net) != (
        int(expected_w_to_c),
        int(expected_c_to_w),
        int(expected_net),
    ):
        raise RuntimeError(
            f"top-1 transition parity differs: {(w_to_c, c_to_w, net)}"
        )
    return {"rows": len(observed), "w_to_c": w_to_c, "c_to_w": c_to_w, "net": net}


if __name__ == "__main__":
    main()
