#!/usr/bin/env python3
"""Freeze the exact outcome-defined WeMath greedy-recovery population."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ANALYSIS = PROJECT / "outputs/wemath2pro_mcts_label_analysis_v1"
MCTS_ROOT = PROJECT / "outputs/label_regeneration/wemath2pro_cap400_v2"
OUTPUT = PROJECT / "outputs/label_regeneration/wemath2pro_greedy_recovery_v1"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MODEL_PATH = (
    "/data/dataset/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/"
    f"snapshots/{MODEL_REVISION}"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    suitability_path = ANALYSIS / "per_sample_training_suitability_v1.jsonl"
    cache_index_path = ANALYSIS / "cache_record_index_v1.jsonl"
    source_manifest_path = MCTS_ROOT / "manifest/wemath2pro_valid_mcts_v1.jsonl"
    source_contract_path = MCTS_ROOT / "frozen_execution_contract_cap400_v5.json"

    suitability = read_jsonl(suitability_path)
    cache_index = {row["uid"]: row for row in read_jsonl(cache_index_path)}
    source = {row["uid"]: row for row in read_jsonl(source_manifest_path)}
    selected = sorted(
        (
            row
            for row in suitability
            if row["current_all_on_status"] == "wrong" and int(row["raw_valid_routes"]) == 0
        ),
        key=lambda row: int(str(row["uid"]).split(":", 1)[1]),
    )
    if len(selected) != 2278:
        raise RuntimeError(f"recovery population drift: {len(selected)} != 2278")
    if len({row["uid"] for row in selected}) != len(selected):
        raise RuntimeError("duplicate recovery UID")
    if len({row["image_group_id"] for row in selected}) != 1104:
        raise RuntimeError("recovery image-group count drift")

    manifest_rows = []
    for diagnostic in selected:
        uid = diagnostic["uid"]
        if uid not in source or uid not in cache_index:
            raise RuntimeError(f"missing source/cache linkage for {uid}")
        index = cache_index[uid]
        record_path = PROJECT / index["record_path"]
        if not record_path.is_file():
            raise RuntimeError(f"missing MCTS record for {uid}: {record_path}")
        if sha256_file(record_path) != index["record_sha256"]:
            raise RuntimeError(f"MCTS record checksum mismatch for {uid}")
        manifest_rows.append(
            {
                **source[uid],
                "recovery_population": "current_all_on_wrong_and_zero_valid_cap400_mcts_routes",
                "mcts_evaluated_routes": int(diagnostic["evaluated_routes"]),
                "mcts_valid_routes": 0,
                "mcts_record_path": str(record_path.relative_to(PROJECT)),
                "mcts_record_sha256": index["record_sha256"],
                "mcts_requested_simulations": int(index["requested_simulations"]),
            }
        )

    manifest = OUTPUT / "manifest/recovery_manifest_v1.jsonl"
    atomic_text(
        manifest,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest_rows),
    )
    manifest_sha = sha256_file(manifest)
    atomic_text(manifest.with_suffix(manifest.suffix + ".sha256"), f"{manifest_sha}  {manifest.name}\n")

    contract = {
        "schema_version": "wemath2pro_greedy_recovery_contract_v1",
        "population": {
            "records": 2278,
            "image_groups": 1104,
            "definition": "current all-ON wrong AND zero valid routes in cap-400 MCTS cache",
            "manifest_path": str(manifest.relative_to(PROJECT)),
            "manifest_sha256": manifest_sha,
        },
        "model": {"path": MODEL_PATH, "revision": MODEL_REVISION},
        "environment": {"venv": ".venv", "transformers": "5.3.0"},
        "executor": "label_regeneration.runtime.RouteEvaluator/current verified binary executor",
        "route_semantics": {"bits": 28, "one": "VISUAL_ON", "zero": "TEXT_ONLY"},
        "processing": {"native_qwen_defaults": True, "custom_max_image_tokens": None},
        "generation": {"deterministic_greedy": True, "max_new_tokens": 96},
        "scoring": {
            "metric": "wemath2pro_mathruler_accuracy",
            "correctness_threshold": 1.0,
            "timeout_seconds": 5.0,
        },
        "phase1": {
            "orders": [
                "early_to_late",
                "late_to_early",
                "center_out",
                "outside_in",
                *[f"random:{seed}" for seed in range(20260714, 20260720)],
            ],
            "score_tolerance": 1e-9,
        },
        "phase2": {
            "seed": 20260720,
            "random_per_budget": 2,
            "local_per_operation": 4,
            "max_bases": 3,
        },
        "derived_supervision": {
            "max_valid_routes_per_sample": 50,
            "raw_successes_truncated": False,
        },
        "source_hashes": {
            "suitability": sha256_file(suitability_path),
            "cache_index": sha256_file(cache_index_path),
            "mcts_manifest": sha256_file(source_manifest_path),
            "mcts_contract": sha256_file(source_contract_path),
        },
    }
    contract_path = OUTPUT / "frozen_execution_contract_v1.json"
    atomic_text(contract_path, json.dumps(contract, indent=2, sort_keys=True) + "\n")
    contract_sha = sha256_file(contract_path)
    atomic_text(
        contract_path.with_suffix(contract_path.suffix + ".sha256"),
        f"{contract_sha}  {contract_path.name}\n",
    )
    audit = {
        "status": "PASS",
        "records": len(manifest_rows),
        "unique_uids": len({row["uid"] for row in manifest_rows}),
        "unique_image_groups": len({row["image_group_id"] for row in manifest_rows}),
        "difficulty_counts": dict(sorted(Counter(row["difficulty"] for row in manifest_rows).items())),
        "all_current_all_on_wrong": True,
        "all_cap400_mcts_valid_route_count_zero": True,
        "all_mcts_record_checksums_pass": True,
        "manifest_sha256": manifest_sha,
        "contract_file_sha256": contract_sha,
    }
    audit_path = OUTPUT / "manifest/recovery_manifest_audit_v1.json"
    atomic_text(audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    atomic_text(
        audit_path.with_suffix(audit_path.suffix + ".sha256"),
        f"{sha256_file(audit_path)}  {audit_path.name}\n",
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
