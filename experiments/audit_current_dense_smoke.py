#!/usr/bin/env python3
"""Enforce complete repeatability, feature parity, and provenance smoke gates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from dense_failure_stage1.contract import (
    canonical_json_sha256,
    file_sha256,
    validate_smoke_gate,
)
from experiments.run_current_dense_regeneration import (
    feature_provenance,
    validate_feature_provenance,
)
from tools.research_analysis.dense_failure_stage1 import canonical_payload_sha256


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_once(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"refusing to overwrite differing smoke audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def compare(left: list[dict], right: list[dict]) -> dict:
    left_index = {row["uid"]: row for row in left}
    right_index = {row["uid"]: row for row in right}
    if len(left_index) != len(left) or len(right_index) != len(right):
        raise ValueError("duplicate UID in smoke output")
    fields = (
        "dense_generated_token_ids",
        "dense_generated_text",
        "normalized_prediction",
        "evaluator_score",
        "current_dense_correct",
        "current_dense_wrong",
    )
    return {
        "left_records": len(left),
        "right_records": len(right),
        "uid_set_identical": set(left_index) == set(right_index),
        "field_mismatch_counts": {
            field: sum(
                left_index[uid].get(field) != right_index.get(uid, {}).get(field)
                for uid in left_index
            )
            for field in fields
        },
        "mismatched_uids": sorted(
            uid
            for uid in left_index
            if any(
                left_index[uid].get(field)
                != right_index.get(uid, {}).get(field)
                for field in fields
            )
        ),
    }


def comparison_passed(comparison: dict) -> bool:
    return (
        comparison["left_records"] == 24
        and comparison["right_records"] == 24
        and comparison["uid_set_identical"]
        and not any(comparison["field_mismatch_counts"].values())
    )


def exact_uid_coverage(rows: list[dict], expected_uids: list[str]) -> bool:
    actual = [row["uid"] for row in rows]
    return (
        len(actual) == 24
        and len(actual) == len(set(actual))
        and set(actual) == set(expected_uids)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--smoke-manifest", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    root = Path(args.output_root)
    smoke_root = root / "smoke"
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    if canonical_payload_sha256(contract) != contract["contract_sha256"]:
        raise ValueError("frozen dense contract self-hash is invalid")
    if (
        file_sha256(Path(args.candidate_manifest))
        != contract["integrity"]["candidate_manifest_sha256"]
    ):
        raise ValueError("candidate manifest differs from the frozen contract")
    if (
        file_sha256(Path(args.smoke_manifest))
        != contract["integrity"]["smoke_manifest_sha256"]
    ):
        raise ValueError("smoke manifest differs from the frozen contract")
    smoke_value = json.loads(Path(args.smoke_manifest).read_text(encoding="utf-8"))
    expected_rows = smoke_value["records"]
    expected_uids = [row["uid"] for row in expected_rows]
    expected_index = {row["uid"]: row for row in expected_rows}
    if len(expected_uids) != 24 or len(set(expected_uids)) != 24:
        raise ValueError("frozen smoke manifest is not exactly 24 unique UIDs")

    rank_markers = sorted((smoke_root / "shards").glob("rank*.complete.json"))
    if len(rank_markers) != 4:
        raise SystemExit(
            f"smoke partial-rank failure: found {len(rank_markers)} / 4 rank markers"
        )
    run1 = []
    run2 = []
    hook = []
    feature_by_uid = {}
    resume_pass = True
    marker_hashes = {}
    observed_ranks = set()
    for marker_path in rank_markers:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        rank = int(marker.get("rank", -1))
        observed_ranks.add(rank)
        expected_rank_uids = [
            uid for index, uid in enumerate(expected_uids) if index % 4 == rank
        ]
        if (
            marker.get("schema_version") != "current_dense_smoke_rank_marker_v1"
            or marker.get("contract_id") != contract["contract_sha256"]
            or marker.get("run_id") != args.run_id
            or marker.get("world_size") != 4
            or marker.get("records") != len(expected_rank_uids)
            or marker.get("uids") != expected_rank_uids
            or marker.get("completion_state") != "completed_with_features"
        ):
            raise SystemExit(f"smoke rank marker provenance failed: {marker_path}")
        rank_paths = {
            "run1": Path(marker["run1_path"]),
            "run2": Path(marker["run2_path"]),
            "hook": Path(marker["hook_path"]),
            "feature": Path(marker["feature_path"]),
        }
        for name, path in rank_paths.items():
            recorded = marker[f"{name}_sha256"]
            if not path.is_file() or file_sha256(path) != recorded:
                raise SystemExit(f"smoke rank artifact hash failed: {path}")
        rank_run1 = read_jsonl(rank_paths["run1"])
        rank_run2 = read_jsonl(rank_paths["run2"])
        rank_hook = read_jsonl(rank_paths["hook"])
        if any(
            [row["uid"] for row in rows] != expected_rank_uids
            for rows in (rank_run1, rank_run2, rank_hook)
        ):
            raise SystemExit(f"smoke rank UID sequence failed: rank {rank}")
        run1.extend(rank_run1)
        run2.extend(rank_run2)
        hook.extend(rank_hook)
        payload = torch.load(rank_paths["feature"], map_location="cpu", weights_only=False)
        expected_rank_provenance = feature_provenance(
            contract,
            run_id=args.run_id,
            generation_sha256=marker["hook_sha256"],
        )
        validate_feature_provenance(payload, expected_rank_provenance)
        if payload.get("uids") != expected_rank_uids:
            raise SystemExit(f"smoke rank feature UID sequence failed: rank {rank}")
        for offset, uid in enumerate(payload["uids"]):
            if uid in feature_by_uid:
                raise SystemExit(f"duplicate smoke feature UID: {uid}")
            feature_by_uid[uid] = {
                key: payload[key][offset]
                for key in ("text_final", "text_mean", "visual_mean")
            }
        resume_pass = resume_pass and (
            marker.get("exact_feature_resume_accepted") is True
            and marker.get("incompatible_mode_rejected") is True
            and marker.get("feature_provenance") == expected_rank_provenance
        )
        marker_hashes[marker_path.name] = file_sha256(marker_path)
    if observed_ranks != {0, 1, 2, 3}:
        raise SystemExit(f"smoke rank coverage failed: {sorted(observed_ranks)}")
    order = {uid: index for index, uid in enumerate(expected_uids)}
    run1.sort(key=lambda row: order[row["uid"]])
    run2.sort(key=lambda row: order[row["uid"]])
    hook.sort(key=lambda row: order[row["uid"]])
    run1_path = smoke_root / "smoke_run1.jsonl"
    run2_path = smoke_root / "smoke_run2.jsonl"
    hook_path = smoke_root / "smoke_hook_run.jsonl"
    write_once(
        run1_path,
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in run1),
    )
    write_once(
        run2_path,
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in run2),
    )
    write_once(
        hook_path,
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in hook),
    )
    coverage_pass = all(
        exact_uid_coverage(rows, expected_uids) for rows in (run1, run2, hook)
    )
    if not coverage_pass:
        raise SystemExit("smoke output UID completeness/uniqueness failed")

    repeat = compare(run1, run2)
    hook_parity = compare(run1, hook)
    repeat_pass = comparison_passed(repeat)
    hook_pass = comparison_passed(hook_parity)
    evaluator_pass = (
        repeat["field_mismatch_counts"]["evaluator_score"] == 0
        and repeat["field_mismatch_counts"]["current_dense_correct"] == 0
        and hook_parity["field_mismatch_counts"]["evaluator_score"] == 0
        and hook_parity["field_mismatch_counts"]["current_dense_correct"] == 0
    )
    contract_pass = all(
        row.get("contract_id") == contract["contract_sha256"]
        and row.get("model_revision") == contract["integrity"]["model"]["revision"]
        and row.get("source_manifest_sha256")
        == contract["integrity"]["candidate_manifest_sha256"]
        for rows in (run1, run2, hook)
        for row in rows
    )
    image_sha_pass = all(
        row.get("image_content_sha256")
        == row.get("consumed_image_sha256")
        == expected_index[row["uid"]]["image_content_sha256"]
        for rows in (run1, run2, hook)
        for row in rows
    )

    feature_path = smoke_root / "smoke_hook_features.pt"
    expected_provenance = feature_provenance(
        contract,
        run_id=args.run_id,
        generation_sha256=file_sha256(hook_path),
    )
    feature_payload = {
        "schema_version": "current_dense_features_v1",
        "provenance": expected_provenance,
        "uids": expected_uids,
        "layer_ids": list(range(28)),
        "text_final": torch.stack([feature_by_uid[uid]["text_final"] for uid in expected_uids]),
        "text_mean": torch.stack([feature_by_uid[uid]["text_mean"] for uid in expected_uids]),
        "visual_mean": torch.stack([feature_by_uid[uid]["visual_mean"] for uid in expected_uids]),
    }
    if feature_path.exists():
        existing = torch.load(feature_path, map_location="cpu", weights_only=False)
        validate_feature_provenance(existing, expected_provenance)
        if existing.get("uids") != expected_uids:
            raise ValueError("existing merged smoke feature UID order differs")
        feature_payload = existing
    else:
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feature_payload, feature_path)
    validate_feature_provenance(feature_payload, expected_provenance)
    feature_shapes = {
        key: list(feature_payload[key].shape)
        for key in ("text_final", "text_mean", "visual_mean")
    }
    feature_valid = (
        feature_payload.get("uids") == [row["uid"] for row in hook]
        and feature_payload.get("layer_ids") == list(range(28))
        and all(
            shape[0] == 24
            and shape[1] == 28
            and shape[2] == contract["features"]["schema"]["hidden_size"]
            for shape in feature_shapes.values()
        )
        and feature_payload.get("provenance") == expected_provenance
    )

    gate = {
        "schema_version": "current_dense_smoke_gate_v1",
        "contract_id": contract["contract_sha256"],
        "candidate_manifest_sha256": contract["integrity"][
            "candidate_manifest_sha256"
        ],
        "smoke_manifest_sha256": contract["integrity"]["smoke_manifest_sha256"],
        "run_id": args.run_id,
        "records": 24,
        "missing_uids": 0,
        "duplicate_uids": 0,
        "repeatability_pass": repeat_pass,
        "evaluator_parity_pass": evaluator_pass,
        "image_sha_pass": image_sha_pass,
        "contract_integrity_pass": contract_pass,
        "global_completeness_pass": coverage_pass,
        "resume_validation_pass": resume_pass,
        "hook_token_parity_pass": hook_pass,
        "feature_provenance_pass": feature_valid,
        "artifact_sha256": {
            "smoke_run1.jsonl": file_sha256(run1_path),
            "smoke_run2.jsonl": file_sha256(run2_path),
            "smoke_hook_run.jsonl": file_sha256(hook_path),
            "smoke_hook_features.pt": file_sha256(feature_path),
            **marker_hashes,
        },
        "feature_shapes": feature_shapes,
    }
    validate_smoke_gate(
        gate,
        contract_id=contract["contract_sha256"],
        candidate_manifest_sha256=contract["integrity"][
            "candidate_manifest_sha256"
        ],
        require_features=True,
    )

    counts = Counter((row["dataset"], row["historical_bucket"]) for row in run1)
    smoke_md = f"""# Current Dense Determinism Smoke

- Records: {len(run1)} / 24; missing UIDs: 0; duplicate UIDs: 0
- Stratification: {dict(sorted((f'{key[0]}/{key[1]}', value) for key, value in counts.items()))}
- Contract ID: `{contract['contract_sha256']}`
- Contract/source/model hashes on every row: **{'PASS' if contract_pass else 'FAIL'}**
- Immediately consumed image SHA checks: **{'PASS' if image_sha_pass else 'FAIL'}**
- Generated token ID mismatches across repeats: {repeat['field_mismatch_counts']['dense_generated_token_ids']}
- Generated text mismatches: {repeat['field_mismatch_counts']['dense_generated_text']}
- Normalized-answer mismatches: {repeat['field_mismatch_counts']['normalized_prediction']}
- Evaluator-score mismatches: {repeat['field_mismatch_counts']['evaluator_score']}
- Correctness mismatches: {repeat['field_mismatch_counts']['current_dense_correct']}
- Exact resume accepted and incompatible resume rejected: **{'PASS' if resume_pass else 'FAIL'}**
- Current-runtime deterministic repeatability: **{'PASS' if repeat_pass else 'FAIL'}**

This gate compares two native dense all-on runs under the same frozen current runtime.
Historical output parity is not part of this decision.
"""
    hook_md = f"""# Dense Feature-Extraction Parity

- Records: {len(hook)} / 24
- Generated token ID mismatches vs hook-free run: {hook_parity['field_mismatch_counts']['dense_generated_token_ids']}
- Generated text mismatches: {hook_parity['field_mismatch_counts']['dense_generated_text']}
- Correctness mismatches: {hook_parity['field_mismatch_counts']['current_dense_correct']}
- Feature shapes: `{feature_shapes}`
- Feature schema SHA-256: `{contract['integrity']['feature_schema_sha256']}`
- Feature provenance and shard hash: **{'PASS' if feature_valid else 'FAIL'}**
- Passive feature extraction exact-token parity: **{'PASS' if hook_pass else 'FAIL'}**

The collector observes post-layer dense states and returns no replacement output.
Feature extraction remains unavailable to full execution without this gate file.
"""
    write_once(smoke_root / "smoke_report.md", smoke_md)
    write_once(smoke_root / "hook_parity_report.md", hook_md)
    write_once(
        root / "features" / "feature_schema.json",
        json.dumps(contract["features"]["schema"], indent=2, sort_keys=True) + "\n",
    )
    if (
        canonical_json_sha256(contract["features"]["schema"])
        != contract["integrity"]["feature_schema_sha256"]
    ):
        raise ValueError("frozen feature schema hash differs")
    write_once(
        smoke_root / "smoke_gate.json",
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(gate, sort_keys=True))


if __name__ == "__main__":
    main()
