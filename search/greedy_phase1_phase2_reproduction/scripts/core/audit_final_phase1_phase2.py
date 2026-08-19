#!/usr/bin/env python3
"""Independently verify checksums and row counts for the final 10k mask dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "10k_dataset_mask" / "final_phase1_phase2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def hash_and_count(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    line_count = 0
    byte_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            line_count += chunk.count(b"\n")
            byte_count += len(chunk)
    return digest.hexdigest(), line_count, byte_count


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    summary_path = input_dir / "summary.json"
    checksum_path = input_dir / "checksums.sha256"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("decision") != "pass_complete_no_ranking_constructed":
        raise RuntimeError(f"final summary is not accepted: {summary.get('decision')}")
    expected_rows = {
        "evaluated_mask_candidates.jsonl": int(summary["combined_unique_evaluated_candidates"]),
        "phase1_permutation_final_masks.jsonl": int(summary["phase1_permutation_finals"]),
        "phase2_route_requests.jsonl": int(summary["phase2_requests"]),
        "sample_index.jsonl": int(summary["samples"]),
    }
    expected_checksums: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        checksum, separator, filename = line.partition("  ")
        if not separator or not checksum or not filename:
            raise RuntimeError(f"invalid checksum row: {line}")
        expected_checksums[filename] = checksum

    file_rows = []
    for filename, expected_checksum in sorted(expected_checksums.items()):
        path = input_dir / filename
        if not path.is_file():
            raise RuntimeError(f"missing checksummed output: {path}")
        actual_checksum, line_count, byte_count = hash_and_count(path)
        if actual_checksum != expected_checksum:
            raise RuntimeError(f"checksum mismatch for {filename}")
        if filename in expected_rows and line_count != expected_rows[filename]:
            raise RuntimeError(
                f"row-count mismatch for {filename}: expected={expected_rows[filename]} actual={line_count}"
            )
        file_rows.append(
            {
                "filename": filename,
                "sha256": actual_checksum,
                "bytes": byte_count,
                "rows": line_count if filename.endswith(".jsonl") else None,
            }
        )

    temp_files = sorted(path.name for path in input_dir.glob("*.tmp"))
    if temp_files:
        raise RuntimeError(f"temporary outputs remain: {temp_files}")
    if int(summary["phase1_candidates"]) + int(summary["phase2_new_candidates"]) != int(
        summary["combined_unique_evaluated_candidates"]
    ):
        raise RuntimeError("combined candidate arithmetic mismatch")
    if int(summary["phase2_reused_phase1_requests"]) + int(summary["phase2_new_candidates"]) != int(
        summary["phase2_requests"]
    ):
        raise RuntimeError("Phase-2 request arithmetic mismatch")
    if int(summary["samples"]) != 10000:
        raise RuntimeError("unexpected final sample count")
    for phase in ("phase1_raw_audit", "phase2_raw_audit"):
        if int(summary[phase]["sample_files"]) != 10000 or int(summary[phase]["error_rows"]) != 0:
            raise RuntimeError(f"raw audit did not pass: {phase}")

    audit: dict[str, Any] = {
        "decision": "pass_final_phase1_phase2_integrity_audit",
        "dataset_version": summary["dataset_version"],
        "samples": summary["samples"],
        "phase1_candidates": summary["phase1_candidates"],
        "phase2_requests": summary["phase2_requests"],
        "phase2_reused_phase1_requests": summary["phase2_reused_phase1_requests"],
        "phase2_new_candidates": summary["phase2_new_candidates"],
        "combined_unique_evaluated_candidates": summary["combined_unique_evaluated_candidates"],
        "checksum_files_verified": len(file_rows),
        "row_counts_verified": expected_rows,
        "temporary_files": [],
        "files": file_rows,
    }
    output_path = input_dir / "audit_summary.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
