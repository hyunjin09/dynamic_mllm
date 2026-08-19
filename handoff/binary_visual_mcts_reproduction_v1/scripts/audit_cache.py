#!/usr/bin/env python3
"""Audit terminal route-cache records across all current and resumed shards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {str(row["uid"]) for row in manifest}
    records_by_uid: dict[str, list[str]] = defaultdict(list)
    invalid_records: list[dict] = []
    status_counts = Counter()
    positive_counts: list[int] = []
    record_paths = sorted((Path(args.output_root) / "raw_route_cache").glob("shard_*_of_*/samples/*.json"))
    for path in record_paths:
        try:
            record = read_json(path)
            uid = str(record["sample"]["uid"])
            checks = {
                "contract": record.get("runtime", {}).get("contract_sha256") == args.contract_sha256,
                "simulations": record.get("mcts", {}).get("completed_simulations") == record.get("mcts", {}).get("requested_simulations"),
                "candidates": isinstance(record.get("candidate_executions"), list),
                "uid_expected": uid in expected,
            }
            if not all(checks.values()):
                invalid_records.append({"path": str(path), "uid": uid, "checks": checks})
                continue
            records_by_uid[uid].append(str(path))
            status_counts[str(record["sample"].get("current_all_on_status"))] += 1
            positive_counts.append(len(record.get("successful_route_ids", [])))
        except Exception as exc:
            invalid_records.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    duplicates = {uid: paths for uid, paths in records_by_uid.items() if len(paths) > 1}
    completed = set(records_by_uid)
    error_paths = sorted((Path(args.output_root) / "raw_route_cache").glob("shard_*_of_*/errors/*.json"))
    passed = completed == expected and not duplicates and not invalid_records and not error_paths
    report = {
        "schema_version": "portable_mcts_cache_audit_v1",
        "passed": passed,
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "contract_sha256": args.contract_sha256,
        "expected_records": len(expected),
        "valid_terminal_records": len(completed),
        "missing_uids": sorted(expected - completed),
        "unexpected_uids": sorted(completed - expected),
        "duplicate_records": duplicates,
        "invalid_records": invalid_records,
        "error_record_paths": [str(path) for path in error_paths],
        "current_all_on_status_counts": dict(status_counts),
        "valid_route_count_summary": {
            "minimum": min(positive_counts) if positive_counts else None,
            "maximum": max(positive_counts) if positive_counts else None,
            "mean": sum(positive_counts) / len(positive_counts) if positive_counts else None,
            "zero_positive_records": sum(count == 0 for count in positive_counts),
        },
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256(output.read_bytes()).hexdigest()}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"passed": passed, "completed": len(completed), "expected": len(expected)}))
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
