#!/usr/bin/env python3
"""Verify that complete pre-repair We-Math records are safe to resume around."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_label_regeneration import find_completed_record, index_existing_records
from label_regeneration.data import safe_sample_filename


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--old-contract", required=True)
    parser.add_argument("--new-contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_root = Path(args.output_root).resolve()
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    existing = index_existing_records(output_root)
    completed: list[dict] = []
    contract_counts: Counter[str] = Counter()
    for row in rows:
        path = find_completed_record(
            existing,
            filename=safe_sample_filename(row["uid"]),
            uid=row["uid"],
            contract_hash=args.new_contract,
            compatible_contract_hashes=(args.old_contract,),
        )
        if path is None:
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        contract = str(record["runtime"]["contract_sha256"])
        contract_counts[contract] += 1
        completed.append(
            {
                "uid": row["uid"],
                "path": str(path),
                "contract_sha256": contract,
                "completed_simulations": int(record["mcts"]["completed_simulations"]),
                "requested_simulations": int(record["mcts"]["requested_simulations"]),
            }
        )

    checks = {
        "manifest_has_4544_records": len(rows) == 4544,
        "manifest_uids_unique": len({row["uid"] for row in rows}) == len(rows),
        "at_least_observed_1150_complete": len(completed) >= 1150,
        "all_retained_contracts_explicitly_compatible": set(contract_counts).issubset(
            {args.old_contract, args.new_contract}
        ),
        "all_retained_simulations_complete": all(
            row["completed_simulations"] == row["requested_simulations"]
            for row in completed
        ),
    }
    report = {
        "schema_version": "wemath2pro_resume_compatibility_audit_v2",
        "passed": all(checks.values()),
        "checks": checks,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "old_contract_sha256": args.old_contract,
        "new_contract_sha256": args.new_contract,
        "manifest_record_count": len(rows),
        "compatible_complete_record_count": len(completed),
        "remaining_record_count": len(rows) - len(completed),
        "retained_contract_counts": dict(sorted(contract_counts.items())),
        "completed_records": completed,
    }
    output = Path(args.output).resolve()
    atomic_json(output, report)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256(output.read_bytes()).hexdigest()}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "passed", "compatible_complete_record_count", "remaining_record_count",
        "retained_contract_counts",
    )}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
