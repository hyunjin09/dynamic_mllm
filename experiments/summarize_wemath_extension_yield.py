#!/usr/bin/env python3
"""Summarize whether We-Math MCTS extensions discover routes after simulation 400."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cache_root = Path(args.cache_root).resolve()
    paths = sorted(cache_root.glob("shard_*_of_*/samples/*.json"))
    seen: dict[str, str] = {}
    requested_counts: Counter[int] = Counter()
    extended = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        uid = str(record["sample"]["uid"])
        if uid in seen:
            raise ValueError(f"duplicate completed UID {uid}: {seen[uid]} and {path}")
        seen[uid] = str(path)
        mcts = record["mcts"]
        requested = int(mcts["requested_simulations"])
        completed = int(mcts["completed_simulations"])
        if requested != completed:
            raise ValueError(f"nonterminal record {uid}: {completed}/{requested}")
        requested_counts[requested] += 1
        if requested != 600:
            continue
        simulations = mcts["simulations"]
        if len(simulations) != 600:
            raise ValueError(f"invalid simulation trace for {uid}")
        pre400_success = any(float(row["reward"]) > 0 for row in simulations[:400])
        post400_success_rows = [
            row for row in simulations[400:] if float(row["reward"]) > 0
        ]
        unique_new = [row for row in post400_success_rows if not bool(row["evaluation_reused"])]
        successful_masks = {row["evaluated_mask_key"] for row in post400_success_rows}
        unique_new_masks = {row["evaluated_mask_key"] for row in unique_new}
        extended.append(
            {
                "uid": uid,
                "path": str(path),
                "extension_reason": mcts.get("extension_reason"),
                "pre400_simulation_success": pre400_success,
                "found_success_after_400": bool(post400_success_rows),
                "first_success_simulation_after_400": (
                    min(int(row["simulation"]) for row in post400_success_rows)
                    if post400_success_rows
                    else None
                ),
                "successful_simulations_401_600": len(post400_success_rows),
                "distinct_successful_masks_401_600": len(successful_masks),
                "new_successful_masks_401_600": len(unique_new_masks),
                "final_successful_mask_count": len(mcts["successful_masks"]),
            }
        )

    successes = [row for row in extended if row["found_success_after_400"]]
    checks = {
        "completed_uids_unique": len(seen) == len(paths),
        "all_extensions_had_no_pre400_simulation_success": not any(
            row["pre400_simulation_success"] for row in extended
        ),
        "all_extension_reasons_match": all(
            row["extension_reason"] == "no_correcting_route_after_400"
            for row in extended
        ),
    }
    report = {
        "schema_version": "wemath2pro_extension_yield_snapshot_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "cache_root": str(cache_root),
        "snapshot_record_count": len(paths),
        "requested_simulation_counts": {
            str(key): value for key, value in sorted(requested_counts.items())
        },
        "extended_to_600_count": len(extended),
        "extensions_finding_success_after_400_count": len(successes),
        "extensions_finding_success_after_400_fraction": (
            len(successes) / len(extended) if extended else None
        ),
        "new_successful_masks_found_401_600_total": sum(
            row["new_successful_masks_401_600"] for row in extended
        ),
        "successful_extension_records": successes,
        "all_extension_records": extended,
    }
    output = Path(args.output).resolve()
    atomic_json(output, report)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256(output.read_bytes()).hexdigest()}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({
        "snapshot_record_count": report["snapshot_record_count"],
        "requested_simulation_counts": report["requested_simulation_counts"],
        "extended_to_600_count": report["extended_to_600_count"],
        "extensions_finding_success_after_400_count": report[
            "extensions_finding_success_after_400_count"
        ],
        "new_successful_masks_found_401_600_total": report[
            "new_successful_masks_found_401_600_total"
        ],
        "passed": report["passed"],
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
