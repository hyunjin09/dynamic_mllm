#!/usr/bin/env python3
"""Build a self-contained capped cache from compatible terminal predecessors."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_label_regeneration import record_complete
from label_regeneration.data import safe_sample_filename


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--destination-root", required=True)
    parser.add_argument("--new-contract", required=True)
    parser.add_argument("--contract-artifact", required=True)
    parser.add_argument("--compatible-contract", action="append", default=[])
    parser.add_argument("--max-simulations-per-sample", type=int, required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    source_root = Path(args.source_root).resolve()
    destination_root = Path(args.destination_root).resolve()
    contract_artifact = Path(args.contract_artifact).resolve()
    if destination_root.exists() and any(destination_root.iterdir()):
        raise FileExistsError(f"destination must be absent or empty: {destination_root}")
    if args.max_simulations_per_sample != 400:
        raise ValueError("this staging action is frozen to a 400-simulation cap")
    contract = json.loads(contract_artifact.read_text(encoding="utf-8"))
    if contract.get("contract_sha256") != args.new_contract:
        raise ValueError("contract artifact hash does not match --new-contract")
    if contract.get("mcts", {}).get("max_simulations_per_sample") != 400:
        raise ValueError("contract artifact does not freeze the 400-simulation cap")

    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_uid = {str(row["uid"]): row for row in rows}
    if len(by_uid) != len(rows) or len(rows) != 4544:
        raise ValueError("manifest must contain exactly 4,544 unique UIDs")

    source_paths = sorted(source_root.glob("raw_route_cache/shard_*_of_*/samples/*.json"))
    seen: dict[str, Path] = {}
    retained: list[dict] = []
    excluded_above_cap: list[dict] = []
    contracts = tuple(sorted(set(args.compatible_contract)))
    for path in source_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        uid = str(record.get("sample", {}).get("uid") or "")
        if uid not in by_uid:
            raise ValueError(f"unexpected cache UID: {uid}")
        if uid in seen:
            raise ValueError(f"duplicate cache UID {uid}: {seen[uid]} and {path}")
        seen[uid] = path
        requested = int(record["mcts"]["requested_simulations"])
        if requested > args.max_simulations_per_sample:
            excluded_above_cap.append({"uid": uid, "requested_simulations": requested, "path": str(path)})
            continue
        if not record_complete(
            path,
            uid=uid,
            contract_hash=args.new_contract,
            compatible_contract_hashes=contracts,
            max_simulations_per_sample=args.max_simulations_per_sample,
        ):
            raise ValueError(f"nonterminal or incompatible retained record: {path}")
        retained.append({
            "uid": uid,
            "requested_simulations": requested,
            "source_path": str(path),
            "source_sha256": sha256(path.read_bytes()).hexdigest(),
        })

    sample_root = destination_root / "raw_route_cache" / "shard_legacy_of_cap400" / "samples"
    sample_root.mkdir(parents=True, exist_ok=False)
    for row in retained:
        source = Path(row["source_path"])
        destination = sample_root / safe_sample_filename(row["uid"])
        shutil.copy2(source, destination)
        digest = sha256(destination.read_bytes()).hexdigest()
        if digest != row["source_sha256"]:
            raise RuntimeError(f"copy checksum mismatch: {row['uid']}")
        row["destination_path"] = str(destination)
        row["destination_sha256"] = digest

    destination_manifest = destination_root / "manifest" / manifest_path.name
    destination_manifest.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest_path, destination_manifest)
    manifest_digest = sha256(destination_manifest.read_bytes()).hexdigest()
    destination_manifest.with_suffix(destination_manifest.suffix + ".sha256").write_text(
        f"{manifest_digest}  {destination_manifest.name}\n", encoding="utf-8"
    )
    destination_contract = destination_root / "frozen_execution_contract.json"
    shutil.copy2(contract_artifact, destination_contract)
    contract_digest = sha256(destination_contract.read_bytes()).hexdigest()
    destination_contract.with_suffix(destination_contract.suffix + ".sha256").write_text(
        f"{contract_digest}  {destination_contract.name}\n", encoding="utf-8"
    )

    counts = Counter(row["requested_simulations"] for row in retained)
    checks = {
        "manifest_count_4544": len(rows) == 4544,
        "source_uids_unique": len(seen) == len(source_paths),
        "all_retained_at_or_below_cap": all(
            row["requested_simulations"] <= args.max_simulations_per_sample for row in retained
        ),
        "all_above_cap_excluded": all(
            row["requested_simulations"] > args.max_simulations_per_sample
            for row in excluded_above_cap
        ),
        "destination_count_matches": len(list(sample_root.glob("*.json"))) == len(retained),
        "retained_and_remaining_cover_manifest": len(retained) + (len(rows) - len(retained)) == len(rows),
    }
    report = {
        "schema_version": "wemath2pro_cap400_resume_cache_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "max_simulations_per_sample": args.max_simulations_per_sample,
        "new_contract_sha256": args.new_contract,
        "contract_artifact_path": str(destination_contract),
        "contract_artifact_file_sha256": contract_digest,
        "compatible_contract_sha256": list(contracts),
        "manifest_path": str(destination_manifest),
        "manifest_sha256": manifest_digest,
        "manifest_count": len(rows),
        "source_terminal_record_count": len(source_paths),
        "retained_record_count": len(retained),
        "retained_requested_simulation_counts": {
            str(key): value for key, value in sorted(counts.items())
        },
        "excluded_above_cap_count": len(excluded_above_cap),
        "remaining_record_count": len(rows) - len(retained),
        "retained_records": retained,
        "excluded_above_cap_records": excluded_above_cap,
    }
    output = destination_root / "cap400_resume_audit_v1.json"
    atomic_json(output, report)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256(output.read_bytes()).hexdigest()}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({
        "passed": report["passed"],
        "retained_record_count": len(retained),
        "excluded_above_cap_count": len(excluded_above_cap),
        "remaining_record_count": report["remaining_record_count"],
        "retained_requested_simulation_counts": report["retained_requested_simulation_counts"],
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
