#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_CONFIG = Path("configs/sequential_four_action_label_conversion.yaml")


def _claim_path(claim_root: Path, uid: str) -> Path:
    return claim_root / f"{sha256(uid.encode()).hexdigest()}.json"


def prepare_deferred_dataset_claims(
    rows: Iterable[dict[str, Any]],
    *,
    claim_root: Path,
    deferred_datasets: set[str],
    claimant: str,
    resume: bool,
) -> dict[str, Any]:
    """Preclaim deferred datasets so one launch processes only active datasets."""
    materialized = list(rows)
    uids = [str(row["uid"]) for row in materialized]
    if len(uids) != len(set(uids)):
        raise ValueError("source manifest contains duplicate UIDs")

    datasets = {str(row["dataset"]) for row in materialized}
    unknown = sorted(set(deferred_datasets) - datasets)
    if unknown:
        raise ValueError(f"deferred datasets absent from source manifest: {unknown}")

    deferred = [
        row for row in materialized if str(row["dataset"]) in deferred_datasets
    ]
    active = [
        row for row in materialized if str(row["dataset"]) not in deferred_datasets
    ]
    claim_root = Path(claim_root)
    claim_root.mkdir(parents=True, exist_ok=True)

    expected = {
        _claim_path(claim_root, str(row["uid"])): {
            "uid": str(row["uid"]),
            "claimant": str(claimant),
        }
        for row in deferred
    }
    existing = [path for path in expected if path.exists()]
    if existing and not resume:
        raise FileExistsError(
            f"claim exists without --resume: {existing[0]}"
        )

    created = 0
    verified = 0
    for path, payload in expected.items():
        if path.exists():
            observed = json.loads(path.read_text(encoding="utf-8"))
            if observed != payload:
                raise RuntimeError(f"unexpected existing claim: {path}")
            verified += 1
            continue
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        created += 1

    return {
        "total_samples": len(materialized),
        "active_samples": len(active),
        "deferred_samples": len(deferred),
        "active_by_dataset": dict(
            sorted(Counter(str(row["dataset"]) for row in active).items())
        ),
        "deferred_by_dataset": dict(
            sorted(Counter(str(row["dataset"]) for row in deferred).items())
        ),
        "claims_created": created,
        "claims_verified": verified,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Defer selected datasets in one sequential-conversion launch."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--defer", action="append", required=True)
    parser.add_argument("--launch-token", default=os.environ.get("SLURM_JOB_ID"))
    parser.add_argument(
        "--restart-count", default=os.environ.get("SLURM_RESTART_COUNT", "0")
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.launch_token:
        raise RuntimeError("--launch-token or SLURM_JOB_ID is required")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parents[1]
    source_manifest = (project_root / config["source_manifest"]).resolve()
    output_root = (project_root / config["output_root"]).resolve()
    analysis_root = (project_root / config["analysis_root"]).resolve()
    launch_name = f"{args.launch_token}_{args.restart_count}"
    claim_root = output_root / "full" / "claims" / launch_name

    summary = prepare_deferred_dataset_claims(
        _read_jsonl(source_manifest),
        claim_root=claim_root,
        deferred_datasets=set(args.defer),
        claimant="deferred-dataset-order",
        resume=args.resume,
    )
    summary.update(
        {
            "schema_version": "sequential_dataset_priority_claims_v1",
            "launch_token": str(args.launch_token),
            "restart_count": str(args.restart_count),
            "claim_root": str(claim_root),
            "source_manifest": str(source_manifest),
            "deferred_datasets": sorted(set(args.defer)),
        }
    )
    summary_path = analysis_root / f"dataset_priority_{launch_name}.json"
    if summary_path.exists():
        if not args.resume:
            raise FileExistsError(f"summary exists without --resume: {summary_path}")
        observed = json.loads(summary_path.read_text(encoding="utf-8"))
        stable_fields = {
            key: value
            for key, value in summary.items()
            if key not in {"claims_created", "claims_verified"}
        }
        observed_stable = {
            key: value
            for key, value in observed.items()
            if key not in {"claims_created", "claims_verified"}
        }
        if observed_stable != stable_fields:
            raise RuntimeError(f"existing priority summary differs: {summary_path}")
    else:
        _write_atomic_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
