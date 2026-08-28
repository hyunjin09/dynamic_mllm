#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from tools.research_analysis.four_action.route_conditioned import (
    balance_work_units,
    finalize_anchor_rows,
    select_stratified_pilot,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_sidecar(path: Path) -> None:
    path.with_name(path.name + ".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
    )


def write_jsonl_once(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    add_sidecar(path)


def write_json_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    add_sidecar(path)


def write_parquet_once(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    add_sidecar(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze current-valid route-conditioned anchors.")
    parser.add_argument("--config", type=Path, default=Path("configs/four_action_route_conditioned.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(config["anchor_validation_root"])
    candidate_rows = read_jsonl(Path(config["candidate_manifest"]))
    result_paths = sorted(root.glob("shard_*/results.jsonl"))
    runtime_paths = sorted(root.glob("shard_*/runtime.json"))
    failure_paths = sorted(root.glob("shard_*/failures.jsonl"))
    results = [row for path in result_paths for row in read_jsonl(path)]
    runtimes = [json.loads(path.read_text(encoding="utf-8")) for path in runtime_paths]
    failures = [row for path in failure_paths for row in read_jsonl(path)]
    passed_uids = {row["uid"] for row in results if row.get("passed")}
    disqualifying_failures = [row for row in failures if row.get("uid") not in passed_uids]
    worker_gate = len(runtimes) == 8 and {row["gpu_index"] for row in runtimes} == set(range(8))
    if not worker_gate:
        raise RuntimeError("anchor validation is missing the exact eight-GPU runtime contract")
    if disqualifying_failures:
        raise RuntimeError(f"anchor validation has {len(disqualifying_failures)} unrecovered failures")
    anchors, exclusions = finalize_anchor_rows(candidate_rows, results)
    units = balance_work_units(anchors, work_unit_count=int(config["work_unit_count"]))
    assigned_anchors = sorted(
        (row for unit in units for row in unit["samples"]), key=lambda row: row["uid"]
    )
    pilot = select_stratified_pilot(assigned_anchors, total=int(config["pilot_count"]))
    pilot_units = balance_work_units(pilot, work_unit_count=8)
    pilot_assigned = []
    for index, unit in enumerate(pilot_units):
        for source in unit["samples"]:
            row = dict(source)
            row["pilot_worker_index"] = index
            pilot_assigned.append(row)
    pilot_assigned.sort(key=lambda row: (row["pilot_worker_index"], row["uid"]))
    output_root = Path(config["output_root"])
    anchor_path = Path(config["anchor_manifest"])
    write_jsonl_once(anchor_path, assigned_anchors)
    write_parquet_once(anchor_path.with_suffix(".parquet"), assigned_anchors)
    write_jsonl_once(output_root / "anchor_exclusions.jsonl", exclusions)
    write_jsonl_once(output_root / "pilot_manifest.jsonl", pilot_assigned)
    unit_rows = [
        {
            "schema_version": "route_conditioned_work_unit_v1",
            "work_unit_id": unit["work_unit_id"],
            "expected_new_cells": unit["expected_new_cells"],
            "sample_count": len(unit["samples"]),
            "uids": [row["uid"] for row in unit["samples"]],
        }
        for unit in units
    ]
    write_jsonl_once(output_root / "work_unit_manifest.jsonl", unit_rows)
    off_counts = [row["anchor_off_count"] for row in assigned_anchors]
    summary = {
        "schema_version": "route_conditioned_anchor_summary_v1",
        "passed": True,
        "frozen_a_plus_count": len(candidate_rows),
        "validated_anchor_count": len(assigned_anchors),
        "excluded_no_current_correct_anchor_count": len(exclusions),
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in assigned_anchors).items())),
        "fallback_sample_count": sum(row["anchor_fallback_count"] > 0 for row in assigned_anchors),
        "fallback_evaluations": sum(row["anchor_fallback_count"] for row in assigned_anchors),
        "anchor_off_count_distribution": dict(sorted(Counter(off_counts).items())),
        "expected_new_cells_3k": 3 * sum(off_counts),
        "pilot_count": len(pilot_assigned),
        "pilot_dataset_counts": dict(sorted(Counter(row["dataset"] for row in pilot_assigned).items())),
        "work_unit_count": len(unit_rows),
        "work_unit_expected_cell_range": [
            min(row["expected_new_cells"] for row in unit_rows),
            max(row["expected_new_cells"] for row in unit_rows),
        ],
        "all_eight_shards_present": worker_gate,
        "observed_validation_rows": len(results),
        "failure_artifact_count": len(failures),
        "disqualifying_failure_count": len(disqualifying_failures),
        "anchor_manifest_sha256": sha256_file(anchor_path),
        "anchor_manifest_parquet_sha256": sha256_file(anchor_path.with_suffix(".parquet")),
    }
    write_json_once(output_root / "anchor_route_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
