#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from tools.research_analysis.four_action.sequential_label_jobs import file_sha256


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate exact sequential conversion compute.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sequential_four_action_label_conversion.yaml"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/4action_sequential_label_conversion/full_compute_estimate_v1.json"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    smoke_root = Path(config["output_root"]) / "smoke" / "records"
    smoke = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(smoke_root.glob("*.json"))]
    if len(smoke) != 8:
        raise RuntimeError(f"compute estimate requires 8 smoke records, found {len(smoke)}")
    source = read_jsonl(Path(config["source_manifest"]))
    seconds_per_declared_cost: dict[str, list[float]] = {"W2C": [], "C2C": []}
    smoke_manifest = {
        row["uid"]: row
        for row in read_jsonl(Path(config["output_root"]) / "smoke" / "smoke_manifest_v1.jsonl")
    }
    for record in smoke:
        cost = max(1, int(smoke_manifest[record["uid"]]["estimated_conversion_cost"]))
        seconds_per_declared_cost[record["route_type"]].append(
            float(record["runtime"]["elapsed_seconds"]) / cost
        )
    medians = {
        route_type: statistics.median(values)
        for route_type, values in seconds_per_declared_cost.items()
    }
    predicted_worker_seconds = 0.0
    type_counts = {"W2C": 0, "C2C": 0}
    for row in source:
        route_type = "W2C" if row["source_current_all_on_status"] == "wrong" else "C2C"
        type_counts[route_type] += 1
        predicted_worker_seconds += medians[route_type] * max(
            1, int(row["estimated_conversion_cost"])
        )
    scheduling_efficiency = 0.80
    wall_seconds = predicted_worker_seconds / (16 * scheduling_efficiency)
    report = {
        "schema_version": "exact_sequential_four_action_compute_estimate_v1",
        "method": "smoke median seconds per source estimated-cost unit, by current-runtime route type",
        "warning": (
            "Exact branching is data-dependent; this estimate is provisional and branch explosion "
            "can make the actual run materially slower."
        ),
        "smoke_samples": len(smoke),
        "full_samples": len(source),
        "full_type_counts": type_counts,
        "median_seconds_per_declared_cost": medians,
        "predicted_total_worker_hours": predicted_worker_seconds / 3600.0,
        "assumed_scheduling_efficiency": scheduling_efficiency,
        "predicted_wall_hours_16_workers": wall_seconds / 3600.0,
        "predicted_allocated_gpu_hours": wall_seconds * 8 / 3600.0,
        "source_manifest_sha256": file_sha256(Path(config["source_manifest"])),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {args.output}")
        if args.output.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("existing compute estimate differs from recomputed evidence")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
