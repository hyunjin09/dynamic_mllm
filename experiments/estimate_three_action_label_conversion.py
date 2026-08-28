#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import floor
from pathlib import Path
import statistics
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from tools.research_analysis.four_action.label_jobs import safe_filename
from tools.research_analysis.four_action.three_action_jobs import file_sha256


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_compute_estimate(
    pilot_records: list[dict[str, Any]],
    *,
    pilot_manifest: list[dict[str, Any]],
    full_manifest: list[dict[str, Any]],
    workers: int,
    gpus: int,
) -> dict[str, Any]:
    pilot_source = {str(row["uid"]): row for row in pilot_manifest}
    rates: dict[str, list[float]] = defaultdict(list)
    actual_runtime = 0.0
    actual_route_evaluations = 0
    for record in pilot_records:
        uid = str(record["uid"])
        source = pilot_source[uid]
        static_cost = max(1, int(source["estimated_conversion_cost"]))
        elapsed = float(record["runtime"]["elapsed_seconds"])
        status = str(source["source_current_all_on_status"])
        rates[status].append(elapsed / static_cost)
        actual_runtime += elapsed
        actual_route_evaluations += int(
            record.get("route_evaluation_cache", {}).get("cache_misses", 0)
        )
    if set(pilot_source) != {str(row["uid"]) for row in pilot_records}:
        raise ValueError("pilot records do not exactly match the frozen pilot manifest")

    full_cost: dict[str, int] = defaultdict(int)
    full_samples: dict[str, int] = defaultdict(int)
    for row in full_manifest:
        status = str(row["source_current_all_on_status"])
        full_cost[status] += int(row["estimated_conversion_cost"])
        full_samples[status] += 1
    missing_strata = sorted(set(full_cost) - set(rates))
    if missing_strata:
        raise ValueError(f"pilot lacks full-population cost strata: {missing_strata}")

    rate_summary = {
        status: {
            "count": len(values),
            "p25": _quantile(values, 0.25),
            "median": statistics.median(values),
            "p75": _quantile(values, 0.75),
        }
        for status, values in rates.items()
    }
    estimates = {}
    for label, percentile in (("low", "p25"), ("central", "median"), ("high", "p75")):
        worker_seconds = sum(
            full_cost[status] * rate_summary[status][percentile]
            for status in full_cost
        )
        ideal_hours = worker_seconds / workers / 3600.0
        conservative_hours = ideal_hours / 0.80
        estimates[label] = {
            "worker_hours": worker_seconds / 3600.0,
            "ideal_wall_hours": ideal_hours,
            "conservative_wall_hours_at_80pct_scheduling_efficiency": conservative_hours,
            "ideal_allocated_gpu_hours": ideal_hours * gpus,
            "conservative_allocated_gpu_hours": conservative_hours * gpus,
        }
    return {
        "schema_version": "three_action_answer_aligned_full_compute_estimate_v1",
        "method": (
            "pilot per-sample seconds per frozen route/OFF static cost, stratified by "
            "historical FULL status; p25/median/p75 projected to the complete source inventory"
        ),
        "pilot_samples": len(pilot_records),
        "full_samples": len(full_manifest),
        "workers": workers,
        "gpus": gpus,
        "pilot_seconds_per_static_cost_by_source_status": rate_summary,
        "pilot_observed_unique_route_evaluations": actual_route_evaluations,
        "pilot_observed_worker_seconds": actual_runtime,
        "pilot_route_evaluations_per_worker_second": (
            actual_route_evaluations / actual_runtime if actual_runtime else None
        ),
        "projected_route_evaluations_per_wall_second_at_configured_workers": (
            actual_route_evaluations * workers / actual_runtime if actual_runtime else None
        ),
        "full_samples_by_source_status": dict(full_samples),
        "full_static_cost_by_source_status": dict(full_cost),
        "estimates": estimates,
        "limitations": [
            "The pilot is deliberately stress-stratified rather than a random sample.",
            "Historical FULL status can shift under the current unified runtime.",
            "The static source cost is a projection proxy; exact cache overlap is represented only through observed pilot elapsed time.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate full three-action conversion compute from the passing pilot.")
    parser.add_argument("--config", type=Path, default=Path("configs/three_action_label_conversion.yaml"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/three_action_answer_aligned_label_conversion/full_compute_estimate_v1.json"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pilot_root = Path(config["output_root"]) / "pilot"
    pilot_manifest = read_jsonl(pilot_root / "pilot_manifest_v1.jsonl")
    records = []
    for row in pilot_manifest:
        path = pilot_root / "records" / safe_filename(str(row["uid"]))
        if not path.is_file():
            raise RuntimeError(f"pilot is incomplete: {path}")
        records.append(json.loads(path.read_text(encoding="utf-8")))
    report = build_compute_estimate(
        records,
        pilot_manifest=pilot_manifest,
        full_manifest=read_jsonl(Path(config["source_manifest"])),
        workers=int(config["worker_count"]),
        gpus=int(config["gpu_count"]),
    )
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {args.output}")
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
        checksum_valid = (
            sidecar.is_file()
            and bool(sidecar.read_text(encoding="utf-8").split())
            and sidecar.read_text(encoding="utf-8").split()[0] == file_sha256(args.output)
        )
        if existing != report or not checksum_valid:
            raise RuntimeError("existing compute estimate differs from the passing pilot")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{file_sha256(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
