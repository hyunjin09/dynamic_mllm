#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_full_estimate(
    anchor_summary: Mapping[str, Any],
    pilot_benchmark: Mapping[str, Any],
    *,
    gpu_count: int = 8,
) -> dict[str, Any]:
    selected_name = str(pilot_benchmark["selected_configuration"])
    matches = [
        row for row in pilot_benchmark["configurations"] if str(row["name"]) == selected_name
    ]
    if len(matches) != 1:
        raise ValueError("pilot benchmark does not contain exactly one selected configuration")
    selected = matches[0]
    throughput = float(selected["useful_new_cells_per_second"])
    if throughput <= 0 or not math.isfinite(throughput):
        raise ValueError("selected pilot throughput must be finite and positive")
    cells = int(anchor_summary["expected_new_cells_3k"])
    wall_seconds = cells / throughput
    return {
        "schema_version": "route_conditioned_compute_estimate_v1",
        "validated_anchor_sample_count": int(anchor_summary["validated_anchor_count"]),
        "expected_new_intervention_cells": cells,
        "production_evaluations_per_off_position": 3,
        "selected_configuration": selected_name,
        "selected_replicas_per_gpu": int(pilot_benchmark["selected_replicas_per_gpu"]),
        "pilot_useful_new_cells_per_second": throughput,
        "gpu_count": int(gpu_count),
        "expected_wall_seconds": wall_seconds,
        "expected_wall_hours": wall_seconds / 3600.0,
        "expected_gpu_hours": wall_seconds * gpu_count / 3600.0,
        "planning_wall_hours_with_20_percent_contingency": wall_seconds * 1.2 / 3600.0,
        "estimation_basis": (
            "identical-manifest pilot valid new cells divided by maximum accumulated "
            "worker time; GPU utilization is diagnostic only"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Record route-conditioned full-sweep compute estimate.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/four_action_route_conditioned.yaml"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(config["output_root"])
    anchor_path = root / "anchor_route_summary.json"
    pilot_path = root / "pilot_benchmark_summary.json"
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if not anchor.get("passed") or not pilot.get("passed"):
        raise RuntimeError("compute estimate requires passing anchor and pilot summaries")
    estimate = compute_full_estimate(anchor, pilot)
    estimate["input_hashes"] = {
        "anchor_route_summary": _sha256_file(anchor_path),
        "pilot_benchmark_summary": _sha256_file(pilot_path),
    }
    output = root / "compute_estimate.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(estimate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_name(output.name + ".sha256").write_text(
        f"{_sha256_file(output)}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps(estimate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
