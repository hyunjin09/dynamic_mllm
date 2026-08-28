#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

from tools.research_analysis.four_action.route_conditioned import (
    choose_pilot_configuration,
    summarize_gpu_metrics,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(text, encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare route-conditioned pilot concurrency.")
    parser.add_argument("--config", type=Path, default=Path("configs/four_action_route_conditioned.yaml"))
    parser.add_argument("--tags", nargs="+", default=["one_replica", "two_replicas"])
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pilot_root = Path(config["pilot_root"])
    configurations = []
    for tag in args.tags:
        directory = pilot_root / f"pilot__{tag}"
        summary = json.loads((directory / "stage_summary.json").read_text(encoding="utf-8"))
        with (directory / "gpu_metrics.csv").open(newline="", encoding="utf-8") as handle:
            gpu = summarize_gpu_metrics(csv.DictReader(handle))
        configurations.append({"name": tag, **summary, "gpu_metrics": gpu})
    if len({row["manifest_sha256"] for row in configurations}) != 1:
        raise RuntimeError("pilot configurations did not use the identical pilot manifest")
    if len({row["sample_count"] for row in configurations}) != 1:
        raise RuntimeError("pilot configurations have different sample coverage")
    selected = choose_pilot_configuration(configurations)
    baseline = min(configurations, key=lambda row: row["replicas_per_gpu"])
    for row in configurations:
        row["throughput_ratio_to_one_replica"] = (
            row["useful_new_cells_per_second"] / baseline["useful_new_cells_per_second"]
        )
    benchmark = {
        "schema_version": "route_conditioned_pilot_benchmark_v1",
        "passed": all(row["passed"] for row in configurations),
        "semantic_and_numerical_gates_pass": all(
            row["all_sample_gates_pass"] and row["all_pilot_m00_reproductions_pass"]
            for row in configurations
        ),
        "configurations": configurations,
        "selected_configuration": selected["name"],
        "selected_replicas_per_gpu": selected["replicas_per_gpu"],
        "selection_objective": "maximum passing useful new intervention cells per worker wall second",
    }
    if not benchmark["passed"] or not benchmark["semantic_and_numerical_gates_pass"]:
        raise RuntimeError("pilot did not pass every semantic/numerical stage gate")
    output_root = Path(config["output_root"])
    write_once(
        output_root / "pilot_benchmark_summary.json",
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n",
    )
    table = []
    for row in configurations:
        gpu = row["gpu_metrics"]
        table.append(
            "| {name} | {replicas} | {cells:.4f} | {samples:.4f} | {ratio:.3f}x | "
            "{memory:.0f} MiB | {util:.1f}% | 0 |".format(
                name=row["name"],
                replicas=row["replicas_per_gpu"],
                cells=row["useful_new_cells_per_second"],
                samples=row["useful_samples_per_second"],
                ratio=row["throughput_ratio_to_one_replica"],
                memory=gpu["peak_memory_used_mib"],
                util=gpu["mean_gpu_utilization_percent"],
            )
        )
    report = f"""# Route-Conditioned Four-Action Pilot Report

## Scope

The frozen pilot contains {selected['sample_count']} samples selected across
GQA/TextVQA and small, medium, and large validated anchor-OFF-count strata.
Every configuration used the identical manifest and all eight H100s.

## Semantic and numerical gates

- Anchor BOTH_OFF generation, evaluator correctness, and fixed-target scores
  reproduce the current validated anchor within the frozen tolerance.
- Every four-action branch starts from the same anchor pre-layer state.
- Every non-target layer retains its exact anchor FULL/IGNORE action.
- READ_ONLY, WRITE_ONLY, FULL, and explicit pilot IGNORE satisfy their READ,
  WRITE, visual-row, two-call target, and heterogeneous-cache contracts.
- Correct and original-FULL-wrong answer targets remain fixed across states.
- Resume/shard coverage is unique and complete; no disqualifying failure or OOM
  remains.

All semantic and numerical gates passed in both concurrency configurations.

## Throughput benchmark

Useful intervention cells per wall second is the primary optimization target;
GPU utilization is diagnostic only.

| Configuration | Replicas/GPU | New cells/s | Samples/s | Ratio | Peak VRAM | Mean GPU util | Failures/OOM |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Selected configuration: **{selected['name']}**
({selected['replicas_per_gpu']} replica(s) per GPU), because it has the highest
passing useful intervention throughput.

## Full-launch gate

**PASS.** The full route-conditioned A+ sweep may launch automatically with
the selected concurrency. Completed pilot artifacts remain separate and are
not reused as production cells.
"""
    write_once(output_root / "pilot_report.md", report)
    print(json.dumps(benchmark, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
