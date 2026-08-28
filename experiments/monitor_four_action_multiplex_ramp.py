#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from experiments.summarize_four_action_stage import (
    sample_gate_semantically_passes,
    worker_contract_passes,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def throughput_assessment(
    *,
    first_count: int,
    final_count: int,
    steady_seconds: float,
    baseline_samples_per_minute: float,
    minimum_speedup: float,
) -> dict[str, Any]:
    if steady_seconds <= 0.0 or final_count <= first_count:
        return {
            "samples_per_minute": None,
            "baseline_samples_per_minute": baseline_samples_per_minute,
            "speedup": None,
            "minimum_speedup": minimum_speedup,
            "passed": False,
        }
    rate = (final_count - first_count) * 60.0 / steady_seconds
    speedup = rate / baseline_samples_per_minute
    return {
        "samples_per_minute": rate,
        "baseline_samples_per_minute": baseline_samples_per_minute,
        "speedup": speedup,
        "minimum_speedup": minimum_speedup,
        "passed": speedup >= minimum_speedup,
    }


def slurm_state(job_id: str) -> str:
    live = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if live:
        return live.splitlines()[0]
    history = subprocess.run(
        ["sacct", "-j", job_id, "--format=State", "-n", "-X"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return history.splitlines()[0].strip().split()[0] if history else "UNKNOWN"


def gpu_snapshot() -> tuple[list[dict[str, float | int]], str | None]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            detail = f"nvidia-smi exit {exc.returncode}: {exc.stderr or exc.stdout or ''}".strip()
        else:
            detail = str(exc)
        return [], detail
    rows = []
    for line in output.splitlines():
        index, utilization, memory = (part.strip() for part in line.split(","))
        rows.append(
            {
                "gpu_index": int(index),
                "utilization_percent": float(utilization),
                "memory_used_mib": float(memory),
            }
        )
    return rows, None


def replica_paths(root: Path, stem: str, suffix: str) -> list[Path]:
    return sorted(root.glob(f"shard_*/{stem}_replica_*{suffix}"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate the first multi-replica four-action samples.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("analysis/4action_answer_alignment/primary__unified_v1"),
    )
    parser.add_argument("--minimum-new-samples", type=int, default=64)
    parser.add_argument("--expected-workers", type=int, default=16)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--baseline-samples-per-minute", type=float, default=18.0)
    parser.add_argument("--minimum-speedup", type=float, default=1.20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/4action_answer_alignment/multiplex2_ramp_report.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    started = time.monotonic()
    first_time = None
    first_count = 0
    snapshots: list[list[dict[str, float | int]]] = []
    telemetry_errors: list[str] = []
    terminal_problem = None
    while True:
        result_paths = replica_paths(args.root, "results", ".jsonl")
        runtime_paths = replica_paths(args.root, "runtime", ".json")
        failure_paths = replica_paths(args.root, "failures", ".jsonl")
        count = sum(sum(1 for line in path.open() if line.strip()) for path in result_paths)
        failures = [row for path in failure_paths for row in read_jsonl(path)]
        state = slurm_state(args.job_id)
        if runtime_paths:
            snapshot, telemetry_error = gpu_snapshot()
            if snapshot:
                snapshots.append(snapshot)
            if telemetry_error and telemetry_error not in telemetry_errors:
                telemetry_errors.append(telemetry_error)
        if failures:
            terminal_problem = "replica failure artifact appeared"
            break
        if count > 0 and first_time is None:
            first_time = time.monotonic()
            first_count = count
        if count >= args.minimum_new_samples:
            break
        if state not in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING"}:
            terminal_problem = f"job entered {state} before the ramp completed"
            break
        time.sleep(args.poll_seconds)

    finished = time.monotonic()
    result_paths = replica_paths(args.root, "results", ".jsonl")
    runtime_paths = replica_paths(args.root, "runtime", ".json")
    failure_paths = replica_paths(args.root, "failures", ".jsonl")
    rows = [row for path in result_paths for row in read_jsonl(path)]
    runtimes = [json.loads(path.read_text(encoding="utf-8")) for path in runtime_paths]
    failures = [row for path in failure_paths for row in read_jsonl(path)]
    throughput = throughput_assessment(
        first_count=first_count,
        final_count=len(rows),
        steady_seconds=0.0 if first_time is None else finished - first_time,
        baseline_samples_per_minute=args.baseline_samples_per_minute,
        minimum_speedup=args.minimum_speedup,
    )
    flat_snapshots = [row for snapshot in snapshots for row in snapshot]
    gpu_summary = {
        "snapshot_count": len(snapshots),
        "mean_utilization_percent": (
            statistics.mean(row["utilization_percent"] for row in flat_snapshots)
            if flat_snapshots
            else None
        ),
        "median_utilization_percent": (
            statistics.median(row["utilization_percent"] for row in flat_snapshots)
            if flat_snapshots
            else None
        ),
        "maximum_memory_used_mib": (
            max(row["memory_used_mib"] for row in flat_snapshots)
            if flat_snapshots
            else None
        ),
        "collection_errors": telemetry_errors,
        "availability_is_diagnostic_only": True,
    }
    ids = [row["uid"] for row in rows]
    checks = {
        "minimum_new_samples": len(rows) >= args.minimum_new_samples,
        "expected_worker_metadata": len(runtimes) == args.expected_workers,
        "all_eight_gpu_worker_contract": worker_contract_passes(runtimes),
        "unique_new_samples": len(ids) == len(set(ids)),
        "semantic_gates_pass": all(sample_gate_semantically_passes(row) for row in rows),
        "no_failure_artifacts": not failures,
        "material_throughput_improvement": throughput["passed"],
        "job_still_healthy": slurm_state(args.job_id) in {"RUNNING", "COMPLETING", "COMPLETED"},
    }
    report = {
        "schema_version": "four_action_multiplex_ramp_report_v1",
        "job_id": args.job_id,
        "new_sample_count": len(rows),
        "monitor_elapsed_seconds": finished - started,
        "terminal_problem": terminal_problem,
        "throughput": throughput,
        "gpu": gpu_summary,
        "checks": checks,
        "passed": all(checks.values()) and terminal_problem is None,
        "failure_rows": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
