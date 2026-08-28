#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from tools.research_analysis.four_action.three_action_jobs import file_sha256


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_full_audit(
    records: list[dict[str, Any]],
    *,
    manifest: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    checksum_errors: list[str],
    epsilon_sha256: str,
    slurm_jobs: list[dict[str, Any]],
    progress_rows: list[dict[str, Any]],
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    expected = {str(row["uid"]): row for row in manifest}
    observed = [str(row.get("uid")) for row in records]
    unresolved_failure_rows = [
        row for row in failure_rows if str(row.get("uid")) not in set(observed)
    ]
    raw = [conversion for row in records for conversion in row.get("raw_conversions", [])]
    converted = [row for row in raw if row.get("status") == "converted"]
    positives = [route for row in converted for route in row.get("positive_routes", [])]
    unique = [route for row in records for route in row.get("unique_valid_three_action_routes", [])]
    contracts = {row.get("execution_contract", {}).get("contract_sha256") for row in records}
    dynamic_starts = [
        row for row in progress_rows
        if row.get("event") == "worker_start"
        and row.get("work_assignment") == "atomic_dynamic"
    ]
    source_ids_exact = True
    for record in records:
        uid = str(record.get("uid"))
        if uid not in expected:
            source_ids_exact = False
            continue
        expected_ids = {
            str(route["source_binary_route_id"])
            for route in expected[uid]["source_positive_routes"]
        }
        observed_ids = {
            str(route.get("source_binary_route_id")) for route in record.get("raw_conversions", [])
        }
        if expected_ids != observed_ids or len(observed_ids) != len(record.get("raw_conversions", [])):
            source_ids_exact = False
    c2c_gain = True
    for record in records:
        if record.get("route_type") != "C2C":
            continue
        baseline = float(record["current_unified_full"]["S_correct"])
        epsilon = float(record["epsilon"])
        for conversion in record.get("raw_conversions", []):
            for route in conversion.get("positive_routes", []):
                if not (
                    bool(route["evaluation"].get("correct"))
                    and float(route["evaluation"]["S_correct"]) > baseline + epsilon
                ):
                    c2c_gain = False
    w2c_coverage = all(
        record.get("route_type") != "W2C"
        or int(record.get("source_route_replay_valid_count", 0)) == 0
        or bool(record.get("unique_valid_three_action_routes"))
        for record in records
    )
    actions_valid = all(
        set(route.get("route", [])) <= {"FULL", "READ_OFF", "WRITE_OFF", "BOTH_OFF"}
        for route in unique
    )
    total_job_seconds = sum(
        float(row.get("elapsed_seconds", 0)) for row in slurm_jobs
    )
    checks = {
        "all_manifest_samples_completed": set(observed) == set(expected),
        "no_duplicate_sample_records": len(observed) == len(set(observed)),
        "all_sample_records_passed": bool(records) and all(bool(row.get("passed")) for row in records),
        "no_unresolved_worker_failures": not unresolved_failure_rows,
        "all_record_checksums_valid": not checksum_errors,
        "source_route_accounting_exact": len(raw) == sum(
            int(row["source_positive_route_count"]) for row in manifest
        ),
        "source_route_ids_reconcile_exactly": source_ids_exact,
        "replay_status_accounting_exact": all(
            int(row.get("source_route_replay_valid_count", 0))
            + int(row.get("source_route_replay_failure_count", 0))
            == int(row.get("source_positive_route_count", 0))
            for row in records
        ),
        "all_positive_routes_jointly_correct": all(
            bool(row["evaluation"].get("correct")) for row in positives
        ),
        "all_unique_routes_jointly_correct": all(
            bool(row["evaluation"].get("correct")) for row in unique
        ),
        "c2c_positive_routes_have_support_gain": c2c_gain,
        "w2c_replay_valid_samples_have_positive_route": w2c_coverage,
        "only_declared_three_action_values": actions_valid,
        "epsilon_artifact_bound_to_every_record": bool(records) and all(
            row.get("execution_contract", {}).get("epsilon_sha256") == epsilon_sha256
            for row in records
        ),
        "single_execution_contract": len(contracts) == 1 and all(contracts),
        "slurm_job_provenance_present": bool(slurm_jobs),
        "final_declared_slurm_job_completed": bool(slurm_jobs)
        and str(slurm_jobs[-1].get("state", "")).startswith("COMPLETED")
        and str(slurm_jobs[-1].get("exit_code", "0:0")).startswith("0:"),
        "all_sixteen_dynamic_workers_observed": {
            int(row["rank"]) for row in dynamic_starts
        } == set(range(16)),
        "two_replicas_on_all_eight_gpus_observed": {
            (int(row["gpu_index"]), int(row["replica_index"]))
            for row in dynamic_starts
        } >= {(gpu, replica) for gpu in range(8) for replica in range(2)},
        "telemetry_covers_all_eight_gpus": int(telemetry.get("samples", 0)) > 0
        and telemetry.get("gpu_indices") == list(range(8)),
    }
    return {
        "schema_version": "three_action_answer_aligned_full_integrity_audit_v1",
        "passed": all(checks.values()),
        "expected_samples": len(expected),
        "completed_samples": len(records),
        "missing_uids": sorted(set(expected) - set(observed)),
        "extra_uids": sorted(set(observed) - set(expected)),
        "source_routes": len(raw),
        "replay_valid_routes": len(converted),
        "replay_failure_routes": len(raw) - len(converted),
        "positive_route_occurrences": len(positives),
        "unique_positive_routes": len(unique),
        "samples_without_positive_three_action_route": sum(
            not bool(row.get("unique_valid_three_action_routes")) for row in records
        ),
        "execution_contract_sha256": sorted(
            "<missing>" if value is None else str(value) for value in contracts
        ),
        "epsilon_sha256": epsilon_sha256,
        "checks": checks,
        "worker_failure_rows": failure_rows,
        "unresolved_worker_failure_rows": unresolved_failure_rows,
        "checksum_errors": checksum_errors,
        "jobs": slurm_jobs,
        "telemetry": telemetry,
        "throughput": {
            "wall_hours_across_all_launches": total_job_seconds / 3600.0,
            "allocated_gpu_hours_across_all_launches": total_job_seconds * 8 / 3600.0,
            "samples_per_wall_hour": (
                len(records) * 3600.0 / total_job_seconds if total_job_seconds else None
            ),
            "source_routes_per_wall_second": (
                len(raw) / total_job_seconds if total_job_seconds else None
            ),
        },
    }


def _slurm(job_id: str) -> dict[str, Any]:
    output = subprocess.run(
        ["sacct", "-j", job_id, "-X", "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES", "-n", "-P"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    row = output.splitlines()[0].split("|") if output else []
    return (
        {"job_id": row[0], "state": row[1], "exit_code": row[2], "elapsed_seconds": int(row[3]), "allocated_tres": row[4]}
        if len(row) >= 5 else {"job_id": job_id, "raw": output}
    )


def _telemetry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"samples": 0, "gpu_indices": []}
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    utilization = [float(row["utilization_gpu_percent"]) for row in rows]
    memory = [float(row["memory_used_mib"]) for row in rows]
    return {
        "samples": len(rows),
        "gpu_indices": sorted({int(row["gpu_index"]) for row in rows}),
        "mean_utilization_percent": statistics.mean(utilization) if utilization else None,
        "median_utilization_percent": statistics.median(utilization) if utilization else None,
        "maximum_memory_used_mib": max(memory) if memory else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the complete three-action conversion.")
    parser.add_argument("--config", type=Path, default=Path("configs/three_action_label_conversion.yaml"))
    parser.add_argument("--job-ids", required=True)
    parser.add_argument("--telemetry", type=Path, default=Path("analysis/three_action_answer_aligned_label_conversion/full_gpu_telemetry.csv"))
    parser.add_argument("--output", type=Path, default=Path("analysis/three_action_answer_aligned_label_conversion/full_integrity_audit_v1.json"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = read_jsonl(Path(config["source_manifest"]))
    root = Path(config["output_root"]) / "full"
    paths = sorted((root / "records").glob("*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    failures = [row for path in sorted((root / "failures").glob("*.jsonl")) for row in read_jsonl(path)]
    progress = [row for path in sorted((root / "progress").glob("*.jsonl")) for row in read_jsonl(path)]
    checksum_errors = []
    for path in paths:
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not sidecar.exists() or sidecar.read_text(encoding="utf-8").split()[0] != file_sha256(path):
            checksum_errors.append(str(path))
    epsilon_path = Path(config["noise_calibration"]["artifact_path"])
    jobs = [_slurm(value) for value in args.job_ids.split(",") if value]
    report = build_full_audit(
        records,
        manifest=manifest,
        failure_rows=failures,
        checksum_errors=checksum_errors,
        epsilon_sha256=file_sha256(epsilon_path),
        slurm_jobs=jobs,
        progress_rows=progress,
        telemetry=_telemetry(args.telemetry),
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{file_sha256(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
