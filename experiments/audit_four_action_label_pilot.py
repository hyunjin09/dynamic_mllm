#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binary_actions(mask) -> list[str]:
    return ["FULL" if int(value) else "IGNORE" for value in mask]


def _telemetry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"samples": 0}
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    utilizations = [float(row["utilization_gpu_percent"]) for row in rows]
    memories = [float(row["memory_used_mib"]) for row in rows]
    by_gpu = Counter(int(row["gpu_index"]) for row in rows)
    return {
        "samples": len(rows),
        "snapshots": len(rows) // 8,
        "gpu_indices": sorted(by_gpu),
        "mean_utilization_percent": statistics.mean(utilizations) if utilizations else None,
        "median_utilization_percent": statistics.median(utilizations) if utilizations else None,
        "p10_utilization_percent": (
            sorted(utilizations)[max(0, int(0.10 * len(utilizations)) - 1)]
            if utilizations
            else None
        ),
        "maximum_memory_used_mib": max(memories) if memories else None,
    }


def _slurm(job_id: str) -> dict[str, Any]:
    if not job_id:
        return {}
    output = subprocess.run(
        [
            "sacct",
            "-j",
            job_id,
            "-X",
            "--format=JobIDRaw,State,ElapsedRaw,AllocTRES",
            "-n",
            "-P",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    row = output.splitlines()[0].split("|") if output else []
    return (
        {
            "job_id": row[0],
            "state": row[1],
            "elapsed_seconds": int(row[2]),
            "allocated_tres": row[3],
        }
        if len(row) >= 4
        else {"raw": output}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the five-dataset conversion pilot.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/mcts_labels_4action/conversion_v1/pilot"),
    )
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=Path("analysis/4action_label_conversion/pilot_gpu_telemetry.csv"),
    )
    parser.add_argument("--job-id", default="")
    parser.add_argument("--job-ids", default="")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/4action_label_conversion/pilot_audit_v1.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    manifest = read_jsonl(args.root / "pilot_manifest_v1.jsonl")
    expected = {str(row["uid"]): row for row in manifest}
    result_paths = sorted((args.root / "records").glob("*.json"))
    results = [json.loads(path.read_text()) for path in result_paths]
    observed_ids = [str(row["uid"]) for row in results]
    failures = [
        row
        for path in sorted((args.root / "failures").glob("*.jsonl"))
        for row in read_jsonl(path)
    ]
    progress = [
        row
        for path in sorted((args.root / "progress").glob("*.jsonl"))
        for row in read_jsonl(path)
    ]

    checksum_errors = []
    for path in result_paths:
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not sidecar.exists() or sidecar.read_text().split()[0] != file_sha256(path):
            checksum_errors.append(str(path))
    raw = [row for result in results for row in result["raw_conversions"]]
    converted = [row for row in raw if row["status"] == "converted"]
    replay_failures = [row for row in raw if row["status"] == "source_route_replay_failure"]
    unique = [row for result in results for row in result["unique_valid_four_action_routes"]]
    w2c = [row for row in converted if row["label_semantics"] == "corrective_w2c"]
    c2c = [row for row in converted if row["label_semantics"] == "preserving_c2c"]
    binary_checks = [
        check for result in results for check in result["pilot_old_binary_semantic_checks"]
    ]
    worker_starts = [row for row in progress if row["event"] == "worker_start"]
    start_by_rank = {int(row["rank"]): row for row in worker_starts}
    worker_completes = [row for row in progress if row["event"] == "worker_complete"]
    all_off_w2c = [
        row for row in w2c if bool(row["all_off_seed"])
    ]
    action_counts = Counter(
        action for row in unique for action in row["route"]
    )

    checks = {
        "all_manifest_samples_completed": set(observed_ids) == set(expected),
        "no_duplicate_sample_results": len(observed_ids) == len(set(observed_ids)),
        "all_sample_records_passed": all(bool(row.get("passed")) for row in results),
        "no_worker_failure_artifacts": not failures,
        "all_result_checksums_valid": not checksum_errors,
        "source_route_accounting_exact": sum(
            int(row["source_positive_route_count"]) for row in results
        )
        == len(raw),
        "all_final_routes_jointly_correct": all(
            bool(row["evaluation"]["correct"]) for row in unique
        ),
        "w2c_purification_and_refinement_present": bool(w2c)
        and all(
            row["purification"] is not None
            and bool(row["purification"]["evaluation"]["correct"])
            and row["refinement"] is not None
            and bool(row["refinement"]["evaluation"]["correct"])
            for row in w2c
        ),
        "c2c_mechanical_and_unpurified": bool(c2c)
        and all(
            row["final_route"] == _binary_actions(row["source_binary_route"])
            and row["purification"] is None
            and row["refinement"] is None
            for row in c2c
        ),
        "old_binary_semantic_parity": bool(binary_checks)
        and all(
            row["generated_ids_match"]
            and row["generated_answer_match"]
            and row["correctness_match"]
            for row in binary_checks
        ),
        "all_sixteen_workers_started": set(start_by_rank) == set(range(16)),
        "two_replicas_on_all_eight_gpus": set(
            (int(row["gpu_index"]), int(row["replica_index"]))
            for row in start_by_rank.values()
        )
        == {(gpu, replica) for gpu in range(8) for replica in range(2)},
        "all_workers_completed": len({int(row["rank"]) for row in worker_completes}) == 16,
        "deduplication_observed": len(unique) < len(converted),
        "read_or_write_structure_observed": (
            action_counts["READ_ONLY"] + action_counts["WRITE_ONLY"]
        )
        > 0,
        "all_off_w2c_exercised_if_available": bool(all_off_w2c),
        "both_current_semantics_exercised": {
            row["label_semantics"] for row in results
        }
        == {"corrective_w2c", "preserving_c2c"},
    }
    telemetry = _telemetry(args.telemetry)
    selected_job_ids = [value for value in args.job_ids.split(",") if value]
    if args.job_id and not selected_job_ids:
        selected_job_ids = [args.job_id]
    slurm_jobs = [_slurm(job_id) for job_id in selected_job_ids]
    slurm = slurm_jobs[0] if slurm_jobs else {}
    checks["all_declared_slurm_jobs_completed"] = bool(slurm_jobs) and all(
        row.get("state", "").startswith("COMPLETED") for row in slurm_jobs
    )
    report = {
        "schema_version": "four_action_label_conversion_pilot_audit_v1",
        "job": slurm,
        "jobs": slurm_jobs,
        "expected_samples": len(expected),
        "completed_samples": len(results),
        "missing_uids": sorted(set(expected) - set(observed_ids)),
        "extra_uids": sorted(set(observed_ids) - set(expected)),
        "source_routes": len(raw),
        "replay_valid_routes": len(converted),
        "replay_failure_routes": len(replay_failures),
        "unique_valid_routes": len(unique),
        "current_semantics": dict(Counter(row["label_semantics"] for row in results)),
        "unique_action_counts": dict(action_counts),
        "all_off_w2c_converted_routes": len(all_off_w2c),
        "binary_semantic_checks": len(binary_checks),
        "checksum_errors": checksum_errors,
        "failure_rows": failures,
        "telemetry": telemetry,
        "checks": checks,
        "passed": all(checks.values()),
    }
    elapsed = sum(float(row.get("elapsed_seconds", 0)) for row in slurm_jobs)
    if elapsed:
        report["throughput"] = {
            "samples_per_hour": len(results) * 3600.0 / elapsed,
            "source_routes_per_second": len(raw) / elapsed,
            "unique_route_evaluations_per_second": sum(
                row["route_evaluation_cache"]["unique_complete_routes_evaluated"]
                for row in results
            )
            / elapsed,
            "allocated_gpu_hours": elapsed * 8 / 3600.0,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{file_sha256(args.output)}  {args.output.name}\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
