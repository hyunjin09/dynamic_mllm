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


def safe_filename(uid: str) -> str:
    readable = uid.replace(":", "__").replace("/", "_")
    return f"{readable}_{hashlib.sha256(uid.encode()).hexdigest()[:10]}.json"


def slurm_jobs(job_ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for job_id in job_ids:
        output = subprocess.run(
            [
                "sacct",
                "-j",
                job_id,
                "-X",
                "--format=JobIDRaw,State,ElapsedRaw,ExitCode,AllocTRES",
                "-n",
                "-P",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        fields = output.splitlines()[0].split("|") if output else []
        if len(fields) >= 5:
            rows.append(
                {
                    "job_id": fields[0],
                    "state": fields[1],
                    "elapsed_seconds": int(fields[2]),
                    "exit_code": fields[3],
                    "allocated_tres": fields[4],
                }
            )
    return rows


def telemetry_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"samples": 0}
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    utilization = [float(row["utilization_gpu_percent"]) for row in rows]
    memory = [float(row["memory_used_mib"]) for row in rows]
    return {
        "samples": len(rows),
        "snapshots": len(rows) // 8,
        "gpu_indices": sorted({int(row["gpu_index"]) for row in rows}),
        "mean_utilization_percent": statistics.mean(utilization) if utilization else None,
        "median_utilization_percent": statistics.median(utilization) if utilization else None,
        "maximum_memory_used_mib": max(memory) if memory else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the full four-action label conversion.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(
            "datasets/mcts_labels_4action/source_inventory_v1/source_manifest_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/mcts_labels_4action/conversion_v1/full"),
    )
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=Path("analysis/4action_label_conversion/full_gpu_telemetry.csv"),
    )
    parser.add_argument("--job-ids", default="")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/4action_label_conversion/full_integrity_audit_v1.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    sources = read_jsonl(args.source_manifest)
    expected = {str(row["uid"]): row for row in sources}
    observed = {}
    checksum_errors = []
    invalid_records = []
    source_accounting_errors = []
    semantic_errors = []
    provenance_errors = []
    contract_ids = set()
    counters = Counter()
    for uid, source in expected.items():
        path = args.root / "records" / safe_filename(uid)
        if not path.is_file():
            continue
        sidecar = Path(str(path) + ".sha256")
        if not sidecar.is_file() or sidecar.read_text().split()[0] != file_sha256(path):
            checksum_errors.append(str(path))
            continue
        result = json.loads(path.read_text())
        if result.get("uid") != uid or not result.get("passed"):
            invalid_records.append(str(path))
            continue
        observed[uid] = result
        contract_ids.add(result["execution_contract"]["contract_sha256"])
        if result["label_semantics"] == "corrective_w2c" and bool(
            result["current_unified_full"]["correct"]
        ):
            semantic_errors.append(f"{uid}:w2c_full_correct")
        if result["label_semantics"] == "preserving_c2c" and not bool(
            result["current_unified_full"]["correct"]
        ):
            semantic_errors.append(f"{uid}:c2c_full_wrong")
        raw = result["raw_conversions"]
        expected_ids = {
            str(row["source_binary_route_id"]) for row in source["source_positive_routes"]
        }
        raw_ids = [str(row["source_binary_route_id"]) for row in raw]
        if len(raw) != int(source["source_positive_route_count"]) or set(raw_ids) != expected_ids:
            source_accounting_errors.append(uid)
        converted_ids = set()
        for row in raw:
            counters[f"raw_{row['status']}"] += 1
            if row["status"] == "source_route_replay_failure":
                if bool(row["source_route_evaluation"]["correct"]):
                    semantic_errors.append(f"{uid}:{row['source_binary_route_id']}:failure_correct")
                continue
            if row["status"] != "converted":
                semantic_errors.append(f"{uid}:{row['source_binary_route_id']}:unknown_status")
                continue
            converted_ids.add(str(row["source_binary_route_id"]))
            if not bool(row["source_route_evaluation"]["correct"]):
                semantic_errors.append(f"{uid}:{row['source_binary_route_id']}:source_wrong")
            if not bool(row["final_evaluation"]["correct"]):
                semantic_errors.append(f"{uid}:{row['source_binary_route_id']}:final_wrong")
            if row["label_semantics"] == "corrective_w2c":
                if row["purification"] is None or row["refinement"] is None:
                    semantic_errors.append(f"{uid}:{row['source_binary_route_id']}:missing_w2c")
            elif row["label_semantics"] == "preserving_c2c":
                mechanical = ["FULL" if int(value) else "IGNORE" for value in row["source_binary_route"]]
                if row["final_route"] != mechanical or row["purification"] is not None or row["refinement"] is not None:
                    semantic_errors.append(f"{uid}:{row['source_binary_route_id']}:c2c_changed")
            else:
                semantic_errors.append(f"{uid}:{row['source_binary_route_id']}:bad_semantics")

        provenance_ids = []
        unique_keys = set()
        for row in result["unique_valid_four_action_routes"]:
            if not bool(row["evaluation"]["correct"]):
                semantic_errors.append(f"{uid}:{row['route_key']}:unique_wrong")
            if row["route_key"] in unique_keys:
                provenance_errors.append(f"{uid}:duplicate_route_key:{row['route_key']}")
            unique_keys.add(row["route_key"])
            provenance_ids.extend(str(value) for value in row["source_binary_route_ids"])
        if len(provenance_ids) != len(set(provenance_ids)) or set(provenance_ids) != converted_ids:
            provenance_errors.append(f"{uid}:source_provenance_partition")
        canonical = result["canonical_4action_route"]
        if canonical is not None and canonical["route_key"] not in unique_keys:
            provenance_errors.append(f"{uid}:canonical_absent")
        if "current_unified_all_off" not in result:
            semantic_errors.append(f"{uid}:missing_current_all_off")

    failure_rows = [
        row
        for path in sorted((args.root / "failures").glob("*.jsonl"))
        for row in read_jsonl(path)
    ]
    unresolved_failure_rows = [row for row in failure_rows if str(row.get("uid")) not in observed]
    progress = [
        row
        for path in sorted((args.root / "progress").glob("*.jsonl"))
        for row in read_jsonl(path)
    ]
    worker_starts = [row for row in progress if row.get("event") == "worker_start"]
    dynamic_starts = [row for row in worker_starts if row.get("work_assignment") == "atomic_dynamic"]
    top_contract = args.root / "execution_contract_v1.json"
    top_contract_id = None
    if top_contract.is_file():
        top_contract_id = json.loads(top_contract.read_text()).get("contract_sha256")
    jobs = slurm_jobs([value for value in args.job_ids.split(",") if value])
    total_job_seconds = sum(row["elapsed_seconds"] for row in jobs)
    checks = {
        "all_source_samples_completed": set(observed) == set(expected),
        "no_invalid_or_duplicate_source_uids": len(expected) == len(sources) and not invalid_records,
        "all_record_checksums_valid": not checksum_errors,
        "source_route_accounting_exact": not source_accounting_errors
        and sum(int(row["source_positive_route_count"]) for row in sources)
        == counters["raw_converted"] + counters["raw_source_route_replay_failure"],
        "all_conversion_semantics_valid": not semantic_errors,
        "deduplicated_provenance_partitions_valid_sources": not provenance_errors,
        "one_execution_contract": len(contract_ids) == 1,
        "top_level_execution_contract_matches": len(contract_ids) == 1
        and top_contract_id == next(iter(contract_ids)),
        "no_unresolved_worker_failures": not unresolved_failure_rows,
        "all_sixteen_dynamic_workers_observed": set(int(row["rank"]) for row in dynamic_starts)
        == set(range(16)),
        "two_replicas_on_all_eight_gpus_observed": {
            (int(row["gpu_index"]), int(row["replica_index"])) for row in dynamic_starts
        }
        >= {(gpu, replica) for gpu in range(8) for replica in range(2)},
        "slurm_job_provenance_present": bool(jobs),
        "final_slurm_launch_completed": bool(jobs)
        and jobs[-1]["state"].startswith("COMPLETED")
        and jobs[-1]["exit_code"].startswith("0:"),
    }
    telemetry = telemetry_summary(args.telemetry)
    checks["telemetry_covers_all_eight_gpus"] = telemetry.get("gpu_indices") == list(
        range(8)
    ) and int(telemetry.get("samples", 0)) > 0
    report = {
        "schema_version": "four_action_label_full_integrity_audit_v1",
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": file_sha256(args.source_manifest),
        "expected_samples": len(expected),
        "completed_samples": len(observed),
        "missing_uids": sorted(set(expected) - set(observed)),
        "source_positive_routes": sum(
            int(row["source_positive_route_count"]) for row in sources
        ),
        "replay_valid_routes": counters["raw_converted"],
        "replay_failure_routes": counters["raw_source_route_replay_failure"],
        "record_checksum_errors": checksum_errors,
        "invalid_records": invalid_records,
        "source_accounting_errors": source_accounting_errors,
        "semantic_errors": semantic_errors,
        "provenance_errors": provenance_errors,
        "execution_contract_sha256": sorted(contract_ids),
        "worker_failure_rows": failure_rows,
        "unresolved_worker_failure_rows": unresolved_failure_rows,
        "slurm_jobs": jobs,
        "telemetry": telemetry,
        "throughput": {
            "wall_hours_across_all_launches": total_job_seconds / 3600.0,
            "allocated_gpu_hours_across_all_launches": total_job_seconds * 8 / 3600.0,
            "samples_per_wall_hour": (
                None
                if total_job_seconds == 0
                else len(observed) * 3600.0 / total_job_seconds
            ),
            "source_routes_per_wall_second": (
                None
                if total_job_seconds == 0
                else sum(int(row["source_positive_route_count"]) for row in sources)
                / total_job_seconds
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{file_sha256(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
