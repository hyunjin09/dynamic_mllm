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


DATASETS = {"gqa", "textvqa", "chartqa", "wemath20_standard", "wemath2pro"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_pilot_audit(
    records: list[dict[str, Any]],
    *,
    expected_uids: set[str],
    failure_rows: list[dict[str, Any]],
    progress: list[dict[str, Any]],
    epsilon_sha256: str,
    minimum_jaccard: float,
    checksum_errors: list[str],
    slurm_jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    observed = [str(row.get("uid")) for row in records]
    unresolved_failure_rows = [
        row for row in failure_rows if str(row.get("uid")) not in set(observed)
    ]
    conversions = [row for record in records for row in record.get("raw_conversions", [])]
    converted = [row for row in conversions if row.get("status") == "converted"]
    positives = [route for row in converted for route in row.get("positive_routes", [])]
    unique = [route for record in records for route in record.get("unique_valid_three_action_routes", [])]
    binary_checks = [
        check for record in records for check in record.get("pilot_old_binary_semantic_checks", [])
    ]
    starts = {int(row["rank"]): row for row in progress if row.get("event") == "worker_start"}
    completes = {int(row["rank"]) for row in progress if row.get("event") == "worker_complete"}
    hard = [row for row in converted if row.get("label_semantics") == "W2C_HARD_CORRECTIVE"]
    soft = [row for row in converted if row.get("label_semantics") == "W2C_SOFT_ALIGNMENT"]
    c2c = [row for row in converted if row.get("label_semantics") == "C2C_COMPENSATED_ALIGNMENT"]
    stabilities = [row.get("pilot_beam_stability") for row in converted]
    contract_hashes = {
        row.get("execution_contract", {}).get("contract_sha256") for row in records
    }
    target_policy_present = all(
        bool(row.get("current_unified_full", {}).get("correct_target_scores"))
        and bool(row.get("current_unified_full", {}).get("selected_correct_target"))
        and row.get("current_unified_full", {}).get("score_quantity")
        == ("S_correct_minus_S_full_wrong" if row.get("route_type") == "W2C" else "S_correct")
        for row in records
    )

    c2c_global_gain = True
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
                    c2c_global_gain = False

    efficiency_rows = [row["execution_efficiency"] for row in converted]
    checks = {
        "all_expected_samples_present": set(observed) == expected_uids,
        "no_duplicate_samples": len(observed) == len(set(observed)),
        "all_sample_records_passed": bool(records) and all(bool(row.get("passed")) for row in records),
        "no_unresolved_worker_failures": not unresolved_failure_rows,
        "all_record_checksums_valid": not checksum_errors,
        "all_five_datasets_present": {str(row.get("dataset")) for row in records} == DATASETS,
        "source_route_accounting_exact": sum(
            int(row.get("source_positive_route_count", 0)) for row in records
        ) == len(conversions),
        "all_positive_routes_jointly_correct": bool(positives) and all(
            bool(row["evaluation"].get("correct")) for row in positives
        ),
        "all_unique_routes_jointly_correct": bool(unique) and all(
            bool(row["evaluation"].get("correct")) for row in unique
        ),
        "c2c_positive_routes_have_global_support_gain": c2c_global_gain,
        "old_binary_semantic_parity": bool(binary_checks) and all(
            bool(row.get("generated_ids_match"))
            and bool(row.get("generated_answer_match"))
            and bool(row.get("correctness_match"))
            for row in binary_checks
        ),
        "evaluator_compatible_target_policy_present": bool(records) and target_policy_present,
        "w2c_hard_path_exercised": bool(hard),
        "w2c_soft_path_exercised": bool(soft),
        "c2c_alignment_path_exercised": bool(c2c) and any(row.get("positive_routes") for row in c2c),
        "beam_canonical_stable": bool(stabilities) and all(
            bool(row)
            and (bool(row.get("canonical_route_match")) or bool(row.get("both_have_no_positive_route")))
            for row in stabilities
        ),
        "beam_positive_set_overlap_above_floor": bool(stabilities) and all(
            float(row.get("positive_route_jaccard", 0.0)) >= minimum_jaccard
            for row in stabilities if row
        ),
        "epsilon_artifact_bound_to_every_record": bool(records) and all(
            row.get("execution_contract", {}).get("epsilon_sha256") == epsilon_sha256
            for row in records
        ),
        "single_execution_contract": len(contract_hashes) == 1 and all(contract_hashes),
        "all_sixteen_workers_started": set(starts) == set(range(16)),
        "two_replicas_on_all_eight_gpus": {
            (int(row["gpu_index"]), int(row["replica_index"])) for row in starts.values()
        } == {(gpu, replica) for gpu in range(8) for replica in range(2)},
        "all_workers_completed": completes == set(range(16)),
        "three_action_cache_savings_observed": bool(efficiency_rows) and all(
            int(row["decomposition_new_cache_misses"])
            <= 2 * int(row["candidate_positions"])
            and int(row["theoretical_four_state_evaluations_avoided"])
            >= 2 * int(row["candidate_positions"])
            for row in efficiency_rows
        ),
        "all_declared_slurm_jobs_completed": bool(slurm_jobs) and all(
            str(row.get("state", "")).startswith("COMPLETED") for row in slurm_jobs
        ),
    }
    return {
        "schema_version": "three_action_answer_aligned_pilot_audit_v1",
        "passed": all(checks.values()),
        "expected_samples": len(expected_uids),
        "completed_samples": len(records),
        "missing_uids": sorted(expected_uids - set(observed)),
        "extra_uids": sorted(set(observed) - expected_uids),
        "source_routes": len(conversions),
        "replay_valid_routes": len(converted),
        "replay_failure_routes": len(conversions) - len(converted),
        "positive_route_occurrences": len(positives),
        "unique_positive_routes": len(unique),
        "w2c_hard_routes": len(hard),
        "w2c_soft_routes": len(soft),
        "c2c_routes": len(c2c),
        "beam_stability_rows": len(stabilities),
        "execution_contract_sha256": sorted(
            "<missing>" if value is None else str(value) for value in contract_hashes
        ),
        "epsilon_sha256": epsilon_sha256,
        "checks": checks,
        "worker_failure_rows": failure_rows,
        "unresolved_worker_failure_rows": unresolved_failure_rows,
        "checksum_errors": checksum_errors,
        "jobs": slurm_jobs,
    }


def _slurm(job_id: str) -> dict[str, Any]:
    output = subprocess.run(
        ["sacct", "-j", job_id, "-X", "--format=JobIDRaw,State,ElapsedRaw,AllocTRES", "-n", "-P"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    row = output.splitlines()[0].split("|") if output else []
    return (
        {"job_id": row[0], "state": row[1], "elapsed_seconds": int(row[2]), "allocated_tres": row[3]}
        if len(row) >= 4 else {"job_id": job_id, "raw": output}
    )


def _telemetry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"samples": 0}
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
    parser = argparse.ArgumentParser(description="Audit the answer-aligned three-action pilot.")
    parser.add_argument("--config", type=Path, default=Path("configs/three_action_label_conversion.yaml"))
    parser.add_argument("--job-ids", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--telemetry", type=Path, default=Path("analysis/three_action_answer_aligned_label_conversion/pilot_gpu_telemetry.csv"))
    parser.add_argument("--output", type=Path, default=Path("analysis/three_action_answer_aligned_label_conversion/pilot_audit_v1.json"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(config["output_root"]) / "pilot"
    manifest = read_jsonl(root / "pilot_manifest_v1.jsonl")
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
    report = build_pilot_audit(
        records,
        expected_uids={str(row["uid"]) for row in manifest},
        failure_rows=failures,
        progress=progress,
        epsilon_sha256=file_sha256(epsilon_path),
        minimum_jaccard=float(config["pilot_gate"]["minimum_positive_route_jaccard"]),
        checksum_errors=checksum_errors,
        slurm_jobs=jobs,
    )
    report["telemetry"] = _telemetry(args.telemetry)
    report["checks"]["telemetry_covers_all_eight_gpus"] = report["telemetry"].get("gpu_indices") == list(range(8))
    report["passed"] = all(report["checks"].values())
    elapsed = sum(float(row.get("elapsed_seconds", 0)) for row in jobs)
    if elapsed:
        report["throughput"] = {
            "samples_per_hour": len(records) * 3600.0 / elapsed,
            "source_routes_per_second": len([row for record in records for row in record.get("raw_conversions", [])]) / elapsed,
            "allocated_gpu_hours": elapsed * 8 / 3600.0,
        }
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
            raise RuntimeError("existing pilot audit differs from recomputed evidence")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0 if existing.get("passed") else 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{file_sha256(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
