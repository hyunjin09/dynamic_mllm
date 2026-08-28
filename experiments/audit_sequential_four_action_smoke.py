#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from tools.research_analysis.four_action.sequential_label_conversion import (
    binary_to_four_action,
)
from tools.research_analysis.four_action.sequential_label_jobs import file_sha256


DATASETS = {"gqa", "textvqa", "chartqa", "wemath20_standard", "wemath2pro"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _decision_valid(decision: dict[str, Any]) -> bool:
    action = decision.get("action")
    full = bool(decision.get("full_correct"))
    read = decision.get("read_only_correct")
    write = decision.get("write_only_correct")
    if action == "FULL":
        return full and read is None and write is None
    if action == "READ_ONLY":
        return not full and read is True and write in {True, False}
    if action == "WRITE_ONLY":
        return not full and write is True and read in {True, False}
    if action == "IGNORE":
        return not full and read is False and write is False
    return False


def conversion_semantics_valid(conversion: dict[str, Any], *, route_type: str) -> bool:
    if conversion.get("status") != "converted":
        return (
            conversion.get("status") == "source_route_replay_failure"
            and bool(conversion.get("failure_reason"))
            and not conversion.get("final_branches")
            and not conversion.get("steps")
        )
    source = tuple(binary_to_four_action(conversion["source_binary_route"]))
    branches = conversion.get("final_branches", [])
    if not branches or not all(bool(row.get("evaluation", {}).get("correct")) for row in branches):
        return False
    if route_type == "C2C":
        return (
            conversion.get("label_semantics") == "preserving_c2c"
            and conversion.get("steps") == []
            and len(branches) == 1
            and tuple(branches[0].get("route", [])) == source
            and branches[0].get("decisions", []) == []
        )
    if route_type != "W2C" or conversion.get("label_semantics") != "corrective_w2c":
        return False

    off_layers = [index for index, action in enumerate(source) if action == "IGNORE"]
    steps = conversion.get("steps", [])
    if [int(step.get("layer", -1)) for step in steps] != off_layers:
        return False
    for step in steps:
        incoming = int(step.get("incoming_branch_count", -1))
        full = int(step.get("full_restored_count", -1))
        read = int(step.get("read_only_only_count", -1))
        write = int(step.get("write_only_only_count", -1))
        both = int(step.get("both_partial_correct_count", -1))
        ignore = int(step.get("ignore_fallback_count", -1))
        outgoing = int(step.get("outgoing_branch_count", -1))
        if incoming != full + read + write + both + ignore:
            return False
        if outgoing != full + read + write + 2 * both + ignore:
            return False

    for branch in branches:
        decisions = branch.get("decisions", [])
        if [int(row.get("layer", -1)) for row in decisions] != off_layers:
            return False
        reconstructed = list(source)
        for decision in decisions:
            if not _decision_valid(decision):
                return False
            reconstructed[int(decision["layer"])] = str(decision["action"])
        if reconstructed != list(branch.get("route", [])):
            return False
    return True


def _target_texts(evaluation: dict[str, Any]) -> tuple[str, ...] | None:
    rows = evaluation.get("correct_target_scores")
    if rows is None:
        return None
    return tuple(sorted(str(row["text"]) for row in rows))


def _record_targets_stable(record: dict[str, Any]) -> bool:
    reference = _target_texts(record.get("current_unified_full", {}))
    if not reference:
        return False
    evaluations = [record.get("current_unified_all_off", {})]
    for conversion in record.get("raw_conversions", []):
        evaluations.append(conversion.get("source_route_evaluation", {}))
        evaluations.extend(branch.get("evaluation", {}) for branch in conversion.get("final_branches", []))
    evaluations.extend(row.get("evaluation", {}) for row in record.get("unique_valid_four_action_routes", []))
    return all(_target_texts(evaluation) == reference for evaluation in evaluations)


def build_smoke_audit(
    records: list[dict[str, Any]],
    *,
    expected_uids: set[str],
    failure_rows: list[dict[str, Any]],
    progress: list[dict[str, Any]],
    checksum_errors: list[str],
    resume_verified: bool,
    slurm_jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    observed = [str(row.get("uid")) for row in records]
    observed_set = set(observed)
    unresolved_failures = [
        row for row in failure_rows if str(row.get("uid")) not in observed_set
    ]
    conversions = [row for record in records for row in record.get("raw_conversions", [])]
    converted = [row for row in conversions if row.get("status") == "converted"]
    branches = [row for conversion in converted for row in conversion.get("final_branches", [])]
    unique = [
        row
        for record in records
        for row in record.get("unique_valid_four_action_routes", [])
    ]
    parity = [
        row
        for record in records
        for row in record.get("pilot_old_binary_semantic_checks", [])
    ]
    starts = [row for row in progress if row.get("event") == "worker_start"]
    completes = [row for row in progress if row.get("event") == "worker_complete"]
    start_ranks = {int(row["rank"]) for row in starts}
    complete_ranks = {int(row["rank"]) for row in completes}
    layouts = {
        (int(row["gpu_index"]), int(row["replica_index"])) for row in starts
    }
    contracts = {
        row.get("execution_contract", {}).get("contract_sha256") for row in records
    }
    checks = {
        "exactly_eight_expected_samples_present": (
            len(expected_uids) == 8 and observed_set == expected_uids
        ),
        "no_duplicate_samples": len(observed) == len(observed_set),
        "all_sample_records_passed": len(records) == 8 and all(bool(row.get("passed")) for row in records),
        "no_unresolved_worker_failures": not unresolved_failures,
        "all_record_checksums_valid": not checksum_errors,
        "all_five_datasets_present": {str(row.get("dataset")) for row in records} == DATASETS,
        "both_route_types_present": {str(row.get("route_type")) for row in records} == {"W2C", "C2C"},
        "source_route_accounting_exact": sum(
            int(row.get("source_positive_route_count", 0)) for row in records
        ) == len(conversions),
        "source_replay_outcomes_exactly_accounted": bool(conversions)
        and all(
            row.get("status") in {"converted", "source_route_replay_failure"}
            for row in conversions
        )
        and sum(
            int(record.get("source_route_replay_valid_count", 0))
            + int(record.get("source_route_replay_failure_count", 0))
            for record in records
        )
        == len(conversions),
        "all_conversions_follow_exact_sequential_semantics": bool(converted)
        and all(
            conversion_semantics_valid(conversion, route_type=record["route_type"])
            for record in records
            for conversion in record.get("raw_conversions", [])
        ),
        "all_final_branch_occurrences_correct": bool(branches)
        and all(bool(row.get("evaluation", {}).get("correct")) for row in branches),
        "all_unique_routes_correct": bool(unique)
        and all(bool(row.get("evaluation", {}).get("correct")) for row in unique),
        "old_binary_full_ignore_semantic_parity": len(parity) == 8
        and all(
            bool(row.get("generated_ids_match"))
            and bool(row.get("generated_answer_match"))
            and bool(row.get("correctness_match"))
            for row in parity
        ),
        "evaluator_targets_stable_across_routes": bool(records)
        and all(_record_targets_stable(record) for record in records),
        "route_cache_reuse_observed": bool(records)
        and all(int(row.get("route_evaluation_cache", {}).get("cache_hits", 0)) > 0 for row in records),
        "single_execution_contract": len(contracts) == 1 and None not in contracts,
        "all_eight_workers_started": start_ranks == set(range(8)),
        "one_replica_on_all_eight_gpus": layouts == {(gpu, 0) for gpu in range(8)},
        "all_eight_workers_completed": complete_ranks == set(range(8)),
        "exact_resume_verified": bool(resume_verified),
        "declared_smoke_job_completed": bool(slurm_jobs)
        and all(
            str(row.get("state", "")).startswith("COMPLETED")
            and str(row.get("exit_code", "0:0")).startswith("0:")
            for row in slurm_jobs
        ),
    }
    branch_steps = [step for row in converted for step in row.get("steps", [])]
    return {
        "schema_version": "exact_sequential_four_action_smoke_audit_v1",
        "passed": all(checks.values()),
        "expected_samples": len(expected_uids),
        "completed_samples": len(records),
        "missing_uids": sorted(expected_uids - observed_set),
        "extra_uids": sorted(observed_set - expected_uids),
        "source_routes": len(conversions),
        "replay_valid_routes": len(converted),
        "replay_failure_routes": len(conversions) - len(converted),
        "final_branch_occurrences": len(branches),
        "unique_final_routes": len(unique),
        "real_smoke_path_counts": {
            "full_restorations": sum(int(row["full_restored_count"]) for row in branch_steps),
            "read_only_only": sum(int(row["read_only_only_count"]) for row in branch_steps),
            "write_only_only": sum(int(row["write_only_only_count"]) for row in branch_steps),
            "both_partial_correct": sum(int(row["both_partial_correct_count"]) for row in branch_steps),
            "ignore_fallback": sum(int(row["ignore_fallback_count"]) for row in branch_steps),
        },
        "maximum_active_branch_count": max(
            (
                int(record.get("branching_summary", {}).get("maximum_active_branch_count", 0))
                for record in records
            ),
            default=0,
        ),
        "checks": checks,
        "unresolved_worker_failures": unresolved_failures,
        "checksum_errors": checksum_errors,
        "jobs": slurm_jobs,
    }


def _slurm(job_id: str) -> dict[str, Any]:
    output = subprocess.run(
        [
            "sacct",
            "-j",
            job_id,
            "-X",
            "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES",
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
            "exit_code": row[2],
            "elapsed_seconds": int(row[3]),
            "allocated_tres": row[4],
        }
        if len(row) >= 5
        else {"job_id": job_id, "raw": output}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the exact sequential smoke run.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sequential_four_action_label_conversion.yaml"),
    )
    parser.add_argument("--job-ids", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/4action_sequential_label_conversion/smoke_audit_v1.json"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(config["output_root"]) / "smoke"
    manifest = read_jsonl(root / "smoke_manifest_v1.jsonl")
    paths = sorted((root / "records").glob("*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    failures = [
        row
        for path in sorted((root / "failures").glob("*.jsonl"))
        for row in read_jsonl(path)
    ]
    progress = [
        row
        for path in sorted((root / "progress").glob("*.jsonl"))
        for row in read_jsonl(path)
    ]
    checksum_errors = []
    for path in paths:
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if (
            not sidecar.exists()
            or not sidecar.read_text(encoding="utf-8").split()
            or sidecar.read_text(encoding="utf-8").split()[0] != file_sha256(path)
        ):
            checksum_errors.append(str(path))
    resume_path = Path(config["analysis_root"]) / "smoke_resume_verification_v1.json"
    resume_verified = (
        resume_path.is_file()
        and bool(json.loads(resume_path.read_text(encoding="utf-8")).get("passed"))
    )
    report = build_smoke_audit(
        records,
        expected_uids={str(row["uid"]) for row in manifest},
        failure_rows=failures,
        progress=progress,
        checksum_errors=checksum_errors,
        resume_verified=resume_verified,
        slurm_jobs=[_slurm(value) for value in args.job_ids.split(",") if value],
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {args.output}")
        if args.output.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("existing smoke audit differs from recomputed evidence")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        args.output.with_suffix(args.output.suffix + ".sha256").write_text(
            f"{file_sha256(args.output)}  {args.output.name}\n", encoding="utf-8"
        )
    real_paths = report["real_smoke_path_counts"]
    markdown = f"""# Exact Sequential Four-Action Smoke Test

- Result: **{'PASS' if report['passed'] else 'FAIL'}**
- Samples: {report['completed_samples']}/{report['expected_samples']}
- Source routes: {report['source_routes']} ({report['replay_valid_routes']} replay-valid, {report['replay_failure_routes']} replay failures)
- Final branch occurrences: {report['final_branch_occurrences']}
- Unique final routes: {report['unique_final_routes']}
- Maximum active branch count: {report['maximum_active_branch_count']}
- Real-data path counts: FULL restoration {real_paths['full_restorations']}, READ_ONLY-only {real_paths['read_only_only']}, WRITE_ONLY-only {real_paths['write_only_only']}, both-partial branching {real_paths['both_partial_correct']}, IGNORE fallback {real_paths['ignore_fallback']}.
- Synthetic truth-table coverage: `tests/test_sequential_four_action_label_conversion.py`.
- Exact resume verification: {'passed' if report['checks']['exact_resume_verified'] else 'failed'}.
- Old binary FULL/IGNORE semantic parity: {'passed' if report['checks']['old_binary_full_ignore_semantic_parity'] else 'failed'}.
- Worker topology: 8 workers, one replica on each of 8 GPUs.

All check details and Slurm provenance are in `smoke_audit_v1.json`.
"""
    report_targets = [
        Path(config["analysis_root"]) / "smoke_test_report.md",
        Path(config["output_root"]) / "reports" / "smoke_test_report.md",
    ]
    for target in report_targets:
        if target.exists():
            if not args.resume or target.read_text(encoding="utf-8") != markdown:
                raise RuntimeError(f"existing smoke report differs: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(markdown, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
