#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from experiments.audit_sequential_four_action_smoke import conversion_semantics_valid
from tools.research_analysis.four_action.sequential_label_jobs import (
    DATASETS,
    file_sha256,
    safe_filename,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json_atomic(path: Path, payload: dict[str, Any], *, resume: bool) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if path.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {path}")
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"existing artifact differs from recomputed evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{file_sha256(path)}  {path.name}\n", encoding="utf-8"
    )


def _record_valid(record: dict[str, Any], layer_count: int) -> bool:
    conversions = record.get("raw_conversions", [])
    if (
        not record.get("passed")
        or len(conversions) != int(record.get("source_positive_route_count", -1))
        or sum(row.get("status") == "converted" for row in conversions)
        != int(record.get("source_route_replay_valid_count", -1))
        or sum(row.get("status") == "source_route_replay_failure" for row in conversions)
        != int(record.get("source_route_replay_failure_count", -1))
    ):
        return False
    if not all(
        conversion_semantics_valid(conversion, route_type=record["route_type"])
        for conversion in conversions
    ):
        return False
    for row in record.get("unique_valid_four_action_routes", []):
        if (
            len(row.get("four_action_route", [])) != layer_count
            or not bool(row.get("evaluation", {}).get("correct"))
        ):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize exact sequential four-action labels.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sequential_four_action_label_conversion.yaml"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = read_jsonl(Path(config["source_manifest"]))
    source_by_uid = {str(row["uid"]): row for row in source}
    full_root = Path(config["output_root"]) / "full"
    records_root = full_root / "records"
    expected_paths = {
        uid: records_root / safe_filename(uid) for uid in source_by_uid
    }
    missing = sorted(uid for uid, path in expected_paths.items() if not path.is_file())
    observed_paths = sorted(records_root.glob("*.json"))
    expected_names = {path.name for path in expected_paths.values()}
    extra = sorted(path.name for path in observed_paths if path.name not in expected_names)
    failures = [
        row
        for path in sorted((full_root / "failures").glob("*.jsonl"))
        for row in read_jsonl(path)
    ]
    progress = [
        row
        for path in sorted((full_root / "progress").glob("*.jsonl"))
        for row in read_jsonl(path)
    ]
    completed_uids = set(source_by_uid) - set(missing)
    unresolved_failures = [
        row for row in failures if str(row.get("uid")) not in completed_uids
    ]
    checksum_errors = []
    invalid_records = []
    contracts = set()
    source_routes = 0
    replay_valid = 0
    replay_failures = 0
    unique_routes = 0
    final_branches = 0
    starts = [row for row in progress if row.get("event") == "worker_start"]
    completes = [row for row in progress if row.get("event") == "worker_complete"]

    view_root = full_root / "views"
    view_root.mkdir(parents=True, exist_ok=True)
    index_tmp = view_root / f".sample_view_index_v1.jsonl.tmp.{os.getpid()}"
    training_tmps = {
        dataset: view_root / f".{dataset}_training_v1.jsonl.tmp.{os.getpid()}"
        for dataset in DATASETS
    }
    training_handles = {
        dataset: path.open("w", encoding="utf-8")
        for dataset, path in training_tmps.items()
    }
    try:
        with index_tmp.open("w", encoding="utf-8") as index_handle:
            for uid in sorted(completed_uids):
                path = expected_paths[uid]
                sidecar = path.with_suffix(path.suffix + ".sha256")
                digest = file_sha256(path)
                if (
                    not sidecar.is_file()
                    or not sidecar.read_text(encoding="utf-8").split()
                    or sidecar.read_text(encoding="utf-8").split()[0] != digest
                ):
                    checksum_errors.append(str(path))
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("uid") != uid or not _record_valid(
                    record, int(config["layer_count"])
                ):
                    invalid_records.append(uid)
                contracts.add(
                    record.get("execution_contract", {}).get("contract_sha256")
                )
                source_routes += int(record.get("source_positive_route_count", 0))
                replay_valid += int(record.get("source_route_replay_valid_count", 0))
                replay_failures += int(record.get("source_route_replay_failure_count", 0))
                unique = record.get("unique_valid_four_action_routes", [])
                unique_routes += len(unique)
                final_branches += sum(
                    len(row.get("final_branches", []))
                    for row in record.get("raw_conversions", [])
                    if row.get("status") == "converted"
                )
                index_row = {
                    "uid": uid,
                    "dataset": record["dataset"],
                    "sample_id": record["sample_id"],
                    "record_path": str(path.resolve()),
                    "record_sha256": digest,
                    "route_type": record["route_type"],
                    "label_semantics": record["label_semantics"],
                    "source_positive_route_count": record["source_positive_route_count"],
                    "source_route_replay_valid_count": record[
                        "source_route_replay_valid_count"
                    ],
                    "source_route_replay_failure_count": record[
                        "source_route_replay_failure_count"
                    ],
                    "unique_valid_route_count": len(unique),
                    "execution_contract_sha256": record["execution_contract"][
                        "contract_sha256"
                    ],
                }
                index_handle.write(json.dumps(index_row, ensure_ascii=False, sort_keys=True) + "\n")
                for route in unique:
                    label = {
                        "schema_version": "exact_sequential_four_action_training_label_v1",
                        "uid": uid,
                        "dataset": record["dataset"],
                        "sample_id": record["sample_id"],
                        "image_id": record.get("image_id"),
                        "source_split": record["source_split"],
                        "route_type": record["route_type"],
                        "label_semantics": record["label_semantics"],
                        "all_off_seed": any(
                            bool(row.get("all_off_seed"))
                            for row in route.get("conversion_provenance", [])
                        ),
                        "current_full_answer": record["current_unified_full"][
                            "generated_answer"
                        ],
                        "current_full_correctness": record["current_unified_full"][
                            "correct"
                        ],
                        **route,
                        "executor_contract_sha256": record["execution_contract"][
                            "contract_sha256"
                        ],
                        "worker_provenance": record["runtime"],
                    }
                    training_handles[record["dataset"]].write(
                        json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n"
                    )
    finally:
        for handle in training_handles.values():
            handle.close()

    layouts = {
        (int(row["gpu_index"]), int(row["replica_index"])) for row in starts
    }
    checks = {
        "all_source_samples_present": not missing and len(observed_paths) == len(source),
        "no_extra_records": not extra,
        "no_unresolved_worker_failures": not unresolved_failures,
        "all_record_checksums_valid": not checksum_errors,
        "all_records_semantically_valid": not invalid_records,
        "single_execution_contract": len(contracts) == 1 and None not in contracts,
        "source_route_accounting_exact": source_routes
        == replay_valid + replay_failures,
        "all_sixteen_workers_started": {int(row["rank"]) for row in starts}
        == set(range(16)),
        "two_replicas_on_all_eight_gpus": layouts
        == {(gpu, replica) for gpu in range(8) for replica in range(2)},
        "all_sixteen_workers_completed": {int(row["rank"]) for row in completes}
        == set(range(16)),
    }
    report = {
        "schema_version": "exact_sequential_four_action_full_completion_audit_v1",
        "passed": all(checks.values()),
        "counts": {
            "source_samples": len(source),
            "completed_samples": len(completed_uids),
            "source_routes": source_routes,
            "source_replay_valid_routes": replay_valid,
            "source_replay_failure_routes": replay_failures,
            "final_branch_occurrences": final_branches,
            "unique_valid_routes": unique_routes,
        },
        "checks": checks,
        "missing_uids": missing,
        "extra_record_names": extra,
        "invalid_record_uids": invalid_records,
        "checksum_errors": checksum_errors,
        "unresolved_worker_failures": unresolved_failures,
        "execution_contract_sha256": sorted(str(value) for value in contracts),
    }
    if not report["passed"]:
        index_tmp.unlink(missing_ok=True)
        for path in training_tmps.values():
            path.unlink(missing_ok=True)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    index_target = view_root / "sample_view_index_v1.jsonl"
    if index_target.exists():
        if not args.resume or file_sha256(index_target) != file_sha256(index_tmp):
            raise RuntimeError("existing sample view index differs from recomputation")
        index_tmp.unlink()
    else:
        index_tmp.replace(index_target)
    index_target.with_suffix(index_target.suffix + ".sha256").write_text(
        f"{file_sha256(index_target)}  {index_target.name}\n", encoding="utf-8"
    )
    for dataset, temporary in training_tmps.items():
        target = view_root / f"{dataset}_training_v1.jsonl"
        if target.exists():
            if not args.resume or file_sha256(target) != file_sha256(temporary):
                raise RuntimeError(f"existing training view differs: {target}")
            temporary.unlink()
        else:
            temporary.replace(target)
        target.with_suffix(target.suffix + ".sha256").write_text(
            f"{file_sha256(target)}  {target.name}\n", encoding="utf-8"
        )
    output = full_root / "completion_audit_v1.json"
    write_json_atomic(output, report, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
