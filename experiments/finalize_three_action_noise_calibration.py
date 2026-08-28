#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from tools.research_analysis.four_action.three_action_jobs import file_sha256
from tools.research_analysis.four_action.three_action_labels import (
    calibrate_repeatability_epsilon,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_calibration_report(
    records: list[dict[str, Any]],
    *,
    expected_uids: set[str],
    floor: float,
    quantile: float,
    failure_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    observed = [str(row.get("uid")) for row in records]
    unresolved_failure_rows = [
        row for row in failure_rows if str(row.get("uid")) not in set(observed)
    ]
    controls = [control for row in records for control in row.get("repeatability_controls", [])]
    differences = [
        float(value)
        for control in controls
        for value in control.get("signed_differences_from_first", [])
    ]
    contract_hashes = {
        row.get("execution_contract", {}).get("contract_sha256") for row in records
    }
    quantity_matches = all(
        control.get("score_quantity")
        == ("answer_alignment_margin" if row.get("route_type") == "W2C" else "S_correct")
        for row in records
        for control in row.get("repeatability_controls", [])
    )
    checks = {
        "all_expected_samples_present": set(observed) == expected_uids,
        "no_duplicate_samples": len(observed) == len(set(observed)),
        "all_sample_records_passed": bool(records) and all(bool(row.get("passed")) for row in records),
        "no_unresolved_worker_failures": not unresolved_failure_rows,
        "controls_present": bool(controls) and bool(differences),
        "identical_generation_and_correctness": bool(controls) and all(
            bool(row.get("generated_ids_identical")) and bool(row.get("correctness_identical"))
            for row in controls
        ),
        "decision_score_quantity_matches_route_type": quantity_matches,
        "single_execution_contract": len(contract_hashes) == 1 and all(contract_hashes),
    }
    calibration = (
        calibrate_repeatability_epsilon(
            signed_differences=differences,
            floor=floor,
            quantile=quantile,
        )
        if differences
        else {
            "epsilon": None,
            "selection_rule": "unavailable: no within-unified repeat differences",
        }
    )
    return {
        "schema_version": "three_action_noise_calibration_v1",
        "passed": all(checks.values()),
        "threshold_source": "within_unified_identical_route_repeatability",
        "native_vs_unified_drift_used": False,
        "expected_samples": len(expected_uids),
        "completed_samples": len(records),
        "control_count": len(controls),
        "difference_count": len(differences),
        "missing_uids": sorted(expected_uids - set(observed)),
        "extra_uids": sorted(set(observed) - expected_uids),
        "execution_contract_sha256": sorted(
            "<missing>" if value is None else str(value) for value in contract_hashes
        ),
        "checks": checks,
        "historical_failure_count": len(failure_rows),
        "unresolved_failure_count": len(unresolved_failure_rows),
        "worker_failure_rows": failure_rows,
        "unresolved_worker_failure_rows": unresolved_failure_rows,
        **calibration,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze three-action score repeatability epsilon.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/three_action_label_conversion.yaml")
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(config["output_root"]) / "calibration"
    manifest_path = Path(config["output_root"]) / "pilot" / "pilot_manifest_v1.jsonl"
    manifest = read_jsonl(manifest_path)
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "records").glob("*.json"))]
    failure_rows = [
        row for path in sorted((root / "failures").glob("*.jsonl")) for row in read_jsonl(path)
    ]
    checksum_errors = []
    for path in sorted((root / "records").glob("*.json")):
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not sidecar.exists() or sidecar.read_text(encoding="utf-8").split()[0] != file_sha256(path):
            checksum_errors.append(str(path))
    noise = config["noise_calibration"]
    report = build_calibration_report(
        records,
        expected_uids={str(row["uid"]) for row in manifest},
        floor=float(noise["mean_score_floor"]),
        quantile=float(noise["absolute_quantile"]),
        failure_rows=failure_rows,
    )
    report["record_checksum_errors"] = checksum_errors
    report["checks"]["all_record_checksums_valid"] = not checksum_errors
    report["passed"] = all(report["checks"].values())
    output = Path(noise["artifact_path"])
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite frozen epsilon artifact {output}")
        existing = json.loads(output.read_text(encoding="utf-8"))
        sidecar = output.with_suffix(output.suffix + ".sha256")
        checksum_valid = (
            sidecar.is_file()
            and bool(sidecar.read_text(encoding="utf-8").split())
            and sidecar.read_text(encoding="utf-8").split()[0] == file_sha256(output)
        )
        if existing != report or not checksum_valid:
            raise RuntimeError("existing frozen epsilon artifact differs from recomputed calibration")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0 if existing.get("passed") else 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
