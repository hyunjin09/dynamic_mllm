#!/usr/bin/env python3
"""Outcome-blind feasibility audit for the P7 predictor split design."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path


TARGETS = {
    "gqa": {
        "train": 3250,
        "validation": 500,
        "validation_historical_correct": 250,
        "internal_test": 250,
        "internal_test_historical_correct": 125,
    },
    "textvqa": {
        "train": 1625,
        "validation": 250,
        "validation_historical_correct": 125,
        "internal_test": 125,
        "internal_test_historical_correct": 62,
    },
    "chartqa": {
        "train": 1625,
        "validation": 250,
        "validation_historical_correct": 125,
        "internal_test": 125,
        "internal_test_historical_correct": 63,
    },
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def singleton_heldout_sufficiency(
    groups: dict[str, list[dict]], required_correct: int, required_wrong: int
) -> tuple[Counter, bool]:
    """Return singleton source-cell counts and whether they witness feasibility."""
    counts = Counter(
        group[0]["historical_all_on_status"]
        for group in groups.values()
        if len(group) == 1
    )
    return counts, (
        counts["correct"] >= required_correct and counts["wrong"] >= required_wrong
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--overlap-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    overlap = json.loads(args.overlap_audit.read_text())
    if len(rows) != 8000 or not overlap.get("passed"):
        raise RuntimeError("source population or external-overlap gate is not valid")

    dataset_results = {}
    all_feasible = True
    for dataset, target in TARGETS.items():
        dataset_rows = [row for row in rows if row["benchmark"] == dataset]
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in dataset_rows:
            groups[str(row["image_group_id"])].append(row)
        required_correct = (
            target["validation_historical_correct"]
            + target["internal_test_historical_correct"]
        )
        required_wrong = (
            target["validation"]
            - target["validation_historical_correct"]
            + target["internal_test"]
            - target["internal_test_historical_correct"]
        )
        singleton_status, singleton_feasible = singleton_heldout_sufficiency(
            groups, required_correct, required_wrong
        )
        feasible = singleton_feasible and (
            target["train"] + target["validation"] + target["internal_test"]
            == len(dataset_rows)
        )
        all_feasible &= feasible
        dataset_results[dataset] = {
            **target,
            "records": len(dataset_rows),
            "image_groups": len(groups),
            "singleton_groups_by_historical_status": dict(sorted(singleton_status.items())),
            "heldout_singleton_sufficiency_witness": {
                "required_historical_correct": required_correct,
                "required_historical_wrong": required_wrong,
                "feasible": feasible,
            },
        }

    report = {
        "schema_version": "binary_router_p7_split_design_audit_v1",
        "inputs": {
            "source_manifest": str(args.manifest),
            "source_manifest_sha256": file_sha256(args.manifest),
            "external_overlap_audit": str(args.overlap_audit),
            "external_overlap_audit_sha256": file_sha256(args.overlap_audit),
        },
        "recommended_roles": {
            "train_records": 6500,
            "validation_records": 1000,
            "internal_test_records": 500,
            "external_transfer_test_records": 5807,
        },
        "dataset_targets": dataset_results,
        "selection_contract": {
            "group_unit": "image_group_id",
            "seed": 20260809,
            "allowed_selection_fields": [
                "benchmark",
                "image_group_id",
                "historical_all_on_status",
            ],
            "forbidden_selection_fields": [
                "current_all_on_status",
                "valid_route_count",
                "correction_found",
                "route_diversity",
                "predictor_outcomes",
                "external_test_outcomes",
            ],
            "freeze_rule": (
                "stable seed-hash ordering plus deterministic grouped constrained "
                "assignment to the exact dataset/status targets"
            ),
        },
        "external_overlap": overlap["overlap"],
        "passed": all_feasible and all(
            value in (0, [], {}) for value in overlap["overlap"].values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    checksum = file_sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{checksum}  {args.output.name}\n"
    )
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
