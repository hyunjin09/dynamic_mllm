#!/usr/bin/env python3
"""Freeze the outcome-blind P7 binary-predictor train/validation split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


TARGETS = {
    "gqa": {"total": 4000, "validation": 500, "validation_correct": 250},
    "textvqa": {"total": 2000, "validation": 250, "validation_correct": 125},
    "chartqa": {"total": 2000, "validation": 250, "validation_correct": 125},
}
SELECTION_FIELDS = ("benchmark", "image_group_id", "historical_all_on_status")
FORBIDDEN_SELECTION_FIELDS = (
    "current_all_on_status",
    "valid_route_count",
    "correction_found",
    "route_diversity",
    "predictor_outcomes",
    "external_test_outcomes",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_sidecar(path: Path) -> str:
    digest = file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def select_validation_groups(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    target_records: int,
    target_correct: int,
    seed: int,
) -> frozenset[str]:
    """Choose an exact validation subset using only frozen source metadata."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["benchmark"]) != dataset:
            raise ValueError(f"row {row.get('uid')} is not in dataset {dataset}")
        status = str(row["historical_all_on_status"])
        if status not in {"correct", "wrong"}:
            raise ValueError(f"invalid historical status for {row.get('uid')}: {status}")
        groups[str(row["image_group_id"])].append(row)

    ordered_groups = sorted(
        groups,
        key=lambda group_id: (
            sha256(f"{seed}:{dataset}:{group_id}".encode()).hexdigest(),
            group_id,
        ),
    )
    # Each reachable state is stored once. Its parent always comes from an
    # earlier group because additions use a snapshot of the prior states.
    reachable: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {
        (0, 0): None
    }
    target = (target_records, target_correct)
    for group_id in ordered_groups:
        group = groups[group_id]
        size = len(group)
        correct = sum(
            str(row["historical_all_on_status"]) == "correct" for row in group
        )
        additions: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
        for state in tuple(reachable):
            candidate = (state[0] + size, state[1] + correct)
            if candidate[0] > target_records or candidate[1] > target_correct:
                continue
            if candidate not in reachable and candidate not in additions:
                additions[candidate] = (state, group_id)
        reachable.update(additions)
        if target in reachable:
            break

    if target not in reachable:
        raise ValueError(
            f"no exact image-group-disjoint validation subset for {dataset}: "
            f"records={target_records}, historical_correct={target_correct}"
        )

    selected: set[str] = set()
    state = target
    while state != (0, 0):
        parent = reachable[state]
        if parent is None:
            raise RuntimeError(f"broken split backpointer at {state}")
        state, group_id = parent
        selected.add(group_id)
    return frozenset(selected)


def summarize_split(
    source_rows: list[dict[str, Any]],
    assignments: dict[str, str],
    current_by_uid: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dataset in TARGETS:
        summary[dataset] = {}
        for split in ("train", "validation"):
            rows = [
                row
                for row in source_rows
                if row["benchmark"] == dataset and assignments[row["uid"]] == split
            ]
            summary[dataset][split] = {
                "records": len(rows),
                "image_groups": len({row["image_group_id"] for row in rows}),
                "historical_all_on_status": dict(
                    sorted(Counter(row["historical_all_on_status"] for row in rows).items())
                ),
                "current_all_on_status_descriptive_only": dict(
                    sorted(
                        Counter(
                            current_by_uid[row["uid"]]["current_all_on_status"]
                            for row in rows
                        ).items()
                    )
                ),
            }
    return summary


def render_report(audit: dict[str, Any]) -> str:
    lines = [
        "# Label Regeneration P7 Predictor Split",
        "",
        "The exact outcome-blind image-group-disjoint predictor split is frozen.",
        "Selection used only dataset, image group, historical source-cell status,",
        "and seed `20260809`. Current executor outcomes were joined only after",
        "assignment for descriptive auditing.",
        "",
        "| Dataset | Split | Records | Image groups | Historical correct | Historical wrong | Current correct | Current wrong |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("gqa", "textvqa", "chartqa"):
        for split in ("train", "validation"):
            row = audit["counts"][dataset][split]
            historical = row["historical_all_on_status"]
            current = row["current_all_on_status_descriptive_only"]
            lines.append(
                f"| {dataset.upper()} | {split} | {row['records']} | "
                f"{row['image_groups']} | {historical.get('correct', 0)} | "
                f"{historical.get('wrong', 0)} | {current.get('correct', 0)} | "
                f"{current.get('wrong', 0)} |"
            )
    lines.extend(
        [
            "",
            f"- Train: `{audit['totals']['train_records']}` records.",
            f"- Validation: `{audit['totals']['validation_records']}` records.",
            f"- Cross-split image groups: `{audit['integrity']['cross_split_image_groups']}`.",
            f"- Duplicate UIDs: `{audit['integrity']['duplicate_uids']}`.",
            "- Route success, valid-route count, correction discovery, diversity, and predictor/evaluation outcomes were not selection inputs.",
            "- P8, P9, and predictor training were not executed.",
            "",
            "Artifacts:",
            "",
            f"- `{audit['artifacts']['split_manifest']}`",
            f"- `{audit['artifacts']['split_manifest_sha256']}`",
            f"- `{audit['artifacts']['audit_sha256']}` (audit file checksum)",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--current-summary", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    source_rows = read_jsonl(args.source_manifest)
    current_rows = read_jsonl(args.current_summary)
    if len(source_rows) != 8000 or len({row["uid"] for row in source_rows}) != 8000:
        raise ValueError("source manifest must contain 8,000 unique UIDs")
    current_by_uid = {row["uid"]: row for row in current_rows}
    if len(current_rows) != 8000 or set(current_by_uid) != {
        row["uid"] for row in source_rows
    }:
        raise ValueError("current summary must match all 8,000 source UIDs exactly")

    validation_groups: dict[str, frozenset[str]] = {}
    for dataset, target in TARGETS.items():
        rows = [row for row in source_rows if row["benchmark"] == dataset]
        if len(rows) != target["total"]:
            raise ValueError(f"unexpected {dataset} population: {len(rows)}")
        validation_groups[dataset] = select_validation_groups(
            rows,
            dataset=dataset,
            target_records=target["validation"],
            target_correct=target["validation_correct"],
            seed=args.seed,
        )

    assignments = {
        row["uid"]: (
            "validation"
            if row["image_group_id"] in validation_groups[row["benchmark"]]
            else "train"
        )
        for row in source_rows
    }
    manifest_rows = [
        {
            "schema_version": "binary_predictor_split_manifest_v1",
            "uid": row["uid"],
            "sample_id": row["sample_id"],
            "benchmark": row["benchmark"],
            "image_group_id": row["image_group_id"],
            "historical_all_on_status": row["historical_all_on_status"],
            "source_row_sha256": row["source_row_sha256"],
            "split": assignments[row["uid"]],
        }
        for row in sorted(source_rows, key=lambda row: int(row["extraction_index"]))
    ]
    write_jsonl(args.output_manifest, manifest_rows)
    manifest_sha256 = write_sidecar(args.output_manifest)

    train_groups = {
        row["image_group_id"]
        for row in source_rows
        if assignments[row["uid"]] == "train"
    }
    validation_group_set = {
        row["image_group_id"]
        for row in source_rows
        if assignments[row["uid"]] == "validation"
    }
    counts = summarize_split(source_rows, assignments, current_by_uid)
    train_records = sum(value == "train" for value in assignments.values())
    validation_records = sum(value == "validation" for value in assignments.values())
    audit = {
        "schema_version": "binary_predictor_split_audit_v1",
        "passed": True,
        "selection": {
            "algorithm": "sha256_seed_ordered_exact_group_subset_dp_v1",
            "seed": args.seed,
            "group_key": "image_group_id",
            "allowed_fields": list(SELECTION_FIELDS),
            "forbidden_fields": list(FORBIDDEN_SELECTION_FIELDS),
            "historical_status_role": "source-stratum balance only",
            "current_status_role": "descriptive post-assignment audit only",
        },
        "inputs": {
            "source_manifest": str(args.source_manifest),
            "source_manifest_sha256": file_sha256(args.source_manifest),
            "current_summary": str(args.current_summary),
            "current_summary_sha256": file_sha256(args.current_summary),
            "source_plan": str(args.source_plan),
            "source_plan_sha256": file_sha256(args.source_plan),
        },
        "targets": TARGETS,
        "counts": counts,
        "totals": {
            "records": len(source_rows),
            "train_records": train_records,
            "validation_records": validation_records,
            "train_image_groups": len(train_groups),
            "validation_image_groups": len(validation_group_set),
        },
        "integrity": {
            "duplicate_uids": len(manifest_rows) - len({row["uid"] for row in manifest_rows}),
            "cross_split_image_groups": len(train_groups & validation_group_set),
            "missing_source_uids": 0,
            "unexpected_uids": 0,
            "route_outcomes_used_for_selection": False,
        },
        "artifacts": {
            "split_manifest": str(args.output_manifest),
            "split_manifest_sha256": manifest_sha256,
        },
    }
    for dataset, target in TARGETS.items():
        validation = counts[dataset]["validation"]
        if validation["records"] != target["validation"]:
            raise RuntimeError(f"{dataset} validation count mismatch")
        if validation["historical_all_on_status"].get("correct", 0) != target[
            "validation_correct"
        ]:
            raise RuntimeError(f"{dataset} historical balance mismatch")
    if train_records != 7000 or validation_records != 1000:
        raise RuntimeError("global split count mismatch")
    if audit["integrity"]["cross_split_image_groups"] or audit["integrity"]["duplicate_uids"]:
        raise RuntimeError("split integrity failure")

    # The audit checksum is recorded in the report after the final audit is written.
    write_json(args.output_audit, audit)
    audit_sha256 = write_sidecar(args.output_audit)
    audit["artifacts"]["audit_sha256"] = audit_sha256
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(render_report(audit), encoding="utf-8")
    report_sha256 = write_sidecar(args.output_report)
    print(
        json.dumps(
            {
                "passed": True,
                "train": train_records,
                "validation": validation_records,
                "manifest_sha256": manifest_sha256,
                "audit_sha256": audit_sha256,
                "report_sha256": report_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
