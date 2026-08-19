#!/usr/bin/env python3
"""Geometry-only audit used to freeze the binary label representation.

The complete source is about 9 GB, so launch this with
``infra/gpu_scheduler.py --gpus 0``. No MLLM inference is performed.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binary_policy.factorization_audit import direct_representation_gate
from binary_policy.labels import (
    deterministic_group_split,
    iter_source_json,
    load_mcts_example,
    summarize_label_geometry,
)


def digest(path: Path) -> str:
    value = sha256(path.read_bytes()).hexdigest()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    paths = list(iter_source_json(args.source))
    invalid = []
    group_to_split: dict[str, str] = {}
    group_sample_counts: dict[str, int] = {}
    cell_split_counts: dict[str, dict[str, int]] = {}
    cross_split_groups: set[str] = set()
    missing_image_sha256_count = 0

    def examples():
        nonlocal missing_image_sha256_count
        for path in paths:
            try:
                example = load_mcts_example(path, max_valid_routes=None)
                split = deterministic_group_split(example.split_group)
                previous_split = group_to_split.setdefault(example.split_group, split)
                if previous_split != split:
                    cross_split_groups.add(example.split_group)
                group_sample_counts[example.split_group] = group_sample_counts.get(example.split_group, 0) + 1
                cell = f"{example.benchmark}/{example.difficulty}"
                split_counts = cell_split_counts.setdefault(
                    cell, {"train": 0, "validation": 0, "test": 0}
                )
                split_counts[split] += 1
                missing_image_sha256_count += int(not bool(example.image_sha256))
                yield example
            except Exception as exc:
                invalid.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"})

    geometry = summarize_label_geometry(examples())
    representation_gate = direct_representation_gate(geometry) if geometry["samples"] else {"passed": False}
    source_audit_path = Path(args.source) / "final" / "audit_summary_full_v2.json"
    source_audit_match: dict[str, object]
    if source_audit_path.is_file():
        source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
        source_cells = source_audit.get("by_benchmark_difficulty", {})
        cell_comparisons = {}
        for cell_name in sorted(set(source_cells) | set(geometry["cells"])):
            expected = source_cells.get(cell_name, {})
            observed = geometry["cells"].get(cell_name, {})
            sample_match = observed.get("samples") == expected.get("samples")
            valid_match = observed.get("samples_with_valid_route") == expected.get(
                "samples_with_successful_mask"
            )
            cell_comparisons[cell_name] = {
                "expected_samples": expected.get("samples"),
                "observed_samples": observed.get("samples"),
                "expected_samples_with_valid_route": expected.get("samples_with_successful_mask"),
                "observed_samples_with_valid_route": observed.get("samples_with_valid_route"),
                "passed": sample_match and valid_match,
            }
        source_audit_match = {
            "source_audit": str(source_audit_path.resolve()),
            "source_audit_sha256": digest(source_audit_path),
            "expected_found_samples": source_audit.get("found_samples"),
            "observed_samples": geometry["samples"],
            "cells": cell_comparisons,
            "passed": (
                source_audit.get("passed") is True
                and source_audit.get("found_samples") == geometry["samples"]
                and all(row["passed"] for row in cell_comparisons.values())
            ),
        }
    else:
        source_audit_match = {
            "source_audit": str(source_audit_path.resolve()),
            "passed": False,
            "reason": "source audit is missing",
        }

    minimum_cell_split_count = min(
        (count for split_counts in cell_split_counts.values() for count in split_counts.values()),
        default=0,
    )
    group_split_audit = {
        "split_seed": 20260809,
        "fractions": {"train": 0.75, "validation": 0.125, "test": 0.125},
        "unique_group_count": len(group_sample_counts),
        "repeated_group_count": sum(count > 1 for count in group_sample_counts.values()),
        "maximum_samples_per_group": max(group_sample_counts.values(), default=0),
        "missing_image_sha256_count": missing_image_sha256_count,
        "cross_split_group_count": len(cross_split_groups),
        "cross_split_groups": sorted(cross_split_groups),
        "cell_split_counts": cell_split_counts,
        "minimum_cell_split_count": minimum_cell_split_count,
        "prospective_bp2_minimum_per_cell_split": 40,
        "passed": (
            len(group_sample_counts) > 0
            and not cross_split_groups
            and minimum_cell_split_count >= 40
        ),
    }
    report = {
        "source": str(Path(args.source).resolve()),
        "source_record_count": len(paths),
        "runs_model_inference": False,
        "loads_reference_answers": False,
        "uses_only_existing_valid_route_labels_and_mask_geometry": True,
        "geometry": geometry,
        "representation_gate": representation_gate,
        "source_audit_match": source_audit_match,
        "group_split_audit": group_split_audit,
        "invalid_records": invalid,
        "passed": (
            len(paths) == 4000
            and geometry["samples"] == 4000
            and not invalid
            and representation_gate["passed"]
            and source_audit_match["passed"]
            and group_split_audit["passed"]
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest(output)}  {output.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
