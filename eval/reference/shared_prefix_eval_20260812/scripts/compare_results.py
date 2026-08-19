#!/usr/bin/env python3
"""Compare reproduced core-VQA or external evaluation with its reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["external", "core-vqa", "pope"], default="external")
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    if args.suite in {"core-vqa", "pope"}:
        if args.suite == "core-vqa":
            expected_count = 12849
            label = "CORE VQA"
            candidate_default = (
                ROOT / "results/core_vqa_regeneration/reproduced_sw31_core_vqa/merged_final"
            )
            reference_default = ROOT / "results/reference_core_vqa"
        else:
            expected_count = 9000
            label = "POPE"
            candidate_default = ROOT / "results/pope_regeneration/reproduced_sw31_pope/merged_final"
            reference_default = ROOT / "results/reference_pope"
        candidate_dir = args.candidate_dir or candidate_default
        reference_dir = args.reference_dir or reference_default
        candidate_rows = {
            str(row["uid"]): row
            for row in read_jsonl(candidate_dir / "heldout_generation_rows.jsonl")
        }
        reference_rows = {
            str(row["uid"]): row
            for row in read_jsonl(reference_dir / "heldout_generation_rows.jsonl")
        }
        failures: list[str] = []
        if set(candidate_rows) != set(reference_rows):
            failures.append(
                f"{label} UID sets differ: candidate={len(candidate_rows)} "
                f"reference={len(reference_rows)}"
            )
        else:
            exact_fields = [
                "benchmark",
                "metric_name",
                "baseline_prediction",
                "baseline_correct",
                "router_prediction",
                "router_correct",
                "selected_visual_on_mask",
                "selected_num_visual_on_layers",
            ]
            numeric_fields = ["baseline_score", "router_score"]
            for uid in sorted(candidate_rows):
                left = candidate_rows[uid]
                right = reference_rows[uid]
                for field in exact_fields:
                    if left[field] != right[field]:
                        failures.append(
                            f"{uid}.{field}: candidate={left[field]!r} reference={right[field]!r}"
                        )
                        break
                for field in numeric_fields:
                    if abs(float(left[field]) - float(right[field])) > args.tolerance:
                        failures.append(
                            f"{uid}.{field}: candidate={left[field]} reference={right[field]}"
                        )
                        break
                if len(failures) >= 50:
                    break
        if failures:
            print(f"{label} REPRODUCTION COMPARISON FAILED", file=sys.stderr)
            for failure in failures:
                print(f"- {failure}", file=sys.stderr)
            return 1
        print(f"{label} REPRODUCTION COMPARISON PASSED: {expected_count} UID outputs match")
        return 0

    candidate_dir = args.candidate_dir or ROOT / "results/reproduced_prefix_admission_eval"
    reference_dir = args.reference_dir or ROOT / "results/reference_original"

    candidate = json.loads((candidate_dir / "summary.json").read_text())
    reference = json.loads((reference_dir / "summary.json").read_text())
    failures: list[str] = []
    if candidate["selected_prefix_layers"] != reference["selected_prefix_layers"]:
        failures.append("selected prefix differs")

    policies = ["all_on", "ungated_hybrid", "learned_admission", "oracle_admission"]
    fields = [
        "baseline_accuracy",
        "selected_accuracy",
        "accuracy_delta",
        "route_fraction",
        "mean_visual_on_layers",
        "harm_count",
        "rescue_count",
    ]
    for policy in policies:
        left = candidate["external_test"][policy]
        right = reference["external_test"][policy]
        for field in fields:
            if abs(float(left[field]) - float(right[field])) > args.tolerance:
                failures.append(
                    f"{policy}.{field}: candidate={left[field]} reference={right[field]}"
                )

    candidate_rows = {
        str(row["uid"]): row
        for row in read_jsonl(candidate_dir / "external_predictions.jsonl")
    }
    reference_rows = {
        str(row["uid"]): row
        for row in read_jsonl(reference_dir / "external_predictions.jsonl")
    }
    if set(candidate_rows) != set(reference_rows):
        failures.append("external prediction UID sets differ")
    else:
        row_fields = [
            "benchmark",
            "outcome",
            "baseline_correct",
            "router_correct",
            "selected_num_visual_on_layers",
            "use_sparse_hybrid",
        ]
        for uid in sorted(candidate_rows):
            for field in row_fields:
                if candidate_rows[uid][field] != reference_rows[uid][field]:
                    failures.append(
                        f"{uid}.{field}: candidate={candidate_rows[uid][field]} "
                        f"reference={reference_rows[uid][field]}"
                    )
                    break
            if len(failures) >= 50:
                break

    if failures:
        print("REPRODUCTION COMPARISON FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("REPRODUCTION COMPARISON PASSED: summary metrics and 5807 UID decisions match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
