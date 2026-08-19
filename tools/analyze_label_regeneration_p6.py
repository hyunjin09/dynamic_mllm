#!/usr/bin/env python3
"""Compute checksum-bound P6 route-diversity diagnostics."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from label_regeneration.diversity import aggregate_diversity, summarize_record_diversity


EXPECTED_COUNTS = {"gqa": 4000, "textvqa": 2000, "chartqa": 2000}
DATASET_ORDER = ("gqa", "textvqa", "chartqa")


def digest_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def write_checksum(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest(path)}  {path.name}\n", encoding="utf-8"
    )


def markdown_report(summary: dict, *, rows_path: Path, summary_path: Path) -> str:
    overall = summary["overall"]
    primary = overall["sample_balanced"]
    weighted = overall["route_weighted"]
    structural = overall["structural_frequencies"]
    lines = [
        "# Label Regeneration P6 Route-Diversity Analysis",
        "",
        "Status: **PASS**",
        "",
        "This outcome-blind post-generation analysis uses every valid complete mask from the strict "
        "P4-frozen GQA, TextVQA, and ChartQA cache. WeMath2.0-Pro is excluded. No valid-route cap "
        "or later training-view selection was applied.",
        "",
        "## Primary sample-balanced geometry",
        "",
        "Each positive sample contributes one mean so samples with many successful routes do not dominate.",
        "",
        "| Dataset | Positive samples | Valid masks | Mean ON | Mean transitions | Mean Hamming to min | Mean pairwise Hamming | Mean run length |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASET_ORDER:
        data = summary["by_dataset"][dataset]
        sb = data["sample_balanced"]
        lines.append(
            f"| {dataset.upper()} | {data['samples_with_valid_routes']:,} | {data['valid_masks']:,} | "
            f"{sb['mean_visual_on_count']['mean']:.2f} | {sb['mean_transition_count']['mean']:.2f} | "
            f"{sb['mean_hamming_to_minimum']['mean']:.2f} | {sb['mean_pairwise_hamming']['mean']:.2f} | "
            f"{sb['mean_segment_length']['mean']:.2f} |"
        )
    lines.append(
        f"| **Total** | **{overall['samples_with_valid_routes']:,}** | **{overall['valid_masks']:,}** | "
        f"**{primary['mean_visual_on_count']['mean']:.2f}** | "
        f"**{primary['mean_transition_count']['mean']:.2f}** | "
        f"**{primary['mean_hamming_to_minimum']['mean']:.2f}** | "
        f"**{primary['mean_pairwise_hamming']['mean']:.2f}** | "
        f"**{primary['mean_segment_length']['mean']:.2f}** |"
    )
    lines += [
        "",
        "## Route- and pair-weighted distributions",
        "",
        "These summaries describe all masks or all within-sample mask pairs and therefore give more "
        "weight to high-yield samples. They are secondary to the sample-balanced table.",
        "",
        "| Quantity | Count | Mean | Median | P10 | P90 | P95 | Maximum |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("Visual-ON layers", "visual_on_count"),
        ("ON/OFF transitions", "transition_count"),
        ("Hamming to minimum-ON route", "hamming_to_minimum_route"),
        ("Within-sample pairwise Hamming", "pairwise_hamming"),
        ("Maximal run length", "all_segment_length"),
        ("ON-run length", "on_segment_length"),
        ("OFF-run length", "off_segment_length"),
    ):
        stats = weighted[key]
        lines.append(
            f"| {label} | {stats['count']:,} | {stats['mean']:.2f} | {stats['median']:.1f} | "
            f"{stats['p10']:.1f} | {stats['p90']:.1f} | {stats['p95']:.1f} | {stats['maximum']:.1f} |"
        )
    lines += [
        "",
        "## Transition and segment structure",
        "",
        f"- Valid masks with at most 3 transitions: `{structural['transition_le_3']:,}` "
        f"(`{100.0 * structural['transition_le_3_fraction']:.2f}%`).",
        f"- Valid masks with at least 14 transitions: `{structural['transition_ge_14']:,}` "
        f"(`{100.0 * structural['transition_ge_14_fraction']:.2f}%`).",
        f"- Valid ALL-OFF anchors: `{structural['all_off_masks']:,}`; valid ALL-ON anchors: "
        f"`{structural['all_on_masks']:,}`.",
        "- Segment means use maximal contiguous runs of equal ON/OFF decisions; no POLAR segmentation "
        "constraint was imposed during MCTS.",
        "",
        "## Current ALL-ON status stratification",
        "",
        "| Dataset/status | Positive samples | Valid masks | Mean transitions | Mean pairwise Hamming |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASET_ORDER:
        for status in ("correct", "wrong"):
            data = summary["by_dataset_and_current_status"][dataset][status]
            sb = data["sample_balanced"]
            lines.append(
                f"| {dataset.upper()} / {status} | {data['samples_with_valid_routes']:,} | "
                f"{data['valid_masks']:,} | {sb['mean_transition_count']['mean']:.2f} | "
                f"{sb['mean_pairwise_hamming']['mean']:.2f} |"
            )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The observed quantities describe the geometry of MCTS-discovered successful masks. High Hamming "
        "distance or frequent transitions indicates that the raw valid sets are not merely duplicate masks, "
        "but it does not prove that a factorized predictor can generalize, that segment prediction will fail, "
        "or that any route yields real latency reduction. A later POLAR representation remains a controlled "
        "derived baseline from these same masks.",
        "",
        "## Integrity and scope",
        "",
        f"- P4 record-index SHA-256: `{summary['integrity']['p4_record_index_sha256']}`.",
        f"- P5 per-sample SHA-256: `{summary['integrity']['p5_per_sample_sha256']}`.",
        f"- Raw records checksum-verified: `{summary['integrity']['raw_records_checksum_verified']:,}`.",
        f"- Per-sample diversity rows: `{summary['integrity']['per_sample_rows']:,}` at `{rows_path}`.",
        f"- Aggregate JSON: `{summary_path}`.",
        "- P7 splitting, P8 derived supervision, P9 final freeze, and P10 predictor training were not executed.",
        "",
        "## P6 decision",
        "",
        "P6 passes because all valid masks were analyzed without truncation, exact within-sample pairwise "
        "distances were computed, sample-balanced and weighted summaries were both saved, and all inputs "
        "remain checksum-bound. The next bounded action is P7 image-group-disjoint predictor split freezing.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p4-audit", required=True)
    parser.add_argument("--record-index", required=True)
    parser.add_argument("--p5-summary", required=True)
    parser.add_argument("--p5-per-sample", required=True)
    parser.add_argument("--per-sample-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args()

    p4_path = Path(args.p4_audit)
    index_path = Path(args.record_index)
    p5_summary_path = Path(args.p5_summary)
    p5_rows_path = Path(args.p5_per_sample)
    p4 = json.loads(p4_path.read_text(encoding="utf-8"))
    p5_summary = json.loads(p5_summary_path.read_text(encoding="utf-8"))
    if not p4.get("passed") or p4.get("expected_dataset_counts") != EXPECTED_COUNTS:
        raise RuntimeError("P4 is not the passed original-pool 8K audit")
    if digest(index_path) != p4.get("record_index_sha256"):
        raise RuntimeError("P4 record-index checksum mismatch")
    if digest(p5_rows_path) != p5_summary["integrity"]["per_sample_summary_sha256"]:
        raise RuntimeError("P5 per-sample checksum mismatch")

    p5_by_uid = {
        row["uid"]: row
        for row in (
            json.loads(line) for line in p5_rows_path.read_text(encoding="utf-8").splitlines() if line
        )
    }
    index_rows = [
        json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if len(index_rows) != 8000 or len(p5_by_uid) != 8000:
        raise RuntimeError("P4/P5 population is not exactly 8,000 unique records")

    rows = []
    dataset_counts: dict[str, int] = {}
    for index_row in index_rows:
        path = Path(index_row["record_path"])
        payload = path.read_bytes()
        if digest_bytes(payload) != index_row["record_sha256"]:
            raise RuntimeError(f"raw record checksum mismatch: {path}")
        record = json.loads(payload)
        row = summarize_record_diversity(record)
        p5_row = p5_by_uid.get(row["uid"])
        if p5_row is None:
            raise RuntimeError(f"missing P5 binding for {row['uid']}")
        if (
            row["dataset"] != index_row["benchmark"]
            or row["dataset"] != p5_row["dataset"]
            or row["current_all_on_status"] != p5_row["current_all_on_status"]
            or row["valid_route_count"] != p5_row["valid_route_count"]
            or row["minimum_visual_on_valid_route"] != p5_row["minimum_visual_on_valid_route"]
        ):
            raise RuntimeError(f"P4/P5/P6 binding mismatch: {row['uid']}")
        rows.append(row)
        dataset_counts[row["dataset"]] = dataset_counts.get(row["dataset"], 0) + 1
    if dataset_counts != EXPECTED_COUNTS:
        raise RuntimeError(f"unexpected P6 population: {dataset_counts}")

    rows_path = Path(args.per_sample_output)
    summary_path = Path(args.summary_output)
    report_path = Path(args.report_output)
    for path in (rows_path, summary_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    summary = aggregate_diversity(rows)
    summary["integrity"] = {
        "p4_audit_sha256": digest(p4_path),
        "p4_record_index_sha256": digest(index_path),
        "p5_summary_sha256": digest(p5_summary_path),
        "p5_per_sample_sha256": digest(p5_rows_path),
        "raw_records_checksum_verified": len(rows),
        "per_sample_rows": len(rows),
        "per_sample_output_sha256": digest(rows_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(
        markdown_report(summary, rows_path=rows_path, summary_path=summary_path), encoding="utf-8"
    )
    for path in (rows_path, summary_path, report_path):
        write_checksum(path)
    print(
        json.dumps(
            {
                "passed": True,
                "samples": summary["overall"]["samples"],
                "valid_masks": summary["overall"]["valid_masks"],
                "sample_balanced": summary["overall"]["sample_balanced"],
                "summary": str(summary_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
