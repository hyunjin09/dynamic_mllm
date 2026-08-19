#!/usr/bin/env python3
"""Build checksum-bound P5 summaries from the strict P4 record index."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from label_regeneration.summary import aggregate_summaries, summarize_record


EXPECTED_COUNTS = {"gqa": 4000, "textvqa": 2000, "chartqa": 2000}


def digest_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def write_checksum(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest(path)}  {path.name}\n", encoding="utf-8"
    )


def pct(count: int, total: int) -> str:
    return f"{100.0 * count / total:.2f}%" if total else "n/a"


def markdown_report(summary: dict, *, summary_path: Path, rows_path: Path) -> str:
    overall = summary["overall"]
    lines = [
        "# Label Regeneration P5 Outcome Summary",
        "",
        "Status: **PASS**",
        "",
        "P5 summarizes the strict P4-frozen GQA, TextVQA, and ChartQA cache. "
        "WeMath2.0-Pro is excluded. Route-transition, segment, and pairwise-Hamming "
        "analysis remains unopened for P6.",
        "",
        "## Population and current ALL-ON",
        "",
        "| Dataset | Samples | Current correct | Current wrong | ≥1 valid | ≥20 valid | Corrected current-wrong |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("gqa", "textvqa", "chartqa"):
        data = summary["by_dataset"][dataset]
        correct = data["current_all_on"].get("correct", 0)
        wrong = data["current_all_on"].get("wrong", 0)
        recovered = data["correction"]["recovered"]
        lines.append(
            f"| {dataset.upper()} | {data['samples']:,} | {correct:,} | {wrong:,} | "
            f"{data['coverage']['with_at_least_1']:,} | {data['coverage']['with_at_least_20']:,} | "
            f"{recovered:,}/{wrong:,} ({pct(recovered, wrong)}) |"
        )
    oc = overall["current_all_on"].get("correct", 0)
    ow = overall["current_all_on"].get("wrong", 0)
    orecovered = overall["correction"]["recovered"]
    lines.append(
        f"| **Total** | **{overall['samples']:,}** | **{oc:,}** | **{ow:,}** | "
        f"**{overall['coverage']['with_at_least_1']:,}** | "
        f"**{overall['coverage']['with_at_least_20']:,}** | "
        f"**{orecovered:,}/{ow:,} ({pct(orecovered, ow)})** |"
    )
    lines += [
        "",
        "## Valid-route coverage",
        "",
        "| Dataset | Zero | ≥1 | ≥5 | ≥10 | ≥20 | Mean | Median | P10 | P90 | P95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("gqa", "textvqa", "chartqa"):
        coverage = summary["by_dataset"][dataset]["coverage"]
        dist = coverage["valid_route_count"]
        lines.append(
            f"| {dataset.upper()} | {coverage['zero_valid']:,} | {coverage['with_at_least_1']:,} | "
            f"{coverage['with_at_least_5']:,} | {coverage['with_at_least_10']:,} | "
            f"{coverage['with_at_least_20']:,} | {dist['mean']:.2f} | {dist['median']:.1f} | "
            f"{dist['p10']:.1f} | {dist['p90']:.1f} | {dist['p95']:.1f} |"
        )
    coverage = overall["coverage"]
    dist = coverage["valid_route_count"]
    lines.append(
        f"| **Total** | **{coverage['zero_valid']:,}** | **{coverage['with_at_least_1']:,}** | "
        f"**{coverage['with_at_least_5']:,}** | **{coverage['with_at_least_10']:,}** | "
        f"**{coverage['with_at_least_20']:,}** | **{dist['mean']:.2f}** | **{dist['median']:.1f}** | "
        f"**{dist['p10']:.1f}** | **{dist['p90']:.1f}** | **{dist['p95']:.1f}** |"
    )
    lines += [
        "",
        "## Correction and preservation",
        "",
        "A correction is a valid evaluated mask for a sample whose authoritative current ALL-ON route is wrong. "
        "For current-correct samples, preservation reports the least visual computation among valid evaluated masks.",
        "",
        "| Dataset | Correction recovery | Correcting routes/recovered mean | Min ON median | Min ON mean | Mean OFF-layer saving |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("gqa", "textvqa", "chartqa"):
        data = summary["by_dataset"][dataset]
        correction = data["correction"]
        preservation = data["preservation"]
        recovered_dist = correction["correcting_route_count_recovered_only"]
        min_on = preservation["minimum_visual_on_valid_route"]
        lines.append(
            f"| {dataset.upper()} | {correction['recovered']:,}/{correction['eligible_current_wrong']:,} "
            f"({pct(correction['recovered'], correction['eligible_current_wrong'])}) | "
            f"{recovered_dist['mean']:.2f} | {min_on['median']:.1f} | {min_on['mean']:.2f} | "
            f"{100.0 * preservation['mean_visual_compute_saving_fraction']:.2f}% |"
        )
    correction = overall["correction"]
    preservation = overall["preservation"]
    recovered_dist = correction["correcting_route_count_recovered_only"]
    min_on = preservation["minimum_visual_on_valid_route"]
    lines.append(
        f"| **Total** | **{correction['recovered']:,}/{correction['eligible_current_wrong']:,} "
        f"({pct(correction['recovered'], correction['eligible_current_wrong'])})** | "
        f"**{recovered_dist['mean']:.2f}** | **{min_on['median']:.1f}** | **{min_on['mean']:.2f}** | "
        f"**{100.0 * preservation['mean_visual_compute_saving_fraction']:.2f}%** |"
    )
    lines += [
        "",
        "## Execution-contract drift",
        "",
        "Historical easy/hard membership is metadata only; the current executor is authoritative.",
        "",
        "| Dataset | Correct→correct | Correct→wrong | Wrong→correct | Wrong→wrong |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in ("gqa", "textvqa", "chartqa"):
        drift = summary["by_dataset"][dataset]["contract_drift"]
        lines.append(
            f"| {dataset.upper()} | {drift.get('stable_correct', 0):,} | "
            f"{drift.get('historical_correct_to_current_wrong', 0):,} | "
            f"{drift.get('historical_wrong_to_current_correct', 0):,} | "
            f"{drift.get('stable_wrong', 0):,} |"
        )
    drift = overall["contract_drift"]
    lines.append(
        f"| **Total** | **{drift.get('stable_correct', 0):,}** | "
        f"**{drift.get('historical_correct_to_current_wrong', 0):,}** | "
        f"**{drift.get('historical_wrong_to_current_correct', 0):,}** | "
        f"**{drift.get('stable_wrong', 0):,}** |"
    )
    lines += [
        "",
        "## Integrity and scope",
        "",
        f"- Per-sample rows: `{summary['integrity']['per_sample_rows']:,}`.",
        f"- P4 audit SHA-256: `{summary['integrity']['p4_audit_sha256']}`.",
        f"- P4 record-index SHA-256: `{summary['integrity']['p4_record_index_sha256']}`.",
        f"- Raw records reverified against P4 checksums: `{summary['integrity']['raw_records_checksum_verified']:,}`.",
        f"- Per-sample summary: `{rows_path}` (`{summary['integrity']['per_sample_summary_sha256']}`).",
        f"- Aggregate JSON: `{summary_path}`.",
        "- No record was excluded or replaced; no likelihood or predictor training was run.",
        "- P6 diversity/transition analysis, P7 splits, P8 derived views, P9 final freeze, and P10 training were not executed.",
        "",
        "## P5 decision",
        "",
        "P5 passes because all 8,000 P4-frozen records reconcile into complete per-sample summaries and the required "
        "current ALL-ON, correction, preservation, budget, and contract-drift statistics are defined and saved. "
        "The next bounded action is P6 route-diversity and transition-structure analysis.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p4-audit", required=True)
    parser.add_argument("--record-index", required=True)
    parser.add_argument("--per-sample-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args()

    p4_path = Path(args.p4_audit)
    index_path = Path(args.record_index)
    p4 = json.loads(p4_path.read_text(encoding="utf-8"))
    if not p4.get("passed") or p4.get("expected_dataset_counts") != EXPECTED_COUNTS:
        raise RuntimeError("P4 audit is not a passed 8K GQA/TextVQA/ChartQA audit")
    if digest(index_path) != p4.get("record_index_sha256"):
        raise RuntimeError("P4 record-index checksum mismatch")

    index_rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line]
    if len(index_rows) != 8000:
        raise RuntimeError(f"expected 8000 P4 index rows, found {len(index_rows)}")
    summaries = []
    dataset_counts: dict[str, int] = {}
    for index_row in index_rows:
        path = Path(index_row["record_path"])
        payload = path.read_bytes()
        actual_hash = digest_bytes(payload)
        if actual_hash != index_row["record_sha256"]:
            raise RuntimeError(f"raw record checksum mismatch: {path}")
        record = json.loads(payload)
        row = summarize_record(record)
        if row["uid"] != index_row["uid"] or row["dataset"] != index_row["benchmark"]:
            raise RuntimeError(f"P4 index binding mismatch: {path}")
        summaries.append(row)
        dataset_counts[row["dataset"]] = dataset_counts.get(row["dataset"], 0) + 1
    if dataset_counts != EXPECTED_COUNTS:
        raise RuntimeError(f"unexpected P5 population: {dataset_counts}")

    rows_path = Path(args.per_sample_output)
    summary_path = Path(args.summary_output)
    report_path = Path(args.report_output)
    for path in (rows_path, summary_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in summaries), encoding="utf-8"
    )
    summary = aggregate_summaries(summaries)
    summary["integrity"] = {
        "p4_audit_path": str(p4_path),
        "p4_audit_sha256": digest(p4_path),
        "p4_record_index_path": str(index_path),
        "p4_record_index_sha256": digest(index_path),
        "raw_records_checksum_verified": len(summaries),
        "per_sample_rows": len(summaries),
        "per_sample_summary_path": str(rows_path),
        "per_sample_summary_sha256": digest(rows_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(
        markdown_report(summary, summary_path=summary_path, rows_path=rows_path), encoding="utf-8"
    )
    for path in (rows_path, summary_path, report_path):
        write_checksum(path)
    print(
        json.dumps(
            {
                "passed": True,
                "samples": len(summaries),
                "current_all_on": summary["overall"]["current_all_on"],
                "correction": summary["overall"]["correction"],
                "summary": str(summary_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
