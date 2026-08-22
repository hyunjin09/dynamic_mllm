#!/usr/bin/env python3
"""Summarize one cap predictor on the frozen external execution suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evaluate_binary_polar_external import ACTIVE_BENCHMARKS, file_sha256, write_json, write_jsonl
from experiments.merge_binary_polar_external_eval import summarize_rows, verify_and_load


POPE_OVERLAP_CLUSTER = "7b21e833d6fe982ef6c55c793c7c4fc8111b9a4876334ef7f4c470362d20ce55"


def table(groups: dict[str, Any], order: list[str]) -> str:
    lines = [
        "| Benchmark | N | ALL-ON acc | Router acc | Delta | Ratio | Harm | Rescue | Unchanged C/W | Mean ON | ON reduction | ALL-ON | ALL-OFF | Unique | Changed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        item = groups[name]
        ratio = item["predicted_correct_rate"] / item["baseline_correct_rate"] if item["baseline_correct_rate"] else None
        lines.append(
            f"| {name} | {item['records']:,} | {item['baseline_correct_rate']:.4f} | "
            f"{item['predicted_correct_rate']:.4f} | {item['correctness_delta']:+.4f} | "
            f"{ratio:.2%} | {item['full_correct_to_predicted_wrong']} | "
            f"{item['full_wrong_to_predicted_correct']} | {item['unchanged_correct']}/{item['unchanged_wrong']} | "
            f"{item['mean_visual_on_layers']:.2f} | {item['mean_visual_on_reduction_from_full']:.2f} | "
            f"{item['all_on_rate']:.2%} | {item['all_off_rate']:.2%} | "
            f"{item['unique_predicted_masks']} | {item['behavior_changing_executions']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--preflight-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cap", type=int, choices=(26, 24, 22, 20, 18), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    rows, metadata = verify_and_load(
        args.input_root, 1, modality="image_question", preflight_path=args.preflight_path
    )
    groups = {benchmark: [row for row in rows if row["benchmark"] == benchmark] for benchmark in ACTIVE_BENCHMARKS}
    groups.update(
        {
            "core_vqa": [row for row in rows if row["suite"] == "core_vqa"],
            "external_multiple_choice": [row for row in rows if row["suite"] == "external_multiple_choice"],
            "pope": [row for row in rows if row["suite"] == "pope"],
            "pope_image_disjoint": [row for row in rows if row["suite"] == "pope" and row["cluster_key"] != POPE_OVERLAP_CLUSTER],
        }
    )
    if len(groups["pope_image_disjoint"]) != 8982:
        raise RuntimeError("frozen POPE image-disjoint sensitivity must contain 8,982 records")
    summaries = {
        name: summarize_rows(current, "image_question", bootstrap_draws=args.bootstrap_draws, seed=args.seed + 2 * index)
        for index, (name, current) in enumerate(groups.items())
    }
    analysis = {
        "schema_version": "binary_cap_external_analysis_v1",
        "integrity_status": "PASS",
        "cap": args.cap,
        "records": len(rows),
        "modality": "image_question",
        "scientific_baseline": "current_live_all_on",
        "checkpoint": {"path": str(args.checkpoint), "sha256": file_sha256(args.checkpoint)},
        "groups": summaries,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_root / "external_results_v1.jsonl"
    analysis_path = args.output_root / "external_analysis_v1.json"
    write_jsonl(rows_path, rows)
    write_json(analysis_path, analysis)
    manifest = {
        "schema_version": "binary_cap_external_analysis_manifest_v1",
        "integrity_status": "PASS",
        "cap": args.cap,
        "preflight": {"path": str(args.preflight_path), "sha256": file_sha256(args.preflight_path)},
        "results": {"path": str(rows_path), "sha256": file_sha256(rows_path)},
        "analysis": {"path": str(analysis_path), "sha256": file_sha256(analysis_path)},
        "shard_metadata": metadata,
        "records": len(rows),
    }
    write_json(args.output_root / "analysis_manifest_v1.json", manifest)
    order = list(ACTIVE_BENCHMARKS) + ["core_vqa", "external_multiple_choice", "pope", "pope_image_disjoint"]
    content = [
        f"# CAP={args.cap} Image+Question External Evaluation",
        "",
        f"- Integrity: **PASS**; {len(rows):,} frozen active UIDs completed exactly once.",
        "- DocVQA is excluded; the scientific baseline is current live ALL-ON.",
        "- Compute is visual-ON decoder-layer count, not measured latency.",
        "",
        table(summaries, order),
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(content), encoding="utf-8")
    args.report.with_suffix(args.report.suffix + ".sha256").write_text(
        f"{file_sha256(args.report)}  {args.report.name}\n", encoding="utf-8"
    )
    print(json.dumps({"integrity_status": "PASS", "cap": args.cap, "records": len(rows)}))


if __name__ == "__main__":
    main()
