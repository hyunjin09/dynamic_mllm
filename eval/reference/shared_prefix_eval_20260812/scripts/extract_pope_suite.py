#!/usr/bin/env python3
"""Extract a portable POPE-only suite and its frozen SW31 reference rows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


BENCHMARKS = ("pope_adversarial", "pope_popular", "pope_random")
EXPECTED_COUNTS = {benchmark: 3000 for benchmark in BENCHMARKS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument("--source-generation-rows-jsonl", type=Path, required=True)
    parser.add_argument("--output-data-dir", type=Path, required=True)
    parser.add_argument("--output-reference-dir", type=Path, required=True)
    parser.add_argument("--baseline-output-jsonl", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def outcome(row: dict[str, Any]) -> str:
    baseline = bool(row["baseline_correct"])
    router = bool(row["router_correct"])
    if baseline and router:
        return "preserve"
    if baseline and not router:
        return "harm"
    if not baseline and router:
        return "rescue"
    return "unsolved"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {"all": rows}
    groups.update({benchmark: [row for row in rows if row["benchmark"] == benchmark] for benchmark in BENCHMARKS})
    summary: dict[str, Any] = {}
    for name, group in groups.items():
        count = len(group)
        summary[name] = {
            "samples": count,
            "baseline_correct_rate": sum(bool(row["baseline_correct"]) for row in group) / count,
            "router_correct_rate": sum(bool(row["router_correct"]) for row in group) / count,
            "paired_correct_delta": sum(
                int(bool(row["router_correct"])) - int(bool(row["baseline_correct"])) for row in group
            )
            / count,
            "avg_selected_layers": sum(int(row["selected_num_visual_on_layers"]) for row in group) / count,
            "unique_masks": len({str(row["selected_mask_key"]) for row in group}),
            "outcomes": dict(Counter(outcome(row) for row in group)),
        }
    return summary


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# POPE Reference Evaluation",
        "",
        "| split | n | all-on acc | SW31 acc | delta | mean ON | unique masks | harm | rescue |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("all",) + BENCHMARKS:
        group = summary[name]
        lines.append(
            f"| {name} | {group['samples']} | {group['baseline_correct_rate']:.2%} | "
            f"{group['router_correct_rate']:.2%} | {group['paired_correct_delta']:+.2%} | "
            f"{group['avg_selected_layers']:.2f} | {group['unique_masks']} | "
            f"{group['outcomes'].get('harm', 0)} | {group['outcomes'].get('rescue', 0)} |"
        )
    lines.extend(
        [
            "",
            "Prediction is mapped to `yes` only when it starts with `yes`, and to `no` only when it starts with `no`.",
            "The combined row is a 9,000-sample micro-average across the three POPE splits.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    source_manifest_path = args.source_data_dir / "samples.jsonl"
    manifest_rows = [
        row for row in read_jsonl(source_manifest_path) if str(row.get("benchmark")) in BENCHMARKS
    ]
    counts = Counter(str(row["benchmark"]) for row in manifest_rows)
    if len(manifest_rows) != 9000 or dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"unexpected POPE manifest population: rows={len(manifest_rows)}, counts={dict(counts)}")
    if len({str(row["uid"]) for row in manifest_rows}) != 9000:
        raise ValueError("POPE manifest contains duplicate UIDs")

    for row in manifest_rows:
        relpaths = row.get("image_relpaths") or [row["image_relpath"]]
        for relative in relpaths:
            link_or_copy(args.source_data_dir / str(relative), args.output_data_dir / str(relative))

    manifest_path = args.output_data_dir / "samples.jsonl"
    write_jsonl(manifest_path, manifest_rows)
    source_revision = "4db1276663dfa5eb8ad16a52d24c31a09e470896"
    write_json(
        args.output_data_dir / "summary.json",
        {
            "dataset_version": "heldout_pope_v1",
            "source_dataset": "lmms-lab/POPE",
            "source_dataset_name": "Full",
            "source_revision": source_revision,
            "total_count": 9000,
            "counts": EXPECTED_COUNTS,
            "metric_name": "pope_yes_no_accuracy",
            "correctness_threshold": 1.0,
            "max_new_tokens": 128,
            "max_pixels": None,
        },
    )
    write_json(
        args.output_data_dir / "source_manifest_alignment.json",
        {
            "matched": True,
            "rows": 9000,
            "source_manifest_sha256": sha256(source_manifest_path),
            "portable_manifest_sha256": sha256(manifest_path),
        },
    )
    (args.output_data_dir / "README.md").write_text(
        "# POPE Held-out Suite\n\n"
        "This is the exact POPE adversarial/popular/random subset extracted from "
        "`heldout_lmms_recommended_plus_pope_seed_lite_v1`. The authoritative input is "
        "`samples.jsonl`; images are stored under `images/<split>/`.\n",
        encoding="utf-8",
    )

    generation_rows = [
        row
        for row in read_jsonl(args.source_generation_rows_jsonl)
        if str(row.get("benchmark")) in BENCHMARKS
    ]
    indexed = {str(row["uid"]): row for row in generation_rows}
    expected_uids = [str(row["uid"]) for row in manifest_rows]
    if len(generation_rows) != 9000 or set(indexed) != set(expected_uids):
        raise ValueError("POPE generation rows do not match the manifest")
    ordered = [indexed[uid] for uid in expected_uids]
    write_jsonl(args.output_reference_dir / "heldout_generation_rows.jsonl", ordered)
    baseline_rows = [
        {
            "uid": row["uid"],
            "sample_id": row["sample_id"],
            "benchmark": row["benchmark"],
            "metric_name": row["metric_name"],
            "correctness_threshold": row["correctness_threshold"],
            "baseline_prediction": row["baseline_prediction"],
            "baseline_score": row["baseline_score"],
            "baseline_correct": row["baseline_correct"],
        }
        for row in ordered
    ]
    write_jsonl(args.baseline_output_jsonl, baseline_rows)
    summary = summarize(ordered)
    write_json(
        args.output_reference_dir / "summary.json",
        {
            "evaluation_version": "pope_reference_eval_v1",
            "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "router_checkpoint_sha256": "6ecf2f9119b78d5d11c969b4602b93cecc59d27aab43440abacb84421c4af255",
            "manifest_sha256": sha256(manifest_path),
            "baseline_rows_sha256": sha256(args.baseline_output_jsonl),
            "reference_rows_sha256": sha256(args.output_reference_dir / "heldout_generation_rows.jsonl"),
            "summary": summary,
        },
    )
    (args.output_reference_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(f"POPE suite complete: rows={len(manifest_rows)}, counts={dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
