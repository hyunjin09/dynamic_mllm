#!/usr/bin/env python3
"""Merge heldout router eval shards into a final report."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any


DEFAULT_OUT_ROOT = Path("/mnt/hyemin/10k_dataset_mask/heldout_router_generation_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def percentile(sorted_values: list[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def bootstrap_mean_ci(values: list[float], *, repetitions: int, seed: int) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = random.Random(int(seed))
    numeric = [float(value) for value in values]
    bootstrapped = []
    for _ in range(int(repetitions)):
        bootstrapped.append(sum(numeric[rng.randrange(len(numeric))] for _ in numeric) / len(numeric))
    bootstrapped.sort()
    return {
        "n": len(numeric),
        "mean": sum(numeric) / len(numeric),
        "ci_low": percentile(bootstrapped, 0.025),
        "ci_high": percentile(bootstrapped, 0.975),
    }


def summarize(rows: list[dict[str, Any]], *, repetitions: int, seed: int) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"all": rows}
    for row in rows:
        groups.setdefault(str(row["benchmark"]), []).append(row)
    metric_fields = {
        "baseline_score": lambda row: float(row["baseline_score"]),
        "baseline_correct_rate": lambda row: float(bool(row["baseline_correct"])),
        "router_score": lambda row: float(row["router_score"]),
        "router_correct_rate": lambda row: float(bool(row["router_correct"])),
        "paired_correct_delta": lambda row: float(bool(row["router_correct"])) - float(bool(row["baseline_correct"])),
        "router_minus_baseline_score": lambda row: float(row["router_score"]) - float(row["baseline_score"]),
        "avg_selected_layers": lambda row: float(row["selected_num_visual_on_layers"]),
        "avg_selected_transitions": lambda row: float(row["selected_transition_count"]),
    }
    output: dict[str, Any] = {}
    for group_index, (name, group) in enumerate(sorted(groups.items())):
        item: dict[str, Any] = {"samples": len(group)}
        for metric_index, (metric_name, getter) in enumerate(metric_fields.items()):
            item[metric_name] = bootstrap_mean_ci(
                [getter(row) for row in group],
                repetitions=repetitions,
                seed=seed + group_index * 100 + metric_index,
            )
        item["transition_counts"] = dict(Counter(int(row["selected_transition_count"]) for row in group))
        item["unique_masks"] = len({str(row["selected_mask_key"]) for row in group})
        top_masks = Counter(str(row["selected_mask_key"]) for row in group).most_common(10)
        item["top_masks"] = [
            {"mask_key": mask, "count": count, "rate": count / len(group)}
            for mask, count in top_masks
        ]
        output[name] = item
    return output


def markdown_table(summary: dict[str, Any]) -> str:
    lines = [
        "# Heldout Router Eval - Final Ver",
        "",
        "| group | n | baseline correct | router correct | paired delta | baseline score | router score | avg on layers | unique masks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["all", "chartqa", "docvqa", "textvqa", "pope", "seedbench"]
    keys = [key for key in order if key in summary] + sorted(key for key in summary if key not in order)
    for key in keys:
        row = summary[key]
        def fmt(metric: str) -> str:
            stats = row[metric]
            return f"{stats['mean'] * 100:.2f}% [{stats['ci_low'] * 100:.2f}, {stats['ci_high'] * 100:.2f}]"
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    str(row["samples"]),
                    fmt("baseline_correct_rate"),
                    fmt("router_correct_rate"),
                    fmt("paired_correct_delta"),
                    f"{row['baseline_score']['mean']:.4f}",
                    f"{row['router_score']['mean']:.4f}",
                    f"{row['avg_selected_layers']['mean']:.2f}",
                    str(row["unique_masks"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    run_dir = args.out_root / args.run_id
    final_dir = run_dir / "merged_final"
    if final_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{final_dir} already exists; pass --overwrite")
    shard_summaries = []
    rows = []
    for shard_index in range(int(args.num_shards)):
        shard_dir = run_dir / f"shard_{shard_index:03d}_of_{args.num_shards:03d}"
        summary_path = shard_dir / "summary.json"
        rows_path = shard_dir / "heldout_generation_rows.jsonl"
        if not summary_path.exists() or not rows_path.exists():
            raise FileNotFoundError(f"missing shard output under {shard_dir}")
        shard_summaries.append(read_json(summary_path))
        rows.extend(read_jsonl(rows_path))
    uid_counts = Counter(str(row["uid"]) for row in rows)
    duplicates = [uid for uid, count in uid_counts.items() if count > 1]
    if duplicates:
        raise RuntimeError(f"duplicate UIDs across shards: {duplicates[:10]}")
    summary = summarize(rows, repetitions=int(args.bootstrap_repetitions), seed=int(args.bootstrap_seed))
    payload = {
        "evaluation_version": "heldout_online_visual_router_generation_eval_merged_v1",
        "run_id": args.run_id,
        "num_shards": int(args.num_shards),
        "samples": len(rows),
        "shard_summary_paths": [str((run_dir / f"shard_{idx:03d}_of_{args.num_shards:03d}" / "summary.json")) for idx in range(int(args.num_shards))],
        "shard_summaries": shard_summaries,
        "summary": summary,
        "outputs": {
            "rows_jsonl": str(final_dir / "heldout_generation_rows.jsonl"),
            "summary_json": str(final_dir / "summary.json"),
            "report_md": str(final_dir / "README_final_ver.md"),
        },
    }
    write_jsonl(final_dir / "heldout_generation_rows.jsonl", rows)
    write_json(final_dir / "summary.json", payload)
    (final_dir / "README_final_ver.md").write_text(markdown_table(summary), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
