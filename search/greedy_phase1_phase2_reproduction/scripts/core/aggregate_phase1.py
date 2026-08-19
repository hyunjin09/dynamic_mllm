#!/usr/bin/env python3
"""Merge completed Phase-1 sample files without constructing rankings."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "10k_dataset_mask" / "raw" / "search"
DEFAULT_OUTPUT = ROOT / "10k_dataset_mask" / "phase1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    paths = sorted(args.input_dir.glob("shard_*_of_*/samples/*.json"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_count = 0
    trace_count = 0
    final_count = 0
    sample_count = 0
    seen_uids: set[str] = set()
    sample_success_budgets: dict[tuple[str, str], list[float]] = defaultdict(list)
    source_reproduction = Counter()

    with (
        (args.output_dir / "candidate_executions.jsonl").open("w", encoding="utf-8") as candidate_handle,
        (args.output_dir / "search_trace.jsonl").open("w", encoding="utf-8") as trace_handle,
        (args.output_dir / "permutation_finals.jsonl").open("w", encoding="utf-8") as final_handle,
        (args.output_dir / "sample_summary.jsonl").open("w", encoding="utf-8") as sample_handle,
    ):
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            sample = payload["sample"]
            uid = str(sample["uid"])
            if uid in seen_uids:
                raise RuntimeError(f"duplicate completed sample: {uid}")
            seen_uids.add(uid)

            for row in payload["candidate_executions"]:
                output = {"uid": uid, "benchmark": sample["benchmark"], "data_split": sample["data_split"], **row}
                candidate_handle.write(json.dumps(output, ensure_ascii=True, sort_keys=True) + "\n")
                candidate_count += 1
            for row in payload["search_trace"]:
                output = {"benchmark": sample["benchmark"], "data_split": sample["data_split"], **row}
                trace_handle.write(json.dumps(output, ensure_ascii=True, sort_keys=True) + "\n")
                trace_count += 1
            for row in payload["permutation_finals"]:
                output = {"uid": uid, "benchmark": sample["benchmark"], "data_split": sample["data_split"], **row}
                final_handle.write(json.dumps(output, ensure_ascii=True, sort_keys=True) + "\n")
                final_count += 1

            unique_success = {
                tuple(row["final_mask_one_based"]): int(row["final_num_visual_on_layers"])
                for row in payload["permutation_finals"]
                if row["final_correct"]
            }
            if unique_success:
                mean_budget = statistics.fmean(unique_success.values())
                sample_success_budgets[(sample["data_split"], sample["benchmark"])].append(mean_budget)
                sample_success_budgets[("all", sample["benchmark"])].append(mean_budget)
            reproduction = payload["source_anchor_reproduced_by_binary"]
            source_reproduction["samples"] += 1
            source_reproduction["prediction_match"] += bool(reproduction["prediction_match"])
            source_reproduction["score_match"] += bool(reproduction["score_match"])
            sample_row = {
                "uid": uid,
                "sample_id": sample["sample_id"],
                "benchmark": sample["benchmark"],
                "data_split": sample["data_split"],
                "source_bucket": sample["source_bucket"],
                "candidate_execution_count": len(payload["candidate_executions"]),
                "search_trace_count": len(payload["search_trace"]),
                "successful_order_count": sum(row["final_correct"] for row in payload["permutation_finals"]),
                "unique_success_mask_count": len(unique_success),
                "mean_unique_success_budget": statistics.fmean(unique_success.values()) if unique_success else None,
                "source_prediction_reproduced": bool(reproduction["prediction_match"]),
                "source_score_reproduced": bool(reproduction["score_match"]),
                "phase1_sample_file": str(path.relative_to(ROOT)),
            }
            sample_handle.write(json.dumps(sample_row, ensure_ascii=True, sort_keys=True) + "\n")
            sample_count += 1

    budget_rows = []
    budget_lookup: dict[str, Any] = {}
    for (split, benchmark), values in sorted(sample_success_budgets.items()):
        row = {
            "data_split": split,
            "benchmark": benchmark,
            "samples_with_success": len(values),
            "mean_sample_success_budget": statistics.fmean(values),
            "median_sample_success_budget": statistics.median(values),
            "q25_sample_success_budget": quantile(values, 0.25),
            "q75_sample_success_budget": quantile(values, 0.75),
            "rounded_budget_center": int(round(statistics.fmean(values))),
        }
        budget_rows.append(row)
        budget_lookup[f"{split}/{benchmark}"] = row

    write_jsonl(args.output_dir / "benchmark_budget_statistics.jsonl", budget_rows)
    write_json(args.output_dir / "benchmark_budget_statistics.json", budget_lookup)
    write_json(
        args.output_dir / "summary.json",
        {
            "completed_samples": sample_count,
            "candidate_executions": candidate_count,
            "search_trace_rows": trace_count,
            "permutation_final_rows": final_count,
            "source_anchor_prediction_matches": source_reproduction["prediction_match"],
            "source_anchor_score_matches": source_reproduction["score_match"],
            "ranking_constructed": False,
            "preference_pairs_constructed": False,
        },
    )
    print(json.dumps({"completed_samples": sample_count, "candidate_executions": candidate_count}, sort_keys=True))


if __name__ == "__main__":
    main()
