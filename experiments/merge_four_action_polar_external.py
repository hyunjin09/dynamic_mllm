#!/usr/bin/env python3
"""Integrity-check, merge, and summarize one four-action external evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments.train_binary_polar import file_sha256
from four_action_policy.actions import FOUR_ACTIONS
from four_action_policy.external import ACTIVE_BENCHMARKS, EXPECTED_COUNTS, TOTAL_RECORDS


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def clustered_mean_ci(
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    *,
    draws: int,
    seed: int,
) -> list[float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row["cluster_key"])].append(float(value(row)))
    keys = sorted(groups)
    sums = np.asarray([sum(groups[key]) for key in keys], dtype=np.float64)
    counts = np.asarray([len(groups[key]) for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 128):
        stop = min(start + 128, draws)
        indices = rng.integers(0, len(keys), size=(stop - start, len(keys)))
        estimates[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def summarize_rows(
    rows: list[dict[str, Any]], *, bootstrap_draws: int, seed: int
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty population")
    predicted = [row["predicted"] for row in rows]
    correct_delta = [
        float(item["correct"]) - float(row["baseline_correct"])
        for row, item in zip(rows, predicted)
    ]
    score_delta = [
        float(item["score"]) - float(row["baseline_score"])
        for row, item in zip(rows, predicted)
    ]
    route_counts = Counter(str(item["route_key"]) for item in predicted)
    action_totals = Counter()
    for item in predicted:
        action_totals.update(
            {action: int(item["action_counts"][action]) for action in FOUR_ACTIONS}
        )
    return {
        "records": len(rows),
        "cluster_count": len({str(row["cluster_key"]) for row in rows}),
        "baseline_correct_rate": statistics.fmean(float(row["baseline_correct"]) for row in rows),
        "predicted_correct_rate": statistics.fmean(float(item["correct"]) for item in predicted),
        "correctness_delta": statistics.fmean(correct_delta),
        "correctness_delta_clustered_95ci": clustered_mean_ci(
            rows,
            lambda row: float(row["predicted"]["correct"])
            - float(row["baseline_correct"]),
            draws=bootstrap_draws,
            seed=seed,
        ),
        "baseline_mean_score": statistics.fmean(float(row["baseline_score"]) for row in rows),
        "predicted_mean_score": statistics.fmean(float(item["score"]) for item in predicted),
        "score_delta": statistics.fmean(score_delta),
        "score_delta_clustered_95ci": clustered_mean_ci(
            rows,
            lambda row: float(row["predicted"]["score"])
            - float(row["baseline_score"]),
            draws=bootstrap_draws,
            seed=seed + 1,
        ),
        "full_wrong_to_predicted_correct": sum(
            (not row["baseline_correct"]) and item["correct"]
            for row, item in zip(rows, predicted)
        ),
        "full_correct_to_predicted_wrong": sum(
            row["baseline_correct"] and (not item["correct"])
            for row, item in zip(rows, predicted)
        ),
        "unchanged_correct": sum(
            row["baseline_correct"] and item["correct"]
            for row, item in zip(rows, predicted)
        ),
        "unchanged_wrong": sum(
            (not row["baseline_correct"]) and (not item["correct"])
            for row, item in zip(rows, predicted)
        ),
        "behavior_changing_executions": sum(
            item["generated_ids"] != row["baseline_generated_ids"]
            for row, item in zip(rows, predicted)
        ),
        "mean_action_layers": {
            action: action_totals[action] / len(rows) for action in FOUR_ACTIONS
        },
        "mean_non_full_layers": statistics.fmean(
            int(item["non_full_layers"]) for item in predicted
        ),
        "mean_route_transitions": statistics.fmean(
            int(item["transition_count"]) for item in predicted
        ),
        "unique_predicted_routes": len(route_counts),
        "all_full_rate": route_counts.get("|".join(["FULL"] * 28), 0) / len(rows),
        "all_ignore_rate": route_counts.get("|".join(["IGNORE"] * 28), 0) / len(rows),
        "top_predicted_routes": [
            {"route": route, "count": count, "fraction": count / len(rows)}
            for route, count in route_counts.most_common(10)
        ],
    }


def verify_and_load(
    root: Path, num_shards: int, *, preflight_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("passed") is not True:
        raise RuntimeError("external preflight is absent or failed")
    rows = []
    metadata = []
    for index in range(num_shards):
        shard = root / f"shard_{index:03d}_of_{num_shards:03d}"
        current_meta = json.loads((shard / "metadata.json").read_text(encoding="utf-8"))
        if int(current_meta["shard_index"]) != index or int(current_meta["num_shards"]) != num_shards:
            raise RuntimeError(f"invalid shard metadata: {shard}")
        parts = sorted(shard.glob("part_*.jsonl"))
        current = [row for path in parts for row in read_jsonl(path)]
        if len(current) != int(current_meta["records"]):
            raise RuntimeError(f"row count mismatch in {shard}")
        if current_meta.get("preflight_sha256") != file_sha256(preflight_path):
            raise RuntimeError(f"shard was not bound to the selected preflight: {shard}")
        rows.extend(current)
        metadata.append(current_meta)
    for key in (
        "config_sha256",
        "selection_sha256",
        "checkpoint_sha256",
        "preflight_sha256",
        "source_sha256",
    ):
        if len({str(item[key]) for item in metadata}) != 1:
            raise RuntimeError(f"shards disagree on {key}")
    uids = [str(row["uid"]) for row in rows]
    if len(rows) != TOTAL_RECORDS or len(uids) != len(set(uids)):
        raise RuntimeError(
            f"expected {TOTAL_RECORDS:,} unique outputs, found {len(rows)}/{len(set(uids))}"
        )
    counts = Counter(str(row["benchmark"]) for row in rows)
    if counts != Counter(EXPECTED_COUNTS):
        raise RuntimeError(f"benchmark counts differ from the frozen population: {counts}")
    if any("predicted" not in row for row in rows):
        raise RuntimeError("predicted execution is absent from one or more rows")
    return sorted(rows, key=lambda row: str(row["uid"])), metadata


def markdown_table(summary: dict[str, Any], groups: list[str]) -> str:
    lines = [
        "| Population | N | Unified FULL correct | Predicted correct | Delta (95% image-cluster CI) | W→C | C→W | IGNORE | READ_ONLY | WRITE_ONLY | FULL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        item = summary["groups"][group]
        low, high = item["correctness_delta_clustered_95ci"]
        actions = item["mean_action_layers"]
        lines.append(
            f"| {group} | {item['records']:,} | {item['baseline_correct_rate']:.4f} | "
            f"{item['predicted_correct_rate']:.4f} | {item['correctness_delta']:+.4f} "
            f"[{low:+.4f}, {high:+.4f}] | {item['full_wrong_to_predicted_correct']} | "
            f"{item['full_correct_to_predicted_wrong']} | {actions['IGNORE']:.2f} | "
            f"{actions['READ_ONLY']:.2f} | {actions['WRITE_ONLY']:.2f} | {actions['FULL']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--title", default="Four-Action POLAR External Evaluation")
    args = parser.parse_args()

    rows, metadata = verify_and_load(
        args.root, args.num_shards, preflight_path=args.preflight
    )
    groups = {
        benchmark: [row for row in rows if row["benchmark"] == benchmark]
        for benchmark in ACTIVE_BENCHMARKS
    }
    groups["mmmu_pro"] = [row for row in rows if row["suite"] == "mmmu_pro"]
    groups["pope"] = [row for row in rows if row["suite"] == "pope"]
    summary = {
        "schema_version": "four_action_polar_external_analysis_v1",
        "records": len(rows),
        "no_cross_suite_pooling": True,
        "baseline": "current_live_unified_full",
        "bootstrap": {
            "method": "image-cluster resampling with replacement; row-weighted mean",
            "draws": args.bootstrap_draws,
            "confidence": 0.95,
            "seed": args.seed,
        },
        "groups": {},
    }
    for index, (name, current) in enumerate(groups.items()):
        summary["groups"][name] = summarize_rows(
            current, bootstrap_draws=args.bootstrap_draws, seed=args.seed + 2 * index
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    results_path = args.output_root / "external_results_v1.jsonl"
    analysis_path = args.output_root / "external_analysis_v1.json"
    write_jsonl(results_path, rows)
    write_json(analysis_path, summary)
    write_json(
        args.output_root / "analysis_manifest_v1.json",
        {
            "schema_version": "four_action_polar_external_analysis_manifest_v1",
            "integrity_status": "PASS",
            "preflight_sha256": file_sha256(args.preflight),
            "results": {"path": str(results_path), "sha256": file_sha256(results_path)},
            "analysis": {"path": str(analysis_path), "sha256": file_sha256(analysis_path)},
            "shard_metadata": metadata,
            "records": len(rows),
        },
    )
    order = list(ACTIVE_BENCHMARKS) + ["mmmu_pro", "pope"]
    report = [
        f"# {args.title}",
        "",
        "## Integrity",
        "",
        f"- PASS: all {len(rows):,} prospectively selected UIDs completed exactly once.",
        "- The checkpoint was selected using internal four-action validation before any external outcomes were evaluated.",
        "- The scientific baseline is current live unified FULL; no imported historical output cache enters the comparison.",
        "- ChartQA, MMMU-Pro, and POPE are not pooled into a cross-suite metric.",
        "",
        "## Results",
        "",
        markdown_table(summary, order),
        "",
        "## Interpretation boundary",
        "",
        "These are deterministic complete four-action route executions from an Image+Question factorized predictor. The layer-action counts describe selected routes; they are not measured latency or memory savings.",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report), encoding="utf-8")
    args.report.with_suffix(args.report.suffix + ".sha256").write_text(
        f"{file_sha256(args.report)}  {args.report.name}\n", encoding="utf-8"
    )
    print(json.dumps({"records": len(rows), "report": str(args.report)}))


if __name__ == "__main__":
    main()
