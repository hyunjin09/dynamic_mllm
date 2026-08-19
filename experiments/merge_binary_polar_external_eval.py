#!/usr/bin/env python3
"""Validate, merge, and summarize sharded full10 external evaluations."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments.evaluate_binary_polar_external import (
    ACTIVE_BENCHMARKS,
    file_sha256,
    read_jsonl,
    write_json,
    write_jsonl,
)


EXPECTED_COUNTS = {
    "chartqa": 2500,
    "textvqa": 5000,
    "mmstar_val": 1500,
    "mmmu_val": 847,
    "mmmu_pro_standard_test": 1730,
    "mmmu_pro_vision_test": 1730,
    "pope_adversarial": 3000,
    "pope_popular": 3000,
    "pope_random": 3000,
}


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
    block = 128
    for start in range(0, draws, block):
        stop = min(start + block, draws)
        indices = rng.integers(0, len(keys), size=(stop - start, len(keys)))
        estimates[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def summarize_rows(
    rows: list[dict[str, Any]],
    modality: str,
    *,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty population")
    predicted = [row[modality] for row in rows]
    correct_delta = [float(item["correct"]) - float(row["baseline_correct"]) for row, item in zip(rows, predicted)]
    score_delta = [float(item["score"]) - float(row["baseline_score"]) for row, item in zip(rows, predicted)]
    on_counts = [int(item["num_visual_on_layers"]) for item in predicted]
    mask_counts = Counter(str(item["mask_key"]) for item in predicted)
    return {
        "records": len(rows),
        "cluster_count": len({str(row["cluster_key"]) for row in rows}),
        "baseline_correct_rate": statistics.fmean(float(row["baseline_correct"]) for row in rows),
        "predicted_correct_rate": statistics.fmean(float(item["correct"]) for item in predicted),
        "correctness_delta": statistics.fmean(correct_delta),
        "correctness_delta_clustered_95ci": clustered_mean_ci(
            rows,
            lambda row: float(row[modality]["correct"]) - float(row["baseline_correct"]),
            draws=bootstrap_draws,
            seed=seed,
        ),
        "baseline_mean_score": statistics.fmean(float(row["baseline_score"]) for row in rows),
        "predicted_mean_score": statistics.fmean(float(item["score"]) for item in predicted),
        "score_delta": statistics.fmean(score_delta),
        "score_delta_clustered_95ci": clustered_mean_ci(
            rows,
            lambda row: float(row[modality]["score"]) - float(row["baseline_score"]),
            draws=bootstrap_draws,
            seed=seed + 1,
        ),
        "full_wrong_to_predicted_correct": sum(
            (not row["baseline_correct"]) and item["correct"] for row, item in zip(rows, predicted)
        ),
        "full_correct_to_predicted_wrong": sum(
            row["baseline_correct"] and (not item["correct"]) for row, item in zip(rows, predicted)
        ),
        "unchanged_correct": sum(row["baseline_correct"] and item["correct"] for row, item in zip(rows, predicted)),
        "unchanged_wrong": sum((not row["baseline_correct"]) and (not item["correct"]) for row, item in zip(rows, predicted)),
        "mean_visual_on_layers": statistics.fmean(on_counts),
        "median_visual_on_layers": statistics.median(on_counts),
        "mean_visual_on_reduction_from_full": 28.0 - statistics.fmean(on_counts),
        "mean_visual_on_fraction_reduction_from_full": (28.0 - statistics.fmean(on_counts)) / 28.0,
        "visual_on_count_histogram": dict(sorted(Counter(on_counts).items())),
        "mean_mask_transitions": statistics.fmean(int(item["transition_count"]) for item in predicted),
        "unique_predicted_masks": len(mask_counts),
        "all_on_rate": mask_counts.get("1" * 28, 0) / len(rows),
        "top_predicted_masks": [
            {"mask": mask, "count": count, "fraction": count / len(rows)}
            for mask, count in mask_counts.most_common(10)
        ],
    }


def verify_and_load(
    root: Path,
    num_shards: int,
    *,
    modality: str,
    preflight_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight.get("passed"):
        raise RuntimeError("external preflight is absent or failed")
    rows: list[dict[str, Any]] = []
    metadata = []
    for index in range(num_shards):
        shard = root / f"shard_{index:03d}_of_{num_shards:03d}"
        current_meta = json.loads((shard / "metadata.json").read_text(encoding="utf-8"))
        if int(current_meta["shard_index"]) != index or int(current_meta["num_shards"]) != num_shards:
            raise RuntimeError(f"invalid shard metadata: {shard}")
        if current_meta.get("modalities") != [modality]:
            raise RuntimeError(f"unexpected modality in shard metadata: {shard}")
        parts = sorted(shard.glob("part_*.jsonl"))
        current = [row for path in parts for row in read_jsonl(path)]
        if len(current) != int(current_meta["records"]):
            raise RuntimeError(f"row count mismatch in {shard}")
        rows.extend(current)
        metadata.append(current_meta)
    invariant_keys = (
        "config_sha256",
        "question_checkpoint_sha256",
        "image_question_checkpoint_sha256",
        "source_sha256",
    )
    for key in invariant_keys:
        if len({str(item[key]) for item in metadata}) != 1:
            raise RuntimeError(f"shards disagree on {key}")
    uids = [str(row["uid"]) for row in rows]
    if len(rows) != 22307 or len(set(uids)) != len(rows):
        raise RuntimeError(f"expected 22,307 unique outputs, found {len(rows)}/{len(set(uids))}")
    counts = Counter(str(row["benchmark"]) for row in rows)
    if counts != Counter(EXPECTED_COUNTS):
        raise RuntimeError(f"benchmark counts do not match frozen population: {counts}")
    if any(modality not in row for row in rows):
        raise RuntimeError(f"{modality} output is absent from one or more rows")
    return sorted(rows, key=lambda row: str(row["uid"])), metadata


def markdown_table(summary: dict[str, Any], modality: str, groups: list[str]) -> str:
    lines = [
        f"### {modality.replace('_', ' ').title()}",
        "",
        "| Population | N | FULL correct | Predicted correct | Delta (95% clustered CI) | W→C | C→W | Mean ON | ON reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        item = summary[modality][group]
        low, high = item["correctness_delta_clustered_95ci"]
        lines.append(
            f"| {group} | {item['records']:,} | {item['baseline_correct_rate']:.4f} | "
            f"{item['predicted_correct_rate']:.4f} | {item['correctness_delta']:+.4f} "
            f"[{low:+.4f}, {high:+.4f}] | {item['full_wrong_to_predicted_correct']} | "
            f"{item['full_correct_to_predicted_wrong']} | {item['mean_visual_on_layers']:.2f} | "
            f"{item['mean_visual_on_fraction_reduction_from_full']:.2%} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-root", type=Path, required=True)
    parser.add_argument("--image-question-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-path", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--pope-overlap-cluster", action="append", default=[])
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--title", default="Full10 Best-Checkpoint External Evaluation")
    args = parser.parse_args()

    question_rows, question_metadata = verify_and_load(
        args.question_root,
        args.num_shards,
        modality="question",
        preflight_path=args.preflight_path,
    )
    image_rows, image_metadata = verify_and_load(
        args.image_question_root,
        args.num_shards,
        modality="image_question",
        preflight_path=args.preflight_path,
    )
    image_by_uid = {str(row["uid"]): row for row in image_rows}
    rows = []
    common_fields = (
        "uid",
        "sample_id",
        "benchmark",
        "suite",
        "cluster_key",
        "metric_name",
        "correctness_threshold",
        "baseline_prediction",
        "baseline_score",
        "baseline_correct",
        "baseline_generated_ids",
        "reference_cache_prediction",
        "reference_cache_score",
        "reference_cache_correct",
        "reference_cache_exact_match",
        "predictor_text_sha256",
        "visual_tokens",
    )
    for question_row in question_rows:
        image_row = image_by_uid.get(str(question_row["uid"]))
        if image_row is None:
            raise RuntimeError(f"image-question output is missing UID {question_row['uid']}")
        if any(question_row[field] != image_row[field] for field in common_fields):
            raise RuntimeError(f"modality outputs disagree on common fields for {question_row['uid']}")
        merged = {
            **question_row,
            "baseline_source": "current_live_all_on",
            "baseline_source_by_modality": {
                "question": question_row["baseline_source"],
                "image_question": image_row["baseline_source"],
            },
            "image_question": image_row["image_question"],
        }
        rows.append(merged)
    metadata = {
        "question": question_metadata,
        "image_question": image_metadata,
    }
    groups: dict[str, list[dict[str, Any]]] = {
        benchmark: [row for row in rows if row["benchmark"] == benchmark]
        for benchmark in ACTIVE_BENCHMARKS
    }
    groups.update(
        {
            "core_vqa": [row for row in rows if row["suite"] == "core_vqa"],
            "external_multiple_choice": [row for row in rows if row["suite"] == "external_multiple_choice"],
            "pope": [row for row in rows if row["suite"] == "pope"],
            "pope_image_disjoint": [
                row
                for row in rows
                if row["suite"] == "pope" and row["cluster_key"] not in set(args.pope_overlap_cluster)
            ],
        }
    )
    if args.pope_overlap_cluster and len(groups["pope_image_disjoint"]) != 8982:
        raise RuntimeError("frozen POPE image-disjoint sensitivity must contain 8,982 records")
    summary: dict[str, Any] = {
        "schema_version": "binary_polar_external_analysis_v1",
        "records": len(rows),
        "bootstrap": {
            "method": "image-cluster resampling with replacement; row-weighted mean",
            "draws": args.bootstrap_draws,
            "confidence": 0.95,
            "seed": args.seed,
        },
        "pope_overlap_clusters": sorted(args.pope_overlap_cluster),
    }
    for modality_index, modality in enumerate(("question", "image_question")):
        summary[modality] = {}
        for group_index, (name, current) in enumerate(groups.items()):
            summary[modality][name] = summarize_rows(
                current,
                modality,
                bootstrap_draws=args.bootstrap_draws,
                seed=args.seed + 1000 * modality_index + 2 * group_index,
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    merged_path = args.output_root / "external_results_v1.jsonl"
    summary_path = args.output_root / "external_analysis_v1.json"
    write_jsonl(merged_path, rows)
    write_json(summary_path, summary)
    manifest = {
        "schema_version": "binary_polar_external_analysis_manifest_v1",
        "integrity_status": "PASS",
        "preflight_sha256": file_sha256(args.preflight_path),
        "merged_results": {"path": str(merged_path), "sha256": file_sha256(merged_path)},
        "analysis": {"path": str(summary_path), "sha256": file_sha256(summary_path)},
        "shard_metadata": metadata,
        "records": len(rows),
    }
    write_json(args.output_root / "analysis_manifest_v1.json", manifest)

    order = list(ACTIVE_BENCHMARKS) + [
        "core_vqa",
        "external_multiple_choice",
        "pope",
        "pope_image_disjoint",
    ]
    report = [
        f"# {args.title}",
        "",
        "## Integrity",
        "",
        f"- Status: **PASS**; all {len(rows):,} frozen active UIDs completed exactly once.",
        "- DocVQA was excluded prospectively. Core VQA, multiple choice, and POPE are not pooled into one overall metric.",
        "- The checkpoints were selected on internal validation before these external outcomes were evaluated.",
        "- The scientific baseline is current live ALL-ON execution. The historical bundle cache is audit-only because 485/22,307 predictions/scores/correctness tuples differed under the current runtime, including 168 correctness labels.",
        "- Reported compute is visual-ON decoder-layer count, not measured wall-clock acceleration.",
        "",
        "## Route behavior",
        "",
        *[
            f"- {modality.replace('_', ' ').title()} selected ALL-ON for "
            f"{sum(row[modality]['mask_key'] == '1' * 28 for row in rows):,}/{len(rows):,} "
            f"records and a non-ALL-ON mask for "
            f"{sum(row[modality]['mask_key'] != '1' * 28 for row in rows):,}. "
            f"The predicted execution changed prediction, score, or correctness relative to current live "
            f"ALL-ON on {sum((row[modality]['prediction'], row[modality]['score'], row[modality]['correct']) != (row['baseline_prediction'], row['baseline_score'], row['baseline_correct']) for row in rows):,} records."
            for modality in ("question", "image_question")
        ],
        "",
        markdown_table(summary, "question", order),
        "",
        markdown_table(summary, "image_question", order),
        "",
        "## Interpretation boundary",
        "",
        "These are deterministic static-mask executions of validation-selected factorized predictors. External correctness changes and visual-ON counts describe their behavior; they do not by themselves establish deployable latency gains or causal routing mechanisms.",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report), encoding="utf-8")
    (args.report.with_suffix(args.report.suffix + ".sha256")).write_text(
        f"{file_sha256(args.report)}  {args.report.name}\n", encoding="utf-8"
    )
    print(json.dumps({"records": len(rows), "report": str(args.report)}))


if __name__ == "__main__":
    main()
