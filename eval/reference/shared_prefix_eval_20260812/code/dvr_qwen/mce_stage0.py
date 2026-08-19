"""Stage-0 sanity helpers for the minimum counterfactual evidence phase."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dvr_qwen.eval_metrics import score_prediction
from dvr_qwen.mce_inventory import BENCHMARKS, iter_jsonl


PRIMARY_5K_ROWS = Path(
    "/home/aix7101/hyemin/dynamic_mllm/mnt/dvr_qwen/binary_oracle_search/"
    "binary_oracle_5k_float32_lmms_metric_20260613/binary_oracle_search_rows.jsonl"
)
FAILURE_NO_OLD_GT_ROWS = Path(
    "/home/aix7101/hyemin/dynamic_mllm/mnt/dvr_qwen/binary_oracle_search/"
    "no_old_gt_available_failures_20260610_aggregate/binary_oracle_search_rows.jsonl"
)
FAILURE_TCD_ROWS = Path(
    "/home/aix7101/hyemin/dynamic_mllm/mnt/dvr_qwen/binary_oracle_search/"
    "binary_oracle_lmms_metric_wrong_tcd_20260612/binary_oracle_search_rows.jsonl"
)

SOURCE_POOL_TO_ROWS = {
    "primary_5k": PRIMARY_5K_ROWS,
    "primary_5k_router": PRIMARY_5K_ROWS,
    "failure_no_old_gt": FAILURE_NO_OLD_GT_ROWS,
    "failure_tcd_lmms": FAILURE_TCD_ROWS,
}


def close(left: float, right: float, *, atol: float = 1e-9) -> bool:
    return abs(float(left) - float(right)) <= atol


def load_source_indexes(paths_by_pool: dict[str, Path] | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    paths = paths_by_pool or SOURCE_POOL_TO_ROWS
    by_path = {path: {} for path in set(paths.values())}
    for path in by_path:
        for row in iter_jsonl(path):
            by_path[path][str(row["sample_id"])] = row
    return {pool: by_path[path] for pool, path in paths.items()}


def source_row_for_manifest(
    manifest_row: dict[str, Any],
    source_indexes: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    source_pool = str(manifest_row["source_pool"])
    sample_id = str(manifest_row["sample_id"])
    if source_pool not in source_indexes:
        raise KeyError(f"no source index for source_pool={source_pool!r}")
    try:
        return source_indexes[source_pool][sample_id]
    except KeyError as exc:
        raise KeyError(f"missing source row for {source_pool}:{sample_id}") from exc


def build_artifact_record(
    manifest_row: dict[str, Any],
    source_row: dict[str, Any],
    *,
    score_atol: float = 1e-9,
) -> dict[str, Any]:
    full = source_row["full_qwen"]
    all_visual = source_row["binary_all_visual_on"]
    metric_name = source_row["metric_name"]
    answer = source_row["answer"]
    all_answer_norms = source_row.get("all_answer_norms")
    full_score_recomputed = score_prediction(
        metric_name,
        full["prediction"],
        answer,
        all_answer_norms,
    )
    all_visual_score_recomputed = score_prediction(
        metric_name,
        all_visual["prediction"],
        answer,
        all_answer_norms,
    )
    full_score = float(full["score"])
    all_visual_score = float(all_visual["score"])
    return {
        "sample_id": manifest_row["sample_id"],
        "dataset": manifest_row["dataset"],
        "cohort": manifest_row["cohort"],
        "source_pool": manifest_row["source_pool"],
        "metric_name": metric_name,
        "answer": answer,
        "full_prediction": full["prediction"],
        "all_visual_prediction": all_visual["prediction"],
        "full_score": full_score,
        "all_visual_score": all_visual_score,
        "manifest_baseline_score": float(manifest_row["baseline_score"]),
        "full_score_recomputed": full_score_recomputed,
        "all_visual_score_recomputed": all_visual_score_recomputed,
        "full_vs_all_visual_generated_ids_match": full["generated_ids"] == all_visual["generated_ids"],
        "full_vs_all_visual_prediction_match": full["prediction"] == all_visual["prediction"],
        "full_vs_all_visual_score_match": close(full_score, all_visual_score, atol=score_atol),
        "full_score_recomputed_match": close(full_score, full_score_recomputed, atol=score_atol),
        "all_visual_score_recomputed_match": close(all_visual_score, all_visual_score_recomputed, atol=score_atol),
        "full_generated_ids": full["generated_ids"],
        "all_visual_generated_ids": all_visual["generated_ids"],
    }


def build_artifact_records(
    manifest_rows: list[dict[str, Any]],
    source_indexes: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    records = []
    for manifest_row in manifest_rows:
        records.append(build_artifact_record(manifest_row, source_row_for_manifest(manifest_row, source_indexes)))
    return records


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_artifact_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts_by_dataset: dict[str, Counter[str]] = {benchmark: Counter() for benchmark in BENCHMARKS}
    source_pool_counts = Counter()
    for row in records:
        counts_by_dataset[str(row["dataset"])][str(row["cohort"])] += 1
        source_pool_counts[str(row["source_pool"])] += 1

    generated_id_mismatches = [row for row in records if not row["full_vs_all_visual_generated_ids_match"]]
    prediction_mismatches = [row for row in records if not row["full_vs_all_visual_prediction_match"]]
    score_mismatches = [row for row in records if not row["full_vs_all_visual_score_match"]]
    scorer_mismatches = [
        row
        for row in records
        if not row["full_score_recomputed_match"] or not row["all_visual_score_recomputed_match"]
    ]
    per_dataset_scores = defaultdict(lambda: {"full": [], "all_visual": []})
    for row in records:
        per_dataset_scores[str(row["dataset"])]["full"].append(float(row["full_score"]))
        per_dataset_scores[str(row["dataset"])]["all_visual"].append(float(row["all_visual_score"]))

    return {
        "row_count": len(records),
        "counts_by_dataset": {benchmark: dict(counts_by_dataset[benchmark]) for benchmark in BENCHMARKS},
        "source_pool_counts": dict(source_pool_counts),
        "full_qwen_score": mean([float(row["full_score"]) for row in records]),
        "all_visual_on_score": mean([float(row["all_visual_score"]) for row in records]),
        "per_dataset_scores": {
            dataset: {
                "full_qwen_score": mean(values["full"]),
                "all_visual_on_score": mean(values["all_visual"]),
            }
            for dataset, values in sorted(per_dataset_scores.items())
        },
        "generated_id_mismatches": len(generated_id_mismatches),
        "prediction_mismatches": len(prediction_mismatches),
        "score_mismatches": len(score_mismatches),
        "scorer_mismatches": len(scorer_mismatches),
        "generated_id_mismatch_ids": [row["sample_id"] for row in generated_id_mismatches],
        "prediction_mismatch_ids": [row["sample_id"] for row in prediction_mismatches],
        "score_mismatch_ids": [row["sample_id"] for row in score_mismatches],
        "scorer_mismatch_ids": [row["sample_id"] for row in scorer_mismatches],
    }


def metric_smoke_results() -> list[dict[str, Any]]:
    cases = [
        {
            "benchmark": "gqa",
            "metric_name": "exact_match",
            "prediction": "Gold.",
            "answer": "gold",
            "answers": None,
            "expected": 1.0,
        },
        {
            "benchmark": "textvqa",
            "metric_name": "textvqa_consensus",
            "prediction": "perfection",
            "answer": "perfection feed for all stock",
            "answers": ["perfection"] * 4 + ["feed"] * 6,
            "expected": 1.0,
        },
        {
            "benchmark": "chartqa",
            "metric_name": "relaxed_accuracy",
            "prediction": "105",
            "answer": "100",
            "answers": None,
            "expected": 1.0,
        },
        {
            "benchmark": "docvqa",
            "metric_name": "anls",
            "prediction": "invoice total",
            "answer": "invoice total",
            "answers": ["invoice total"],
            "expected": 1.0,
        },
    ]
    results = []
    for case in cases:
        score = score_prediction(
            case["metric_name"],
            case["prediction"],
            case["answer"],
            case["answers"],
        )
        results.append(
            {
                "benchmark": case["benchmark"],
                "metric_name": case["metric_name"],
                "prediction": case["prediction"],
                "answer": case["answer"],
                "score": score,
                "expected": case["expected"],
                "passed": close(score, case["expected"], atol=1e-9),
            }
        )
    return results
