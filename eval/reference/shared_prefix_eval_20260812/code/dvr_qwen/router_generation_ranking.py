"""Comparable ranking utilities for full validation-generation router runs."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable


BENCHMARKS = ("gqa", "chartqa", "docvqa", "textvqa")


def _mean(summary: dict[str, Any], group: str, metric: str) -> float:
    return float(summary[group][metric]["mean"])


def _mask_diversity(payload: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    embedded = payload["summary"]["all"].get("mask_diversity")
    if isinstance(embedded, dict):
        return embedded
    rows_value = payload.get("outputs", {}).get("rows_jsonl")
    rows_path = Path(rows_value) if rows_value else summary_path.parent / "online_generation_rows.jsonl"
    if not rows_path.exists():
        return {"unique_masks": None, "modal_mask_fraction": None}
    counts: Counter[str] = Counter()
    total = 0
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("selected_mask_key")
            if key is None and row.get("selected_visual_on_mask") is not None:
                key = "".join(str(int(value)) for value in row["selected_visual_on_mask"])
            if key is not None:
                counts[str(key)] += 1
            total += 1
    modal_count = counts.most_common(1)[0][1] if counts else 0
    return {
        "unique_masks": len(counts),
        "modal_mask_fraction": modal_count / total if total else None,
    }


def candidate_from_summary(summary_path: Path, *, required_samples: int = 1045) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = payload.get("summary") or {}
    if not all(group in summary for group in ("all", "complete_correct", "complete_wrong", *BENCHMARKS)):
        raise ValueError(f"incomplete comparable groups: {summary_path}")
    sample_count = int(summary["all"]["samples"])
    if sample_count != int(required_samples):
        raise ValueError(
            f"validation sample count mismatch for {summary_path}: {sample_count} != {required_samples}"
        )
    checkpoint = Path(str(payload["checkpoint"]))
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint recorded by {summary_path} does not exist: {checkpoint}")
    diversity = _mask_diversity(payload, summary_path)
    benchmark_correct = [_mean(summary, benchmark, "online_correct_rate") for benchmark in BENCHMARKS]
    return {
        "eval_run_id": summary_path.parent.name,
        "summary_path": str(summary_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "router_threshold": float(payload.get("checkpoint_runtime", {}).get("router_threshold", 0.0)),
        "dataset_dir": payload.get("dataset_dir"),
        "samples": sample_count,
        "overall_correct": _mean(summary, "all", "online_correct_rate"),
        "overall_score": _mean(summary, "all", "online_score"),
        "source_full_correct": _mean(summary, "all", "source_full_correct_rate"),
        "preservation_correct": _mean(summary, "complete_correct", "online_correct_rate"),
        "rescue_correct": _mean(summary, "complete_wrong", "online_correct_rate"),
        "macro_benchmark_correct": sum(benchmark_correct) / len(benchmark_correct),
        "avg_selected_layers": _mean(summary, "all", "avg_selected_layers"),
        "unique_masks": diversity.get("unique_masks"),
        "modal_mask_fraction": diversity.get("modal_mask_fraction"),
        "benchmark_correct": dict(zip(BENCHMARKS, benchmark_correct)),
    }


def rank_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    min_preservation: float = 0.95,
    max_baseline_drop: float = 0.005,
    accuracy_tolerance: float = 0.01,
) -> list[dict[str, Any]]:
    rows = [dict(candidate) for candidate in candidates]
    if not rows:
        return []
    if accuracy_tolerance <= 0.0:
        raise ValueError("accuracy_tolerance must be positive")
    for row in rows:
        preservation_shortfall = max(0.0, min_preservation - float(row["preservation_correct"]))
        accuracy_floor = float(row["source_full_correct"]) - max_baseline_drop
        accuracy_shortfall = max(0.0, accuracy_floor - float(row["overall_correct"]))
        row["preservation_shortfall"] = preservation_shortfall
        row["baseline_accuracy_shortfall"] = accuracy_shortfall
        row["safety_violation"] = preservation_shortfall + accuracy_shortfall
        row["safety_pass"] = preservation_shortfall == 0.0 and accuracy_shortfall == 0.0

    safe_rows = [row for row in rows if bool(row["safety_pass"])]
    best_safe_accuracy = max((float(row["overall_correct"]) for row in safe_rows), default=None)
    for row in rows:
        if best_safe_accuracy is None or not bool(row["safety_pass"]):
            row["accuracy_band"] = None
        else:
            deficit = max(0.0, best_safe_accuracy - float(row["overall_correct"]))
            row["accuracy_band"] = int(math.floor(max(deficit - 1e-12, 0.0) / accuracy_tolerance))

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        if bool(row["safety_pass"]):
            return (
                0,
                int(row["accuracy_band"]),
                float(row["avg_selected_layers"]),
                -float(row["overall_correct"]),
                -float(row["macro_benchmark_correct"]),
                -float(row["rescue_correct"]),
                str(row["eval_run_id"]),
            )
        return (
            1,
            float(row["safety_violation"]),
            -float(row["overall_correct"]),
            -float(row["preservation_correct"]),
            float(row["avg_selected_layers"]),
            str(row["eval_run_id"]),
        )

    ranked = sorted(rows, key=key)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked

