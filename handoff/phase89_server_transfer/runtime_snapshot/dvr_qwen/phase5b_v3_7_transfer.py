"""Train/val transfer diagnostics for Phase 5B v3.7 probes."""

from __future__ import annotations

from typing import Any

import torch

from dvr_qwen.phase5b_v3_6_separability import binary_safe_vs_harm_label
from dvr_qwen.phase5b_v3_7_content_probe import evaluate_probe_scores


def benchmark_probe_metrics(rows: list[dict[str, Any]], scores: torch.Tensor) -> dict[str, dict[str, Any]]:
    by_benchmark: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        if binary_safe_vs_harm_label(row) is None:
            continue
        by_benchmark.setdefault(str(row.get("benchmark", "unknown")), []).append(idx)

    out: dict[str, dict[str, Any]] = {}
    for benchmark, indices in sorted(by_benchmark.items()):
        sub_rows = [rows[idx] for idx in indices]
        sub_scores = scores[torch.tensor(indices, dtype=torch.long)]
        out[benchmark] = evaluate_probe_scores(sub_scores, sub_rows)
    return out


def compact_metric(metrics: dict[str, Any]) -> dict[str, Any]:
    topk = metrics.get("top_k_at_positive_count", {})
    return {
        "auc": metrics.get("roc_auc"),
        "average_precision": metrics.get("average_precision"),
        "topk_precision": topk.get("precision"),
        "topk_tp": int(topk.get("true_positive", 0)),
        "positive": int(metrics.get("positive", 0)),
        "num_examples": int(metrics.get("num_examples", 0)),
    }


def _split_metrics(summary: dict[str, Any], split_name: str) -> dict[str, Any]:
    return compact_metric(summary["linear_safe_harm_probe"]["metrics"][split_name])


def _benchmark_metrics(summary: dict[str, Any], split_name: str) -> dict[str, dict[str, Any]]:
    key = f"{split_name}_by_benchmark"
    return {
        benchmark: compact_metric(metrics)
        for benchmark, metrics in sorted(summary["linear_safe_harm_probe"].get(key, {}).items())
    }


def _metric_delta(current: dict[str, Any], baseline: dict[str, Any], key: str) -> float | None:
    left = current.get(key)
    right = baseline.get(key)
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _benchmark_notes(by_benchmark: dict[str, dict[str, dict[str, Any]]]) -> dict[str, list[str]]:
    calibration = by_benchmark.get("calibration", {})
    val = by_benchmark.get("val", {})
    notes = {
        "lost_topk_from_calibration_to_val": [],
        "zero_topk_on_val": [],
        "zero_topk_on_calibration_and_val": [],
    }
    for benchmark in sorted(set(calibration) | set(val)):
        cal_tp = int(calibration.get(benchmark, {}).get("topk_tp", 0))
        val_tp = int(val.get(benchmark, {}).get("topk_tp", 0))
        if cal_tp > 0 and val_tp == 0:
            notes["lost_topk_from_calibration_to_val"].append(benchmark)
        if val_tp == 0:
            notes["zero_topk_on_val"].append(benchmark)
        if cal_tp == 0 and val_tp == 0:
            notes["zero_topk_on_calibration_and_val"].append(benchmark)
    return notes


def transfer_comparison(feature_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_feature_set: dict[str, dict[str, Any]] = {}
    for name, summary in sorted(feature_summaries.items()):
        by_feature_set[name] = {
            "calibration": _split_metrics(summary, "calibration"),
            "val": _split_metrics(summary, "val"),
            "by_benchmark": {
                "calibration": _benchmark_metrics(summary, "calibration"),
                "val": _benchmark_metrics(summary, "val"),
            },
        }

    deltas: dict[str, dict[str, Any]] = {}
    baseline = by_feature_set.get("v3_6_route_proposal")
    if baseline:
        for name, row in sorted(by_feature_set.items()):
            if name == "v3_6_route_proposal":
                continue
            deltas[name] = {
                "calibration_auc_delta": _metric_delta(row["calibration"], baseline["calibration"], "auc"),
                "val_auc_delta": _metric_delta(row["val"], baseline["val"], "auc"),
                "calibration_topk_tp_delta": _metric_delta(row["calibration"], baseline["calibration"], "topk_tp"),
                "val_topk_tp_delta": _metric_delta(row["val"], baseline["val"], "topk_tp"),
            }

    benchmark_notes = {
        name: _benchmark_notes(row["by_benchmark"])
        for name, row in sorted(by_feature_set.items())
    }
    decision = "needs_review"
    content_delta = deltas.get("v3_7_content_only")
    if content_delta:
        cal_tp_delta = content_delta.get("calibration_topk_tp_delta") or 0.0
        val_tp_delta = content_delta.get("val_topk_tp_delta") or 0.0
        val_auc_delta = content_delta.get("val_auc_delta") or 0.0
        if cal_tp_delta > 0 and (val_tp_delta <= 0 or val_auc_delta < 0.0):
            decision = "content_signal_is_train_internal_not_val_transfer"
        elif val_tp_delta > 0 and val_auc_delta >= 0.0:
            decision = "content_signal_transfers_to_val"
        else:
            decision = "content_signal_mixed_or_marginal"

    return {
        "by_feature_set": by_feature_set,
        "deltas_vs_v3_6": deltas,
        "benchmark_notes": benchmark_notes,
        "decision": decision,
    }
