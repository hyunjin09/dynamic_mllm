#!/usr/bin/env python3
"""Aggregate the frozen Predictability Step-D external transfer evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dense_failure_stage2.predictability_learnability import (  # noqa: E402
    binary_classification_metrics,
    harmful_ranking_metrics,
    high_precision_harmful_metrics,
    regression_metrics,
)
from experiments import run_predictability_stepD_external_transfer as stepd  # noqa: E402


DEFAULT_CONFIG = stepd.DEFAULT_CONFIG
FAMILIES = ("chartqa", "textvqa", "mmmu_pro", "pope")


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    import io

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow(_clean(dict(row)))
    _atomic(path, output.getvalue().encode())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0", ""}:
        return False
    raise ValueError(f"not a boolean: {value}")


def _metric(truth: Sequence[int], prediction: Sequence[float]) -> dict[str, Any]:
    return dict(binary_classification_metrics(truth=np.asarray(truth), prediction=np.asarray(prediction)))


def _stage1_tables(
    output_root: Path,
    predictions: Sequence[Mapping[str, Any]],
    triggers: Sequence[Mapping[str, Any]],
    thresholds: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    outcome = {str(row["uid"]): row for row in triggers}
    fields = {
        "m0_x": "prediction_m0_x",
        "m1": "prediction_m1",
        "m3": "prediction_m3_ensemble",
        "question_knn": "prediction_question_knn",
    }
    benchmark_rows = []
    controls = []
    groups = [(family, [row for row in predictions if row["benchmark_family"] == family]) for family in FAMILIES]
    groups += [("pooled", list(predictions))]
    for name, rows in groups:
        truth = [int(not _bool(outcome[str(row["uid"])]["dense_correct"])) for row in rows]
        metrics = {model: _metric(truth, [float(row[field]) for row in rows]) for model, field in fields.items()}
        uids = sorted({str(row["uid"]) for row in rows})
        dense_c = sum(_bool(outcome[uid]["dense_correct"]) for uid in uids)
        result = {
            "benchmark": name,
            "uids": len(uids),
            "states": len(rows),
            "dense_correct": dense_c,
            "dense_wrong": len(uids) - dense_c,
            "m0_x_auroc": metrics["m0_x"]["auroc"],
            "m1_auroc": metrics["m1"]["auroc"],
            "m3_auroc": metrics["m3"]["auroc"],
            "m3_auprc": metrics["m3"]["auprc"],
            "question_knn_auroc": metrics["question_knn"]["auroc"],
        }
        benchmark_rows.append(result)
        for model, metric in metrics.items():
            controls.append({"benchmark": name, "model": model, **metric})
    macro = {
        "benchmark": "macro",
        "uids": sum(row["uids"] for row in benchmark_rows if row["benchmark"] in FAMILIES),
        "states": sum(row["states"] for row in benchmark_rows if row["benchmark"] in FAMILIES),
        "dense_correct": sum(row["dense_correct"] for row in benchmark_rows if row["benchmark"] in FAMILIES),
        "dense_wrong": sum(row["dense_wrong"] for row in benchmark_rows if row["benchmark"] in FAMILIES),
    }
    for field in ("m0_x_auroc", "m1_auroc", "m3_auroc", "m3_auprc", "question_knn_auroc"):
        macro[field] = float(np.nanmean([row[field] for row in benchmark_rows if row["benchmark"] in FAMILIES]))
    benchmark_rows.insert(4, macro)

    layer_rows = []
    for family in FAMILIES:
        family_rows = [row for row in predictions if row["benchmark_family"] == family]
        for layer in range(28):
            rows = [row for row in family_rows if int(row["layer"]) == layer]
            truth = [int(not _bool(outcome[str(row["uid"])]["dense_correct"])) for row in rows]
            metrics = _metric(truth, [float(row["prediction_m3_ensemble"]) for row in rows])
            layer_rows.append({"benchmark": family, "layer": layer, **metrics})

    seed_rows = []
    for family in FAMILIES:
        rows = [row for row in predictions if row["benchmark_family"] == family]
        truth = [int(not _bool(outcome[str(row["uid"])]["dense_correct"])) for row in rows]
        values = []
        for index in (1, 2, 3):
            metrics = _metric(truth, [float(row[f"prediction_m3_seed{index}"]) for row in rows])
            seed_rows.append({"benchmark": family, "seed_index": index, **metrics})
            values.append(float(metrics["auroc"]))
        seed_rows.append({"benchmark": family, "seed_index": "mean_sd", "support": len(rows), "positives": sum(truth), "negatives": len(rows) - sum(truth), "prevalence": sum(truth) / len(rows), "auroc": float(np.mean(values)), "auprc": None, "auroc_sd": float(np.std(values, ddof=0))})

    by_uid = defaultdict(list)
    for row in predictions:
        by_uid[str(row["uid"])].append(row)
    operating = []
    for threshold_row in thresholds:
        if str(threshold_row["model"]) != "m3":
            continue
        name = str(threshold_row["seed"] if "seed" in threshold_row else threshold_row["seed_index"] if "seed_index" in threshold_row else threshold_row.get("seed_or_ensemble"))
        field = {
            "2026090801": "prediction_m3_seed1",
            "2026090802": "prediction_m3_seed2",
            "2026090803": "prediction_m3_seed3",
            "ensemble": "prediction_m3_ensemble",
        }[name]
        threshold = float(threshold_row["threshold"])
        nominal = float(threshold_row["target_correct_preservation"] if "target_correct_preservation" in threshold_row else threshold_row["nominal"] if "nominal" in threshold_row else threshold_row["nominal_c_preserve"] if "nominal_c_preserve" in threshold_row else threshold_row["nominal_c_preservation"])
        for family in FAMILIES:
            uids = sorted(uid for uid, rows in by_uid.items() if rows[0]["benchmark_family"] == family)
            decisions = []
            for uid in uids:
                rows = sorted(by_uid[uid], key=lambda row: int(row["layer"]))
                layer = next((int(row["layer"]) for row in rows if float(row[field]) > threshold), None)
                decisions.append((uid, layer))
            correct = [(uid, layer) for uid, layer in decisions if _bool(outcome[uid]["dense_correct"])]
            wrong = [(uid, layer) for uid, layer in decisions if not _bool(outcome[uid]["dense_correct"])]
            triggered = [(uid, layer) for uid, layer in decisions if layer is not None]
            operating.append({"benchmark": family, "prediction": name, "internal_nominal_c_preserve": nominal, "threshold": threshold, "external_c_preserve": 1.0 - sum(layer is not None for _, layer in correct) / len(correct), "external_w_recall": sum(layer is not None for _, layer in wrong) / len(wrong), "precision": sum(not _bool(outcome[uid]["dense_correct"]) for uid, _ in triggered) / len(triggered) if triggered else None, "triggered": len(triggered), "median_trigger_layer": float(np.median([layer for _, layer in triggered])) if triggered else None})
    return benchmark_rows, layer_rows, operating, controls + seed_rows


def _robust_funnel(triggers: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for family in (*FAMILIES, "pooled"):
        rows = list(triggers) if family == "pooled" else [row for row in triggers if row["benchmark_family"] == family]
        correct = [row for row in rows if _bool(row["dense_correct"])]
        wrong = [row for row in rows if not _bool(row["dense_correct"])]
        active = [row for row in rows if _bool(row["triggered"])]
        output.append({"benchmark": family, "uids": len(rows), "dense_correct": len(correct), "dense_wrong": len(wrong), "triggered": len(active), "p_trigger_given_c": sum(_bool(row["triggered"]) for row in correct) / len(correct) if correct else None, "p_trigger_given_w": sum(_bool(row["triggered"]) for row in wrong) / len(wrong) if wrong else None, "c_preservation": 1.0 - sum(_bool(row["triggered"]) for row in correct) / len(correct) if correct else None, "w_recall": sum(_bool(row["triggered"]) for row in wrong) / len(wrong) if wrong else None, "precision": sum(not _bool(row["dense_correct"]) for row in active) / len(active) if active else None, "median_trigger_layer": float(np.median([int(row["first_trigger_layer"]) for row in active])) if active else None})
    return output


def _semantic_tables(output_root: Path, predictions: Sequence[Mapping[str, Any]], triggers: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    semantic = stepd.read_jsonl(output_root / "question_semantics/nearest_internal_similarity.jsonl")
    outcome = {str(row["uid"]): row for row in triggers}
    by_uid = defaultdict(list)
    for row in predictions:
        by_uid[str(row["uid"])].append(row)
    summary = []
    conditioned = []
    for family in FAMILIES:
        rows = [row for row in semantic if row["benchmark_family"] == family]
        values = np.asarray([float(row["nearest_internal_similarity"]) for row in rows])
        summary.append({"benchmark": family, "uids": len(rows), "mean": float(values.mean()), "median": float(np.median(values)), "p10": float(np.quantile(values, 0.1)), "p90": float(np.quantile(values, 0.9))})
        median = float(np.median(values))
        for half, uids in (
            ("lower", {str(row["uid"]) for row in rows if float(row["nearest_internal_similarity"]) <= median}),
            ("upper", {str(row["uid"]) for row in rows if float(row["nearest_internal_similarity"]) > median}),
        ):
            layer_rows = [item for uid in uids for item in by_uid[uid]]
            truth = [int(not _bool(outcome[str(row["uid"])]["dense_correct"])) for row in layer_rows]
            conditioned.append({"benchmark": family, "similarity_half": half, "median_edge": median, **_metric(truth, [float(row["prediction_m3_ensemble"]) for row in layer_rows])})
    return summary, conditioned


def _stage2_tables(
    output_root: Path,
    read_predictions: Sequence[Mapping[str, Any]],
    write_predictions: Sequence[Mapping[str, Any]],
    utilities: Sequence[Mapping[str, str]],
    flips: Sequence[Mapping[str, str]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    truth = {str(row["state_id"]): row for row in utilities}
    flip = {str(row["state_id"]): row for row in flips}
    outputs = {}
    controls = []
    layer_rows = []
    relative_rows = []
    similarity_rows = []
    semantic = {str(row["uid"]): float(row["nearest_internal_similarity"]) for row in stepd.read_jsonl(output_root / "question_semantics/nearest_internal_similarity.jsonl")}
    for target, predictions in (("read", read_predictions), ("write", write_predictions)):
        primary_rows = []
        for family in FAMILIES:
            rows = [row for row in predictions if row["benchmark_family"] == family]
            if not rows:
                primary_rows.append({"benchmark": family, "triggered_uids": 0, "states": 0, "spearman": None, "pearson": None, "mae": None, "harmful_auroc": None, "harmful_auprc": None, "precision_at_top5": None, "precision_at_top10": None, "precision_at_top20": None, "m0_x_spearman": None, "question_knn_spearman": None})
                continue
            values = np.asarray([float(truth[str(row["state_id"])]["u_read_w1" if target == "read" else "u_write_r1"]) for row in rows])
            primary = np.asarray([float(row["m3_ensemble"]) for row in rows])
            reg = regression_metrics(truth=values, prediction=primary)
            harm = harmful_ranking_metrics(truth=values, prediction=primary)
            top = high_precision_harmful_metrics(truth=values, prediction=primary, coverages=(0.05, 0.1, 0.2), precision_targets=(0.9, 0.95))
            m0 = regression_metrics(truth=values, prediction=[float(row["m0_x_s20260908"]) for row in rows])
            knn = regression_metrics(truth=values, prediction=[float(row["question_knn"]) for row in rows])
            primary_rows.append({"benchmark": family, "triggered_uids": len({row["uid"] for row in rows}), "states": len(rows), "spearman": reg["spearman"], "pearson": reg["pearson"], "mae": reg["mae"], "harmful_auroc": harm["auroc"], "harmful_auprc": harm["auprc"], "precision_at_top5": top["precision_at_0.05"], "precision_at_top10": top["precision_at_0.1"], "precision_at_top20": top["precision_at_0.2"], "m0_x_spearman": m0["spearman"], "question_knn_spearman": knn["spearman"]})
            for model, field in (("m0_x", "m0_x_s20260908"), ("m1", "m1_s20260908"), ("m3", "m3_ensemble"), ("question_knn", "question_knn")):
                metrics = regression_metrics(truth=values, prediction=[float(row[field]) for row in rows])
                controls.append({"target": target, "benchmark": family, "model": model, **metrics})
            for bin_name, low, high in (("early", 0, 8), ("middle", 9, 18), ("late", 19, 27)):
                subset = [row for row in rows if low <= int(row["layer"]) <= high]
                if len(subset) >= 10:
                    y = [float(truth[str(row["state_id"])]["u_read_w1" if target == "read" else "u_write_r1"]) for row in subset]
                    layer_rows.append({"target": target, "benchmark": family, "depth_bin": bin_name, **regression_metrics(truth=y, prediction=[float(row["m3_ensemble"]) for row in subset])})
            for bin_name, low, high in (("at_trigger", 0, 0), ("after_1_2", 1, 2), ("after_3_5", 3, 5), ("after_6_plus", 6, 27)):
                subset = [row for row in rows if low <= int(row["trigger_relative_depth"]) <= high]
                if len(subset) >= 10:
                    y = [float(truth[str(row["state_id"])]["u_read_w1" if target == "read" else "u_write_r1"]) for row in subset]
                    relative_rows.append({"target": target, "benchmark": family, "trigger_relative_bin": bin_name, **regression_metrics(truth=y, prediction=[float(row["m3_ensemble"]) for row in subset])})
            median = float(np.median([semantic[str(row["uid"])] for row in rows]))
            for half, subset in (("lower", [row for row in rows if semantic[str(row["uid"])] <= median]), ("upper", [row for row in rows if semantic[str(row["uid"])] > median])):
                if len(subset) >= 10:
                    y = [float(truth[str(row["state_id"])]["u_read_w1" if target == "read" else "u_write_r1"]) for row in subset]
                    similarity_rows.append({"target": target, "benchmark": family, "similarity_half": half, "median_edge": median, **regression_metrics(truth=y, prediction=[float(row["m3_ensemble"]) for row in subset])})
        valid = [row for row in primary_rows if row["states"]]
        primary_rows.append({"benchmark": "macro", "triggered_uids": sum(row["triggered_uids"] for row in valid), "states": sum(row["states"] for row in valid), **{field: float(np.nanmean([row[field] for row in valid])) for field in ("spearman", "pearson", "mae", "harmful_auroc", "harmful_auprc", "precision_at_top5", "precision_at_top10", "precision_at_top20", "m0_x_spearman", "question_knn_spearman")}})
        outputs[target] = primary_rows

    strong = []
    for target, predictions in (("read", read_predictions), ("write", write_predictions)):
        flag = "read_harmful_flip_w1" if target == "read" else "write_harmful_flip_r1"
        for family in FAMILIES:
            rows = [row for row in predictions if row["benchmark_family"] == family]
            if not rows:
                continue
            labels = [int(_bool(flip[str(row["state_id"])][flag])) for row in rows]
            metrics = binary_classification_metrics(truth=labels, prediction=[-float(row["m3_ensemble"]) for row in rows]) if 0 < sum(labels) < len(labels) else {"support": len(labels), "positives": sum(labels), "negatives": len(labels)-sum(labels), "prevalence": sum(labels)/len(labels), "auroc": None, "auprc": None}
            strong.append({"target": target, "benchmark": family, "event": flag, **metrics})
    return outputs, controls, layer_rows, relative_rows, similarity_rows + strong


def _local_rescue(utilities: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    output = []
    for family in FAMILIES:
        rows = [row for row in utilities if row["benchmark_family"] == family]
        if not rows:
            output.append({"benchmark": family, "states": 0, "local_rescue": 0, "local_regression": 0, "all_four_correct": 0, "all_four_wrong": 0})
        else:
            output.append({"benchmark": family, "states": len(rows), "local_rescue": sum(_bool(row["local_rescue_exists"]) for row in rows), "local_regression": sum(_bool(row["local_regression_exists"]) for row in rows), "all_four_correct": sum(_bool(row["all_four_correct"]) for row in rows), "all_four_wrong": sum(_bool(row["all_four_wrong"]) for row in rows)})
    return output


def _bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_field: str,
    draws: int,
    seed: int,
    metric,
) -> tuple[float, float, float]:
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row[group_field])].append(index)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        selected = rng.integers(0, len(keys), size=len(keys))
        indices = np.concatenate([np.asarray(groups[keys[int(index)]], dtype=np.int64) for index in selected])
        value = float(metric(indices))
        if math.isfinite(value):
            values.append(value)
    array = np.asarray(values)
    if len(array) != draws:
        raise RuntimeError("Step-D bootstrap produced non-finite draws")
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975)), float(array.mean())


def _statistics(
    config: Mapping[str, Any],
    output_root: Path,
    stage1: Sequence[Mapping[str, Any]],
    triggers: Sequence[Mapping[str, Any]],
    stage2_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    utilities: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    draws = int(config["evaluation"]["bootstrap_draws"])
    seed = int(config["evaluation"]["bootstrap_seed"])
    outcome = {str(row["uid"]): row for row in triggers}
    utility = {str(row["state_id"]): row for row in utilities}
    rows_out = []
    for number, family in enumerate(FAMILIES):
        rows = [row for row in stage1 if row["benchmark_family"] == family and int(row["layer"]) == 20]
        labels = np.asarray([int(not _bool(outcome[str(row["uid"])]["dense_correct"])) for row in rows])
        scores = np.asarray([float(row["prediction_m3_ensemble"]) for row in rows])
        for metric_name, position in (("auroc", "auroc"), ("auprc", "auprc")):
            point = _metric(labels, scores)[position]
            low, high, _ = _bootstrap(rows, group_field="image_group_id", draws=draws, seed=seed + number * 31 + (0 if metric_name == "auroc" else 1), metric=lambda idx, p=position: _metric(labels[idx], scores[idx])[p])
            rows_out.append({"domain": "stage1", "target": "dense_wrong", "benchmark": family, "metric": metric_name, "basis": "internal_peak_layer20", "estimate": point, "ci_low": low, "ci_high": high, "draws": draws, "bootstrap_unit": "image_group_id"})
    for target, predictions in stage2_predictions.items():
        column = "u_read_w1" if target == "read" else "u_write_r1"
        for number, family in enumerate(FAMILIES):
            rows = [row for row in predictions if row["benchmark_family"] == family]
            if not rows:
                continue
            truth = np.asarray([float(utility[str(row["state_id"])][column]) for row in rows])
            score = np.asarray([float(row["m3_ensemble"]) for row in rows])
            functions = {
                "spearman": lambda idx: regression_metrics(truth=truth[idx], prediction=score[idx])["spearman"],
                "harmful_auroc": lambda idx: harmful_ranking_metrics(truth=truth[idx], prediction=score[idx])["auroc"],
                "precision_at_top10": lambda idx: high_precision_harmful_metrics(truth=truth[idx], prediction=score[idx], coverages=(0.1,), precision_targets=(0.9,))["precision_at_0.1"],
            }
            points = {"spearman": regression_metrics(truth=truth, prediction=score)["spearman"], "harmful_auroc": harmful_ranking_metrics(truth=truth, prediction=score)["auroc"], "precision_at_top10": high_precision_harmful_metrics(truth=truth, prediction=score, coverages=(0.1,), precision_targets=(0.9,))["precision_at_0.1"]}
            for offset, (metric_name, function) in enumerate(functions.items()):
                low, high, _ = _bootstrap(rows, group_field="image_group_id", draws=draws, seed=seed + 1000 + number * 37 + offset + (0 if target == "read" else 500), metric=function)
                rows_out.append({"domain": "stage2", "target": target, "benchmark": family, "metric": metric_name, "basis": "all_dense_post_trigger_states", "estimate": points[metric_name], "ci_low": low, "ci_high": high, "draws": draws, "bootstrap_unit": "image_group_id"})
    write_csv(output_root / "statistics/group_bootstrap_ci.csv", rows_out)
    degradation = []
    for row in rows_out:
        if row["domain"] == "stage1" and row["metric"] == "auroc":
            degradation.append({"benchmark": row["benchmark"], "external_layer20_auroc": row["estimate"], "stepB_id_overall_auroc": 0.7869, "external_minus_id": float(row["estimate"]) - 0.7869, "difference_ci_low": float(row["ci_low"]) - 0.7869, "difference_ci_high": float(row["ci_high"]) - 0.7869})
    write_csv(output_root / "statistics/external_vs_internal_degradation.csv", degradation)
    return rows_out


def _plots(output_root: Path, benchmark_rows: Sequence[Mapping[str, Any]], layer_rows: Sequence[Mapping[str, Any]], operating: Sequence[Mapping[str, Any]], stage2: Mapping[str, Sequence[Mapping[str, Any]]], semantic: Sequence[Mapping[str, Any]]) -> None:
    root = output_root / "figures"
    family_rows = [row for row in benchmark_rows if row["benchmark"] in FAMILIES]
    names = [row["benchmark"] for row in family_rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, [row["m3_auroc"] for row in family_rows])
    ax.axhline(0.5, color="black", linewidth=1)
    ax.set_xticks(x, names, rotation=20)
    ax.set_ylabel("AUROC")
    fig.tight_layout(); fig.savefig(root / "stage1_external_benchmark_auroc.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for family in FAMILIES:
        rows = [row for row in layer_rows if row["benchmark"] == family]
        ax.plot([row["layer"] for row in rows], [row["auroc"] for row in rows], label=family)
    ax.axvline(20, color="black", linestyle="--", linewidth=1); ax.set_xlabel("Layer"); ax.set_ylabel("AUROC"); ax.legend(); fig.tight_layout(); fig.savefig(root / "stage1_layer_emergence_external.png", dpi=180); plt.close(fig)

    ensemble = [row for row in operating if row["prediction"] == "ensemble" and row["internal_nominal_c_preserve"] in {0.95, 0.98, 0.99}]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for family in FAMILIES:
        rows = [row for row in ensemble if row["benchmark"] == family]
        ax.plot([row["internal_nominal_c_preserve"] for row in rows], [row["external_c_preserve"] for row in rows], marker="o", label=family)
    ax.plot([0.95, 0.99], [0.95, 0.99], color="black", linestyle="--"); ax.legend(); ax.set_xlabel("Internal nominal C preservation"); ax.set_ylabel("External C preservation"); fig.tight_layout(); fig.savefig(root / "stage1_operating_point_calibration_drift.png", dpi=180); plt.close(fig)

    ladder = [("ID", 0.7869), ("Q1", 0.7828), ("Cluster", 0.7725)] + [(row["benchmark"], row["m3_auroc"]) for row in family_rows]
    fig, ax = plt.subplots(figsize=(9, 4.5)); ax.plot(range(len(ladder)), [value for _, value in ladder], marker="o"); ax.set_xticks(range(len(ladder)), [name for name, _ in ladder], rotation=30); ax.set_ylabel("M3 AUROC"); fig.tight_layout(); fig.savefig(root / "stage1_generalization_ladder_final.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5)); width=.25; ax.bar(x-width,[r["m0_x_auroc"] for r in family_rows],width,label="M0-X"); ax.bar(x,[r["m3_auroc"] for r in family_rows],width,label="M3"); ax.bar(x+width,[r["question_knn_auroc"] for r in family_rows],width,label="kNN"); ax.set_xticks(x,names,rotation=20); ax.legend(); ax.set_ylabel("AUROC"); fig.tight_layout(); fig.savefig(root / "stage1_hidden_vs_knn_vs_nuisance.png",dpi=180); plt.close(fig)

    for target in ("read", "write"):
        rows = [row for row in stage2[target] if row["benchmark"] in FAMILIES]
        fig, ax = plt.subplots(figsize=(8,4.5)); ax.bar(range(len(rows)), [0 if row["spearman"] is None else row["spearman"] for row in rows]); ax.set_xticks(range(len(rows)),[row["benchmark"] for row in rows],rotation=20); ax.axhline(0,color="black",linewidth=1); ax.set_ylabel(f"{target.upper()} Spearman"); fig.tight_layout(); fig.savefig(root/f"stage2_{target}_external_transfer.png",dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,4.5)); read=[row for row in stage2["read"] if row["benchmark"] in FAMILIES]; write=[row for row in stage2["write"] if row["benchmark"] in FAMILIES]; ax.plot(range(3),[0.0416,0.0512,np.nanmean([r["spearman"] for r in read if r["spearman"] is not None])],marker="o",label="READ"); ax.plot(range(3),[0.0347,0.03,np.nanmean([r["spearman"] for r in write if r["spearman"] is not None])],marker="o",label="WRITE"); ax.set_xticks(range(3),["ID","Cluster OOD","External macro"]); ax.legend(); ax.set_ylabel("Spearman"); fig.tight_layout(); fig.savefig(root/"stage2_generalization_ladder_final.png",dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4.5)); ax.bar(range(len(semantic)),[row["median"] for row in semantic]); ax.set_xticks(range(len(semantic)),[row["benchmark"] for row in semantic],rotation=20); ax.set_ylabel("Median nearest-internal cosine"); fig.tight_layout(); fig.savefig(root/"external_semantic_similarity.png",dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4.5)); ax.bar([0,1],[float(np.mean([row["m3_auroc"] for row in family_rows]))-.5,np.nanmean([abs(row["spearman"]) for target in ("read","write") for row in stage2[target] if row["benchmark"] in FAMILIES and row["spearman"] is not None])]); ax.set_xticks([0,1],["Stage-1 AUROC-0.5","Stage-2 |rho|"]); fig.tight_layout(); fig.savefig(root/"stage1_stage2_asymmetry_final.png",dpi=180); plt.close(fig)


def _category(config: Mapping[str, Any], benchmark_rows: Sequence[Mapping[str, Any]], stage2: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[str, str]:
    rules = config["interpretation_rules"]
    rows = {row["benchmark"]: row for row in benchmark_rows if row["benchmark"] in FAMILIES}
    useful = {name: row["m3_auroc"] >= float(rules["stage1_useful_auroc_minimum"]) and row["m3_auroc"] - max(row["m0_x_auroc"], row["question_knn_auroc"]) >= float(rules["stage1_control_advantage_minimum"]) for name, row in rows.items()}
    if all(useful.values()):
        d1 = "D1-A"
    elif useful.get("chartqa") and useful.get("textvqa") and not (useful.get("mmmu_pro") and useful.get("pope")):
        d1 = "D1-B"
    else:
        d1 = "D1-C"
    strong = []
    for target in ("read", "write"):
        for row in stage2[target]:
            if row["benchmark"] in FAMILIES and row["spearman"] is not None:
                strong.append((target, row["benchmark"], abs(float(row["spearman"])) >= float(rules["stage2_nontrivial_absolute_spearman_minimum"]) and abs(float(row["spearman"])) - max(abs(float(row["m0_x_spearman"])), abs(float(row["question_knn_spearman"]))) >= float(rules["stage2_control_advantage_minimum"])))
    count = sum(value for _, _, value in strong)
    if count >= int(rules["stage2_broad_requires_supported_benchmarks"]):
        d2 = "D2-C"
    elif any(value and family in {"chartqa", "textvqa"} for _, family, value in strong):
        d2 = "D2-B"
    else:
        d2 = "D2-A"
    return d1, d2


def aggregate(config_path: Path) -> None:
    contract, output_root, _ = stepd.verify_contract(config_path)
    stepd._verify_frozen_predictions(contract, output_root)
    parity = stepd.read_json(output_root / "stage2_measurement/parity_report.json")
    if not parity.get("complete") or not parity.get("full_branch_and_state_parity"):
        raise RuntimeError("Step-D aggregation requires complete parity-valid measurements")
    config = contract["static_config"]
    stage1_predictions = stepd.read_jsonl(output_root / "predictions_frozen_before_labels/stage1_external_predictions.jsonl")
    read_predictions = stepd.read_jsonl(output_root / "predictions_frozen_before_labels/stage2_read_external_predictions.jsonl")
    write_predictions = stepd.read_jsonl(output_root / "predictions_frozen_before_labels/stage2_write_external_predictions.jsonl")
    triggers = stepd.read_jsonl(output_root / "external_manifests/p90_trigger_manifest.jsonl")
    thresholds = read_csv(output_root / "refit/stage1_internal_calibration_thresholds.csv")
    utilities = read_csv(output_root / "stage2_measurement/utility_labels.csv")
    flips = read_csv(output_root / "stage2_measurement/correctness_flip_labels.csv")

    benchmark_rows, layer_rows, operating, controls_seeds = _stage1_tables(output_root, stage1_predictions, triggers, thresholds)
    controls = [row for row in controls_seeds if "model" in row]
    seeds = [row for row in controls_seeds if "seed_index" in row]
    write_csv(output_root / "stage1/benchmark_metrics.csv", benchmark_rows)
    write_csv(output_root / "stage1/overall_metrics.csv", [row for row in benchmark_rows if row["benchmark"] in {"macro", "pooled"}])
    write_csv(output_root / "stage1/layer_metrics.csv", layer_rows)
    write_csv(output_root / "stage1/operating_point_transfer.csv", operating)
    funnel = _robust_funnel(triggers)
    write_csv(output_root / "stage1/robust_p90_external_funnel.csv", funnel)
    write_csv(output_root / "stage1/model_control_comparison.csv", controls)
    write_csv(output_root / "stage1/question_knn_metrics.csv", [row for row in controls if row["model"] == "question_knn"])
    write_csv(output_root / "stage1/seed_metrics.csv", seeds)
    semantic_summary, conditioned = _semantic_tables(output_root, stage1_predictions, triggers)
    write_csv(output_root / "stage1/semantic_similarity_metrics.csv", conditioned)
    write_csv(output_root / "question_semantics/benchmark_similarity_summary.csv", semantic_summary)

    stage2, control2, layer2, relative2, similarity_strong = _stage2_tables(output_root, read_predictions, write_predictions, utilities, flips)
    write_csv(output_root / "stage2_predictability/read_benchmark_metrics.csv", stage2["read"])
    write_csv(output_root / "stage2_predictability/write_benchmark_metrics.csv", stage2["write"])
    macro = [{"target": target, **next(row for row in stage2[target] if row["benchmark"] == "macro")} for target in ("read", "write")]
    write_csv(output_root / "stage2_predictability/macro_metrics.csv", macro)
    write_csv(output_root / "stage2_predictability/control_comparison.csv", control2)
    strong = [row for row in similarity_strong if "event" in row]
    similarities2 = [row for row in similarity_strong if "similarity_half" in row]
    write_csv(output_root / "stage2_predictability/strong_flip_metrics.csv", strong)
    write_csv(output_root / "stage2_predictability/layer_metrics.csv", layer2)
    write_csv(output_root / "stage2_predictability/trigger_relative_metrics.csv", relative2)
    write_csv(output_root / "stage2_predictability/semantic_similarity_metrics.csv", similarities2)
    seed2 = []
    utility = {str(row["state_id"]): row for row in utilities}
    for target, rows in (("read", read_predictions), ("write", write_predictions)):
        column = "u_read_w1" if target == "read" else "u_write_r1"
        for family in FAMILIES:
            subset = [row for row in rows if row["benchmark_family"] == family]
            for index, seed in enumerate((2026090801, 2026090802, 2026090803), 1):
                if not subset:
                    continue
                metric = regression_metrics(truth=[float(utility[str(row["state_id"])][column]) for row in subset], prediction=[float(row[f"m3_s{seed}"]) for row in subset])
                seed2.append({"target": target, "benchmark": family, "seed": seed, **metric})
    write_csv(output_root / "stage2_predictability/seed_metrics.csv", seed2)
    rescue = _local_rescue(utilities)
    write_csv(output_root / "stage2_measurement/local_rescue_summary.csv", rescue)
    distribution = []
    for target, column in (("read", "u_read_w1"), ("write", "u_write_r1")):
        for family in FAMILIES:
            values = np.asarray([float(row[column]) for row in utilities if row["benchmark_family"] == family])
            if len(values):
                distribution.append({"target": target, "benchmark": family, "states": len(values), "mean": float(values.mean()), "median": float(np.median(values)), "negative": int((values < 0).sum()), "zero": int((values == 0).sum()), "positive": int((values > 0).sum()), "p10": float(np.quantile(values,.1)), "p90": float(np.quantile(values,.9))})
    write_csv(output_root / "stage2_measurement/utility_distribution_summary.csv", distribution)
    _statistics(config, output_root, stage1_predictions, triggers, {"read": read_predictions, "write": write_predictions}, utilities)
    _plots(output_root, benchmark_rows, layer_rows, operating, stage2, semantic_summary)
    d1, d2 = _category(config, benchmark_rows, stage2)

    final_ladder = [
        ("Step-B ID", 0.7869), ("Step-C least-similar Q1", 0.7828), ("Step-C cluster OOD", 0.7725),
        ("Historical to Canonical", 0.4342), ("Canonical to Historical", 0.5563),
        ("LODO GQA", 0.5508), ("LODO ChartQA", 0.6043), ("LODO TextVQA", 0.6006),
    ] + [(f"External {row['benchmark']}", row["m3_auroc"]) for row in benchmark_rows if row["benchmark"] in (*FAMILIES, "macro")]
    write_csv(output_root / "stage1/generalization_ladder.csv", [{"regime": name, "m3_auroc": value} for name, value in final_ladder])
    write_csv(output_root / "stage2_predictability/generalization_ladder.csv", [{"target": "read", "regime": "ID", "spearman": .0416}, {"target": "write", "regime": "ID", "spearman": .0347}, {"target": "read", "regime": "cluster_OOD", "spearman": .0512}, {"target": "write", "regime": "cluster_OOD", "spearman": .03}] + [{"target": target, "regime": f"external_{row['benchmark']}", "spearman": row["spearman"]} for target in ("read", "write") for row in stage2[target] if row["benchmark"] in (*FAMILIES, "macro")])

    main_rows = {row["benchmark"]: row for row in benchmark_rows}
    read_rows = {row["benchmark"]: row for row in stage2["read"]}
    write_rows = {row["benchmark"]: row for row in stage2["write"]}
    funnel_rows = {row["benchmark"]: row for row in funnel}
    peaks = {family: max((row for row in layer_rows if row["benchmark"] == family), key=lambda row: -1 if row["auroc"] is None else row["auroc"]) for family in FAMILIES}
    summary = f"""# Predictability Step-D external-transfer summary

Contract: `{contract['contract_sha256']}`

## Required answers

1. **Full refits:** yes; all frozen Stage-1/Stage-2 M0-X, M1, and M3 tasks trained on all internal data for the prospectively frozen Step-B-median epochs.
2. **Dense parity:** yes; all 19,960 UIDs matched the established generated tokens, task score/correctness, images, and robust trigger trace.
3. **P90 domain:** {sum(row['triggered'] for row in funnel if row['benchmark'] in FAMILIES):,} external UIDs entered Stage 2.
4. **Trigger parity:** yes; the exact prior P90 trigger census was reproduced, including zero POPE triggers.
5. **Stage-1 external M3:** ChartQA AUROC/AUPRC {main_rows['chartqa']['m3_auroc']:.4f}/{main_rows['chartqa']['m3_auprc']:.4f}; TextVQA {main_rows['textvqa']['m3_auroc']:.4f}/{main_rows['textvqa']['m3_auprc']:.4f}; MMMU-Pro {main_rows['mmmu_pro']['m3_auroc']:.4f}/{main_rows['mmmu_pro']['m3_auprc']:.4f}; POPE {main_rows['pope']['m3_auroc']:.4f}/{main_rows['pope']['m3_auprc']:.4f}.
6. **Stage-1 macro AUROC:** {main_rows['macro']['m3_auroc']:.4f}.
7. **Peak layers:** {', '.join(f"{family}=L{peaks[family]['layer']} ({peaks[family]['auroc']:.4f})" for family in FAMILIES)}.
8. **Layer 20:** exact layer-20 values are in `stage1/layer_metrics.csv`; no benchmark-specific deployment layer was selected.
9. **Calibration drift:** internally calibrated 95/98/99% points are reported without external repair in `stage1/operating_point_transfer.csv`.
10. **Existing robust P90:** external funnel is preserved in `stage1/robust_p90_external_funnel.csv`; total triggers={funnel_rows['pooled']['triggered']}.
11. **M3 versus nuisance:** per-benchmark differences are in the main table/control file; no nuisance feature contains source or dataset ID.
12. **M3 versus question-kNN:** per-benchmark comparisons are in `stage1/model_control_comparison.csv`.
13. **Semantic similarity:** benchmark mean/median/P10/P90 are in `question_semantics/benchmark_similarity_summary.csv`.
14. **Similarity-conditioned transfer:** lower/upper external halves are in `stage1/semantic_similarity_metrics.csv`.
15. **External Stage-2 measurement:** {len(utilities):,} dense post-trigger states.
16. **Four-branch validity:** FULL token/score/correctness/q parity and utility algebra passed for all {len(utilities):,} states.
17. **Local opportunities:** per-benchmark rescue/regression/all-correct/all-wrong counts are in `stage2_measurement/local_rescue_summary.csv`.
18. **READ utility:** ChartQA/TextVQA/MMMU-Pro Spearman {read_rows['chartqa']['spearman']!s}/{read_rows['textvqa']['spearman']!s}/{read_rows['mmmu_pro']['spearman']!s}; POPE is N/A because it never triggered.
19. **WRITE utility:** ChartQA/TextVQA/MMMU-Pro Spearman {write_rows['chartqa']['spearman']!s}/{write_rows['textvqa']['spearman']!s}/{write_rows['mmmu_pro']['spearman']!s}; POPE is N/A.
20. **Stage-2 niche:** fixed rule classification is `{d2}`; detailed uncertainty and controls prevent selecting an isolated favorable cell.
21. **Stage-2 controls:** M3, transfer-safe nuisance, M1, and exact-layer question-kNN are compared in `stage2_predictability/control_comparison.csv`.
22. **Asymmetry:** Stage-1 category `{d1}` and Stage-2 category `{d2}` provide the frozen external answer.
23. **Stage-1 category:** `{d1}`.
24. **Stage-2 category:** `{d2}`.
25. **Not established:** deployment routing gain, causal mechanism, impossibility of richer-history/counterfactual Stage 2, or transfer of a redesigned Stage-1 representation.

## Main Stage-1 table

| Benchmark | N | Dense C | Dense W | M0-X AUROC | M1 AUROC | M3 AUROC | M3 AUPRC | kNN AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(f"| {name} | {main_rows[name]['uids']} | {main_rows[name]['dense_correct']} | {main_rows[name]['dense_wrong']} | {main_rows[name]['m0_x_auroc']:.4f} | {main_rows[name]['m1_auroc']:.4f} | {main_rows[name]['m3_auroc']:.4f} | {main_rows[name]['m3_auprc']:.4f} | {main_rows[name]['question_knn_auroc']:.4f} |" for name in (*FAMILIES, "macro", "pooled")) + "\n"
    _atomic(output_root / "summaries/stepD_external_transfer_summary.md", summary.encode())
    phase_summary = f"""# Predictability phase A-to-D final summary

- **Measurement (Step A):** controlled local READ/WRITE effects and correctness flips exist under complete four-branch intervention measurement.
- **In-domain learnability (Step B):** eventual Dense failure is learnable (M3 AUROC 0.7869), while immediate READ/WRITE utility is only weakly predictable (rho 0.0416/0.0347).
- **Internal generalization (Step C):** Stage 1 survives semantic distance/clusters but not source/dataset shift; Stage 2 remains weak.
- **External transfer (Step D):** Stage 1 is `{d1}` and Stage 2 is `{d2}` under zero external retuning across ChartQA, TextVQA, MMMU-Pro, and POPE.
- **Deployment:** none of these predictability/measurement results alone establishes positive routed generation performance.
"""
    _atomic(output_root / "summaries/predictability_phase_final_summary.md", phase_summary.encode())
    if d1 == "D1-A" and d2 == "D2-A":
        implication = "Keep failure detection as the handoff, but do not use current-state local READ/WRITE utility classification as the default Stage-2 formulation; the next separately authorized experiment should distinguish whether history/counterfactual probes add treatment information."
    elif d1 == "D1-B" and d2 == "D2-A":
        implication = "Treat failure detection as task-family calibrated and abandon current-state local READ/WRITE utility classification as the default Stage-2 formulation; the next separately authorized experiment should distinguish representation shift from missing nonlocal treatment information."
    elif d1 == "D1-C" and d2 == "D2-A":
        implication = "The next method must jointly address robust failure representation and nonlocal intervention selection; first distinguish source-calibration failure from representation failure before adding routing complexity."
    elif d2 == "D2-B":
        implication = "Characterize only the fixed-rule restricted Stage-2 niche before any general routing claim; the next separately authorized experiment should test whether that niche repeats on an independent same-family population."
    else:
        implication = "Audit the unexpected broad external utility signal for parity, leakage, and distribution artifacts before any method claim or deployment experiment."
    implication_md = f"""# One next-method implication

{implication}

Evidence: Step-D categories `{d1}` / `{d2}`, with the exact benchmark metrics and controls in the final summary.

This recommendation does not establish deployment gain, causal mechanism, or that richer Stage-2 information will succeed. It is not authorization to run the next experiment.
"""
    _atomic(output_root / "summaries/next_method_implication.md", implication_md.encode())
    manifest_files = {}
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or "work" in path.relative_to(output_root).parts or path.name == "artifact_manifest.json":
            continue
        relative = str(path.relative_to(output_root))
        manifest_files[relative] = stepd.file_sha256(path)
    artifact = {"schema_version": "predictability_stepD_artifact_manifest_v1", "contract_sha256": contract["contract_sha256"], "stage1_category": d1, "stage2_category": d2, "files": manifest_files}
    artifact["artifact_manifest_sha256"] = stepd.canonical_hash(artifact)
    stepd.atomic_json(output_root / "artifact_manifest.json", artifact)
    print(json.dumps({"complete": True, "contract_sha256": contract["contract_sha256"], "artifact_manifest_sha256": artifact["artifact_manifest_sha256"], "stage1_category": d1, "stage2_category": d2}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    aggregate(arguments.config)


if __name__ == "__main__":
    main()
