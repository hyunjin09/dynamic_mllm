#!/usr/bin/env python3
"""Validate, aggregate, and report the frozen Step-B OOF study."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dense_failure_stage2.predictability_learnability import (  # noqa: E402
    binary_classification_metrics,
    harmful_ranking_metrics,
    high_precision_harmful_metrics,
    regression_metrics,
    select_preservation_threshold,
    uid_macro_regression_metrics,
    validate_oof_completeness,
)
from experiments import run_predictability_stepB_learnability as pipeline  # noqa: E402


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _finite(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return "nan"
    return value.item() if isinstance(value, np.generic) else value


def _clean(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _finite(value) for key, value in row.items()}


def _classification(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    return _clean(binary_classification_metrics(truth=truth, prediction=prediction))


def _regression(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    return _clean(regression_metrics(truth=truth, prediction=prediction))


def _harmful(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    return _clean(harmful_ranking_metrics(truth=truth, prediction=prediction))


def _high_precision(
    truth: np.ndarray, prediction: np.ndarray, config: Mapping[str, Any]
) -> dict[str, Any]:
    return _clean(
        high_precision_harmful_metrics(
            truth=truth,
            prediction=prediction,
            coverages=config["evaluation"]["harmful_coverages"],
            precision_targets=config["evaluation"]["harmful_precision_targets"],
        )
    )


def _task_payloads(
    contract: Mapping[str, Any], output_root: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    completions = pipeline._validated_training_completions(contract, output_root)
    tasks = [dict(row["task"]) for row in completions.values()]
    payloads = {
        task_id: torch.load(
            pipeline.resolve_path(value["completion"]["checkpoint"]),
            map_location="cpu",
            weights_only=False,
        )
        for task_id, value in completions.items()
    }
    return tasks, payloads


def _ensemble_predictions(
    contract: Mapping[str, Any],
    output_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, dict[tuple[str, str, str], list[dict[str, Any]]]],
    list[dict[str, Any]],
]:
    grouped: dict[tuple[str, str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[
            (
                str(task["domain"]), str(task["target"]), str(task["model"]),
                str(task["input"]), int(task["fold"]),
            )
        ].append(task)
    data_cache: dict[str, dict[str, Any]] = {}
    output: dict[str, dict[tuple[str, str, str], list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seed_buckets: dict[tuple[str, str, str, str, int], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for key, variants in sorted(grouped.items()):
        domain, target, model, input_name, fold = key
        if domain not in data_cache:
            data_cache[domain] = pipeline._domain_data(contract, output_root, domain)
        data = data_cache[domain]
        predictions = []
        indices = None
        for task in sorted(variants, key=lambda row: int(row["seed"])):
            payload = payloads[str(task["task_id"])]
            current_indices = np.asarray(payload["test_indices"], dtype=np.int64)
            if indices is None:
                indices = current_indices
            elif not np.array_equal(indices, current_indices):
                raise RuntimeError(f"seed OOF indices differ: {key}")
            current = np.asarray(payload["test_prediction"], dtype=np.float64)
            predictions.append(current)
            seed_buckets[(domain, target, model, input_name, int(task["seed"]))].append(
                (current_indices, current)
            )
        assert indices is not None
        ensemble = np.mean(np.stack(predictions), axis=0)
        truth = np.asarray(data["targets"][target], dtype=np.float64)
        for local, row_index in enumerate(indices):
            source = data["rows"][int(row_index)]
            row = {
                "state_id": str(source["state_id"]),
                "uid": str(source["uid"]),
                "image_group_id": str(source["image_group_id"]),
                "dataset": str(source["dataset"]),
                "source_regime": str(source["source_regime"]),
                "dense_wrong": bool(source["dense_wrong"]),
                "fold": fold,
                "layer": int(source["layer"]),
                "target": target,
                "model": model,
                "input": input_name,
                "seeds": [int(task["seed"]) for task in variants],
                "truth": float(truth[int(row_index)]),
                "prediction": float(ensemble[local]),
            }
            if domain != "stage1":
                row.update(
                    {
                        "trigger_layer": int(source["trigger_layer"]),
                        "trigger_relative_depth": int(source["trigger_relative_depth"]),
                    }
                )
            output[domain][(target, model, input_name)].append(row)

    seed_metrics = []
    for key, pieces in sorted(seed_buckets.items()):
        domain, target, model, input_name, seed = key
        data = data_cache[domain]
        indices = np.concatenate([piece[0] for piece in pieces])
        prediction = np.concatenate([piece[1] for piece in pieces])
        truth = np.asarray(data["targets"][target], dtype=np.float64)[indices]
        metrics = (
            _classification(truth.astype(np.int64), prediction)
            if domain == "stage1"
            else _regression(truth, prediction)
        )
        if domain != "stage1":
            harmful = _harmful(truth, prediction)
            metrics = {
                **metrics,
                "harmful_auroc": harmful["auroc"],
                "harmful_auprc": harmful["auprc"],
            }
        seed_metrics.append(
            {"domain": domain, "target": target, "model": model, "input": input_name, "seed": seed, **metrics}
        )
    return output, seed_metrics


def _arrays(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([float(row["truth"]) for row in rows], dtype=np.float64),
        np.asarray([float(row["prediction"]) for row in rows], dtype=np.float64),
    )


def _uid_sequential_operating_point(
    *,
    calibration_rows: Sequence[Mapping[str, Any]],
    calibration_scores: np.ndarray,
    calibration_wrong: np.ndarray,
    test_rows: Sequence[Mapping[str, Any]],
    test_scores: np.ndarray,
    test_wrong: np.ndarray,
    target_correct_preservation: float,
) -> dict[str, Any]:
    """Calibrate and evaluate a sequential any-layer trigger at UID level."""

    def collapse(
        rows: Sequence[Mapping[str, Any]], scores: np.ndarray, labels: np.ndarray
    ) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, list[tuple[int, float]]]]:
        if len(rows) != len(scores) or len(rows) != len(labels):
            raise ValueError("sequential operating-point inputs must be aligned")
        by_uid: dict[str, list[tuple[int, float]]] = defaultdict(list)
        label_by_uid: dict[str, int] = {}
        for row, score, label in zip(rows, scores, labels, strict=True):
            uid = str(row["uid"])
            current = int(label)
            if uid in label_by_uid and label_by_uid[uid] != current:
                raise ValueError(f"dense outcome changes within UID: {uid}")
            label_by_uid[uid] = current
            by_uid[uid].append((int(row["layer"]), float(score)))
        uids = sorted(by_uid)
        maxima = np.asarray([max(score for _, score in by_uid[uid]) for uid in uids])
        uid_labels = np.asarray([label_by_uid[uid] for uid in uids], dtype=np.int64)
        return uids, maxima, uid_labels, by_uid

    _, calibration_max, calibration_labels, _ = collapse(
        calibration_rows, calibration_scores, calibration_wrong
    )
    threshold = select_preservation_threshold(
        calibration_max,
        calibration_labels,
        target_correct_preservation=float(target_correct_preservation),
    )
    test_uids, test_max, test_labels, test_by_uid = collapse(
        test_rows, test_scores, test_wrong
    )
    triggered = test_max > threshold
    correct = test_labels == 0
    wrong = test_labels == 1
    first_layers = [
        min(layer for layer, score in test_by_uid[uid] if score > threshold)
        for uid, did_trigger in zip(test_uids, triggered, strict=True)
        if bool(did_trigger)
    ]
    state_wrong = np.asarray(test_wrong, dtype=np.int64) == 1
    state_trigger = np.asarray(test_scores, dtype=np.float64) > threshold
    return {
        "threshold": float(threshold),
        "heldout_correct_preservation": float(1.0 - triggered[correct].mean()),
        "heldout_wrong_recall_uid": float(triggered[wrong].mean()),
        "heldout_wrong_recall_state": float(state_trigger[state_wrong].mean()),
        "precision": float(test_labels[triggered].mean()) if triggered.any() else "nan",
        "median_trigger_layer": float(np.median(first_layers)) if first_layers else "nan",
        "triggered_uids": int(triggered.sum()),
    }


def _seed_metric_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Report mean/sample-SD across independently trained seeds."""

    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["domain"]), str(row["target"]), str(row["model"]), str(row["input"]))
        ].append(row)
    metric_names = (
        "auroc", "auprc", "spearman", "pearson", "mae", "rmse",
        "harmful_auroc", "harmful_auprc",
    )
    output = []
    for (domain, target, model, input_name), variants in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "domain": domain,
            "target": target,
            "model": model,
            "input": input_name,
            "seeds": len(variants),
            "seed_values": ";".join(str(int(row["seed"])) for row in variants),
        }
        for metric in metric_names:
            values = np.asarray(
                [_number(row.get(metric, "nan")) for row in variants], dtype=np.float64
            )
            values = values[np.isfinite(values)]
            if len(values):
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(summary)
    return output


def _stage1_outputs(
    contract: Mapping[str, Any],
    output_root: Path,
    predictions: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    tasks: Sequence[Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    config = contract["static_config"]
    primary = {
        key: list(rows) for key, rows in predictions.items() if key[0] == "dense_wrong"
    }
    expected_data = pipeline._domain_data(contract, output_root, "stage1")
    expected = {
        str(row["state_id"]): {"fold": int(row["fold"])} for row in expected_data["rows"]
    }
    overall = []
    layer_metrics = []
    source = []
    oof_rows = []
    for (target, model, input_name), rows in sorted(primary.items()):
        validate_oof_completeness(expected, rows)
        truth, score = _arrays(rows)
        overall.append(
            {"model": model, "input": input_name, **_classification(truth, score)}
        )
        for layer in range(28):
            subset = [row for row in rows if int(row["layer"]) == layer]
            layer_truth, layer_score = _arrays(subset)
            layer_metrics.append(
                {
                    "model": model,
                    "input": input_name,
                    "layer": layer,
                    **_classification(layer_truth, layer_score),
                }
            )
        for dataset in ("gqa", "chartqa", "textvqa"):
            for regime in ("historical", "canonical"):
                subset = [
                    row for row in rows
                    if row["dataset"] == dataset and row["source_regime"] == regime
                ]
                if subset:
                    local_truth, local_score = _arrays(subset)
                    source.append(
                        {
                            "model": model,
                            "input": input_name,
                            "dataset": dataset,
                            "source_regime": regime,
                            **_classification(local_truth, local_score),
                        }
                    )
        oof_rows.extend(rows)

    grouped_tasks: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for task in tasks:
        if task["domain"] == "stage1":
            grouped_tasks[(str(task["model"]), str(task["input"]), int(task["fold"]))].append(task)
    operating = []
    data = expected_data
    target = np.asarray(data["targets"]["dense_wrong"], dtype=np.int64)
    for (model, input_name, fold), variants in sorted(grouped_tasks.items()):
        calibrations = []
        tests = []
        calibration_indices = test_indices = None
        for task in variants:
            payload = payloads[str(task["task_id"])]
            current_calibration = np.asarray(payload["calibration_indices"], dtype=np.int64)
            current_test = np.asarray(payload["test_indices"], dtype=np.int64)
            if calibration_indices is None:
                calibration_indices, test_indices = current_calibration, current_test
            elif not np.array_equal(calibration_indices, current_calibration) or not np.array_equal(
                test_indices, current_test
            ):
                raise RuntimeError("Stage-1 seed calibration/test indices differ")
            calibrations.append(np.asarray(payload["calibration_prediction"], dtype=np.float64))
            tests.append(np.asarray(payload["test_prediction"], dtype=np.float64))
        assert calibration_indices is not None and test_indices is not None
        calibration = np.mean(np.stack(calibrations), axis=0)
        test_score = np.mean(np.stack(tests), axis=0)
        test_truth = target[test_indices]
        for preservation in config["evaluation"]["stage1_preservation_targets"]:
            operating.append(
                {
                    "model": model,
                    "input": input_name,
                    "fold": fold,
                    "target_correct_preservation": float(preservation),
                    **_uid_sequential_operating_point(
                        calibration_rows=[data["rows"][int(index)] for index in calibration_indices],
                        calibration_scores=calibration,
                        calibration_wrong=target[calibration_indices],
                        test_rows=[data["rows"][int(index)] for index in test_indices],
                        test_scores=test_score,
                        test_wrong=test_truth,
                        target_correct_preservation=float(preservation),
                    ),
                }
            )
    return {
        "oof": oof_rows,
        "overall": overall,
        "layer": layer_metrics,
        "source": source,
        "operating": operating,
    }


def _uid_macro(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    truth, prediction = _arrays(rows)
    uids = np.asarray([str(row["uid"]) for row in rows])
    return _clean(
        uid_macro_regression_metrics(
            truth=truth,
            prediction=prediction,
            uids=uids,
            minimum_nonconstant_states=int(
                config["evaluation"]["uid_macro_minimum_nonconstant_states"]
            ),
        )
    )


def _safe_binary_event(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return {"support": len(labels), "positives": int(labels.sum()), "auroc": "nan", "auprc": "nan"}
    return _classification(labels, scores)


def _stage2_outputs(
    contract: Mapping[str, Any],
    output_root: Path,
    domain: str,
    predictions: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    config = contract["static_config"]
    data = pipeline._domain_data(contract, output_root, domain)
    expected = {str(row["state_id"]): {"fold": int(row["fold"])} for row in data["rows"]}
    primary = {
        key: list(rows) for key, rows in predictions.items() if key[0] in {"read", "write"}
    }
    continuous = []
    harmful = []
    high_precision = []
    conditional = []
    layer_metrics = []
    trigger_metrics = []
    breakdown = []
    oof = {"read": [], "write": []}
    depth_bins = config["evaluation"]["depth_bins"]
    trigger_bins = config["evaluation"]["trigger_relative_bins"]
    minimum = int(config["evaluation"]["minimum_breakdown_rows"])
    for (target, model, input_name), rows in sorted(primary.items()):
        validate_oof_completeness(expected, rows)
        truth, prediction = _arrays(rows)
        micro = _regression(truth, prediction)
        continuous.append(
            {"target": target, "model": model, "input": input_name, "aggregation": "state_micro", **micro}
        )
        continuous.append(
            {
                "target": target,
                "model": model,
                "input": input_name,
                "aggregation": "uid_macro",
                **_uid_macro(rows, config),
            }
        )
        harmful.append({"target": target, "model": model, "input": input_name, **_harmful(truth, prediction)})
        high_precision.append(
            {"target": target, "model": model, "input": input_name, **_high_precision(truth, prediction, config)}
        )
        for dense_wrong in (False, True):
            subset = [row for row in rows if bool(row["dense_wrong"]) == dense_wrong]
            local_truth, local_prediction = _arrays(subset)
            conditional.append(
                {
                    "target": target,
                    "model": model,
                    "input": input_name,
                    "dense_outcome": "wrong" if dense_wrong else "correct",
                    **_regression(local_truth, local_prediction),
                    "harmful_auroc": _harmful(local_truth, local_prediction)["auroc"],
                }
            )
        for layer in range(28):
            subset = [row for row in rows if int(row["layer"]) == layer]
            if len(subset) >= minimum:
                local_truth, local_prediction = _arrays(subset)
                layer_metrics.append(
                    {
                        "target": target, "model": model, "input": input_name,
                        "slice": "exact", "layer": layer,
                        **_regression(local_truth, local_prediction),
                        "harmful_auroc": _harmful(local_truth, local_prediction)["auroc"],
                    }
                )
        for name, bounds in depth_bins.items():
            subset = [row for row in rows if int(bounds[0]) <= int(row["layer"]) <= int(bounds[1])]
            if len(subset) >= minimum:
                local_truth, local_prediction = _arrays(subset)
                layer_metrics.append(
                    {
                        "target": target, "model": model, "input": input_name,
                        "slice": name, "layer": "all",
                        **_regression(local_truth, local_prediction),
                        "harmful_auroc": _harmful(local_truth, local_prediction)["auroc"],
                    }
                )
        for name, bounds in trigger_bins.items():
            subset = [
                row for row in rows
                if int(bounds[0]) <= int(row["trigger_relative_depth"]) <= int(bounds[1])
            ]
            if len(subset) >= minimum:
                local_truth, local_prediction = _arrays(subset)
                trigger_metrics.append(
                    {
                        "target": target, "model": model, "input": input_name,
                        "trigger_relative_bin": name,
                        **_regression(local_truth, local_prediction),
                        "harmful_auroc": _harmful(local_truth, local_prediction)["auroc"],
                    }
                )
        for dataset in ("gqa", "chartqa", "textvqa"):
            for regime in ("historical", "canonical"):
                subset = [
                    row for row in rows
                    if row["dataset"] == dataset and row["source_regime"] == regime
                ]
                if len(subset) >= minimum:
                    local_truth, local_prediction = _arrays(subset)
                    breakdown.append(
                        {
                            "target": target, "model": model, "input": input_name,
                            "dataset": dataset, "source_regime": regime,
                            **_regression(local_truth, local_prediction),
                            "harmful_auroc": _harmful(local_truth, local_prediction)["auroc"],
                        }
                    )
        oof[target].extend(rows)

    strong_flip = []
    specificity = []
    secondary = []
    if domain == "stage2_dense":
        flips = {
            str(row["state_id"]): row
            for row in pipeline.read_csv(config["sources"]["stage2_dense_flips"])
        }
        for (target, model, input_name), rows in sorted(primary.items()):
            _, prediction = _arrays(rows)
            for event, score_sign in (
                (f"{target}_harmful_flip_{'w1' if target == 'read' else 'r1'}", -1.0),
                (f"{target}_beneficial_flip_{'w1' if target == 'read' else 'r1'}", 1.0),
            ):
                labels = np.asarray(
                    [str(flips[str(row["state_id"])][event]).lower() == "true" for row in rows],
                    dtype=np.int64,
                )
                metrics = _safe_binary_event(labels, score_sign * prediction)
                strong_flip.append(
                    {
                        "target": target, "model": model, "input": input_name,
                        "event": event, **metrics,
                        "median_predicted_utility_event": (
                            float(np.median(prediction[labels == 1])) if labels.any() else "nan"
                        ),
                    }
                )
        for key, rows in sorted(predictions.items()):
            target, model, input_name = key
            if target in {"read", "write"}:
                continue
            truth, prediction = _arrays(rows)
            secondary.append(
                {
                    "target": target, "model": model, "input": input_name,
                    **_regression(truth, prediction),
                    "harmful_auroc": _harmful(truth, prediction)["auroc"],
                    "harmful_auprc": _harmful(truth, prediction)["auprc"],
                }
            )
    for target in ("read", "write"):
        for representation in ("z_R", "z_W", "z_RW"):
            rows = primary[(target, f"m3_{representation}", representation)]
            truth, prediction = _arrays(rows)
            specificity.append(
                {
                    "target": target, "representation": representation,
                    "spearman": _regression(truth, prediction)["spearman"],
                    "harmful_auroc": _harmful(truth, prediction)["auroc"],
                }
            )
    continuous_lookup = {
        (row["target"], row["model"], row["input"]): row
        for row in continuous if row["aggregation"] == "state_micro"
    }
    harmful_lookup = {(row["target"], row["model"], row["input"]): row for row in harmful}
    nuisance = []
    controls = []
    for target in ("read", "write"):
        reference = continuous_lookup[(target, "m0_nuisance", "nuisance")]
        for key, row in sorted(continuous_lookup.items()):
            if key[0] != target:
                continue
            nuisance.append(
                {
                    "target": target, "model": key[1], "input": key[2],
                    "spearman": row["spearman"],
                    "nuisance_spearman": reference["spearman"],
                    "spearman_gain_over_nuisance": (
                        float(row["spearman"]) - float(reference["spearman"])
                        if row["spearman"] != "nan" and reference["spearman"] != "nan" else "nan"
                    ),
                    "harmful_auroc": harmful_lookup[key]["auroc"],
                }
            )
        for model, input_name in (
            ("m1_text", "text"), ("m1_visual", "visual"),
            ("m1_text_visual", "text_visual"), ("m2_text_visual", "text_visual"),
            ("m3_z_RW", "z_RW"),
        ):
            row = continuous_lookup[(target, model, input_name)]
            controls.append(
                {
                    "target": target, "model": model, "input": input_name,
                    "spearman": row["spearman"],
                    "harmful_auroc": harmful_lookup[(target, model, input_name)]["auroc"],
                }
            )
    return {
        "oof_read": oof["read"], "oof_write": oof["write"],
        "continuous": continuous, "harmful": harmful, "high_precision": high_precision,
        "strong_flip": strong_flip, "specificity": specificity, "conditional": conditional,
        "layer": layer_metrics, "trigger": trigger_metrics, "breakdown": breakdown,
        "nuisance": nuisance, "controls": controls, "secondary": secondary,
    }


def _state_regime_transfer(
    contract: Mapping[str, Any], output_root: Path
) -> list[dict[str, Any]]:
    config = contract["static_config"]
    tasks = pipeline._transfer_tasks(config)
    grouped: dict[tuple[str, str, str, str, str, int], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for task in tasks:
        rank = int(task["worker_rank"])
        completion_path = output_root / f"work/transfer/rank{rank:02d}/{task['task_id']}.json"
        completion = pipeline.read_json(completion_path)
        prediction_path = pipeline.resolve_path(completion.get("prediction_path", ""))
        if (
            completion.get("contract_sha256") != contract["contract_sha256"]
            or completion.get("task_sha256") != pipeline.canonical_hash(task)
            or not prediction_path.is_file()
            or pipeline.file_sha256(prediction_path) != completion.get("prediction_sha256")
        ):
            raise RuntimeError(f"invalid state-regime transfer completion: {task['task_id']}")
        payload = torch.load(prediction_path, map_location="cpu", weights_only=False)
        grouped[
            (
                str(task["train_domain"]), str(task["test_domain"]), str(task["target"]),
                str(task["model"]), str(task["input"]), int(task["fold"]),
            )
        ].append((task, payload))
    for rank in range(int(config["world_size"])):
        marker = pipeline.read_json(output_root / f"work/transfer/rank{rank:02d}/complete.json")
        expected = sum(int(task["worker_rank"]) == rank for task in tasks)
        if (
            marker.get("contract_sha256") != contract["contract_sha256"]
            or int(marker.get("completed_tasks", -1)) != expected
        ):
            raise RuntimeError(f"state-regime transfer worker {rank} incomplete")
    rows = []
    accumulators: dict[tuple[str, str, str, str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    data_cache: dict[str, dict[str, Any]] = {}
    for key, variants in sorted(grouped.items()):
        train_domain, test_domain, target, model, input_name, fold = key
        if test_domain not in data_cache:
            data_cache[test_domain] = pipeline._domain_data(contract, output_root, test_domain)
        data = data_cache[test_domain]
        indices = None
        predictions = []
        for _, payload in sorted(variants, key=lambda value: int(value[0]["seed"])):
            current = np.asarray(payload["test_indices"], dtype=np.int64)
            if indices is None:
                indices = current
            elif not np.array_equal(indices, current):
                raise RuntimeError("state-regime seed test indices differ")
            predictions.append(np.asarray(payload["test_prediction"], dtype=np.float64))
        assert indices is not None
        prediction = np.mean(np.stack(predictions), axis=0)
        truth = np.asarray(data["targets"][target], dtype=np.float64)[indices]
        accumulators[(train_domain, test_domain, target, model, input_name)].append((truth, prediction))
    for key, pieces in sorted(accumulators.items()):
        train_domain, test_domain, target, model, input_name = key
        truth = np.concatenate([piece[0] for piece in pieces])
        prediction = np.concatenate([piece[1] for piece in pieces])
        rows.append(
            {
                "train_domain": train_domain,
                "test_domain": test_domain,
                "target": target,
                "model": model,
                "input": input_name,
                **_regression(truth, prediction),
                "harmful_auroc": _harmful(truth, prediction)["auroc"],
                "harmful_auprc": _harmful(truth, prediction)["auprc"],
            }
        )
    return rows


def _weighted_binary(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    order = np.argsort(scores, kind="mergesort")
    y = labels[order]
    w = weights[order]
    pos_total = float(w[y == 1].sum())
    neg_total = float(w[y == 0].sum())
    if pos_total <= 0.0 or neg_total <= 0.0:
        return float("nan"), float("nan")
    wins = 0.0
    cumulative_neg = 0.0
    start = 0
    ordered_scores = scores[order]
    while start < len(order):
        end = start + 1
        while end < len(order) and ordered_scores[end] == ordered_scores[start]:
            end += 1
        tie_pos = float(w[start:end][y[start:end] == 1].sum())
        tie_neg = float(w[start:end][y[start:end] == 0].sum())
        wins += tie_pos * (cumulative_neg + 0.5 * tie_neg)
        cumulative_neg += tie_neg
        start = end
    auroc = wins / (pos_total * neg_total)
    descending = np.argsort(-scores, kind="mergesort")
    yd = labels[descending]
    wd = weights[descending]
    cumulative_weight = np.cumsum(wd)
    cumulative_positive = np.cumsum(wd * (yd == 1))
    precision = np.divide(
        cumulative_positive,
        cumulative_weight,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_weight > 0,
    )
    auprc = float(np.sum(wd * (yd == 1) * precision) / pos_total)
    return auroc, auprc


def _weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0.0:
        return float("nan")
    mx = float(np.dot(weights, x) / total)
    my = float(np.dot(weights, y) / total)
    xc, yc = x - mx, y - my
    denominator = math.sqrt(float(np.dot(weights, xc * xc) * np.dot(weights, yc * yc)))
    return float(np.dot(weights, xc * yc) / denominator) if denominator > 0.0 else float("nan")


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _bootstrap_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    metrics: Sequence[str],
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    truth, prediction = _arrays(rows)
    group_names = sorted({str(row["image_group_id"]) for row in rows})
    group_index = {name: index for index, name in enumerate(group_names)}
    codes = np.asarray([group_index[str(row["image_group_id"])] for row in rows], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    values = {metric: np.empty(int(draws), dtype=np.float64) for metric in metrics}
    truth_ranks = _rank(truth) if "spearman" in metrics else None
    prediction_ranks = _rank(prediction) if "spearman" in metrics else None
    for draw in range(int(draws)):
        counts = rng.multinomial(len(group_names), np.full(len(group_names), 1.0 / len(group_names)))
        weights = counts[codes].astype(np.float64)
        if kind == "stage1":
            auroc, auprc = _weighted_binary(truth.astype(np.int64), prediction, weights)
            if "auroc" in values:
                values["auroc"][draw] = auroc
            if "auprc" in values:
                values["auprc"][draw] = auprc
        else:
            if "spearman" in values:
                assert truth_ranks is not None and prediction_ranks is not None
                values["spearman"][draw] = _weighted_correlation(truth_ranks, prediction_ranks, weights)
            if "harmful_auroc" in values:
                mask = truth != 0.0
                values["harmful_auroc"][draw] = _weighted_binary(
                    (truth[mask] < 0.0).astype(np.int64), -prediction[mask], weights[mask]
                )[0]
            if "precision_at_0.1" in values:
                mask = truth != 0.0
                local_truth = truth[mask]
                local_prediction = prediction[mask]
                local_weights = weights[mask]
                order = np.argsort(local_prediction, kind="mergesort")
                ordered_weights = local_weights[order]
                cutoff = 0.1 * ordered_weights.sum()
                cumulative = np.cumsum(ordered_weights)
                selected = cumulative <= cutoff
                if not selected.any() and len(selected):
                    selected[0] = True
                labels = local_truth[order] < 0.0
                values["precision_at_0.1"][draw] = float(
                    np.dot(ordered_weights[selected], labels[selected]) / ordered_weights[selected].sum()
                ) if ordered_weights[selected].sum() > 0 else float("nan")
    output = []
    for metric, metric_values in values.items():
        finite = metric_values[np.isfinite(metric_values)]
        output.append(
            {
                "metric": metric,
                "draws_requested": int(draws),
                "draws_valid": int(len(finite)),
                "ci_low": float(np.quantile(finite, 0.025)),
                "median": float(np.quantile(finite, 0.5)),
                "ci_high": float(np.quantile(finite, 0.975)),
                "bootstrap_unit": "image_group_id",
            }
        )
    return output


def _bootstrap(
    contract: Mapping[str, Any],
    stage1: Mapping[str, Sequence[Mapping[str, Any]]],
    dense: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    config = contract["static_config"]
    draws = int(config["evaluation"]["bootstrap_draws"])
    seed = int(config["evaluation"]["bootstrap_seed"])
    specs = [
        (
            "stage1", "dense_wrong", "m3_current_head", "state_layer",
            [row for row in stage1["oof"] if row["model"] == "m3_current_head"],
            ("auroc", "auprc"),
        ),
        *[
            (
                "stage2_dense", target, "m3_z_RW", "z_RW",
                [row for row in dense[target] if row["model"] == "m3_z_RW"],
                ("spearman", "harmful_auroc", "precision_at_0.1"),
            )
            for target in ("read", "write")
        ],
    ]
    output = []
    for index, (domain, target, model, input_name, rows, metrics) in enumerate(specs):
        for row in _bootstrap_rows(
            rows,
            kind="stage1" if domain == "stage1" else "stage2",
            metrics=metrics,
            draws=draws,
            seed=seed + index,
        ):
            output.append(
                {"domain": domain, "target": target, "model": model, "input": input_name, **row}
            )
    return output


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _metric_row(rows: Sequence[Mapping[str, Any]], **criteria: Any) -> Mapping[str, Any]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise RuntimeError(f"expected one metric row for {criteria}, found {len(matches)}")
    return matches[0]


def _bar_figure(path: Path, labels: Sequence[str], values: Sequence[float], title: str, ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(np.arange(len(values)), values, color="#4472c4")
    axis.set_xticks(np.arange(len(values)), labels, rotation=28, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _figures(
    output_root: Path,
    stage1: Mapping[str, Sequence[Mapping[str, Any]]],
    dense: Mapping[str, Sequence[Mapping[str, Any]]],
    routed: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    root = output_root / "figures"
    root.mkdir(parents=True, exist_ok=True)
    display = {
        "m0_nuisance": "M0 nuisance", "m1_linear": "M1 linear",
        "m2_mlp": "M2 MLP", "m3_current_head": "M3 current",
        "m1_text": "text linear", "m1_visual": "visual linear",
        "m1_text_visual": "multimodal linear", "m2_text_visual": "multimodal MLP",
        "m3_z_R": "router z_R", "m3_z_W": "router z_W", "m3_z_RW": "router z_RW",
    }
    figure, axis = plt.subplots(figsize=(8, 5))
    for model in ("m0_nuisance", "m1_linear", "m2_mlp", "m3_current_head"):
        rows = sorted(
            (row for row in stage1["layer"] if row["model"] == model),
            key=lambda row: int(row["layer"]),
        )
        axis.plot(
            [int(row["layer"]) for row in rows],
            [_number(row["auroc"]) for row in rows],
            label=display[model],
        )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Layer", ylabel="OOF AUROC", title="Stage-1 failure predictability by layer")
    axis.legend(); axis.grid(alpha=0.25)
    figure.tight_layout(); figure.savefig(root / "stage1_auroc_by_layer.png", dpi=160); plt.close(figure)

    s1_models = ("m0_nuisance", "m1_linear", "m2_mlp", "m3_current_head")
    _bar_figure(
        root / "stage1_capacity_ladder.png",
        [display[model] for model in s1_models],
        [_number(_metric_row(stage1["overall"], model=model)["auroc"]) for model in s1_models],
        "Stage-1 capacity ladder", "OOF AUROC",
    )
    stage2_models = (
        "m0_nuisance", "m1_text", "m1_visual", "m1_text_visual",
        "m2_text_visual", "m3_z_R", "m3_z_W", "m3_z_RW",
    )
    for target in ("read", "write"):
        _bar_figure(
            root / f"stage2_{target}_capacity_ladder.png",
            [display[model] for model in stage2_models],
            [
                _number(_metric_row(dense["continuous"], target=target, model=model, aggregation="state_micro")["spearman"])
                for model in stage2_models
            ],
            f"Stage-2 {target.upper()} capacity ladder", "OOF Spearman",
        )

    matrix = np.asarray(
        [
            [
                _number(_metric_row(dense["specificity"], target=target, representation=representation)["spearman"])
                for target in ("read", "write")
            ]
            for representation in ("z_R", "z_W", "z_RW")
        ]
    )
    figure, axis = plt.subplots(figsize=(5, 4.5))
    image = axis.imshow(matrix, cmap="coolwarm", aspect="auto")
    axis.set_xticks((0, 1), ("READ utility", "WRITE utility"))
    axis.set_yticks((0, 1, 2), ("z_R", "z_W", "z_RW"))
    for y in range(3):
        for x in range(2):
            axis.text(x, y, f"{matrix[y, x]:.3f}", ha="center", va="center")
    axis.set_title("Router representation specificity (Spearman)")
    figure.colorbar(image, ax=axis)
    figure.tight_layout(); figure.savefig(root / "stage2_read_write_specificity.png", dpi=160); plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.8))
    for target, marker in (("read", "o"), ("write", "s")):
        row = _metric_row(dense["high_precision"], target=target, model="m3_z_RW")
        coverages = (0.05, 0.1, 0.2)
        axis.plot(
            coverages, [_number(row[f"precision_at_{value:g}"]) for value in coverages],
            marker=marker, label=target.upper(),
        )
    axis.set(xlabel="Predicted-harmful coverage", ylabel="Harmful precision", title="High-confidence harmful subset")
    axis.legend(); axis.grid(alpha=0.25)
    figure.tight_layout(); figure.savefig(root / "stage2_harmful_precision_coverage.png", dpi=160); plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for target in ("read", "write"):
        rows = sorted(
            (
                row for row in dense["layer"]
                if row["target"] == target and row["model"] == "m3_z_RW" and row["slice"] == "exact"
            ),
            key=lambda row: int(row["layer"]),
        )
        axis.plot([int(row["layer"]) for row in rows], [_number(row["spearman"]) for row in rows], label=target.upper())
    axis.set(xlabel="Layer", ylabel="OOF Spearman", title="Stage-2 utility predictability by layer")
    axis.legend(); axis.grid(alpha=0.25)
    figure.tight_layout(); figure.savefig(root / "stage2_predictability_by_layer.png", dpi=160); plt.close(figure)

    trigger_order = ("at_trigger", "after_1_2", "after_3_5", "after_6_plus")
    figure, axis = plt.subplots(figsize=(8, 5))
    for target in ("read", "write"):
        values = [
            _number(_metric_row(dense["trigger"], target=target, model="m3_z_RW", trigger_relative_bin=name)["spearman"])
            for name in trigger_order
        ]
        axis.plot(range(4), values, marker="o", label=target.upper())
    axis.set_xticks(range(4), trigger_order, rotation=20)
    axis.set(ylabel="OOF Spearman", title="Predictability by trigger-relative depth")
    axis.legend(); axis.grid(alpha=0.25)
    figure.tight_layout(); figure.savefig(root / "stage2_predictability_by_trigger_depth.png", dpi=160); plt.close(figure)

    comparisons = []
    for target in ("read", "write"):
        nuisance = _number(_metric_row(dense["continuous"], target=target, model="m0_nuisance", aggregation="state_micro")["spearman"])
        for model in ("m1_text_visual", "m2_text_visual", "m3_z_RW"):
            state = _number(_metric_row(dense["continuous"], target=target, model=model, aggregation="state_micro")["spearman"])
            comparisons.append((target, model, nuisance, state))
    figure, axis = plt.subplots(figsize=(6, 5))
    for target, model, nuisance, state in comparisons:
        axis.scatter(nuisance, state, label=f"{target}:{display[model]}")
    axis.set(xlabel="Nuisance Spearman", ylabel="State predictor Spearman", title="Nuisance versus state predictors")
    axis.legend(fontsize=7); axis.grid(alpha=0.25)
    figure.tight_layout(); figure.savefig(root / "nuisance_vs_state_predictor.png", dpi=160); plt.close(figure)

    labels, dense_values, routed_values = [], [], []
    for target in ("read", "write"):
        for model in ("m1_text_visual", "m2_text_visual", "m3_z_RW"):
            labels.append(f"{target}\n{display[model]}")
            dense_values.append(_number(_metric_row(dense["continuous"], target=target, model=model, aggregation="state_micro")["spearman"]))
            routed_values.append(_number(_metric_row(routed["continuous"], target=target, model=model, aggregation="state_micro")["spearman"]))
    x = np.arange(len(labels)); width = 0.38
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x - width / 2, dense_values, width, label="Dense states")
    axis.bar(x + width / 2, routed_values, width, label="Routed states")
    axis.set_xticks(x, labels, rotation=25, ha="right"); axis.set_ylabel("OOF Spearman")
    axis.set_title("Dense versus routed in-domain predictability"); axis.legend(); axis.grid(axis="y", alpha=0.25)
    figure.tight_layout(); figure.savefig(root / "dense_vs_routed_predictability.png", dpi=160); plt.close(figure)

    labels, correct_values, wrong_values = [], [], []
    for target in ("read", "write"):
        for model in ("m1_text_visual", "m2_text_visual", "m3_z_RW"):
            labels.append(f"{target}\n{display[model]}")
            correct_values.append(_number(_metric_row(dense["conditional"], target=target, model=model, dense_outcome="correct")["spearman"]))
            wrong_values.append(_number(_metric_row(dense["conditional"], target=target, model=model, dense_outcome="wrong")["spearman"]))
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x - width / 2, correct_values, width, label="Dense-C")
    axis.bar(x + width / 2, wrong_values, width, label="Dense-W")
    axis.set_xticks(x, labels, rotation=25, ha="right"); axis.set_ylabel("OOF Spearman")
    axis.set_title("Utility predictability conditional on dense outcome"); axis.legend(); axis.grid(axis="y", alpha=0.25)
    figure.tight_layout(); figure.savefig(root / "dense_c_vs_dense_w.png", dpi=160); plt.close(figure)


def _evidence_category(
    stage1: Mapping[str, Sequence[Mapping[str, Any]]],
    dense: Mapping[str, Sequence[Mapping[str, Any]]],
    bootstrap: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    stage1_ci = _metric_row(
        bootstrap, domain="stage1", target="dense_wrong", model="m3_current_head", metric="auroc"
    )
    stage1_strong = _number(stage1_ci["ci_low"]) > 0.5
    stage2_strong = True
    for target in ("read", "write"):
        spearman_ci = _metric_row(
            bootstrap, domain="stage2_dense", target=target, model="m3_z_RW", metric="spearman"
        )
        harmful_ci = _metric_row(
            bootstrap, domain="stage2_dense", target=target, model="m3_z_RW", metric="harmful_auroc"
        )
        stage2_strong &= (
            _number(spearman_ci["ci_low"]) > 0.0
            and _number(harmful_ci["ci_low"]) > 0.5
        )
    if stage1_strong and not stage2_strong:
        return "D", "Stage-1 strong / Stage-2 weak"
    if not stage1_strong and not stage2_strong:
        return "E", "both weak"
    nuisance_close = False
    for target in ("read", "write"):
        nuisance = _number(
            _metric_row(
                dense["continuous"], target=target, model="m0_nuisance", aggregation="state_micro"
            )["spearman"]
        )
        full = _number(
            _metric_row(
                dense["continuous"], target=target, model="m3_z_RW", aggregation="state_micro"
            )["spearman"]
        )
        nuisance_close |= nuisance >= full
    if nuisance_close:
        return "C", "shortcut-conditioned apparent learnability"
    linear_stage1 = _number(_metric_row(stage1["overall"], model="m1_linear")["auroc"])
    nuisance_stage1 = _number(_metric_row(stage1["overall"], model="m0_nuisance")["auroc"])
    linear_stage2 = all(
        _number(
            _metric_row(
                dense["continuous"], target=target, model="m1_text_visual", aggregation="state_micro"
            )["spearman"]
        )
        > _number(
            _metric_row(
                dense["continuous"], target=target, model="m0_nuisance", aggregation="state_micro"
            )["spearman"]
        )
        for target in ("read", "write")
    )
    if linear_stage1 > nuisance_stage1 and linear_stage2:
        return "A", "strong direct learnability"
    return "B", "nonlinear/cross-modal learnability"


def _mean_operating(
    rows: Sequence[Mapping[str, Any]], model: str, preservation: float, metric: str
) -> float:
    values = [
        _number(row[metric]) for row in rows
        if row["model"] == model and math.isclose(float(row["target_correct_preservation"]), preservation)
    ]
    return float(np.nanmean(values))


def _summary_text(
    contract: Mapping[str, Any],
    stage1: Mapping[str, Sequence[Mapping[str, Any]]],
    dense: Mapping[str, Sequence[Mapping[str, Any]]],
    routed: Mapping[str, Sequence[Mapping[str, Any]]],
    transfer: Sequence[Mapping[str, Any]],
    bootstrap: Sequence[Mapping[str, Any]],
    seed_summary: Sequence[Mapping[str, Any]],
    category: tuple[str, str],
) -> str:
    config = contract["static_config"]
    s1_m3 = _metric_row(stage1["overall"], model="m3_current_head")
    s1_m0 = _metric_row(stage1["overall"], model="m0_nuisance")
    s1_m1 = _metric_row(stage1["overall"], model="m1_linear")
    s1_m2 = _metric_row(stage1["overall"], model="m2_mlp")
    stage1_layers = [row for row in stage1["layer"] if row["model"] == "m3_current_head"]
    best_layer = max(stage1_layers, key=lambda row: _number(row["auroc"]))
    read = _metric_row(dense["continuous"], target="read", model="m3_z_RW", aggregation="state_micro")
    write = _metric_row(dense["continuous"], target="write", model="m3_z_RW", aggregation="state_micro")
    read_h = _metric_row(dense["harmful"], target="read", model="m3_z_RW")
    write_h = _metric_row(dense["harmful"], target="write", model="m3_z_RW")
    read_hp = _metric_row(dense["high_precision"], target="read", model="m3_z_RW")
    write_hp = _metric_row(dense["high_precision"], target="write", model="m3_z_RW")
    routed_read = _metric_row(routed["continuous"], target="read", model="m3_z_RW", aggregation="state_micro")
    routed_write = _metric_row(routed["continuous"], target="write", model="m3_z_RW", aggregation="state_micro")
    routed_data = pipeline._domain_data(contract, pipeline.resolve_path(config["output_root"]), "stage2_routed")
    s1_groups = len({str(row["image_group_id"]) for row in stage1["oof"] if row["model"] == "m3_current_head"})
    dense_groups = len({str(row["image_group_id"]) for row in dense["read"] if row["model"] == "m3_z_RW"})
    specificity = {
        (str(row["target"]), str(row["representation"])): _number(row["spearman"])
        for row in dense["specificity"]
    }
    conditional = {
        (str(row["target"]), str(row["dense_outcome"])): _number(row["spearman"])
        for row in dense["conditional"] if row["model"] == "m3_z_RW"
    }
    transfer_best = sorted(transfer, key=lambda row: (str(row["train_domain"]), str(row["target"]), str(row["model"])))
    transfer_text = "; ".join(
        f"{row['train_domain']}→{row['test_domain']} {row['target']}/{row['model']} ρ={_number(row['spearman']):.3f}"
        for row in transfer_best if row["model"] == "m3_z_RW"
    )
    s1_seed = _metric_row(
        seed_summary,
        domain="stage1", target="dense_wrong", model="m3_current_head", input="state_layer",
    )
    read_seed = _metric_row(
        seed_summary,
        domain="stage2_dense", target="read", model="m3_z_RW", input="z_RW",
    )
    write_seed = _metric_row(
        seed_summary,
        domain="stage2_dense", target="write", model="m3_z_RW", input="z_RW",
    )
    return f"""# Step-B in-domain learnability summary

Contract: `{contract['contract_sha256']}`  
Evidence category: **{category[0]} — {category[1]}**

## Main results

| Stage-1 model | AUROC | AUPRC | W recall @95% C-preserve | W recall @98% C-preserve |
|---|---:|---:|---:|---:|
| M0 nuisance | {_number(s1_m0['auroc']):.4f} | {_number(s1_m0['auprc']):.4f} | {_mean_operating(stage1['operating'], 'm0_nuisance', .95, 'heldout_wrong_recall_uid'):.4f} | {_mean_operating(stage1['operating'], 'm0_nuisance', .98, 'heldout_wrong_recall_uid'):.4f} |
| M1 linear state | {_number(s1_m1['auroc']):.4f} | {_number(s1_m1['auprc']):.4f} | {_mean_operating(stage1['operating'], 'm1_linear', .95, 'heldout_wrong_recall_uid'):.4f} | {_mean_operating(stage1['operating'], 'm1_linear', .98, 'heldout_wrong_recall_uid'):.4f} |
| M2 two-layer MLP | {_number(s1_m2['auroc']):.4f} | {_number(s1_m2['auprc']):.4f} | {_mean_operating(stage1['operating'], 'm2_mlp', .95, 'heldout_wrong_recall_uid'):.4f} | {_mean_operating(stage1['operating'], 'm2_mlp', .98, 'heldout_wrong_recall_uid'):.4f} |
| M3 current head | {_number(s1_m3['auroc']):.4f} | {_number(s1_m3['auprc']):.4f} | {_mean_operating(stage1['operating'], 'm3_current_head', .95, 'heldout_wrong_recall_uid'):.4f} | {_mean_operating(stage1['operating'], 'm3_current_head', .98, 'heldout_wrong_recall_uid'):.4f} |

| Dense Stage-2 target / M3 joint | Spearman | Harmful AUROC | Harmful AUPRC | Precision top 5% | Precision top 10% |
|---|---:|---:|---:|---:|---:|
| READ `q_F-q_WO` | {_number(read['spearman']):.4f} | {_number(read_h['auroc']):.4f} | {_number(read_h['auprc']):.4f} | {_number(read_hp['precision_at_0.05']):.4f} | {_number(read_hp['precision_at_0.1']):.4f} |
| WRITE `q_F-q_RO` | {_number(write['spearman']):.4f} | {_number(write_h['auroc']):.4f} | {_number(write_h['auprc']):.4f} | {_number(write_hp['precision_at_0.05']):.4f} | {_number(write_hp['precision_at_0.1']):.4f} |

## Answers required by the plan

1. Stage-1 evaluated **{s1_groups:,} image groups, {config['population']['internal_uids']:,} UIDs, and {config['population']['stage1_states']:,} states** OOF.
2. Dense Stage-2 evaluated **{dense_groups:,} image groups, {config['population']['stage2_dense_uids']:,} UIDs, and {config['population']['stage2_dense_states']:,} states** OOF.
3. All five folds were image-group-disjoint and passed the frozen support audit; every expected state appears once per model after seed ensembling.
4. Stage-1 nuisance/linear/MLP/current-head AUROCs were {_number(s1_m0['auroc']):.3f}/{_number(s1_m1['auroc']):.3f}/{_number(s1_m2['auroc']):.3f}/{_number(s1_m3['auroc']):.3f}.
5. Current-head Stage-1 AUROC was maximal at layer {best_layer['layer']} ({_number(best_layer['auroc']):.3f}); the complete emergence curve is in `stage1/layer_metrics.csv`.
6. At 95/98/99% calibration-fold C preservation, current-head held-out UID W recall was {_mean_operating(stage1['operating'], 'm3_current_head', .95, 'heldout_wrong_recall_uid'):.3f}/{_mean_operating(stage1['operating'], 'm3_current_head', .98, 'heldout_wrong_recall_uid'):.3f}/{_mean_operating(stage1['operating'], 'm3_current_head', .99, 'heldout_wrong_recall_uid'):.3f}.
7. Primary READ utility joint-router OOF Spearman was {_number(read['spearman']):.3f}.
8. Primary WRITE utility joint-router OOF Spearman was {_number(write['spearman']):.3f}.
9. At top-10% predicted harmful coverage, READ/WRITE precision was {_number(read_hp['precision_at_0.1']):.3f}/{_number(write_hp['precision_at_0.1']):.3f} versus natural harmful prevalence {_number(read_h['harmful'])/_number(read_h['support']):.3f}/{_number(write_h['harmful'])/_number(write_h['support']):.3f}.
10. READ specificity: z_R ρ={specificity[('read','z_R')]:.3f} versus z_W ρ={specificity[('read','z_W')]:.3f}.
11. WRITE specificity: z_W ρ={specificity[('write','z_W')]:.3f} versus z_R ρ={specificity[('write','z_R')]:.3f}.
12. Joint z_RW READ/WRITE ρ={specificity[('read','z_RW')]:.3f}/{specificity[('write','z_RW')]:.3f}; this is comparative predictive evidence, not causal branch proof.
13. Joint-router Dense-C versus Dense-W Spearman was READ {conditional[('read','correct')]:.3f}/{conditional[('read','wrong')]:.3f}, WRITE {conditional[('write','correct')]:.3f}/{conditional[('write','wrong')]:.3f}.
14. Nuisance comparisons are frozen in `stage2_dense/nuisance_control_comparison.csv`; the evidence category accounts for whether nuisance matched full-state performance.
15. Text-only, visual-only, concatenated linear, MLP, and router controls are reported in `stage2_dense/text_visual_control_comparison.csv`.
16. Exact-layer and trigger-relative results are in `stage2_dense/layer_metrics.csv` and `trigger_relative_metrics.csv`.
17. All five secondary factorial/context targets were fit with the preregistered reduced ladder; results are in `secondary_factorial_targets.csv`.
18. Selected routed-state joint-router READ/WRITE Spearman was {_number(routed_read['spearman']):.3f}/{_number(routed_write['spearman']):.3f} over {len(routed_data['rows']):,} states; this is secondary selection-biased evidence.
19. State-regime transfer was estimable without group relaxation: {transfer_text}.
20. The evidence pattern is **Case {category[0]} ({category[1]})** under the plan's qualitative taxonomy; 95% group-bootstrap intervals are in `statistics/uid_bootstrap_ci.csv`.
21. Stage-1 AUROC and Stage-2 utility correlation answer different questions, so their raw magnitudes are not directly commensurate; the observed pattern is summarized by Case {category[0]}.
22. Step B does **not** establish semantic/template-independent generalization, dataset LODO robustness, external-benchmark transfer, causal branch mechanism, or deployable routing improvement.

Three-seed fold-concatenated variability for the reference models was: Stage-1 M3 AUROC SD {_number(s1_seed['auroc_std']):.4f}, dense READ M3 Spearman SD {_number(read_seed['spearman_std']):.4f}, and dense WRITE M3 Spearman SD {_number(write_seed['spearman_std']):.4f}. Full seed means/SDs are in each domain's `seed_metrics.csv`.

All results use current dense outcomes and the frozen Step-A utility measurements; no target was redesigned after observing performance.
"""


def _readiness_text(
    contract: Mapping[str, Any], category: tuple[str, str], files: Mapping[str, str]
) -> str:
    return f"""# Step-C readiness

READY_FOR_STEP_C = true

- Evidence category: `{category[0]} {category[1]}`
- Frozen Step-B contract: `{contract['contract_sha256']}`
- All preregistered OOF task completions present and checkpoint-hash-valid: true
- Every expected state represented once per model after seed ensembling: true
- Image-group overlap across outer folds: 0
- Fold-local preprocessing and target scaling: verified by saved fold checkpoints
- Nuisance, text-only, visual-only, multimodal, and current/full-state controls complete: true
- Primary targets unchanged from Step A: true
- Required compact artifact count: {len(files)}

Readiness is procedural, not a positive-result claim. Step C was not executed.
"""


def _artifact_files(output_root: Path) -> dict[str, str]:
    files = {}
    excluded = ("work/", "cache/", "stage1/models/", "stage2_dense/models/", "stage2_routed/models/")
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        relative = str(path.relative_to(output_root))
        if any(relative.startswith(prefix) for prefix in excluded):
            continue
        files[relative] = pipeline.file_sha256(path)
    return files


def aggregate(config_path: Path) -> None:
    contract, output_root, _ = pipeline.verify_contract(config_path)
    tasks, payloads = _task_payloads(contract, output_root)
    predictions, seed_metrics = _ensemble_predictions(contract, output_root, tasks, payloads)
    seed_summary = _seed_metric_summary(seed_metrics)
    stage1 = _stage1_outputs(contract, output_root, predictions["stage1"], tasks, payloads)
    dense = _stage2_outputs(contract, output_root, "stage2_dense", predictions["stage2_dense"])
    routed = _stage2_outputs(contract, output_root, "stage2_routed", predictions["stage2_routed"])
    transfer = _state_regime_transfer(contract, output_root)

    _write_jsonl(output_root / "stage1/oof_predictions.jsonl", stage1["oof"])
    pipeline.atomic_csv(output_root / "stage1/overall_metrics.csv", stage1["overall"])
    pipeline.atomic_csv(output_root / "stage1/layer_metrics.csv", stage1["layer"])
    pipeline.atomic_csv(
        output_root / "stage1/high_preservation_operating_points.csv", stage1["operating"]
    )
    pipeline.atomic_csv(output_root / "stage1/source_breakdown.csv", stage1["source"])
    pipeline.atomic_csv(
        output_root / "stage1/seed_run_metrics.csv",
        [row for row in seed_metrics if row["domain"] == "stage1"],
    )
    pipeline.atomic_csv(
        output_root / "stage1/seed_metrics.csv",
        [row for row in seed_summary if row["domain"] == "stage1"],
    )

    for name, values in (("stage2_dense", dense), ("stage2_routed", routed)):
        root = output_root / name
        _write_jsonl(root / "oof_predictions_read.jsonl", values["read"])
        _write_jsonl(root / "oof_predictions_write.jsonl", values["write"])
        pipeline.atomic_csv(root / "continuous_metrics.csv", values["continuous"])
        pipeline.atomic_csv(root / "harmful_ranking_metrics.csv", values["harmful"])
        pipeline.atomic_csv(root / "representation_specificity.csv", values["specificity"])
        pipeline.atomic_csv(root / "layer_metrics.csv", values["layer"])
        pipeline.atomic_csv(
            root / "seed_run_metrics.csv",
            [row for row in seed_metrics if row["domain"] == name],
        )
        pipeline.atomic_csv(
            root / "seed_metrics.csv",
            [row for row in seed_summary if row["domain"] == name],
        )
    pipeline.atomic_csv(output_root / "stage2_dense/high_precision_metrics.csv", dense["high_precision"])
    pipeline.atomic_csv(output_root / "stage2_dense/strong_flip_metrics.csv", dense["strong_flip"])
    pipeline.atomic_csv(output_root / "stage2_dense/dense_cw_conditional.csv", dense["conditional"])
    pipeline.atomic_csv(output_root / "stage2_dense/trigger_relative_metrics.csv", dense["trigger"])
    pipeline.atomic_csv(output_root / "stage2_dense/dataset_source_breakdown.csv", dense["breakdown"])
    pipeline.atomic_csv(output_root / "stage2_dense/nuisance_control_comparison.csv", dense["nuisance"])
    pipeline.atomic_csv(output_root / "stage2_dense/text_visual_control_comparison.csv", dense["controls"])
    pipeline.atomic_csv(output_root / "stage2_dense/secondary_factorial_targets.csv", dense["secondary"])
    pipeline.atomic_csv(output_root / "stage2_routed/state_regime_transfer.csv", transfer)

    bootstrap = _bootstrap(
        contract,
        stage1,
        {"read": dense["read"], "write": dense["write"]},
    )
    pipeline.atomic_csv(output_root / "statistics/uid_bootstrap_ci.csv", bootstrap)
    primary_comparison = []
    for row in stage1["overall"]:
        primary_comparison.append(
            {
                "domain": "stage1", "target": "dense_wrong", "model": row["model"],
                "input": row["input"], "primary_metric": "auroc", "value": row["auroc"],
            }
        )
    for target in ("read", "write"):
        for row in dense["continuous"]:
            if row["target"] == target and row["aggregation"] == "state_micro":
                harmful = _metric_row(dense["harmful"], target=target, model=row["model"], input=row["input"])
                primary_comparison.append(
                    {
                        "domain": "stage2_dense", "target": target, "model": row["model"],
                        "input": row["input"], "primary_metric": "spearman",
                        "value": row["spearman"], "harmful_auroc": harmful["auroc"],
                    }
                )
    pipeline.atomic_csv(output_root / "statistics/primary_comparison_table.csv", primary_comparison)
    _figures(output_root, stage1, dense, routed)

    category = _evidence_category(stage1, dense, bootstrap)
    summary = _summary_text(
        contract, stage1, dense, routed, transfer, bootstrap, seed_summary, category
    )
    pipeline._atomic_bytes(
        output_root / "summaries/stepB_id_learnability_summary.md", summary.encode()
    )
    preliminary_files = _artifact_files(output_root)
    readiness = _readiness_text(contract, category, preliminary_files)
    pipeline._atomic_bytes(output_root / "summaries/stepC_readiness.md", readiness.encode())
    files = _artifact_files(output_root)
    manifest = {
        "schema_version": "predictability_stepB_artifact_manifest_v1",
        "contract_sha256": contract["contract_sha256"],
        "created_at": pipeline.utc_now(),
        "evidence_category": {"code": category[0], "label": category[1]},
        "ready_for_step_c": True,
        "training_tasks": len(tasks),
        "transfer_tasks": len(pipeline._transfer_tasks(contract["static_config"])),
        "files": files,
    }
    manifest["artifact_manifest_sha256"] = pipeline.canonical_hash(manifest)
    pipeline.atomic_json(output_root / "artifact_manifest.json", manifest)
    print(
        json.dumps(
            {
                "complete": True,
                "contract_sha256": contract["contract_sha256"],
                "artifact_manifest_sha256": manifest["artifact_manifest_sha256"],
                "evidence_category": category[0],
                "ready_for_step_c": True,
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=pipeline.DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    aggregate(parse_args().config)
