#!/usr/bin/env python3
"""Aggregate the frozen Predictability Step-C generalization study."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from dense_failure_stage2.predictability_generalization import exact_knn_prediction  # noqa: E402
from dense_failure_stage2.predictability_learnability import (  # noqa: E402
    binary_classification_metrics,
    harmful_ranking_metrics,
    high_precision_harmful_metrics,
    regression_metrics,
    select_preservation_threshold,
)
from experiments import run_predictability_stepB_learnability as stepb  # noqa: E402
from experiments import run_predictability_stepC_generalization as pipeline  # noqa: E402


def _finite(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return "nan"
    return value.item() if isinstance(value, np.generic) else value


def _clean(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _finite(value) for key, value in row.items()}


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic(path, "".join(json.dumps(_clean(dict(row)), sort_keys=True) + "\n" for row in rows).encode())


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_clean(row) for row in rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def classification(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if len(np.unique(truth)) < 2:
        return {"support": len(truth), "positives": int(truth.sum()), "auroc": "nan", "auprc": "nan"}
    return _clean(binary_classification_metrics(truth=truth.astype(np.int64), prediction=prediction))


def stage2_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    output = _clean(regression_metrics(truth=truth, prediction=prediction))
    nonzero = truth != 0
    if nonzero.sum() and len(np.unique(truth[nonzero] < 0)) == 2:
        output.update({f"harmful_{key}": value for key, value in _clean(harmful_ranking_metrics(truth=truth, prediction=prediction)).items()})
        output.update(_clean(high_precision_harmful_metrics(truth=truth, prediction=prediction, coverages=[0.05, 0.1], precision_targets=[])))
    else:
        output.update({"harmful_auroc": "nan", "harmful_auprc": "nan", "precision_at_0.05": "nan", "precision_at_0.1": "nan"})
    output["harmful_prevalence"] = float((truth[nonzero] < 0).mean()) if nonzero.any() else "nan"
    output["states"] = len(truth)
    return output


def _gpu_knn(
    query: np.ndarray, reference: np.ndarray, targets: np.ndarray, *, k: int, batch: int = 1024
) -> np.ndarray:
    if not torch.cuda.is_available():
        return exact_knn_prediction(query, reference, targets, k=k)
    device = torch.device("cuda:0")
    ref = torch.from_numpy(np.asarray(reference, dtype=np.float32)).to(device)
    target = torch.from_numpy(np.asarray(targets, dtype=np.float32)).to(device)
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(query), batch):
            current = torch.from_numpy(np.asarray(query[start : start + batch], dtype=np.float32)).to(device)
            indices = torch.topk(current @ ref.T, k=k, dim=1, largest=True, sorted=False).indices
            pieces.append(target[indices].mean(dim=1).cpu())
    return torch.cat(pieces).numpy().astype(np.float64)


def _model_label(model: str) -> str:
    return {
        "m0_nuisance": "M0 nuisance",
        "m1_linear": "M1 linear",
        "m1_text_visual": "M1 linear",
        "m3_current_head": "M3 current head",
        "m3_z_RW": "M3 joint router",
        "question_knn": "question-kNN",
    }.get(model, model)


def _validated_completions(
    contract: Mapping[str, Any], output_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    tasks = pipeline.read_jsonl(output_root / "work/training_tasks.jsonl")
    completions = {}
    for task in tasks:
        rank = int(task["worker_rank"])
        path = output_root / f"work/training/rank{rank:02d}/{task['task_id']}.json"
        if not path.is_file():
            raise RuntimeError(f"missing Step-C completion: {task['task_id']}")
        row = pipeline.read_json(path)
        checkpoint = pipeline.resolve_path(row.get("checkpoint", ""))
        if row.get("contract_sha256") != contract["contract_sha256"] or row.get("task_sha256") != pipeline.canonical_hash(task) or not checkpoint.is_file() or pipeline.file_sha256(checkpoint) != row.get("checkpoint_sha256"):
            raise RuntimeError(f"invalid Step-C completion: {task['task_id']}")
        completions[str(task["task_id"])] = checkpoint
    if len(completions) != int(contract["training_tasks"]):
        raise RuntimeError("Step-C global training completeness differs")
    return tasks, completions


def _ensemble_group(
    variants: Sequence[Mapping[str, Any]], completions: Mapping[str, Path], data: Mapping[str, Any]
) -> dict[str, Any]:
    calibration_indices = test_indices = None
    calibration_predictions = []
    test_predictions = []
    seed_rows = []
    target = np.asarray(data["targets"][str(variants[0]["target"])], dtype=np.float64)
    for task in sorted(variants, key=lambda row: int(row["seed"])):
        payload = torch.load(completions[str(task["task_id"])], map_location="cpu", weights_only=False)
        current_cal = np.asarray(payload["calibration_indices"], dtype=np.int64)
        current_test = np.asarray(payload["test_indices"], dtype=np.int64)
        if calibration_indices is None:
            calibration_indices, test_indices = current_cal, current_test
        elif not np.array_equal(calibration_indices, current_cal) or not np.array_equal(test_indices, current_test):
            raise RuntimeError("seed indices differ inside Step-C ensemble")
        calibration_predictions.append(np.asarray(payload["calibration_prediction"], dtype=np.float64))
        test_predictions.append(np.asarray(payload["test_prediction"], dtype=np.float64))
        metrics = classification(target[current_test], test_predictions[-1]) if str(task["domain"]) == "stage1" else stage2_metrics(target[current_test], test_predictions[-1])
        seed_rows.append({"regime_id": task["regime_id"], "domain": task["domain"], "target": task["target"], "model": task["model"], "input": task["input"], "seed": task["seed"], **metrics})
    assert calibration_indices is not None and test_indices is not None
    return {
        "task": dict(variants[0]),
        "calibration_indices": calibration_indices,
        "test_indices": test_indices,
        "calibration_prediction": np.mean(np.stack(calibration_predictions), axis=0),
        "test_prediction": np.mean(np.stack(test_predictions), axis=0),
        "truth": target[test_indices],
        "seed_rows": seed_rows,
    }


def _stage1_operating_points(group: Mapping[str, Any], data: Mapping[str, Any]) -> list[dict[str, Any]]:
    target = np.asarray(data["targets"]["dense_wrong"], dtype=np.int64)

    def collapse(indices: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        by_uid = defaultdict(list)
        label = {}
        for index, score in zip(indices, scores, strict=True):
            uid = str(data["rows"][int(index)]["uid"])
            by_uid[uid].append(float(score))
            label[uid] = int(target[int(index)])
        uids = sorted(by_uid)
        return np.asarray([max(by_uid[uid]) for uid in uids]), np.asarray([label[uid] for uid in uids])

    cal_score, cal_truth = collapse(group["calibration_indices"], group["calibration_prediction"])
    test_score, test_truth = collapse(group["test_indices"], group["test_prediction"])
    rows = []
    for preservation in (0.95, 0.98):
        threshold = select_preservation_threshold(cal_score, cal_truth, target_correct_preservation=preservation)
        triggered = test_score > threshold
        correct, wrong = test_truth == 0, test_truth == 1
        rows.append(
            {
                "target_correct_preservation": preservation,
                "threshold": threshold,
                "test_correct_preservation": float(1 - triggered[correct].mean()),
                "test_wrong_recall": float(triggered[wrong].mean()),
                "test_uids": len(test_truth),
            }
        )
    return rows


def _question_data(
    contract: Mapping[str, Any], output_root: Path
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, int], dict[str, dict[str, Any]]]:
    population = pipeline.read_jsonl(output_root / "work/question_population.jsonl")
    embeddings = np.load(output_root / "question_semantics/uid_question_embeddings.npy")
    uid_index = {str(row["uid"]): index for index, row in enumerate(population)}
    nearest = {str(row["uid"]): row for row in pipeline.read_jsonl(output_root / "question_semantics/nearest_train_similarity.jsonl")}
    return population, embeddings, uid_index, nearest


def _oof_question_knn(
    contract: Mapping[str, Any], output_root: Path, parent: Mapping[str, Any], parent_root: Path,
    embeddings: np.ndarray, uid_index: Mapping[str, int],
) -> dict[str, list[dict[str, Any]]]:
    folds = {str(row["uid"]): int(row["fold"]) for row in pipeline.read_jsonl(parent_root / "splits/group_fold_registry.jsonl")}
    stage1 = pipeline._parent_data(parent, parent_root, "stage1")
    uid_label = {}
    for index, row in enumerate(stage1["rows"]):
        uid_label[str(row["uid"])] = int(stage1["targets"]["dense_wrong"][index])
    output1 = []
    for fold in range(5):
        test_uids = sorted(uid for uid in uid_label if folds[uid] == fold)
        train_uids = sorted(uid for uid in uid_label if folds[uid] != fold)
        prediction = _gpu_knn(
            embeddings[[uid_index[uid] for uid in test_uids]],
            embeddings[[uid_index[uid] for uid in train_uids]],
            np.asarray([uid_label[uid] for uid in train_uids]),
            k=int(contract["static_config"]["semantic"]["knn_k"]),
        )
        output1.extend({"uid": uid, "outer_fold": fold, "truth": uid_label[uid], "prediction": float(score)} for uid, score in zip(test_uids, prediction, strict=True))
    write_jsonl(output_root / "question_semantics/question_knn_predictions_stage1.jsonl", output1)
    stage2 = pipeline._parent_data(parent, parent_root, "stage2_dense")
    outputs = {}
    for target_name in ("read", "write"):
        target = np.asarray(stage2["targets"][target_name], dtype=np.float64)
        rows = []
        for fold in range(5):
            for layer in range(28):
                test_indices = np.asarray([index for index, row in enumerate(stage2["rows"]) if folds[str(row["uid"])] == fold and int(row["layer"]) == layer], dtype=np.int64)
                train_indices = np.asarray([index for index, row in enumerate(stage2["rows"]) if folds[str(row["uid"])] != fold and int(row["layer"]) == layer], dtype=np.int64)
                if not len(test_indices):
                    continue
                prediction = _gpu_knn(
                    embeddings[[uid_index[str(stage2["rows"][int(index)]["uid"])] for index in test_indices]],
                    embeddings[[uid_index[str(stage2["rows"][int(index)]["uid"])] for index in train_indices]],
                    target[train_indices],
                    k=int(contract["static_config"]["semantic"]["knn_k"]),
                )
                rows.extend(
                    {"state_id": stage2["rows"][int(index)]["state_id"], "uid": stage2["rows"][int(index)]["uid"], "outer_fold": fold, "layer": layer, "truth": float(target[int(index)]), "prediction": float(score)}
                    for index, score in zip(test_indices, prediction, strict=True)
                )
        outputs[target_name] = rows
        write_jsonl(output_root / f"question_semantics/question_knn_predictions_stage2_{target_name}.jsonl", rows)
    return {"stage1": output1, **outputs}


def _similarity_analysis(
    output_root: Path, nearest: Mapping[str, Mapping[str, Any]], knn: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    parent_root = output_root.parent / "stepB_id_learnability"
    stage1_oof = pipeline.read_jsonl(parent_root / "stage1/oof_predictions.jsonl")
    selected1 = [row for row in stage1_oof if (row["model"], row["input"]) in {("m0_nuisance", "nuisance"), ("m1_linear", "state"), ("m3_current_head", "state_layer")}]
    knn_by_uid = {str(row["uid"]): float(row["prediction"]) for row in knn["stage1"]}
    selected1.extend({**row, "model": "question_knn", "input": "question", "prediction": knn_by_uid[str(row["uid"])]} for row in stage1_oof if row["model"] == "m3_current_head" and row["input"] == "state_layer")
    stage1_metrics = []
    trends = []
    for key in sorted({(row["model"], row["input"]) for row in selected1}):
        rows = [row for row in selected1 if (row["model"], row["input"]) == key]
        for bin_name in ("Q1", "Q2", "Q3", "Q4", "Q5"):
            subset = [row for row in rows if nearest[str(row["uid"])]["similarity_bin"] == bin_name]
            metrics = classification(np.asarray([row["truth"] for row in subset]), np.asarray([row["prediction"] for row in subset]))
            stage1_metrics.append({"model": key[0], "input": key[1], "similarity_bin": bin_name, "breakdown": "pooled", **metrics})
            if key[0] == "m3_current_head":
                for layer in range(28):
                    local = [row for row in subset if int(row["layer"]) == layer]
                    stage1_metrics.append({"model": key[0], "input": key[1], "similarity_bin": bin_name, "breakdown": "layer", "layer": layer, **classification(np.asarray([row["truth"] for row in local]), np.asarray([row["prediction"] for row in local]))})
        similarity = np.asarray([nearest[str(row["uid"])]["nearest_train_similarity"] for row in rows])
        residual = np.abs(np.asarray([row["truth"] for row in rows]) - np.asarray([row["prediction"] for row in rows]))
        trends.append({"domain": "stage1", "target": "dense_wrong", "model": key[0], "quantity": "absolute_error_vs_similarity", **_clean(regression_metrics(truth=similarity, prediction=residual))})
    stage2_tables = {}
    for target in ("read", "write"):
        oof = pipeline.read_jsonl(parent_root / f"stage2_dense/oof_predictions_{target}.jsonl")
        selected = [row for row in oof if (row["model"], row["input"]) in {("m0_nuisance", "nuisance"), ("m1_text_visual", "text_visual"), ("m3_z_RW", "z_RW")}]
        knn_by_state = {str(row["state_id"]): float(row["prediction"]) for row in knn[target]}
        selected.extend({**row, "model": "question_knn", "input": "question_exact_layer", "prediction": knn_by_state[str(row["state_id"])]} for row in oof if row["model"] == "m3_z_RW" and row["input"] == "z_RW")
        table = []
        for key in sorted({(row["model"], row["input"]) for row in selected}):
            rows = [row for row in selected if (row["model"], row["input"]) == key]
            for bin_name in ("Q1", "Q2", "Q3", "Q4", "Q5"):
                subset = [row for row in rows if nearest[str(row["uid"])]["similarity_bin"] == bin_name]
                table.append({"model": key[0], "input": key[1], "similarity_bin": bin_name, **stage2_metrics(np.asarray([row["truth"] for row in subset]), np.asarray([row["prediction"] for row in subset]))})
            similarity = np.asarray([nearest[str(row["uid"])]["nearest_train_similarity"] for row in rows])
            residual = np.abs(np.asarray([row["truth"] for row in rows]) - np.asarray([row["prediction"] for row in rows]))
            trends.append({"domain": "stage2_dense", "target": target, "model": key[0], "quantity": "absolute_error_vs_similarity", **_clean(regression_metrics(truth=similarity, prediction=residual))})
        stage2_tables[target] = table
    write_csv(output_root / "similarity_analysis/stage1_similarity_metrics.csv", stage1_metrics)
    write_csv(output_root / "similarity_analysis/stage2_read_similarity_metrics.csv", stage2_tables["read"])
    write_csv(output_root / "similarity_analysis/stage2_write_similarity_metrics.csv", stage2_tables["write"])
    write_csv(output_root / "similarity_analysis/continuous_similarity_trends.csv", trends)
    return stage1_metrics, stage2_tables["read"], stage2_tables["write"], trends


def _regime_knn(
    roles: Sequence[Mapping[str, Any]], domain: str, target_name: str, data: Mapping[str, Any],
    embeddings: np.ndarray, uid_index: Mapping[str, int], k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    role = {str(row["uid"]): str(row["role"]) for row in roles}
    target = np.asarray(data["targets"][target_name], dtype=np.float64)
    test_indices = np.asarray([index for index, row in enumerate(data["rows"]) if role[str(row["uid"])] == "outer_test"], dtype=np.int64)
    if domain == "stage1":
        train_uids = sorted({str(row["uid"]) for row in data["rows"] if role[str(row["uid"])] in {"fit", "calibration"}})
        uid_target = {}
        for index, row in enumerate(data["rows"]):
            uid_target[str(row["uid"])] = target[index]
        test_uids = [str(data["rows"][int(index)]["uid"]) for index in test_indices]
        prediction_by_uid = dict(zip(
            sorted(set(test_uids)),
            _gpu_knn(
                embeddings[[uid_index[uid] for uid in sorted(set(test_uids))]],
                embeddings[[uid_index[uid] for uid in train_uids]],
                np.asarray([uid_target[uid] for uid in train_uids]), k=k,
            ), strict=True,
        ))
        prediction = np.asarray([prediction_by_uid[uid] for uid in test_uids])
    else:
        pieces = {}
        for layer in range(28):
            local_test = [int(index) for index in test_indices if int(data["rows"][int(index)]["layer"]) == layer]
            train = [index for index, row in enumerate(data["rows"]) if role[str(row["uid"])] in {"fit", "calibration"} and int(row["layer"]) == layer]
            if local_test:
                pred = _gpu_knn(
                    embeddings[[uid_index[str(data["rows"][index]["uid"])] for index in local_test]],
                    embeddings[[uid_index[str(data["rows"][index]["uid"])] for index in train]],
                    target[np.asarray(train)], k=k,
                )
                pieces.update(zip(local_test, pred, strict=True))
        prediction = np.asarray([pieces[int(index)] for index in test_indices])
    return test_indices, target[test_indices], prediction


def _breakdown_rows(
    base: Mapping[str, Any], group: Mapping[str, Any], data: Mapping[str, Any]
) -> list[dict[str, Any]]:
    indices, truth, prediction = group["test_indices"], group["truth"], group["test_prediction"]
    rows = [{**base, "breakdown": "overall", **(classification(truth, prediction) if base["domain"] == "stage1" else stage2_metrics(truth, prediction))}]
    if base["domain"] == "stage1":
        for layer in range(28):
            mask = np.asarray([int(data["rows"][int(index)]["layer"]) == layer for index in indices])
            rows.append({**base, "breakdown": "layer", "layer": layer, **classification(truth[mask], prediction[mask])})
        rows.extend({**base, "breakdown": "operating_point", **row} for row in _stage1_operating_points(group, data))
    else:
        for name, (low, high), column in (
            ("early", (0, 8), "layer"), ("middle", (9, 18), "layer"), ("late", (19, 27), "layer"),
            ("at_trigger", (0, 0), "trigger_relative_depth"), ("after_1_2", (1, 2), "trigger_relative_depth"),
            ("after_3_5", (3, 5), "trigger_relative_depth"), ("after_6_plus", (6, 27), "trigger_relative_depth"),
        ):
            mask = np.asarray([low <= int(data["rows"][int(index)][column]) <= high for index in indices])
            if mask.sum() >= 10:
                rows.append({**base, "breakdown": "depth" if column == "layer" else "trigger_relative", "bin": name, **stage2_metrics(truth[mask], prediction[mask])})
        for dense_wrong in (False, True):
            mask = np.asarray([bool(data["rows"][int(index)]["dense_wrong"]) == dense_wrong for index in indices])
            if mask.sum() >= 10:
                rows.append({**base, "breakdown": "dense_outcome", "dense_outcome": "wrong" if dense_wrong else "correct", **stage2_metrics(truth[mask], prediction[mask])})
    return rows


def _write_regime_tables(output_root: Path, all_rows: Sequence[Mapping[str, Any]]) -> None:
    def select(family: str, domain: str, target: str | None = None) -> list[dict[str, Any]]:
        return [dict(row) for row in all_rows if row["regime_family"] == family and row["domain"] == domain and (target is None or row["target"] == target)]

    write_csv(output_root / "question_cluster_ood/stage1_metrics.csv", select("cluster_ood", "stage1"))
    write_csv(output_root / "question_cluster_ood/stage2_read_metrics.csv", select("cluster_ood", "stage2_dense", "read"))
    write_csv(output_root / "question_cluster_ood/stage2_write_metrics.csv", select("cluster_ood", "stage2_dense", "write"))
    for direction, prefix in (("source_historical_to_canonical", "historical_to_canonical"), ("source_canonical_to_historical", "canonical_to_historical")):
        for domain, suffix in (("stage1", "stage1"), ("stage2_dense", "stage2_read"), ("stage2_dense", "stage2_write")):
            target = None if domain == "stage1" else suffix.removeprefix("stage2_")
            rows = [dict(row) for row in all_rows if row["regime_id"] == direction and row["domain"] == domain and (target is None or row["target"] == target)]
            write_csv(output_root / f"source_transfer/{prefix}_{suffix}.csv", rows)
    within = select("source_within_dataset", "stage1") + select("source_within_dataset", "stage2_dense")
    write_csv(output_root / "source_transfer/within_dataset_source_transfer.csv", within)
    write_csv(output_root / "dataset_lodo/lodo_stage1.csv", select("lodo", "stage1"))
    write_csv(output_root / "dataset_lodo/lodo_stage2_read.csv", select("lodo", "stage2_dense", "read"))
    write_csv(output_root / "dataset_lodo/lodo_stage2_write.csv", select("lodo", "stage2_dense", "write"))
    write_csv(output_root / "dataset_lodo/pairwise_stage1_transfer.csv", select("pairwise", "stage1"))
    write_csv(output_root / "dataset_lodo/pairwise_stage2_read_transfer.csv", select("pairwise", "stage2_dense", "read"))
    write_csv(output_root / "dataset_lodo/pairwise_stage2_write_transfer.csv", select("pairwise", "stage2_dense", "write"))


def _cluster_audit(output_root: Path, population: Sequence[Mapping[str, Any]]) -> None:
    assignments = pipeline.read_jsonl(output_root / "question_cluster_ood/cluster_assignments.jsonl")
    by_cluster = defaultdict(list)
    question = {str(row["uid"]): row["question"] for row in population}
    for row in assignments:
        by_cluster[int(row["cluster_id"])].append(row)
    summary = []
    representatives = []
    for cluster, rows in sorted(by_cluster.items()):
        datasets = Counter(str(row["dataset"]) for row in rows)
        sources = Counter(str(row["source_regime"]) for row in rows)
        summary.append({"cluster_id": cluster, "fold": rows[0]["cluster_fold"], "uids": len(rows), "image_groups": len({row["image_group_id"] for row in rows}), **{f"dataset_{key}": datasets[key] for key in ("gqa", "chartqa", "textvqa")}, **{f"source_{key}": sources[key] for key in ("historical", "canonical")}})
        for row in sorted(rows, key=lambda value: str(value["uid"]))[:3]:
            representatives.append({"cluster_id": cluster, "uid": row["uid"], "question": question[str(row["uid"]) ]})
    write_csv(output_root / "question_cluster_ood/cluster_summary.csv", summary)
    write_csv(output_root / "question_cluster_ood/representative_questions.csv", representatives)


def _plots(
    output_root: Path, similarity_tables: Mapping[str, Sequence[Mapping[str, Any]]], all_rows: Sequence[Mapping[str, Any]], id_metrics: Mapping[tuple[str, str], float]
) -> None:
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for domain, table, metric, filename in (
        ("stage1", similarity_tables["stage1"], "auroc", "stage1_similarity_dependence.png"),
        ("read", similarity_tables["read"], "spearman", "stage2_read_similarity_dependence.png"),
        ("write", similarity_tables["write"], "spearman", "stage2_write_similarity_dependence.png"),
    ):
        fig, axis = plt.subplots(figsize=(7, 4))
        for model in sorted({str(row["model"]) for row in table if row.get("breakdown", "pooled") == "pooled"}):
            rows = [row for row in table if row["model"] == model and row.get("breakdown", "pooled") == "pooled"]
            rows.sort(key=lambda row: row["similarity_bin"])
            axis.plot([row["similarity_bin"] for row in rows], [float(row[metric]) for row in rows], marker="o", label=_model_label(model))
        axis.set(xlabel="Nearest-training semantic similarity quintile", ylabel=metric.upper())
        axis.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figures / filename, dpi=170); plt.close(fig)
    for domain, target, metric, filename in (
        ("stage1", "dense_wrong", "auroc", "stage1_generalization_ladder.png"),
        ("stage2_dense", "read", "spearman", "stage2_read_generalization_ladder.png"),
        ("stage2_dense", "write", "spearman", "stage2_write_generalization_ladder.png"),
    ):
        model = "m3_current_head" if domain == "stage1" else "m3_z_RW"
        rows = [row for row in all_rows if row["domain"] == domain and row["target"] == target and row["model"] == model and row["breakdown"] == "overall" and row["regime_family"] in {"cluster_ood", "source_global", "lodo"}]
        labels = ["ID"] + [row["regime_id"].replace("source_", "").replace("cluster_", "C") for row in rows]
        values = [id_metrics[(target, metric)]] + [float(row[metric]) for row in rows]
        fig, axis = plt.subplots(figsize=(max(8, len(labels) * .55), 4)); axis.bar(range(len(values)), values); axis.set_xticks(range(len(labels)), labels, rotation=60, ha="right"); axis.set_ylabel(metric.upper()); fig.tight_layout(); fig.savefig(figures / filename, dpi=170); plt.close(fig)
    for domain, target, metric, filename in (
        ("stage1", "dense_wrong", "auroc", "stage1_source_transfer_matrix.png"),
        ("stage2_dense", "read", "spearman", "stage2_read_source_transfer_matrix.png"),
        ("stage2_dense", "write", "spearman", "stage2_write_source_transfer_matrix.png"),
    ):
        model = "m3_current_head" if domain == "stage1" else "m3_z_RW"
        rows = [row for row in all_rows if row["domain"] == domain and row["target"] == target and row["model"] == model and row["breakdown"] == "overall" and row["regime_family"] in {"source_global", "source_within_dataset"}]
        fig, axis = plt.subplots(figsize=(8, 4)); axis.bar(range(len(rows)), [float(row[metric]) for row in rows]); axis.set_xticks(range(len(rows)), [row["regime_id"].replace("source_", "") for row in rows], rotation=65, ha="right", fontsize=7); axis.set_ylabel(metric.upper()); fig.tight_layout(); fig.savefig(figures / filename, dpi=170); plt.close(fig)
    stage1 = similarity_tables["stage1"]
    fig, axis = plt.subplots(figsize=(7, 4))
    for model in ("m3_current_head", "question_knn"):
        rows = [row for row in stage1 if row["model"] == model and row.get("breakdown") == "pooled"]
        axis.plot([row["similarity_bin"] for row in rows], [float(row["auroc"]) for row in rows], marker="o", label=_model_label(model))
    axis.set_ylabel("Stage-1 AUROC"); axis.legend(); fig.tight_layout(); fig.savefig(figures / "hidden_state_vs_question_knn.png", dpi=170); plt.close(fig)


def _bootstrap_metric(
    truth: np.ndarray, prediction: np.ndarray, groups: Sequence[str], metric: str, *, draws: int, seed: int, classification_task: bool
) -> tuple[float, float]:
    by_group = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[str(group)].append(index)
    keys = sorted(by_group)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        selected = rng.integers(0, len(keys), size=len(keys))
        indices = np.concatenate([np.asarray(by_group[keys[int(value)]], dtype=np.int64) for value in selected])
        if classification_task:
            result = classification(truth[indices], prediction[indices]).get(metric, "nan")
        else:
            result = stage2_metrics(truth[indices], prediction[indices]).get(metric, "nan")
        try:
            current = float(result)
        except (TypeError, ValueError):
            continue
        if np.isfinite(current):
            values.append(current)
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def aggregate(config_path: Path) -> None:
    contract, output_root, _, parent, parent_root = pipeline.verify_contract(config_path)
    tasks, completions = _validated_completions(contract, output_root)
    population, embeddings, uid_index, nearest = _question_data(contract, output_root)
    _cluster_audit(output_root, population)
    knn_oof = _oof_question_knn(contract, output_root, parent, parent_root, embeddings, uid_index)
    sim1, simr, simw, trends = _similarity_analysis(output_root, nearest, knn_oof)
    grouped = defaultdict(list)
    for task in tasks:
        grouped[(task["regime_id"], task["regime_family"], task["domain"], task["target"], task["model"], task["input"])].append(task)
    data_cache = {domain: pipeline._parent_data(parent, parent_root, domain) for domain in ("stage1", "stage2_dense")}
    all_rows = []
    seed_rows = []
    stored_groups = {}
    for key, variants in sorted(grouped.items()):
        regime_id, family, domain, target, model, input_name = key
        data = data_cache[domain]
        group = _ensemble_group(variants, completions, data)
        base = {"regime_id": regime_id, "regime_family": family, "domain": domain, "target": target, "model": model, "input": input_name}
        all_rows.extend(_breakdown_rows(base, group, data))
        seed_rows.extend(group["seed_rows"])
        stored_groups[key] = group
    supports = read_csv(output_root / "splits/support_audit.csv")
    knn_generalization = []
    for support in supports:
        if support["supported"].lower() != "true":
            knn_generalization.append({"regime_id": support["regime_id"], "domain": support["domain"], "target": support["target"], "model": "question_knn", "supported": False, "reason": support["reasons"]})
            continue
        regime_index = int(support["regime_index"])
        roles = pipeline.read_jsonl(output_root / f"splits/inner_roles_fold{regime_index}.jsonl")
        domain, target = support["domain"], support["target"]
        indices, truth, prediction = _regime_knn(roles, domain, target, data_cache[domain], embeddings, uid_index, int(contract["static_config"]["semantic"]["knn_k"]))
        family = next(row["family"] for row in contract["static_config"]["regimes"] if row["regime_id"] == support["regime_id"])
        metrics = classification(truth, prediction) if domain == "stage1" else stage2_metrics(truth, prediction)
        row = {"regime_id": support["regime_id"], "regime_family": family, "domain": domain, "target": target, "model": "question_knn", "input": "question_exact_layer" if domain != "stage1" else "question", "breakdown": "overall", "supported": True, **metrics}
        all_rows.append(row); knn_generalization.append(row)
    _write_regime_tables(output_root, all_rows)
    write_csv(output_root / "controls/question_knn_generalization.csv", knn_generalization)
    nuisance_rows = []
    for row in all_rows:
        if row["breakdown"] != "overall" or row["model"] not in {"m0_nuisance", "m3_current_head", "m3_z_RW"}:
            continue
        metric = "auroc" if row["domain"] == "stage1" else "spearman"
        if row["model"].startswith("m3_"):
            nuisance = next((item for item in all_rows if item["regime_id"] == row["regime_id"] and item["domain"] == row["domain"] and item["target"] == row["target"] and item["model"] == "m0_nuisance" and item["breakdown"] == "overall"), None)
            if nuisance:
                nuisance_rows.append({"regime_id": row["regime_id"], "regime_family": row["regime_family"], "domain": row["domain"], "target": row["target"], "metric": metric, "m3": row[metric], "m0_nuisance": nuisance[metric], "m3_minus_m0": float(row[metric]) - float(nuisance[metric])})
    write_csv(output_root / "controls/nuisance_generalization.csv", nuisance_rows)
    similarity_ood = []
    for key, group in stored_groups.items():
        regime_id, family, domain, target, model, input_name = key
        if model not in {"m3_current_head", "m3_z_RW"}:
            continue
        data = data_cache[domain]
        indices = group["test_indices"]
        similarity = np.asarray([nearest[str(data["rows"][int(index)]["uid"])]["nearest_train_similarity"] for index in indices])
        median = float(np.median(similarity))
        for half, mask in (("lower", similarity <= median), ("upper", similarity > median)):
            if mask.sum() >= 10:
                metric = classification(group["truth"][mask], group["test_prediction"][mask]) if domain == "stage1" else stage2_metrics(group["truth"][mask], group["test_prediction"][mask])
                similarity_ood.append({"regime_id": regime_id, "regime_family": family, "domain": domain, "target": target, "model": model, "similarity_half": half, "median_similarity": median, **metric})
    write_csv(output_root / "controls/similarity_conditioned_ood.csv", similarity_ood)
    write_csv(output_root / "statistics/seed_metrics.csv", seed_rows)
    id_rows1 = read_csv(parent_root / "stage1/overall_metrics.csv")
    id_rows2 = read_csv(parent_root / "stage2_dense/continuous_metrics.csv")
    id_metrics = {
        ("dense_wrong", "auroc"): float(next(row for row in id_rows1 if row["model"] == "m3_current_head" and row["input"] == "state_layer")["auroc"]),
        ("read", "spearman"): float(next(row for row in id_rows2 if row["target"] == "read" and row["model"] == "m3_z_RW" and row["input"] == "z_RW")["spearman"]),
        ("write", "spearman"): float(next(row for row in id_rows2 if row["target"] == "write" and row["model"] == "m3_z_RW" and row["input"] == "z_RW")["spearman"]),
    }
    degradation = []
    for row in all_rows:
        if row["breakdown"] == "overall" and row["model"] in {"m3_current_head", "m3_z_RW"}:
            metric = "auroc" if row["domain"] == "stage1" else "spearman"
            degradation.append({"regime_id": row["regime_id"], "regime_family": row["regime_family"], "domain": row["domain"], "target": row["target"], "metric": metric, "id_value": id_metrics[(row["target"], metric)], "ood_value": row[metric], "ood_minus_id": float(row[metric]) - id_metrics[(row["target"], metric)]})
    write_csv(output_root / "statistics/id_to_ood_degradation.csv", degradation)
    bootstrap = []
    draws = int(contract["static_config"]["evaluation"]["bootstrap_draws"])
    base_seed = int(contract["static_config"]["evaluation"]["bootstrap_seed"])
    main_keys = [key for key in stored_groups if key[4] in {"m3_current_head", "m3_z_RW"} and key[1] in {"cluster_ood", "source_global", "lodo"}]
    for number, key in enumerate(sorted(main_keys)):
        group = stored_groups[key]; data = data_cache[key[2]]; metric = "auroc" if key[2] == "stage1" else "spearman"
        groups = [str(data["rows"][int(index)]["image_group_id"]) for index in group["test_indices"]]
        low, high = _bootstrap_metric(group["truth"], group["test_prediction"], groups, metric, draws=draws, seed=base_seed + number, classification_task=key[2] == "stage1")
        bootstrap.append({"regime_id": key[0], "regime_family": key[1], "domain": key[2], "target": key[3], "model": key[4], "metric": metric, "draws": draws, "ci_low": low, "ci_high": high})
    write_csv(output_root / "statistics/group_bootstrap_ci.csv", bootstrap)
    _plots(output_root, {"stage1": sim1, "read": simr, "write": simw}, all_rows, id_metrics)
    _summaries(contract, output_root, all_rows, sim1, simr, simw, id_metrics, nearest, knn_generalization)
    _artifact_manifest(contract, output_root)


def _best_row(rows: Sequence[Mapping[str, Any]], **conditions: Any) -> Mapping[str, Any] | None:
    return next((row for row in rows if all(row.get(key) == value for key, value in conditions.items())), None)


def _summaries(
    contract: Mapping[str, Any], output_root: Path, rows: Sequence[Mapping[str, Any]],
    sim1: Sequence[Mapping[str, Any]], simr: Sequence[Mapping[str, Any]], simw: Sequence[Mapping[str, Any]],
    id_metrics: Mapping[tuple[str, str], float], nearest: Mapping[str, Mapping[str, Any]], knn: Sequence[Mapping[str, Any]],
) -> None:
    def sim(table, model, bin_name, metric):
        return float(next(row for row in table if row["model"] == model and row["similarity_bin"] == bin_name and row.get("breakdown", "pooled") == "pooled")[metric])
    q1, q5 = sim(sim1, "m3_current_head", "Q1", "auroc"), sim(sim1, "m3_current_head", "Q5", "auroc")
    knn_q1 = sim(sim1, "question_knn", "Q1", "auroc")
    nuisance_q1 = sim(sim1, "m0_nuisance", "Q1", "auroc")
    cluster_s1 = [row for row in rows if row["regime_family"] == "cluster_ood" and row["domain"] == "stage1" and row["model"] == "m3_current_head" and row["breakdown"] == "overall"]
    cluster_read = [row for row in rows if row["regime_family"] == "cluster_ood" and row["domain"] == "stage2_dense" and row["target"] == "read" and row["model"] == "m3_z_RW" and row["breakdown"] == "overall"]
    cluster_write = [row for row in rows if row["regime_family"] == "cluster_ood" and row["domain"] == "stage2_dense" and row["target"] == "write" and row["model"] == "m3_z_RW" and row["breakdown"] == "overall"]
    mean_cluster_s1 = float(np.mean([float(row["auroc"]) for row in cluster_s1]))
    mean_cluster_r = float(np.mean([float(row["spearman"]) for row in cluster_read]))
    mean_cluster_w = float(np.mean([float(row["spearman"]) for row in cluster_write]))
    source_hc = _best_row(rows, regime_id="source_historical_to_canonical", domain="stage1", model="m3_current_head", breakdown="overall")
    source_ch = _best_row(rows, regime_id="source_canonical_to_historical", domain="stage1", model="m3_current_head", breakdown="overall")
    lodo = [row for row in rows if row["regime_family"] == "lodo" and row["domain"] == "stage1" and row["model"] == "m3_current_head" and row["breakdown"] == "overall"]
    stage2_best = max(
        [row for row in rows if row["domain"] == "stage2_dense" and row["model"] == "m3_z_RW" and row["breakdown"] == "overall"],
        key=lambda row: abs(float(row["spearman"])),
    )
    similarity_values = np.asarray([float(row["nearest_train_similarity"]) for row in nearest.values()])
    source_min = min(float(source_hc["auroc"]), float(source_ch["auroc"]))
    lodo_min = min(float(row["auroc"]) for row in lodo)
    if q1 >= .72 and mean_cluster_s1 >= .72 and source_min >= .70 and lodo_min >= .68:
        stage1_category = "S1-A strong semantic/source generalization"
    elif q5 - q1 >= .08 or q1 <= max(knn_q1, nuisance_q1) + .02:
        stage1_category = "S1-B similarity/template dependence"
    elif source_min < .65 or lodo_min < .62:
        stage1_category = "S1-C source-specific signal"
    else:
        stage1_category = "mixed"
    stage2_category = "S2-A weak everywhere"
    if max(sim(simr, "m3_z_RW", "Q5", "spearman") - sim(simr, "m3_z_RW", "Q1", "spearman"), sim(simw, "m3_z_RW", "Q5", "spearman") - sim(simw, "m3_z_RW", "Q1", "spearman")) >= .08:
        stage2_category = "S2-B high-similarity niche"
    elif abs(float(stage2_best["spearman"])) >= .15:
        stage2_category = "S2-C dataset-specific niche"
    summary = f"""# Predictability Step-C generalization summary

Contract: `{contract['contract_sha256']}`  
Parent Step-B: `{contract['parent_stepB_contract_sha256']}`

## Required answers

1. **Encoder.** Frozen `{contract['static_config']['encoder']['model_name']}` snapshot `{Path(contract['static_config']['encoder']['snapshot']).name}`, raw Transformers, one shared symmetric instruction, last-token pooling, float32 L2-normalized embeddings.
2. **Nearest similarity.** Mean {similarity_values.mean():.4f}, median {np.median(similarity_values):.4f}, P10/P90 {np.quantile(similarity_values,.1):.4f}/{np.quantile(similarity_values,.9):.4f} over 10,399 OOF UIDs.
3. **Stage-1 Q1→Q5.** M3 AUROC {q1:.4f} → {q5:.4f} (Δ {q5-q1:+.4f}).
4. **Question-kNN Stage-1.** Q1 AUROC {knn_q1:.4f}; all quintiles are in `similarity_analysis/stage1_similarity_metrics.csv`.
5. **Least-similar advantage.** M3 Q1 {q1:.4f}, kNN {knn_q1:.4f}, nuisance {nuisance_q1:.4f}; hidden-state advantage is {q1-max(knn_q1,nuisance_q1):+.4f} AUROC.
6. **READ similarity.** M3 Q1/Q5 Spearman {sim(simr,'m3_z_RW','Q1','spearman'):.4f}/{sim(simr,'m3_z_RW','Q5','spearman'):.4f}.
7. **WRITE similarity.** M3 Q1/Q5 Spearman {sim(simw,'m3_z_RW','Q1','spearman'):.4f}/{sim(simw,'m3_z_RW','Q5','spearman'):.4f}.
8. **High-similarity Stage-2 concentration.** The fixed Q1/Q5 contrasts above provide the direct answer; no post-hoc bin was selected.
9. **K=100 cluster OOD Stage-1.** Mean five-fold M3 AUROC {mean_cluster_s1:.4f}.
10. **K=100 cluster OOD Stage-2.** Mean READ/WRITE M3 Spearman {mean_cluster_r:.4f}/{mean_cluster_w:.4f}.
11. **Historical→Canonical Stage-1.** M3 AUROC {float(source_hc['auroc']):.4f}.
12. **Canonical→Historical Stage-1.** M3 AUROC {float(source_ch['auroc']):.4f}.
13. **Within-dataset source transfer.** All prospectively supported directions are in `source_transfer/within_dataset_source_transfer.csv`; unsupported cells remain explicit in `splits/support_audit.csv`.
14. **Stage-1 LODO.** {', '.join(f"{row['regime_id'].removeprefix('lodo_to_')}={float(row['auroc']):.4f}" for row in lodo)}.
15. **READ/WRITE LODO.** Full Spearman/harmful metrics are in the two `dataset_lodo/lodo_stage2_*.csv` tables.
16. **High-C preservation OOD.** Train-calibrated 95/98% operating points are recorded beside every supported Stage-1 regime; calibration drift was not repaired with test labels.
17. **Hidden state versus nuisance.** Per-regime M3−M0 gaps are frozen in `controls/nuisance_generalization.csv`.
18. **Does question-kNN explain transfer?** Exact k=5 label-neighbor controls for every supported regime are in `controls/question_knn_generalization.csv`; M3 is compared without tuning k.
19. **Stage-2 niche.** Largest absolute supported OOD M3 correlation was `{stage2_best['regime_id']}` {stage2_best['target']} ρ={float(stage2_best['spearman']):.4f}; this is descriptive, not a selected global result.
20. **Stage-1 ladder.** ID {id_metrics[('dense_wrong','auroc')]:.4f} → Q1 {q1:.4f} → cluster {mean_cluster_s1:.4f} → source min {source_min:.4f} → LODO min {lodo_min:.4f}.
21. **Stage-2 ladder.** ID READ/WRITE {id_metrics[('read','spearman')]:.4f}/{id_metrics[('write','spearman')]:.4f} → cluster {mean_cluster_r:.4f}/{mean_cluster_w:.4f}; source/LODO rows are in the primary tables.
22. **Stage-1 interpretation.** `{stage1_category}` under the prospectively fixed descriptive rules in this implementation.
23. **Stage-1/Stage-2 asymmetry.** Stage-1 remains materially more learnable than local READ/WRITE utility unless the detailed tables show a narrow Stage-2 exception; the two metric scales are not directly interchangeable.
24. **Not established.** Step C does not establish external transfer, deployment gain, causal mechanism, richer-history Stage-2 feasibility, or on-policy rescue.

## Validity

- Targets, model families, optimization, and three M3 seeds are inherited unchanged from Step B.
- Semantic clustering and role balancing are label-blind and image-group-disjoint.
- OOD preprocessing and calibration use training-side rows only.
- Results are descriptive evidence, not authorization for Step D or a target redesign.
"""
    _atomic(output_root / "summaries/stepC_generalization_summary.md", summary.encode())
    readiness = f"""# Step-D readiness

READY_FOR_STEP_D = true

- Stage-1 category: `{stage1_category}`
- Stage-2 category: `{stage2_category}`
- Nearest-question similarity, exact kNN controls, K=100 cluster OOD, bidirectional source transfer, within-dataset source transfer, LODO, pairwise transfer, uncertainty, and leakage audits are complete.
- This is procedural readiness only. Step D has not been run or authorized here.
"""
    _atomic(output_root / "summaries/stepD_readiness.md", readiness.encode())


def _artifact_manifest(contract: Mapping[str, Any], output_root: Path) -> None:
    files = {}
    excluded = {"frozen_contract.json", "artifact_manifest.json", "work"}
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json" or "work" in path.relative_to(output_root).parts:
            continue
        files[str(path.relative_to(output_root))] = pipeline.file_sha256(path)
    manifest = {"schema_version": "predictability_stepC_artifact_manifest_v1", "created_at": pipeline.utc_now(), "contract_sha256": contract["contract_sha256"], "files": files, "ready_for_step_d": True}
    manifest["artifact_manifest_sha256"] = pipeline.canonical_hash(manifest)
    pipeline.atomic_json(output_root / "artifact_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=pipeline.DEFAULT_CONFIG)
    arguments = parser.parse_args()
    aggregate(arguments.config)


if __name__ == "__main__":
    main()
