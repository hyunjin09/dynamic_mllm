#!/usr/bin/env python3
"""Retrospective artifact-only audit of historical Stage-1 population shortcuts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from dense_failure_stage1.historical_shortcut_audit import (
    average_precision,
    binary_auroc,
    deterministic_group_folds,
    exact_stratum_match,
    fit_centroid_probe,
    spearman_correlation,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/stage1_historical_population_shortcut_audit_v1.json"
DATASETS = ("gqa", "chartqa", "textvqa")
LAYERS = tuple(range(28))
BLOCKS = ("text_final", "text_mean", "visual_mean")
BOUND_CODE = (
    "configs/stage1_historical_population_shortcut_audit_v1.json",
    "dense_failure_stage1/historical_shortcut_audit.py",
    "experiments/analyze_stage1_historical_population_shortcuts.py",
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(payload.encode()).hexdigest()


def command(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n" for row in rows).encode(),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def percentile(values: Sequence[float], q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.quantile(array[np.isfinite(array)], q)) if np.isfinite(array).any() else float("nan")


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "mean": math.nan, "std": math.nan, "median": math.nan, "q25": math.nan, "q75": math.nan}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def standardized_mean_difference(a: Sequence[float], b: Sequence[float]) -> float:
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    first, second = first[np.isfinite(first)], second[np.isfinite(second)]
    if len(first) < 2 or len(second) < 2:
        return float("nan")
    pooled = math.sqrt((first.var() + second.var()) / 2.0)
    return float((second.mean() - first.mean()) / pooled) if pooled > 0 else 0.0


def image_metadata(rows: list[dict[str, Any]]) -> None:
    cache: dict[str, tuple[int, int, str]] = {}
    for index, row in enumerate(rows, 1):
        group = str(row["image_group_id"])
        if group not in cache:
            path = Path(str(row["local_image_path"]))
            with Image.open(path) as image:
                cache[group] = (int(image.width), int(image.height), str(image.format or path.suffix.lstrip(".")).lower())
        width, height, image_format = cache[group]
        row.update(
            image_width=width,
            image_height=height,
            image_area=width * height,
            aspect_ratio=width / max(height, 1),
            image_format=image_format,
            image_size_bytes=Path(str(row["local_image_path"])).stat().st_size,
        )
        if index % 2000 == 0:
            print(f"image metadata: {index}/{len(rows)}", flush=True)


def load_population(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = config["sources"]
    portable = {str(row["uid"]): row for row in read_jsonl(resolve(sources["portable_historical_manifest"]))}
    old_candidate = {str(row["uid"]): row for row in read_jsonl(resolve(sources["historical_candidate_manifest"]))}
    old_dense = {str(row["uid"]): row for row in read_jsonl(resolve(sources["historical_dense_outputs"]))}
    old_split = {str(row["uid"]): row for row in read_jsonl(resolve(sources["historical_split_manifest"]))}
    old_scores: dict[str, dict[str, Any]] = {}
    for key in ("historical_trigger_train", "historical_trigger_val", "historical_trigger_test"):
        for row in read_jsonl(resolve(sources[key])):
            old_scores[str(row["uid"])] = row
    if set(old_dense) != set(old_split) or set(old_dense) != set(old_scores):
        raise ValueError("historical dense/split/score UIDs differ")

    old: list[dict[str, Any]] = []
    for uid in sorted(old_dense):
        dense, candidate, split, score, source = (
            old_dense[uid], old_candidate[uid], old_split[uid], old_scores[uid], portable[uid]
        )
        row = {
            "uid": uid,
            "population": f"historical_{split['split']}",
            "source_family": "historical_selected_quota",
            "split": str(split["split"]),
            "dataset": str(dense["dataset"]),
            "dense_wrong": int(bool(dense["current_dense_wrong"])),
            "correctness": "W" if dense["current_dense_wrong"] else "C",
            "historical_bucket": str(candidate["historical_bucket"]),
            "image_group_id": str(dense["image_group_id"]),
            "image_content_sha256": str(dense["image_content_sha256"]),
            "local_image_path": str(candidate["local_image_path"]),
            "question": str(candidate["question"]),
            "question_token_count": int(dense["user_text_token_count"]),
            "prompt_token_count": int(dense["prompt_token_count"]),
            "answer_token_count": int(dense["generation_length"]),
            "visual_token_count": int(dense["visual_token_count"]),
            "source_file": str(source.get("source_manifest") or ""),
            "source_index": int(candidate["source_manifest_index"]),
            "source_split": str(source.get("data_split") or "unknown"),
            "source_subtype": str(source.get("source_runtime_policy") or "unknown"),
            "triggered": int(bool(score["triggered"])),
        }
        for layer in LAYERS:
            row[f"score_l{layer}"] = float(score[f"score_l{layer}"])
        row["score_max"] = max(row[f"score_l{layer}"] for layer in LAYERS)
        old.append(row)

    new_candidate = {str(row["uid"]): row for row in read_jsonl(resolve(sources["new_candidate_manifest"]))}
    new_dense = {str(row["uid"]): row for row in read_jsonl(resolve(sources["new_dense_outputs"]))}
    new_scores = {str(row["uid"]): row for row in read_jsonl(resolve(sources["new_trigger_map"]))}
    if set(new_dense) != set(new_candidate) or set(new_dense) != set(new_scores):
        raise ValueError("new dense/candidate/score UIDs differ")
    new: list[dict[str, Any]] = []
    for uid in sorted(new_dense):
        dense, candidate, score = new_dense[uid], new_candidate[uid], new_scores[uid]
        row = {
            "uid": uid,
            "population": "canonical_new",
            "source_family": "canonical_outcome_blind",
            "split": "canonical_new",
            "dataset": str(dense["dataset"]),
            "dense_wrong": int(bool(dense["current_dense_wrong"])),
            "correctness": "W" if dense["current_dense_wrong"] else "C",
            "historical_bucket": "not_applicable",
            "image_group_id": str(dense["image_group_id"]),
            "image_content_sha256": str(dense["image_content_sha256"]),
            "local_image_path": str(candidate["local_image_path"]),
            "question": str(candidate["question"]),
            "question_token_count": int(dense["user_text_token_count"]),
            "prompt_token_count": int(dense["prompt_token_count"]),
            "answer_token_count": int(dense["generation_length"]),
            "visual_token_count": int(dense["visual_token_count"]),
            "source_file": str(candidate.get("source_dataset") or ""),
            "source_index": str(candidate.get("native_row_key") or ""),
            "source_split": str(candidate.get("source_split") or "unknown"),
            "source_subtype": str(candidate.get("source_stratum") or "unknown"),
            "triggered": int(bool(score["triggered"])),
        }
        for layer in LAYERS:
            row[f"score_l{layer}"] = float(score[f"p_{layer}"])
        row["score_max"] = max(row[f"score_l{layer}"] for layer in LAYERS)
        new.append(row)
    image_metadata(old + new)
    return old, new


def load_feature_tensor(rows: list[dict[str, Any]], index_path: Path) -> np.ndarray:
    target = np.empty((len(rows), 28, 10752), dtype=np.float16)
    position = {str(row["uid"]): index for index, row in enumerate(rows)}
    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(index_path):
        by_shard[str(row["shard"])].append(row)
    observed: set[str] = set()
    for shard_number, (name, entries) in enumerate(sorted(by_shard.items()), 1):
        path = resolve(name)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        matrix = torch.cat([payload[key] for key in BLOCKS], dim=-1).to(torch.float16).numpy()
        for entry in entries:
            uid = str(entry["uid"])
            if uid not in position:
                continue
            row_index = int(entry["row_index"])
            if str(payload["uids"][row_index]) != uid:
                raise ValueError(f"feature index mismatch for {uid}")
            target[position[uid]] = matrix[row_index]
            observed.add(uid)
        if shard_number % 16 == 0:
            print(f"features: {shard_number}/{len(by_shard)} shards", flush=True)
    if observed != set(position):
        raise ValueError(f"missing features: {len(set(position) - observed)}")
    return target


def deterministic_take(indices: Sequence[int], rows: Sequence[Mapping[str, Any]], count: int, seed: int) -> list[int]:
    return sorted(indices, key=lambda i: sha256(f"{seed}:{rows[i]['uid']}".encode()).hexdigest())[:count]


def source_subset_indices(rows: list[dict[str, Any]], dataset: str, stratum: str, seed: int) -> list[int]:
    selected = [i for i, row in enumerate(rows) if row["dataset"] == dataset]
    if stratum == "C":
        return [i for i in selected if row_is(rows[i], "correctness", "C")]
    if stratum == "W":
        return [i for i in selected if row_is(rows[i], "correctness", "W")]
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in selected:
        cells[(str(rows[i]["source_family"]), str(rows[i]["correctness"]))].append(i)
    count = min(map(len, cells.values()))
    if count == 0:
        return []
    output: list[int] = []
    for key, indices in sorted(cells.items()):
        output.extend(deterministic_take(indices, rows, count, seed + int(sha256(str(key).encode()).hexdigest()[:6], 16)))
    return sorted(output)


def row_is(row: Mapping[str, Any], key: str, value: str) -> bool:
    return str(row[key]) == value


def source_probes(rows: list[dict[str, Any]], features: np.ndarray, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seed = int(config["seed"])
    n_folds = int(config["source_probe_folds"])
    for dataset in DATASETS:
        for stratum in ("C", "W", "balanced_CW"):
            indices = source_subset_indices(rows, dataset, stratum, seed)
            labels = np.asarray([int(rows[i]["source_family"] == "canonical_outcome_blind") for i in indices])
            support = Counter(labels)
            if min(support.values(), default=0) < 15:
                for layer in LAYERS:
                    results.append({"dataset": dataset, "correctness_stratum": stratum, "layer": layer, "status": "insufficient_support", "n_old": support.get(0, 0), "n_new": support.get(1, 0), "heldout_auroc": math.nan})
                continue
            groups = [str(rows[i]["image_group_id"]) for i in indices]
            folds = deterministic_group_folds(groups, seed=seed + 17, n_folds=n_folds)
            train_local = np.where(folds != 0)[0]
            test_local = np.where(folds == 0)[0]
            if len(set(labels[train_local])) < 2 or len(set(labels[test_local])) < 2:
                raise ValueError(f"source probe fold lacks a class: {dataset}/{stratum}")
            for layer in LAYERS:
                x_train = features[np.asarray(indices)[train_local], layer].astype(np.float32)
                x_test = features[np.asarray(indices)[test_local], layer].astype(np.float32)
                probe = fit_centroid_probe(x_train, labels[train_local])
                scores = probe.score(x_test)
                results.append(
                    {
                        "dataset": dataset,
                        "correctness_stratum": stratum,
                        "layer": layer,
                        "method": "train_standardized_centroid_linear_probe",
                        "status": "ok",
                        "n_old": support.get(0, 0),
                        "n_new": support.get(1, 0),
                        "train_n": len(train_local),
                        "heldout_n": len(test_local),
                        "heldout_auroc": binary_auroc(labels[test_local], scores),
                        "heldout_auprc": average_precision(labels[test_local], scores),
                    }
                )
        print(f"source probes: {dataset}", flush=True)
    return results


def nuisance_probes(rows: list[dict[str, Any]], features: np.ndarray, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    variables = ("visual_token_count", "aspect_ratio", "image_size_bytes", "question_token_count")
    selected_layers = tuple(int(value) for value in config["selected_layers"])
    folds_count = int(config["nuisance_probe_folds"])
    for dataset in DATASETS:
        indices = [i for i, row in enumerate(rows) if row["dataset"] == dataset]
        groups = [str(rows[i]["image_group_id"]) for i in indices]
        folds = deterministic_group_folds(groups, seed=int(config["seed"]) + 31, n_folds=folds_count)
        for variable in variables:
            raw = np.asarray([float(rows[i][variable]) for i in indices])
            threshold = float(np.median(raw))
            labels = (raw > threshold).astype(np.int64)
            if min(Counter(labels).values(), default=0) < 20:
                continue
            for layer in selected_layers:
                fold_scores = np.full(len(indices), np.nan)
                for fold in range(folds_count):
                    train = np.where(folds != fold)[0]
                    test = np.where(folds == fold)[0]
                    if len(set(labels[train])) < 2 or len(set(labels[test])) < 2:
                        continue
                    probe = fit_centroid_probe(features[np.asarray(indices)[train], layer].astype(np.float32), labels[train])
                    fold_scores[test] = probe.score(features[np.asarray(indices)[test], layer].astype(np.float32))
                output.append(
                    {
                        "dataset": dataset,
                        "population": "historical_and_canonical",
                        "property": variable,
                        "target": f"greater_than_dataset_median_{threshold:g}",
                        "layer": layer,
                        "method": f"{folds_count}_fold_group_disjoint_centroid_linear_probe",
                        "records": int(np.isfinite(fold_scores).sum()),
                        "positive_rate": float(labels.mean()),
                        "cv_auroc": binary_auroc(labels, fold_scores),
                        "cv_auprc": average_precision(labels, fold_scores),
                    }
                )
        print(f"nuisance probes: {dataset}", flush=True)
    return output


def bootstrap_source_effect(rows: list[dict[str, Any]], score_key: str, seed: int, reps: int) -> tuple[float, float, float]:
    old = [row for row in rows if row["source_family"] == "historical_selected_quota"]
    new = [row for row in rows if row["source_family"] == "canonical_outcome_blind"]
    point = float(np.mean([row[score_key] for row in new]) - np.mean([row[score_key] for row in old]))
    old_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    new_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in old:
        old_groups[str(row["image_group_id"])].append(row)
    for row in new:
        new_groups[str(row["image_group_id"])].append(row)
    rng = np.random.default_rng(seed)
    estimates = []
    old_keys, new_keys = list(old_groups), list(new_groups)
    for _ in range(reps):
        a = rng.choice(old_keys, len(old_keys), replace=True)
        b = rng.choice(new_keys, len(new_keys), replace=True)
        old_values = [row[score_key] for key in a for row in old_groups[key]]
        new_values = [row[score_key] for key in b for row in new_groups[key]]
        estimates.append(float(np.mean(new_values) - np.mean(old_values)))
    return point, percentile(estimates, 0.025), percentile(estimates, 0.975)


def metadata_outputs(old: list[dict[str, Any]], new: list[dict[str, Any]], config: Mapping[str, Any], root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows = old + new
    variables = ("visual_token_count", "image_width", "image_height", "image_area", "aspect_ratio", "image_size_bytes", "question_token_count", "prompt_token_count", "answer_token_count")
    distributions: list[dict[str, Any]] = []
    for population in ("historical_train", "historical_val", "historical_test", "canonical_new"):
        for dataset in DATASETS:
            for correctness in ("C", "W"):
                selected = [row for row in all_rows if row["population"] == population and row["dataset"] == dataset and row["correctness"] == correctness]
                for variable in variables:
                    stats = summarize([row[variable] for row in selected])
                    old_train_values = [row[variable] for row in old if row["population"] == "historical_train" and row["dataset"] == dataset and row["correctness"] == correctness]
                    distributions.append({"population": population, "dataset": dataset, "correctness": correctness, "variable": variable, "statistic": "numeric_summary", **stats, "smd_vs_historical_train": standardized_mean_difference(old_train_values, [row[variable] for row in selected])})
                for variable in ("image_format", "source_split", "source_subtype", "source_file"):
                    counts = Counter(str(row[variable]) for row in selected)
                    for category, count in sorted(counts.items()):
                        distributions.append(
                            {
                                "population": population,
                                "dataset": dataset,
                                "correctness": correctness,
                                "variable": variable,
                                "statistic": "category_fraction",
                                "category": category,
                                "n": count,
                                "fraction": count / max(len(selected), 1),
                            }
                        )
    write_csv(root / "metrics/metadata_distribution.csv", distributions)

    token_rows: list[dict[str, Any]] = []
    for key, grouped in _group(all_rows, ("population", "dataset", "correctness")):
        counts = Counter(int(row["visual_token_count"]) for row in grouped)
        total = len(grouped)
        for mode, count in sorted(counts.items()):
            token_rows.append({"population": key[0], "dataset": key[1], "correctness": key[2], "visual_token_count": mode, "records": count, "fraction": count / total})
    write_csv(root / "metrics/visual_token_distribution.csv", token_rows)

    primary: list[dict[str, Any]] = []
    for key, grouped in _group(all_rows, ("population", "dataset", "correctness")):
        primary.append({"population": key[0], "dataset": key[1], "correctness": key[2], "records": len(grouped), "trigger_rate": np.mean([row["triggered"] for row in grouped]), "median_l0_score": np.median([row["score_l0"] for row in grouped]), "median_max_score": np.median([row["score_max"] for row in grouped]), "median_visual_tokens": np.median([row["visual_token_count"] for row in grouped])})
    write_csv(root / "population/old_vs_new_population_summary.csv", primary)

    score_summary: list[dict[str, Any]] = []
    score_keys = [f"score_l{layer}" for layer in config["selected_layers"]] + ["score_max"]
    for key, grouped in _group(all_rows, ("population", "dataset", "correctness")):
        for score_key in score_keys:
            stats = summarize([row[score_key] for row in grouped])
            score_summary.append({"population": key[0], "dataset": key[1], "correctness": key[2], "score": score_key, **stats})
    for dataset in DATASETS:
        for correctness in ("C", "W"):
            subset = [row for row in all_rows if row["dataset"] == dataset and row["correctness"] == correctness and row["population"] in {"historical_train", "canonical_new"}]
            for score_key in score_keys:
                point, low, high = bootstrap_source_effect(subset, score_key, int(config["seed"]) + len(score_summary), int(config["bootstrap_replicates"]))
                score_summary.append({"population": "canonical_minus_historical_train", "dataset": dataset, "correctness": correctness, "score": score_key, "mean_difference": point, "group_bootstrap_ci_low": low, "group_bootstrap_ci_high": high})
    write_csv(root / "metrics/old_head_score_by_population.csv", score_summary)
    return distributions, primary


def _group(rows: Iterable[dict[str, Any]], keys: Sequence[str]) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return sorted(grouped.items(), key=lambda item: tuple(map(str, item[0])))


def score_associations(rows: list[dict[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    variables = ("visual_token_count", "image_width", "image_height", "image_area", "aspect_ratio", "image_size_bytes", "question_token_count", "prompt_token_count", "answer_token_count")
    score_keys = [f"score_l{layer}" for layer in config["selected_layers"]] + ["score_max"]
    output = []
    for key, grouped in _group(rows, ("population", "dataset", "correctness")):
        for score_key in score_keys:
            for variable in variables:
                output.append({"population": key[0], "dataset": key[1], "correctness": key[2], "score": score_key, "nuisance": variable, "records": len(grouped), "spearman_rho": spearman_correlation([row[variable] for row in grouped], [row[score_key] for row in grouped])})
    return output


def nuisance_matrix(rows: list[dict[str, Any]], *, include_dataset: bool) -> tuple[np.ndarray, list[str]]:
    numeric = ["visual_token_count", "image_width", "image_height", "image_area", "aspect_ratio", "image_size_bytes", "question_token_count", "prompt_token_count", "answer_token_count"]
    x = np.asarray([[math.log1p(float(row[key])) if key in {"image_area", "image_size_bytes"} else float(row[key]) for key in numeric] for row in rows], dtype=np.float32)
    names = list(numeric)
    if include_dataset:
        one_hot = np.asarray([[int(row["dataset"] == dataset) for dataset in DATASETS] for row in rows], dtype=np.float32)
        x = np.concatenate([x, one_hot], axis=1)
        names.extend([f"dataset_{dataset}" for dataset in DATASETS])
    return x, names


def nuisance_only_correctness(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dataset in (*DATASETS, "overall"):
        train = [row for row in old if row["population"] == "historical_train" and (dataset == "overall" or row["dataset"] == dataset)]
        x_train, names = nuisance_matrix(train, include_dataset=dataset == "overall")
        y_train = np.asarray([row["dense_wrong"] for row in train])
        probe = fit_centroid_probe(x_train, y_train)
        for evaluation, source in (("historical_val", old), ("historical_test", old), ("canonical_new", new)):
            selected = [row for row in source if row["population"] == evaluation and (dataset == "overall" or row["dataset"] == dataset)]
            x, _ = nuisance_matrix(selected, include_dataset=dataset == "overall")
            y = np.asarray([row["dense_wrong"] for row in selected])
            score = probe.score(x)
            output.append({"dataset": dataset, "train_population": "historical_train", "evaluation_population": evaluation, "method": "train_standardized_centroid_linear_metadata_baseline", "inputs": "+".join(names), "records": len(selected), "correct": int(np.sum(y == 0)), "wrong": int(np.sum(y == 1)), "auroc": binary_auroc(y, score), "auprc": average_precision(y, score), "ranking_direction": "wrong_high" if binary_auroc(y, score) >= 0.5 else "inverted_wrong_low"})
    return output


def head_metrics(rows: list[dict[str, Any]], score_key: str) -> dict[str, float | int]:
    labels = [row["dense_wrong"] for row in rows]
    scores = [row[score_key] for row in rows]
    return {"records": len(rows), "correct": int(sum(label == 0 for label in labels)), "wrong": int(sum(label == 1 for label in labels)), "auroc": binary_auroc(labels, scores), "auprc": average_precision(labels, scores)}


def matched_metrics(old: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    strata_output: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        for dataset in (*DATASETS, "overall"):
            base = [dict(row) for row in old if row["split"] == split and (dataset == "overall" or row["dataset"] == dataset)]
            for row in base:
                row["token_stratum"] = f"{row['dataset']}|vt={row['visual_token_count']}"
                row["multi_stratum"] = f"{row['dataset']}|vt={row['visual_token_count']}|ar={int(np.digitize(row['aspect_ratio'], [0.75, 1.25, 1.75]))}|q={int(np.digitize(row['question_token_count'], [12, 20, 32]))}"
            for method, selected in (
                ("unmatched", base),
                ("exact_visual_token", exact_stratum_match(base, label_key="dense_wrong", stratum_key="token_stratum", seed=seed)),
                ("coarsened_visual_aspect_question", exact_stratum_match(base, label_key="dense_wrong", stratum_key="multi_stratum", seed=seed + 1)),
            ):
                for score_key in ("score_l0", "score_l21", "score_max"):
                    output.append({"split": split, "dataset": dataset, "matching": method, "score": score_key, **head_metrics(selected, score_key), "effective_sample_fraction": len(selected) / max(len(base), 1)})
            if dataset != "overall":
                for key, grouped in _group(base, ("token_stratum",)):
                    counts = Counter(row["dense_wrong"] for row in grouped)
                    if set(counts) == {0, 1} and min(counts.values()) >= 10:
                        strata_output.append({"split": split, "dataset": dataset, "stratum": key[0], "score": "score_max", **head_metrics(grouped, "score_max")})
    return output, strata_output


def normalization_and_geometry(rows: list[dict[str, Any]], features: np.ndarray, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normal = torch.load(resolve(config["sources"]["historical_normalization"]), map_location="cpu", weights_only=False)
    old_mean = normal["mean"].float().numpy()
    old_std = np.maximum(normal["std"].float().numpy(), 1e-6)
    norm_rows: list[dict[str, Any]] = []
    cohorts = ("historical_train", "historical_val", "historical_test", "canonical_new")
    for cohort in cohorts:
        indices = np.asarray([i for i, row in enumerate(rows) if row["population"] == cohort])
        for layer in LAYERS:
            for block_index, block in enumerate(BLOCKS):
                sl = slice(block_index * 3584, (block_index + 1) * 3584)
                x = features[indices, layer, sl].astype(np.float32)
                z = (x - old_mean[sl]) / old_std[sl]
                dimension_mean = z.mean(axis=0)
                raw_std_ratio = x.std(axis=0) / old_std[sl]
                norms = np.linalg.norm(z, axis=1) / math.sqrt(z.shape[1])
                norm_rows.append({"population": cohort, "layer": layer, "feature_block": block, "records": len(indices), "rms_normalized_mean_shift": float(np.sqrt(np.mean(dimension_mean ** 2))), "median_raw_std_ratio_to_old": float(np.median(raw_std_ratio)), "fraction_dimensions_abs_mean_gt2": float(np.mean(np.abs(dimension_mean) > 2)), "fraction_dimensions_abs_mean_gt3": float(np.mean(np.abs(dimension_mean) > 3)), "median_normalized_rms_norm": float(np.median(norms)), "q25_normalized_rms_norm": float(np.quantile(norms, .25)), "q75_normalized_rms_norm": float(np.quantile(norms, .75))})
        print(f"normalization: {cohort}", flush=True)

    # Absolute distance from the pooled normalization is depth-dominated.  Add
    # the decision-relevant, label-conditioned source comparison at identical
    # dataset/layer/block coordinates.
    for reference_population in ("historical_train", "historical_val"):
        for dataset in DATASETS:
            for correctness in ("C", "W"):
                reference_indices = np.asarray(
                    [
                        i
                        for i, row in enumerate(rows)
                        if row["population"] == reference_population
                        and row["dataset"] == dataset
                        and row["correctness"] == correctness
                    ]
                )
                canonical_indices = np.asarray(
                    [
                        i
                        for i, row in enumerate(rows)
                        if row["population"] == "canonical_new"
                        and row["dataset"] == dataset
                        and row["correctness"] == correctness
                    ]
                )
                if not len(reference_indices) or not len(canonical_indices):
                    continue
                for layer in LAYERS:
                    for block_index, block in enumerate(BLOCKS):
                        sl = slice(block_index * 3584, (block_index + 1) * 3584)
                        reference = features[reference_indices, layer, sl].astype(np.float32)
                        canonical = features[canonical_indices, layer, sl].astype(np.float32)
                        reference_mean = ((reference - old_mean[sl]) / old_std[sl]).mean(axis=0)
                        canonical_mean = ((canonical - old_mean[sl]) / old_std[sl]).mean(axis=0)
                        reference_std = reference.std(axis=0)
                        canonical_std = canonical.std(axis=0)
                        norm_rows.append(
                            {
                                "population": f"canonical_new_minus_{reference_population}",
                                "reference_population": reference_population,
                                "dataset": dataset,
                                "correctness": correctness,
                                "layer": layer,
                                "feature_block": block,
                                "records": len(canonical_indices),
                                "reference_records": len(reference_indices),
                                "rms_normalized_mean_difference": float(
                                    np.sqrt(np.mean((canonical_mean - reference_mean) ** 2))
                                ),
                                "median_canonical_to_reference_std_ratio": float(
                                    np.median(canonical_std / np.maximum(reference_std, 1e-6))
                                ),
                                "fraction_dimensions_abs_normalized_mean_difference_gt2": float(
                                    np.mean(np.abs(canonical_mean - reference_mean) > 2)
                                ),
                                "fraction_dimensions_abs_normalized_mean_difference_gt3": float(
                                    np.mean(np.abs(canonical_mean - reference_mean) > 3)
                                ),
                            }
                        )
        print(f"normalization contrasts: {reference_population}", flush=True)

    geometry: list[dict[str, Any]] = []
    for dataset in DATASETS:
        train_c = np.asarray([i for i, row in enumerate(rows) if row["population"] == "historical_train" and row["dataset"] == dataset and row["correctness"] == "C"])
        train_w = np.asarray([i for i, row in enumerate(rows) if row["population"] == "historical_train" and row["dataset"] == dataset and row["correctness"] == "W"])
        for layer in LAYERS:
            c = (features[train_c, layer].astype(np.float32) - old_mean) / old_std
            w = (features[train_w, layer].astype(np.float32) - old_mean) / old_std
            centroid_c, centroid_w = c.mean(axis=0), w.mean(axis=0)
            direction = centroid_w - centroid_c
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            midpoint = 0.5 * (centroid_c + centroid_w)
            for population in ("historical_train", "historical_val", "historical_test", "canonical_new"):
                for correctness in ("C", "W"):
                    indices = np.asarray([i for i, row in enumerate(rows) if row["population"] == population and row["dataset"] == dataset and row["correctness"] == correctness])
                    if not len(indices):
                        continue
                    z = (features[indices, layer].astype(np.float32) - old_mean) / old_std
                    projection = (z - midpoint) @ direction
                    distance_c = np.sqrt(np.mean((z - centroid_c) ** 2, axis=1))
                    distance_w = np.sqrt(np.mean((z - centroid_w) ** 2, axis=1))
                    geometry.append({"dataset": dataset, "layer": layer, "population": population, "correctness": correctness, "records": len(indices), "projection_mean": float(projection.mean()), "projection_median": float(np.median(projection)), "projection_q25": float(np.quantile(projection, .25)), "projection_q75": float(np.quantile(projection, .75)), "rms_distance_to_old_correct_centroid": float(distance_c.mean()), "rms_distance_to_old_wrong_centroid": float(distance_w.mean()), "relative_distance_wrong_minus_correct": float((distance_w - distance_c).mean())})
        print(f"geometry: {dataset}", flush=True)
    return norm_rows, geometry


def construction_outputs(old: list[dict[str, Any]], new: list[dict[str, Any]], config: Mapping[str, Any], root: Path) -> None:
    source_manifest = []
    for row in old:
        source_manifest.append({key: row[key] for key in ("uid", "dataset", "source_file", "source_index", "source_split", "image_content_sha256", "image_group_id", "local_image_path", "question", "dense_wrong", "historical_bucket", "split") } | {"selection_reason": f"fixed_quota_from_previous_dense_{row['historical_bucket']}_bucket"})
    write_jsonl(root / "population/historical_source_manifest.jsonl", source_manifest)

    split_rows = []
    split_groups: dict[str, set[str]] = {}
    split_uids: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        selected = [row for row in old if row["split"] == split]
        split_groups[split] = {str(row["image_group_id"]) for row in selected}
        split_uids[split] = {str(row["uid"]) for row in selected}
        for dataset in DATASETS:
            cell = [row for row in selected if row["dataset"] == dataset]
            split_rows.append({"split": split, "dataset": dataset, "records": len(cell), "correct": sum(row["correctness"] == "C" for row in cell), "wrong": sum(row["correctness"] == "W" for row in cell), "image_groups": len({row["image_group_id"] for row in cell}), "original_source_train": sum(row["source_split"] == "train" for row in cell), "original_source_validation": sum(row["source_split"] == "validation" for row in cell)})
    overlaps = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps.append({"split": f"overlap:{a}:{b}", "dataset": "all", "records": len(split_uids[a] & split_uids[b]), "image_groups": len(split_groups[a] & split_groups[b])})
    write_csv(root / "population/split_reconstruction.csv", split_rows + overlaps)

    requested = {"gqa": 2000, "chartqa": 1000, "textvqa": 1000}
    selection = []
    for dataset in DATASETS:
        for correctness in ("C", "W"):
            selected = [row for row in old if row["dataset"] == dataset and row["correctness"] == correctness]
            selection.append({"dataset": dataset, "correctness": correctness, "requested_records": requested[dataset], "selected_candidates": requested[dataset], "executed_records": len(selected), "selection_basis": "previous_Qwen_dense_outcome_bucket_then_fixed_quota", "natural_preselection_records": "unavailable", "selection_rate": "unavailable", "missing_reason": "one_transferred_ChartQA_correct_image_missing" if dataset == "chartqa" and correctness == "C" else "none"})
    write_csv(root / "population/selection_bias_summary.csv", selection)

    candidate_audit = read_json(resolve(config["sources"]["new_candidate_audit"]))
    construction = f"""# Historical Population Construction

The historical three-task population came from the frozen 10,000-row `complete_correct`/`complete_wrong` manifest; excluding DocVQA leaves fixed quotas of 2,000 C + 2,000 W GQA and 1,000 C + 1,000 W for both ChartQA and TextVQA. Bucket membership was the previous Qwen dense all-on outcome (`source_full_score` against each task threshold), so the nearly 50:50 population is by construction, not measured natural prevalence. One ChartQA-C image was unavailable on this server, leaving 7,999 current executions (3,999 C / 4,000 W).

The portable manifest records the originating dataset split, image SHA-256, question, previous prediction/score, and deterministic source split. The six raw quota files and any larger pre-quota harvesting pool are absent. Consequently the audit can reconstruct the fixed quota and outcome filter exactly but cannot estimate `P(selected | nuisance, correctness)` against a natural preselection population.

Phase 48 later made a new deterministic greedy split with seed `20260830`, stratified by dataset × current dense outcome and grouped by SHA-256 image content. It contains 6,399/800/800 train/val/test rows with zero UID and image-content-group overlap. Every split is a held-out identity sample from the same historical quota/source mechanism; validation success is therefore same-regime generalization, not evidence of robustness to a new source.

The canonical pool differs fundamentally: its {len(new):,} candidates were frozen outcome-blind from pinned canonical training sources, excluded all historical UIDs/content hashes, matched the 2,000/1,000/1,000 task quotas and observable source strata, and were not rebalanced after current dense inference. The candidate audit records {candidate_audit.get('gqa', {}).get('eligible_question_rows', 'unknown')} eligible GQA questions and {candidate_audit.get('chartqa', {}).get('eligible_rows', 'unknown')} final eligible ChartQA rows; current outcomes are allowed to be naturally imbalanced.
"""
    atomic_write(root / "population/historical_construction.md", construction.encode())


def make_figures(root: Path, rows: list[dict[str, Any]], source_probe: list[dict[str, Any]], nuisance_correctness: list[dict[str, Any]], matched: list[dict[str, Any]], geometry: list[dict[str, Any]], normalization: list[dict[str, Any]], failure_metrics_path: Path) -> None:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"historical_train": "#4C78A8", "canonical_new": "#F58518"}
    for variable, name, ylabel in (("visual_token_count", "old_vs_new_visual_tokens.png", "Visual tokens"), ("score_l0", "old_vs_new_l0_scores.png", "Frozen L0 score"), ("score_max", "old_vs_new_max_scores.png", "Frozen max score")):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
        for axis, dataset in zip(axes, DATASETS):
            labels, values = [], []
            for population in ("historical_train", "canonical_new"):
                for correctness in ("C", "W"):
                    selected = [row[variable] for row in rows if row["population"] == population and row["dataset"] == dataset and row["correctness"] == correctness]
                    labels.append(("old" if population.startswith("historical") else "new") + " " + correctness)
                    values.append(selected)
            axis.boxplot(values, tick_labels=labels, showfliers=False)
            axis.set_title(dataset.upper())
            axis.tick_params(axis="x", rotation=35)
            axis.set_ylabel(ylabel)
        fig.tight_layout(); fig.savefig(figures / name, dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, dataset in zip(axes, DATASETS):
        for stratum, style in (("C", "-"), ("W", "--"), ("balanced_CW", ":")):
            points = [row for row in source_probe if row["dataset"] == dataset and row["correctness_stratum"] == stratum and row["status"] == "ok"]
            axis.plot([row["layer"] for row in points], [row["heldout_auroc"] for row in points], style, label=stratum)
        axis.axhline(.5, color="black", lw=.8); axis.set_title(dataset.upper()); axis.set_xlabel("Layer")
    axes[0].set_ylabel("Old-vs-new held-out AUROC"); axes[-1].legend(); fig.tight_layout(); fig.savefig(figures / "source_probe_by_layer.png", dpi=170); plt.close(fig)

    failure = list(csv.DictReader(failure_metrics_path.open()))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, dataset in zip(axes, DATASETS):
        source = [row for row in source_probe if row["dataset"] == dataset and row["correctness_stratum"] == "C" and row["status"] == "ok"]
        old_failure = [row for row in failure if row["split"] == "test" and row["dataset"] == dataset]
        axis.plot([row["layer"] for row in source], [row["heldout_auroc"] for row in source], label="source on C")
        axis.plot([int(row["layer"]) for row in old_failure], [float(row["auroc"]) for row in old_failure], label="old failure")
        axis.axhline(.5, color="black", lw=.8); axis.set_title(dataset.upper()); axis.set_xlabel("Layer")
    axes[0].set_ylabel("AUROC"); axes[-1].legend(); fig.tight_layout(); fig.savefig(figures / "source_vs_failure_predictability.png", dpi=170); plt.close(fig)

    overall = [row for row in nuisance_correctness if row["dataset"] == "overall"]
    fig, axis = plt.subplots(figsize=(6, 4)); axis.bar([row["evaluation_population"] for row in overall], [row["auroc"] for row in overall]); axis.axhline(.5, color="black", lw=.8); axis.set_ylabel("Nuisance-only correctness AUROC"); axis.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(figures / "nuisance_only_generalization.png", dpi=170); plt.close(fig)

    test = [row for row in matched if row["split"] == "test" and row["dataset"] == "overall" and row["score"] == "score_max"]
    fig, axis = plt.subplots(figsize=(7, 4)); axis.bar([row["matching"] for row in test], [row["auroc"] for row in test]); axis.set_ylim(.5, 1); axis.set_ylabel("Frozen max-score AUROC"); axis.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(figures / "matched_vs_unmatched_head_auroc.png", dpi=170); plt.close(fig)

    geo = [row for row in geometry if row["layer"] == 21]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, dataset in zip(axes, DATASETS):
        selected = [row for row in geo if row["dataset"] == dataset and row["population"] in {"historical_train", "canonical_new"}]
        labels = [("old" if row["population"].startswith("historical") else "new") + " " + row["correctness"] for row in selected]
        axis.bar(labels, [row["projection_mean"] for row in selected]); axis.axhline(0, color="black", lw=.8); axis.tick_params(axis="x", rotation=35); axis.set_title(dataset.upper())
    axes[0].set_ylabel("Projection on old C→W direction"); fig.tight_layout(); fig.savefig(figures / "historical_failure_direction_projection.png", dpi=170); plt.close(fig)

    norm = [row for row in normalization if row["feature_block"] == "visual_mean" and row["population"] in {"historical_train", "historical_val", "canonical_new"}]
    fig, axis = plt.subplots(figsize=(7, 4))
    for population in ("historical_train", "historical_val", "canonical_new"):
        points = [row for row in norm if row["population"] == population]
        axis.plot([row["layer"] for row in points], [row["rms_normalized_mean_shift"] for row in points], label=population)
    axis.set_xlabel("Layer"); axis.set_ylabel("RMS normalized mean shift"); axis.legend(); fig.tight_layout(); fig.savefig(figures / "normalization_shift.png", dpi=170); plt.close(fig)


def summary_outputs(root: Path, old: list[dict[str, Any]], new: list[dict[str, Any]], source_probe: list[dict[str, Any]], nuisance_probes_rows: list[dict[str, Any]], nuisance_correctness: list[dict[str, Any]], matched: list[dict[str, Any]], geometry: list[dict[str, Any]], normalization: list[dict[str, Any]]) -> None:
    source_c = {dataset: next(row["heldout_auroc"] for row in source_probe if row["dataset"] == dataset and row["correctness_stratum"] == "C" and row["layer"] == 21 and row["status"] == "ok") for dataset in DATASETS}
    old_test = {dataset: next(row["auroc"] for row in matched if row["split"] == "test" and row["dataset"] == dataset and row["matching"] == "unmatched" and row["score"] == "score_max") for dataset in DATASETS}
    token_test = {dataset: next(row["auroc"] for row in matched if row["split"] == "test" and row["dataset"] == dataset and row["matching"] == "exact_visual_token" and row["score"] == "score_max") for dataset in DATASETS}
    new_head = {dataset: binary_auroc([row["dense_wrong"] for row in new if row["dataset"] == dataset], [row["score_max"] for row in new if row["dataset"] == dataset]) for dataset in DATASETS}
    nuisance_old = {dataset: next(row["auroc"] for row in nuisance_correctness if row["dataset"] == dataset and row["evaluation_population"] == "historical_test") for dataset in DATASETS}
    nuisance_new = {dataset: next(row["auroc"] for row in nuisance_correctness if row["dataset"] == dataset and row["evaluation_population"] == "canonical_new") for dataset in DATASETS}
    nuisance_best = {dataset: max((row["cv_auroc"] for row in nuisance_probes_rows if row["dataset"] == dataset), default=math.nan) for dataset in DATASETS}
    geo_new_c = {dataset: next(row["projection_mean"] for row in geometry if row["dataset"] == dataset and row["layer"] == 21 and row["population"] == "canonical_new" and row["correctness"] == "C") for dataset in DATASETS}
    geo_old_w = {dataset: next(row["projection_mean"] for row in geometry if row["dataset"] == dataset and row["layer"] == 21 and row["population"] == "historical_train" and row["correctness"] == "W") for dataset in DATASETS}
    normalization_contrasts = [
        row
        for row in normalization
        if row["population"] == "canonical_new_minus_historical_val"
    ]
    norm_contrast_max = max(
        row["rms_normalized_mean_difference"] for row in normalization_contrasts
    )
    norm_contrast_median = float(
        np.median(
            [row["rms_normalized_mean_difference"] for row in normalization_contrasts]
        )
    )
    numeric_variables = (
        "visual_token_count",
        "image_width",
        "image_height",
        "image_area",
        "aspect_ratio",
        "image_size_bytes",
        "question_token_count",
        "prompt_token_count",
        "answer_token_count",
    )
    largest_metadata_shift: dict[str, tuple[str, str, float]] = {}
    for dataset in DATASETS:
        candidates: list[tuple[str, str, float]] = []
        for correctness in ("C", "W"):
            historical = [row for row in old if row["population"] == "historical_train" and row["dataset"] == dataset and row["correctness"] == correctness]
            canonical = [row for row in new if row["dataset"] == dataset and row["correctness"] == correctness]
            for variable in numeric_variables:
                candidates.append((correctness, variable, standardized_mean_difference([row[variable] for row in historical], [row[variable] for row in canonical])))
        largest_metadata_shift[dataset] = max(candidates, key=lambda item: abs(item[2]))

    table = ["| Diagnostic | GQA | ChartQA | TextVQA |", "|---|---:|---:|---:|"]
    for label, values in (("Source-ID probe AUROC on C, L21", source_c), ("Best nuisance probe AUROC", nuisance_best), ("Nuisance-only old test C/W AUROC", nuisance_old), ("Frozen head old test AUROC", old_test), ("Frozen head token-matched old test AUROC", token_test), ("Frozen head new AUROC", new_head)):
        table.append(f"| {label} | {values['gqa']:.3f} | {values['chartqa']:.3f} | {values['textvqa']:.3f} |")
    source_strong = float(np.nanmean(list(source_c.values()))) >= .8
    mean_match_drop = float(np.nanmean([old_test[d] - token_test[d] for d in DATASETS]))
    if source_strong and mean_match_drop >= .10:
        decision = "Case A / substantial observable shortcut dependence"
    elif source_strong:
        decision = "Case B with Case-D geometry: source is strongly encoded for ChartQA/TextVQA, but measured nuisance matching does not explain most old ranking"
    else:
        decision = "Case C: measured source probe is weak despite canonical ranking failure"
    shortcut = f"""# Old-Head Shortcut Audit Summary

{os.linesep.join(table)}

## Findings

- Historical-vs-canonical source is evaluated with a train-standardized, group-held-out centroid linear probe using exactly the Shared Random-4 input blocks. At L21 the Dense-C AUROCs are {source_c}. This establishes feature availability, not causal head use.
- The strongest cross-validated nuisance-property AUROCs are {nuisance_best}. Frozen-score associations by source, outcome, layer, and measured nuisance are preserved in `metrics/score_nuisance_association.csv`.
- The nuisance-only old-test/new AUROCs are respectively {nuisance_old} and {nuisance_new}; this tests shortcut opportunity without hidden states.
- Exact visual-token matching changes per-dataset old-test max-score AUROC from {old_test} to {token_test}; mean drop is {mean_match_drop:.3f}. Remaining performance is evidence that visual-token count alone is insufficient.
- At L21 the new-correct projections on the historical C→W direction are {geo_new_c}; historical wrong reference means are {geo_old_w}. This is descriptive decision-geometry shift.
- After conditioning on dataset, correctness, layer, and feature block, canonical-vs-historical-validation RMS normalized mean differences have median {norm_contrast_median:.3f} and maximum {norm_contrast_max:.3f}. This measures raw feature/source shift in frozen-normalized units; it does not show that the normalization transform itself is the dominant cause.

## Decision

The evidence is most consistent with **{decision}**. The old population provided abundant selection/source shortcut opportunity and the fitted boundary is source-sensitive. However, feature availability and score association are observational; visual-token matching cannot prove a causal shortcut, and residual same-regime AUROC means transferable failure information may coexist with nuisance/source dependence.
"""
    atomic_write(root / "summaries/old_head_shortcut_summary.md", shortcut.encode())

    population = f"""# Population Construction Summary

1. The raw authority is the frozen 10K previous-Qwen all-on manifest; the Stage-1 subset removes 2K DocVQA and keeps fixed GQA 4K, ChartQA 2K, TextVQA 2K quotas.
2. Correct/wrong membership was defined by the previous dense all-on task score, then exact per-dataset C/W quotas were consumed. It was not a natural prevalence sample.
3. One missing ChartQA-C image produces the executable 7,999 = 3,999 C + 4,000 W population.
4. Phase-48 train/val/test are 6,399/800/800 group-disjoint identities from the same quota-selected source mechanism. UID and SHA-256 image-content-group overlap are zero.
5. The canonical 4K was selected outcome-blind from pinned canonical training sources, disjoint from the legacy content population, then retained its observed current outcome skew ({sum(not row['dense_wrong'] for row in new):,} C / {sum(row['dense_wrong'] for row in new):,} W).
6. Observable shifts are fully tabulated in `metadata_distribution.csv`, including categorical image format/source strata. The largest absolute numeric standardized mean differences per dataset are {largest_metadata_shift}; none is assumed causal.
7. The original larger pre-quota harvesting pool and six raw quota JSONLs are not present, so natural prevalence and selection rates cannot be reconstructed.
8. Old validation/test success is evidence for same-selection-regime identity generalization; it is not source-shift robustness. Phase-49 and this audit directly show the distinction.
"""
    atomic_write(root / "summaries/population_construction_summary.md", population.encode())

    repair = f"""# Next Stage-1 Repair Recommendation

Do not execute a repair in this phase.

The smallest discriminating next experiment is **same head + frozen old normalization + canonical training labels**. Keep architecture, features, optimizer, split discipline, and evaluation fixed. If this recovers held-out canonical ranking, the old fitted boundary/population regime—not an inherent feature limitation—is sufficient to explain the deployment failure.

Only if that arm fails should a second arm recompute normalization from the canonical training fold. That follow-up distinguishes normalization amplification from a deeper representation/head limitation. A source-balanced old+new mixed fit should be later still; it tests mixture robustness but is not the smallest first diagnostic.

Why this is the smallest defensible action: source is {'strongly' if source_strong else 'not strongly'} linearly available overall (strong for ChartQA/TextVQA, weak for GQA), exact token matching changes old test AUROC by {mean_match_drop:.3f} on average, and the frozen old score shifts sharply for canonical correct ChartQA/TextVQA. A single same-architecture fit directly tests old-boundary failure while preserving the Stage-1 target.
"""
    atomic_write(root / "summaries/next_stage1_repair_recommendation.md", repair.encode())


def write_manifest(root: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            entries.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    manifest = {"schema_version": "stage1_historical_population_shortcut_audit_manifest_v1", "files": entries, "file_count": len(entries), "all_hashes_verified": all(file_sha256(resolve(row["path"])) == row["sha256"] for row in entries)}
    write_json(root / "artifact_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = read_json(config_path)
    if config.get("schema_version") != "stage1_historical_population_shortcut_audit_config_v1":
        raise ValueError("unsupported audit config")
    root = resolve(config["output_root"])
    for directory in ("population", "metrics", "figures", "summaries"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    source_hashes = {name: file_sha256(resolve(path)) for name, path in config["sources"].items()}
    protocol_record = {
        "schema_version": "stage1_historical_population_shortcut_audit_protocol_v1",
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_branch": command("git", "branch", "--show-current"),
        "git_status_porcelain": command("git", "status", "--porcelain=v1", "--untracked-files=all").splitlines(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "config": config,
        "config_sha256": file_sha256(config_path),
        "source_sha256": source_hashes,
        "bound_code_sha256": {path: file_sha256(resolve(path)) for path in BOUND_CODE},
        "rules": {
            "source_probe": "train-standardized nearest-centroid linear discriminant; deterministic SHA group holdout; source positive=new",
            "all_source_probe_balance": "equal counts in old/new × C/W cells before group holdout",
            "nuisance_probes": "dataset-specific median binary target; 3-fold SHA-group-disjoint centroid linear probe",
            "matching": "equal C/W counts within exact dataset+visual-token strata; secondary fixed coarsened aspect/question bins",
            "bootstrap": "independent image-group bootstrap of new-minus-old mean score",
            "geometry": "frozen-normalized historical-train centroid C-to-W direction",
            "interpretation": "feature availability, score association, and matching dependence remain observational",
        },
    }
    protocol_record["protocol_sha256"] = canonical_hash(protocol_record)
    protocol = f"""# Frozen Historical-Population Shortcut Audit Protocol

- Protocol SHA-256: `{protocol_record['protocol_sha256']}`
- Git commit: `{protocol_record['git_commit']}` (dirty worktree recorded in the JSON block below)
- Artifact-only retrospective analysis; no MLLM inference, label regeneration, threshold change, Stage-1/2 training, or corrective search.
- Source-ID probes use the exact 10,752-value Stage-1 state (`text_final + text_mean + visual_mean`), a deterministic image-group held-out split, and a train-standardized nearest-centroid linear discriminant. This is a simple linear accessibility test, not a new Stage-1 model.
- Nuisance matching uses fixed exact visual-token strata and a secondary fixed coarsening: aspect cut points 0.75/1.25/1.75 and question-token cut points 12/20/32.
- All tests condition on current LMMS Dense-C/W labels. Historical buckets are selection provenance only.

```json
{json.dumps(protocol_record, indent=2, sort_keys=True)}
```
"""
    atomic_write(root / "protocol.md", protocol.encode())

    old, new = load_population(config)
    construction_outputs(old, new, config, root)
    distributions, primary = metadata_outputs(old, new, config, root)
    associations = score_associations(old + new, config)
    write_csv(root / "metrics/score_nuisance_association.csv", associations)
    nuisance_correctness = nuisance_only_correctness(old, new)
    write_csv(root / "metrics/nuisance_only_correctness_probe.csv", nuisance_correctness)
    matched, within = matched_metrics(old, int(config["seed"]))
    write_csv(root / "metrics/matched_head_performance.csv", matched)
    write_csv(root / "metrics/within_stratum_head_performance.csv", within)

    print("loading frozen feature tensors", flush=True)
    old_features = load_feature_tensor(old, resolve(config["sources"]["historical_feature_index"]))
    new_features = load_feature_tensor(new, resolve(config["sources"]["new_feature_index"]))
    rows = old + new
    features = np.concatenate([old_features, new_features], axis=0)
    del old_features, new_features
    source_probe = source_probes(rows, features, config)
    write_csv(root / "metrics/source_probe.csv", source_probe)
    nuisance_probe_rows = nuisance_probes(rows, features, config)
    write_csv(root / "metrics/nuisance_probes.csv", nuisance_probe_rows)

    failure_metrics = list(csv.DictReader(resolve(config["sources"]["historical_failure_metrics"]).open()))
    layerwise = []
    for row in source_probe:
        if row["correctness_stratum"] != "C" or row["status"] != "ok":
            continue
        matches = [item for item in failure_metrics if item["split"] == "test" and item["dataset"] == row["dataset"] and int(item["layer"]) == int(row["layer"])]
        layerwise.append({"dataset": row["dataset"], "layer": row["layer"], "source_id_auroc_within_C": row["heldout_auroc"], "historical_failure_test_auroc": float(matches[0]["auroc"]) if matches else math.nan})
    write_csv(root / "metrics/layerwise_source_vs_failure_probe.csv", layerwise)

    normalization, geometry = normalization_and_geometry(rows, features, config)
    write_csv(root / "metrics/normalization_shift.csv", normalization)
    write_csv(root / "metrics/feature_geometry.csv", geometry)
    make_figures(root, rows, source_probe, nuisance_correctness, matched, geometry, normalization, resolve(config["sources"]["historical_failure_metrics"]))
    summary_outputs(root, old, new, source_probe, nuisance_probe_rows, nuisance_correctness, matched, geometry, normalization)
    manifest = write_manifest(root)
    print(json.dumps({"protocol_sha256": protocol_record["protocol_sha256"], "old_records": len(old), "new_records": len(new), "artifacts": manifest["file_count"], "hashes_verified": manifest["all_hashes_verified"]}, indent=2))


if __name__ == "__main__":
    main()
