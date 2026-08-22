#!/usr/bin/env python3
"""Matched-budget cross-dataset visual-access analysis for motivation_check4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from experiments.analyze_mcts_bce_labels import svg_bars, svg_lines
from experiments.analyze_wemath2pro_mcts_labels import validate_record as validate_wemath
from experiments.analyze_wemath_visual_access_placement import route_metrics, select_route_set
from label_regeneration.audit import validate_record as validate_legacy
from label_regeneration.bce_geometry import as_mask


PROJECT = Path(__file__).resolve().parents[1]
LEGACY = PROJECT / "outputs/label_regeneration/v1"
WEMATH = PROJECT / "outputs/label_regeneration/wemath2pro_cap400_v2"
LEGACY_INDEX = LEGACY / "post_generation/cache_record_index_v1.jsonl"
LEGACY_MANIFEST = LEGACY / "source_manifest_v1.jsonl"
WEMATH_INDEX = PROJECT / "outputs/wemath2pro_mcts_label_analysis_v1/cache_record_index_v1.jsonl"
WEMATH_MANIFEST = WEMATH / "manifest/wemath2pro_valid_mcts_v1.jsonl"
WEMATH_COMPLETION = PROJECT / "outputs/wemath2pro_mcts_label_analysis_v1/completion_audit_v1.json"
DEFAULT_OUTPUT = PROJECT / "outputs/cross_dataset_visual_access_v1"

DATASETS = ("gqa", "textvqa", "chartqa", "wemath2pro")
DISPLAY = {"gqa": "GQA", "textvqa": "TextVQA", "chartqa": "ChartQA", "wemath2pro": "WeMath2.0-Pro"}
COLORS = {"gqa": "#3568b8", "textvqa": "#d47a1f", "chartqa": "#2f8f5b", "wemath2pro": "#9a4f9f"}
ALL_ON = (1,) * 28
ALL_OFF = (0,) * 28
B_CORRECT = 200
B_WRONG = 400
DELTAS = (0, 2, 4)
BUDGETS = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28)
COMPACT_BUDGETS = {4, 8, 12, 16, 20, 24, 28}
BUDGET_BINS = ((1, 8, "1-8"), (9, 12, "9-12"), (13, 16, "13-16"),
               (17, 20, "17-20"), (21, 27, "21-27"), (28, 28, "28"))
VISUAL_TOKEN_BINS = ((0, 256, "<=256"), (257, 512, "257-512"),
                     (513, 1024, "513-1024"), (1025, 2048, "1025-2048"),
                     (2049, 10**9, ">2048"))
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260822
PLACEMENT_METRICS = ("first_on", "last_on", "normalized_centroid", "span",
                     "early_fraction", "middle_fraction", "late_fraction",
                     "on_segments", "reentries", "has_late_reentry")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def mean_ci(values: Sequence[float], clusters: Sequence[str], *, seed: int) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[str(cluster)].append(float(value))
    names = sorted(grouped)
    sums = np.asarray([sum(grouped[name]) for name in names], dtype=float)
    counts = np.asarray([len(grouped[name]) for name in names], dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for start in range(0, BOOTSTRAP_DRAWS, 250):
        size = min(250, BOOTSTRAP_DRAWS - start)
        indices = rng.integers(0, len(names), size=(size, len(names)))
        estimates[start:start + size] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def summarize(values: Sequence[float], clusters: Sequence[str], *, seed: int) -> dict[str, Any]:
    if not values:
        return {key: None for key in ("mean", "ci_low", "ci_high", "sd", "p10", "q25",
                                       "median", "q75", "p90", "minimum", "maximum")}
    array = np.asarray(values, dtype=float)
    low, high = mean_ci(array.tolist(), clusters, seed=seed)
    return {
        "mean": float(array.mean()), "ci_low": low, "ci_high": high,
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p10": float(np.quantile(array, 0.10)), "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)), "q75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)), "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def matched_candidates(record: dict[str, Any], budget: int) -> list[dict[str, Any]]:
    """Return anchors plus routes referenced by the first ``budget`` simulations."""
    by_mask = {as_mask(route["visual_on_mask"]): route for route in record["candidate_executions"]}
    if ALL_ON not in by_mask or ALL_OFF not in by_mask:
        raise ValueError("missing FULL/ALL-OFF anchor")
    simulations = record["mcts"]["simulations"]
    if len(simulations) < budget:
        raise ValueError(f"record has {len(simulations)} simulations, requires {budget}")
    masks = [ALL_ON, ALL_OFF]
    masks.extend(as_mask(row["evaluated_mask"]) for row in simulations[:budget])
    unique = dict.fromkeys(masks)
    missing = [mask for mask in unique if mask not in by_mask]
    if missing:
        raise ValueError("matched simulation route missing from candidate cache")
    return [by_mask[mask] for mask in unique]


def classify(full_correct: bool, alloff_correct: bool, positive_valid_masks: Sequence[tuple[int, ...]]) -> str:
    if full_correct:
        return "V0" if alloff_correct else "V+"
    if alloff_correct:
        return "A0"
    return "A+" if positive_valid_masks else "D"


def sample_placement(row: dict[str, Any], masks: Sequence[tuple[int, ...]], delta: int) -> dict[str, Any]:
    minimum = min(sum(mask) for mask in masks)
    selected = select_route_set(masks, minimum_on=minimum, delta=delta)
    route_values = [route_metrics(mask) for mask in selected]
    profile = np.asarray(selected, dtype=float).mean(axis=0)
    result = {
        "dataset": row["dataset"], "uid": row["uid"], "image_group_id": row["image_group_id"],
        "actual_visual_tokens": row["actual_visual_tokens"], "actual_text_tokens": row["actual_text_tokens"],
        "min_positive_on": minimum, "delta": delta, "selected_route_count": len(selected),
    }
    for metric in PLACEMENT_METRICS:
        result[metric] = float(np.mean([item[metric] for item in route_values]))
    for layer, value in enumerate(profile):
        result[f"layer_{layer:02d}"] = float(value)
    return result


def profile_distance(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {
        "l1_distance": float(np.abs(a - b).sum()),
        "l2_distance": float(np.linalg.norm(a - b)),
        "cosine_similarity": float(np.dot(a, b) / denominator) if denominator else math.nan,
    }


def fit_ols(rows: Sequence[dict[str, Any]], outcome: str, covariates: Sequence[str]) -> tuple[np.ndarray, float, list[str]]:
    names = ["intercept", "dataset:textvqa", "dataset:chartqa", "dataset:wemath2pro", *covariates]
    matrix = []
    y = []
    for row in rows:
        matrix.append([1.0, float(row["dataset"] == "textvqa"), float(row["dataset"] == "chartqa"),
                       float(row["dataset"] == "wemath2pro"), *[float(row[name]) for name in covariates]])
        y.append(float(row[outcome]))
    x = np.asarray(matrix, dtype=float)
    target = np.asarray(y, dtype=float)
    beta = np.linalg.lstsq(x, target, rcond=None)[0]
    prediction = x @ beta
    denominator = float(np.square(target - target.mean()).sum())
    r2 = 1.0 - float(np.square(target - prediction).sum()) / denominator if denominator else 0.0
    return beta, r2, names


def bootstrap_ols(rows: Sequence[dict[str, Any]], outcome: str, covariates: Sequence[str], *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["dataset"]][row["image_group_id"]].append(row)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled: list[dict[str, Any]] = []
        for dataset in DATASETS:
            names = sorted(groups[dataset])
            picks = rng.integers(0, len(names), size=len(names))
            for pick in picks:
                sampled.extend(groups[dataset][names[int(pick)]])
        estimates.append(fit_ols(sampled, outcome, covariates)[0])
    array = np.asarray(estimates)
    return np.quantile(array, 0.025, axis=0), np.quantile(array, 0.975, axis=0)


def model_rows(rows: Sequence[dict[str, Any]], outcome: str, covariates: Sequence[str], *, model: str, seed: int) -> list[dict[str, Any]]:
    beta, r2, names = fit_ols(rows, outcome, covariates)
    low, high = bootstrap_ols(rows, outcome, covariates, seed=seed)
    return [{
        "model": model, "outcome": outcome, "coefficient_name": name, "n": len(rows),
        "coefficient": float(beta[index]), "ci_low": float(low[index]), "ci_high": float(high[index]),
        "r_squared": r2, "reference_dataset": "gqa", "interpretation": "descriptive image-cluster bootstrap; not causal",
    } for index, name in enumerate(names)]


def _load_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    legacy = json.loads((LEGACY / "frozen_execution_contract.json").read_text())
    wemath = json.loads((WEMATH / "frozen_execution_contract.json").read_text())
    invariants = ("executor", "attention_implementation", "dtype", "model_revision", "processor_use_fast")
    for key in invariants:
        if legacy[key] != wemath[key]:
            raise RuntimeError(f"executor semantic invariant differs: {key}")
    if legacy["route_semantics"]["num_layers"] != 28 or wemath["route_semantics"]["num_layers"] != 28:
        raise RuntimeError("route length differs")
    for source in ("binary_policy/executor/layers.py", "binary_policy/executor/masks.py", "binary_policy/executor/model.py"):
        if legacy["source_code_sha256"][source] != wemath["source_code_sha256"][source]:
            raise RuntimeError(f"executor source differs: {source}")
    return legacy, wemath


def load_and_validate() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    legacy_contract, wemath_contract = _load_contracts()
    legacy_sources = {row["uid"]: row for row in read_jsonl(LEGACY_MANIFEST)}
    wemath_sources = {row["uid"]: row for row in read_jsonl(WEMATH_MANIFEST)}
    accepted = set(json.loads(WEMATH_COMPLETION.read_text())["accepted_contracts"])
    indices = [("legacy", row) for row in read_jsonl(LEGACY_INDEX)]
    indices += [("wemath", row) for row in read_jsonl(WEMATH_INDEX)]
    rows: list[dict[str, Any]] = []
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    thresholds: dict[str, set[float]] = defaultdict(set)
    budget_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rollups = {dataset: hashlib.sha256() for dataset in DATASETS}
    for position, (source_type, item) in enumerate(indices, start=1):
        path = PROJECT / item["record_path"]
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != item["record_sha256"]:
            raise RuntimeError(f"raw record checksum mismatch: {item['uid']}")
        record = json.loads(payload)
        if source_type == "legacy":
            failures = validate_legacy(record, legacy_sources[item["uid"]], legacy_contract["contract_sha256"])
            if failures:
                raise RuntimeError(f"legacy validation failure {item['uid']}: {failures[:5]}")
            dataset = record["sample"]["benchmark"]
        else:
            validate_wemath(record, wemath_sources[item["uid"]], accepted_contracts=accepted)
            dataset = "wemath2pro"
        rollups[dataset].update(f"{item['uid']}:{digest}\n".encode())
        sample = record["sample"]
        candidates = record["candidate_executions"]
        by_mask = {as_mask(route["visual_on_mask"]): route for route in candidates}
        full_correct = bool(by_mask[ALL_ON]["result_correct"])
        alloff_correct = bool(by_mask[ALL_OFF]["result_correct"])
        budget = B_CORRECT if full_correct else B_WRONG
        primary = matched_candidates(record, budget)
        primary_valid = [as_mask(route["visual_on_mask"]) for route in primary
                         if route["result_correct"] and sum(route["visual_on_mask"]) > 0]
        all_valid = [as_mask(route["visual_on_mask"]) for route in candidates
                     if route["result_correct"] and sum(route["visual_on_mask"]) > 0]
        base = {
            "uid": item["uid"], "dataset": dataset, "image_group_id": sample["image_group_id"],
            "full_correct": full_correct, "alloff_correct": alloff_correct,
            "actual_text_tokens": int(sample["actual_text_tokens"]),
            "actual_visual_tokens": int(sample["actual_visual_tokens"]),
            "reference_answer_whitespace_tokens": max(1, len(str(sample["answer"]).split())),
            "reference_answer_characters": len(str(sample["answer"])),
            "question_characters": len(str(sample["question"])),
            "correctness_threshold": float(sample["correctness_threshold"]),
            "metric_name": sample["metric_name"], "requested_simulations": int(record["mcts"]["requested_simulations"]),
            "completed_simulations": int(record["mcts"]["completed_simulations"]),
            "primary_regime": classify(full_correct, alloff_correct, primary_valid),
            "all_available_regime": classify(full_correct, alloff_correct, all_valid),
            "primary_valid_masks": primary_valid, "all_available_valid_masks": all_valid,
        }
        rows.append(base)
        source_counts[dataset]["records"] += 1
        source_counts[dataset]["evaluated_routes"] += len(candidates)
        source_counts[dataset]["valid_routes"] += sum(route["result_correct"] for route in candidates)
        source_counts[dataset]["full_anchor"] += ALL_ON in by_mask
        source_counts[dataset]["alloff_anchor"] += ALL_OFF in by_mask
        thresholds[dataset].add(float(sample["correctness_threshold"]))
        status = "FULL_correct" if full_correct else "FULL_wrong"
        budget_rows[(dataset, status)].append(record)
        if position % 500 == 0:
            print(f"validated {position}/{len(indices)} raw records", flush=True)
    if Counter(row["dataset"] for row in rows) != Counter({"gqa": 4000, "textvqa": 2000, "chartqa": 2000, "wemath2pro": 4544}):
        raise RuntimeError("source population mismatch")
    integrity_rows = []
    for dataset in DATASETS:
        counts = source_counts[dataset]
        integrity_rows.append({
            "dataset": dataset, "eligible_samples": counts["records"],
            "evaluated_routes_all_available": counts["evaluated_routes"],
            "valid_routes_all_available": counts["valid_routes"],
            "full_anchors": counts["full_anchor"], "alloff_anchors": counts["alloff_anchor"],
            "route_length": 28, "correctness_thresholds": ";".join(map(str, sorted(thresholds[dataset]))),
            "model_revision": legacy_contract["model_revision"], "executor": legacy_contract["executor"],
            "route_semantics": "ON=native visual+text with visual K/V; OFF=visual bypass and text excludes visual K/V",
            "raw_hash_rollup_sha256": rollups[dataset].hexdigest(), "status": "PASS",
        })
    search_rows = []
    for dataset in DATASETS:
        for status in ("FULL_correct", "FULL_wrong"):
            members = budget_rows[(dataset, status)]
            requested = [int(row["mcts"]["requested_simulations"]) for row in members]
            completed = [int(row["mcts"]["completed_simulations"]) for row in members]
            search_rows.append({
                "dataset": dataset, "full_status": status, "n": len(members),
                "requested_distribution": ";".join(f"{key}:{value}" for key, value in sorted(Counter(requested).items())),
                "minimum_completed": min(completed), "maximum_completed": max(completed),
                "anchor_routes_outside_simulations": 2, "early_stopping": False,
                "extension_count": sum(value > (200 if status == "FULL_correct" else 400) for value in requested),
                "matched_budget": B_CORRECT if status == "FULL_correct" else B_WRONG,
                "primary_prefix": f"FULL+ALL_OFF+first_{B_CORRECT if status == 'FULL_correct' else B_WRONG}_simulations",
            })
    return rows, search_rows, {"integrity_rows": integrity_rows, "legacy_contract": legacy_contract,
                               "wemath_contract": wemath_contract}


def build_analysis(sample_rows: list[dict[str, Any]], search_rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    taxonomy_rows, dependence_rows, min_rows, feasibility_rows = [], [], [], []
    placement_rows, profiles_by_delta, distance_rows = [], {delta: [] for delta in DELTAS}, []
    all_placements: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    aplus_taxonomy, aplus_min, aplus_place = [], [], []
    for scope, mask_key, regime_key in (("matched_prefix", "primary_valid_masks", "primary_regime"),
                                         ("all_available", "all_available_valid_masks", "all_available_regime")):
        for dataset_index, dataset in enumerate(DATASETS):
            members = [row for row in sample_rows if row["dataset"] == dataset]
            counts = Counter(row[regime_key] for row in members)
            full_correct = sum(row["full_correct"] for row in members)
            full_wrong = len(members) - full_correct
            taxonomy_rows.append({"scope": scope, "dataset": dataset, "eligible_n": len(members),
                                  "V0": counts["V0"], "V+": counts["V+"], "A0": counts["A0"],
                                  "A+": counts["A+"], "D": counts["D"]})
            if scope == "matched_prefix":
                dependence_rows.append({
                    "dataset": dataset, "eligible_n": len(members), "full_accuracy": full_correct / len(members),
                    "full_correct_n": full_correct, "v0_n": counts["V0"], "v0_given_full_correct": counts["V0"] / full_correct,
                    "vplus_n": counts["V+"], "vplus_given_full_correct": counts["V+"] / full_correct,
                    "alloff_accuracy": sum(row["alloff_correct"] for row in members) / len(members),
                    "unique_images": len({row["image_group_id"] for row in members}),
                })
            vplus = [row for row in members if row[regime_key] == "V+"]
            min_on = [min(map(sum, row[mask_key])) for row in vplus]
            stats = summarize(min_on, [row["image_group_id"] for row in vplus], seed=BOOTSTRAP_SEED + dataset_index)
            min_rows.append({"scope": scope, "dataset": dataset, "vplus_n": len(vplus), **stats,
                             "mean_removable_layers": 28 - stats["mean"] if stats["mean"] is not None else None,
                             "cheaper_than_full_fraction": float(np.mean(np.asarray(min_on) < 28)) if min_on else None})
            for budget in BUDGETS:
                feasibility_rows.append({"scope": scope, "dataset": dataset, "budget": budget, "vplus_n": len(vplus),
                                         "feasible_fraction": float(np.mean(np.asarray(min_on) <= budget)) if min_on else None,
                                         "compact_table_budget": budget in COMPACT_BUDGETS})
            for delta in DELTAS:
                placed = [sample_placement(row, row[mask_key], delta) for row in vplus]
                all_placements[(scope, dataset, delta)] = placed
                summary = {"scope": scope, "dataset": dataset, "delta": delta, "vplus_n": len(placed)}
                for metric_index, metric in enumerate(PLACEMENT_METRICS):
                    metric_stats = summarize([row[metric] for row in placed], [row["image_group_id"] for row in placed],
                                             seed=BOOTSTRAP_SEED + 1000 + dataset_index * 100 + delta * 10 + metric_index)
                    for key, value in metric_stats.items():
                        summary[f"{metric}_{key}"] = value
                placement_rows.append(summary)
                profile = np.asarray([[row[f"layer_{layer:02d}"] for layer in range(28)] for row in placed], dtype=float)
                means = profile.mean(axis=0) if len(profile) else np.full(28, np.nan)
                for layer, value in enumerate(means):
                    profiles_by_delta[delta].append({"scope": scope, "dataset": dataset, "delta": delta,
                                                     "layer": layer, "vplus_n": len(placed),
                                                     "sample_balanced_access_probability": float(value)})
            aplus_taxonomy.append({
                "scope": scope, "dataset": dataset, "full_wrong_n": full_wrong, "a0_n": counts["A0"],
                "a0_fraction": counts["A0"] / full_wrong, "aplus_n": counts["A+"],
                "aplus_fraction": counts["A+"] / full_wrong, "no_correction_n": counts["D"],
                "no_correction_fraction": counts["D"] / full_wrong,
            })
            aplus = [row for row in members if row[regime_key] == "A+"]
            aplus_values = [min(map(sum, row[mask_key])) for row in aplus]
            astats = summarize(aplus_values, [row["image_group_id"] for row in aplus], seed=BOOTSTRAP_SEED + 5000 + dataset_index)
            aplus_min.append({"scope": scope, "dataset": dataset, "aplus_n": len(aplus), **astats})
            placed = [sample_placement(row, row[mask_key], 0) for row in aplus]
            arow = {"scope": scope, "dataset": dataset, "aplus_n": len(placed)}
            for metric_index, metric in enumerate(("normalized_centroid", "last_on", "late_fraction", "on_segments", "has_late_reentry")):
                astat = summarize([row[metric] for row in placed], [row["image_group_id"] for row in placed],
                                  seed=BOOTSTRAP_SEED + 6000 + dataset_index * 10 + metric_index)
                for key, value in astat.items():
                    arow[f"{metric}_{key}"] = value
            for layer in range(28):
                arow[f"layer_{layer:02d}"] = float(np.mean([row[f"layer_{layer:02d}"] for row in placed])) if placed else None
            aplus_place.append(arow)
    for delta in DELTAS:
        profiles = {(row["scope"], row["dataset"]): [item["sample_balanced_access_probability"] for item in profiles_by_delta[delta]
                    if item["scope"] == row["scope"] and item["dataset"] == row["dataset"]]
                    for row in profiles_by_delta[delta]}
        for scope in ("matched_prefix", "all_available"):
            for left, right in combinations(DATASETS, 2):
                distance_rows.append({"scope": scope, "delta": delta, "dataset_left": left, "dataset_right": right,
                                      **profile_distance(profiles[(scope, left)], profiles[(scope, right)])})

    model_output: list[dict[str, Any]] = []
    matched_exact = []
    for dataset in DATASETS:
        matched_exact.extend(all_placements[("matched_prefix", dataset, 0)])
    for delta in DELTAS:
        members = sum((all_placements[("matched_prefix", dataset, delta)] for dataset in DATASETS), [])
        for metric_index, outcome in enumerate(("normalized_centroid", "late_fraction", "has_late_reentry")):
            model_output.extend(model_rows(members, outcome, ("min_positive_on",),
                                           model=f"dataset_plus_min_on_delta{delta}",
                                           seed=BOOTSTRAP_SEED + 8000 + delta * 10 + metric_index))
        for dataset in DATASETS:
            dataset_members = [row for row in members if row["dataset"] == dataset]
            for low, high, label in BUDGET_BINS:
                cell = [row for row in dataset_members if low <= row["min_positive_on"] <= high]
                model_output.append({"model": "fixed_min_on_bin", "delta": delta, "dataset": dataset,
                                     "min_on_bin": label, "n": len(cell), "sparse_cell": len(cell) < 10,
                                     "normalized_centroid_mean": float(np.mean([row["normalized_centroid"] for row in cell])) if cell else None,
                                     "late_fraction_mean": float(np.mean([row["late_fraction"] for row in cell])) if cell else None,
                                     "late_reentry_rate": float(np.mean([row["has_late_reentry"] for row in cell])) if cell else None})

    token_rows: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(DATASETS):
        all_members = [row for row in sample_rows if row["dataset"] == dataset]
        vplus_members = [row for row in all_members if row["primary_regime"] == "V+"]
        for cohort, members in (("all", all_members), ("V+", vplus_members)):
            values = [row["actual_visual_tokens"] for row in members]
            token_rows.append({"row_type": "visual_token_summary", "dataset": dataset, "cohort": cohort,
                               "n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
                               "p90": float(np.quantile(values, .9)), "maximum": max(values)})
        for low, high, label in VISUAL_TOKEN_BINS:
            cell = [row for row in vplus_members if low <= row["actual_visual_tokens"] <= high]
            token_rows.append({"row_type": "visual_token_bin", "dataset": dataset, "cohort": "V+", "visual_token_bin": label,
                               "n": len(cell), "min_positive_on_mean": float(np.mean([min(map(sum, row["primary_valid_masks"])) for row in cell])) if cell else None,
                               "normalized_centroid_mean": float(np.mean([p["normalized_centroid"] for p in matched_exact if p["dataset"] == dataset and p["uid"] in {r["uid"] for r in cell}])) if cell else None})
    for row in matched_exact:
        row["log1p_visual_tokens"] = math.log1p(row["actual_visual_tokens"])
    min_model_rows = [{**row, "min_positive_on": min(map(sum, row["primary_valid_masks"])),
                       "log1p_visual_tokens": math.log1p(row["actual_visual_tokens"])}
                      for row in sample_rows if row["primary_regime"] == "V+"]
    token_rows.extend({"row_type": "model_coefficient", **row} for row in
                      model_rows(min_model_rows, "min_positive_on", ("log1p_visual_tokens",),
                                 model="min_on_dataset_plus_log_visual_tokens", seed=BOOTSTRAP_SEED + 9000))
    token_rows.extend({"row_type": "model_coefficient", **row} for row in
                      model_rows(matched_exact, "normalized_centroid", ("min_positive_on", "log1p_visual_tokens"),
                                 model="centroid_dataset_plus_min_on_plus_log_visual_tokens", seed=BOOTSTRAP_SEED + 9001))

    context_rows = []
    format_context = {
        "gqa": ("open-ended short phrase", "exact match ignoring case/punctuation"),
        "textvqa": ("open-ended with 10 human answers", "EvalAI consensus; >=0.5"),
        "chartqa": ("open-ended/structured chart answer", "relaxed accuracy; exact validity threshold 1.0"),
        "wemath2pro": ("structured math answer in <answer> tags", "MathRuler accuracy"),
    }
    for dataset in DATASETS:
        members = [row for row in sample_rows if row["dataset"] == dataset]
        context_rows.append({
            "dataset": dataset, "n": len(members), "unique_images": len({row["image_group_id"] for row in members}),
            "mean_prompt_text_tokens": float(np.mean([row["actual_text_tokens"] for row in members])),
            "median_prompt_text_tokens": float(np.median([row["actual_text_tokens"] for row in members])),
            "mean_visual_tokens": float(np.mean([row["actual_visual_tokens"] for row in members])),
            "mean_reference_answer_whitespace_tokens": float(np.mean([row["reference_answer_whitespace_tokens"] for row in members])),
            "mean_reference_answer_characters": float(np.mean([row["reference_answer_characters"] for row in members])),
            "answer_format": format_context[dataset][0], "scoring": format_context[dataset][1],
            "correctness_thresholds": ";".join(map(str, sorted({row["correctness_threshold"] for row in members}))),
            "image_count_per_sample": 1, "answer_length_measure": "reference whitespace tokens/chars (tokenizer target IDs unavailable in raw cache)",
        })

    write_csv(output / "search_budget_audit.csv", search_rows)
    write_csv(output / "full_alloff_taxonomy_by_dataset.csv", taxonomy_rows)
    write_csv(output / "visual_dependence_by_dataset.csv", dependence_rows)
    write_csv(output / "vplus_min_on_by_dataset.csv", min_rows)
    write_csv(output / "vplus_budget_feasibility.csv", feasibility_rows)
    write_csv(output / "vplus_placement_by_dataset.csv", placement_rows)
    for delta, filename in ((0, "vplus_layer_profiles_min.csv"), (2, "vplus_layer_profiles_plus2.csv"), (4, "vplus_layer_profiles_plus4.csv")):
        write_csv(output / filename, profiles_by_delta[delta])
    write_csv(output / "profile_distance_matrix.csv", distance_rows)
    write_csv(output / "amount_adjusted_placement_models.csv", model_output)
    write_csv(output / "visual_token_control.csv", token_rows)
    write_csv(output / "dataset_context_audit.csv", context_rows)
    write_csv(output / "aplus_taxonomy_by_dataset.csv", aplus_taxonomy)
    write_csv(output / "aplus_min_on_by_dataset.csv", aplus_min)
    write_csv(output / "aplus_placement_by_dataset.csv", aplus_place)

    dep = {row["dataset"]: row for row in dependence_rows}
    primary_min = {row["dataset"]: row for row in min_rows if row["scope"] == "matched_prefix"}
    primary_place = {(row["dataset"], row["delta"]): row for row in placement_rows if row["scope"] == "matched_prefix"}
    primary_aplus = {row["dataset"]: row for row in aplus_min if row["scope"] == "matched_prefix"}
    svg_bars(figures / "01_visual_dependence_v0_vplus_by_dataset.svg", [DISPLAY[d] for d in DATASETS],
             {"V0/FULL-correct": [dep[d]["v0_given_full_correct"] for d in DATASETS],
              "V+/FULL-correct": [dep[d]["vplus_given_full_correct"] for d in DATASETS]}, "Direct visual dependence by dataset", y_max=1.0)
    svg_bars(figures / "02_vplus_min_positive_on_by_dataset.svg", [DISPLAY[d] for d in DATASETS],
             {"mean minimum ON": [primary_min[d]["mean"] for d in DATASETS]}, "V+ minimum positive visual-ON", y_max=28)
    svg_lines(figures / "03_vplus_budget_feasibility_by_dataset.svg",
              {DISPLAY[d]: [next(row["feasible_fraction"] for row in feasibility_rows if row["scope"] == "matched_prefix" and row["dataset"] == d and row["budget"] == budget) for budget in BUDGETS] for d in DATASETS},
              "V+ budget feasibility (grid index)", y_min=0, y_max=1)
    for delta, number, label in ((0, "04", "exact minimum"), (2, "04b", "minimum +2"), (4, "04c", "minimum +4")):
        svg_lines(figures / f"{number}_vplus_layer_access_profile_{'exact_min' if delta == 0 else 'plus'+str(delta)}.svg",
                  {DISPLAY[d]: [next(row["sample_balanced_access_probability"] for row in profiles_by_delta[delta] if row["scope"] == "matched_prefix" and row["dataset"] == d and row["layer"] == layer) for layer in range(28)] for d in DATASETS},
                  f"V+ layer access profile: {label}", y_min=0, y_max=1)
    svg_bars(figures / "05_vplus_centroid_by_dataset.svg", [DISPLAY[d] for d in DATASETS],
             {"normalized centroid": [primary_place[(d, 0)]["normalized_centroid_mean"] for d in DATASETS]},
             "V+ exact-min normalized centroid", y_max=1)
    svg_bars(figures / "06_vplus_early_middle_late_fraction.svg", [DISPLAY[d] for d in DATASETS],
             {region: [primary_place[(d, 0)][f"{region}_fraction_mean"] for d in DATASETS] for region in ("early", "middle", "late")},
             "V+ exact-min access by decoder third", y_max=1)
    svg_bars(figures / "07_vplus_late_reentry_by_dataset.svg", [DISPLAY[d] for d in DATASETS],
             {"late reentry rate": [primary_place[(d, 0)]["has_late_reentry_mean"] for d in DATASETS]},
             "V+ exact-min late reentry", y_max=1)
    svg_lines(figures / "08_min_plus2_plus4_sensitivity.svg",
              {DISPLAY[d]: [primary_place[(d, delta)]["normalized_centroid_mean"] for delta in DELTAS] for d in DATASETS},
              "Centroid sensitivity (delta 0,2,4)", y_min=0, y_max=1)
    svg_bars(figures / "09_visual_token_count_by_dataset.svg", [DISPLAY[d] for d in DATASETS],
             {"mean visual tokens": [next(row["mean"] for row in token_rows if row.get("row_type") == "visual_token_summary" and row["dataset"] == d and row["cohort"] == "all") for d in DATASETS]},
             "Native visual-token count")
    svg_bars(figures / "10_aplus_min_correcting_on_by_dataset.svg", [DISPLAY[d] for d in DATASETS],
             {"mean minimum correcting ON": [primary_aplus[d]["mean"] for d in DATASETS]},
             "A+ minimum correcting positive ON", y_max=28)
    svg_lines(figures / "11_aplus_layer_access_profile_by_dataset.svg",
              {DISPLAY[d]: [next(row[f"layer_{layer:02d}"] for row in aplus_place if row["scope"] == "matched_prefix" and row["dataset"] == d) for layer in range(28)] for d in DATASETS},
              "A+ minimum correcting-route layer profile", y_min=0, y_max=1)

    return {
        "status": "PASS", "matched_budgets": {"FULL_correct": B_CORRECT, "FULL_wrong": B_WRONG},
        "population": {dataset: sum(row["dataset"] == dataset for row in sample_rows) for dataset in DATASETS},
        "visual_dependence": dep, "vplus_min_on_matched": primary_min,
        "vplus_placement_matched": {f"{dataset}:delta{delta}": primary_place[(dataset, delta)] for dataset in DATASETS for delta in DELTAS},
        "aplus_min_on_matched": primary_aplus,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sample_rows, search_rows, integrity = load_and_validate()
    write_csv(args.output / "source_integrity.csv", integrity["integrity_rows"])
    summary = build_analysis(sample_rows, search_rows, args.output)
    write_json(args.output / "analysis_summary.json", summary)
    sources = {
        "plan": sha256_file(PROJECT / "plans/motivation_check4.md"),
        "analysis_code": sha256_file(Path(__file__)),
        "legacy_index": sha256_file(LEGACY_INDEX), "legacy_contract": sha256_file(LEGACY / "frozen_execution_contract.json"),
        "wemath_index": sha256_file(WEMATH_INDEX), "wemath_contract": sha256_file(WEMATH / "frozen_execution_contract.json"),
    }
    output_hashes = {str(path.relative_to(PROJECT)): sha256_file(path) for path in sorted(args.output.rglob("*"))
                     if path.is_file() and path.name != "analysis_manifest.json"}
    manifest = {
        "schema_version": "cross_dataset_visual_access_analysis_manifest_v1", "status": "PASS",
        "analysis_type": "read-only matched-prefix raw-cache analysis", "datasets": list(DATASETS),
        "route_semantics": {"layers": 28, "ON": "visual and text execute with visual K/V available",
                            "OFF": "visual bypasses and text executes without visual K/V"},
        "matched_search": {"FULL_correct": "anchors + first 200 simulations", "FULL_wrong": "anchors + first 400 simulations"},
        "all_available_sensitivity": True, "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                                                          "cluster": "image_group within dataset"},
        "source_hashes": sources, "output_hashes": output_hashes,
    }
    write_json(args.output / "analysis_manifest.json", manifest)
    print(json.dumps({"status": "PASS", "records": len(sample_rows), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
