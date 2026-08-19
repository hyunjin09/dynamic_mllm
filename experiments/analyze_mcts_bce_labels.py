#!/usr/bin/env python3
"""Deterministic label-only analysis for ``plans/mcts_bce_analysis.md``."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from label_regeneration.bce_geometry import (
    Mask,
    as_mask,
    binary_entropy,
    connected_components,
    diversity_balanced_indices,
    effective_component_count,
    hamming,
    layer_marginals,
    pairwise_distances,
    pareto_efficient_indices,
    polar_route_weights,
    threshold_mask,
    transition_count,
)


PROJECT = Path(__file__).resolve().parents[1]
POST = PROJECT / "outputs/label_regeneration/v1/post_generation"
DEFAULT_OUTPUT = PROJECT / "outputs/binary_mcts_label_geometry_v1"
NUM_LAYERS = 28
ALL_ON: Mask = (1,) * NUM_LAYERS
ALL_OFF: Mask = (0,) * NUM_LAYERS
DATASETS = ("gqa", "textvqa", "chartqa")
RADII = (1, 2, 4)
SELECTION_K = (4, 8, 16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quantile_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "minimum": None, "p10": None, "p25": None,
                "median": None, "p75": None, "p90": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def fmean_defined(values: Any) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return statistics.fmean(defined) if defined else None


def mask_key(mask: Mask) -> str:
    return "".join(map(str, mask))


def mask_entropy(mask_counts: Counter[str]) -> float:
    total = sum(mask_counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in mask_counts.values())


def summarize_masks(masks: list[Mask]) -> dict[str, Any]:
    on_counts = [sum(mask) for mask in masks]
    distances = list(pairwise_distances(masks))
    marginals = layer_marginals(masks) if masks else []
    summary: dict[str, Any] = {
        "route_count": len(masks),
        "unique_route_count": len(set(masks)),
        "duplicate_fraction": 0.0 if not masks else 1.0 - len(set(masks)) / len(masks),
        "on_count": quantile_summary(on_counts),
        "distance_to_all_on": quantile_summary([hamming(mask, ALL_ON) for mask in masks]),
        "pairwise_hamming": quantile_summary(distances),
        "all_on_present": ALL_ON in masks,
        "all_off_present": ALL_OFF in masks,
        "layer_marginals": marginals,
        "mean_bit_entropy": statistics.fmean(binary_entropy(q) for q in marginals) if marginals else None,
    }
    for radius in RADII:
        components = connected_components(masks, radius)
        summary[f"clusters_r{radius}"] = len(components)
        summary[f"largest_cluster_fraction_r{radius}"] = (
            len(components[0]) / len(masks) if masks else None
        )
        summary[f"effective_modes_r{radius}"] = effective_component_count(components)
    return summary


def classify_group(current_status: str, raw_masks: list[Mask]) -> str:
    if current_status == "wrong":
        return "A" if raw_masks else "D"
    if current_status != "correct":
        raise ValueError(f"unexpected current ALL-ON status: {current_status!r}")
    return "B" if any(sum(mask) < NUM_LAYERS for mask in raw_masks) else "C"


def validate_raw_record(record: dict[str, Any], expected_uid: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if record["sample"]["uid"] != expected_uid:
        raise ValueError(f"raw UID mismatch for {expected_uid}")
    candidates = record.get("candidate_executions", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"sample {expected_uid} has no candidate executions")
    for route in candidates:
        mask = as_mask(route["visual_on_mask"])
        if len(mask) != NUM_LAYERS or route["mask_key"] != mask_key(mask):
            raise ValueError(f"malformed route mask for {expected_uid}")
        expected_valid = float(route["score"]) >= float(route["correctness_threshold"])
        if bool(route["result_correct"]) != expected_valid:
            raise ValueError(f"stored validity/threshold mismatch for {expected_uid}")
        if int(route["num_visual_on_layers"]) != sum(mask):
            raise ValueError(f"stored ON count mismatch for {expected_uid}")
    valid = [route for route in candidates if route["result_correct"]]
    return candidates, valid


def oracle_metrics(masks: list[Mask], *, weighted: bool) -> dict[str, Any]:
    if not masks:
        raise ValueError("BCE oracle requires at least one valid mask")
    weights = polar_route_weights(masks) if weighted else None
    marginals = layer_marginals(masks, weights)
    oracle = threshold_mask(marginals)
    nearest = min(hamming(oracle, mask) for mask in masks)
    return {
        "marginals": marginals,
        "entropy": [binary_entropy(value) for value in marginals],
        "mask": oracle,
        "mask_key": mask_key(oracle),
        "on_count": sum(oracle),
        "all_on": oracle == ALL_ON,
        "all_off": oracle == ALL_OFF,
        "valid_hit_at_1": oracle in set(masks),
        "nearest_valid_hamming": nearest,
    }


def sample_record(
    *,
    manifest: dict[str, Any],
    raw_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    uid = manifest["uid"]
    candidates, raw_valid = validate_raw_record(raw_record, uid)
    raw_masks_occurrences = [as_mask(route["visual_on_mask"]) for route in raw_valid]
    raw_unique_routes: dict[Mask, float] = {}
    for route in raw_valid:
        raw_unique_routes.setdefault(as_mask(route["visual_on_mask"]), float(route["score"]))
    raw_masks = sorted(raw_unique_routes, key=mask_key)
    selected_routes = manifest.get("valid_routes", [])
    selected_masks = [as_mask(route["mask"]) for route in selected_routes]
    if len(selected_masks) != len(set(selected_masks)):
        raise ValueError(f"selected manifest contains duplicate masks for {uid}")
    if not set(selected_masks).issubset(set(raw_masks)):
        raise ValueError(f"selected masks are not a subset of raw valid masks for {uid}")
    if len(raw_masks_occurrences) != int(manifest["raw_valid_route_count"]):
        raise ValueError(f"raw valid count mismatch for {uid}")
    if len(selected_masks) != int(manifest["selected_valid_route_count"]):
        raise ValueError(f"selected valid count mismatch for {uid}")

    current_status = raw_record["sample"]["current_all_on_status"]
    if current_status != manifest["current_all_on_status"]:
        raise ValueError(f"current ALL-ON status mismatch for {uid}")
    group = classify_group(current_status, raw_masks)
    base = {
        "uid": uid,
        "dataset": manifest["benchmark"],
        "split": manifest["split"],
        "group": group,
        "current_all_on_status": current_status,
        "raw_evaluated_route_count": len(candidates),
        "raw_valid_route_occurrences": len(raw_masks_occurrences),
        "raw_valid_unique_routes": len(raw_masks),
        "raw_duplicate_fraction": (
            0.0 if not raw_masks_occurrences else 1.0 - len(raw_masks) / len(raw_masks_occurrences)
        ),
        "selected_route_count": len(selected_masks),
        "route_cap_applied": bool(manifest["route_cap_applied"]),
    }
    if not selected_masks:
        base.update({
            "raw_min_on": None,
            "selected_min_on": None,
            "weighted_oracle_key": None,
            "unweighted_oracle_key": None,
        })
        return base, {
            "raw_masks": raw_masks,
            "raw_utilities": raw_unique_routes,
            "selected_masks": selected_masks,
            "selected_utilities": [float(route["score"]) for route in selected_routes],
            "group": group,
        }

    raw_geometry = summarize_masks(raw_masks)
    selected_geometry = summarize_masks(selected_masks)
    weighted_oracle = oracle_metrics(selected_masks, weighted=True)
    unweighted_oracle = oracle_metrics(selected_masks, weighted=False)
    base.update({
        "raw_min_on": raw_geometry["on_count"]["minimum"],
        "raw_median_on": raw_geometry["on_count"]["median"],
        "raw_mean_on": raw_geometry["on_count"]["mean"],
        "raw_mean_pairwise_hamming": raw_geometry["pairwise_hamming"]["mean"],
        "raw_mean_distance_to_all_on": raw_geometry["distance_to_all_on"]["mean"],
        "raw_all_on_present": raw_geometry["all_on_present"],
        "raw_all_off_present": raw_geometry["all_off_present"],
        "raw_mean_bit_entropy": raw_geometry["mean_bit_entropy"],
        "selected_min_on": selected_geometry["on_count"]["minimum"],
        "selected_median_on": selected_geometry["on_count"]["median"],
        "selected_mean_on": selected_geometry["on_count"]["mean"],
        "selected_mean_pairwise_hamming": selected_geometry["pairwise_hamming"]["mean"],
        "selected_mean_distance_to_all_on": selected_geometry["distance_to_all_on"]["mean"],
        "selected_all_on_present": selected_geometry["all_on_present"],
        "selected_all_off_present": selected_geometry["all_off_present"],
        "selected_mean_bit_entropy": selected_geometry["mean_bit_entropy"],
        "weighted_mean_marginal": statistics.fmean(weighted_oracle["marginals"]),
        "weighted_mean_entropy": statistics.fmean(weighted_oracle["entropy"]),
        "weighted_near_tie_fraction": sum(0.45 <= q <= 0.55 for q in weighted_oracle["marginals"]) / NUM_LAYERS,
        "weighted_high_on_fraction": sum(q >= 0.9 for q in weighted_oracle["marginals"]) / NUM_LAYERS,
        "weighted_high_off_fraction": sum(q <= 0.1 for q in weighted_oracle["marginals"]) / NUM_LAYERS,
        "weighted_oracle_key": weighted_oracle["mask_key"],
        "weighted_oracle_on": weighted_oracle["on_count"],
        "weighted_oracle_all_on": weighted_oracle["all_on"],
        "weighted_oracle_all_off": weighted_oracle["all_off"],
        "weighted_oracle_valid": weighted_oracle["valid_hit_at_1"],
        "weighted_oracle_nearest_hamming": weighted_oracle["nearest_valid_hamming"],
        "unweighted_mean_marginal": statistics.fmean(unweighted_oracle["marginals"]),
        "unweighted_mean_entropy": statistics.fmean(unweighted_oracle["entropy"]),
        "unweighted_oracle_key": unweighted_oracle["mask_key"],
        "unweighted_oracle_on": unweighted_oracle["on_count"],
        "unweighted_oracle_all_on": unweighted_oracle["all_on"],
        "unweighted_oracle_all_off": unweighted_oracle["all_off"],
        "unweighted_oracle_valid": unweighted_oracle["valid_hit_at_1"],
        "unweighted_oracle_nearest_hamming": unweighted_oracle["nearest_valid_hamming"],
        "oracle_weighting_mask_disagreement": weighted_oracle["mask"] != unweighted_oracle["mask"],
        "oracle_weighting_hamming": hamming(weighted_oracle["mask"], unweighted_oracle["mask"]),
    })
    for radius in RADII:
        for prefix, geometry in (("raw", raw_geometry), ("selected", selected_geometry)):
            base[f"{prefix}_clusters_r{radius}"] = geometry[f"clusters_r{radius}"]
            base[f"{prefix}_largest_cluster_fraction_r{radius}"] = geometry[f"largest_cluster_fraction_r{radius}"]
            base[f"{prefix}_effective_modes_r{radius}"] = geometry[f"effective_modes_r{radius}"]

    return base, {
        "raw_masks": raw_masks,
        "raw_utilities": raw_unique_routes,
        "selected_masks": selected_masks,
        "selected_utilities": [float(route["score"]) for route in selected_routes],
        "group": group,
        "weighted_oracle": weighted_oracle,
        "unweighted_oracle": unweighted_oracle,
        "raw_geometry": raw_geometry,
        "selected_geometry": selected_geometry,
    }


def stratum_members(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    strata: dict[str, list[dict[str, Any]]] = {"overall": list(rows)}
    for row in rows:
        keys = (
            f"dataset:{row['dataset']}",
            f"split:{row['split']}",
            f"group:{row['group']}",
            f"dataset_group:{row['dataset']}:{row['group']}",
            f"split_dataset:{row['split']}:{row['dataset']}",
        )
        for key in keys:
            strata.setdefault(key, []).append(row)
    return strata


def population_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for stratum, members in stratum_members(rows).items():
        counts = Counter(row["group"] for row in members)
        output.append({
            "stratum": stratum,
            "records": len(members),
            "current_all_on_correct": sum(row["current_all_on_status"] == "correct" for row in members),
            "current_all_on_wrong": sum(row["current_all_on_status"] == "wrong" for row in members),
            "positive_records": sum(row["selected_route_count"] > 0 for row in members),
            "zero_positive_records": sum(row["selected_route_count"] == 0 for row in members),
            **{f"group_{group}": counts.get(group, 0) for group in "ABCD"},
        })
    return output


def route_geometry_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for stratum, members in stratum_members(rows).items():
        for field in (
            "raw_evaluated_route_count", "raw_valid_route_occurrences", "raw_valid_unique_routes",
            "selected_route_count", "raw_min_on", "raw_median_on", "raw_mean_on",
            "selected_min_on", "selected_median_on", "selected_mean_on",
            "raw_mean_pairwise_hamming", "selected_mean_pairwise_hamming",
            "raw_mean_distance_to_all_on", "selected_mean_distance_to_all_on",
            "raw_mean_bit_entropy", "selected_mean_bit_entropy",
        ):
            values = [float(row[field]) for row in members if row.get(field) is not None]
            output.append({"stratum": stratum, "metric": field, **quantile_summary(values)})
    return output


def oracle_summary_table(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    positive = [row for row in rows if row["selected_route_count"] > 0]
    output: list[dict[str, Any]] = []
    for stratum, members in stratum_members(positive).items():
        mask_counts = Counter(row[f"{prefix}_oracle_key"] for row in members)
        output.append({
            "stratum": stratum,
            "oracle": prefix,
            "records": len(members),
            "all_on_fraction": statistics.fmean(row[f"{prefix}_oracle_all_on"] for row in members),
            "all_off_fraction": statistics.fmean(row[f"{prefix}_oracle_all_off"] for row in members),
            "mean_on_layers": statistics.fmean(row[f"{prefix}_oracle_on"] for row in members),
            "unique_masks": len(mask_counts),
            "mask_entropy_nats": mask_entropy(mask_counts),
            "valid_set_hit_at_1": statistics.fmean(row[f"{prefix}_oracle_valid"] for row in members),
            "mean_nearest_valid_hamming": statistics.fmean(row[f"{prefix}_oracle_nearest_hamming"] for row in members),
            "median_nearest_valid_hamming": statistics.median(row[f"{prefix}_oracle_nearest_hamming"] for row in members),
        })
    return output


def layer_marginal_table(
    sample_rows: list[dict[str, Any]], details: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    positive = [row for row in sample_rows if row["selected_route_count"] > 0]
    output: list[dict[str, Any]] = []
    for stratum, members in stratum_members(positive).items():
        for layer in range(NUM_LAYERS):
            raw_values = [details[row["uid"]]["raw_geometry"]["layer_marginals"][layer] for row in members]
            selected_values = [details[row["uid"]]["unweighted_oracle"]["marginals"][layer] for row in members]
            weighted_values = [details[row["uid"]]["weighted_oracle"]["marginals"][layer] for row in members]
            output.append({
                "stratum": stratum,
                "layer": layer,
                "records": len(members),
                "raw_unweighted_q": statistics.fmean(raw_values),
                "selected_unweighted_q": statistics.fmean(selected_values),
                "selected_weighted_q": statistics.fmean(weighted_values),
                "selected_weighted_entropy": statistics.fmean(binary_entropy(value) for value in weighted_values),
                "selected_weighted_near_tie_fraction": statistics.fmean(0.45 <= value <= 0.55 for value in weighted_values),
                "selected_weighted_q_ge_0_9_fraction": statistics.fmean(value >= 0.9 for value in weighted_values),
                "selected_weighted_q_le_0_1_fraction": statistics.fmean(value <= 0.1 for value in weighted_values),
            })
    return output


def add_counterfactuals(
    sample_rows: list[dict[str, Any]], details: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in sample_rows:
        detail = details[row["uid"]]
        selected_masks = detail["selected_masks"]
        if not selected_masks:
            continue
        utilities = detail["selected_utilities"]
        efficient_indices = pareto_efficient_indices(selected_masks, utilities)
        efficient_masks = [selected_masks[index] for index in efficient_indices]
        pareto_oracle = oracle_metrics(efficient_masks, weighted=True)
        all_on_dominated = ALL_ON in selected_masks and selected_masks.index(ALL_ON) not in efficient_indices
        row.update({
            "selected_pareto_efficient_count": len(efficient_masks),
            "selected_pareto_dominated_fraction": 1.0 - len(efficient_masks) / len(selected_masks),
            "selected_all_on_dominated": all_on_dominated,
            "pareto_oracle_key": pareto_oracle["mask_key"],
            "pareto_oracle_on": pareto_oracle["on_count"],
            "pareto_oracle_all_on": pareto_oracle["all_on"],
            "pareto_oracle_valid": pareto_oracle["valid_hit_at_1"],
            "pareto_oracle_nearest_hamming": pareto_oracle["nearest_valid_hamming"],
        })
        output.append({
            "uid": row["uid"], "dataset": row["dataset"], "split": row["split"], "group": row["group"],
            "label_set": "pareto_selected", "route_count": len(efficient_masks),
            "oracle_key": pareto_oracle["mask_key"], "oracle_on": pareto_oracle["on_count"],
            "oracle_all_on": pareto_oracle["all_on"], "oracle_valid": pareto_oracle["valid_hit_at_1"],
            "oracle_nearest_hamming": pareto_oracle["nearest_valid_hamming"],
        })

        raw_masks = detail["raw_masks"]
        for limit in SELECTION_K:
            indices = diversity_balanced_indices(raw_masks, min(limit, len(raw_masks)))
            selected = [raw_masks[index] for index in indices]
            diagnostic = oracle_metrics(selected, weighted=True)
            output.append({
                "uid": row["uid"], "dataset": row["dataset"], "split": row["split"], "group": row["group"],
                "label_set": f"diversity_raw_k{limit}", "route_count": len(selected),
                "oracle_key": diagnostic["mask_key"], "oracle_on": diagnostic["on_count"],
                "oracle_all_on": diagnostic["all_on"], "oracle_valid": diagnostic["valid_hit_at_1"],
                "oracle_nearest_hamming": diagnostic["nearest_valid_hamming"],
            })
    return output


def counterfactual_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    label_sets = sorted({row["label_set"] for row in rows})
    for label_set in label_sets:
        selected = [row for row in rows if row["label_set"] == label_set]
        for stratum, members in stratum_members(selected).items():
            counts = Counter(row["oracle_key"] for row in members)
            output.append({
                "label_set": label_set, "stratum": stratum, "records": len(members),
                "mean_route_count": statistics.fmean(row["route_count"] for row in members),
                "mean_oracle_on": statistics.fmean(row["oracle_on"] for row in members),
                "oracle_all_on_fraction": statistics.fmean(row["oracle_all_on"] for row in members),
                "oracle_valid_hit_at_1": statistics.fmean(row["oracle_valid"] for row in members),
                "mean_nearest_hamming": statistics.fmean(row["oracle_nearest_hamming"] for row in members),
                "unique_oracle_masks": len(counts), "oracle_mask_entropy_nats": mask_entropy(counts),
            })
    return output


def balance_pressure_table(
    sample_rows: list[dict[str, Any]], details: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    positive = [row for row in sample_rows if row["selected_route_count"] > 0]
    wrong = [row for row in positive if row["group"] == "A"]
    correct = [row for row in positive if row["group"] in ("B", "C")]
    if not wrong or not correct:
        raise ValueError("balance counterfactual requires both correction and FULL-correct examples")
    output: list[dict[str, Any]] = []
    scenarios = (("observed", len(wrong) / len(correct)), ("1:1", 1.0), ("2:1", 2.0), ("3:1", 3.0))
    for name, ratio in scenarios:
        wrong_weight = ratio / len(wrong)
        correct_weight = 1.0 / len(correct)
        denominator = ratio + 1.0
        global_q = []
        for layer in range(NUM_LAYERS):
            numerator = sum(
                wrong_weight * details[row["uid"]]["weighted_oracle"]["marginals"][layer]
                for row in wrong
            ) + sum(
                correct_weight * details[row["uid"]]["weighted_oracle"]["marginals"][layer]
                for row in correct
            )
            global_q.append(numerator / denominator)
        decoded = threshold_mask(global_q)
        all_on_probability = math.exp(sum(math.log(max(value, 1e-300)) for value in global_q))
        for layer, value in enumerate(global_q):
            output.append({
                "scenario": name, "wrong_to_correct_mass_ratio": ratio, "layer": layer,
                "global_weighted_on_marginal": value,
                "mean_global_on_probability": statistics.fmean(global_q),
                "decoded_global_on_layers": sum(decoded),
                "factorized_global_all_on_probability": all_on_probability,
            })
    return output


def route_frequency_rows(
    sample_rows: list[dict[str, Any]], details: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    positive = [row for row in sample_rows if row["selected_route_count"] > 0]
    output: list[dict[str, Any]] = []
    for dataset in ("overall", *DATASETS):
        members = positive if dataset == "overall" else [row for row in positive if row["dataset"] == dataset]
        route_sets = {
            "selected_occurrences": [mask for row in members for mask in details[row["uid"]]["selected_masks"]],
            "minimum_on_representative": [
                min(details[row["uid"]]["selected_masks"], key=lambda mask: (sum(mask), mask_key(mask)))
                for row in members
            ],
            "pareto_representative": [
                min(
                    (
                        details[row["uid"]]["selected_masks"][index]
                        for index in pareto_efficient_indices(
                            details[row["uid"]]["selected_masks"],
                            details[row["uid"]]["selected_utilities"],
                        )
                    ),
                    key=lambda mask: (sum(mask), mask_key(mask)),
                )
                for row in members
            ],
        }
        for route_set, masks in route_sets.items():
            counts = Counter(mask_key(mask) for mask in masks)
            total = sum(counts.values())
            ranked = counts.most_common()
            row = {
                "dataset": dataset, "route_set": route_set, "records": len(members),
                "route_occurrences": total, "unique_masks": len(counts),
                "mask_entropy_nats": mask_entropy(counts),
                "all_on_occurrence_fraction": counts.get(mask_key(ALL_ON), 0) / total if total else 0.0,
            }
            for k in (1, 5, 10, 50):
                row[f"top_{k}_occurrence_coverage"] = sum(count for _, count in ranked[:k]) / total if total else 0.0
            output.append(row)

        selected_sample_sets = [set(map(mask_key, details[row["uid"]]["selected_masks"])) for row in members]
        selected_occurrence_counts = Counter(mask_key(mask) for row in members for mask in details[row["uid"]]["selected_masks"])
        ranked_keys = [key for key, _ in selected_occurrence_counts.most_common()]
        sample_coverage = {
            k: statistics.fmean(bool(sample_set.intersection(ranked_keys[:k])) for sample_set in selected_sample_sets)
            for k in (1, 5, 50)
        }
        output.append({
            "dataset": dataset, "route_set": "selected_sample_coverage", "records": len(members),
            "route_occurrences": sum(map(len, selected_sample_sets)), "unique_masks": len(selected_occurrence_counts),
            "mask_entropy_nats": mask_entropy(selected_occurrence_counts),
            "all_on_occurrence_fraction": statistics.fmean(mask_key(ALL_ON) in masks for masks in selected_sample_sets),
            "top_1_occurrence_coverage": sample_coverage[1],
            "top_5_occurrence_coverage": sample_coverage[5],
            "top_10_occurrence_coverage": None,
            "top_50_occurrence_coverage": sample_coverage[50],
        })
    return output


def predictor_aggregate_comparison(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use frozen aggregate histories; no per-record BCE predictions were saved."""
    validation = [row for row in sample_rows if row["split"] == "validation" and row["selected_route_count"] > 0]
    oracle = next(row for row in oracle_summary_table(validation, "weighted") if row["stratum"] == "overall")
    output: list[dict[str, Any]] = []
    for modality in ("question", "image_question"):
        history_path = PROJECT / f"outputs/binary_polar/full10_bce/{modality}_v1/history.json"
        history = json.loads(history_path.read_text())
        best = max(history, key=lambda epoch: epoch["validation"]["overall"]["top1_valid_route_coverage"])
        actual = best["validation"]["overall"]
        output.append({
            "modality": modality,
            "checkpoint_epoch": best["epoch"],
            "records": actual["examples"],
            "actual_mean_on": actual["average_predicted_visual_on"],
            "actual_all_on_fraction": actual["fraction_top1_all_on"],
            "actual_all_off_fraction": actual["fraction_top1_all_off"],
            "actual_unique_masks": len(actual["top1_mask_counts"]),
            "actual_mask_entropy_nats": actual["top1_mask_entropy_nats"],
            "actual_valid_set_hit_at_1": actual["top1_valid_route_coverage"],
            "actual_nearest_valid_hamming": actual["nearest_valid_hamming"],
            "oracle_mean_on": oracle["mean_on_layers"],
            "oracle_all_on_fraction": oracle["all_on_fraction"],
            "oracle_unique_masks": oracle["unique_masks"],
            "oracle_mask_entropy_nats": oracle["mask_entropy_nats"],
            "oracle_valid_set_hit_at_1": oracle["valid_set_hit_at_1"],
            "oracle_nearest_valid_hamming": oracle["mean_nearest_valid_hamming"],
            "per_record_comparison_available": False,
            "unavailable_reason": "full10 BCE histories retain aggregate validation masks only, not UID-level predictions",
        })
    return output


def _svg_frame(title: str, body: str, *, width: int = 1000, height: int = 560) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="24" y="32" font-family="sans-serif" font-size="20" font-weight="bold">{title}</text>'
        f'{body}</svg>\n'
    )


def svg_histogram(path: Path, values: list[float], title: str, *, bins: int = 29) -> None:
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=bins)
    maximum = max(int(counts.max()), 1)
    left, top, plot_width, plot_height = 70, 60, 880, 420
    bars = []
    for index, count in enumerate(counts):
        x = left + index * plot_width / len(counts)
        width = plot_width / len(counts) - 1
        height = plot_height * count / maximum
        bars.append(f'<rect x="{x:.2f}" y="{top + plot_height - height:.2f}" width="{width:.2f}" height="{height:.2f}" fill="#3b82f6"/>')
    labels = (
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>'
        f'<text x="{left}" y="{top + plot_height + 24}" font-family="sans-serif" font-size="12">{edges[0]:.2f}</text>'
        f'<text x="{left + plot_width - 45}" y="{top + plot_height + 24}" font-family="sans-serif" font-size="12">{edges[-1]:.2f}</text>'
        f'<text x="15" y="{top + 10}" font-family="sans-serif" font-size="12">max n={maximum}</text>'
    )
    path.write_text(_svg_frame(title, "".join(bars) + labels))


def svg_lines(path: Path, series: dict[str, list[float]], title: str, *, y_min: float = 0.0, y_max: float = 1.0) -> None:
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c")
    left, top, plot_width, plot_height = 70, 60, 880, 420
    elements = [
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
    ]
    for series_index, (name, values) in enumerate(series.items()):
        color = colors[series_index % len(colors)]
        points = []
        for index, value in enumerate(values):
            x = left + plot_width * index / max(len(values) - 1, 1)
            y = top + plot_height * (y_max - value) / max(y_max - y_min, 1e-12)
            points.append(f"{x:.2f},{y:.2f}")
        elements.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        elements.append(f'<text x="{left + 150 * series_index}" y="535" fill="{color}" font-family="sans-serif" font-size="12">{name}</text>')
    elements.extend([
        f'<text x="20" y="{top + 5}" font-family="sans-serif" font-size="12">{y_max:.2f}</text>',
        f'<text x="20" y="{top + plot_height}" font-family="sans-serif" font-size="12">{y_min:.2f}</text>',
        f'<text x="{left}" y="{top + plot_height + 22}" font-family="sans-serif" font-size="12">layer 0</text>',
        f'<text x="{left + plot_width - 50}" y="{top + plot_height + 22}" font-family="sans-serif" font-size="12">layer 27</text>',
    ])
    path.write_text(_svg_frame(title, "".join(elements)))


def svg_bars(path: Path, labels: list[str], series: dict[str, list[float]], title: str, *, y_max: float | None = None) -> None:
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")
    maximum = y_max if y_max is not None else max(max(values) for values in series.values()) * 1.05
    maximum = max(maximum, 1e-12)
    left, top, plot_width, plot_height = 70, 60, 880, 400
    group_width = plot_width / max(len(labels), 1)
    bar_width = group_width / (len(series) + 1)
    elements = []
    for series_index, (name, values) in enumerate(series.items()):
        color = colors[series_index % len(colors)]
        for index, value in enumerate(values):
            height = plot_height * value / maximum
            x = left + index * group_width + series_index * bar_width
            elements.append(f'<rect x="{x:.2f}" y="{top + plot_height - height:.2f}" width="{bar_width - 2:.2f}" height="{height:.2f}" fill="{color}"/>')
        elements.append(f'<text x="{left + series_index * 180}" y="535" fill="{color}" font-family="sans-serif" font-size="12">{name}</text>')
    for index, label in enumerate(labels):
        elements.append(f'<text x="{left + index * group_width}" y="{top + plot_height + 20}" font-family="sans-serif" font-size="11">{label}</text>')
    path.write_text(_svg_frame(title, "".join(elements)))


def create_figures(
    output_dir: Path,
    sample_rows: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
    predictor_rows: list[dict[str, Any]],
    counter_rows: list[dict[str, Any]],
) -> list[Path]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    positive = [row for row in sample_rows if row["selected_route_count"] > 0]
    paths: list[Path] = []

    def histogram(name: str, values: list[float], title: str, bins: int = 29) -> None:
        path = figures / name; svg_histogram(path, values, title, bins=bins); paths.append(path)

    histogram("01_valid_route_count.svg", [row["raw_valid_unique_routes"] for row in sample_rows], "Raw valid-route count per sample", 40)
    histogram("02_minimum_route_on_count.svg", [row["raw_min_on"] for row in positive], "Minimum valid-route VISUAL_ON layers", 29)
    histogram("03_median_route_on_count.svg", [row["raw_median_on"] for row in positive], "Median valid-route VISUAL_ON layers", 29)
    histogram("04_pairwise_hamming.svg", [row["raw_mean_pairwise_hamming"] for row in positive if row["raw_mean_pairwise_hamming"] is not None], "Per-sample mean pairwise Hamming", 29)
    histogram("07_bce_oracle_on_count.svg", [row["weighted_oracle_on"] for row in positive], "Weighted BCE-label-oracle VISUAL_ON layers", 29)
    histogram("10_bce_oracle_nearest_hamming.svg", [row["weighted_oracle_nearest_hamming"] for row in positive], "BCE oracle nearest selected-valid Hamming", 29)

    overall_layers = [row for row in layer_rows if row["stratum"] == "overall"]
    path = figures / "05_layer_marginals_raw_selected.svg"
    svg_lines(path, {
        "raw": [row["raw_unweighted_q"] for row in overall_layers],
        "selected": [row["selected_unweighted_q"] for row in overall_layers],
        "weighted": [row["selected_weighted_q"] for row in overall_layers],
    }, "Global per-layer ON marginals"); paths.append(path)
    path = figures / "06_group_layer_marginals.svg"
    svg_lines(path, {
        group: [row["selected_weighted_q"] for row in layer_rows if row["stratum"] == f"group:{group}"]
        for group in ("A", "B", "C")
        if any(row["stratum"] == f"group:{group}" for row in layer_rows)
    }, "Weighted ON marginals by population group"); paths.append(path)

    validation_oracle = next(row for row in oracle_summary_table(sample_rows, "weighted") if row["stratum"] == "split:validation")
    path = figures / "08_actual_vs_oracle_on_count.svg"
    svg_bars(path, [row["modality"] for row in predictor_rows], {
        "actual": [row["actual_mean_on"] for row in predictor_rows],
        "label oracle": [validation_oracle["mean_on_layers"] for _ in predictor_rows],
    }, "Actual full10 BCE vs label-oracle mean ON", y_max=28); paths.append(path)
    path = figures / "09_actual_vs_oracle_all_on.svg"
    svg_bars(path, [row["modality"] for row in predictor_rows], {
        "actual": [row["actual_all_on_fraction"] for row in predictor_rows],
        "label oracle": [validation_oracle["all_on_fraction"] for _ in predictor_rows],
    }, "Actual full10 BCE vs label-oracle ALL-ON fraction", y_max=1); paths.append(path)

    path = figures / "11_raw_vs_selected.svg"
    svg_bars(path, ["mean ON", "pairwise H", "entropy"], {
        "raw": [statistics.fmean(row["raw_mean_on"] for row in positive), fmean_defined(row["raw_mean_pairwise_hamming"] for row in positive), statistics.fmean(row["raw_mean_bit_entropy"] for row in positive)],
        "selected": [statistics.fmean(row["selected_mean_on"] for row in positive), fmean_defined(row["selected_mean_pairwise_hamming"] for row in positive), statistics.fmean(row["selected_mean_bit_entropy"] for row in positive)],
    }, "Raw versus max-50 selected geometry"); paths.append(path)

    pareto = [row for row in counter_rows if row["label_set"] == "pareto_selected"]
    path = figures / "12_pareto_oracle_comparison.svg"
    svg_bars(path, ["mean ON", "ALL-ON", "Hit@1"], {
        "original": [statistics.fmean(row["weighted_oracle_on"] for row in positive), statistics.fmean(row["weighted_oracle_all_on"] for row in positive), statistics.fmean(row["weighted_oracle_valid"] for row in positive)],
        "Pareto": [statistics.fmean(row["oracle_on"] for row in pareto), statistics.fmean(row["oracle_all_on"] for row in pareto), statistics.fmean(row["oracle_valid"] for row in pareto)],
    }, "Original versus Pareto-filtered BCE oracle", y_max=28); paths.append(path)
    return paths


def cluster_summary_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positive = [row for row in rows if row["selected_route_count"] > 0]
    output: list[dict[str, Any]] = []
    for stratum, members in stratum_members(positive).items():
        for source in ("raw", "selected"):
            for radius in RADII:
                output.append({
                    "stratum": stratum, "source": source, "radius": radius, "records": len(members),
                    "mean_cluster_count": statistics.fmean(row[f"{source}_clusters_r{radius}"] for row in members),
                    "median_cluster_count": statistics.median(row[f"{source}_clusters_r{radius}"] for row in members),
                    "mean_largest_cluster_fraction": statistics.fmean(row[f"{source}_largest_cluster_fraction_r{radius}"] for row in members),
                    "mean_effective_modes": statistics.fmean(row[f"{source}_effective_modes_r{radius}"] for row in members),
                })
    return output


def invalid_hybrid_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positive = [row for row in rows if row["selected_route_count"] > 0]
    output: list[dict[str, Any]] = []
    for stratum, members in stratum_members(positive).items():
        for valid in (True, False):
            subset = [row for row in members if row["weighted_oracle_valid"] is valid]
            if not subset:
                continue
            output.append({
                "stratum": stratum, "oracle_valid": valid, "records": len(subset),
                "mean_oracle_on": statistics.fmean(row["weighted_oracle_on"] for row in subset),
                "mean_nearest_hamming": statistics.fmean(row["weighted_oracle_nearest_hamming"] for row in subset),
                "mean_selected_bit_entropy": statistics.fmean(row["weighted_mean_entropy"] for row in subset),
                "mean_near_tie_fraction": statistics.fmean(row["weighted_near_tie_fraction"] for row in subset),
                "mean_selected_routes": statistics.fmean(row["selected_route_count"] for row in subset),
                "mean_effective_modes_r2": statistics.fmean(row["selected_effective_modes_r2"] for row in subset),
                "mean_largest_cluster_fraction_r2": statistics.fmean(row["selected_largest_cluster_fraction_r2"] for row in subset),
            })
    return output


def pareto_summary_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positive = [row for row in rows if row["selected_route_count"] > 0]
    output: list[dict[str, Any]] = []
    for stratum, members in stratum_members(positive).items():
        all_on_present = [row for row in members if row["selected_all_on_present"]]
        output.append({
            "stratum": stratum, "records": len(members),
            "total_selected_routes": sum(row["selected_route_count"] for row in members),
            "total_pareto_efficient_routes": sum(row["selected_pareto_efficient_count"] for row in members),
            "route_weighted_dominated_fraction": 1.0 - (
                sum(row["selected_pareto_efficient_count"] for row in members)
                / sum(row["selected_route_count"] for row in members)
            ),
            "mean_efficient_routes": statistics.fmean(row["selected_pareto_efficient_count"] for row in members),
            "median_efficient_routes": statistics.median(row["selected_pareto_efficient_count"] for row in members),
            "mean_dominated_fraction": statistics.fmean(row["selected_pareto_dominated_fraction"] for row in members),
            "all_on_present_records": len(all_on_present),
            "all_on_dominated_fraction_when_present": (
                statistics.fmean(row["selected_all_on_dominated"] for row in all_on_present)
                if all_on_present else None
            ),
        })
    return output


def raw_selected_summary_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positive = [row for row in rows if row["selected_route_count"] > 0]
    output: list[dict[str, Any]] = []
    for stratum, members in stratum_members(positive).items():
        output.append({
            "stratum": stratum, "records": len(members),
            "mean_raw_routes": statistics.fmean(row["raw_valid_unique_routes"] for row in members),
            "mean_selected_routes": statistics.fmean(row["selected_route_count"] for row in members),
            "mean_raw_on": statistics.fmean(row["raw_mean_on"] for row in members),
            "mean_selected_on": statistics.fmean(row["selected_mean_on"] for row in members),
            "mean_raw_pairwise_hamming": fmean_defined(row["raw_mean_pairwise_hamming"] for row in members),
            "mean_selected_pairwise_hamming": fmean_defined(row["selected_mean_pairwise_hamming"] for row in members),
            "mean_raw_entropy": statistics.fmean(row["raw_mean_bit_entropy"] for row in members),
            "mean_selected_entropy": statistics.fmean(row["selected_mean_bit_entropy"] for row in members),
            "raw_all_on_presence": statistics.fmean(row["raw_all_on_present"] for row in members),
            "selected_all_on_presence": statistics.fmean(row["selected_all_on_present"] for row in members),
            "route_cap_fraction": statistics.fmean(row["route_cap_applied"] for row in members),
        })
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "cache_index": POST / "cache_record_index_v1.jsonl",
        "selected_manifest": POST / "binary_predictor_manifest_v1.jsonl",
        "split_manifest": POST / "predictor_split_manifest_v1.jsonl",
        "p4_audit": POST / "cache_audit_v1.json",
        "p8_audit": POST / "derived_supervision_audit_v1.json",
        "bce_config": PROJECT / "configs/binary_polar_full10_polar_bce_v1.yaml",
        "dataset_code": PROJECT / "binary_policy/dataset.py",
        "decode_code": PROJECT / "binary_policy/decode.py",
        "analysis_code": PROJECT / "experiments/analyze_mcts_bce_labels.py",
        "geometry_code": PROJECT / "label_regeneration/bce_geometry.py",
        "source_plan": PROJECT / "plans/mcts_bce_analysis.md",
    }
    index_rows = load_jsonl(source_paths["cache_index"])
    manifest_rows = load_jsonl(source_paths["selected_manifest"])
    split_rows = load_jsonl(source_paths["split_manifest"])
    if len(index_rows) != 8000 or len(manifest_rows) != 8000 or len(split_rows) != 8000:
        raise ValueError("expected exactly 8,000 rows in cache, selected, and split manifests")
    index_by_uid = {row["uid"]: row for row in index_rows}
    manifest_by_uid = {row["uid"]: row for row in manifest_rows}
    split_by_uid = {row["uid"]: row for row in split_rows}
    if not (set(index_by_uid) == set(manifest_by_uid) == set(split_by_uid)):
        raise ValueError("UID sets differ across frozen source manifests")
    if len(index_by_uid) != 8000:
        raise ValueError("duplicate UID in frozen source manifests")

    sample_rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    for position, index in enumerate(index_rows, start=1):
        uid = index["uid"]
        raw_path = PROJECT / index["record_path"]
        payload = raw_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != index["record_sha256"]:
            raise ValueError(f"raw-record checksum mismatch for {uid}")
        raw_record = json.loads(payload)
        manifest = manifest_by_uid[uid]
        if manifest["split"] != split_by_uid[uid]["split"]:
            raise ValueError(f"split mismatch for {uid}")
        row, detail = sample_record(manifest=manifest, raw_record=raw_record)
        sample_rows.append(row)
        details[uid] = detail
        if position % 250 == 0:
            print(f"analyzed {position}/8000", flush=True)

    if Counter(row["dataset"] for row in sample_rows) != Counter({"gqa": 4000, "textvqa": 2000, "chartqa": 2000}):
        raise ValueError("dataset counts differ from frozen 4K/2K/2K contract")
    if Counter(row["split"] for row in sample_rows) != Counter({"train": 7000, "validation": 1000}):
        raise ValueError("split counts differ from frozen 7K/1K contract")

    counter_rows = add_counterfactuals(sample_rows, details)
    tables = {
        "population_taxonomy.csv": population_table(sample_rows),
        "route_geometry_summary.csv": route_geometry_table(sample_rows),
        "layer_marginals.csv": layer_marginal_table(sample_rows, details),
        "weighted_bce_oracle_summary.csv": oracle_summary_table(sample_rows, "weighted"),
        "unweighted_bce_oracle_summary.csv": oracle_summary_table(sample_rows, "unweighted"),
        "invalid_hybrid_summary.csv": invalid_hybrid_table(sample_rows),
        "cluster_summary.csv": cluster_summary_table(sample_rows),
        "raw_selected_summary.csv": raw_selected_summary_table(sample_rows),
        "pareto_summary.csv": pareto_summary_table(sample_rows),
        "counterfactual_oracle_summary.csv": counterfactual_summary(counter_rows),
        "balance_pressure.csv": balance_pressure_table(sample_rows, details),
        "cross_sample_route_diversity.csv": route_frequency_rows(sample_rows, details),
        "actual_predictor_aggregate_comparison.csv": predictor_aggregate_comparison(sample_rows),
    }
    for filename, rows in tables.items():
        write_csv(output_dir / filename, rows)
    write_jsonl(output_dir / "per_sample_geometry.jsonl", sample_rows)
    write_jsonl(output_dir / "counterfactual_oracles.jsonl", counter_rows)
    write_json(output_dir / "cross_sample_interchangeability.json", {
        "status": "not_executed",
        "reason": "The raw cache records only routes evaluated on their own sample; it does not contain a systematic donor-route-by-target execution matrix. New Qwen execution is forbidden by the label-only plan.",
    })
    layer_rows = tables["layer_marginals.csv"]
    predictor_rows = tables["actual_predictor_aggregate_comparison.csv"]
    figure_paths = create_figures(output_dir, sample_rows, layer_rows, predictor_rows, counter_rows)

    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    output_paths = sorted(path for path in output_dir.rglob("*") if path.is_file())
    output_hashes = {str(path.relative_to(PROJECT)): sha256_file(path) for path in output_paths}
    manifest = {
        "schema_version": "binary_mcts_label_geometry_v1",
        "source_plan": "plans/mcts_bce_analysis.md",
        "scope": {"records": 8000, "train": 7000, "validation": 1000, "datasets": {"gqa": 4000, "textvqa": 2000, "chartqa": 2000}},
        "route_semantics": {"bits": 28, "one": "VISUAL_ON", "zero": "TEXT_ONLY", "full": mask_key(ALL_ON)},
        "validity": "stored route score >= stored benchmark-specific correctness threshold; verified against result_correct",
        "bce_weighting": "per-sample normalized polar_full_downweight_0.3: ALL-ON raw weight 0.3 iff a cheaper selected valid route coexists; all other routes raw weight 1.0",
        "decode_rule": "m_l = 1 when q_l >= 0.5; exact ties resolve ON to match binary_policy.decode.decode_threshold",
        "raw_route_policy": "deduplicate exact masks for geometry/oracle diagnostics; occurrence duplicates reported separately",
        "clustering": {"method": "connected components under Hamming <= radius", "radii": list(RADII), "effective_modes": "inverse Simpson over component sizes"},
        "pareto": "route b dominates a when stored score(b) >= score(a) and ON(b) < ON(a)",
        "diversity_counterfactual": {"start": "lowest ON then lexical", "selection": "greedy maximum minimum Hamming; ties lower ON then lexical", "k": list(SELECTION_K)},
        "source_hashes": source_hashes,
        "output_hashes_before_manifest": output_hashes,
        "figures": [str(path.relative_to(PROJECT)) for path in figure_paths],
        "integrity": "PASS",
    }
    write_json(output_dir / "analysis_manifest.json", manifest)
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), "positive_records": sum(row["selected_route_count"] > 0 for row in sample_rows)}), flush=True)


if __name__ == "__main__":
    main()
