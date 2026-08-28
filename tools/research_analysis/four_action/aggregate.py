from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np


EFFECT_NAMES = ("read_w1", "read_w0", "write_r1", "write_r0", "interaction")
ACTIONS = ("IGNORE", "READ_ONLY", "WRITE_ONLY", "FULL")


def classify_rescue(ignore: bool, read_only: bool, write_only: bool) -> str:
    if read_only and not write_only:
        return "write_removal_only"
    if write_only and not read_only:
        return "read_removal_only"
    if read_only and write_only:
        return "either_removal_sufficient"
    if ignore:
        return "joint_removal_only"
    return "no_local_rescue"


def hamming_stratum(distance: int | None) -> str:
    if distance is None:
        return "not_available"
    if distance == 1:
        return "1"
    if distance == 2:
        return "2"
    if distance <= 4:
        return "3-4"
    if distance <= 8:
        return "5-8"
    return ">8"


def route_metadata(binary_routes: dict[str, Any] | None, layer_count: int) -> dict[str, Any]:
    if not binary_routes:
        return {
            "nearest_distance": None,
            "nearest_route_id": None,
            "nearest_mask": None,
            "correcting_route_count": 0,
            "off_frequency": None,
            "minimum_visual_on_count": None,
            "minimum_on_route_count": 0,
            "minimum_on_off_frequency": None,
        }
    nearest = list(binary_routes.get("nearest_correcting_routes") or [])
    correcting = list(binary_routes.get("correcting_routes") or [])
    nearest_row = sorted(nearest, key=lambda row: str(row.get("route_id", "")))[0] if nearest else None
    masks = [list(map(int, row["mask"])) for row in correcting]
    if any(len(mask) != layer_count for mask in masks):
        raise ValueError("correcting route mask has unexpected layer count")
    off_frequency = None
    if masks:
        off_frequency = [
            sum(mask[layer] == 0 for mask in masks) / len(masks)
            for layer in range(layer_count)
        ]
    minimum_visual_on_count = binary_routes.get("minimum_correcting_visual_on_count")
    minimum_on_masks = [
        mask
        for row, mask in zip(correcting, masks)
        if int(row.get("visual_on_count", sum(mask))) == minimum_visual_on_count
    ]
    minimum_on_off_frequency = None
    if minimum_on_masks:
        minimum_on_off_frequency = [
            sum(mask[layer] == 0 for mask in minimum_on_masks) / len(minimum_on_masks)
            for layer in range(layer_count)
        ]
    nearest_mask = None if nearest_row is None else list(map(int, nearest_row["mask"]))
    if nearest_mask is not None and len(nearest_mask) != layer_count:
        raise ValueError("nearest correcting route mask has unexpected layer count")
    return {
        "nearest_distance": binary_routes.get("nearest_correcting_route_distance"),
        "nearest_route_id": None if nearest_row is None else nearest_row.get("route_id"),
        "nearest_mask": nearest_mask,
        "correcting_route_count": len(correcting),
        "off_frequency": off_frequency,
        "minimum_visual_on_count": minimum_visual_on_count,
        "minimum_on_route_count": len(minimum_on_masks),
        "minimum_on_off_frequency": minimum_on_off_frequency,
    }


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    x = _average_ranks(left)
    y = _average_ranks(right)
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def flatten_samples(samples: Iterable[dict[str, Any]], layer_count: int = 28) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for sample in samples:
        metadata = route_metadata(sample.get("binary_routes"), layer_count)
        nearest_mask = metadata["nearest_mask"]
        off_frequency = metadata["off_frequency"]
        minimum_on_off_frequency = metadata["minimum_on_off_frequency"]
        for layer_row in sample["layers"]:
            layer = int(layer_row["layer"])
            states = layer_row["states"]
            effects = {name: float(layer_row["effects"][name]) for name in EFFECT_NAMES}
            rescue = classify_rescue(
                bool(states["IGNORE"]["correct"]),
                bool(states["READ_ONLY"]["correct"]),
                bool(states["WRITE_ONLY"]["correct"]),
            )
            base = {
                "uid": sample["uid"],
                "dataset": sample["dataset"],
                "cohort": sample["cohort"],
                "sample_id": sample["sample_id"],
                "image_id": sample["image_id"],
                "image_group_id": sample["image_group_id"],
                "visual_token_count": int(sample["visual_token_count"]),
                "layer": layer,
                "rescue_category": rescue,
                "nearest_correcting_route_distance": metadata["nearest_distance"],
                "hamming_stratum": hamming_stratum(metadata["nearest_distance"]),
                "nearest_route_id": metadata["nearest_route_id"],
                "nearest_route_layer_off": None if nearest_mask is None else nearest_mask[layer] == 0,
                "correcting_route_count": metadata["correcting_route_count"],
                "correcting_route_off_frequency": None if off_frequency is None else off_frequency[layer],
                "minimum_correcting_visual_on_count": metadata["minimum_visual_on_count"],
                "minimum_on_route_count": metadata["minimum_on_route_count"],
                "minimum_on_route_off_frequency": (
                    None if minimum_on_off_frequency is None else minimum_on_off_frequency[layer]
                ),
                "M00": float(states["IGNORE"]["margin"]),
                "M10": float(states["READ_ONLY"]["margin"]),
                "M01": float(states["WRITE_ONLY"]["margin"]),
                "M11": float(states["FULL"]["margin"]),
                **effects,
            }
            for action in ACTIONS:
                state = states[action]
                flat.append(
                    {
                        **base,
                        "action": action,
                        "S_correct": float(state["S_correct"]),
                        "S_full_wrong": state["S_full_wrong"],
                        "margin": float(state["margin"]),
                        "generated_answer": state["generated_answer"],
                        "correctness_score": float(state["correctness_score"]),
                        "correct": bool(state["correct"]),
                    }
                )
    return flat


def primary_layer_rows(flat_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in flat_rows
        if row["cohort"] == "primary_a_plus" and row["action"] == "FULL"
    ]


def magnitude_thresholds(layer_rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    thresholds = {}
    for name in EFFECT_NAMES:
        magnitudes = np.abs(np.asarray([float(row[name]) for row in layer_rows], dtype=np.float64))
        thresholds[name] = {
            "q50_absolute": float(np.quantile(magnitudes, 0.50)),
            "q75_absolute": float(np.quantile(magnitudes, 0.75)),
            "q90_absolute": float(np.quantile(magnitudes, 0.90)),
        }
    return thresholds


def _bootstrap_stat(
    rows: Sequence[dict[str, Any]],
    field: str,
    statistic: str,
    group_field: str | None,
    seed: int,
    replicates: int,
) -> dict[str, float]:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    observed = float(np.mean(values) if statistic == "mean" else np.median(values))
    if not rows:
        return {"value": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    if group_field is None:
        for index in range(replicates):
            sampled = values[rng.integers(0, len(values), len(values))]
            draws[index] = np.mean(sampled) if statistic == "mean" else np.median(sampled)
    else:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            grouped[str(row[group_field])].append(float(row[field]))
        keys = sorted(grouped)
        for index in range(replicates):
            sampled_keys = rng.choice(keys, size=len(keys), replace=True)
            sampled = np.asarray(
                [value for key in sampled_keys for value in grouped[str(key)]], dtype=np.float64
            )
            draws[index] = np.mean(sampled) if statistic == "mean" else np.median(sampled)
    return {
        "value": observed,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def layer_effect_table(
    layer_rows: Sequence[dict[str, Any]],
    thresholds: dict[str, dict[str, float]],
    seed: int = 20260823,
    replicates: int = 2000,
) -> list[dict[str, Any]]:
    table = []
    datasets = sorted({row["dataset"] for row in layer_rows})
    for dataset in [*datasets, "joint"]:
        for layer in sorted({int(row["layer"]) for row in layer_rows}):
            rows = [
                row for row in layer_rows
                if int(row["layer"]) == layer and (dataset == "joint" or row["dataset"] == dataset)
            ]
            for name in EFFECT_NAMES:
                values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
                sample_ci = _bootstrap_stat(rows, name, "mean", None, seed + layer, replicates)
                image_ci = _bootstrap_stat(
                    rows, name, "mean", "image_group_id", seed + 1000 + layer, replicates
                )
                table.append(
                    {
                        "dataset": dataset,
                        "layer": layer,
                        "effect": name,
                        "count": len(rows),
                        "mean": sample_ci["value"],
                        "median": float(np.median(values)),
                        "sample_ci_low": sample_ci["ci_low"],
                        "sample_ci_high": sample_ci["ci_high"],
                        "image_group_ci_low": image_ci["ci_low"],
                        "image_group_ci_high": image_ci["ci_high"],
                        "negative_fraction": float(np.mean(values < 0.0)),
                        "strong_negative_q75_fraction": float(
                            np.mean(values <= -thresholds[name]["q75_absolute"])
                        ),
                        "strong_negative_q90_fraction": float(
                            np.mean(values <= -thresholds[name]["q90_absolute"])
                        ),
                    }
                )
    return table


def rescue_tables(
    layer_rows: Sequence[dict[str, Any]],
    seed: int = 20260823,
    replicates: int = 2000,
) -> dict[str, Any]:
    categories = (
        "write_removal_only",
        "read_removal_only",
        "either_removal_sufficient",
        "joint_removal_only",
        "no_local_rescue",
    )
    per_layer = []
    per_dataset = []
    for dataset in [*sorted({row["dataset"] for row in layer_rows}), "joint"]:
        selected = [row for row in layer_rows if dataset == "joint" or row["dataset"] == dataset]
        for layer in sorted({int(row["layer"]) for row in selected}):
            rows = [row for row in selected if int(row["layer"]) == layer]
            for category in categories:
                count = sum(row["rescue_category"] == category for row in rows)
                indicator_rows = [
                    {**row, "indicator": float(row["rescue_category"] == category)}
                    for row in rows
                ]
                sample_ci = _bootstrap_stat(
                    indicator_rows, "indicator", "mean", None,
                    seed + 10_000 + layer, replicates,
                )
                image_ci = _bootstrap_stat(
                    indicator_rows, "indicator", "mean", "image_group_id",
                    seed + 20_000 + layer, replicates,
                )
                per_layer.append(
                    {"dataset": dataset, "layer": layer, "category": category,
                     "count": count, "fraction": count / len(rows),
                     "sample_ci_low": sample_ci["ci_low"],
                     "sample_ci_high": sample_ci["ci_high"],
                     "image_group_ci_low": image_ci["ci_low"],
                     "image_group_ci_high": image_ci["ci_high"]}
                )
        sample_layers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            sample_layers[row["uid"]].append(row)
        rescued = {
            uid: [row for row in rows if row["rescue_category"] != "no_local_rescue"]
            for uid, rows in sample_layers.items()
        }
        per_dataset.append(
            {
                "dataset": dataset,
                "sample_count": len(sample_layers),
                "samples_with_local_rescue": sum(bool(rows) for rows in rescued.values()),
                "rescue_layers_per_sample": {
                    "mean": float(np.mean([len(rows) for rows in rescued.values()])),
                    "median": float(np.median([len(rows) for rows in rescued.values()])),
                    "maximum": max((len(rows) for rows in rescued.values()), default=0),
                },
            }
        )
    return {"per_layer": per_layer, "per_dataset": per_dataset}


def route_overlap_table(layer_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    per_sample = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in layer_rows:
        grouped[row["uid"]].append(row)
    for uid, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["layer"]))
        metrics = {
            "read_w1": [float(row["read_w1"]) for row in rows],
            "write_r1": [float(row["write_r1"]) for row in rows],
            "ignore_gain_m00_minus_m11": [float(row["M00"] - row["M11"]) for row in rows],
            "strongest_local_harmfulness": [
                max(0.0, -min(float(row["read_w1"]), float(row["write_r1"])))
                for row in rows
            ],
        }
        harmfulness = metrics["strongest_local_harmfulness"]
        nearest_off = [row["nearest_route_layer_off"] for row in rows]
        off_frequency = [row["correcting_route_off_frequency"] for row in rows]
        if any(value is None for value in nearest_off) or any(value is None for value in off_frequency):
            continue
        top_layers = sorted(range(len(rows)), key=lambda index: (-harmfulness[index], index))
        off_count = sum(bool(value) for value in nearest_off)
        top_same_size = set(top_layers[:off_count])
        result = {
                "uid": uid,
                "dataset": rows[0]["dataset"],
                "nearest_distance": rows[0]["nearest_correcting_route_distance"],
                "nearest_off_count": off_count,
                "minimum_correcting_visual_on_count": rows[0]["minimum_correcting_visual_on_count"],
                "minimum_on_route_count": rows[0]["minimum_on_route_count"],
                "top_same_size_recall": (
                    sum(bool(nearest_off[index]) for index in top_same_size) / off_count
                    if off_count else math.nan
                ),
                "off_frequency_harmfulness_spearman": spearman(off_frequency, harmfulness),
            }
        for name, values in metrics.items():
            off_values = [value for value, is_off in zip(values, nearest_off) if is_off]
            on_values = [value for value, is_off in zip(values, nearest_off) if not is_off]
            result[f"nearest_off_mean_{name}"] = float(np.mean(off_values)) if off_values else math.nan
            result[f"nearest_on_mean_{name}"] = float(np.mean(on_values)) if on_values else math.nan
            result[f"nearest_off_minus_on_{name}"] = (
                float(np.mean(off_values) - np.mean(on_values))
                if off_values and on_values else math.nan
            )
            result[f"off_frequency_{name}_spearman"] = spearman(off_frequency, values)
        per_sample.append(result)
    difference_key = "nearest_off_minus_on_strongest_local_harmfulness"
    valid_differences = [row[difference_key] for row in per_sample if math.isfinite(row[difference_key])]
    valid_correlations = [row["off_frequency_harmfulness_spearman"] for row in per_sample if math.isfinite(row["off_frequency_harmfulness_spearman"])]
    return {
        "per_sample": per_sample,
        "aggregate": {
            "sample_count": len(per_sample),
            "mean_nearest_off_minus_on_harmfulness": float(np.mean(valid_differences)) if valid_differences else math.nan,
            "median_nearest_off_minus_on_harmfulness": float(np.median(valid_differences)) if valid_differences else math.nan,
            "mean_within_sample_off_frequency_spearman": float(np.mean(valid_correlations)) if valid_correlations else math.nan,
            "median_within_sample_off_frequency_spearman": float(np.median(valid_correlations)) if valid_correlations else math.nan,
            "metric_comparisons": {
                name: {
                    "mean_nearest_off_minus_on": float(np.mean(values)) if values else math.nan,
                    "median_nearest_off_minus_on": float(np.median(values)) if values else math.nan,
                    "mean_within_sample_off_frequency_spearman": (
                        float(np.mean(correlations)) if correlations else math.nan
                    ),
                    "median_within_sample_off_frequency_spearman": (
                        float(np.median(correlations)) if correlations else math.nan
                    ),
                }
                for name in metrics
                for values in [[
                    row[f"nearest_off_minus_on_{name}"]
                    for row in per_sample
                    if math.isfinite(row[f"nearest_off_minus_on_{name}"])
                ]]
                for correlations in [[
                    row[f"off_frequency_{name}_spearman"]
                    for row in per_sample
                    if math.isfinite(row[f"off_frequency_{name}_spearman"])
                ]]
            },
        },
    }
