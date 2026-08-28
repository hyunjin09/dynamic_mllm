#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from tools.research_analysis.four_action.aggregate import (
    EFFECT_NAMES,
    flatten_samples,
    layer_effect_table,
    magnitude_thresholds,
    primary_layer_rows,
    rescue_tables,
    route_overlap_table,
    spearman,
)


MODES = ("primary", "control_no_correction", "control_vision_required")
VALIDATION_MODES = ("preflight", "smoke", "pilot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze completed four-action causal stages.")
    parser.add_argument("--config", type=Path, default=Path("configs/four_action_answer_alignment.yaml"))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--input-tag", default="unified_v1")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require_completed_stage(root: Path, mode: str, input_tag: str) -> list[dict[str, Any]]:
    mode_directory = mode if not input_tag else f"{mode}__{input_tag}"
    summary_path = root / mode_directory / "stage_summary.json"
    result_path = root / mode_directory / "merged_results.jsonl"
    if not summary_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"{mode} must be merged and gated before aggregate analysis")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("passed"):
        raise RuntimeError(f"{mode} stage did not pass its gates")
    return read_jsonl(result_path)


def write_once(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def write_json(path: Path, payload: Any) -> None:
    write_once(path, (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def finite_or_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def write_frame(root: Path, name: str, frame: pd.DataFrame) -> None:
    parquet_path = root / f"{name}.parquet"
    jsonl_path = root / f"{name}.jsonl"
    if parquet_path.exists() or jsonl_path.exists():
        raise FileExistsError(f"refusing to overwrite existing {name} table")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    parquet_digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    parquet_path.with_suffix(".parquet.sha256").write_text(
        f"{parquet_digest}  {parquet_path.name}\n", encoding="utf-8"
    )
    jsonl_data = frame.to_json(orient="records", lines=True, force_ascii=False).encode()
    write_once(jsonl_path, jsonl_data)


def hamming_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for dataset in [*sorted({row["dataset"] for row in rows}), "joint"]:
        selected = [row for row in rows if dataset == "joint" or row["dataset"] == dataset]
        for stratum in ("1", "2", "3-4", "5-8", ">8"):
            group = [row for row in selected if row["hamming_stratum"] == stratum]
            if not group:
                continue
            sample_count = len({row["uid"] for row in group})
            table.append(
                {
                    "dataset": dataset,
                    "hamming_stratum": stratum,
                    "sample_count": sample_count,
                    "layer_cell_count": len(group),
                    "mean_read_w1": float(np.mean([row["read_w1"] for row in group])),
                    "mean_write_r1": float(np.mean([row["write_r1"] for row in group])),
                    "mean_abs_interaction": float(np.mean(np.abs([row["interaction"] for row in group]))),
                    "mean_negative_component_count_per_sample": float(
                        sum(min(row["read_w1"], row["write_r1"]) < 0.0 for row in group)
                        / sample_count
                    ),
                    "samples_with_local_rescue": len(
                        {row["uid"] for row in group if row["rescue_category"] != "no_local_rescue"}
                    ),
                }
            )
    return table


def per_sample_effect_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["uid"], []).append(row)
    output = []
    for uid, group in sorted(grouped.items()):
        read = np.asarray([float(row["read_w1"]) for row in group], dtype=np.float64)
        write = np.asarray([float(row["write_r1"]) for row in group], dtype=np.float64)
        interaction = np.asarray([float(row["interaction"]) for row in group], dtype=np.float64)
        strongest = np.minimum(read, write)
        rescue_count = sum(row["rescue_category"] != "no_local_rescue" for row in group)
        output.append(
            {
                "uid": uid,
                "dataset": group[0]["dataset"],
                "image_group_id": group[0]["image_group_id"],
                "nearest_correcting_route_distance": group[0]["nearest_correcting_route_distance"],
                "hamming_stratum": group[0]["hamming_stratum"],
                "negative_read_layer_count": int(np.sum(read < 0.0)),
                "negative_write_layer_count": int(np.sum(write < 0.0)),
                "negative_either_layer_count": int(np.sum(strongest < 0.0)),
                "rescue_layer_count": int(rescue_count),
                "strongest_negative_read_magnitude": float(max(0.0, -read.min())),
                "strongest_negative_write_magnitude": float(max(0.0, -write.min())),
                "strongest_negative_component_magnitude": float(max(0.0, -strongest.min())),
                "mean_negative_component_magnitude": float(
                    np.mean(np.maximum(0.0, -strongest))
                ),
                "mean_absolute_interaction": float(np.mean(np.abs(interaction))),
                "maximum_absolute_interaction": float(np.max(np.abs(interaction))),
            }
        )
    return output


def distance_association_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "negative_read_layer_count",
        "negative_write_layer_count",
        "negative_either_layer_count",
        "rescue_layer_count",
        "strongest_negative_read_magnitude",
        "strongest_negative_write_magnitude",
        "strongest_negative_component_magnitude",
        "mean_negative_component_magnitude",
        "mean_absolute_interaction",
        "maximum_absolute_interaction",
    )
    output = []
    for dataset in [*sorted({row["dataset"] for row in rows}), "joint"]:
        selected = [row for row in rows if dataset == "joint" or row["dataset"] == dataset]
        distances = [float(row["nearest_correcting_route_distance"]) for row in selected]
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            output.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "sample_count": len(selected),
                    "spearman_with_nearest_route_distance": spearman(distances, values),
                    "pearson_with_nearest_route_distance": (
                        float(np.corrcoef(distances, values)[0, 1])
                        if np.std(distances) > 0.0 and np.std(values) > 0.0
                        else math.nan
                    ),
                }
            )
    return output


def layer_extrema_summary(
    layer_effects: list[dict[str, Any]], rescue_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    effect_extrema = []
    for dataset in sorted({row["dataset"] for row in layer_effects}):
        for effect in EFFECT_NAMES:
            selected = [
                row for row in layer_effects
                if row["dataset"] == dataset and row["effect"] == effect
            ]
            minimum = min(selected, key=lambda row: (row["mean"], row["layer"]))
            maximum = max(selected, key=lambda row: (row["mean"], -row["layer"]))
            effect_extrema.append(
                {
                    "dataset": dataset,
                    "effect": effect,
                    "minimum_mean_layer": minimum["layer"],
                    "minimum_mean": minimum["mean"],
                    "maximum_mean_layer": maximum["layer"],
                    "maximum_mean": maximum["mean"],
                }
            )
    rescue_peaks = []
    for dataset in sorted({row["dataset"] for row in rescue_rows}):
        for category in sorted({row["category"] for row in rescue_rows}):
            selected = [
                row for row in rescue_rows
                if row["dataset"] == dataset and row["category"] == category
            ]
            maximum = max(selected, key=lambda row: (row["fraction"], -row["layer"]))
            rescue_peaks.append(
                {
                    "dataset": dataset,
                    "category": category,
                    "peak_layer": maximum["layer"],
                    "peak_fraction": maximum["fraction"],
                    "peak_count": maximum["count"],
                }
            )
    return {"effect_extrema": effect_extrema, "rescue_peaks": rescue_peaks}


def cohort_comparison_table(
    rows: list[dict[str, Any]], thresholds: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    table = []
    for cohort in sorted({row["cohort"] for row in rows}):
        for dataset in [*sorted({row["dataset"] for row in rows}), "joint"]:
            group = [
                row for row in rows
                if row["cohort"] == cohort and (dataset == "joint" or row["dataset"] == dataset)
            ]
            if not group:
                continue
            for effect in EFFECT_NAMES:
                values = np.asarray([row[effect] for row in group], dtype=np.float64)
                table.append(
                    {
                        "cohort": cohort,
                        "dataset": dataset,
                        "effect": effect,
                        "sample_count": len({row["uid"] for row in group}),
                        "layer_cell_count": len(group),
                        "mean": float(values.mean()),
                        "median": float(np.median(values)),
                        "negative_fraction": float(np.mean(values < 0.0)),
                        "strong_negative_q75_fraction": float(
                            np.mean(values <= -thresholds[effect]["q75_absolute"])
                        ),
                        "strong_negative_q90_fraction": float(
                            np.mean(values <= -thresholds[effect]["q90_absolute"])
                        ),
                    }
                )
    return table


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(array.max()),
    }


def native_drift_rows(
    samples: list[dict[str, Any]], analysis_set: str
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        diagnostic = sample["native_full_external"]["diagnostic"]
        row = {
            "uid": sample["uid"],
            "dataset": sample["dataset"],
            "cohort": sample["cohort"],
            "analysis_set": analysis_set,
            "generated_ids_match": diagnostic["generated_ids_match"],
            "generated_answer_match": diagnostic["generated_answer_match"],
            "evaluator_score_match": diagnostic["evaluator_score_match"],
            "correctness_match": diagnostic["correctness_match"],
        }
        for quantity, value in diagnostic["signed_drift"].items():
            row[f"{quantity}_signed_drift"] = float(value)
            row[f"{quantity}_absolute_drift"] = abs(float(value))
        rows.append(row)
    return rows


def drift_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    cohorts = sorted({row["cohort"] for row in rows})
    datasets = sorted({row["dataset"] for row in rows})
    for analysis_set in sorted({row["analysis_set"] for row in rows}):
        for cohort in [*cohorts, "all"]:
            for dataset in [*datasets, "joint"]:
                selected = [
                    row for row in rows
                    if row["analysis_set"] == analysis_set
                    and (cohort == "all" or row["cohort"] == cohort)
                    and (dataset == "joint" or row["dataset"] == dataset)
                ]
                if not selected:
                    continue
                for quantity in ("S_correct", "S_full_wrong", "margin"):
                    signed_key = f"{quantity}_signed_drift"
                    available = [row for row in selected if signed_key in row]
                    if not available:
                        continue
                    signed = [float(row[signed_key]) for row in available]
                    table.append(
                        {
                            "cohort": cohort,
                            "dataset": dataset,
                            "analysis_set": analysis_set,
                            "quantity": quantity,
                            "signed": distribution(signed),
                            "absolute": distribution([abs(value) for value in signed]),
                        }
                    )
    return table


def validation_semantic_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    full_fields = ("generated_ids_match", "generated_answer_match", "evaluator_score_match", "correctness_match")
    full_counts = {
        field: sum(sample["native_full_external"]["diagnostic"][field] for sample in samples)
        for field in full_fields
    }
    ignore_rows = [
        layer["old_binary_ignore_external"]
        for sample in samples
        for layer in sample["layers"]
        if "old_binary_ignore_external" in layer
    ]
    return {
        "validation_sample_count": len(samples),
        "unified_full_vs_native": {
            "comparisons": len(samples),
            **{f"{field}_count": count for field, count in full_counts.items()},
        },
        "unified_ignore_vs_old_binary_single_off": {
            "comparisons": len(ignore_rows),
            **{
                f"{field}_count": sum(row[field] for row in ignore_rows)
                for field in full_fields
            },
        },
    }


def answer_erosion_rows(
    samples: list[dict[str, Any]], primary_rows: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    local_by_uid: dict[str, list[dict[str, Any]]] = {}
    for row in primary_rows:
        local_by_uid.setdefault(row["uid"], []).append(row)
    rng = np.random.default_rng(seed)
    output = []
    for sample in samples:
        trajectory = sample["unified_full_answer_trajectory"]
        row = {
            "uid": sample["uid"],
            "dataset": sample["dataset"],
            "cohort": sample["cohort"],
            "image_group_id": sample["image_group_id"],
            "margin_by_layer": trajectory["margin_by_layer"],
            "maximum_intermediate_margin": trajectory["maximum_intermediate_margin"],
            "peak_layer": trajectory["peak_layer"],
            "final_margin": trajectory["final_margin"],
            "peak_to_final_erosion": trajectory["peak_to_final_erosion"],
            "largest_adjacent_change": trajectory["largest_adjacent_change"],
            "largest_drop_arrival_layer": trajectory["largest_drop_arrival_layer"],
        }
        local = sorted(local_by_uid.get(sample["uid"], []), key=lambda item: item["layer"])
        if local:
            culprit = min(
                local,
                key=lambda item: (
                    min(float(item["read_w1"]), float(item["write_r1"])),
                    int(item["layer"]),
                ),
            )
            read_value = float(culprit["read_w1"])
            write_value = float(culprit["write_r1"])
            culprit_operation = "READ" if read_value <= write_value else "WRITE"
            distance = abs(int(culprit["layer"]) - int(trajectory["largest_drop_arrival_layer"]))
            random_layers = rng.integers(0, 28, size=2000)
            random_distances = np.abs(random_layers - int(trajectory["largest_drop_arrival_layer"]))
            row.update(
                {
                    "strongest_harmful_operation": culprit_operation,
                    "strongest_harmful_layer": int(culprit["layer"]),
                    "strongest_harmful_effect": min(read_value, write_value),
                    "culprit_drop_layer_distance": distance,
                    "culprit_within_1_of_drop": distance <= 1,
                    "culprit_within_2_of_drop": distance <= 2,
                    "random_layer_mean_drop_distance": float(random_distances.mean()),
                    "random_layer_within_1_fraction": float(np.mean(random_distances <= 1)),
                    "random_layer_within_2_fraction": float(np.mean(random_distances <= 2)),
                }
            )
        output.append(row)
    return output


def answer_erosion_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for cohort in sorted({row["cohort"] for row in rows}):
        for dataset in [*sorted({row["dataset"] for row in rows}), "joint"]:
            selected = [
                row for row in rows
                if row["cohort"] == cohort and (dataset == "joint" or row["dataset"] == dataset)
            ]
            if not selected:
                continue
            entry = {
                "cohort": cohort,
                "dataset": dataset,
                "sample_count": len(selected),
                "mean_peak_to_final_erosion": float(
                    np.mean([row["peak_to_final_erosion"] for row in selected])
                ),
                "median_peak_to_final_erosion": float(
                    np.median([row["peak_to_final_erosion"] for row in selected])
                ),
                "mean_largest_adjacent_change": float(
                    np.mean([row["largest_adjacent_change"] for row in selected])
                ),
                "fraction_positive_intermediate_margin": float(
                    np.mean([max(row["margin_by_layer"][:-1]) > 0.0 for row in selected])
                ),
            }
            aligned = [row for row in selected if "culprit_drop_layer_distance" in row]
            if aligned:
                entry.update(
                    {
                        "mean_culprit_drop_layer_distance": float(
                            np.mean([row["culprit_drop_layer_distance"] for row in aligned])
                        ),
                        "culprit_within_1_fraction": float(
                            np.mean([row["culprit_within_1_of_drop"] for row in aligned])
                        ),
                        "culprit_within_2_fraction": float(
                            np.mean([row["culprit_within_2_of_drop"] for row in aligned])
                        ),
                        "random_layer_within_1_fraction": float(
                            np.mean([row["random_layer_within_1_fraction"] for row in aligned])
                        ),
                        "random_layer_within_2_fraction": float(
                            np.mean([row["random_layer_within_2_fraction"] for row in aligned])
                        ),
                    }
                )
            output.append(entry)
    return output


def numerical_consistency_markdown(
    drift: list[dict[str, Any]], semantic: dict[str, Any]
) -> str:
    lines = [
        "# Numerical Consistency Report",
        "",
        "## Definitions",
        "",
        "Native Qwen FULL is an external semantic/cohort diagnostic. Unified",
        "materialized-mask FULL is M11. Every reported READ/WRITE factorial",
        "effect is computed only within the unified executor. Native/unified",
        "drift is never an effect threshold.",
        "",
        "## Validation semantics",
        "",
        "```json",
        json.dumps(semantic, indent=2, sort_keys=True),
        "```",
        "",
        "## Native FULL versus unified FULL drift",
        "",
        "Signed drift is `unified - native`. Absolute drift is reported",
        "separately. Values are length-normalized teacher-forced log-probability",
        "or correct-vs-frozen-FULL-wrong margin units.",
        "",
        "| Analysis set | Cohort | Dataset | Quantity | Distribution | Mean | Median | Std | P90 | P95 | P99 | Min | Max |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in drift:
        for name in ("signed", "absolute"):
            values = row[name]
            lines.append(
                f"| {row['analysis_set']} | {row['cohort']} | {row['dataset']} | {row['quantity']} | {name} | "
                f"{values['mean']:.6f} | {values['median']:.6f} | {values['std']:.6f} | "
                f"{values['p90']:.6f} | {values['p95']:.6f} | {values['p99']:.6f} | "
                f"{values['minimum']:.6f} | {values['maximum']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Separation from causal effects",
            "",
            "M00, M10, M01, and M11 share the unified-FULL prefix and suffix.",
            "No value in this report is subtracted from, used to calibrate, or",
            "used to threshold a within-unified causal effect.",
            "",
        ]
    )
    return "\n".join(lines)


def save_figure(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_figures(
    root: Path,
    primary: pd.DataFrame,
    layer_effects: pd.DataFrame,
    rescue: pd.DataFrame,
    hamming: pd.DataFrame,
) -> None:
    figures = root / "figures"
    joint = layer_effects[layer_effects.dataset == "joint"]
    for family, effects, title in (
        ("read_effect_vs_layer", ("read_w1", "read_w0"), "Conditional READ effects"),
        ("write_effect_vs_layer", ("write_r1", "write_r0"), "Conditional WRITE effects"),
    ):
        plt.figure(figsize=(8, 4.5))
        for effect in effects:
            subset = joint[joint.effect == effect].sort_values("layer")
            plt.plot(subset.layer, subset["mean"], marker="o", markersize=2.5, label=effect)
            plt.fill_between(
                subset.layer,
                subset.sample_ci_low,
                subset.sample_ci_high,
                alpha=0.15,
            )
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Decoder layer")
        plt.ylabel("Correct-vs-FULL-wrong margin effect")
        plt.title(title)
        plt.legend()
        save_figure(figures / f"{family}.png")

    subset = joint[joint.effect == "interaction"].sort_values("layer")
    plt.figure(figsize=(8, 4.5))
    plt.plot(subset.layer, subset["mean"], marker="o", markersize=2.5)
    plt.fill_between(subset.layer, subset.sample_ci_low, subset.sample_ci_high, alpha=0.15)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Decoder layer")
    plt.ylabel("READ×WRITE interaction")
    plt.title("Factorial interaction by layer")
    save_figure(figures / "interaction_vs_layer.png")

    rescue_joint = rescue[(rescue.dataset == "joint") & (rescue.category != "no_local_rescue")]
    plt.figure(figsize=(8, 4.5))
    for category, group in rescue_joint.groupby("category"):
        group = group.sort_values("layer")
        plt.plot(group.layer, group.fraction, marker="o", markersize=2.5, label=category)
    plt.xlabel("Decoder layer")
    plt.ylabel("Fraction of A+ samples")
    plt.title("Local rescue prevalence")
    plt.legend(fontsize=8)
    save_figure(figures / "local_rescue_prevalence_vs_layer.png")

    plt.figure(figsize=(8, 4.5))
    values = [primary[name].to_numpy() for name in EFFECT_NAMES]
    plt.violinplot(values, showmeans=True, showextrema=False)
    plt.xticks(range(1, len(EFFECT_NAMES) + 1), EFFECT_NAMES, rotation=25)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Effect on answer margin")
    plt.title("A+ local-effect distributions")
    save_figure(figures / "effect_distributions.png")

    if not hamming.empty:
        order = [item for item in ("1", "2", "3-4", "5-8", ">8") if item in set(hamming.hamming_stratum)]
        joint_hamming = hamming[hamming.dataset == "joint"].set_index("hamming_stratum").loc[order]
        plt.figure(figsize=(8, 4.5))
        plt.plot(order, joint_hamming.mean_read_w1, marker="o", label="read_w1")
        plt.plot(order, joint_hamming.mean_write_r1, marker="o", label="write_r1")
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Nearest correcting-route Hamming distance")
        plt.ylabel("Mean conditional effect")
        plt.title("Local effects by binary-route distance")
        plt.legend()
        save_figure(figures / "hamming_distance_stratification.png")

    route_rows = primary.dropna(subset=["correcting_route_off_frequency"]).copy()
    if not route_rows.empty:
        route_rows["harmfulness"] = np.maximum(
            0.0, -np.minimum(route_rows.read_w1, route_rows.write_r1)
        )
        bins = pd.cut(route_rows.correcting_route_off_frequency, bins=np.linspace(0, 1, 11), include_lowest=True)
        binned = route_rows.groupby(bins, observed=False).agg(
            off_frequency=("correcting_route_off_frequency", "mean"),
            harmfulness=("harmfulness", "mean"),
        ).dropna()
        plt.figure(figsize=(7, 4.5))
        plt.plot(binned.off_frequency, binned.harmfulness, marker="o")
        plt.xlabel("Correcting-route OFF frequency")
        plt.ylabel("Mean strongest local harmfulness")
        plt.title("Binary route suppression vs local harmfulness")
        save_figure(figures / "binary_off_frequency_vs_local_harmfulness.png")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(config["output_root"])
    samples = [
        sample
        for mode in MODES
        for sample in require_completed_stage(root, mode, args.input_tag)
    ]
    validation_stage_samples = {
        mode: require_completed_stage(root, mode, args.input_tag)
        for mode in VALIDATION_MODES
    }
    validation_samples = [
        sample for rows in validation_stage_samples.values() for sample in rows
    ]
    flat = flatten_samples(samples, layer_count=len(config["layer_grid"]))
    primary_rows = primary_layer_rows(flat)
    all_layer_rows = [row for row in flat if row["action"] == "FULL"]
    thresholds = magnitude_thresholds(primary_rows)
    layer_effects = layer_effect_table(
        primary_rows,
        thresholds,
        seed=int(config["seed"]),
        replicates=args.bootstrap_replicates,
    )
    rescue = rescue_tables(
        primary_rows,
        seed=int(config["seed"]),
        replicates=args.bootstrap_replicates,
    )
    hamming = hamming_table(primary_rows)
    sample_effects = per_sample_effect_summary(primary_rows)
    distance_associations = distance_association_summary(sample_effects)
    overlap = route_overlap_table(primary_rows)
    comparisons = cohort_comparison_table(all_layer_rows, thresholds)
    drift_rows = native_drift_rows(samples, "production")
    for mode, rows in validation_stage_samples.items():
        drift_rows.extend(native_drift_rows(rows, f"validation_{mode}"))
    drift_statistics = drift_summary(drift_rows)
    semantic_summary = validation_semantic_summary(validation_samples)
    erosion = answer_erosion_rows(samples, primary_rows, int(config["seed"]))
    erosion_summary = answer_erosion_summary(erosion)
    extrema = layer_extrema_summary(layer_effects, rescue["per_layer"])

    tables = root / "aggregate"
    flat_frame = pd.DataFrame(flat)
    layer_frame = pd.DataFrame(layer_effects)
    rescue_layer_frame = pd.DataFrame(rescue["per_layer"])
    hamming_frame = pd.DataFrame(hamming)
    write_frame(root, "per_sample_layer_actions", flat_frame)
    write_frame(tables, "per_layer_effects", layer_frame)
    write_frame(tables, "rescue_taxonomy_per_layer", rescue_layer_frame)
    write_frame(tables, "hamming_strata", hamming_frame)
    write_frame(tables, "per_sample_effect_summary", pd.DataFrame(sample_effects))
    write_frame(tables, "hamming_distance_associations", pd.DataFrame(distance_associations))
    write_frame(tables, "cohort_comparisons", pd.DataFrame(comparisons))
    write_frame(tables, "route_overlap_per_sample", pd.DataFrame(overlap["per_sample"]))
    write_frame(tables, "native_unified_full_drift", pd.DataFrame(drift_rows))
    write_frame(tables, "answer_erosion", pd.DataFrame(erosion))
    write_json(tables / "effect_magnitude_thresholds.json", thresholds)
    write_json(tables / "rescue_taxonomy_summary.json", rescue["per_dataset"])
    write_json(tables / "route_overlap_summary.json", finite_or_none(overlap["aggregate"]))
    write_json(tables / "native_unified_full_drift_summary.json", drift_statistics)
    write_json(tables / "validation_semantic_summary.json", semantic_summary)
    write_json(tables / "answer_erosion_summary.json", erosion_summary)
    write_json(
        tables / "analysis_summary.json",
        finite_or_none(
            {
                "schema_version": "four_action_analysis_summary_v1",
                "sample_counts": {
                    "all_analyzed": len(samples),
                    "primary": len({row["uid"] for row in primary_rows}),
                    "by_cohort_dataset": {
                        f"{cohort}/{dataset}": sum(
                            sample["cohort"] == cohort and sample["dataset"] == dataset
                            for sample in samples
                        )
                        for cohort in sorted({sample["cohort"] for sample in samples})
                        for dataset in sorted({sample["dataset"] for sample in samples})
                    },
                },
                "within_unified_effect_thresholds_descriptive_only": thresholds,
                "layer_extrema": extrema,
                "rescue_summary": rescue["per_dataset"],
                "route_overlap_summary": overlap["aggregate"],
                "hamming_distance_associations": distance_associations,
                "cohort_comparisons": comparisons,
                "answer_erosion_summary": erosion_summary,
                "native_unified_drift_summary_diagnostic_only": drift_statistics,
                "validation_semantic_summary": semantic_summary,
            }
        ),
    )
    write_once(
        root / "numerical_consistency_report.md",
        numerical_consistency_markdown(drift_statistics, semantic_summary).encode(),
    )
    make_figures(
        root,
        pd.DataFrame(primary_rows),
        layer_frame,
        rescue_layer_frame,
        hamming_frame,
    )
    drift_frame = pd.DataFrame(drift_rows)
    margin_drift = drift_frame.loc[
        drift_frame.analysis_set == "production", "margin_signed_drift"
    ]
    plt.figure(figsize=(8, 4.5))
    plt.hist(margin_drift, bins=60)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Unified FULL minus native FULL margin")
    plt.ylabel("Samples")
    plt.title("Native–unified FULL margin drift (diagnostic only)")
    save_figure(root / "figures" / "native_unified_full_margin_drift.png")
    erosion_frame = pd.DataFrame(erosion)
    plt.figure(figsize=(8, 4.5))
    for cohort, group in erosion_frame.groupby("cohort"):
        curves = np.asarray(group.margin_by_layer.tolist(), dtype=np.float64)
        plt.plot(range(curves.shape[1]), curves.mean(axis=0), label=cohort)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Decoder layer")
    plt.ylabel("Mean correct-vs-FULL-wrong logit-lens margin")
    plt.title("Unified FULL answer-alignment trajectories")
    plt.legend(fontsize=8)
    save_figure(root / "figures" / "answer_erosion_curves.png")
    aligned = erosion_frame.dropna(subset=["strongest_harmful_layer"])
    if not aligned.empty:
        plt.figure(figsize=(6, 5.5))
        plt.scatter(
            aligned.strongest_harmful_layer,
            aligned.largest_drop_arrival_layer,
            alpha=0.25,
            s=10,
        )
        plt.plot([0, 27], [0, 27], color="black", linewidth=0.8)
        plt.xlabel("Strongest harmful local operation layer")
        plt.ylabel("Largest erosion arrival layer")
        plt.title("Culprit-layer versus collapse-layer alignment")
        save_figure(root / "figures" / "culprit_layer_vs_collapse_layer.png")
    print(
        json.dumps(
            {
                "sample_count": len(samples),
                "primary_sample_count": len({row["uid"] for row in primary_rows}),
                "flat_action_rows": len(flat),
                "bootstrap_replicates": args.bootstrap_replicates,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
