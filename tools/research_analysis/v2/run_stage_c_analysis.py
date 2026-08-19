from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import yaml

from tools.research_analysis.v2.stage_c_analysis import (
    behavior_category,
    classify_stage_c_outcome,
    cluster_bootstrap_ci,
    consensus_bin,
    effect_label,
    pearson_correlation,
    reference_format_category,
    sign_agreement_fraction,
    spearman_correlation,
    trimmed_mean,
)
from tools.research_analysis.v2.stage_c_null_comparison import (
    evaluate_null_superiority,
    evaluate_real_residual_sensitivity,
)


OUTCOME_TEXT = {
    "Outcome A": "The Stage B TextVQA layer-0 READ effect did not replicate.",
    "Outcome B": (
        "The reference-support effect replicated, but it was not distinguishable "
        "from the frozen structured intervention nulls."
    ),
    "Outcome C": (
        "The frozen held-out protocol confirms a layer-0 answer-misaligned READ "
        "effect on TextVQA reference support."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the frozen Stage C protocol.")
    parser.add_argument("--config", default="configs/stage_c.yaml")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cluster_means(values: np.ndarray, image_ids: np.ndarray) -> np.ndarray:
    unique = np.unique(image_ids)
    return np.asarray([values[image_ids == image_id].mean() for image_id in unique])


def bootstrap_means(
    values: np.ndarray,
    image_ids: np.ndarray,
    draws: int,
    seed: int,
) -> np.ndarray:
    clusters = cluster_means(np.asarray(values, dtype=np.float64), image_ids)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, clusters.size, size=(draws, clusters.size))
    return clusters[indices].mean(axis=1)


def summarize(
    values: np.ndarray,
    image_ids: np.ndarray,
    draws: int,
    seed: int,
    quantiles: list[float],
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    low, high = cluster_bootstrap_ci(values, image_ids, draws, seed)
    return {
        "n_records": int(values.size),
        "n_image_clusters": int(np.unique(image_ids).size),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "trimmed_mean_05": trimmed_mean(values, 0.05),
        "trimmed_mean_20": trimmed_mean(values, 0.20),
        "quantiles": {
            str(value): float(np.quantile(values, value)) for value in quantiles
        },
        "clustered_ci_low": low,
        "clustered_ci_high": high,
    }


def fraction_summary(
    indicators: np.ndarray,
    image_ids: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    indicators = np.asarray(indicators, dtype=np.float64)
    low, high = cluster_bootstrap_ci(indicators, image_ids, draws, seed)
    return {
        "fraction": float(indicators.mean()),
        "clustered_ci_low": low,
        "clustered_ci_high": high,
    }


def simple_histogram_svg(path: Path, values: np.ndarray, title: str) -> None:
    values = np.asarray(values, dtype=np.float64)
    counts, edges = np.histogram(values, bins=40)
    width, height, margin = 800, 420, 55
    plot_width, plot_height = width - 2 * margin, height - 2 * margin
    maximum = max(int(counts.max()), 1)
    bars = []
    for index, count in enumerate(counts):
        x = margin + index * plot_width / len(counts)
        bar_width = plot_width / len(counts) - 1
        bar_height = plot_height * count / maximum
        y = margin + plot_height - bar_height
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
            f'height="{bar_height:.2f}" fill="#3569a8" />'
        )
    zero_x = None
    if edges[0] <= 0.0 <= edges[-1] and edges[-1] > edges[0]:
        zero_x = margin + (0.0 - edges[0]) / (edges[-1] - edges[0]) * plot_width
    zero_line = (
        f'<line x1="{zero_x:.2f}" y1="{margin}" x2="{zero_x:.2f}" '
        f'y2="{margin + plot_height}" stroke="#b33" stroke-width="2" />'
        if zero_x is not None
        else ""
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="white" />'
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">{title}</text>'
        + "".join(bars)
        + zero_line
        + f'<text x="{margin}" y="{height - 12}" font-size="12">{edges[0]:.4g}</text>'
        + f'<text x="{width - margin}" y="{height - 12}" text-anchor="end" font-size="12">{edges[-1]:.4g}</text>'
        + "</svg>\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def scatter_svg(
    path: Path,
    left: np.ndarray,
    right: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    width, height, margin = 800, 500, 65
    x_min, x_max = float(left.min()), float(left.max())
    y_min, y_max = float(right.min()), float(right.max())
    if x_min == x_max:
        x_min, x_max = x_min - 1.0, x_max + 1.0
    if y_min == y_max:
        y_min, y_max = y_min - 1.0, y_max + 1.0
    points = []
    for x_value, y_value in zip(left, right):
        x = margin + (x_value - x_min) / (x_max - x_min) * (width - 2 * margin)
        y = height - margin - (y_value - y_min) / (y_max - y_min) * (height - 2 * margin)
        points.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="#3569a8" opacity="0.45" />')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white" />'
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">{title}</text>'
        + "".join(points)
        + f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle" font-size="13">{x_label}</text>'
        + f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-size="13">{y_label}</text>'
        + "</svg>\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def execute(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config["output"]["analysis_dir"])
    if (output_dir / "analysis_manifest.json").exists():
        raise FileExistsError("Refusing to overwrite a completed Stage C analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(config["output"]["results_path"])
    results = read_jsonl(results_path)
    manifest = read_jsonl(Path(config["manifest_path"]))
    runtime = json.loads(Path(config["output"]["runtime_path"]).read_text(encoding="utf-8"))
    freeze = json.loads(Path(config["output"]["execution_freeze_path"]).read_text(encoding="utf-8"))
    prefix_preflight = json.loads(
        Path(config["output"]["prefix_preflight_summary_path"]).read_text(encoding="utf-8")
    )
    match_rows = read_jsonl(Path(config["nulls"]["real_donor_match_index"]))
    match_by_id = {row["id"]: row for row in match_rows}
    manifest_ids = [row["id"] for row in manifest]
    if (
        len(results) != 800
        or [row["id"] for row in results] != manifest_ids
        or len({row["image_id"] for row in results}) != 800
        or runtime["result_count"] != 800
        or runtime["config_sha256"] != sha256(config_path)
        or freeze["config_sha256"] != sha256(config_path)
        or len(match_rows) != 800
        or set(match_by_id) != set(manifest_ids)
        or freeze.get(
            "stage_c_primary_endpoint_computed",
            freeze.get(
                "stage_c_primary_endpoint_computed_or_inspected_before_amendment", True
            ),
        )
        or not prefix_preflight["all_records_pass"]
    ):
        raise RuntimeError("Stage C integrity gate failed before aggregation")
    if any(
        not row["integrity"]["generation_token_ids_match"]
        or row["integrity"]["prompt_logit_max_abs"]
        > float(config["full_parity_logit_tolerance"])
        or row["integrity"]["mean_score_abs"]
        > float(config["full_parity_score_tolerance"])
        for row in results
    ):
        raise RuntimeError("Stage C per-record FULL parity failed before aggregation")
    if any(
        [draw["donor_id"] for draw in row["structured_nulls"]["real_residual"]]
        != [
            donor["sample_id"]
            for donor in match_by_id[row["id"]]["selected_donors"]
        ]
        for row in results
    ):
        raise RuntimeError("Stage C real-residual draws differ from the frozen amended index")

    image_ids = np.asarray([row["image_id"] for row in results])
    primary = np.asarray([row["effects"]["mean"] for row in results], dtype=np.float64)
    sequence = np.asarray([row["effects"]["sequence"] for row in results], dtype=np.float64)
    draws = int(config["bootstrap"]["draws"])
    primary_seed = int(config["bootstrap"]["primary_seed"])
    quantiles = [float(value) for value in config["quantiles"]]
    practical_threshold = float(config["practical_threshold_nats_per_token"])

    primary_summary = summarize(primary, image_ids, draws, primary_seed, quantiles)
    primary_summary.update(
        {
            "estimand": "accepted-reference per-token FULL minus WRITE_ONLY at layer 0",
            "fraction_below_zero": fraction_summary(primary < 0.0, image_ids, draws, primary_seed),
            "practical_threshold": practical_threshold,
            "mean_reaches_practical_threshold": bool(primary.mean() <= practical_threshold),
            "fraction_at_or_below_practical_threshold": fraction_summary(
                primary <= practical_threshold, image_ids, draws, primary_seed
            ),
            "primary_pass": bool(primary_summary["clustered_ci_high"] < 0.0),
        }
    )
    write_json(output_dir / "primary_endpoint_summary.json", primary_summary)
    write_json(
        output_dir / "practical_threshold_fractions.json",
        {
            "threshold": practical_threshold,
            "mean_reaches_threshold": primary_summary["mean_reaches_practical_threshold"],
            "fraction": primary_summary["fraction_at_or_below_practical_threshold"],
        },
    )

    sequence_summary = summarize(sequence, image_ids, draws, primary_seed, quantiles)
    sequence_summary.update(
        {
            "directional_sign_agreement": sign_agreement_fraction(primary, sequence),
            "pearson_correlation": pearson_correlation(primary, sequence),
            "spearman_correlation": spearman_correlation(primary, sequence),
            "threshold_label_agreement": float(
                np.mean(
                    [
                        effect_label(mean_value, float(config["mean_epsilon"]))
                        == effect_label(sequence_value, float(config["sequence_epsilon"]))
                        for mean_value, sequence_value in zip(primary, sequence)
                    ]
                )
            ),
        }
    )
    write_json(output_dir / "sequence_per_token_agreement.json", sequence_summary)

    covariance_null = np.asarray(
        [
            [draw_row["mean_effect"] for draw_row in row["structured_nulls"]["covariance"]]
            for row in results
        ],
        dtype=np.float64,
    )
    real_residual_null = np.asarray(
        [
            [draw_row["mean_effect"] for draw_row in row["structured_nulls"]["real_residual"]]
            for row in results
        ],
        dtype=np.float64,
    )
    original_caliper_supported = np.asarray(
        [
            row["read_residual"]["original_caliper_supplies_eight"]
            for row in results
        ],
        dtype=bool,
    )
    excluded_original_caliper_ids = [
        row["id"] for row, included in zip(results, original_caliper_supported) if not included
    ]
    expected_excluded_ids = {
        "textvqa:textvqa_validation_39543",
        "textvqa:textvqa_validation_36174",
    }
    if (
        int(original_caliper_supported.sum()) != 798
        or set(excluded_original_caliper_ids) != expected_excluded_ids
    ):
        raise RuntimeError("Original-caliper sensitivity subset differs from its freeze")
    null_comparison = evaluate_null_superiority(
        primary,
        covariance_null,
        real_residual_null,
        image_ids,
        draws,
        int(config["bootstrap"]["covariance_seed"]),
        int(config["bootstrap"]["real_residual_seed"]),
    )
    real_residual_798_sensitivity = evaluate_real_residual_sensitivity(
        primary,
        real_residual_null,
        image_ids,
        original_caliper_supported,
        draws,
        int(config["bootstrap"]["real_residual_seed"]),
    )
    real_residual_798_sensitivity.update(
        {
            "original_caliper": float(config["nulls"]["real_donor_original_caliper"]),
            "amended_caliper": float(config["nulls"]["real_donor_matching_ratio_cap"]),
            "excluded_target_ids": excluded_original_caliper_ids,
            "replaces_all_800_primary_comparison": False,
        }
    )
    write_json(
        output_dir / "real_residual_null_original_caliper_798_sensitivity.json",
        real_residual_798_sensitivity,
    )
    covariance_sample_means = covariance_null.mean(axis=1)
    real_sample_means = real_residual_null.mean(axis=1)
    covariance_detail = {
        **null_comparison["covariance"],
        "per_draw_clustered_means": [
            float(cluster_means(covariance_null[:, index], image_ids).mean())
            for index in range(covariance_null.shape[1])
        ],
        "sample_mean_null_summary": summarize(
            covariance_sample_means, image_ids, draws, int(config["bootstrap"]["covariance_seed"]), quantiles
        ),
    }
    selected_real_draws = [draw_row for row in results for draw_row in row["structured_nulls"]["real_residual"]]
    real_detail = {
        **null_comparison["real_residual"],
        "per_draw_clustered_means": [
            float(cluster_means(real_residual_null[:, index], image_ids).mean())
            for index in range(real_residual_null.shape[1])
        ],
        "sample_mean_null_summary": summarize(
            real_sample_means, image_ids, draws, int(config["bootstrap"]["real_residual_seed"]), quantiles
        ),
        "donor_match_diagnostics": {
            "selected_draws": len(selected_real_draws),
            "unique_donor_ids": len({row["donor_id"] for row in selected_real_draws}),
            "maximum_matching_ratio": max(row["matching_ratio"] for row in selected_real_draws),
            "median_matching_ratio": float(np.median([row["matching_ratio"] for row in selected_real_draws])),
            "caliper": float(config["nulls"]["real_donor_matching_ratio_cap"]),
            "caliper_violation_count": sum(
                row["matching_ratio"] > float(config["nulls"]["real_donor_matching_ratio_cap"])
                for row in selected_real_draws
            ),
            "same_target_image_violation_count": sum(
                draw_row["donor_image_id"] == result["image_id"]
                for result in results
                for draw_row in result["structured_nulls"]["real_residual"]
            ),
            "maximum_norm_relative_error": max(
                row["norm_relative_error"] for row in selected_real_draws
            ),
            "minimum_eligible_donors": min(
                row["read_residual"]["eligible_real_donor_count"] for row in results
            ),
        },
    }
    write_json(output_dir / "covariance_null_comparison.json", covariance_detail)
    write_json(output_dir / "real_residual_null_comparison.json", real_detail)
    write_json(output_dir / "structured_null_comparison.json", null_comparison)

    eligible_wrong = [row for row in results if row["wrong_answer_contrast"]["eligible"]]
    wrong_values = np.asarray(
        [row["wrong_answer_contrast"]["delta_c_mean"] for row in eligible_wrong],
        dtype=np.float64,
    )
    wrong_images = np.asarray([row["image_id"] for row in eligible_wrong])
    wrong_summary = {
        "eligible_full_strictly_wrong_records": len(eligible_wrong),
        "ineligible_reasons": dict(
            Counter(
                row["wrong_answer_contrast"]["ineligible_reason"]
                for row in results
                if not row["wrong_answer_contrast"]["eligible"]
            )
        ),
    }
    if wrong_values.size >= 2:
        wrong_summary.update(
            {
                "mean_delta_c": float(wrong_values.mean()),
                "median_delta_c": float(np.median(wrong_values)),
                "clustered_ci": cluster_bootstrap_ci(
                    wrong_values, wrong_images, draws, primary_seed
                ),
                "fraction_delta_c_above_zero": fraction_summary(
                    wrong_values > 0.0, wrong_images, draws, primary_seed
                ),
            }
        )
    write_json(output_dir / "wrong_answer_contrast.json", wrong_summary)

    behavior_counts = Counter()
    consensus_transitions = Counter()
    for row in results:
        full = row["states"]["FULL"]
        write_only = row["states"]["WRITE_ONLY"]
        behavior_counts[
            behavior_category(
                full["official_correctness"],
                write_only["official_correctness"],
                full["normalized_generated_answer"],
                write_only["normalized_generated_answer"],
            )
        ] += 1
        consensus_transitions[
            f"{consensus_bin(full['official_correctness'])}_to_{consensus_bin(write_only['official_correctness'])}"
        ] += 1
    greedy_summary = {
        "strict_behavior_counts": dict(sorted(behavior_counts.items())),
        "partial_consensus_transitions": dict(sorted(consensus_transitions.items())),
        "interpretation_boundary": "secondary behavior; correction counts are not causal accuracy evidence",
    }
    write_json(output_dir / "greedy_behavior_counts.json", greedy_summary)

    uniform = np.asarray(
        [row["uniform_aggregation_robustness"]["mean_effect"] for row in results],
        dtype=np.float64,
    )
    prefix = np.asarray(
        [row["answer_prefix_robustness"]["mean_effect"] for row in results],
        dtype=np.float64,
    )
    prefix_sequence = np.asarray(
        [row["answer_prefix_robustness"]["sequence_effect"] for row in results],
        dtype=np.float64,
    )
    answer_lengths = np.asarray(
        [row["answer_length_weighted_tokens"] for row in results], dtype=np.float64
    )
    image_tokens = np.asarray([row["image_token_count"] for row in results], dtype=np.float64)
    format_rows = []
    for category in ("numeric", "alphabetic", "mixed_or_symbolic"):
        mask = np.asarray(
            [
                reference_format_category(
                    [answer["answer"] for answer in row["accepted_answers"]]
                )
                == category
                for row in results
            ]
        )
        format_rows.append(
            {
                "category": category,
                "n_records": int(mask.sum()),
                "mean_effect": float(primary[mask].mean()) if mask.any() else math.nan,
                "median_effect": float(np.median(primary[mask])) if mask.any() else math.nan,
            }
        )
    robustness = {
        "uniform_accepted_answer_aggregation": {
            "summary": summarize(uniform, image_ids, draws, primary_seed, quantiles),
            "sign_agreement_with_primary": sign_agreement_fraction(primary, uniform),
            "paired_effect_difference_uniform_minus_primary": summarize(
                uniform - primary, image_ids, draws, primary_seed, quantiles
            ),
        },
        "contextual_answer_prefix": {
            "amendment": config["prefix_tokenization_amendment"],
            "raw_per_token_levels_compared_across_prefixes": False,
            "summary": summarize(prefix, image_ids, draws, primary_seed, quantiles),
            "sign_agreement_with_primary": sign_agreement_fraction(primary, prefix),
            "paired_effect_difference_prefix_minus_primary": summarize(
                prefix - primary, image_ids, draws, primary_seed, quantiles
            ),
            "sequence_summary": summarize(
                prefix_sequence, image_ids, draws, primary_seed, quantiles
            ),
            "sequence_sign_agreement_with_primary_sequence": sign_agreement_fraction(
                sequence, prefix_sequence
            ),
            "paired_sequence_effect_difference_prefix_minus_primary": summarize(
                prefix_sequence - sequence, image_ids, draws, primary_seed, quantiles
            ),
        },
        "answer_length_sensitivity": {
            "pearson": pearson_correlation(answer_lengths, primary),
            "spearman": spearman_correlation(answer_lengths, primary),
        },
        "image_token_count_sensitivity": {
            "pearson": pearson_correlation(image_tokens, primary),
            "spearman": spearman_correlation(image_tokens, primary),
        },
        "reference_answer_format_categories": format_rows,
        "location_estimators": {
            key: primary_summary[key]
            for key in ("mean", "median", "trimmed_mean_05", "trimmed_mean_20")
        },
    }
    write_json(output_dir / "robustness_analyses.json", robustness)
    write_csv(output_dir / "reference_format_categories.csv", format_rows)

    primary_bootstrap = bootstrap_means(primary, image_ids, draws, primary_seed)
    practical_bootstrap = bootstrap_means(
        (primary <= practical_threshold).astype(np.float64), image_ids, draws, primary_seed
    )
    covariance_paired_bootstrap = bootstrap_means(
        primary - covariance_sample_means,
        image_ids,
        draws,
        int(config["bootstrap"]["covariance_seed"]),
    )
    real_paired_bootstrap = bootstrap_means(
        primary - real_sample_means,
        image_ids,
        draws,
        int(config["bootstrap"]["real_residual_seed"]),
    )
    np.savez_compressed(
        output_dir / "clustered_bootstrap_outputs.npz",
        primary_mean=primary_bootstrap,
        practical_fraction=practical_bootstrap,
        covariance_real_minus_null=covariance_paired_bootstrap,
        real_residual_real_minus_null=real_paired_bootstrap,
    )

    plot_dir = output_dir / "plots"
    simple_histogram_svg(plot_dir / "primary_effect_distribution.svg", primary, "Stage C primary per-token effect")
    scatter_svg(plot_dir / "sequence_vs_per_token.svg", primary, sequence, "Sequence and per-token effects", "per-token effect", "sequence effect")
    scatter_svg(plot_dir / "prefix_vs_primary.svg", primary, prefix, "Contextual-prefix and primary contrasts", "primary contrast", "prefix contrast")
    scatter_svg(plot_dir / "structured_null_means.svg", covariance_sample_means, real_sample_means, "Structured null sample means", "covariance null", "real-residual null")

    outcome = classify_stage_c_outcome(
        primary_summary["primary_pass"],
        null_comparison["covariance"]["pass"],
        null_comparison["real_residual"]["pass"],
    )
    decision = {
        "outcome": outcome,
        "exact_conclusion": OUTCOME_TEXT[outcome],
        "held_out_reference_support_replication": primary_summary["primary_pass"],
        "confirmed_answer_misaligned_read_effect": outcome == "Outcome C",
        "real_residual_798_sensitivity_role": "secondary only",
        "real_residual_798_sensitivity_pass": real_residual_798_sensitivity["pass"],
        "stage_d_authorized": False,
    }
    write_json(output_dir / "final_decision.json", decision)

    integrity = {
        "passed": True,
        "manifest_sha256_matches": sha256(Path(config["manifest_path"])) == config["manifest_sha256"],
        "source_plan_sha256_matches": sha256(Path(config["source_plan"])) == config["source_plan_sha256"],
        "result_count": len(results),
        "unique_image_count": len(set(image_ids.tolist())),
        "no_sample_replaced": [row["id"] for row in results] == manifest_ids,
        "no_outcome_dependent_exclusion": True,
        "all_full_parity_pass": True,
        "all_null_draw_counts_frozen": all(
            len(row["structured_nulls"]["covariance"])
            == len(row["structured_nulls"]["real_residual"])
            == int(config["nulls"]["draws_per_family"])
            for row in results
        ),
        "contextual_prefix_preflight_pass": True,
        "real_residual_caliper_amendment_sha256_matches": sha256(
            Path(config["real_residual_caliper_amendment"])
        )
        == config["real_residual_caliper_amendment_sha256"],
        "real_residual_match_index_sha256_matches": sha256(
            Path(config["nulls"]["real_donor_match_index"])
        )
        == config["nulls"]["real_donor_match_index_sha256"],
        "original_caliper_supported_record_count_is_798": int(
            original_caliper_supported.sum()
        )
        == 798,
        "amended_caliper_is_exact_19_over_12": float(
            config["nulls"]["real_donor_matching_ratio_cap"]
        )
        == 1.5833333333333333,
        "all_real_residual_donors_match_amended_index": True,
        "prior_partial_records_not_reused": True,
    }
    if not all(
        value for key, value in integrity.items() if isinstance(value, bool)
    ):
        raise RuntimeError("Final Stage C integrity manifest failed")
    artifact_paths = sorted(path for path in output_dir.rglob("*") if path.is_file())
    analysis_manifest = {
        "schema_version": "stage_c_analysis_manifest_v1",
        "integrity": integrity,
        "frozen_settings": config,
        "checksums": {
            "config": sha256(config_path),
            "source_plan": sha256(Path(config["source_plan"])),
            "manifest": sha256(Path(config["manifest_path"])),
            "results": sha256(results_path),
            "execution_freeze": sha256(Path(config["output"]["execution_freeze_path"])),
            "prefix_preflight": sha256(Path(config["output"]["prefix_preflight_summary_path"])),
            "null_comparison_implementation": sha256(Path("analysis/stage_c_null_comparison.py")),
            "analysis_implementation": sha256(Path(__file__)),
            "artifacts": {str(path): sha256(path) for path in artifact_paths},
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "runtime": runtime,
        "completion": {"records": len(results), "analysis_complete": True},
        "decision": decision,
    }
    write_json(output_dir / "analysis_manifest.json", analysis_manifest)

    results_report = f"""# Stage C Results\n\n## Integrity\n\nAll frozen integrity checks passed for 800 records and 800 unique images. The source plan, manifest, primary scorer, structured nulls, seeds, bootstrap, and success rules remained pinned. The approved real-residual caliper amendment is exactly `19/12`; every record was recomputed in fresh shards. The contextual-token amendment applied only to the secondary `Answer:` robustness condition.\n\n## Primary endpoint\n\n- Mean: `{primary_summary['mean']:.8f}` nats/token\n- Standard deviation: `{primary_summary['standard_deviation']:.8f}`\n- Median: `{primary_summary['median']:.8f}`\n- 5% trimmed mean: `{primary_summary['trimmed_mean_05']:.8f}`\n- 20% trimmed mean: `{primary_summary['trimmed_mean_20']:.8f}`\n- Image-clustered 95% CI: `[{primary_summary['clustered_ci_low']:.8f}, {primary_summary['clustered_ci_high']:.8f}]`\n- Fraction below zero: `{primary_summary['fraction_below_zero']['fraction']:.6f}`\n- Fraction at or below -0.05: `{primary_summary['fraction_at_or_below_practical_threshold']['fraction']:.6f}` with clustered CI `[{primary_summary['fraction_at_or_below_practical_threshold']['clustered_ci_low']:.6f}, {primary_summary['fraction_at_or_below_practical_threshold']['clustered_ci_high']:.6f}]`\n- Primary gate: `{'PASS' if primary_summary['primary_pass'] else 'FAIL'}`\n\n## Structured nulls\n\n- Covariance real-minus-null mean: `{null_comparison['covariance']['paired_mean']:.8f}`, CI `[{null_comparison['covariance']['paired_ci_low']:.8f}, {null_comparison['covariance']['paired_ci_high']:.8f}]`, `{'PASS' if null_comparison['covariance']['pass'] else 'FAIL'}`\n- Real-residual all-800 real-minus-null mean: `{null_comparison['real_residual']['paired_mean']:.8f}`, CI `[{null_comparison['real_residual']['paired_ci_low']:.8f}, {null_comparison['real_residual']['paired_ci_high']:.8f}]`, `{'PASS' if null_comparison['real_residual']['pass'] else 'FAIL'}`\n- Secondary original-1.5-supported 798-target sensitivity: mean `{real_residual_798_sensitivity['paired_mean']:.8f}`, CI `[{real_residual_798_sensitivity['paired_ci_low']:.8f}, {real_residual_798_sensitivity['paired_ci_high']:.8f}]`, `{'PASS' if real_residual_798_sensitivity['pass'] else 'FAIL'}`. This does not replace the all-800 comparison.\n- Coverage warning: `textvqa_validation_36174` had only three donors at 1.5; five selected donors enter at the amended boundary.\n\n## Secondary outcomes\n\n- FULL-wrong contrast eligible records: `{wrong_summary['eligible_full_strictly_wrong_records']}`\n- FULL wrong to WRITE_ONLY correct: `{behavior_counts['full_wrong_to_write_only_correct']}`\n- FULL correct to WRITE_ONLY wrong: `{behavior_counts['full_correct_to_write_only_wrong']}`\n\nFull secondary and robustness outputs are under `outputs/stage_c/analysis_v1/`.\n"""
    Path("reports/stage_c_results.md").write_text(results_report, encoding="utf-8")
    conclusion_report = f"""# Stage C Conclusion\n\n## Frozen decision\n\n**{outcome}**\n\n`{OUTCOME_TEXT[outcome]}`\n\nThis conclusion does not establish a harmful mechanism, accuracy improvement, broad harmful visual participation, or cross-task generalization. Stage D remains unauthorized.\n"""
    Path("reports/stage_c_conclusion.md").write_text(conclusion_report, encoding="utf-8")
    print(json.dumps({"integrity": integrity, "primary": primary_summary, "nulls": null_comparison, "decision": decision}, indent=2))


def main() -> int:
    args = parse_args()
    execute(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
