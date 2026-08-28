#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("analysis/4action_answer_alignment")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    value = float(value)
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def joint(rows: list[dict[str, Any]], **conditions: Any) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if all(row.get(key) == value for key, value in conditions.items())
    ]


def main() -> None:
    aggregate = read_json(ROOT / "aggregate" / "analysis_summary.json")
    trajectory = read_json(ROOT / "trajectory_rescue" / "summary.json")
    cohort = read_json(ROOT / "cohort" / "cohort_summary_v1.json")
    eligibility = read_json(ROOT / "cohort_eligibility__unified_v1" / "summary.json")
    semantic = aggregate["validation_semantic_summary"]
    drift = aggregate["native_unified_drift_summary_diagnostic_only"]
    margin_drift = joint(
        drift,
        analysis_set="production",
        cohort="all",
        dataset="joint",
        quantity="margin",
    )[0]
    extrema = joint(aggregate["layer_extrema"]["effect_extrema"], dataset="joint")
    rescues = {row["dataset"]: row for row in aggregate["rescue_summary"]}
    route = aggregate["route_overlap_summary"]
    distance = joint(aggregate["hamming_distance_associations"], dataset="joint")
    erosion = {
        (row["cohort"], row["dataset"]): row
        for row in aggregate["answer_erosion_summary"]
    }
    controls = aggregate["cohort_comparisons"]
    primary_count = aggregate["sample_counts"]["primary"]
    primary_by_dataset = aggregate["sample_counts"]["by_cohort_dataset"]
    candidate_primary_count = cohort["primary_rows"]
    excluded_primary_count = candidate_primary_count - primary_count

    lines = [
        "# Four-Action Answer-Unaligned Report",
        "",
        "## Scope and causal contract",
        "",
        f"The matched cache supplied {candidate_primary_count:,} candidate A+ samples. "
        f"After freezing current unified-FULL correctness, the primary sweep contains "
        f"{primary_count:,} eligible samples: "
        f"{primary_by_dataset.get('primary_a_plus/gqa', 0):,} GQA and "
        f"{primary_by_dataset.get('primary_a_plus/textvqa', 0):,} TextVQA; "
        f"{excluded_primary_count:,} candidates were excluded because current unified "
        "FULL no longer satisfied the frozen cohort's FULL-wrong condition. "
        "Every sample was evaluated at all 28 decoder layers with M00=IGNORE, "
        "M10=READ_ONLY, M01=WRITE_ONLY, and M11=unified FULL.",
        "",
        "All factorial effects below are internal to the unified executor. Native "
        "Qwen FULL is used only for the matched-cache candidate cohort, generation/correctness sanity "
        "checks, and implementation-drift measurement. No native/unified drift value "
        "is used as a causal-effect threshold.",
        "",
        f"The eligibility freeze evaluated {eligibility['candidate_count']:,} total "
        f"primary/control candidates and retained {eligibility['eligible_count']:,}; "
        "the per-cohort and per-dataset counts are preserved in the eligibility summary.",
        "",
        "## Semantic validation and implementation drift",
        "",
        "| Comparison | Comparisons | Token-ID matches | Answer matches | Evaluator-score matches | Correctness matches |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("Unified FULL vs native FULL", "unified_full_vs_native"),
        ("Unified IGNORE vs old binary single-OFF", "unified_ignore_vs_old_binary_single_off"),
    ):
        row = semantic[key]
        lines.append(
            f"| {label} | {row['comparisons']} | {row['generated_ids_match_count']} | "
            f"{row['generated_answer_match_count']} | {row['evaluator_score_match_count']} | "
            f"{row['correctness_match_count']} |"
        )
    lines.extend(
        [
            "",
            "Native-to-unified FULL margin drift is signed as unified minus native:",
            "",
            "| Distribution | Mean | Median | Std | P90 | P95 | P99 | Max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ("signed", "absolute"):
        row = margin_drift[label]
        lines.append(
            f"| {label} | {number(row['mean'])} | {number(row['median'])} | "
            f"{number(row['std'])} | {number(row['p90'])} | {number(row['p95'])} | "
            f"{number(row['p99'])} | {number(row['maximum'])} |"
        )

    lines.extend(
        [
            "",
            "## Within-unified layerwise factorial landscape",
            "",
            "These extrema summarize population means; the full distributions and "
            "sample- and image-group bootstrap intervals are in the aggregate tables.",
            "",
            "| Effect | Most negative mean layer | Mean | Most positive mean layer | Mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in extrema:
        lines.append(
            f"| {row['effect']} | {row['minimum_mean_layer']} | {number(row['minimum_mean'])} | "
            f"{row['maximum_mean_layer']} | {number(row['maximum_mean'])} |"
        )
    lines.extend(["", "## Discrete local rescues", ""])
    for dataset in ("gqa", "textvqa", "joint"):
        row = rescues[dataset]
        lines.append(
            f"- {dataset}: {row['samples_with_local_rescue']:,}/{row['sample_count']:,} "
            f"samples had at least one local rescue; median rescue layers per sample "
            f"was {number(row['rescue_layers_per_sample']['median'], 1)}."
        )

    metric_comparisons = route["metric_comparisons"]
    lines.extend(
        [
            "",
            "## Relationship to binary correcting routes",
            "",
            "Positive OFF-minus-ON values for ignore gain or harmfulness mean the "
            "nearest correcting routes preferentially turn off more locally harmful "
            "layers. Negative OFF-minus-ON READ/WRITE values mean the OFF layers have "
            "more negative conditional effects.",
            "",
            "| Local quantity | Mean nearest-route OFF minus ON | Mean within-sample OFF-frequency Spearman |",
            "|---|---:|---:|",
        ]
    )
    for metric, row in metric_comparisons.items():
        lines.append(
            f"| {metric} | {number(row['mean_nearest_off_minus_on'])} | "
            f"{number(row['mean_within_sample_off_frequency_spearman'])} |"
        )

    lines.extend(
        [
            "",
            "## Hamming-distance stratification",
            "",
            "| Sample-level quantity | Spearman with nearest correcting-route distance | Pearson |",
            "|---|---:|---:|",
        ]
    )
    for row in distance:
        lines.append(
            f"| {row['metric']} | {number(row['spearman_with_nearest_route_distance'])} | "
            f"{number(row['pearson_with_nearest_route_distance'])} |"
        )

    primary_erosion = erosion[("primary_a_plus", "joint")]
    lines.extend(
        [
            "",
            "## Answer erosion and local causal alignment",
            "",
            f"In the primary A+ cohort, {number(100 * primary_erosion['fraction_positive_intermediate_margin'], 1)}% "
            "of samples had a positive intermediate correct-vs-FULL-wrong margin. "
            f"Mean peak-to-final erosion was {number(primary_erosion['mean_peak_to_final_erosion'])}. "
            f"The strongest harmful local operation lay within two layers of the largest "
            f"trajectory drop for {number(100 * primary_erosion['culprit_within_2_fraction'], 1)}% "
            f"of samples, versus a deterministic random-layer reference of "
            f"{number(100 * primary_erosion['random_layer_within_2_fraction'], 1)}%.",
            "",
            "Population-level single-operation trajectory reruns:",
            "",
            "| Culprit operation | Reruns | Mean final-margin improvement | Mean erosion reduction | Fraction final margin improved |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for operation, row in trajectory["by_operation"].items():
        lines.append(
            f"| {operation} | {row['count']} | "
            f"{number(row['final_margin_improvement']['mean'])} | "
            f"{number(row['peak_to_final_erosion_reduction']['mean'])} | "
            f"{number(row['fraction_positive_final_margin_improvement'])} |"
        )

    lines.extend(
        [
            "",
            "## Controls",
            "",
            "The no-correction control means only that no correcting route was found "
            "under the matched binary-search budget. The full cohort-comparison table "
            "contains negative fractions and distribution-derived q75/q90 enrichment "
            "for each effect, separately by dataset and jointly.",
            "",
            "| Cohort | Effect | Mean | Median | Negative fraction | Strong-negative q90 fraction |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in controls:
        if row["dataset"] != "joint":
            continue
        lines.append(
            f"| {row['cohort']} | {row['effect']} | {number(row['mean'])} | "
            f"{number(row['median'])} | {number(row['negative_fraction'])} | "
            f"{number(row['strong_negative_q90_fraction'])} |"
        )

    lines.extend(
        [
            "",
            "## Evidence inventory and interpretation boundary",
            "",
            "Raw per-sample/layer/action scores, generated answers, evaluator decisions, "
            "factorial effects, route metadata, trajectories, bootstrap aggregates, and "
            "plots are retained under this analysis directory with SHA-256 sidecars. "
            "Intermediate logit-lens trajectories are supporting evidence; the exact "
            "within-unified four-action interventions are the primary causal evidence.",
            "",
        ]
    )
    write_once(ROOT / "4action_answer_unaligned_report.md", "\n".join(lines))
    print(json.dumps({"report": str(ROOT / '4action_answer_unaligned_report.md')}, indent=2))


if __name__ == "__main__":
    main()
