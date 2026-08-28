#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    numeric = float(value)
    return "NA" if not math.isfinite(numeric) else f"{numeric:.{digits}f}"


def _percent(value: Any, digits: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.{digits}f}%"


def _rows_for(rows: Iterable[Mapping[str, Any]], **conditions: Any) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if all(row.get(key) == value for key, value in conditions.items())
    ]


def _depth_conclusion(depth: Mapping[str, Any]) -> str:
    difference = depth.get("read_minus_write_mean_layer")
    low = depth.get("read_minus_write_ci_low")
    high = depth.get("read_minus_write_ci_high")
    if difference is None or low is None or high is None:
        return "There were too few READ- or WRITE-mediated positions for a resolved depth comparison."
    if float(high) < 0:
        return "READ-mediated positions occurred earlier on average than WRITE-mediated positions."
    if float(low) > 0:
        return "READ-mediated positions occurred later on average than WRITE-mediated positions."
    return "The image-group bootstrap interval includes zero, so mean depth did not differ clearly."


def render_report(
    anchor: Mapping[str, Any],
    pilot: Mapping[str, Any],
    stage: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    route_size_rows: list[Mapping[str, Any]],
    continuous_rows: list[Mapping[str, Any]],
    estimate: Mapping[str, Any],
    taxonomy_rows: list[Mapping[str, Any]] | None = None,
) -> str:
    taxonomy = aggregate["taxonomy"]
    necessary = taxonomy["individually_necessary"]
    redundant = taxonomy["redundant"]
    off_count = int(stage["anchor_off_position_count"])
    mechanism_names = (
        ("read_mediated", "READ suppression required"),
        ("write_mediated", "WRITE suppression required"),
        ("either_removal_sufficient", "Either removal sufficient"),
        ("both_required", "Both READ and WRITE required OFF"),
    )
    context = aggregate["context_comparison"]["joint"]
    within = aggregate["context_comparison"].get("within_sample", {})
    depth = aggregate["category_depth_comparison"]["joint"]
    partial_count = sum(int(taxonomy[name]["count"]) for name, _ in mechanism_names[:3])
    necessary_count = int(necessary["count"])
    partial_fraction = partial_count / necessary_count if necessary_count else 0.0
    selected_pilot = next(
        row
        for row in pilot["configurations"]
        if row["name"] == pilot["selected_configuration"]
    )

    lines = [
        "# Route-Conditioned READ/WRITE Decomposition Report",
        "",
        "## Scope and estimand",
        "",
        "This experiment decomposes one deterministic, current-runtime-correct binary "
        "anchor route per frozen A+ sample. At each anchor-OFF position, every other "
        "layer remains fixed to that correcting route while the target layer is evaluated "
        "as M00=BOTH_OFF, M10=WRITE_OFF/READ_ONLY, M01=READ_OFF/WRITE_ONLY, and "
        "M11=FULL restoration. All continuous effects are within the unified executor.",
        "",
        "This is **route-conditioned** evidence. The earlier experiment is a distinct "
        "**FULL-context** local intervention in which every non-target layer is FULL. "
        "Neither result is global causal attribution, and the two contexts are never "
        "conflated below.",
        "",
        "## Integrity, pilot, and execution",
        "",
        f"The full merge contains {stage['sample_count']:,} samples, {off_count:,} "
        f"anchor-OFF positions, and {stage['flat_action_row_count']:,} saved action rows. "
        "Every sample/layer/action gate and exact-coverage check passed.",
        "",
        f"The matched all-eight-H100 pilot selected `{pilot['selected_configuration']}` "
        f"({pilot['selected_replicas_per_gpu']} replica(s)/GPU) at "
        f"{_number(selected_pilot['useful_new_cells_per_second'])} valid new cells/s. "
        f"The prelaunch estimate was {_number(estimate['expected_wall_hours'], 2)} wall "
        f"hours and {_number(estimate['expected_gpu_hours'], 2)} GPU-hours.",
        "",
        "## Required final questions",
        "",
        "### 1. How many frozen A+ samples had a validated current-runtime anchor?",
        "",
        f"{anchor['validated_anchor_count']:,}/{anchor['frozen_a_plus_count']:,} frozen "
        f"A+ samples had a validated current-runtime correcting anchor: "
        f"{anchor['dataset_counts'].get('gqa', 0):,} GQA and "
        f"{anchor['dataset_counts'].get('textvqa', 0):,} TextVQA. "
        f"{anchor['excluded_no_current_correct_anchor_count']:,} were excluded because "
        "no cached correcting route remained correct; no route was invented or searched.",
        "",
        "### 2. How many anchor-OFF positions were individually necessary?",
        "",
        f"{necessary_count:,}/{off_count:,} ({_percent(necessary['fraction'])}, "
        f"image-group bootstrap 95% CI {_percent(necessary['ci_low'])} to "
        f"{_percent(necessary['ci_high'])}) were individually necessary: restoring FULL "
        "made the correcting anchor wrong. The remaining "
        f"{int(redundant['count']):,}/{off_count:,} ({_percent(redundant['fraction'])}) "
        "were redundant in this anchor-route context.",
        "",
        "### 3. Which suppression mechanism preserved correction among necessary positions?",
        "",
        "| Mechanism | Count | Share among necessary | Image-group bootstrap 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for name, label in mechanism_names:
        row = taxonomy[name]
        lines.append(
            f"| {label} | {int(row['count']):,} | "
            f"{_percent(row['conditional_among_necessary_fraction'])} | "
            f"{_percent(row['conditional_among_necessary_ci_low'])}–"
            f"{_percent(row['conditional_among_necessary_ci_high'])} |"
        )
    if taxonomy_rows:
        lines.extend(
            [
                "",
                "Dataset-specific conditional shares are retained with the same "
                "image-group bootstrap contract:",
                "",
                "| Dataset | Mechanism | Count | Share among necessary | 95% CI |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for dataset in ("gqa", "textvqa", "joint"):
            for name, label in mechanism_names:
                matches = _rows_for(taxonomy_rows, dataset=dataset, metric=name)
                if not matches:
                    continue
                row = matches[0]
                lines.append(
                    f"| {dataset} | {label} | {int(row['count']):,} | "
                    f"{_percent(row.get('conditional_necessary_estimate'))} | "
                    f"{_percent(row.get('conditional_necessary_ci_low'))}–"
                    f"{_percent(row.get('conditional_necessary_ci_high'))} |"
                )
    lines.extend(
        [
            "",
            "### 4. Are READ- and WRITE-mediated corrections distributed differently across depth?",
            "",
            f"Mean layer was {_number(depth.get('read_mediated_mean_layer'), 2)} for "
            f"READ-mediated and {_number(depth.get('write_mediated_mean_layer'), 2)} for "
            f"WRITE-mediated positions. The READ-minus-WRITE difference was "
            f"{_number(depth.get('read_minus_write_mean_layer'), 2)} (image-group "
            f"bootstrap 95% CI {_number(depth.get('read_minus_write_ci_low'), 2)} to "
            f"{_number(depth.get('read_minus_write_ci_high'), 2)}). "
            f"{_depth_conclusion(depth)} Layerwise counts and effects for all 28 layers "
            "are in `aggregate/depth_taxonomy.*`, `aggregate/depth_effects.*`, and "
            "`figures/`.",
            "",
            "### 5. Do long correcting routes contain necessary operations or redundancy?",
            "",
            "| Anchor OFF-count stratum | Samples | OFF positions | Necessary | Redundant |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    joint_route_size = _rows_for(route_size_rows, dataset="joint")
    for row in joint_route_size:
        lines.append(
            f"| {row['route_size_stratum']} | {int(row['sample_count']):,} | "
            f"{int(row['off_position_count']):,} | {_percent(row['necessary_fraction'])} | "
            f"{_percent(row['redundant_fraction'])} |"
        )
    if len(joint_route_size) >= 2:
        shortest, longest = joint_route_size[0], joint_route_size[-1]
        relation = "more" if longest["redundant_fraction"] > shortest["redundant_fraction"] else "no more"
        lines.append(
            f"The longest populated stratum had {relation} redundancy than the shortest "
            f"({_percent(longest['redundant_fraction'])} versus "
            f"{_percent(shortest['redundant_fraction'])}); the full stratification retains "
            "category-specific shares and confidence intervals."
        )
    lines.extend(
        [
            "",
            "### 6. How often does FULL-context local harmfulness agree with route-conditioned necessity?",
            "",
            f"The discrete FULL-context-local-rescue versus route-necessity classification "
            f"agreed on {_percent(context['discrete_context_agreement_fraction'])} of "
            f"{context['matched_cell_count']:,} matched sample/layer positions. FULL-context "
            f"local rescue recalled {_percent(context['route_necessity_recall_from_full_context'])} "
            "of route-necessary positions and had "
            f"{_percent(context['full_context_rescue_precision_for_route_necessity'])} precision "
            "for route necessity. Continuous harmful-effect sign agreement was "
            f"{_percent(context['read_harm_sign_agreement_fraction'])} for READ and "
            f"{_percent(context['write_harm_sign_agreement_fraction'])} for WRITE; pooled "
            f"Spearman correlations were {_number(context['read_effect_spearman'])} and "
            f"{_number(context['write_effect_spearman'])}, respectively.",
            "",
            "### 7. How often did route conditioning reveal operations missed in FULL context?",
            "",
            f"{context['route_necessary_full_context_missed_count']:,} route-necessary "
            f"positions ({_percent(context['route_necessary_full_context_missed_fraction'])} "
            "of all route-necessary positions) had no discrete W→C rescue in the earlier "
            "FULL-context single-layer sweep. This is direct evidence that the final "
            "behavioral effect of a layer can depend on the other suppressions in the "
            "successful trajectory. Median within-sample FULL-versus-route rank correlation "
            f"was {_number(within.get('median_read_spearman'))} for READ and "
            f"{_number(within.get('median_write_spearman'))} for WRITE.",
            "",
            "### 8. Does binary routing correct errors by suppressing answer-unaligned READ/WRITE?",
            "",
        ]
    )
    if necessary_count:
        lines.append(
            f"Conditionally, yes for {necessary_count:,} individually necessary OFF positions: "
            f"{partial_count:,} ({_percent(partial_fraction)}) permitted at least one "
            "READ/WRITE component to be restored while preserving correction, whereas "
            f"{taxonomy['both_required']['count']:,} required both components suppressed. "
            "This supports answer-unaligned READ and/or WRITE suppression as a mechanism "
            "inside these selected correcting routes, but does not make the corresponding "
            "operation globally harmful or prove that every OFF choice caused the route."
        )
    else:
        lines.append(
            "No anchor-OFF position was individually necessary, so this experiment does "
            "not support a component-level suppression mechanism for the cached routes."
        )
    lines.extend(
        [
            "",
            "### 9. Does the evidence justify a true four-action trajectory search/router?",
            "",
        ]
    )
    if partial_count:
        pilot_cells = 64 * 4 * 8 * 2
        rate = float(selected_pilot["useful_new_cells_per_second"])
        hours = pilot_cells / rate / 3600.0
        lines.append(
            f"The {partial_count:,} component-relaxable necessary positions justify a "
            "separately approved bounded joint-refinement pilot, but not an immediate claim "
            "that a four-action search/router will improve accuracy or compute. Individual "
            "relaxations were tested one at a time; simultaneous relaxations may interact. "
            "A suitable next proposal is a 64-sample, route-size-stratified beam refinement "
            "from the validated anchors, beam width 4, at most 8 OFF positions, testing at "
            f"most two partial restorations per expansion (upper bound {pilot_cells:,} "
            f"evaluations, approximately {_number(hours, 2)} eight-GPU wall hours or "
            f"{_number(8 * hours, 2)} GPU-hours at the measured throughput). Do not launch "
            "this proposed experiment without a separate decision."
        )
    else:
        lines.append(
            "No. No component-specific relaxation preserved correction, so these results "
            "do not motivate a four-action search/router. Do not launch joint refinement."
        )
    lines.extend(
        [
            "",
            "## Continuous route-conditioned factorial effects",
            "",
            "The sign convention is restoration minus suppression. Because M00 is the "
            "correct anchor, a negative effect means restoring the named operation shifts "
            "the fixed answer margin away from the correct answer. No magnitude threshold "
            "was imposed.",
            "",
            "| Effect | Mean | Median | Fraction negative |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in _rows_for(continuous_rows, dataset="joint", taxonomy="all"):
        lines.append(
            f"| {row['effect']} | {_number(row['estimate'])} | {_number(row['median'])} | "
            f"{_percent(row['negative_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Sample structure and evidence inventory",
            "",
            "Sample-level mechanism counts: "
            + ", ".join(
                f"{name}={count:,}"
                for name, count in sorted(aggregate["sample_structure_counts"].items())
            )
            + ".",
            "",
            "Raw per-sample/layer/action outputs, fixed targets, generated answers, "
            "evaluator correctness, continuous scores, effects, route metadata, worker "
            "provenance, exact mergers, aggregate tables, and figures are retained under "
            "`analysis/4action_route_conditioned/` with SHA-256 sidecars.",
            "",
        ]
    )
    return "\n".join(lines)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the final route-conditioned report.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/four_action_route_conditioned.yaml"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(config["output_root"])
    anchor = json.loads((root / "anchor_route_summary.json").read_text(encoding="utf-8"))
    pilot = json.loads((root / "pilot_benchmark_summary.json").read_text(encoding="utf-8"))
    stage = json.loads((Path(config["full_root"]) / "full" / "stage_summary.json").read_text(encoding="utf-8"))
    aggregate = json.loads((root / "aggregate_summary.json").read_text(encoding="utf-8"))
    estimate = json.loads((root / "compute_estimate.json").read_text(encoding="utf-8"))
    route_size = _read_jsonl(root / "aggregate" / "route_size_stratification.jsonl")
    continuous = _read_jsonl(root / "aggregate" / "continuous_effects.jsonl")
    taxonomy = _read_jsonl(root / "aggregate" / "necessity_taxonomy.jsonl")
    report = render_report(
        anchor,
        pilot,
        stage,
        aggregate,
        route_size,
        continuous,
        estimate,
        taxonomy,
    )
    output = root / "route_conditioned_decomposition_report.md"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(report, encoding="utf-8")
    output.with_name(output.name + ".sha256").write_text(
        f"{_sha256_file(output)}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"report": str(output), "sha256": _sha256_file(output)}, indent=2))


if __name__ == "__main__":
    main()
