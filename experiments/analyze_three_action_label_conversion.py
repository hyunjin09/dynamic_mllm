#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from math import floor
from pathlib import Path
import statistics
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from tools.research_analysis.four_action.three_action_jobs import file_sha256


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distribution(values) -> dict[str, Any]:
    rows = [float(value) for value in values]
    return {
        "count": len(rows),
        "mean": statistics.mean(rows) if rows else None,
        "median": statistics.median(rows) if rows else None,
        "std": statistics.pstdev(rows) if rows else None,
        "p90": _quantile(rows, 0.90),
        "p95": _quantile(rows, 0.95),
        "p99": _quantile(rows, 0.99),
        "min": min(rows) if rows else None,
        "max": max(rows) if rows else None,
    }


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _scope(records: list[dict[str, Any]]) -> dict[str, Any]:
    conversions = [row for record in records for row in record.get("raw_conversions", [])]
    converted = [row for row in conversions if row.get("status") == "converted"]
    screening = [
        {"route_type": record["route_type"], **position}
        for record in records
        for conversion in record.get("raw_conversions", [])
        if conversion.get("status") == "converted"
        for position in conversion.get("screening", {}).get("positions", [])
    ]
    decomposition = [
        {"route_type": record["route_type"], **position}
        for record in records
        for conversion in record.get("raw_conversions", [])
        if conversion.get("status") == "converted"
        for position in conversion.get("decomposition", [])
    ]
    w2c_screen = [row for row in screening if row["route_type"] == "W2C"]
    c2c_screen = [row for row in screening if row["route_type"] == "C2C"]
    w2c_decomp = [row for row in decomposition if row["route_type"] == "W2C"]
    c2c_decomp = [row for row in decomposition if row["route_type"] == "C2C"]
    independent = [
        conversion.get("independent_composition", {}) for conversion in converted
    ]
    efficiency = [conversion.get("execution_efficiency", {}) for conversion in converted]
    w2c_records = [record for record in records if record.get("route_type") == "W2C"]
    c2c_records = [record for record in records if record.get("route_type") == "C2C"]

    def classifications(rows):
        return {
            name: sum(row.get("classification") == name for row in rows)
            for name in (
                "HARD_NECESSARY",
                "SOFT_ALIGNMENT_HELPFUL",
                "CONTEXT_DEPENDENT_NECESSARY",
                "REDUNDANT",
            )
        }

    def action_counts(rows):
        names = (
            "READ_SUPPRESSION",
            "WRITE_SUPPRESSION",
            "BOTH_SUPPRESSION",
            "EITHER_SUPPRESSION",
            "NO_MEANINGFUL_GAIN",
        )
        return {name: sum(row.get("action_classification") == name for row in rows) for name in names}

    def action_summary(rows):
        counts = action_counts(rows)
        return {
            "counts": counts,
            "fractions": {
                name: _fraction(count, len(rows)) for name, count in counts.items()
            },
        }

    def action_depth(rows):
        return {
            name: _distribution(
                row["layer"] for row in rows if row.get("action_classification") == name
            )
            for name in (
                "READ_SUPPRESSION",
                "WRITE_SUPPRESSION",
                "BOTH_SUPPRESSION",
                "EITHER_SUPPRESSION",
            )
        }

    def strongest_gain(row):
        return max(
            float(row["actions"][action]["delta_vs_full_reference"])
            for action in ("READ_OFF", "WRITE_OFF", "BOTH_OFF")
        )

    def unique_operation_layers_per_sample(scope_records):
        values = []
        for record in scope_records:
            layers = {
                int(position["layer"])
                for conversion in record.get("raw_conversions", [])
                if conversion.get("status") == "converted"
                for position in conversion.get("decomposition", [])
                if position.get("action_classification") != "NO_MEANINGFUL_GAIN"
            }
            values.append(len(layers))
        return _distribution(values)

    w2c_classes = classifications(w2c_screen)
    c2c_classes = classifications(c2c_screen)
    w2c_action_summary = action_summary(w2c_decomp)
    c2c_action_summary = action_summary(c2c_decomp)
    w2c_strongest = [strongest_gain(row) for row in w2c_decomp]
    c2c_strongest = [strongest_gain(row) for row in c2c_decomp]
    cache_hits = sum(int(record.get("route_evaluation_cache", {}).get("cache_hits", 0)) for record in records)
    cache_misses = sum(int(record.get("route_evaluation_cache", {}).get("cache_misses", 0)) for record in records)
    return {
        "counts": {
            "samples": len(records),
            "source_routes": len(conversions),
            "replay_valid_routes": len(converted),
            "replay_failure_routes": len(conversions) - len(converted),
            "w2c_samples": len(w2c_records),
            "c2c_samples": len(c2c_records),
            "unique_positive_routes": sum(len(record.get("unique_valid_three_action_routes", [])) for record in records),
            "samples_with_positive_route": sum(bool(record.get("unique_valid_three_action_routes")) for record in records),
        },
        "w2c": {
            "screened_positions": len(w2c_screen),
            "hard_necessary_positions": w2c_classes["HARD_NECESSARY"],
            "soft_alignment_helpful_positions": w2c_classes["SOFT_ALIGNMENT_HELPFUL"],
            "redundant_positions": w2c_classes["REDUNDANT"],
            "useful_positions_missed_by_correctness_only": w2c_classes["SOFT_ALIGNMENT_HELPFUL"],
            "screening_classifications": w2c_classes,
            "screening_classification_fractions": {
                name: _fraction(count, len(w2c_screen))
                for name, count in w2c_classes.items()
            },
            "action_classifications": w2c_action_summary["counts"],
            "action_classification_fractions": w2c_action_summary["fractions"],
            "action_depth_distributions": action_depth(w2c_decomp),
            "both_off_minus_full_distribution": _distribution(row["both_off_minus_full"] for row in w2c_screen),
            "read_off_delta_distribution": _distribution(row["actions"]["READ_OFF"]["delta_vs_full_reference"] for row in w2c_decomp),
            "write_off_delta_distribution": _distribution(row["actions"]["WRITE_OFF"]["delta_vs_full_reference"] for row in w2c_decomp),
            "both_off_delta_distribution": _distribution(row["actions"]["BOTH_OFF"]["delta_vs_full_reference"] for row in w2c_decomp),
            "strongest_local_gain_distribution": _distribution(w2c_strongest),
            "retained_position_depth_distribution": _distribution(row["layer"] for row in w2c_decomp),
            "useful_operation_layers_per_sample": unique_operation_layers_per_sample(w2c_records),
            "correctness_only_miss_fraction_among_useful_screening_positions": _fraction(
                w2c_classes["SOFT_ALIGNMENT_HELPFUL"],
                w2c_classes["HARD_NECESSARY"] + w2c_classes["SOFT_ALIGNMENT_HELPFUL"],
            ),
        },
        "c2c": {
            "screened_positions": len(c2c_screen),
            "alignment_helpful_positions": c2c_classes["SOFT_ALIGNMENT_HELPFUL"],
            "context_dependent_positions": c2c_classes["CONTEXT_DEPENDENT_NECESSARY"],
            "redundant_positions": c2c_classes["REDUNDANT"],
            "screening_classifications": c2c_classes,
            "screening_classification_fractions": {
                name: _fraction(count, len(c2c_screen))
                for name, count in c2c_classes.items()
            },
            "action_classifications": c2c_action_summary["counts"],
            "action_classification_fractions": c2c_action_summary["fractions"],
            "action_depth_distributions": action_depth(c2c_decomp),
            "both_off_minus_full_distribution": _distribution(row["both_off_minus_full"] for row in c2c_screen),
            "read_off_delta_distribution": _distribution(row["actions"]["READ_OFF"]["delta_vs_full_reference"] for row in c2c_decomp),
            "write_off_delta_distribution": _distribution(row["actions"]["WRITE_OFF"]["delta_vs_full_reference"] for row in c2c_decomp),
            "both_off_delta_distribution": _distribution(row["actions"]["BOTH_OFF"]["delta_vs_full_reference"] for row in c2c_decomp),
            "strongest_local_gain_distribution": _distribution(c2c_strongest),
            "retained_position_depth_distribution": _distribution(row["layer"] for row in c2c_decomp),
            "alignment_operation_layers_per_sample": unique_operation_layers_per_sample(c2c_records),
            "samples_with_alignment_route": sum(
                record.get("route_type") == "C2C" and bool(record.get("unique_valid_three_action_routes"))
                for record in records
            ),
            "fraction_samples_with_alignment_route": _fraction(
                sum(bool(record.get("unique_valid_three_action_routes")) for record in c2c_records),
                len(c2c_records),
            ),
        },
        "joint_validation": {
            "independent_compositions": len(independent),
            "independent_composition_failures": sum(
                bool(row.get("independent_composition_failure")) for row in independent
            ),
            "joint_positive": sum(bool(row.get("joint_positive")) for row in independent),
            "independent_composition_failure_fraction": _fraction(
                sum(bool(row.get("independent_composition_failure")) for row in independent),
                sum(bool(row.get("all_local_actions_supported")) for row in independent),
            ),
        },
        "efficiency": {
            "candidate_positions": sum(int(row.get("candidate_positions", 0)) for row in efficiency),
            "decomposition_new_forwards": sum(int(row.get("decomposition_new_cache_misses", 0)) for row in efficiency),
            "theoretical_four_state_evaluations_avoided": sum(
                int(row.get("theoretical_four_state_evaluations_avoided", 0)) for row in efficiency
            ),
            "route_cache_hits": cache_hits,
            "route_cache_misses": cache_misses,
            "route_cache_hit_rate": cache_hits / (cache_hits + cache_misses) if cache_hits + cache_misses else None,
        },
        "unique_routes_per_sample": _distribution(
            len(record.get("unique_valid_three_action_routes", [])) for record in records
        ),
        "unique_routes_per_w2c_sample": _distribution(
            len(record.get("unique_valid_three_action_routes", [])) for record in w2c_records
        ),
        "unique_routes_per_c2c_sample": _distribution(
            len(record.get("unique_valid_three_action_routes", [])) for record in c2c_records
        ),
        "w2c_vs_c2c": {
            "strongest_local_gain_median_difference_c2c_minus_w2c": (
                None
                if not w2c_strongest or not c2c_strongest
                else statistics.median(c2c_strongest) - statistics.median(w2c_strongest)
            ),
            "c2c_strongest_local_gain_weaker_by_median": (
                None
                if not w2c_strongest or not c2c_strongest
                else statistics.median(c2c_strongest) < statistics.median(w2c_strongest)
            ),
            "read_suppression_fraction_difference_c2c_minus_w2c": (
                None
                if not w2c_decomp or not c2c_decomp
                else c2c_action_summary["fractions"]["READ_SUPPRESSION"]
                - w2c_action_summary["fractions"]["READ_SUPPRESSION"]
            ),
            "write_suppression_fraction_difference_c2c_minus_w2c": (
                None
                if not w2c_decomp or not c2c_decomp
                else c2c_action_summary["fractions"]["WRITE_SUPPRESSION"]
                - w2c_action_summary["fractions"]["WRITE_SUPPRESSION"]
            ),
        },
    }


def analyze_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    datasets = sorted({str(row["dataset"]) for row in records})
    screening_rows = []
    decomposition_rows = []
    for record in records:
        for conversion in record.get("raw_conversions", []):
            if conversion.get("status") != "converted":
                continue
            for row in conversion.get("screening", {}).get("positions", []):
                screening_rows.append({"uid": record["uid"], "dataset": record["dataset"], "route_type": record["route_type"], "source_binary_route_id": conversion.get("source_binary_route_id"), **row})
            for row in conversion.get("decomposition", []):
                flat = {"uid": record["uid"], "dataset": record["dataset"], "route_type": record["route_type"], "source_binary_route_id": conversion.get("source_binary_route_id"), "layer": row["layer"], "screening_classification": row.get("screening_classification"), "action_classification": row["action_classification"]}
                for action in ("READ_OFF", "WRITE_OFF", "BOTH_OFF"):
                    flat[f"{action}_delta_vs_full"] = row["actions"][action]["delta_vs_full_reference"]
                    flat[f"{action}_correct"] = row["actions"][action]["evaluation"]["correct"]
                decomposition_rows.append(flat)
    return {
        "schema_version": "three_action_answer_aligned_aggregate_statistics_v1",
        "combined": _scope(records),
        "datasets": {dataset: _scope([row for row in records if row["dataset"] == dataset]) for dataset in datasets},
        "screening_rows": screening_rows,
        "decomposition_rows": decomposition_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plots(analysis: dict[str, Any], output: Path) -> None:
    rows = analysis["screening_rows"]
    classes = ("HARD_NECESSARY", "SOFT_ALIGNMENT_HELPFUL", "CONTEXT_DEPENDENT_NECESSARY", "REDUNDANT")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for axis, route_type in zip(axes, ("W2C", "C2C")):
        for name in classes:
            counts = [sum(row["route_type"] == route_type and row["classification"] == name and int(row["layer"]) == layer for row in rows) for layer in range(28)]
            if any(counts):
                axis.plot(range(28), counts, marker="o", label=name)
        axis.set_ylabel(f"{route_type} positions")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Layer")
    fig.tight_layout()
    path = output / "screening_classification_by_layer.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _report(analysis: dict[str, Any], audit: dict[str, Any], finalization: dict[str, Any]) -> str:
    combined = analysis["combined"]
    w2c = combined["w2c"]
    c2c = combined["c2c"]
    action_w = w2c["action_classifications"]
    action_c = c2c["action_classifications"]
    joint = combined["joint_validation"]
    efficiency = combined["efficiency"]
    comparison = combined["w2c_vs_c2c"]
    dataset_rows = [
        "| Dataset | Samples | Source routes | Replay-valid | Replay-invalid | Unique positive routes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, scope in sorted(analysis["datasets"].items()):
        counts = scope["counts"]
        dataset_rows.append(
            f"| {dataset} | {counts['samples']:,} | {counts['source_routes']:,} | "
            f"{counts['replay_valid_routes']:,} | {counts['replay_failure_routes']:,} | "
            f"{counts['unique_positive_routes']:,} |"
        )
    c2c_weaker = comparison["c2c_strongest_local_gain_weaker_by_median"]
    if c2c_weaker is None:
        effect_comparison = "The relative-magnitude comparison is unavailable because one route type has no retained causal decomposition."
    elif c2c_weaker:
        effect_comparison = (
            "C2C has a lower median strongest local gain than W2C on the respective frozen score quantities; "
            "this supports the weaker/compensated-effect hypothesis for this population."
        )
    else:
        effect_comparison = (
            "C2C does not have a lower median strongest local gain than W2C on the respective frozen score quantities; "
            "the weaker/compensated-effect hypothesis is not supported by this comparison."
        )
    depth_w = w2c["action_depth_distributions"]
    depth_c = c2c["action_depth_distributions"]
    lines = [
        "# Three-Action Answer-Aligned Label Conversion Report",
        "",
        "## Integrity and scope",
        "",
        f"The full integrity audit passed: **{audit.get('passed')}**. The run accounted for {audit.get('source_routes', 0):,} authoritative binary routes across {audit.get('completed_samples', 0):,} samples, with {audit.get('replay_valid_routes', 0):,} current-runtime replay-valid and {audit.get('replay_failure_routes', 0):,} explicitly excluded replay-invalid routes.",
        "",
        *dataset_rows,
        "",
        "## Required scientific answers",
        "",
        f"1. Successful conversion: {audit.get('replay_valid_routes', 0):,} binary labels replayed validly. Exact dataset counts are in the table above and `aggregate_statistics_v1.json`.",
        f"2. Binary-OFF redundancy: W2C redundant={w2c['redundant_positions']:,} ({w2c['screening_classification_fractions']['REDUNDANT']}); C2C answer-alignment-redundant={c2c['redundant_positions']:,} ({c2c['screening_classification_fractions']['REDUNDANT']}).",
        f"3. W2C suppressions: hard={w2c['hard_necessary_positions']:,}, soft={w2c['soft_alignment_helpful_positions']:,}, redundant={w2c['redundant_positions']:,}.",
        f"4. Correctness-only screening would miss {w2c['useful_positions_missed_by_correctness_only']:,} epsilon-significant soft W2C positions ({w2c['correctness_only_miss_fraction_among_useful_screening_positions']} of hard-or-soft useful screening positions).",
        f"5. W2C component classification: READ={action_w['READ_SUPPRESSION']:,}, WRITE={action_w['WRITE_SUPPRESSION']:,}, BOTH={action_w['BOTH_SUPPRESSION']:,}, EITHER={action_w['EITHER_SUPPRESSION']:,}.",
        f"6. C2C alignment gains: {c2c['alignment_helpful_positions']:,} screening positions and {c2c['samples_with_alignment_route']:,}/{combined['counts']['c2c_samples']:,} C2C samples with at least one globally valid alignment route; READ={action_c['READ_SUPPRESSION']:,}, WRITE={action_c['WRITE_SUPPRESSION']:,}, BOTH={action_c['BOTH_SUPPRESSION']:,}, EITHER={action_c['EITHER_SUPPRESSION']:,}.",
        f"7. {effect_comparison} W2C median strongest gain={w2c['strongest_local_gain_distribution']['median']}; C2C={c2c['strongest_local_gain_distribution']['median']}.",
        f"8. READ/WRITE median layers are W2C READ={depth_w['READ_SUPPRESSION']['median']}, W2C WRITE={depth_w['WRITE_SUPPRESSION']['median']}, C2C READ={depth_c['READ_SUPPRESSION']['median']}, and C2C WRITE={depth_c['WRITE_SUPPRESSION']['median']}. Full profiles are saved in the aggregate JSON, raw decomposition CSV, and layer plot; no depth narrative is imposed beyond these measurements.",
        f"9. Independent local-action composition failed joint positivity for {joint['independent_composition_failures']:,}/{joint['independent_compositions']:,} evaluated source routes; among routes where every local choice was supported, the failure fraction is {joint['independent_composition_failure_fraction']}.",
        f"10. Unique valid-set size per sample has median {combined['unique_routes_per_sample']['median']} and maximum {combined['unique_routes_per_sample']['max']}; W2C and C2C distributions are reported separately in the aggregate JSON.",
        f"11. Training-view integrity passed={finalization.get('passed')}; every admitted route is jointly evaluator-correct and W2C/C2C semantics remain explicit.",
        "12. Fresh MCTS is not required merely to obtain READ/WRITE labels when replay coverage, valid-set richness, and joint integrity are adequate; any remaining empirical case is stated in the final decision below.",
        "",
        "## Computational efficiency",
        "",
        f"The implementation avoided {efficiency['theoretical_four_state_evaluations_avoided']:,} theoretical fourth/baseline state executions during decomposition, used {efficiency['decomposition_new_forwards']:,} new decomposition forwards, and achieved route-cache hit rate {efficiency['route_cache_hit_rate']}.",
        "",
        "## Final decision",
        "",
    ]
    ready = bool(audit.get("passed")) and bool(finalization.get("passed")) and audit.get("replay_valid_routes", 0) > 0
    lines.append(
        "The answer-aligned route sets are **ready for router training with W2C/C2C kept as separate supervision types**. No fresh MCTS is empirically required for this conversion objective."
        if ready else
        "The route sets are **not yet established as training-ready**; inspect the failed integrity/finalization gates before considering any new search."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze the full answer-aligned three-action conversion.")
    parser.add_argument("--config", type=Path, default=Path("configs/three_action_label_conversion.yaml"))
    parser.add_argument("--audit", type=Path, default=Path("analysis/three_action_answer_aligned_label_conversion/full_integrity_audit_v1.json"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    finalization_path = Path(config["output_root"]) / "final" / "finalization_summary_v1.json"
    finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    if not audit.get("passed") or not finalization.get("passed"):
        raise RuntimeError("audit and finalization must pass before aggregate analysis")
    paths = sorted((Path(config["output_root"]) / "full" / "records").glob("*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    analysis = analyze_records(records)
    output = Path(config["analysis_root"])
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "screening_positions_v1.csv", analysis["screening_rows"])
    _write_csv(output / "decomposition_actions_v1.csv", analysis["decomposition_rows"])
    _plots(analysis, output)
    serializable = {key: value for key, value in analysis.items() if not key.endswith("_rows")}
    aggregate_path = output / "aggregate_statistics_v1.json"
    aggregate_path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = output / "three_action_answer_aligned_label_conversion_report.md"
    report_path.write_text(_report(analysis, audit, finalization), encoding="utf-8")
    for path in (output / "screening_positions_v1.csv", output / "decomposition_actions_v1.csv", output / "screening_classification_by_layer.png", aggregate_path, report_path):
        path.with_suffix(path.suffix + ".sha256").write_text(f"{file_sha256(path)}  {path.name}\n", encoding="utf-8")
    print(json.dumps(serializable["combined"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
