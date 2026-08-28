#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ("gqa", "textvqa", "chartqa", "wemath20_standard", "wemath2pro")
ACTIONS = ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")
SUBSTANTIAL_READ_WRITE_ROUTE_FRACTION = 0.10


def safe_filename(uid: str) -> str:
    readable = uid.replace(":", "__").replace("/", "_")
    return f"{readable}_{hashlib.sha256(uid.encode()).hexdigest()[:10]}.json"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def describe(values) -> dict[str, float | int | None]:
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    return {
        "count": int(data.size),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p10": float(np.quantile(data, 0.10)),
        "p90": float(np.quantile(data, 0.90)),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def _new_group() -> dict[str, Any]:
    return {
        "counts": Counter(),
        "source_off": [],
        "w2c_source_off": [],
        "c2c_source_off": [],
        "purified_ignore": [],
        "redundant_off_removed": [],
        "w2c_final_cost": [],
        "c2c_final_cost": [],
        "w2c_margin_change": [],
        "c2c_margin_change": [],
        "unique_routes_per_sample": [],
        "layer_actions": defaultdict(Counter),
        "w2c_source_off_by_layer": Counter(),
        "w2c_purification_restored_by_layer": Counter(),
        "w2c_refinement_actions_by_layer": defaultdict(Counter),
        "remaining_anchor_actions": Counter(),
        "w2c_final_action_counts": Counter(),
        "w2c_all_off_correct_final_action_counts": Counter(),
        "w2c_all_off_wrong_final_action_counts": Counter(),
        "w2c_all_off_strata": {
            name: {
                "counts": Counter(),
                "source_off": [],
                "purified_ignore": [],
                "redundant_off_removed": [],
                "final_cost": [],
                "margin_change": [],
                "remaining_anchor_actions": Counter(),
                "joint": Counter(),
            }
            for name in ("all_off_correct", "all_off_wrong")
        },
        "c2c_final_action_counts": Counter(),
        "joint": Counter(),
        "runtime_seconds": [],
        "peak_gpu_memory_bytes": [],
        "max_rss_kib": [],
        "execution_contract_sha256": set(),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def build_final_decisions(
    combined: dict[str, Any], full_audit: dict[str, Any]
) -> dict[str, Any]:
    """Turn the plan's final questions into explicit, auditable decisions.

    The 10% route-level rule is declared before the full result is available and
    applies only to the qualitative word "substantial". Exact counts and
    fractions remain the primary evidence.
    """
    counts = combined["counts"]
    w2c_unique = int(counts.get("corrective_w2c_unique_routes", 0))
    read_only = int(
        counts.get("corrective_w2c_unique_routes_using_read_only", 0)
    )
    write_only = int(
        counts.get("corrective_w2c_unique_routes_using_write_only", 0)
    )
    either = int(
        counts.get(
            "corrective_w2c_unique_routes_using_read_or_write_only", 0
        )
    )
    either_fraction = _ratio(either, w2c_unique)
    zero_valid_samples = int(counts.get("samples_without_current_valid_route", 0))
    unresolved_failures = len(full_audit.get("unresolved_worker_failure_rows", []))
    integrity_clean = bool(full_audit.get("passed")) and unresolved_failures == 0
    training_ready = (
        integrity_clean
        and int(counts.get("unique_valid_routes", 0)) > 0
        and zero_valid_samples == 0
    )
    if training_ready:
        readiness = "ready"
    elif integrity_clean and int(counts.get("unique_valid_routes", 0)) > 0:
        readiness = "ready_with_documented_coverage_exclusions"
    else:
        readiness = "not_ready"
    if training_ready:
        search_decision = "not_needed"
        search_reason = (
            "Conversion provides at least one current-valid jointly executed label "
            "for every source sample."
        )
    elif integrity_clean and zero_valid_samples > 0:
        search_decision = "needed_only_if_excluded_sample_coverage_is_required"
        search_reason = (
            f"{zero_valid_samples} source samples have no currently valid converted "
            "route; fresh search is not needed for the retained valid set itself."
        )
    else:
        search_decision = "not_assessable"
        search_reason = "The full integrity gate has not established a clean label set."
    return {
        "successfully_converted_binary_labels": int(
            counts.get("source_replay_valid_routes", 0)
        ),
        "source_binary_replay_failures": int(
            counts.get("source_replay_failure_routes", 0)
        ),
        "unique_valid_four_action_labels": int(
            counts.get("unique_valid_routes", 0)
        ),
        "read_write_structure": {
            "w2c_unique_routes": w2c_unique,
            "using_read_only": read_only,
            "using_write_only": write_only,
            "using_either": either,
            "using_read_only_fraction": _ratio(read_only, w2c_unique),
            "using_write_only_fraction": _ratio(write_only, w2c_unique),
            "using_either_fraction": either_fraction,
            "substantial": bool(
                either_fraction is not None
                and either_fraction >= SUBSTANTIAL_READ_WRITE_ROUTE_FRACTION
            ),
            "substantial_criterion": (
                "at least 10% of unique corrective W2C routes contain READ_ONLY "
                "or WRITE_ONLY; exact prevalence is the primary evidence"
            ),
        },
        "keep_w2c_and_c2c_separate": True,
        "training_readiness": {
            "decision": readiness,
            "full_integrity_audit_passed": bool(full_audit.get("passed")),
            "unresolved_worker_failure_rows": unresolved_failures,
            "samples_without_current_valid_route": zero_valid_samples,
        },
        "fresh_four_action_search": {
            "decision": search_decision,
            "reason": search_reason,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze the completed four-action label conversion.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("datasets/mcts_labels_4action/source_inventory_v1/source_manifest_v1.jsonl"),
    )
    parser.add_argument(
        "--records-root",
        type=Path,
        default=Path("datasets/mcts_labels_4action/conversion_v1/full/records"),
    )
    parser.add_argument(
        "--source-summary",
        type=Path,
        default=Path(
            "datasets/mcts_labels_4action/source_inventory_v1/source_inventory_summary_v1.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/4action_label_conversion/final"),
    )
    parser.add_argument(
        "--pilot-audit",
        type=Path,
        default=Path("analysis/4action_label_conversion/pilot_audit_v1.json"),
    )
    parser.add_argument(
        "--full-audit",
        type=Path,
        default=Path("analysis/4action_label_conversion/full_integrity_audit_v1.json"),
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.pilot_audit.is_file() or not args.full_audit.is_file():
        raise FileNotFoundError("pilot and full integrity audits must exist before analysis")
    pilot_audit = json.loads(args.pilot_audit.read_text())
    full_audit = json.loads(args.full_audit.read_text())
    source_summary = json.loads(args.source_summary.read_text())
    if not pilot_audit.get("passed") or not full_audit.get("passed"):
        raise RuntimeError("refusing final analysis because an integrity audit did not pass")

    sources = read_jsonl(args.source_manifest)
    groups = {name: _new_group() for name in (*DATASETS, "combined")}
    for source in sources:
        path = args.records_root / safe_filename(str(source["uid"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text())
        if not result.get("passed") or result["uid"] != source["uid"]:
            raise ValueError(f"invalid result: {path}")
        converted_for_sample = [
            row for row in result["raw_conversions"] if row["status"] == "converted"
        ]
        has_all_off_w2c = any(
            row["label_semantics"] == "corrective_w2c" and bool(row["all_off_seed"])
            for row in converted_for_sample
        )
        all_off_correct = bool(result["current_unified_all_off"]["correct"])
        for group_name in (source["dataset"], "combined"):
            group = groups[group_name]
            counts = group["counts"]
            counts["source_samples"] += 1
            counts["source_positive_routes"] += int(source["source_positive_route_count"])
            counts[f"{result['label_semantics']}_samples"] += 1
            counts["source_replay_valid_routes"] += int(result["source_route_replay_valid_count"])
            counts["source_replay_failure_routes"] += int(result["source_route_replay_failure_count"])
            counts["unique_valid_routes"] += len(result["unique_valid_four_action_routes"])
            counts["samples_without_current_valid_route"] += result["canonical_4action_route"] is None
            counts["source_full_correctness_match_samples"] += bool(
                result["source_full_diagnostic"].get("correctness_match")
            )
            counts["source_full_correctness_shift_samples"] += (
                result["source_full_diagnostic"].get("correctness_match") is False
            )
            if result["label_semantics"] == "corrective_w2c":
                counts["w2c_samples_with_all_off_seed"] += has_all_off_w2c
                counts["w2c_samples_without_all_off_seed"] += not has_all_off_w2c
                counts[
                    "w2c_samples_all_off_correct"
                    if all_off_correct
                    else "w2c_samples_all_off_wrong"
                ] += 1
            group["unique_routes_per_sample"].append(
                len(result["unique_valid_four_action_routes"])
            )
            group["runtime_seconds"].append(float(result["runtime"]["elapsed_seconds"]))
            group["peak_gpu_memory_bytes"].append(
                int(result["runtime"]["peak_gpu_memory_bytes"])
            )
            group["max_rss_kib"].append(int(result["runtime"]["max_rss_kib"]))
            group["execution_contract_sha256"].add(
                result["execution_contract"]["contract_sha256"]
            )

        for conversion in result["raw_conversions"]:
            for group_name in (source["dataset"], "combined"):
                group = groups[group_name]
                group["source_off"].append(int(conversion["source_off_count"]))
                if conversion["status"] != "converted":
                    continue
                semantics = conversion["label_semantics"]
                group["counts"][f"{semantics}_routes"] += 1
                if conversion["all_off_seed"] and semantics == "corrective_w2c":
                    group["counts"]["all_off_w2c_routes"] += 1
                final = conversion["final_route"]
                group[
                    "w2c_final_action_counts"
                    if semantics == "corrective_w2c"
                    else "c2c_final_action_counts"
                ].update(final)
                if semantics == "corrective_w2c":
                    stratum = "all_off_correct" if all_off_correct else "all_off_wrong"
                    stratum_group = group["w2c_all_off_strata"][stratum]
                    stratum_group["counts"]["converted_routes"] += 1
                    stratum_group["source_off"].append(int(conversion["source_off_count"]))
                    group[f"w2c_{stratum}_final_action_counts"].update(final)
                    group["counts"][f"w2c_{stratum}_routes"] += 1
                final_cost = sum(
                    {"FULL": 0, "READ_ONLY": 1, "WRITE_ONLY": 1, "IGNORE": 2}[action]
                    for action in final
                )
                group[
                    "w2c_final_cost" if semantics == "corrective_w2c" else "c2c_final_cost"
                ].append(final_cost)
                if semantics == "corrective_w2c":
                    stratum_group["final_cost"].append(final_cost)
                group[
                    "w2c_source_off" if semantics == "corrective_w2c" else "c2c_source_off"
                ].append(int(conversion["source_off_count"]))
                source_margin = conversion["source_route_evaluation"].get(
                    "answer_alignment_margin"
                )
                final_margin = conversion["final_evaluation"].get("answer_alignment_margin")
                if source_margin is not None and final_margin is not None:
                    margin_change = float(final_margin) - float(source_margin)
                    group[
                        "w2c_margin_change"
                        if semantics == "corrective_w2c"
                        else "c2c_margin_change"
                    ].append(margin_change)
                    if semantics == "corrective_w2c":
                        stratum_group["margin_change"].append(margin_change)
                if semantics == "corrective_w2c":
                    purification = conversion["purification"]
                    purified = purification["route"]
                    purified_ignore = purified.count("IGNORE")
                    group["purified_ignore"].append(purified_ignore)
                    stratum_group["purified_ignore"].append(purified_ignore)
                    group["redundant_off_removed"].append(
                        int(conversion["source_off_count"]) - purified_ignore
                    )
                    stratum_group["redundant_off_removed"].append(
                        int(conversion["source_off_count"]) - purified_ignore
                    )
                    for layer, (source_on, purified_action) in enumerate(
                        zip(conversion["source_binary_route"], purified)
                    ):
                        if int(source_on) == 0:
                            group["w2c_source_off_by_layer"][layer] += 1
                        if int(source_on) == 0 and purified_action == "FULL":
                            group["w2c_purification_restored_by_layer"][layer] += 1
                    for index, anchor_action in enumerate(purified):
                        if anchor_action == "IGNORE":
                            group["remaining_anchor_actions"][final[index]] += 1
                            stratum_group["remaining_anchor_actions"][final[index]] += 1
                            group["w2c_refinement_actions_by_layer"][index][
                                final[index]
                            ] += 1
                    if conversion["all_off_seed"]:
                        group["counts"]["all_off_w2c_routes_selective"] += any(
                            action != "IGNORE" for action in final
                        )
                        group["counts"]["all_off_w2c_routes_refined_to_full"] += all(
                            action == "FULL" for action in final
                        )
                    refinement = conversion["refinement"]
                    for field in (
                        "first_round_candidate_count",
                        "first_round_correct_count",
                        "composite_candidate_count",
                        "composite_correct_count",
                        "independently_supported_composite_count",
                        "independent_composition_failure_count",
                    ):
                        group["joint"][field] += int(refinement.get(field, 0))
                        stratum_group["joint"][field] += int(refinement.get(field, 0))

        for unique in result["unique_valid_four_action_routes"]:
            for group_name in (source["dataset"], "combined"):
                group = groups[group_name]
                prefix = unique["label_semantics"]
                group["counts"][f"{prefix}_unique_routes"] += 1
                group["counts"][f"{prefix}_unique_routes_using_read_only"] += (
                    "READ_ONLY" in unique["route"]
                )
                group["counts"][f"{prefix}_unique_routes_using_write_only"] += (
                    "WRITE_ONLY" in unique["route"]
                )
                group["counts"][f"{prefix}_unique_routes_using_ignore"] += (
                    "IGNORE" in unique["route"]
                )
                group["counts"][
                    f"{prefix}_unique_routes_using_read_or_write_only"
                ] += (
                    "READ_ONLY" in unique["route"]
                    or "WRITE_ONLY" in unique["route"]
                )
                for layer, action in enumerate(unique["route"]):
                    group["layer_actions"][(unique["label_semantics"], layer)][action] += 1

    summaries = {}
    layer_rows = []
    refinement_rows = []
    for name, group in groups.items():
        counts = dict(group["counts"])
        valid = counts.get("source_replay_valid_routes", 0)
        unique_count = counts.get("unique_valid_routes", 0)
        w2c_routes = counts.get("corrective_w2c_routes", 0)
        w2c_unique = counts.get("corrective_w2c_unique_routes", 0)
        joint = dict(group["joint"])
        remaining_count = sum(group["remaining_anchor_actions"].values())
        w2c_action_total = sum(group["w2c_final_action_counts"].values())
        c2c_action_total = sum(group["c2c_final_action_counts"].values())
        all_off_strata = {}
        for stratum in ("all_off_correct", "all_off_wrong"):
            action_counts = group[f"w2c_{stratum}_final_action_counts"]
            action_total = sum(action_counts.values())
            stratum_group = group["w2c_all_off_strata"][stratum]
            stratum_remaining_total = sum(stratum_group["remaining_anchor_actions"].values())
            stratum_joint = dict(stratum_group["joint"])
            all_off_strata[stratum] = {
                "samples": counts.get(f"w2c_samples_{stratum}", 0),
                "converted_routes": counts.get(f"w2c_{stratum}_routes", 0),
                "source_off_count": describe(stratum_group["source_off"]),
                "purified_ignore_count": describe(stratum_group["purified_ignore"]),
                "redundant_off_removed": describe(stratum_group["redundant_off_removed"]),
                "final_suppression_component_cost": describe(stratum_group["final_cost"]),
                "source_to_refined_margin_change": describe(stratum_group["margin_change"]),
                "final_action_counts": dict(action_counts),
                "final_action_fractions": {
                    action: _ratio(action_counts[action], action_total)
                    for action in ACTIONS
                },
                "remaining_anchor_action_counts": dict(
                    stratum_group["remaining_anchor_actions"]
                ),
                "remaining_anchor_action_fractions": {
                    action: _ratio(
                        stratum_group["remaining_anchor_actions"][action],
                        stratum_remaining_total,
                    )
                    for action in ACTIONS
                },
                "joint_refinement": {
                    **stratum_joint,
                    "independent_composition_failure_fraction": _ratio(
                        stratum_joint.get("independent_composition_failure_count", 0),
                        stratum_joint.get("independently_supported_composite_count", 0),
                    ),
                },
            }
        summary = {
            "counts": counts,
            "replay_valid_fraction": _ratio(valid, counts.get("source_positive_routes", 0)),
            "deduplication_ratio_unique_over_converted": _ratio(unique_count, valid),
            "source_off_count": describe(group["source_off"]),
            "w2c_source_off_count": describe(group["w2c_source_off"]),
            "c2c_source_off_count": describe(group["c2c_source_off"]),
            "w2c_purified_ignore_count": describe(group["purified_ignore"]),
            "w2c_redundant_off_removed": describe(group["redundant_off_removed"]),
            "w2c_final_suppression_component_cost": describe(group["w2c_final_cost"]),
            "c2c_final_suppression_component_cost": describe(group["c2c_final_cost"]),
            "w2c_source_to_refined_margin_change": describe(group["w2c_margin_change"]),
            "c2c_source_to_mapped_margin_change": describe(group["c2c_margin_change"]),
            "unique_routes_per_sample": describe(group["unique_routes_per_sample"]),
            "remaining_w2c_anchor_action_counts": dict(group["remaining_anchor_actions"]),
            "remaining_w2c_anchor_action_fractions": {
                action: _ratio(group["remaining_anchor_actions"][action], remaining_count)
                for action in ACTIONS
            },
            "w2c_raw_final_action_counts": dict(group["w2c_final_action_counts"]),
            "w2c_raw_final_action_fractions": {
                action: _ratio(group["w2c_final_action_counts"][action], w2c_action_total)
                for action in ACTIONS
            },
            "c2c_raw_action_counts": dict(group["c2c_final_action_counts"]),
            "c2c_raw_action_fractions": {
                action: _ratio(group["c2c_final_action_counts"][action], c2c_action_total)
                for action in ACTIONS
            },
            "joint_refinement": {
                **joint,
                "first_round_correct_fraction": _ratio(
                    joint.get("first_round_correct_count", 0),
                    joint.get("first_round_candidate_count", 0),
                ),
                "composite_correct_fraction": _ratio(
                    joint.get("composite_correct_count", 0),
                    joint.get("composite_candidate_count", 0),
                ),
                "independent_composition_failure_fraction": _ratio(
                    joint.get("independent_composition_failure_count", 0),
                    joint.get("independently_supported_composite_count", 0),
                ),
            },
            "w2c_read_only_route_fraction": _ratio(
                counts.get("corrective_w2c_unique_routes_using_read_only", 0), w2c_unique
            ),
            "w2c_write_only_route_fraction": _ratio(
                counts.get("corrective_w2c_unique_routes_using_write_only", 0), w2c_unique
            ),
            "w2c_still_ignore_route_fraction": _ratio(
                counts.get("corrective_w2c_unique_routes_using_ignore", 0), w2c_unique
            ),
            "all_off_w2c_fraction": _ratio(
                counts.get("all_off_w2c_routes", 0), w2c_routes
            ),
            "all_off_w2c_selective_refinement_fraction": _ratio(
                counts.get("all_off_w2c_routes_selective", 0),
                counts.get("all_off_w2c_routes", 0),
            ),
            "w2c_current_all_off_strata": all_off_strata,
            "sample_runtime_seconds": describe(group["runtime_seconds"]),
            "peak_gpu_memory_bytes": describe(group["peak_gpu_memory_bytes"]),
            "max_rss_kib": describe(group["max_rss_kib"]),
            "execution_contract_sha256": sorted(group["execution_contract_sha256"]),
        }
        summaries[name] = summary
        for layer in range(28):
            source_off = group["w2c_source_off_by_layer"][layer]
            restored = group["w2c_purification_restored_by_layer"][layer]
            refinement_actions = group["w2c_refinement_actions_by_layer"][layer]
            remaining = sum(refinement_actions.values())
            refinement_rows.append(
                {
                    "dataset": name,
                    "layer": layer,
                    "w2c_source_off_count": source_off,
                    "restored_to_full_during_purification_count": restored,
                    "restored_to_full_during_purification_fraction": _ratio(
                        restored, source_off
                    ),
                    "remaining_purified_ignore_count": remaining,
                    **{
                        f"refined_{action}_count": refinement_actions[action]
                        for action in ACTIONS
                    },
                    **{
                        f"refined_{action}_fraction": _ratio(
                            refinement_actions[action], remaining
                        )
                        for action in ACTIONS
                    },
                }
            )
        for (semantics, layer), action_counts in sorted(group["layer_actions"].items()):
            total = sum(action_counts.values())
            layer_rows.append(
                {
                    "dataset": name,
                    "label_semantics": semantics,
                    "layer": layer,
                    "total_unique_routes": total,
                    **{f"{action}_count": action_counts[action] for action in ACTIONS},
                    **{
                        f"{action}_fraction": _ratio(action_counts[action], total)
                        for action in ACTIONS
                    },
                }
            )

    summaries["operational"] = {
        "pilot": {
            "job": pilot_audit.get("job"),
            "jobs": pilot_audit.get("jobs"),
            "throughput": pilot_audit.get("throughput"),
            "telemetry": pilot_audit.get("telemetry"),
        },
        "full": {
            "jobs": full_audit.get("slurm_jobs"),
            "throughput": full_audit.get("throughput"),
            "telemetry": full_audit.get("telemetry"),
            "worker_failure_rows": len(full_audit.get("worker_failure_rows", [])),
            "unresolved_worker_failure_rows": len(
                full_audit.get("unresolved_worker_failure_rows", [])
            ),
        },
    }
    combined = summaries["combined"]
    final_decisions = build_final_decisions(combined, full_audit)
    summaries["final_decisions"] = final_decisions
    summary_path = args.output_dir / "aggregate_statistics_v1.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    layer_frame = pd.DataFrame(layer_rows)
    layer_frame.to_csv(args.output_dir / "layer_action_frequencies_v1.csv", index=False)
    refinement_frame = pd.DataFrame(refinement_rows)
    refinement_frame.to_csv(
        args.output_dir / "w2c_refinement_by_layer_v1.csv", index=False
    )

    combined_layers = layer_frame[
        (layer_frame["dataset"] == "combined")
        & (layer_frame["label_semantics"] == "corrective_w2c")
    ]
    if not combined_layers.empty:
        figure, axis = plt.subplots(figsize=(10, 5))
        for action in ACTIONS:
            axis.plot(
                combined_layers["layer"],
                combined_layers[f"{action}_fraction"],
                marker="o",
                markersize=3,
                label=action,
            )
        axis.set_xlabel("Decoder layer")
        axis.set_ylabel("Fraction of unique W2C labels")
        axis.set_ylim(0, 1)
        axis.set_title("Four-action composition by layer (combined W2C)")
        axis.legend(ncol=4)
        figure.tight_layout()
        figure.savefig(args.output_dir / "w2c_action_fraction_by_layer.png", dpi=180)
        plt.close(figure)

    combined_refinement = refinement_frame[refinement_frame["dataset"] == "combined"]
    if not combined_refinement.empty:
        figure, axis = plt.subplots(figsize=(10, 5))
        for action in ("READ_ONLY", "WRITE_ONLY", "IGNORE", "FULL"):
            axis.plot(
                combined_refinement["layer"],
                combined_refinement[f"refined_{action}_fraction"],
                marker="o",
                markersize=3,
                label=action,
            )
        axis.set_xlabel("Decoder layer")
        axis.set_ylabel("Fraction among purified corrective positions")
        axis.set_ylim(0, 1)
        axis.set_title("W2C refinement action by model depth")
        axis.legend(ncol=4)
        figure.tight_layout()
        figure.savefig(args.output_dir / "w2c_refinement_action_by_layer.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(
            combined_refinement["layer"],
            combined_refinement["restored_to_full_during_purification_fraction"],
            marker="o",
            markersize=3,
        )
        axis.set_xlabel("Decoder layer")
        axis.set_ylabel("Fraction of source W2C OFF restored to FULL")
        axis.set_ylim(0, 1)
        axis.set_title("Binary OFF redundancy removed by depth")
        figure.tight_layout()
        figure.savefig(args.output_dir / "w2c_purification_restoration_by_layer.png", dpi=180)
        plt.close(figure)

    redundancy = [groups[name]["redundant_off_removed"] for name in DATASETS]
    if any(redundancy):
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.boxplot(redundancy, tick_labels=DATASETS, showfliers=False)
        axis.set_ylabel("Binary OFF positions restored during purification")
        axis.set_title("W2C binary-route redundancy removed")
        axis.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        figure.savefig(args.output_dir / "w2c_redundancy_by_dataset.png", dpi=180)
        plt.close(figure)

    w2c_costs = [groups[name]["w2c_final_cost"] for name in DATASETS]
    if any(w2c_costs):
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.boxplot(w2c_costs, tick_labels=DATASETS, showfliers=False)
        axis.set_ylabel("Final suppression-component cost")
        axis.set_title("Corrective W2C route suppression by dataset")
        axis.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        figure.savefig(args.output_dir / "w2c_final_cost_by_dataset.png", dpi=180)
        plt.close(figure)

    table_lines = [
        "| Dataset | Samples | Source routes | Replay valid | Replay invalid | W2C routes | C2C routes | Unique routes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        counts = summaries[dataset]["counts"]
        table_lines.append(
            f"| {dataset} | {counts.get('source_samples', 0):,} | "
            f"{counts.get('source_positive_routes', 0):,} | "
            f"{counts.get('source_replay_valid_routes', 0):,} | "
            f"{counts.get('source_replay_failure_routes', 0):,} | "
            f"{counts.get('corrective_w2c_routes', 0):,} | "
            f"{counts.get('preserving_c2c_routes', 0):,} | "
            f"{counts.get('unique_valid_routes', 0):,} |"
        )
    report = [
        "# Four-Action Label Conversion Report",
        "",
        "This report keeps current-runtime source replay failures, corrective W2C labels, and preserving C2C labels separate. Every retained label is a jointly executed complete trajectory.",
        "",
        "## Authoritative source artifacts",
        "",
        f"- GQA: `{source_summary['source_details']['vqa']['predictor_manifest']}` (SHA-256 `{source_summary['source_details']['vqa']['predictor_manifest_sha256']}`).",
        f"- TextVQA: the same frozen regenerated predictor manifest, filtered to TextVQA.",
        f"- ChartQA: the same frozen regenerated predictor manifest, filtered to ChartQA.",
        f"- WeMath2.0 Standard: `{source_summary['source_details']['wemath20_standard']['cache_root']}` (contract SHA-256 `{source_summary['source_details']['wemath20_standard']['contract_sha256']}`).",
        f"- WeMath2.0 Pro: `{source_summary['source_details']['wemath2pro']['cache_root']}` (contract SHA-256 `{source_summary['source_details']['wemath2pro']['contract_sha256']}`).",
        f"- Frozen normalized inventory SHA-256: `{source_summary['source_manifest_sha256']}`.",
        "",
        "## Exact conversion totals",
        "",
        *table_lines,
        "",
        "## Corrective W2C findings",
        "",
        f"- Binary OFF positions restored during purification: {combined['w2c_redundant_off_removed']}.",
        f"- Final suppression-component cost: {combined['w2c_final_suppression_component_cost']}.",
        f"- Remaining purified-IGNORE positions by final action: {combined['remaining_w2c_anchor_action_counts']}.",
        f"- Remaining-position action fractions: {combined['remaining_w2c_anchor_action_fractions']}.",
        f"- Joint refinement diagnostics: {combined['joint_refinement']}.",
        f"- Source-to-refined answer-margin change: {combined['w2c_source_to_refined_margin_change']}.",
        f"- ALL-OFF seed selective-refinement fraction: {combined['all_off_w2c_selective_refinement_fraction']}.",
        f"- Current unified ALL-OFF correctness strata (kept separate): {combined['w2c_current_all_off_strata']}.",
        "",
        "## Preserving C2C interpretation",
        "",
        "C2C labels remain mechanical FULL/IGNORE conversions and are interpreted only as correctness-preserving efficiency/redundancy supervision, not evidence of harmful visual computation.",
        f"C2C action frequencies: {combined['c2c_raw_action_fractions']}.",
        "",
        "## Final deliverable decisions",
        "",
        "1. The exact authoritative artifacts and hashes are listed above.",
        f"2. Successfully converted binary labels: {final_decisions['successfully_converted_binary_labels']:,}; current-runtime replay failures retained as exclusions: {final_decisions['source_binary_replay_failures']:,}.",
        f"3. Unique valid four-action labels: {final_decisions['unique_valid_four_action_labels']:,}.",
        f"4. READ_ONLY/WRITE_ONLY structure beyond binary FULL/IGNORE: {final_decisions['read_write_structure']}.",
        "5. W2C and C2C remain separate supervision types because correction and correctness-preserving efficiency have different semantics.",
        f"6. Training readiness: {final_decisions['training_readiness']}.",
        f"7. Fresh four-action search: {final_decisions['fresh_four_action_search']}.",
        "",
        "## Execution and integrity",
        "",
        f"- Execution contract SHA-256: {combined['execution_contract_sha256']}.",
        f"- Sample runtime distribution: {combined['sample_runtime_seconds']}.",
        f"- Per-process peak GPU allocation distribution: {combined['peak_gpu_memory_bytes']}.",
        f"- Pilot throughput/utilization: {summaries['operational']['pilot']}.",
        f"- Full-run throughput/utilization/GPU-hours: {summaries['operational']['full']}.",
        "",
        "## Artifacts",
        "",
        "- `aggregate_statistics_v1.json`",
        "- `layer_action_frequencies_v1.csv`",
        "- `w2c_refinement_by_layer_v1.csv`",
        "- `w2c_action_fraction_by_layer.png`",
        "- `w2c_refinement_action_by_layer.png`",
        "- `w2c_purification_restoration_by_layer.png`",
        "- `w2c_redundancy_by_dataset.png`",
        "- `w2c_final_cost_by_dataset.png`",
        "",
    ]
    (args.output_dir / "four_action_label_conversion_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    for dataset in DATASETS:
        (args.output_dir / f"{dataset}_summary.md").write_text(
            f"# {dataset} Four-Action Label Summary\n\n```json\n"
            + json.dumps(summaries[dataset], indent=2, sort_keys=True)
            + "\n```\n",
            encoding="utf-8",
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
