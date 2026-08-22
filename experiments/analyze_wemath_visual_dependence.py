#!/usr/bin/env python3
"""Reanalyze frozen WeMath routes after separating ALL-OFF from positive vision.

The script is label-only.  It verifies every exact ALL-OFF anchor directly in
the authoritative raw MCTS records, then analyzes the checksum-bound raw-route
statistics.  It never loads Qwen, executes a route, or reads capped predictor
supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments import analyze_wemath_visual_compute_difficulty as prior


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "outputs/wemath2pro_mcts_label_analysis_v1"
PRIOR_OUTPUT = PROJECT / "outputs/wemath2pro_visual_compute_difficulty_v1"
OUTPUT = PROJECT / "outputs/wemath2pro_visual_dependence_reanalysis_v1"
DIFFICULTIES = prior.DIFFICULTIES
DEGREE = prior.DEGREE
TRANSITIONS = prior.TRANSITIONS
BOOTSTRAP_DRAWS = 5_000
BOOTSTRAP_SEED = 20260823
BUDGETS = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28)
REPORT_BUDGETS = {4, 8, 12, 16, 20, 24, 28}
ALL_OFF_KEY = "0" * 28
ALL_ON_KEY = "1" * 28


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def grouped_rows(rows: Sequence[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    return prior.grouped_rows(rows)


def verify_exact_anchors(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Verify exact ALL-OFF/ALL-ON records without recomputing model outcomes."""
    source_manifest = json.loads((SOURCE / "analysis_manifest.json").read_text(encoding="utf-8"))
    index_path = SOURCE / "cache_record_index_v1.jsonl"
    expected_index_hash = source_manifest["output_hashes_before_manifest"][
        str(index_path.relative_to(PROJECT))
    ]
    actual_index_hash = sha256_file(index_path)
    if actual_index_hash != expected_index_hash:
        raise RuntimeError("authoritative raw-record index checksum mismatch")
    index = read_jsonl(index_path)
    by_uid = {row["uid"]: row for row in rows}
    if len(index) != len(rows) or {item["uid"] for item in index} != set(by_uid):
        raise RuntimeError("authoritative raw-record index UID mismatch")

    exact_off_correct = 0
    exact_on_correct = 0
    anchor_rollup = hashlib.sha256()
    for item in index:
        path = PROJECT / item["record_path"]
        # The raw cache is large. Read each record once so the direct anchor
        # audit and checksum verification share the same immutable bytes.
        record_bytes = path.read_bytes()
        actual_hash = hashlib.sha256(record_bytes).hexdigest()
        if actual_hash != item["record_sha256"]:
            raise RuntimeError(f"raw record checksum mismatch: {item['uid']}")
        anchor_rollup.update(f"{item['uid']}:{actual_hash}\n".encode())
        record = json.loads(record_bytes)
        candidates = record.get("candidate_executions", [])
        off = [route for route in candidates if route.get("mask_key") == ALL_OFF_KEY]
        on = [route for route in candidates if route.get("mask_key") == ALL_ON_KEY]
        if len(off) != 1 or len(on) != 1:
            raise RuntimeError(f"missing or duplicate exact anchor: {item['uid']}")
        for name, route in (("ALL-OFF", off[0]), ("ALL-ON", on[0])):
            mask = route.get("visual_on_mask")
            if not isinstance(mask, list) or len(mask) != 28:
                raise RuntimeError(f"malformed {name} mask: {item['uid']}")
            expected_bit = 0 if name == "ALL-OFF" else 1
            if any(int(value) != expected_bit for value in mask):
                raise RuntimeError(f"nonexact {name} anchor: {item['uid']}")
            expected_correct = float(route["score"]) >= float(route["correctness_threshold"])
            if bool(route["result_correct"]) != expected_correct:
                raise RuntimeError(f"anchor scorer mismatch: {item['uid']} {name}")
        derived = by_uid[item["uid"]]
        off_correct = bool(off[0]["result_correct"])
        on_correct = bool(on[0]["result_correct"])
        if off_correct != bool(derived["all_off_correct"]):
            raise RuntimeError(f"ALL-OFF raw/index mismatch: {item['uid']}")
        if on_correct != (derived["current_all_on_status"] == "correct"):
            raise RuntimeError(f"ALL-ON raw/index mismatch: {item['uid']}")
        exact_off_correct += off_correct
        exact_on_correct += on_correct
    return {
        "status": "PASS",
        "raw_records_verified": len(index),
        "exact_all_off_anchors": len(index),
        "exact_all_on_anchors": len(index),
        "exact_all_off_correct": exact_off_correct,
        "exact_all_on_correct": exact_on_correct,
        "anchor_mismatches": 0,
        "raw_record_index_sha256": actual_index_hash,
        "raw_record_hash_rollup_sha256": anchor_rollup.hexdigest(),
    }


def classify_visual_regimes(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        full_correct = row["current_all_on_status"] == "correct"
        off_correct = bool(row["all_off_correct"])
        if full_correct:
            row["visual_regime"] = "V0" if off_correct else "V+"
            counts[row["visual_regime"]] += 1
            if (float(row["raw_min_on"]) == 0.0) != off_correct:
                raise RuntimeError(f"V0/minimum-ON identity mismatch: {row['uid']}")
            if not off_correct and not 1 <= int(row["raw_min_on"]) <= 28:
                raise RuntimeError(f"V+ positive minimum-ON invariant failed: {row['uid']}")
        else:
            if off_correct:
                row["correction_regime"] = "A0"
            elif int(row["raw_valid_routes"]) > 0:
                row["correction_regime"] = "A+"
            else:
                row["correction_regime"] = "no_correction"
            counts[row["correction_regime"]] += 1
            if off_correct and int(row["raw_min_on"]) != 0:
                raise RuntimeError(f"A0/minimum-ON identity mismatch: {row['uid']}")
            if row["correction_regime"] == "A+" and not 1 <= int(row["raw_min_on"]) <= 28:
                raise RuntimeError(f"A+ positive minimum-ON invariant failed: {row['uid']}")
    return dict(sorted(counts.items()))


def v0_vplus_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    full_correct = [row for row in rows if row["current_all_on_status"] == "correct"]
    output = []
    for index, (group_type, group, members) in enumerate(grouped_rows(full_correct)):
        v0 = [row for row in members if row["visual_regime"] == "V0"]
        vplus = [row for row in members if row["visual_regime"] == "V+"]
        indicators = [float(row["visual_regime"] == "V0") for row in members]
        low, high = prior.cluster_bootstrap_mean_ci(
            indicators,
            [row["question_id"] for row in members],
            draws=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED + index,
        )
        output.append({
            "group_type": group_type,
            "group": group,
            "eligible_n": sum(
                row["difficulty"] == group if group_type == "difficulty"
                else row["difficulty_degree"] == int(group)
                for row in rows
            ),
            "full_correct_n": len(members),
            "v0_n": len(v0),
            "v0_fraction": len(v0) / len(members),
            "v0_fraction_ci_low": low,
            "v0_fraction_ci_high": high,
            "vplus_n": len(vplus),
            "vplus_fraction": len(vplus) / len(members),
        })
    return output


def decomposition_row(
    *, group_type: str, group: str, values: Sequence[float], vplus_values: Sequence[float]
) -> dict[str, Any]:
    original_mean = float(np.mean(values))
    vplus_fraction = len(vplus_values) / len(values)
    conditional_mean = float(np.mean(vplus_values)) if vplus_values else None
    reconstructed = vplus_fraction * conditional_mean if conditional_mean is not None else 0.0
    return {
        "group_type": group_type,
        "group": group,
        "full_correct_n": len(values),
        "original_mean_min_on": original_mean,
        "v0_fraction": 1.0 - vplus_fraction,
        "vplus_fraction": vplus_fraction,
        "vplus_mean_min_positive_on": conditional_mean,
        "reconstructed_original_mean": reconstructed,
        "reconstruction_abs_error": abs(original_mean - reconstructed),
    }


def mean_decomposition(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    full_correct = [row for row in rows if row["current_all_on_status"] == "correct"]
    output = []
    for group_type, group, members in grouped_rows(full_correct):
        vplus = [row for row in members if row["visual_regime"] == "V+"]
        output.append(decomposition_row(
            group_type=group_type,
            group=group,
            values=[float(row["raw_min_on"]) for row in members],
            vplus_values=[float(row["raw_min_on"]) for row in vplus],
        ))
    if max(row["reconstruction_abs_error"] for row in output) > 1e-12:
        raise RuntimeError("mean minimum-ON decomposition identity failed")

    # Symmetric two-factor decomposition relative to base/degree 0:
    # delta(p*m) = delta(p)*average(m) + delta(m)*average(p).
    for group_type in ("difficulty", "degree"):
        members = [row for row in output if row["group_type"] == group_type]
        reference_group = "base" if group_type == "difficulty" else "0"
        reference = next(row for row in members if row["group"] == reference_group)
        p0 = reference["vplus_fraction"]
        m0 = reference["vplus_mean_min_positive_on"]
        for row in members:
            p1 = row["vplus_fraction"]
            m1 = row["vplus_mean_min_positive_on"]
            observed = row["original_mean_min_on"] - reference["original_mean_min_on"]
            composition = (p1 - p0) * (m1 + m0) / 2.0
            conditional = (m1 - m0) * (p1 + p0) / 2.0
            row.update({
                "reference_group": reference_group,
                "observed_mean_difference_vs_reference": observed,
                "v0_composition_component": composition,
                "vplus_conditional_budget_component": conditional,
                "component_sum_error": abs(observed - composition - conditional),
                "composition_share_of_observed": (
                    composition / observed if abs(observed) > 1e-12 else None
                ),
                "conditional_share_of_observed": (
                    conditional / observed if abs(observed) > 1e-12 else None
                ),
            })
    if max(row["component_sum_error"] for row in output) > 1e-12:
        raise RuntimeError("two-factor mean decomposition identity failed")
    return output


def summarize_positive_min_on(members: Sequence[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    values = [float(row["raw_min_on"]) for row in members]
    result = prior.distribution_summary(values)
    if not values:
        return result
    low, high = prior.cluster_bootstrap_mean_ci(
        values,
        [row["question_id"] for row in members],
        draws=BOOTSTRAP_DRAWS,
        seed=seed,
    )
    cheaper = [float(value < 28) for value in values]
    removable = [28.0 - value for value in values]
    result.update({
        "mean_ci_low": low,
        "mean_ci_high": high,
        "cheaper_positive_route_n": int(sum(cheaper)),
        "cheaper_positive_route_fraction": float(np.mean(cheaper)),
        "mean_removable_direct_visual_layers": float(np.mean(removable)),
        "median_removable_direct_visual_layers": float(np.median(removable)),
    })
    return result


def vplus_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    vplus = [row for row in rows if row.get("visual_regime") == "V+"]
    output = []
    for index, (group_type, group, members) in enumerate(grouped_rows(vplus)):
        output.append({
            "group_type": group_type,
            "group": group,
            **summarize_positive_min_on(members, seed=BOOTSTRAP_SEED + 100 + index),
        })
    return output


def vplus_feasibility(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    vplus = [row for row in rows if row.get("visual_regime") == "V+"]
    output = []
    for index, (group_type, group, members) in enumerate(grouped_rows(vplus)):
        values = [int(row["raw_min_on"]) for row in members]
        clusters = [row["question_id"] for row in members]
        for budget in BUDGETS:
            indicators = [float(value <= budget) for value in values]
            low, high = prior.cluster_bootstrap_mean_ci(
                indicators,
                clusters,
                draws=BOOTSTRAP_DRAWS,
                seed=BOOTSTRAP_SEED + 300 + index * 100 + budget,
            )
            output.append({
                "group_type": group_type,
                "group": group,
                "budget_visual_on": budget,
                "n": len(values),
                "feasible_n": int(sum(indicators)),
                "feasible_fraction": float(np.mean(indicators)),
                "ci_low": low,
                "ci_high": high,
                "in_compact_report_table": budget in REPORT_BUDGETS,
            })
    return output


def trend_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    cohorts = (
        ("all_full_correct", [row for row in rows if row["current_all_on_status"] == "correct"]),
        ("vplus_only", [row for row in rows if row.get("visual_regime") == "V+"]),
    )
    output = []
    for index, (cohort, members) in enumerate(cohorts):
        degrees = [float(row["difficulty_degree"]) for row in members]
        values = [float(row["raw_min_on"]) for row in members]
        rho = prior.spearman(degrees, values)
        low, high = prior.cluster_bootstrap_spearman_ci(
            degrees,
            values,
            [row["question_id"] for row in members],
            draws=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED + 2_000 + index,
        )
        output.append({
            "cohort": cohort,
            "n": len(members),
            "spearman_rho": rho,
            "cluster_bootstrap_ci_low": low,
            "cluster_bootstrap_ci_high": high,
            "degree_means": {
                str(degree): float(np.mean([
                    row["raw_min_on"] for row in members if row["difficulty_degree"] == degree
                ])) for degree in range(4)
            },
            "degree_medians": {
                str(degree): float(np.median([
                    row["raw_min_on"] for row in members if row["difficulty_degree"] == degree
                ])) for degree in range(4)
            },
        })
    return output


def paired_analysis(
    rows: Sequence[dict[str, Any]], *, cohort: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if cohort == "all_full_correct":
        selected_rows = [row for row in rows if row["current_all_on_status"] == "correct"]
    elif cohort == "vplus_only":
        selected_rows = [row for row in rows if row.get("visual_regime") == "V+"]
    else:
        raise ValueError(cohort)
    by_family: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in selected_rows:
        by_family[row["question_id"]][row["difficulty"]].append(row)
    output = []
    all_pairs = []
    for transition_index, (source, target) in enumerate(TRANSITIONS):
        pairs = []
        for family, labels in by_family.items():
            if len(labels.get(source, [])) != 1 or len(labels.get(target, [])) != 1:
                continue
            left, right = labels[source][0], labels[target][0]
            pairs.append({
                "family": family,
                "delta": float(right["raw_min_on"] - left["raw_min_on"]),
                "same_image": left["image_group_id"] == right["image_group_id"],
            })
        all_pairs.extend({"transition": f"{source}->{target}", **pair} for pair in pairs)
        for scope, scoped in (
            ("same_family", pairs),
            ("same_family_same_image", [pair for pair in pairs if pair["same_image"]]),
        ):
            deltas = [pair["delta"] for pair in scoped]
            if deltas:
                low, high = prior.cluster_bootstrap_mean_ci(
                    deltas,
                    [pair["family"] for pair in scoped],
                    draws=BOOTSTRAP_DRAWS,
                    seed=BOOTSTRAP_SEED + 3_000 + transition_index * 10 + (scope != "same_family")
                    + (1_000 if cohort == "vplus_only" else 0),
                )
            else:
                low = high = None
            output.append({
                "cohort": cohort,
                "scope": scope,
                "transition": f"{source}->{target}",
                "source_degree": DEGREE[source],
                "target_degree": DEGREE[target],
                "paired_families": len(scoped),
                "mean_delta": float(np.mean(deltas)) if deltas else None,
                "median_delta": float(np.median(deltas)) if deltas else None,
                "increase_n": sum(delta > 0 for delta in deltas),
                "increase_fraction": sum(delta > 0 for delta in deltas) / len(deltas) if deltas else None,
                "equal_n": sum(delta == 0 for delta in deltas),
                "equal_fraction": sum(delta == 0 for delta in deltas) / len(deltas) if deltas else None,
                "decrease_n": sum(delta < 0 for delta in deltas),
                "decrease_fraction": sum(delta < 0 for delta in deltas) / len(deltas) if deltas else None,
                "mean_delta_ci_low": low,
                "mean_delta_ci_high": high,
            })
    deltas = [pair["delta"] for pair in all_pairs]
    if deltas:
        low, high = prior.cluster_bootstrap_mean_ci(
            deltas,
            [pair["family"] for pair in all_pairs],
            draws=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED + 5_999 + (1_000 if cohort == "vplus_only" else 0),
        )
    else:
        low = high = None
    aggregate = {
        "cohort": cohort,
        "scope": "all_supported_transitions",
        "transition": "aggregate",
        "usable_families": len({pair["family"] for pair in all_pairs}),
        "paired_transition_occurrences": len(all_pairs),
        "mean_delta": float(np.mean(deltas)) if deltas else None,
        "median_delta": float(np.median(deltas)) if deltas else None,
        "increase_fraction": float(np.mean([delta > 0 for delta in deltas])) if deltas else None,
        "equal_fraction": float(np.mean([delta == 0 for delta in deltas])) if deltas else None,
        "decrease_fraction": float(np.mean([delta < 0 for delta in deltas])) if deltas else None,
        "mean_delta_ci_low": low,
        "mean_delta_ci_high": high,
    }
    output.append(aggregate)
    return output, aggregate


def contingency_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for group_type, group, members in grouped_rows(rows):
        categories = Counter(
            ("FULL_correct" if row["current_all_on_status"] == "correct" else "FULL_wrong")
            + "__"
            + ("ALLOFF_correct" if row["all_off_correct"] else "ALLOFF_wrong")
            for row in members
        )
        payload: dict[str, Any] = {
            "group_type": group_type,
            "group": group,
            "eligible_n": len(members),
        }
        for category in (
            "FULL_correct__ALLOFF_correct",
            "FULL_correct__ALLOFF_wrong",
            "FULL_wrong__ALLOFF_correct",
            "FULL_wrong__ALLOFF_wrong",
        ):
            payload[f"{category}_n"] = categories[category]
            payload[f"{category}_fraction"] = categories[category] / len(members)
        output.append(payload)
    return output


def group_a_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    full_wrong = [row for row in rows if row["current_all_on_status"] == "wrong"]
    output = []
    for index, (group_type, group, members) in enumerate(grouped_rows(full_wrong)):
        a0 = [row for row in members if row["correction_regime"] == "A0"]
        aplus = [row for row in members if row["correction_regime"] == "A+"]
        none = [row for row in members if row["correction_regime"] == "no_correction"]
        summary = summarize_positive_min_on(aplus, seed=BOOTSTRAP_SEED + 7_000 + index)
        output.append({
            "group_type": group_type,
            "group": group,
            "full_wrong_n": len(members),
            "a0_n": len(a0),
            "a0_fraction": len(a0) / len(members),
            "aplus_n": len(aplus),
            "aplus_fraction": len(aplus) / len(members),
            "no_correction_n": len(none),
            "no_correction_fraction": len(none) / len(members),
            "aplus_mean_min_positive_on": summary["mean"],
            "aplus_median_min_positive_on": summary["median"],
            "aplus_q25": summary["q25"],
            "aplus_q75": summary["q75"],
            "aplus_mean_ci_low": summary.get("mean_ci_low"),
            "aplus_mean_ci_high": summary.get("mean_ci_high"),
        })
    return output


def axis_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    full_correct = [row for row in rows if row["current_all_on_status"] == "correct"]
    output = []
    for axis in "xyz":
        for present in (False, True):
            members = [row for row in full_correct if (axis in row["difficulty"]) == present]
            vplus = [row for row in members if row["visual_regime"] == "V+"]
            output.append({
                "axis": axis,
                "axis_present": present,
                "full_correct_n": len(members),
                "v0_fraction": sum(row["visual_regime"] == "V0" for row in members) / len(members),
                "vplus_n": len(vplus),
                "vplus_mean_min_positive_on": float(np.mean([row["raw_min_on"] for row in vplus])),
                "vplus_median_min_positive_on": float(np.median([row["raw_min_on"] for row in vplus])),
            })
    return output


def _frame(title: str, body: str, *, width: int = 920, height: int = 540) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>'
        f'{body}</svg>\n'
    )


def svg_bars(path: Path, labels: Sequence[str], values: Sequence[float], title: str, y_label: str) -> None:
    width, height, left, right, top, bottom = 920, 540, 80, 30, 55, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = max(values) * 1.15 if max(values) else 1.0
    body = []
    for tick in np.linspace(0, maximum, 6):
        y = top + plot_h * (1 - tick / maximum)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ddd"/>')
        body.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick:.2f}</text>')
    for index, (label, value) in enumerate(zip(labels, values)):
        cell = plot_w / len(labels)
        x = left + index * cell + cell * 0.18
        bar_w = cell * 0.64
        y = top + plot_h * (1 - value / maximum)
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top+plot_h-y:.1f}" fill="#3182bd"/>')
        body.append(f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.2f}</text>')
        body.append(f'<text x="{x+bar_w/2:.1f}" y="{height-45}" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(label)}</text>')
    body.append(f'<text transform="translate(20,{top+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(y_label)}</text>')
    path.write_text(_frame(title, "".join(body), width=width, height=height), encoding="utf-8")


def svg_stacked(path: Path, labels: Sequence[str], series: Sequence[tuple[str, Sequence[float], str]], title: str) -> None:
    width, height, left, right, top, bottom = 920, 540, 80, 30, 55, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    body = []
    for tick in np.linspace(0, 1, 6):
        y = top + plot_h * (1 - tick)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ddd"/>')
        body.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick:.1f}</text>')
    for index, label in enumerate(labels):
        cell = plot_w / len(labels)
        x, bar_w = left + index * cell + cell * 0.18, cell * 0.64
        cumulative = 0.0
        for _, values, color in series:
            value = values[index]
            y = top + plot_h * (1 - cumulative - value)
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{plot_h*value:.1f}" fill="{color}"/>')
            cumulative += value
        body.append(f'<text x="{x+bar_w/2:.1f}" y="{height-45}" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(label)}</text>')
    for index, (name, _, color) in enumerate(series):
        y = 58 + index * 18
        body.append(f'<rect x="{width-180}" y="{y-10}" width="14" height="14" fill="{color}"/>')
        body.append(f'<text x="{width-160}" y="{y+2}" font-family="sans-serif" font-size="11">{html.escape(name)}</text>')
    path.write_text(_frame(title, "".join(body), width=width, height=height), encoding="utf-8")


def svg_grouped_means(
    path: Path, labels: Sequence[str], first: Sequence[float], second: Sequence[float], title: str
) -> None:
    width, height, left, right, top, bottom = 920, 540, 80, 30, 55, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = 28.0
    body = []
    for tick in range(0, 29, 4):
        y = top + plot_h * (1 - tick / maximum)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ddd"/>')
        body.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick}</text>')
    for index, label in enumerate(labels):
        cell = plot_w / len(labels)
        for offset, value, color in ((0.18, first[index], "#9ecae1"), (0.51, second[index], "#de2d26")):
            x, bar_w = left + index * cell + cell * offset, cell * 0.28
            y = top + plot_h * (1 - value / maximum)
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top+plot_h-y:.1f}" fill="{color}"/>')
        body.append(f'<text x="{left+(index+0.5)*cell:.1f}" y="{height-45}" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(label)}</text>')
    body.append(f'<text x="{width-185}" y="65" font-family="sans-serif" font-size="11" fill="#3182bd">all FULL-correct</text>')
    body.append(f'<text x="{width-185}" y="82" font-family="sans-serif" font-size="11" fill="#de2d26">V+ only</text>')
    path.write_text(_frame(title, "".join(body), width=width, height=height), encoding="utf-8")


def svg_degree_trends(path: Path, trends: Sequence[dict[str, Any]]) -> None:
    width, height, left, right, top, bottom = 920, 540, 80, 30, 55, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    body = []
    for tick in range(0, 17, 2):
        y = top + plot_h * (1 - tick / 16)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ddd"/>')
    for row, color in zip(trends, ("#3182bd", "#de2d26")):
        points = []
        for degree in range(4):
            x = left + plot_w * degree / 3
            y = top + plot_h * (1 - row["degree_means"][str(degree)] / 16)
            points.append(f"{x:.1f},{y:.1f}")
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
    for degree in range(4):
        x = left + plot_w * degree / 3
        body.append(f'<text x="{x:.1f}" y="{height-45}" text-anchor="middle" font-family="sans-serif" font-size="12">degree {degree}</text>')
    body.append(f'<text x="{width-220}" y="65" font-family="sans-serif" font-size="11" fill="#3182bd">all FULL-correct</text>')
    body.append(f'<text x="{width-220}" y="82" font-family="sans-serif" font-size="11" fill="#de2d26">V+ only</text>')
    path.write_text(_frame("Original versus V+-conditional difficulty trend", "".join(body), width=width, height=height), encoding="utf-8")


def svg_paired(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    selected = [row for row in rows if row["cohort"] == "vplus_only" and row["scope"] == "same_family" and row["transition"] != "aggregate"]
    labels = [row["transition"] for row in selected]
    values = [row["mean_delta"] or 0.0 for row in selected]
    width, height, left, right, top, bottom = 1080, 560, 80, 30, 55, 120
    plot_w, plot_h = width-left-right, height-top-bottom
    limit = max(1.0, max(abs(value) for value in values) * 1.25)
    zero = top + plot_h / 2
    body = [f'<line x1="{left}" y1="{zero:.1f}" x2="{width-right}" y2="{zero:.1f}" stroke="#222"/>']
    for index, (label, value) in enumerate(zip(labels, values)):
        cell = plot_w / len(labels)
        x, bar_w = left + index*cell + cell*0.2, cell*0.6
        y_value = zero - value / (2*limit) * plot_h
        body.append(f'<rect x="{x:.1f}" y="{min(zero,y_value):.1f}" width="{bar_w:.1f}" height="{abs(zero-y_value):.1f}" fill="{"#de2d26" if value < 0 else "#3182bd"}"/>')
        body.append(f'<text transform="translate({x+bar_w/2:.1f},{height-45}) rotate(-45)" text-anchor="end" font-family="sans-serif" font-size="10">{html.escape(label)}</text>')
    path.write_text(_frame("V+ same-family transition deltas", "".join(body), width=width, height=height), encoding="utf-8")


def create_figures(
    rows: Sequence[dict[str, Any]],
    v0vplus: Sequence[dict[str, Any]],
    decomposition: Sequence[dict[str, Any]],
    feasibility: Sequence[dict[str, Any]],
    contingency: Sequence[dict[str, Any]],
    trends: Sequence[dict[str, Any]],
    paired: Sequence[dict[str, Any]],
) -> list[Path]:
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [figure_dir / name for name in (
        "01_all_off_rate_by_stratum.svg",
        "02_full_correct_v0_vplus_composition.svg",
        "03_original_vs_vplus_min_on_by_stratum.svg",
        "04_vplus_min_on_by_degree.svg",
        "05_vplus_budget_feasibility_by_degree.svg",
        "06_vplus_budget_feasibility_by_stratum.svg",
        "07_original_vs_conditional_difficulty_trend.svg",
        "08_full_vs_alloff_2x2_by_stratum.svg",
        "09_paired_vplus_transition_deltas.svg",
    )]
    stratum_regime = [row for row in v0vplus if row["group_type"] == "difficulty"]
    labels = [row["group"] for row in stratum_regime]
    v0_values = [row["v0_fraction"] for row in stratum_regime]
    vplus_values = [row["vplus_fraction"] for row in stratum_regime]
    svg_bars(paths[0], labels, v0_values, "ALL-OFF correctness among FULL-correct", "P(V0 | FULL correct)")
    svg_stacked(paths[1], labels, (("V0", v0_values, "#9ecae1"), ("V+", vplus_values, "#de2d26")), "FULL-correct V0/V+ composition")
    stratum_decomp = [row for row in decomposition if row["group_type"] == "difficulty"]
    svg_grouped_means(
        paths[2],
        [row["group"] for row in stratum_decomp],
        [row["original_mean_min_on"] for row in stratum_decomp],
        [row["vplus_mean_min_positive_on"] for row in stratum_decomp],
        "Original versus V+-conditional minimum ON",
    )
    vplus = [row for row in rows if row.get("visual_regime") == "V+"]
    prior.svg_boxplot(
        paths[3],
        {f"degree {degree}": [row["raw_min_on"] for row in vplus if row["difficulty_degree"] == degree] for degree in range(4)},
        "V+ minimum discovered positive visual-access depth",
        "minimum positive VISUAL_ON layers",
    )
    prior.svg_feasibility(paths[4], feasibility, group_type="degree", title="V+ visual-budget feasibility by degree")
    prior.svg_feasibility(paths[5], feasibility, group_type="difficulty", title="V+ visual-budget feasibility by stratum")
    svg_degree_trends(paths[6], trends)
    cont = [row for row in contingency if row["group_type"] == "difficulty"]
    category_keys = (
        ("FULL+/OFF+", "FULL_correct__ALLOFF_correct_fraction", "#9ecae1"),
        ("FULL+/OFF-", "FULL_correct__ALLOFF_wrong_fraction", "#de2d26"),
        ("FULL-/OFF+", "FULL_wrong__ALLOFF_correct_fraction", "#74c476"),
        ("FULL-/OFF-", "FULL_wrong__ALLOFF_wrong_fraction", "#bdbdbd"),
    )
    svg_stacked(paths[7], [row["group"] for row in cont], tuple(
        (name, [row[key] for row in cont], color) for name, key, color in category_keys
    ), "FULL versus ALL-OFF outcomes")
    svg_paired(paths[8], paired)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", choices=("PENDING", "A", "B", "C", "D", "E"), default="PENDING")
    parser.add_argument("--outcome-rationale", default="Interpretation pending aggregate review.")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    rows, source_integrity = prior.load_and_validate()
    anchor_audit = verify_exact_anchors(rows)
    regime_counts = classify_visual_regimes(rows)
    if regime_counts.get("V0", 0) + regime_counts.get("V+", 0) != 841:
        raise RuntimeError("FULL-correct V0/V+ cohort does not total 841")

    v0vplus = v0_vplus_rows(rows)
    decomposition = mean_decomposition(rows)
    vplus = vplus_summary(rows)
    feasibility = vplus_feasibility(rows)
    trends = trend_rows(rows)
    old_paired, old_aggregate = paired_analysis(rows, cohort="all_full_correct")
    conditional_paired, conditional_aggregate = paired_analysis(rows, cohort="vplus_only")
    prior_summary = json.loads((PRIOR_OUTPUT / "analysis_summary.json").read_text(encoding="utf-8"))
    expected_old = prior_summary["paired_family_aggregate"]
    if (
        old_aggregate["paired_transition_occurrences"] != expected_old["paired_transition_occurrences"]
        or abs(old_aggregate["mean_delta"] - expected_old["mean_paired_delta"]) > 1e-12
        or abs(old_aggregate["median_delta"] - expected_old["median_paired_delta"]) > 1e-12
    ):
        raise RuntimeError("old paired-analysis audit did not reproduce")
    paired = [*old_paired, *conditional_paired]
    contingency = contingency_rows(rows)
    group_a = group_a_rows(rows)
    axes = axis_rows(rows)

    paths = {
        "v0vplus": OUTPUT / "full_correct_v0_vplus_by_difficulty.csv",
        "decomposition": OUTPUT / "mean_min_on_decomposition.csv",
        "vplus": OUTPUT / "vplus_min_on_by_difficulty.csv",
        "feasibility": OUTPUT / "vplus_budget_feasibility.csv",
        "contingency": OUTPUT / "full_alloff_contingency.csv",
        "paired": OUTPUT / "paired_vplus_transitions.csv",
        "group_a": OUTPUT / "group_a0_aplus_by_difficulty.csv",
        "axes": OUTPUT / "axis_summary.csv",
        "trends": OUTPUT / "trend_comparison.json",
        "anchor_audit": OUTPUT / "exact_anchor_audit.json",
    }
    for key, payload in (
        ("v0vplus", v0vplus),
        ("decomposition", decomposition),
        ("vplus", vplus),
        ("feasibility", feasibility),
        ("contingency", contingency),
        ("paired", paired),
        ("group_a", group_a),
        ("axes", axes),
    ):
        prior.write_csv(paths[key], payload)
    write_json(paths["trends"], trends)
    write_json(paths["anchor_audit"], anchor_audit)
    figures = create_figures(rows, v0vplus, decomposition, feasibility, contingency, trends, paired)

    summary = {
        "schema_version": "wemath2pro_visual_dependence_reanalysis_v1",
        "status": "PASS",
        "source_integrity": source_integrity,
        "exact_anchor_audit": anchor_audit,
        "population": {
            "eligible": len(rows),
            "full_correct": 841,
            "full_wrong": len(rows) - 841,
            "regime_counts": regime_counts,
        },
        "full_correct_v0_vplus_by_degree": {
            row["group"]: row for row in v0vplus if row["group_type"] == "degree"
        },
        "mean_decomposition_by_degree": {
            row["group"]: row for row in decomposition if row["group_type"] == "degree"
        },
        "vplus_min_on_by_degree": {
            row["group"]: row for row in vplus if row["group_type"] == "degree"
        },
        "trend_comparison": trends,
        "old_paired_audit": {"status": "PASS", **old_aggregate},
        "vplus_paired_aggregate": conditional_aggregate,
        "axis_summary": axes,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "cluster": "official question_id seed family",
        },
        "claim_boundaries": [
            "ALL-OFF removes direct decoder access to encoded visual K/V but may retain structural side channels.",
            "V+ minimum positive ON is search-conditioned within the tested binary route space.",
            "The analysis cannot assess REPEAT, recurrence, or more than 28 visual executions.",
        ],
        "outcome": args.outcome,
        "outcome_rationale": args.outcome_rationale,
    }
    summary_path = OUTPUT / "analysis_summary.json"
    write_json(summary_path, summary)
    output_paths = [*paths.values(), *figures, summary_path]
    manifest = {
        "schema_version": "wemath2pro_visual_dependence_reanalysis_manifest_v1",
        "status": "PASS",
        "source_policy": "all authoritative raw records checked for exact anchors; all route geometry is uncapped raw-derived",
        "source_integrity": source_integrity,
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "output_hashes": {
            str(path.relative_to(PROJECT)): sha256_file(path) for path in output_paths
        },
    }
    write_json(OUTPUT / "analysis_manifest.json", manifest)
    print(json.dumps({
        "status": "PASS",
        "records": len(rows),
        "full_correct": 841,
        "regime_counts": regime_counts,
        "outcome": args.outcome,
        "output": str(OUTPUT.relative_to(PROJECT)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
