#!/usr/bin/env python3
"""Exhaustively decompose the frozen Phase-69 full-benchmark result."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dense_failure_stage2.full_benchmark_exhaustive_audit import (
    ACTION_NAMES,
    NONFULL_ACTIONS,
    answer_change_record,
    classify_bottleneck,
    safe_rate,
    summarize_action_behavior,
    summarize_funnel,
    validate_audit_rows,
)


FAMILY_ORDER = ("chartqa", "textvqa", "mmmu_pro", "pope")
EXPECTED_FAMILY_COUNTS = {"chartqa": 2500, "textvqa": 5000, "mmmu_pro": 3460, "pope": 9000}
EXPECTED_CONTRACT = "63379eeff80fd5b046cb980cdfccea7fab232e15eb5b14924a24b60393327e83"
DEFAULT_INPUT = ROOT / "analysis/dense_failure_stage2/full_benchmark_eval/paired/all_paired.jsonl"
DEFAULT_OUTPUT = ROOT / "analysis/dense_failure_stage2/full_benchmark_exhaustive_audit"
DEFAULT_P90 = ROOT / "analysis/dense_failure_stage1/robust_operating_points_and_compatibility/thresholds/named_operating_points.csv"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_text(
        path,
        "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stats(values: Sequence[int | float]) -> dict[str, int | float | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def family_rows(rows: Sequence[Mapping[str, Any]], family: str) -> list[Mapping[str, Any]]:
    return list(rows) if family == "overall" else [row for row in rows if row["benchmark_family"] == family]


def markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    output = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    output.extend("| " + " | ".join(cell(row.get(field)) for field in fields) + " |" for row in rows)
    return "\n".join(output)


def load_p90_reference(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(row for row in csv.DictReader(handle) if row["operating_point"] == "P90")
    return {
        "threshold": float(row["threshold"]),
        "historical_c_false_admission": 1.0 - float(row["historical_c_preservation"]),
        "canonical_c_false_admission": 1.0 - float(row["canonical_c_preservation"]),
        "pooled_w_recall": float(row["pooled_w_recall"]),
        "historical_w_recall": float(row["historical_w_recall"]),
        "canonical_w_recall": float(row["canonical_w_recall"]),
    }


def build_tables(rows: list[dict[str, Any]], p90: Mapping[str, float]) -> dict[str, list[dict[str, Any]]]:
    funnels: dict[str, dict[str, Any]] = {}
    funnel_rows: list[dict[str, Any]] = []
    action_behavior: list[dict[str, Any]] = []
    stage1: list[dict[str, Any]] = []
    stage2: list[dict[str, Any]] = []
    treatment: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []
    score_distributions: list[dict[str, Any]] = []

    for family in (*FAMILY_ORDER, "overall"):
        selected = family_rows(rows, family)
        funnel = summarize_funnel(selected)
        funnels[family] = funnel
        funnel_rows.append({"family": family, **funnel})
        stage1.append(
            {
                "family": family,
                "dense_w": funnel["dense_w"],
                "triggered_w": funnel["triggered_w"],
                "w_recall": funnel["p_trigger_given_w"],
                "dense_c": funnel["dense_c"],
                "triggered_c": funnel["triggered_c"],
                "c_false_admission_rate": funnel["p_trigger_given_c"],
                "reference_p90_pooled_w_recall": p90["pooled_w_recall"],
                "delta_vs_reference_pooled_w_recall": (
                    funnel["p_trigger_given_w"] - p90["pooled_w_recall"]
                    if funnel["p_trigger_given_w"] is not None
                    else None
                ),
                "reference_p90_historical_c_false_admission": p90["historical_c_false_admission"],
                "reference_p90_canonical_c_false_admission": p90["canonical_c_false_admission"],
            }
        )
        treatment.append(
            {
                "family": family,
                "nonfull_triggered_w": funnel["triggered_w_nonfull"],
                "w_to_c": funnel["w_to_c"],
                "p_w_to_c_given_nonfull_triggered_w": funnel["p_w_to_c_given_trigger_nonfull_w"],
                "nonfull_triggered_c": funnel["triggered_c_nonfull"],
                "c_to_w": funnel["c_to_w"],
                "p_c_to_w_given_nonfull_triggered_c": funnel["p_c_to_w_given_trigger_nonfull_c"],
                "net": funnel["net"],
            }
        )
        label = classify_bottleneck(funnel)
        rationale = {
            "INACTIVE": "Stage-1 admitted no C or W rows, so Stage-2 was never called.",
            "TREATMENT_QUALITY_LIMITED": "At least 20 triggered-W rows received non-FULL but none was rescued.",
            "PRESERVATION_LIMITED": "C→W is at least twice W→C and determines the negative net.",
            "INTERVENTION_LIMITED": "Fewer than 10% of triggered-W rows received non-FULL.",
            "ADMISSION_LIMITED": "Fewer than 5% of Dense-W rows reached Stage-1 trigger.",
            "MIXED": "No single fixed descriptive rule dominates.",
        }[label]
        classifications.append({"family": family, "classification": label, "quantitative_justification": rationale, **funnel})

        for dense_correct in (False, True):
            behavior = summarize_action_behavior(selected, dense_correct=dense_correct)
            action_behavior.append({"family": family, **behavior})
            triggered = [row for row in selected if row["triggered"] and bool(row["dense_correct"]) is dense_correct]
            nonfull = [row for row in triggered if row["any_non_full"]]
            first_layers = [int(row["first_non_full_layer"]) for row in nonfull]
            delays = [int(row["trigger_to_first_non_full_delay"]) for row in nonfull]
            counts = [int(row["non_full_count"]) for row in triggered]
            first_stats, delay_stats, count_stats = stats(first_layers), stats(delays), stats(counts)
            stage2.append(
                {
                    "family": family,
                    "dense_state": behavior["dense_state"],
                    "triggered": len(triggered),
                    "any_nonfull": len(nonfull),
                    "p_nonfull_given_trigger": safe_rate(len(nonfull), len(triggered)),
                    "mean_nonfull_count_per_trigger": count_stats["mean"],
                    "mean_first_nonfull_layer": first_stats["mean"],
                    "median_first_nonfull_layer": first_stats["median"],
                    "mean_delay": delay_stats["mean"],
                    "median_delay": delay_stats["median"],
                }
            )
            timing.append(
                {
                    "family": family,
                    "dense_state": behavior["dense_state"],
                    **{f"first_layer_{key}": value for key, value in first_stats.items()},
                    **{f"delay_{key}": value for key, value in delay_stats.items()},
                    **{f"nonfull_count_{key}": value for key, value in count_stats.items()},
                }
            )

        for state_name, predicate in (
            ("W", lambda row: not bool(row["dense_correct"])),
            ("C", lambda row: bool(row["dense_correct"])),
            ("ALL", lambda row: True),
        ):
            group = [row for row in selected if predicate(row)]
            values = [float(row["stage1_max_score"]) for row in group]
            summary = stats(values)
            score_distributions.append(
                {
                    "family": family,
                    "dense_state": state_name,
                    **summary,
                    "p95": float(np.quantile(values, 0.95)) if values else None,
                    "threshold": p90["threshold"],
                    "max_score_to_threshold": p90["threshold"] - max(values) if values else None,
                }
            )

    action_distribution: list[dict[str, Any]] = []
    for family in (*FAMILY_ORDER, "overall"):
        selected = family_rows(rows, family)
        for dense_correct, state in ((False, "W"), (True, "C")):
            triggered = [row for row in selected if row["triggered"] and bool(row["dense_correct"]) is dense_correct]
            for action in ACTION_NAMES:
                sample_count = sum(action in row["actions"] for row in triggered) if action != "FULL" else sum(not row["any_non_full"] for row in triggered)
                occurrence_count = sum(row["actions"].count(action) for row in triggered)
                first_count = sum(
                    row["first_non_full_layer"] is not None
                    and row["actions"][int(row["first_non_full_layer"])] == action
                    for row in triggered
                ) if action != "FULL" else sum(not row["any_non_full"] for row in triggered)
                action_distribution.append(
                    {
                        "family": family,
                        "dense_state": state,
                        "action": action,
                        "triggered_samples": len(triggered),
                        "sample_use_count": sample_count,
                        "sample_use_rate": safe_rate(sample_count, len(triggered)),
                        "occurrence_count": occurrence_count,
                        "first_action_count": first_count,
                    }
                )

    dataset_summary: list[dict[str, Any]] = []
    for benchmark in sorted({str(row["benchmark"]) for row in rows}):
        selected = [row for row in rows if row["benchmark"] == benchmark]
        dataset_summary.append({"benchmark": benchmark, "family": selected[0]["benchmark_family"], **summarize_funnel(selected)})

    return {
        "benchmark_funnel": funnel_rows[:-1],
        "pooled_funnel": [funnel_rows[-1]],
        "action_behavior": action_behavior,
        "stage1_admission": stage1,
        "stage2_intervention": stage2,
        "treatment_quality": treatment,
        "action_distribution": action_distribution,
        "trigger_action_timing": timing,
        "bottleneck_classification": classifications,
        "stage1_score_distribution": score_distributions,
        "dataset_bottleneck_summary": dataset_summary,
    }


def build_change_summaries(changes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for transition in ("W→C", "C→W"):
        selected = [row for row in changes if row["transition"] == transition]
        for metric in ("trigger_score", "stage1_max_score", "trigger_layer", "first_non_full_layer", "trigger_to_first_non_full_delay", "non_full_count"):
            output.append({"transition": transition, "summary_type": "metric", "item": metric, **stats([row[metric] for row in selected])})
        first = Counter(str(row["first_non_full_action"]) for row in selected)
        used = Counter(action["action"] for row in selected for action in row["non_full_actions"])
        sample_used = Counter(action for row in selected for action in {item["action"] for item in row["non_full_actions"]})
        for action in NONFULL_ACTIONS:
            output.append(
                {
                    "transition": transition,
                    "summary_type": "action",
                    "item": action,
                    "n": len(selected),
                    "first_count": first[action],
                    "sample_use_count": sample_used[action],
                    "occurrence_count": used[action],
                }
            )
    return output


def make_figures(root: Path, tables: Mapping[str, list[dict[str, Any]]], changes: Sequence[Mapping[str, Any]]) -> None:
    figure_root = root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    families = list(FAMILY_ORDER)
    funnels = {row["family"]: row for row in tables["benchmark_funnel"]}

    x = np.arange(len(families))
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.22
    ax.bar(x - width, [funnels[f]["p_trigger_given_w"] for f in families], width, label="P(trigger|W)")
    ax.bar(x, [funnels[f]["p_nonfull_given_trigger_w"] or 0 for f in families], width, label="P(nonFULL|trigger,W)")
    ax.bar(x + width, [funnels[f]["p_w_to_c_given_trigger_nonfull_w"] or 0 for f in families], width, label="P(W→C|trigger,W,nonFULL)")
    ax.set_xticks(x, families)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("Dense-W pipeline funnel by benchmark")
    fig.tight_layout(); fig.savefig(figure_root / "funnel_by_benchmark.png", dpi=180); plt.close(fig)

    behavior = {(row["family"], row["dense_state"]): row for row in tables["action_behavior"]}
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, [safe_rate(behavior[(f, "W")]["any_nonfull"], behavior[(f, "W")]["triggered"]) or 0 for f in families], width, label="Triggered W")
    ax.bar(x + width / 2, [safe_rate(behavior[(f, "C")]["any_nonfull"], behavior[(f, "C")]["triggered"]) or 0 for f in families], width, label="Triggered C")
    ax.set_xticks(x, families); ax.set_ylabel("P(any non-FULL | trigger)"); ax.set_ylim(0, 0.4); ax.legend()
    ax.set_title("Stage-2 activation is not strongly W-selective")
    fig.tight_layout(); fig.savefig(figure_root / "trigger_vs_nonfull_by_correctness.png", dpi=180); plt.close(fig)

    trans = ("W→C", "C→W")
    first_counts = {name: Counter(row["first_non_full_action"] for row in changes if row["transition"] == name) for name in trans}
    fig, ax = plt.subplots(figsize=(7, 5))
    bottom = np.zeros(2)
    for action in NONFULL_ACTIONS:
        values = np.asarray([first_counts[name][action] for name in trans])
        ax.bar(trans, values, bottom=bottom, label=action); bottom += values
    ax.set_ylabel("Samples"); ax.set_title("First non-FULL action among answer changes"); ax.legend()
    fig.tight_layout(); fig.savefig(figure_root / "rescue_vs_regression_actions.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([[row["trigger_to_first_non_full_delay"] for row in changes if row["transition"] == name] for name in trans], tick_labels=trans)
    ax.set_ylabel("Trigger → first non-FULL delay (layers)"); ax.set_title("Answer-change intervention timing")
    fig.tight_layout(); fig.savefig(figure_root / "rescue_vs_regression_timing.png", dpi=180); plt.close(fig)

    text_regressions = [row for row in changes if row["benchmark_family"] == "textvqa" and row["transition"] == "C→W"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist([row["first_non_full_layer"] for row in text_regressions], bins=np.arange(13.5, 28.5, 1), color="#d95f02")
    axes[0].set_xlabel("First non-FULL layer"); axes[0].set_ylabel("Regressions")
    counts = Counter(row["first_non_full_action"] for row in text_regressions)
    axes[1].bar(list(NONFULL_ACTIONS), [counts[a] for a in NONFULL_ACTIONS], color="#7570b3")
    axes[1].tick_params(axis="x", rotation=25); axes[1].set_ylabel("Regressions")
    fig.suptitle("TextVQA C→W regression breakdown"); fig.tight_layout(); fig.savefig(figure_root / "textvqa_regression_breakdown.png", dpi=180); plt.close(fig)

    matrix = np.asarray([[funnels[f]["p_trigger_given_w"] or 0, funnels[f]["p_nonfull_given_trigger_w"] or 0, funnels[f]["p_w_to_c_given_trigger_nonfull_w"] or 0, funnels[f]["p_c_to_w_given_trigger_nonfull_c"] or 0] for f in families])
    fig, ax = plt.subplots(figsize=(9, 4.5)); image = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=max(0.35, matrix.max()))
    ax.set_yticks(range(len(families)), families); ax.set_xticks(range(4), ["W admission", "W intervention", "W treatment success", "C preservation risk"], rotation=20, ha="right")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]): ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", color="white" if matrix[i,j] > 0.17 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="Rate"); ax.set_title("Full-benchmark bottleneck map")
    fig.tight_layout(); fig.savefig(figure_root / "benchmark_bottleneck_map.png", dpi=180); plt.close(fig)


def benchmark_audit_text(family: str, funnel: Mapping[str, Any], behavior: Mapping[tuple[str, str], Mapping[str, Any]], changes: Sequence[Mapping[str, Any]], score_rows: Sequence[Mapping[str, Any]]) -> str:
    local = [row for row in changes if row["benchmark_family"] == family]
    w, c = behavior[(family, "W")], behavior[(family, "C")]
    lines = [
        f"# {family.replace('_', ' ').title()} exhaustive audit",
        "",
        markdown_table([{"Dense-W": funnel["dense_w"], "Triggered-W": funnel["triggered_w"], "W non-FULL": funnel["triggered_w_nonfull"], "W→C": funnel["w_to_c"], "Dense-C": funnel["dense_c"], "Triggered-C": funnel["triggered_c"], "C non-FULL": funnel["triggered_c_nonfull"], "C→W": funnel["c_to_w"]}], ["Dense-W", "Triggered-W", "W non-FULL", "W→C", "Dense-C", "Triggered-C", "C non-FULL", "C→W"]),
        "",
        f"Stage-1 W admission is {funnel['p_trigger_given_w'] or 0:.4f}; C false admission is {funnel['p_trigger_given_c'] or 0:.4f}. Stage-2 uses non-FULL on {funnel['p_nonfull_given_trigger_w'] or 0:.4f} of triggered W and {funnel['p_nonfull_given_trigger_c'] or 0:.4f} of triggered C.",
        "",
        f"Conditional treatment success is {funnel['p_w_to_c_given_trigger_nonfull_w'] or 0:.4f}; conditional preservation risk is {funnel['p_c_to_w_given_trigger_nonfull_c'] or 0:.4f}. First non-FULL counts are W: WRITE_ONLY={w['first_write_only']}, IGNORE={w['first_ignore']}, READ_ONLY={w['first_read_only']}; C: WRITE_ONLY={c['first_write_only']}, IGNORE={c['first_ignore']}, READ_ONLY={c['first_read_only']}.",
        "",
        f"All {len(local)} correctness-changing trajectories are preserved in `../answer_changes/all_22_answer_changes.jsonl`; the descriptions are associations, not per-action causal attributions.",
    ]
    if family == "textvqa":
        lines.extend(["", "TextVQA dominates the negative net because 255 Dense-C rows are admitted versus 186 Dense-W rows, 57 triggered-C rows receive non-FULL versus 26 triggered-W rows, and those interventions produce 12 regressions versus 2 rescues. Both exposure and conditional outcome are unfavorable: C regression risk is 12/57 while W rescue success is 2/26."])
    elif family == "mmmu_pro":
        lines.extend(["", "MMMU-Pro is treatment-quality limited under the fixed rule: Stage-1 admits 261 W rows and Stage-2 intervenes on 82 of them, yet none is rescued. The four regressions occur on 26 non-FULL-treated C rows. This does not imply MMMU-Pro can never benefit."])
    elif family == "pope":
        score = next(row for row in score_rows if row["family"] == "pope" and row["dense_state"] == "ALL")
        lines.extend(["", f"POPE is inactive, not successfully preserved: its maximum Stage-1 score is {score['max']:.6f}, still {score['max_score_to_threshold']:.6f} below the strict P90 threshold {score['threshold']:.6f}. Stage-2 was never called, so this audit provides no evidence about Stage-2 treatment quality on POPE."])
    elif family == "chartqa":
        lines.extend(["", "ChartQA has one rescue and three regressions. Although only 20 triggered rows receive non-FULL, preservation loss still exceeds rescue, and C regression risk conditional on non-FULL is 3/9 versus W rescue success 1/11."])
    return "\n".join(lines)


def write_reports(root: Path, rows: list[dict[str, Any]], tables: Mapping[str, list[dict[str, Any]]], changes: list[dict[str, Any]], p90: Mapping[str, float], source_path: Path) -> None:
    funnels = {row["family"]: row for row in [*tables["benchmark_funnel"], *tables["pooled_funnel"]]}
    behavior = {(row["family"], row["dense_state"]): row for row in tables["action_behavior"]}
    regressions = [row for row in changes if row["transition"] == "C→W"]
    rescues = [row for row in changes if row["transition"] == "W→C"]
    class_counts = Counter(label for row in regressions for label in row["regression_classes"])
    reg_first = Counter(row["first_non_full_action"] for row in regressions)
    rescue_first = Counter(row["first_non_full_action"] for row in rescues)
    pooled = funnels["overall"]

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()
    protocol = f"""# Full-benchmark exhaustive audit protocol

- Analysis type: deterministic offline reconstruction; no model inference, training, search, or threshold change.
- Frozen source: `{source_path.relative_to(ROOT)}` ({file_sha256(source_path)}).
- Phase-69 contract: `{EXPECTED_CONTRACT}`.
- Population: exactly 19,960 unique paired UIDs: ChartQA 2,500; TextVQA 5,000; MMMU-Pro 3,460; POPE 9,000.
- Frozen gate: five-head mean, strict score > P90 threshold `{p90['threshold']}`.
- Git HEAD at audit execution: `{git_head}`; worktree was dirty and preserved. Status SHA-256: `{sha256(git_status.encode()).hexdigest()}`.
- Immediate means first non-FULL at the trigger layer (delay 0); delayed means delay > 0.
- Regression labels overlap: R1 one non-FULL; R2 more than one; R3 delay 0; R4 delay > 0; R5 any READ_ONLY; R6 any WRITE_ONLY; R7 IGNORE count exceeds all READ_ONLY+WRITE_ONLY occurrences; R8 more than one non-FULL action type.
- Fixed bottleneck rule order: INACTIVE when no triggers; TREATMENT_QUALITY_LIMITED when >=20 W interventions yield zero rescue; PRESERVATION_LIMITED when C→W >= 2×max(1,W→C); INTERVENTION_LIMITED when P(nonFULL|trigger,W)<0.10; ADMISSION_LIMITED when P(trigger|W)<0.05; otherwise MIXED.
- These are descriptive accounting rules. An answer change following a non-FULL trace does not causally identify one action as the cause.
"""
    write_text(root / "protocol.md", protocol)

    primary_rows = [{"Family": f, "P(trigger|W)": funnels[f]["p_trigger_given_w"], "P(trigger|C)": funnels[f]["p_trigger_given_c"], "P(nonFULL|trig,W)": funnels[f]["p_nonfull_given_trigger_w"], "P(nonFULL|trig,C)": funnels[f]["p_nonfull_given_trigger_c"], "W→C": funnels[f]["w_to_c"], "C→W": funnels[f]["c_to_w"], "Net": funnels[f]["net"]} for f in (*FAMILY_ORDER, "overall")]
    summary = f"""# Exhaustive full-benchmark regression/rescue audit

The frozen result is primarily **preservation-limited in pooled accounting**, with benchmark-specific treatment failure and inactivity. Stage-1 admits 496/4,380 Dense-W and 405/15,580 Dense-C. Stage-2 then uses at least one non-FULL action on 119/496 triggered W and 92/405 triggered C—similar conditional activation rates of {pooled['p_nonfull_given_trigger_w']:.4f} and {pooled['p_nonfull_given_trigger_c']:.4f}. The decisive asymmetry is outcome: only 3/119 W interventions rescue, while 19/92 C interventions regress.

## Primary funnel

{markdown_table(primary_rows, ['Family','P(trigger|W)','P(trigger|C)','P(nonFULL|trig,W)','P(nonFULL|trig,C)','W→C','C→W','Net'])}

## Answers to the plan questions

1. Dense-W reaching Stage-1 trigger: ChartQA 49/353, TextVQA 186/712, MMMU-Pro 261/2,235, POPE 0/1,080; pooled 496/4,380.
2. Non-FULL among triggered W: ChartQA 11/49, TextVQA 26/186, MMMU-Pro 82/261, POPE 0/0; pooled 119/496.
3. Non-FULL among triggered C: ChartQA 9/58, TextVQA 57/255, MMMU-Pro 26/92, POPE 0/0; pooled 92/405.
4. Conditional W→C among non-FULL triggered W is 3/119 ({pooled['p_w_to_c_given_trigger_nonfull_w']:.4f}) pooled.
5. Conditional C→W among non-FULL triggered C is 19/92 ({pooled['p_c_to_w_given_trigger_nonfull_c']:.4f}) pooled.
6. The 19 regressions have first action WRITE_ONLY={reg_first['WRITE_ONLY']}, IGNORE={reg_first['IGNORE']}, READ_ONLY={reg_first['READ_ONLY']}; {class_counts['R1_SINGLE_INTERVENTION']} are single-intervention, {class_counts['R2_MULTIPLE_INTERVENTIONS']} multiple, {class_counts['R3_IMMEDIATE_INTERVENTION']} immediate, and {class_counts['R4_DELAYED_INTERVENTION']} delayed. WRITE_ONLY occurs in {class_counts['R6_WRITE_RELATED']} regression traces and IGNORE dominates {class_counts['R7_IGNORE_DOMINATED']}; no regression uses READ_ONLY. These patterns are exhaustive associations, not causal isolation.
7. The 3 rescues start with WRITE_ONLY={rescue_first['WRITE_ONLY']} and IGNORE={rescue_first['IGNORE']}; all are delayed by 2–5 layers. Two are exact one-non-FULL trajectories, while the other uses WRITE_ONLY at three consecutive layers.
8. TextVQA dominates because Stage-2 acts on 57 triggered C but only 26 triggered W, then regresses 12/57 C versus rescuing 2/26 W. It contributes -10 of the -16 net.
9. MMMU-Pro is not merely admission-limited: 261 W trigger and 82 receive non-FULL, but 0 are rescued; 4/26 treated C regress. This is treatment-quality limited under the fixed rule.
10. POPE is completely inactive because no score crosses P90. Its maximum score is {next(row['max'] for row in tables['stage1_score_distribution'] if row['family']=='pope' and row['dense_state']=='ALL'):.6f}, {next(row['max_score_to_threshold'] for row in tables['stage1_score_distribution'] if row['family']=='pope' and row['dense_state']=='ALL'):.6f} below threshold; Stage-2 performance on POPE is unobserved.
11. Pooled failure is preservation-limited by the fixed accounting rule, with a mixed family picture: ChartQA/TextVQA preservation-limited, MMMU-Pro treatment-quality-limited, and POPE inactive.
12. TextVQA contributes most: 12/19 regressions and -10/−16 net correction.
13. This audit does not prove any individual action caused a transition, justify lowering Stage-1 threshold, show POPE preservation, show dynamic routing is impossible, or validate a replacement method.

## Development-calibration context

The P90 development expectation was pooled W recall {p90['pooled_w_recall']:.4f}; external pooled W admission is {pooled['p_trigger_given_w']:.4f}. This calibration shift is real but is not the main current net-loss accounting: among W already admitted and treated, success is only 2.52%, while C treatment risk is 20.65%. Expanding admission without improving treatment/preservation is therefore not supported.
"""
    write_text(root / "summaries/exhaustive_audit_summary.md", summary)

    recommendation = """# One next-improvement recommendation

## Recommendation

Test a **preservation-calibrated Stage-2 abstention rule** on frozen Stage-2 logits, allowing a non-FULL action only when its confidence advantage over FULL exceeds one prospectively selected margin. Keep Stage 1, P90, the Stage-2 checkpoint, action meanings, and evaluator fixed.

## Why it matters

The pooled method loses 16 net answers because 19 correct answers regress while only 3 wrong answers are rescued. Stage-2 activation itself is not W-selective: 24.0% of triggered W and 22.7% of triggered C receive non-FULL.

## Observed failure pattern

TextVQA supplies 12 regressions versus 2 rescues; MMMU-Pro supplies 4 regressions and no rescue. Across the full population, W treatment succeeds for 3/119 non-FULL cases while C treatment regresses 19/92. This makes harmful action admission—not simply low Stage-1 W recall—the largest direct contributor to net loss.

## Separately authorized test

On a development-only population, freeze one margin from Stage-2's existing FULL-versus-best-non-FULL scores under a prospective C-preservation constraint, then evaluate that single frozen margin on the untouched full-benchmark traces or a new paired run as technically required. Do not select the margin on these 22 answer changes.

## Interpretation

- Positive: substantially fewer C→W while retaining enough W→C to make net correction nonnegative would support action-level abstention as the next local improvement.
- Negative: if regressions and rescues cannot be separated by existing Stage-2 confidence, the present checkpoint lacks useful deployment selectivity and action/timing supervision becomes the next diagnosis.
- Not justified by one negative result: abandoning four-action routing, changing Stage 1, or claiming corrective interventions cannot work.

This is a recommendation only; no calibration or evaluation is executed in Phase 70.
"""
    write_text(root / "summaries/next_improvement_recommendation.md", recommendation)

    for family, filename in (("chartqa", "chartqa_audit.md"), ("textvqa", "textvqa_audit.md"), ("mmmu_pro", "mmmu_pro_audit.md"), ("pope", "pope_audit.md")):
        write_text(root / "benchmarks" / filename, benchmark_audit_text(family, funnels[family], behavior, changes, tables["stage1_score_distribution"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--p90-reference", type=Path, default=DEFAULT_P90)
    args = parser.parse_args()
    source = args.input if args.input.is_absolute() else ROOT / args.input
    output = args.output if args.output.is_absolute() else ROOT / args.output
    p90_path = args.p90_reference if args.p90_reference.is_absolute() else ROOT / args.p90_reference
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit root: {output}")

    rows = read_jsonl(source)
    counts = validate_audit_rows(rows, expected_contract=EXPECTED_CONTRACT)
    if len(rows) != 19960 or counts != EXPECTED_FAMILY_COUNTS:
        raise ValueError(f"frozen population mismatch: n={len(rows)} counts={counts}")
    transition_counts = Counter(str(row["transition"]) for row in rows)
    if transition_counts["W→C"] != 3 or transition_counts["C→W"] != 19:
        raise ValueError(f"frozen answer-change count mismatch: {dict(transition_counts)}")
    p90 = load_p90_reference(p90_path)
    if not math.isclose(p90["threshold"], float(rows[0]["stage1_threshold"]), abs_tol=1e-15):
        raise ValueError("P90 reference differs from paired-row threshold")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    tables = build_tables(rows, p90)
    changes = [answer_change_record(row) for row in rows if row["transition"] in {"W→C", "C→W"}]
    for row in changes:
        row["single_route_like"] = row["non_full_count"] == 1

    write_csv(stage / "funnel/benchmark_funnel.csv", tables["benchmark_funnel"])
    write_csv(stage / "funnel/pooled_funnel.csv", tables["pooled_funnel"])
    write_csv(stage / "funnel/triggered_c_vs_w_action_behavior.csv", tables["action_behavior"])
    write_csv(stage / "funnel/bottleneck_classification.csv", tables["bottleneck_classification"])
    write_jsonl(stage / "answer_changes/all_22_answer_changes.jsonl", changes)
    write_jsonl(stage / "answer_changes/rescues_3.jsonl", [row for row in changes if row["transition"] == "W→C"])
    write_jsonl(stage / "answer_changes/regressions_19.jsonl", [row for row in changes if row["transition"] == "C→W"])
    write_csv(stage / "answer_changes/rescue_vs_regression_summary.csv", build_change_summaries(changes))
    for name in ("stage1_admission", "stage2_intervention", "treatment_quality", "action_distribution", "trigger_action_timing", "dataset_bottleneck_summary"):
        write_csv(stage / "metrics" / f"{name}.csv", tables[name])
    write_csv(stage / "metrics/stage1_score_distribution.csv", tables["stage1_score_distribution"])
    make_figures(stage, tables, changes)
    write_reports(stage, rows, tables, changes, p90, source)

    files = sorted(path for path in stage.rglob("*") if path.is_file())
    manifest = {
        "schema_version": "full_benchmark_exhaustive_audit_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source.relative_to(ROOT)),
        "source_sha256": file_sha256(source),
        "phase69_contract_sha256": EXPECTED_CONTRACT,
        "population_n": len(rows),
        "family_counts": counts,
        "transition_counts": dict(transition_counts),
        "files": [{"path": str(path.relative_to(stage)), "bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in files],
    }
    write_json(stage / "artifact_manifest.json", manifest)
    for item in manifest["files"]:
        if file_sha256(stage / item["path"]) != item["sha256"]:
            raise ValueError(f"artifact hash verification failed: {item['path']}")
    os.replace(stage, output)
    print(json.dumps({"output": str(output), "population": len(rows), "transitions": dict(transition_counts), "artifact_files": len(manifest["files"]), "no_new_inference": True}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
