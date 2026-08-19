#!/usr/bin/env python3
"""Render the shared-prefix admission experiment as a concise paper audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--external-summary", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def pp(value: float) -> str:
    return f"{100.0 * float(value):+.2f}"


def policy_table(result: dict[str, Any]) -> list[str]:
    labels = (
        ("All-on", "all_on"),
        ("Ungated shared-prefix SW31", "ungated_hybrid"),
        ("Learned admission", "learned_admission"),
        ("Oracle admission", "oracle_admission"),
    )
    lines = [
        "| Policy | Accuracy | Delta (pp) | Routed | Harm / rescue | Mean ON | Proxy saving |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in labels:
        row = result[key]
        lines.append(
            f"| {label} | {percent(row['selected_accuracy'])} | "
            f"{pp(row['accuracy_delta'])} | {percent(row['route_fraction'])} | "
            f"{row['harm_count']} / {row['rescue_count']} | "
            f"{row['mean_visual_on_layers']:.2f} | "
            f"{percent(row['route_sensitive_layer_saving_fraction'])} |"
        )
    return lines


def make_figure(summary: dict[str, Any], path: Path) -> None:
    result = summary["external_test"]
    policies = [
        ("All-on", result["all_on"], "#555555"),
        ("Ungated", result["ungated_hybrid"], "#c44e52"),
        ("Learned", result["learned_admission"], "#007f5f"),
        ("Oracle", result["oracle_admission"], "#2f6fbb"),
    ]
    benchmarks = list(result["by_benchmark"])
    learned_delta = [
        100.0 * result["by_benchmark"][name]["learned_admission"]["accuracy_delta"]
        for name in benchmarks
    ]
    oracle_delta = [
        100.0 * result["by_benchmark"][name]["oracle_admission"]["accuracy_delta"]
        for name in benchmarks
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), constrained_layout=True)
    for name, row, color in policies:
        axes[0].scatter(
            row["mean_visual_on_layers"],
            100.0 * row["selected_accuracy"],
            s=70,
            color=color,
            label=name,
            zorder=3,
        )
        axes[0].annotate(
            name,
            (row["mean_visual_on_layers"], 100.0 * row["selected_accuracy"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    axes[0].set_xlabel("Mean visual-on layers (of 28)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("External accuracy-contextualization frontier")
    axes[0].grid(alpha=0.25)

    x = np.arange(len(benchmarks))
    width = 0.36
    axes[1].bar(x - width / 2, learned_delta, width, label="Learned", color="#007f5f")
    axes[1].bar(x + width / 2, oracle_delta, width, label="Oracle", color="#2f6fbb")
    axes[1].axhline(0.0, color="#444444", linewidth=0.8)
    axes[1].set_xticks(x, [name.upper() for name in benchmarks], rotation=20, ha="right")
    axes[1].set_ylabel("Accuracy delta vs all-on (pp)")
    axes[1].set_title("Task-shift breakdown")
    axes[1].legend(frameon=False, loc="best")
    axes[1].grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    selection = load(args.selection_summary)
    external = load(args.external_summary)
    result = external["external_test"]
    make_figure(external, args.output_figure)

    lines = [
        "# Shared Dense-Prefix Actual-Policy Admission",
        "",
        "## Question and validity contract",
        "",
        "Can an early, continuation-compatible dense prefix expose enough internal state "
        "to choose between exact all-on continuation and a frozen sparse SW31 policy?",
        "",
        "- Prefix candidates are selected only on the canonical calibration split.",
        "- The external MMStar/MMMU population is UID-disjoint and opened only after selection.",
        "- Benchmark identity is not a gate input.",
        "- Correctness is actual benchmark-scored generation, not pair accuracy.",
        "- Mean ON is a route-sensitive contextualization proxy, not wall-clock latency.",
        "- K=2 and K=4 preserve the original SW31 execution on this population; K=8 is a "
        "trajectory-changing intervention.",
        "",
        "## Prefix selection",
        "",
        f"Accuracy-first selected `K={selection['selected_accuracy_prefix_layers']}`; "
        f"efficiency-first selected `K={selection['selected_efficiency_prefix_layers']}`.",
        "",
        "| K | Hybrid accuracy | Hybrid mean ON | Oracle accuracy | Calibration accuracy mode | Calibration accuracy |",
        "|---:|---:|---:|---:|---|---:|",
    ]
    for key in sorted(selection["prefixes"], key=int):
        item = selection["prefixes"][key]
        population = item["actual_policy_full_population"]
        candidate = item["selected_accuracy_candidate"]
        lines.append(
            f"| {key} | {percent(population['ungated_hybrid']['selected_accuracy'])} | "
            f"{population['ungated_hybrid']['mean_visual_on_layers']:.2f} | "
            f"{percent(population['oracle_admission']['selected_accuracy'])} | "
            f"{candidate['score_mode']} | "
            f"{percent(candidate['accuracy_point']['selected_accuracy'])} |"
        )

    lines.extend(["", "## Locked external result", ""])
    lines.extend(policy_table(result))
    ci = result["learned_admission"].get("accuracy_delta_95_ci")
    if ci:
        lines.extend(
            [
                "",
                f"Learned admission accuracy delta bootstrap 95% CI: "
                f"`[{pp(ci[0])}, {pp(ci[1])}] pp`.",
            ]
        )
    lines.extend(
        [
            "",
            f"![Shared-prefix admission]({args.output_figure})",
            "",
            "## Benchmark breakdown",
            "",
            "| Benchmark | n | All-on | Learned | Delta (pp) | Mean ON | Oracle delta (pp) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for benchmark, values in result["by_benchmark"].items():
        learned = values["learned_admission"]
        lines.append(
            f"| {benchmark.upper()} | {learned['n']} | "
            f"{percent(values['all_on']['selected_accuracy'])} | "
            f"{percent(learned['selected_accuracy'])} | {pp(learned['accuracy_delta'])} | "
            f"{learned['mean_visual_on_layers']:.2f} | "
            f"{pp(values['oracle_admission']['accuracy_delta'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "A positive oracle point establishes route availability only. The learned method is "
            "supported only if its external paired accuracy interval satisfies the declared "
            "non-inferiority tolerance with nonzero saving. A calibration gain that reverses "
            "externally is treated as task-shift failure, not as evidence for deployment.",
            "",
            "## Artifacts",
            "",
            f"- Selection summary: `{args.selection_summary}`",
            f"- External summary: `{args.external_summary}`",
            "- Canonical UID decisions: `prefix_admission_selection_v1/canonical_predictions.jsonl`",
            "- External UID decisions: `prefix_admission_external_eval_v1/external_predictions.jsonl`",
        ]
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
