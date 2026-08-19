#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "sw31_bt_leg_s41": "SW31 BT",
    "plmax27_bt_eff0_s41": "BT eff0",
    "plmax27_bt_s31": "BT s31",
    "plmax27_dpo_r70_s41": "DPO r70",
    "plmax27_dpo_r80_s41": "DPO r80",
    "drllm_maxcorr_unique_4k": "DR-LLM",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    population = json.loads(args.summary.read_text(encoding="utf-8"))["populations"]["all"]
    baseline = population["proposer_union_oracle"]["baseline_accuracy"]

    fig, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, (name, result) in enumerate(sorted(population["single_policy"].items())):
        actual = result["actual"]
        oracle = result["oracle"]
        color = colors(index)
        x = [actual["mean_visual_on_layers"], oracle["mean_visual_on_layers"]]
        y = [100 * actual["router_accuracy"], 100 * oracle["oracle_accuracy"]]
        axis.plot(x, y, color=color, alpha=0.55, linewidth=1.2)
        axis.scatter(x[0], y[0], color=color, marker="x", s=52)
        axis.scatter(x[1], y[1], color=color, marker="o", s=46, label=LABELS[name])

    union = population["proposer_union_oracle"]
    axis.scatter(28, 100 * baseline, marker="*", s=180, color="black", label="All-on")
    axis.scatter(
        union["mean_visual_on_layers"],
        100 * union["oracle_accuracy"],
        marker="D",
        s=72,
        color="#8c2d4f",
        label="Six-route oracle",
        zorder=5,
    )
    axis.axhline(100 * baseline, color="black", linewidth=0.9, alpha=0.35)
    axis.set_xlabel("Mean visual-on layers (lower is cheaper)")
    axis.set_ylabel("Generation accuracy (%)")
    axis.set_title("Natural held-out accuracy-compute headroom")
    axis.grid(alpha=0.22)
    axis.text(
        0.02,
        0.025,
        "x: ungated proposer   circle: perfect all-on admission",
        transform=axis.transAxes,
        fontsize=9,
        color="#444444",
    )
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
