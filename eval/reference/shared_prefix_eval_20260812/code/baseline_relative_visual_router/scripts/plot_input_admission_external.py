#!/usr/bin/env python3
"""Plot the locked external accuracy-compute result for input-only admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = summary["external_test"]
    styles = {
        "all_on": ("All-on", "#333333", "o"),
        "ungated_sparse": ("Ungated SW31", "#9b59b6", "X"),
        "efficiency_first": ("Input gate: efficiency", "#d97706", "s"),
        "accuracy_first": ("Input gate: accuracy", "#2563eb", "D"),
        "oracle": ("Perfect admission", "#15803d", "*"),
    }
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for key, (label, color, marker) in styles.items():
        row = result[key]
        x = 100 * row["route_sensitive_layer_saving_fraction"]
        y = 100 * row["accuracy_delta"]
        yerr = None
        if "accuracy_delta_95_ci" in row:
            low, high = [100 * value for value in row["accuracy_delta_95_ci"]]
            yerr = [[y - low], [high - y]]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=color,
            marker=marker,
            markersize=10 if key == "oracle" else 7,
            capsize=4,
            linestyle="none",
            label=label,
        )
    ax.axhline(0, color="#666666", linewidth=1)
    ax.set_xlabel("Visual contextualization proxy saving (%)")
    ax.set_ylabel("Accuracy change vs. all-on (percentage points)")
    ax.set_title("UID-disjoint external tasks: MMStar and MMMU")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
