#!/usr/bin/env python3
"""Create a no-pooling comparison of BCE and NLL external analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.train_binary_polar import file_sha256
from four_action_policy.external import ACTIVE_BENCHMARKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bce-analysis", type=Path, required=True)
    parser.add_argument("--nll-analysis", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    bce = json.loads(args.bce_analysis.read_text(encoding="utf-8"))
    nll = json.loads(args.nll_analysis.read_text(encoding="utf-8"))
    if bce.get("records") != 14960 or nll.get("records") != 14960:
        raise RuntimeError("both objective analyses must cover all 14,960 records")
    order = list(ACTIVE_BENCHMARKS) + ["mmmu_pro", "pope"]
    lines = [
        "# Four-Action POLAR: BCE versus NLL",
        "",
        "Both checkpoints were selected independently on the same internal validation split before external evaluation. Metrics are shown per benchmark/suite; no cross-suite overall score is computed.",
        "",
        "| Population | BCE predicted correct | BCE Δ vs unified FULL | NLL predicted correct | NLL Δ vs unified FULL |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in order:
        left = bce["groups"][group]
        right = nll["groups"][group]
        if left["records"] != right["records"]:
            raise RuntimeError(f"objective populations differ for {group}")
        lines.append(
            f"| {group} | {left['predicted_correct_rate']:.4f} | "
            f"{left['correctness_delta']:+.4f} | {right['predicted_correct_rate']:.4f} | "
            f"{right['correctness_delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "This table is descriptive. Any claim that one loss is better should use the paired per-sample outputs and image-cluster bootstrap, not only the point estimates above.",
            "",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    args.report.with_suffix(args.report.suffix + ".sha256").write_text(
        f"{file_sha256(args.report)}  {args.report.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
