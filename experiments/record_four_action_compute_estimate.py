#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the gated unified pilot compute estimate.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("analysis/4action_answer_alignment/pilot__unified_v1/stage_summary.json"),
    )
    parser.add_argument(
        "--experiment-log",
        type=Path,
        default=Path("analysis/4action_answer_alignment/experiment_log.md"),
    )
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if not summary.get("passed"):
        raise RuntimeError("refusing to record or authorize full-run compute from a failed pilot")
    estimate = summary.get("compute_estimate")
    timing = summary.get("timing_seconds_per_sample")
    if not estimate or not timing:
        raise RuntimeError("pilot summary contains no throughput/compute estimate")
    marker = "## Unified pilot throughput estimate"
    existing = args.experiment_log.read_text(encoding="utf-8")
    if marker in existing:
        raise FileExistsError("unified pilot estimate is already recorded")
    block = (
        f"\n{marker}\n\n"
        f"- Mean seconds/sample: `{timing['mean']:.6f}`\n"
        f"- Median seconds/sample: `{timing['median']:.6f}`\n"
        f"- Estimated primary GPU-hours: `{estimate['primary_gpu_hours']:.3f}`\n"
        f"- Estimated primary wall-hours at 8 workers: "
        f"`{estimate['primary_wall_hours_at_eight_workers']:.3f}`\n"
        f"- Estimated all-controls GPU-hours: `{estimate['all_controls_gpu_hours']:.3f}`\n"
        "- Basis: gated 56-example unified-materialized, all-28-layer pilot; "
        "includes old-binary semantic checks omitted from production, so the "
        "primary/control estimates are conservative upper estimates.\n"
    )
    with args.experiment_log.open("a", encoding="utf-8") as handle:
        handle.write(block)
    print(json.dumps({"timing": timing, "compute_estimate": estimate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
