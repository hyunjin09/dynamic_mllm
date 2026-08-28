#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tools.research_analysis.four_action.followup import followup_thresholds, select_followups


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_once(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze population-level trajectory followups.")
    parser.add_argument(
        "--primary",
        type=Path,
        default=Path("analysis/4action_answer_alignment/primary__unified_v1/merged_results.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/4action_answer_alignment/trajectory_rescue"),
    )
    args = parser.parse_args()
    samples = read_jsonl(args.primary)
    thresholds = followup_thresholds(samples)
    rows = select_followups(samples, thresholds)
    for index, row in enumerate(rows):
        row["selection_id"] = f"trajectory_{index:06d}"
        row["shard"] = int(hashlib.sha256(row["selection_id"].encode()).hexdigest(), 16) % 8
    manifest = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode() for row in rows
    )
    summary = {
        "schema_version": "four_action_trajectory_selection_v1",
        "primary_sample_count": len(samples),
        "selection_count": len(rows),
        "unique_sample_count": len({row["uid"] for row in rows}),
        "dataset_counts": dict(Counter(row["dataset"] for row in rows)),
        "operation_counts": dict(Counter(row["culprit_operation"] for row in rows)),
        "action_counts": dict(Counter(row["suppressed_action"] for row in rows)),
        "thresholds": thresholds,
        "threshold_rule": "pooled primary absolute q90, used only for representation followup selection",
        "all_discrete_rescue_cells_included": True,
        "shard_counts": dict(Counter(str(row["shard"]) for row in rows)),
    }
    write_once(args.output_dir / "manifest.jsonl", manifest)
    write_once(
        args.output_dir / "selection_summary.json",
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
