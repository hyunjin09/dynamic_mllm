#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_filename(uid: str) -> str:
    readable = uid.replace(":", "__").replace("/", "_")
    return f"{readable}_{hashlib.sha256(uid.encode()).hexdigest()[:10]}.json"


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p25": ordered[round(0.25 * (len(ordered) - 1))],
        "median": statistics.median(ordered),
        "p75": ordered[round(0.75 * (len(ordered) - 1))],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate full four-action label cost from pilot.")
    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=Path("datasets/mcts_labels_4action/conversion_v1/pilot"),
    )
    parser.add_argument(
        "--full-manifest",
        type=Path,
        default=Path(
            "datasets/mcts_labels_4action/source_inventory_v1/source_manifest_v1.jsonl"
        ),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/4action_label_conversion/full_compute_estimate_v1.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    pilot_manifest = {
        row["uid"]: row for row in read_jsonl(args.pilot_root / "pilot_manifest_v1.jsonl")
    }
    pilot_results = []
    for uid, source in pilot_manifest.items():
        path = args.pilot_root / "records" / safe_filename(str(uid))
        if not path.is_file():
            raise RuntimeError(f"pilot is incomplete: {path}")
        result = json.loads(path.read_text())
        pilot_results.append((source, result))

    rates: dict[str, list[float]] = defaultdict(list)
    semantics_rates: dict[str, list[float]] = defaultdict(list)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for source, result in pilot_results:
        source_status = str(source["source_current_all_on_status"])
        semantics = str(result["label_semantics"])
        static_cost = max(1, int(source["estimated_conversion_cost"]))
        rate = float(result["runtime"]["elapsed_seconds"]) / static_cost
        rates[source_status].append(rate)
        semantics_rates[semantics].append(rate)
        confusion[source_status][semantics] += 1

    full_rows = read_jsonl(args.full_manifest)
    full_cost_by_status: dict[str, int] = defaultdict(int)
    full_samples_by_status: dict[str, int] = defaultdict(int)
    for row in full_rows:
        status = str(row["source_current_all_on_status"])
        full_cost_by_status[status] += int(row["estimated_conversion_cost"])
        full_samples_by_status[status] += 1

    rate_summary = {status: quantiles(values) for status, values in rates.items()}
    estimates = {}
    for label, percentile in (("low", "p25"), ("central", "median"), ("high", "p75")):
        worker_seconds = sum(
            full_cost_by_status[status] * rate_summary[status][percentile]
            for status in full_cost_by_status
        )
        ideal_wall_hours = worker_seconds / args.workers / 3600.0
        # Dynamic claims remove fixed-bin tails but cannot eliminate decoding and
        # prompt-length variability. Report both ideal and an explicit 80%
        # scheduling-efficiency envelope rather than one false-precision ETA.
        conservative_wall_hours = ideal_wall_hours / 0.80
        estimates[label] = {
            "worker_hours": worker_seconds / 3600.0,
            "ideal_wall_hours_at_16_workers": ideal_wall_hours,
            "conservative_wall_hours_at_80pct_scheduling_efficiency": conservative_wall_hours,
            "ideal_allocated_gpu_hours": ideal_wall_hours * args.gpus,
            "conservative_allocated_gpu_hours": conservative_wall_hours * args.gpus,
        }

    report = {
        "schema_version": "four_action_label_full_compute_estimate_v1",
        "method": (
            "pilot seconds per frozen static route/OFF cost, stratified by historical FULL "
            "status; p25/median/p75 rates projected to all frozen rows"
        ),
        "pilot_samples": len(pilot_results),
        "full_samples": len(full_rows),
        "workers": args.workers,
        "gpus": args.gpus,
        "pilot_rate_seconds_per_static_cost_by_source_status": rate_summary,
        "pilot_rate_seconds_per_static_cost_by_current_semantics": {
            semantics: quantiles(values) for semantics, values in semantics_rates.items()
        },
        "pilot_source_status_to_current_semantics": {
            status: dict(values) for status, values in confusion.items()
        },
        "full_samples_by_source_status": dict(full_samples_by_status),
        "full_static_cost_by_source_status": dict(full_cost_by_status),
        "estimates": estimates,
        "limitations": [
            "The pilot is deliberately stress-stratified rather than a random sample.",
            "Historical FULL status can shift under the current unified runtime.",
            "Per-sample route-cache overlap is represented only through observed pilot timing.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
