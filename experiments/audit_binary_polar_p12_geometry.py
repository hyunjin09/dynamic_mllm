#!/usr/bin/env python3
"""P12 maximal-run geometry and complete selected-mask round-trip audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binary_policy.structured import (
    mask_to_p12_targets,
    p12_targets_to_mask,
    summarize_segment_geometry,
)
from experiments.train_binary_polar import file_sha256


BENCHMARKS = ("gqa", "textvqa", "chartqa")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    groups: dict[str, list[tuple[int, ...]]] = defaultdict(list)
    dataset_groups: dict[str, dict[str, list[tuple[int, ...]]]] = {
        benchmark: defaultdict(list) for benchmark in BENCHMARKS
    }
    input_counts = Counter()
    route_occurrences = 0
    round_trip_failures = []
    malformed = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            routes = row.get("valid_routes") or []
            if not routes:
                continue
            benchmark = row["benchmark"]
            input_counts[benchmark] += 1
            masks = [tuple(int(value) for value in route["mask"]) for route in routes]
            if any(len(mask) != 28 or any(value not in (0, 1) for value in mask) for mask in masks):
                malformed.append({"line": line_number, "uid": row["uid"]})
                continue
            if len(set(masks)) != len(masks):
                malformed.append({"line": line_number, "uid": row["uid"], "reason": "duplicate"})
                continue
            has_all_on = (1,) * 28 in masks
            minimum = min(masks, key=lambda mask: (sum(mask), mask))
            for mask in masks:
                route_occurrences += 1
                boundaries, operations = mask_to_p12_targets(mask)
                reconstructed = tuple(p12_targets_to_mask(boundaries, operations))
                if reconstructed != mask:
                    round_trip_failures.append({"uid": row["uid"], "mask": "".join(map(str, mask))})
                category = "all_on_masks" if sum(mask) == 28 else "non_all_on_masks"
                groups["all_selected_masks"].append(mask)
                groups[category].append(mask)
                dataset_groups[benchmark]["all_selected_masks"].append(mask)
                dataset_groups[benchmark][category].append(mask)
                if has_all_on:
                    groups["masks_from_all_on_valid_inputs"].append(mask)
                    dataset_groups[benchmark]["masks_from_all_on_valid_inputs"].append(mask)
            groups["minimum_on_mask_per_input"].append(minimum)
            dataset_groups[benchmark]["minimum_on_mask_per_input"].append(minimum)

    if malformed or round_trip_failures:
        raise RuntimeError(
            f"P12 geometry gate failed: malformed={len(malformed)}, round_trip={len(round_trip_failures)}"
        )
    required_groups = (
        "all_selected_masks",
        "all_on_masks",
        "non_all_on_masks",
        "minimum_on_mask_per_input",
        "masks_from_all_on_valid_inputs",
    )
    payload = {
        "schema_version": "binary_polar_p12_segment_geometry_v1",
        "passed": True,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "canonical_rule": "maximal contiguous runs of equal bits with explicit boundary[0]=1",
        "minimum_on_tie_rule": "minimum VISUAL_ON count, then lexicographically smallest complete mask",
        "positive_inputs": sum(input_counts.values()),
        "positive_inputs_by_benchmark": dict(sorted(input_counts.items())),
        "selected_route_occurrences": route_occurrences,
        "round_trip": {
            "checked_route_occurrences": route_occurrences,
            "exact_matches": route_occurrences,
            "failures": 0,
            "accuracy": 1.0,
            "ambiguous_canonicalizations": 0,
            "missing_masks": 0,
            "injectivity_argument": (
                "The deterministic inverse reconstructed every complete source mask; therefore two distinct "
                "source masks cannot share one canonical target."
            ),
        },
        "overall": {name: summarize_segment_geometry(groups[name]) for name in required_groups},
        "by_benchmark": {
            benchmark: {
                name: summarize_segment_geometry(dataset_groups[benchmark][name])
                for name in required_groups
            }
            for benchmark in BENCHMARKS
        },
        "interpretation_boundary": (
            "Geometry is an admission diagnostic only. Compressibility does not establish learned routing "
            "utility, and frequent transitions do not by themselves invalidate the lossless representation."
        ),
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P12 geometry audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
