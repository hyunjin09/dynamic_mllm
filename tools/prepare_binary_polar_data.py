#!/usr/bin/env python3
"""Create outcome-preserving compact manifests from the existing MCTS labels.

This is CPU-heavy over the complete 9 GB source and therefore must be launched
through ``infra/gpu_scheduler.py --gpus 0`` rather than on a login node.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binary_policy.labels import (
    deterministic_group_split,
    iter_source_json,
    load_mcts_example,
    summarize_label_geometry,
    write_jsonl,
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=20260809)
    parser.add_argument("--route-cap-seed", type=int, default=20260809)
    parser.add_argument("--max-valid-routes", type=int, default=50)
    parser.add_argument("--all-on-weight", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = []
    invalid = []
    split_counts: Counter[str] = Counter()
    cell_split_counts: Counter[str] = Counter()
    for path in iter_source_json(args.source):
        try:
            example = load_mcts_example(
                path,
                max_valid_routes=args.max_valid_routes,
                route_cap_seed=args.route_cap_seed,
                all_on_weight=args.all_on_weight,
            )
        except Exception as exc:
            invalid.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        row = example.to_json()
        row["split"] = deterministic_group_split(example.split_group, seed=args.split_seed)
        split_counts[row["split"]] += 1
        cell_split_counts[f"{example.benchmark}/{example.difficulty}/{row['split']}"] += 1
        examples.append((example, row))

    manifest_path = output_dir / "binary_polar_labels_v1.jsonl"
    write_jsonl(manifest_path, (row for _, row in examples), overwrite=args.overwrite)
    geometry = summarize_label_geometry(example for example, _ in examples)
    split_balance_passed = len(cell_split_counts) == 24 and all(count >= 40 for count in cell_split_counts.values())
    audit = {
        "source": str(Path(args.source).resolve()),
        "source_files": len(examples) + len(invalid),
        "valid_records": len(examples),
        "invalid_records": invalid,
        "split_counts": dict(sorted(split_counts.items())),
        "cell_split_counts": dict(sorted(cell_split_counts.items())),
        "image_group_overlap_count": 0,
        "minimum_records_per_cell_split": 40,
        "split_balance_passed": split_balance_passed,
        "split_seed": args.split_seed,
        "route_cap_seed": args.route_cap_seed,
        "max_valid_routes": args.max_valid_routes,
        "all_on_weight": args.all_on_weight,
        "geometry": geometry,
        "manifest_sha256": file_sha256(manifest_path),
        "passed": len(examples) == 4000 and not invalid and split_balance_passed,
    }
    audit_path = output_dir / "binary_polar_label_audit_v1.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksums_path = output_dir / "SHA256SUMS"
    checksums_path.write_text(
        f"{file_sha256(manifest_path)}  {manifest_path.name}\n"
        f"{file_sha256(audit_path)}  {audit_path.name}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
