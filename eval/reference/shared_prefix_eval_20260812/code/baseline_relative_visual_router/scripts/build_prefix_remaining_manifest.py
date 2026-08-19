#!/usr/bin/env python3
"""Build an exact UID-aligned manifest for unfinished shared-prefix outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

from baseline_relative_visual_router.input_features import align_manifest_policy_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--baseline-rows-jsonl", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-layers", default="2,4,8")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cached_uids(path: Path) -> set[str]:
    result: set[str] = set()
    for part in sorted(path.glob("prefix_*_shard_*_part_*.pt")):
        rows = torch.load(part, map_location="cpu", weights_only=False)["rows"]
        current = {str(row["uid"]) for row in rows}
        if result & current:
            raise RuntimeError(f"duplicate cached UIDs under {path}")
        result.update(current)
    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    prefixes = [int(value) for value in args.prefix_layers.split(",")]
    seen = {
        prefix: cached_uids(args.cache_root / f"prefix_{prefix:02d}")
        for prefix in prefixes
    }
    reference = seen[prefixes[0]]
    if any(values != reference for values in seen.values()):
        raise RuntimeError(
            f"prefix caches are not synchronized: "
            f"{ {prefix: len(values) for prefix, values in seen.items()} }"
        )
    aligned = align_manifest_policy_rows(
        read_jsonl(args.manifest_jsonl), read_jsonl(args.baseline_rows_jsonl)
    )
    missing = [pair for pair in aligned if str(pair[0]["uid"]) not in reference]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "samples.jsonl", [pair[0] for pair in missing])
    write_jsonl(args.output_dir / "baseline_rows.jsonl", [pair[1] for pair in missing])
    summary = {
        "schema_version": "shared_prefix_remaining_manifest_v1",
        "original_count": len(aligned),
        "cached_count": len(reference),
        "remaining_count": len(missing),
        "prefix_layers": prefixes,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
