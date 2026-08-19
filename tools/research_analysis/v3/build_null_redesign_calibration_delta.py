from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the deterministic calibration enlargement delta.")
    parser.add_argument(
        "--initial", default="data_manifests/v3_null_redesign_calibration_2000_v1.jsonl"
    )
    parser.add_argument(
        "--expanded", default="data_manifests/v3_null_redesign_calibration_4000_v2.jsonl"
    )
    parser.add_argument(
        "--output", default="data_manifests/v3_null_redesign_calibration_delta_2000_v2.jsonl"
    )
    return parser.parse_args()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute(args: argparse.Namespace) -> None:
    initial_path = Path(args.initial)
    expanded_path = Path(args.expanded)
    initial = read(initial_path)
    expanded = read(expanded_path)
    initial_ids = {row["id"] for row in initial}
    initial_images = {row["image_id"] for row in initial}
    if not initial_ids.issubset({row["id"] for row in expanded}):
        raise RuntimeError("Expanded pool does not preserve every initial record")
    delta = [row for row in expanded if row["id"] not in initial_ids]
    if len(delta) != 2000 or Counter(row["dataset"] for row in delta) != Counter({"gqa": 1000, "textvqa": 1000}):
        raise RuntimeError("Expected exactly 1,000 new records per dataset")
    if any(row["image_id"] in initial_images for row in delta):
        raise RuntimeError("Delta contains an initial-pool image")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in delta:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    meta = output.with_suffix(".manifest.json")
    meta.write_text(
        json.dumps(
            {
                "schema_version": "v3_null_redesign_calibration_delta_v1",
                "initial_sha256": sha256(initial_path),
                "expanded_sha256": sha256(expanded_path),
                "delta_sha256": sha256(output),
                "record_count": len(delta),
                "dataset_counts": dict(Counter(row["dataset"] for row in delta)),
                "image_disjoint_from_initial": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"output": str(output), "sha256": sha256(output)}, sort_keys=True))


if __name__ == "__main__":
    execute(parse_args())
