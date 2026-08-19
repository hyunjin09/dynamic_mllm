#!/usr/bin/env python3
"""Create a deterministic, outcome-blind smoke manifest from a full manifest."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random


MIXED_MASKS = (
    tuple(index % 2 for index in range(28)),
    tuple(int(index >= 14) for index in range(28)),
    tuple(int(index % 3 != 0) for index in range(28)),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-benchmark", type=int, default=5)
    parser.add_argument("--mixed-records", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected: list[dict] = []
    rng = random.Random(args.seed)
    for benchmark in sorted({str(row["benchmark"]) for row in rows}):
        candidates = sorted((row for row in rows if str(row["benchmark"]) == benchmark), key=lambda row: row["uid"])
        if len(candidates) < args.per_benchmark:
            raise ValueError(f"{benchmark} has fewer than {args.per_benchmark} records")
        selected.extend(rng.sample(candidates, args.per_benchmark))
    selected.sort(key=lambda row: (str(row["benchmark"]), str(row["uid"])))
    mixed_rows: list[dict] = []
    for benchmark in sorted({str(row["benchmark"]) for row in selected}):
        mixed_rows.append(next(row for row in selected if str(row["benchmark"]) == benchmark))
    mixed_rows.extend(row for row in selected if row not in mixed_rows)
    for index, row in enumerate(mixed_rows[: args.mixed_records]):
        row["mixed_masks"] = [list(MIXED_MASKS[index % len(MIXED_MASKS)])]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256(output.read_bytes()).hexdigest()}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"records": len(selected), "sha256": sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
