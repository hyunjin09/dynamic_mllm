#!/usr/bin/env python3
"""Merge sharded actual-execution results without rescoring or inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evaluate_binary_polar_internal import summarize
from experiments.train_binary_polar import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    shards = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    reference = shards[0]
    for shard in shards:
        if shard["config_sha256"] != reference["config_sha256"] or shard["checkpoint"] != reference["checkpoint"]:
            raise RuntimeError("cannot merge shards with different config/checkpoint contracts")
    rows = [row for shard in shards for row in shard["rows"]]
    if len({row["uid"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate UID across execution shards")
    rows.sort(key=lambda row: row["uid"])
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite merged evaluation: {output}")
    payload = {
        "schema_version": "binary_polar_internal_execution_merged_v1",
        "mode": reference["mode"],
        "config": reference["config"],
        "config_sha256": reference["config_sha256"],
        "checkpoint": reference["checkpoint"],
        "input_shards": [{"path": path, "sha256": file_sha256(Path(path))} for path in args.inputs],
        "rows": rows,
        "summary": {
            "overall": summarize(rows),
            "by_benchmark": {
                benchmark: summarize([row for row in rows if row["benchmark"] == benchmark])
                for benchmark in ("gqa", "textvqa", "chartqa")
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
