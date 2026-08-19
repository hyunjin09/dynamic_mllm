#!/usr/bin/env python3
"""Wait for both one-GPU external evaluations, then run frozen aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    args = parser.parse_args()
    question = args.root / "question"
    image_question = args.root / "image_question"
    required = [
        question / "shard_000_of_001/metadata.json",
        image_question / "shard_000_of_001/metadata.json",
    ]
    deadline = time.monotonic() + args.timeout_hours * 3600
    while not all(path.is_file() for path in required):
        if time.monotonic() >= deadline:
            missing = [str(path) for path in required if not path.is_file()]
            raise TimeoutError(f"external evaluation did not complete before timeout: {missing}")
        time.sleep(60)
    contract = json.loads((args.root / "evaluation_contract_v1.json").read_text(encoding="utf-8"))
    clusters = contract["pope_image_overlap"]["cluster_keys"]
    if len(clusters) != 1:
        raise RuntimeError("expected exactly one frozen POPE overlap cluster")
    command = [
        ".venv/bin/python",
        "experiments/merge_binary_polar_external_eval.py",
        "--question-root",
        str(question),
        "--image-question-root",
        str(image_question),
        "--output-root",
        str(args.root),
        "--preflight-path",
        str(args.root / "preflight_v1.json"),
        "--num-shards",
        "1",
        "--pope-overlap-cluster",
        clusters[0],
        "--bootstrap-draws",
        "5000",
        "--seed",
        "20260813",
        "--report",
        "reports/binary_polar_full10_external_eval.md",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
