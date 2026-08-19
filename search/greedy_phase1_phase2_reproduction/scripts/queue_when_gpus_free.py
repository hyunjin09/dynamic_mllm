#!/usr/bin/env python3
"""Wait for all requested GPUs to be idle, then run the full pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--min-free-gb", type=float, default=40.0)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--log", type=Path, required=True)
    return parser.parse_args()


def gpu_statuses() -> dict[int, dict[str, float | int | str]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    statuses: dict[int, dict[str, float | int | str]] = {}
    for line in output.splitlines():
        index, name, total, used, free, utilization = [part.strip() for part in line.split(",")]
        statuses[int(index)] = {
            "index": int(index),
            "name": name,
            "total_gb": int(total) / 1024.0,
            "used_gb": int(used) / 1024.0,
            "free_gb": int(free) / 1024.0,
            "utilization": int(utilization),
        }
    return statuses


def write_log(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    requested = [int(value) for value in args.gpu_ids.split(",") if value.strip()]
    if not requested:
        raise ValueError("no GPU IDs were provided")
    package_root = Path(__file__).resolve().parents[1]
    runner = package_root / "scripts" / "run_pipeline.sh"
    if not args.env_file.is_file():
        raise FileNotFoundError(args.env_file)

    stable = 0
    write_log(
        args.log,
        {
            "event": "queue_started",
            "gpu_ids": requested,
            "min_free_gb": args.min_free_gb,
            "max_utilization": args.max_utilization,
            "stable_checks": args.stable_checks,
        },
    )
    while stable < args.stable_checks:
        try:
            statuses = gpu_statuses()
            selected = [statuses[index] for index in requested]
            ready = all(
                float(status["free_gb"]) >= args.min_free_gb
                and int(status["utilization"]) <= args.max_utilization
                for status in selected
            )
            stable = stable + 1 if ready else 0
            write_log(
                args.log,
                {"event": "gpu_poll", "ready": ready, "stable": stable, "gpus": selected},
            )
        except Exception as exc:
            stable = 0
            write_log(args.log, {"event": "gpu_poll_error", "error": repr(exc)})
        if stable < args.stable_checks:
            time.sleep(args.poll_seconds)

    write_log(args.log, {"event": "pipeline_start"})
    env = {**os.environ, "ENV_FILE": str(args.env_file.resolve())}
    result = subprocess.run([str(runner), "all"], cwd=package_root, env=env, check=False)
    write_log(args.log, {"event": "pipeline_exit", "returncode": result.returncode})
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
