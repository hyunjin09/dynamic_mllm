#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


RUNNING = True


def stop(*_args) -> None:
    global RUNNING
    RUNNING = False


def csv_open_contract(path: Path, *, resume: bool) -> tuple[str, bool]:
    if not path.exists():
        return "w", True
    if not resume:
        raise FileExistsError(f"refusing to overwrite {path}")
    return "a", path.stat().st_size == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample per-GPU utilization for a bounded run.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        raise ValueError("interval must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    mode, write_header = csv_open_contract(args.output, resume=args.resume)
    with args.output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                [
                    "timestamp_utc",
                    "gpu_index",
                    "memory_used_mib",
                    "memory_total_mib",
                    "utilization_gpu_percent",
                    "utilization_memory_percent",
                    "power_draw_watts",
                ]
            )
        handle.flush()
        while RUNNING:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            timestamp = datetime.now(timezone.utc).isoformat()
            for line in completed.stdout.splitlines():
                writer.writerow([timestamp, *[value.strip() for value in line.split(",")]])
            handle.flush()
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
