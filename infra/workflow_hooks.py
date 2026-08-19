#!/usr/bin/env python3
"""Workflow hooks for keeping lightweight project reports fresh."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def refresh_report_index(root: Path) -> None:
    """Refresh a small Markdown index of scheduler/run state.

    This is intentionally best-effort. Scheduler actions should not fail just
    because report generation fails.
    """
    try:
        root = Path(root).resolve()
        state_dir = root / "state"
        report_dir = root / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        queue_path = state_dir / "gpu_experiment_queue.json"
        queue = _read_json(queue_path) or {"jobs": []}
        jobs = queue.get("jobs", []) if isinstance(queue, dict) else []

        runs_dir = state_dir / "runs"
        run_files = sorted(runs_dir.glob("*.json")) if runs_dir.exists() else []

        lines: list[str] = []
        lines.append("# Workflow Report Index")
        lines.append("")
        lines.append(f"Project: `{root}`")
        lines.append("")

        lines.append("## GPU Jobs")
        lines.append("")
        if jobs:
            lines.append("| id | status | exp_id | gpu_type | slurm_job_id | log |")
            lines.append("|---|---|---|---|---|---|")
            for job in jobs:
                lines.append(
                    "| {id} | {status} | {exp_id} | {gpu_type} | {slurm_job_id} | {log_path} |".format(
                        id=job.get("id", ""),
                        status=job.get("status", ""),
                        exp_id=job.get("exp_id", ""),
                        gpu_type=job.get("gpu_type", ""),
                        slurm_job_id=job.get("slurm_job_id", ""),
                        log_path=job.get("log_path", ""),
                    )
                )
        else:
            lines.append("No GPU jobs recorded.")
        lines.append("")

        lines.append("## Run States")
        lines.append("")
        if run_files:
            lines.append("| exp_id | status | current_step | log | result |")
            lines.append("|---|---|---|---|---|")
            for path in run_files:
                state = _read_json(path)
                if not isinstance(state, dict):
                    continue
                lines.append(
                    "| {exp_id} | {status} | {current_step} | {log_path} | {result_path} |".format(
                        exp_id=state.get("exp_id", path.stem),
                        status=state.get("status", ""),
                        current_step=str(state.get("current_step", "")).replace("|", "\\|"),
                        log_path=state.get("log_path", ""),
                        result_path=state.get("result_path", ""),
                    )
                )
        else:
            lines.append("No run states recorded.")
        lines.append("")

        (report_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        # Keep this hook non-blocking.
        return
