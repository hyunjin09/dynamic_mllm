#!/usr/bin/env python3
"""Shared state helpers for the research workflow harness."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


GPU_TYPES = {"auto", "a100", "a4000", "a5000", "a6000"}
GPU_JOB_STATUSES = {"queued", "running", "succeeded", "failed", "cancelled", "blocked"}


class HarnessError(RuntimeError):
    """Raised for recoverable harness/scheduler errors."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def project_root(project: str | Path) -> Path:
    root = Path(project).expanduser().resolve()
    if not root.exists():
        raise HarnessError(f"Project path does not exist: {root}")
    if not root.is_dir():
        raise HarnessError(f"Project path is not a directory: {root}")
    return root


def state_dir(root: Path) -> Path:
    path = root / "state"
    path.mkdir(parents=True, exist_ok=True)
    return path


def gpu_queue_path(root: Path) -> Path:
    return state_dir(root) / "gpu_experiment_queue.json"


def run_state_dir(root: Path) -> Path:
    path = state_dir(root) / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_state_path(root: Path, exp_id: str) -> Path:
    safe = exp_id.replace("/", "_").replace(" ", "_")
    return run_state_dir(root) / f"{safe}.json"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HarnessError(f"Invalid JSON file: {path}: {exc}") from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def default_gpu_queue(project: str | Path) -> dict[str, Any]:
    return {
        "project": str(Path(project).expanduser().resolve()),
        "max_user_gpus": 8,
        "jobs": [],
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }


def load_gpu_queue(root: Path) -> dict[str, Any]:
    path = gpu_queue_path(root)
    if not path.exists():
        queue = default_gpu_queue(root)
        write_gpu_queue(root, queue)
        return queue

    queue = read_json(path, default={})
    queue.setdefault("project", str(root))
    queue.setdefault("max_user_gpus", 8)
    queue.setdefault("jobs", [])
    return queue


def write_gpu_queue(root: Path, queue: dict[str, Any]) -> None:
    queue["updated_at"] = now_iso()
    write_json(gpu_queue_path(root), queue)


def mutate_gpu_queue(root: Path, fn: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    queue = load_gpu_queue(root)
    fn(queue)
    write_gpu_queue(root, queue)
    return queue


def default_run_state(exp_id: str) -> dict[str, Any]:
    timestamp = now_iso()
    return {
        "exp_id": exp_id,
        "status": "created",
        "owner_agent": "",
        "current_step": "",
        "tmux_session": "",
        "slurm_job_name": "",
        "slurm_job_id": "",
        "gpu_type": "",
        "node": "",
        "log_path": "",
        "result_path": "",
        "display_summary": "",
        "history": [],
        "created_at": timestamp,
        "started_at": "",
        "updated_at": timestamp,
    }


def load_run_state(root: Path, exp_id: str) -> dict[str, Any]:
    path = run_state_path(root, exp_id)
    if not path.exists():
        return default_run_state(exp_id)
    state = read_json(path, default={})
    base = default_run_state(exp_id)
    base.update(state)
    return base


def write_run_state(root: Path, exp_id: str, data: dict[str, Any]) -> None:
    data["updated_at"] = now_iso()
    write_json(run_state_path(root, exp_id), data)


def mutate_run_state(root: Path, exp_id: str, fn: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    data = load_run_state(root, exp_id)
    fn(data)
    write_run_state(root, exp_id, data)
    return data


def append_run_history(data: dict[str, Any], event: str, message: str) -> None:
    data.setdefault("history", [])
    data["history"].append(
        {
            "time": now_iso(),
            "event": event,
            "message": message,
        }
    )
