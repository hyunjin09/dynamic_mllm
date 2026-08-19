#!/usr/bin/env python3
"""Migrate and validate the project-local environment under Slurm."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


EXPECTED_TRANSFORMERS = "5.3.0"


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> dict:
    print("RUN", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout.splitlines()[-100:],
        "stderr_tail": completed.stderr.splitlines()[-100:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    project = Path(args.project).resolve()
    report_path = (project / args.report).resolve() if not Path(args.report).is_absolute() else Path(args.report)
    if not report_path.is_relative_to(project):
        raise ValueError("report must be inside the project")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is unavailable")
    python = project / ".venv/bin/python"
    requirements = project / "requirements.txt"
    if not python.exists() or not requirements.exists():
        raise FileNotFoundError("project .venv or requirements.txt is missing")
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(project / ".uv-cache")
    env["UV_LINK_MODE"] = "copy"
    commands = [
        [uv, "pip", "install", "--python", str(python), "-r", str(requirements)],
        [uv, "pip", "check", "--python", str(python)],
        [
            str(python),
            "-c",
            (
                "import json, torch, torchvision, transformers, accelerate, datasets; "
                "print(json.dumps({'torch':torch.__version__,'torchvision':torchvision.__version__,"
                "'transformers':transformers.__version__,'accelerate':accelerate.__version__,"
                "'datasets':datasets.__version__},sort_keys=True))"
            ),
        ],
        [str(python), "-m", "unittest", "tests.test_stage_a_utils"],
        [str(python), "tools/run_binary_policy_contracts.py"],
    ]
    results = []
    for command in commands:
        result = run(command, cwd=project, env=env)
        results.append(result)
        if result["returncode"] != 0:
            break
    version = None
    if len(results) >= 3 and results[2]["returncode"] == 0 and results[2]["stdout_tail"]:
        try:
            version = json.loads(results[2]["stdout_tail"][-1]).get("transformers")
        except (json.JSONDecodeError, AttributeError):
            version = None
    passed = len(results) == len(commands) and all(row["returncode"] == 0 for row in results) and version == EXPECTED_TRANSFORMERS
    report = {
        "migration": "transformers_4.51.3_to_5.3.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project": str(project),
        "requirements": str(requirements),
        "rollback_requirements": str(project / "workspace/env_migrations/requirements_transformers_4_51_3.txt"),
        "expected_transformers": EXPECTED_TRANSFORMERS,
        "observed_transformers": version,
        "commands": results,
        "passed": passed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
