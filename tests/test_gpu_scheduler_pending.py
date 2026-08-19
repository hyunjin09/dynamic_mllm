"""Contracts for submitting capacity-safe jobs directly to the Slurm queue."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra"))

from gpu_scheduler import pending_submission_jobs


def _job(name: str) -> dict:
    return {
        "id": name,
        "status": "queued",
        "priority": "high",
        "created_at": name,
        "gpu_type": "a6000",
        "gpus": 1,
        "partition": "a6000",
        "node": "",
        "nodelist": "node[02,06-07]",
        "mem": "30G",
        "cpus": 10,
    }


def test_pending_submission_uses_user_cap_without_requiring_current_free_gpu():
    queue = {"jobs": [_job(f"cap{cap}") for cap in (24, 22, 20, 18)]}
    status = {"remaining_user_slots": 4}

    planned = pending_submission_jobs(queue, status)

    assert [job["id"] for job in planned] == ["cap18", "cap20", "cap22", "cap24"]
    assert all(job["nodelist"] == "node[02,06-07]" for job in planned)


def test_pending_submission_never_exceeds_user_gpu_cap():
    queue = {"jobs": [_job(f"cap{cap}") for cap in (24, 22, 20, 18)]}

    planned = pending_submission_jobs(queue, {"remaining_user_slots": 2})

    assert len(planned) == 2
