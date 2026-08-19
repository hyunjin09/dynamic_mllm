#!/usr/bin/env python3
"""Plan and launch bounded parallel GPU experiment jobs."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from harness_state import (
    GPU_JOB_STATUSES,
    GPU_TYPES,
    HarnessError,
    append_run_history,
    default_gpu_queue,
    gpu_queue_path,
    load_gpu_queue,
    mutate_gpu_queue,
    mutate_run_state,
    now_iso,
    project_root,
    write_gpu_queue,
)
from workflow_hooks import refresh_report_index


GPU_PROFILES = {
    "a100": {"partition": "a100", "node": "node01", "nodes": ["node01"], "mem": "16G", "cpus": 8},
    "a4000": {"partition": "a4000", "node": "node05", "nodes": ["node05"], "mem": "16G", "cpus": 8},
    "a5000": {"partition": "a5000", "node": "node04", "nodes": ["node04"], "mem": "16G", "cpus": 8},
    "a6000": {
        "partition": "a6000",
        "node": "node06",
        "nodes": ["node02", "node03", "node06", "node07"],
        "mem": "16G",
        "cpus": 8,
    },
}
NODE_SINGLE_GPU_MEM_LIMITS = {
    "node01": "60G",
    "node05": "25G",
    "node04": "30G",
    "node02": "50G",
    "node03": "50G",
    "node06": "30G",
    "node07": "60G",
}
NODE_CPU_ONLY_MEM_LIMITS = {
    "node07": "480G",
}
GPU_GRES_ALIASES = {
    "rtx6000": "a6000",
    "rtxa6000": "a6000",
}
PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded GPU experiment scheduler.")
    sub = parser.add_subparsers(dest="action", required=True)

    init = sub.add_parser("init", help="Create state/gpu_experiment_queue.json if missing.")
    init.add_argument("--project", required=True)

    add = sub.add_parser("add", help="Add a queued GPU experiment job.")
    add_common(add)
    add.add_argument("--id", required=True)
    add.add_argument("--exp-id", required=True)
    add.add_argument("--command", required=True, dest="experiment_command")
    add.add_argument("--priority", choices=["high", "medium", "low"], default="medium")
    add.add_argument("--owner", dest="owner_agent", default="code_agent")
    add.add_argument("--result-path", default="")

    status = sub.add_parser("status", help="Show current GPU capacity and queue summary.")
    add_status_common(status)

    plan = sub.add_parser("plan", help="Plan jobs that can be launched under the GPU cap.")
    add_status_common(plan)
    plan.add_argument("--json", action="store_true", help="Print machine-readable plan.")

    launch = sub.add_parser("launch", help="Launch planned jobs through sbatch.")
    add_status_common(launch)
    launch.add_argument("--execute", action="store_true", help="Actually submit planned jobs with sbatch.")

    list_cmd = sub.add_parser("list", help="List GPU jobs.")
    list_cmd.add_argument("--project", required=True)
    list_cmd.add_argument("--status", choices=sorted(GPU_JOB_STATUSES))

    update = sub.add_parser("update", help="Update a GPU job status or metadata.")
    update.add_argument("--project", required=True)
    update.add_argument("--id", required=True)
    update.add_argument("--status", choices=sorted(GPU_JOB_STATUSES))
    update.add_argument("--note")
    update.add_argument("--result-path")

    return parser.parse_args()


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", required=True)
    parser.add_argument("--gpu-type", choices=sorted(GPU_TYPES), default="auto")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--partition")
    parser.add_argument("--node")
    parser.add_argument(
        "--nodelist",
        help="Slurm nodelist, e.g. node[02-03]. Mutually exclusive with --node.",
    )
    parser.add_argument("--mem")
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--tmux-session")
    parser.add_argument("--slurm-job-name")
    parser.add_argument("--log-path")
    parser.add_argument(
        "--shell-mode",
        choices=["login", "plain"],
        default="login",
        help="Use a login shell (-lc) or a plain shell (-c) for sbatch --wrap. Use plain when conda run must preserve the target environment.",
    )


def add_status_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", required=True)
    parser.add_argument("--max-user-gpus", type=int)
    parser.add_argument("--squeue-output", help="Mock or captured squeue output file.")
    parser.add_argument("--all-squeue-output", help="Mock or captured all-user running squeue output file.")
    parser.add_argument("--scontrol-output", help="Mock or captured scontrol show node output file.")
    parser.add_argument("--available-gpus", help="Override free GPUs, e.g. a4000=2,a5000=1.")


def read_optional(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8")


def run_capture(command: list[str], timeout: int = 15) -> tuple[str, str]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", str(exc)
    if result.returncode != 0:
        return result.stdout, result.stderr.strip() or f"exit {result.returncode}"
    return result.stdout, ""


def parse_gpu_tokens(text: str) -> int:
    total = 0
    for line in text.splitlines():
        lower = line.lower()
        if not lower.strip() or "jobid" in lower:
            continue
        matches = re.findall(r"(?:gres/)?gpu(?:[:=][a-z0-9_]+)?[:=](\d+)", lower)
        if matches:
            total += sum(int(value) for value in matches)
            continue
        if re.search(r"\bgpu\b", lower):
            total += 1
    return total


def parse_memory_mb(value: str) -> int:
    raw = str(value or "").strip()
    if not raw:
        return 0
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGTkmgt]?B?|[KMGTkmgt])?", raw)
    if not match:
        raise HarnessError(f"Unsupported --mem value: {value}")
    amount = float(match.group(1))
    unit = (match.group(2) or "M").upper()
    if unit.endswith("B"):
        unit = unit[:-1]
    factors = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
    if unit not in factors:
        raise HarnessError(f"Unsupported --mem unit: {value}")
    return int(math.ceil(amount * factors[unit]))


def node_memory_cap_mb(node: str, gpus: int) -> int:
    if int(gpus or 0) <= 0 and node in NODE_CPU_ONLY_MEM_LIMITS:
        return parse_memory_mb(NODE_CPU_ONLY_MEM_LIMITS[node])
    raw_limit = NODE_SINGLE_GPU_MEM_LIMITS.get(node)
    if raw_limit is None:
        raise HarnessError(f"No scheduler memory cap configured for node {node}")
    return parse_memory_mb(raw_limit) * max(1, int(gpus or 0))


def node_memory_cap_label(node: str, gpus: int) -> str:
    if int(gpus or 0) <= 0 and node in NODE_CPU_ONLY_MEM_LIMITS:
        return f"CPU-only {NODE_CPU_ONLY_MEM_LIMITS[node]}"
    raw_limit = NODE_SINGLE_GPU_MEM_LIMITS.get(node)
    if raw_limit is None:
        raise HarnessError(f"No scheduler memory cap configured for node {node}")
    if int(gpus or 0) <= 1:
        return raw_limit
    return f"{raw_limit} per GPU ({format_memory_mb(node_memory_cap_mb(node, gpus))} total)"


def format_memory_mb(value_mb: int) -> str:
    if value_mb % (1024 * 1024) == 0:
        return f"{value_mb // (1024 * 1024)}T"
    if value_mb % 1024 == 0:
        return f"{value_mb // 1024}G"
    return f"{value_mb}M"


def requested_mem_mb(job: dict[str, Any]) -> int:
    return parse_memory_mb(str(job.get("mem") or ""))


def node_satisfies_memory_cap(job: dict[str, Any], node: str) -> bool:
    requested = requested_mem_mb(job)
    if requested <= 0:
        return True
    return requested <= node_memory_cap_mb(node, int(job.get("gpus") or 0))


def validate_node_memory_cap(job: dict[str, Any], node: str) -> None:
    requested = requested_mem_mb(job)
    if requested <= 0:
        return
    limit = node_memory_cap_mb(node, int(job.get("gpus") or 0))
    if requested > limit:
        raise HarnessError(
            f"{node} cap {node_memory_cap_label(node, int(job.get('gpus') or 0))} "
            f"is lower than requested --mem {job.get('mem')}"
        )


def validate_job_memory_cap(job: dict[str, Any]) -> None:
    nodelist_nodes = expand_node_list(str(job.get("nodelist") or ""))
    if nodelist_nodes:
        for node_name in nodelist_nodes:
            validate_node_memory_cap(job, node_name)
        return
    node = str(job.get("node") or "").strip()
    if node:
        validate_node_memory_cap(job, node)
        return

    gpu_type = str(job.get("gpu_type") or "").lower()
    if gpu_type == "auto":
        return
    profile = GPU_PROFILES.get(gpu_type)
    if not profile:
        return
    nodes = profile.get("nodes") or [profile["node"]]
    requested = requested_mem_mb(job)
    if requested <= 0:
        return
    min_limit = min(node_memory_cap_mb(node_name, int(job.get("gpus") or 0)) for node_name in nodes)
    if requested > min_limit:
        raise HarnessError(
            f"Requested --mem {job.get('mem')} exceeds the no-nodelist {gpu_type} cap "
            f"{format_memory_mb(min_limit)}; choose a valid --node or lower --mem"
        )


def validate_add_memory_request(job: dict[str, Any]) -> None:
    if requested_mem_mb(job) <= 0:
        return

    requested_node = str(job.get("node") or "").strip()
    requested_nodelist = str(job.get("nodelist") or "").strip()
    if requested_node and requested_nodelist:
        raise HarnessError("--node and --nodelist are mutually exclusive")
    if requested_node:
        validate_node_memory_cap(job, requested_node)
        return
    if requested_nodelist:
        nodelist_nodes = expand_node_list(requested_nodelist)
        if not nodelist_nodes:
            raise HarnessError(f"Unsupported --nodelist value: {requested_nodelist}")
        for node_name in nodelist_nodes:
            validate_node_memory_cap(job, node_name)
        return

    requested_gpu_type = str(job.get("gpu_type") or "auto").lower()
    if requested_gpu_type == "auto":
        gpu_types = list(GPU_PROFILES)
    else:
        gpu_types = [requested_gpu_type]

    candidates: list[str] = []
    labels: list[str] = []
    for gpu_type in gpu_types:
        profile = GPU_PROFILES.get(gpu_type)
        if not profile:
            continue
        for node in profile.get("nodes") or [profile["node"]]:
            labels.append(f"{node} cap {node_memory_cap_label(node, int(job.get('gpus') or 0))}")
            if node_satisfies_memory_cap(job, node):
                candidates.append(node)
    if candidates:
        return

    cap_summary = ", ".join(labels) if labels else "no configured GPU node caps"
    raise HarnessError(f"Requested --mem {job.get('mem')} exceeds scheduler node caps: {cap_summary}")


def normalize_gpu_type(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if raw in GPU_PROFILES:
        return raw
    return GPU_GRES_ALIASES.get(raw, "")


def partition_gpu_type(line: str) -> str:
    match = re.search(r"\bPartitions=(\S+)", line)
    if not match:
        return ""
    for raw_partition in re.split(r"[, ]+", match.group(1)):
        partition = raw_partition.rstrip("*").lower()
        if partition in GPU_PROFILES:
            return partition
    return ""


def parse_tres_gpu_count(value: str, expected_gpu_type: str = "") -> int:
    total = 0
    for raw_type, raw_count in re.findall(r"gres/gpu(?::([A-Za-z0-9_]+))?=(\d+)", value):
        gpu_type = normalize_gpu_type(raw_type) or expected_gpu_type
        if expected_gpu_type and gpu_type and gpu_type != expected_gpu_type:
            continue
        total += int(raw_count)
    return total


def parse_node_gpu_capacity(line: str) -> tuple[str, int]:
    expected_from_partition = partition_gpu_type(line)

    cfg_match = re.search(r"\bCfgTRES=(\S*)", line)
    if cfg_match:
        for raw_type, raw_count in re.findall(r"gres/gpu(?::([A-Za-z0-9_]+))?=(\d+)", cfg_match.group(1)):
            gpu_type = normalize_gpu_type(raw_type) or expected_from_partition
            if gpu_type in GPU_PROFILES:
                return gpu_type, int(raw_count)

    gres_match = re.search(r"\bGres=(\S+)", line, flags=re.IGNORECASE)
    if not gres_match:
        return "", 0
    for raw_type, raw_count in re.findall(r"gpu(?::([A-Za-z0-9_]+))?:(\d+)", gres_match.group(1), flags=re.IGNORECASE):
        gpu_type = normalize_gpu_type(raw_type) or expected_from_partition
        if gpu_type in GPU_PROFILES:
            return gpu_type, int(raw_count)
    return "", 0


def expand_node_list(value: str) -> list[str]:
    value = value.strip()
    if not value or value in {"(null)", "N/A"}:
        return []
    if "[" not in value:
        return [value]

    prefix, _, rest = value.partition("[")
    inner, _, _ = rest.partition("]")
    nodes: list[str] = []
    for chunk in inner.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            nodes.append(prefix + chunk)
            continue
        start_raw, end_raw = chunk.split("-", 1)
        width = max(len(start_raw), len(end_raw))
        for number in range(int(start_raw), int(end_raw) + 1):
            nodes.append(prefix + str(number).zfill(width))
    return nodes


def requested_nodelist_nodes(job: dict[str, Any]) -> list[str]:
    return expand_node_list(str(job.get("nodelist") or ""))


def parse_squeue_allocations_by_node(text: str) -> dict[str, int]:
    allocations: dict[str, int] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) >= 6:
            # Expected format from gpu_status:
            # job_id|state|partition|gres|node_count|node_list
            state = parts[1].strip().lower()
            gres = parts[3].strip()
            node_list = parts[5].strip()
        else:
            continue
        if state and state not in {"running", "r"}:
            continue
        gpu_count = parse_gpu_tokens(gres)
        if gpu_count <= 0:
            continue
        nodes = expand_node_list(node_list)
        if not nodes:
            continue
        per_node = max(1, gpu_count // len(nodes))
        for node in nodes:
            allocations[node] = allocations.get(node, 0) + per_node
    return allocations


def parse_scontrol_capacity(text: str, squeue_text: str = "") -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    available = {gpu_type: 0 for gpu_type in GPU_PROFILES}
    node_available: dict[str, dict[str, Any]] = {}
    squeue_allocations = parse_squeue_allocations_by_node(squeue_text)
    for line in text.splitlines():
        if not line.strip():
            continue
        node_match = re.search(r"\bNodeName=(\S+)", line)
        if not node_match:
            continue
        node = node_match.group(1)
        state_match = re.search(r"\bState=(\S+)", line)
        state = state_match.group(1).lower() if state_match else ""
        if any(bad in state for bad in ("down", "drain", "fail", "maint")):
            continue
        gpu_type, cfg = parse_node_gpu_capacity(line)
        if not gpu_type or cfg <= 0:
            continue
        alloc_match = re.search(r"\bAllocTRES=(\S*)", line)
        alloc = parse_tres_gpu_count(alloc_match.group(1), expected_gpu_type=gpu_type) if alloc_match else 0
        if alloc <= 0:
            alloc = squeue_allocations.get(node, 0)
        free = max(0, cfg - alloc)
        available[gpu_type] += free
        node_available[node] = {"gpu_type": gpu_type, "available": free, "capacity": cfg, "allocated": alloc}
    return available, node_available


def parse_scontrol_available(text: str, squeue_text: str = "") -> dict[str, int]:
    available, _ = parse_scontrol_capacity(text, squeue_text=squeue_text)
    return available


def parse_available_override(value: str | None) -> dict[str, int]:
    if not value:
        return {}
    result: dict[str, int] = {}
    for chunk in value.split(","):
        if not chunk.strip():
            continue
        key, _, raw_count = chunk.partition("=")
        key = key.strip().lower()
        if key not in GPU_PROFILES:
            raise HarnessError(f"Unknown GPU type in --available-gpus: {key}")
        result[key] = int(raw_count.strip())
    return result


def gpu_status(args: argparse.Namespace, queue: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    squeue_text = read_optional(args.squeue_output)
    if not squeue_text and not args.squeue_output:
        squeue_text, error = run_capture(["squeue", "--me", "-h", "-o", "%i|%j|%T|%b|%D|%R"])
        if error:
            warnings.append(f"squeue unavailable: {error}")
    own_gpus = parse_gpu_tokens(squeue_text)

    available = {gpu_type: 0 for gpu_type in GPU_PROFILES}
    node_available: dict[str, dict[str, Any]] = {}
    override = parse_available_override(args.available_gpus)
    if override:
        available.update(override)
    else:
        all_squeue_text = read_optional(args.all_squeue_output)
        if not all_squeue_text and not args.all_squeue_output:
            all_squeue_text, error = run_capture(["squeue", "-h", "-t", "R", "-o", "%i|%T|%P|%b|%D|%R"])
            if error:
                warnings.append(f"all-user squeue unavailable: {error}")

        scontrol_text = read_optional(args.scontrol_output)
        if not scontrol_text and not args.scontrol_output:
            scontrol_text, error = run_capture(["scontrol", "show", "node", "-o"])
            if error:
                warnings.append(f"scontrol unavailable for all nodes: {error}")
            if not scontrol_text.strip():
                lines = []
                seen_nodes: set[str] = set()
                for profile in GPU_PROFILES.values():
                    for node in profile.get("nodes", [profile["node"]]):
                        if node in seen_nodes:
                            continue
                        seen_nodes.add(node)
                        text, error = run_capture(["scontrol", "show", "node", node, "-o"])
                        if error:
                            warnings.append(f"scontrol unavailable for {node}: {error}")
                        lines.append(text)
                scontrol_text = "\n".join(lines)
        parsed_available, node_available = parse_scontrol_capacity(scontrol_text, squeue_text=all_squeue_text)
        available.update(parsed_available)

    max_user_gpus = args.max_user_gpus or int(queue.get("max_user_gpus") or 8)
    remaining_user_slots = max(0, max_user_gpus - own_gpus)
    return {
        "project": queue.get("project"),
        "own_gpus": own_gpus,
        "max_user_gpus": max_user_gpus,
        "remaining_user_slots": remaining_user_slots,
        "available_gpus": available,
        "available_node_gpus": node_available,
        "warnings": warnings,
    }


def default_job_fields(args: argparse.Namespace) -> dict[str, Any]:
    profile = GPU_PROFILES.get(args.gpu_type if args.gpu_type != "auto" else "a6000", GPU_PROFILES["a6000"])
    exp_id = getattr(args, "exp_id", "")
    job_id = getattr(args, "id", exp_id or "gpu_job")
    slurm_job_name = args.slurm_job_name or f"{job_id}_{args.gpu_type}".replace("auto", "gpu")
    return {
        "id": job_id,
        "exp_id": exp_id,
        "command": getattr(args, "experiment_command", ""),
        "priority": getattr(args, "priority", "medium"),
        "status": "queued",
        "gpu_type": args.gpu_type,
        "gpus": args.gpus,
        "partition": args.partition or ("" if args.gpu_type == "auto" else profile["partition"]),
        "node": args.node or "",
        "nodelist": args.nodelist or "",
        "mem": args.mem or profile["mem"],
        "cpus": args.cpus or profile["cpus"],
        "tmux_session": args.tmux_session or "",
        "slurm_job_name": slurm_job_name,
        "slurm_job_id": "",
        "log_path": args.log_path or f"runs/experiments/{exp_id}/run.log",
        "result_path": getattr(args, "result_path", ""),
        "shell_mode": args.shell_mode,
        "owner_agent": getattr(args, "owner_agent", "code_agent"),
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "notes": "",
    }


def selectable_jobs(queue: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = [job for job in queue.get("jobs", []) if str(job.get("status") or "").lower() == "queued"]
    return sorted(jobs, key=lambda job: (PRIORITY_RANK.get(str(job.get("priority") or "medium"), 1), job.get("created_at", ""), job.get("id", "")))


def candidate_gpu_types(job: dict[str, Any], available: dict[str, int]) -> list[str]:
    requested = str(job.get("gpu_type") or "auto").lower()
    needed = int(job.get("gpus") or 0)
    if requested != "auto":
        return [requested]
    if needed == 0:
        return [gpu_type for gpu_type in ("a4000", "a5000", "a6000", "a100") if gpu_type in GPU_PROFILES]
    return [gpu_type for gpu_type in ("a6000", "a100", "a5000", "a4000") if available.get(gpu_type, 0) >= needed]


def choose_node_for_gpu_type(
    job: dict[str, Any],
    gpu_type: str,
    needed: int,
    node_available: dict[str, dict[str, Any]],
    strict_memory: bool = True,
) -> str:
    requested_node = str(job.get("node") or "")
    if requested_node:
        entry = node_available.get(requested_node)
        if entry and entry.get("gpu_type") == gpu_type and int(entry.get("available") or 0) >= needed:
            validate_node_memory_cap(job, requested_node)
            return requested_node
        return ""

    requested_nodes = requested_nodelist_nodes(job)
    if requested_nodes:
        candidates = [
            (node, int(entry.get("available") or 0))
            for node in requested_nodes
            for entry in [node_available.get(node)]
            if entry
            and entry.get("gpu_type") == gpu_type
            and int(entry.get("available") or 0) >= needed
        ]
        if not candidates:
            return ""
        memory_candidates = [
            (node, available_count)
            for node, available_count in candidates
            if node_satisfies_memory_cap(job, node)
        ]
        if not memory_candidates:
            if strict_memory:
                available_labels = ", ".join(
                    f"{node} cap {node_memory_cap_label(node, int(job.get('gpus') or 0))}"
                    for node, _ in sorted(candidates)
                )
                raise HarnessError(
                    f"Requested memory exceeds available {gpu_type} node caps: "
                    f"{available_labels}; requested --mem {job.get('mem')}"
                )
            return ""
        return sorted(memory_candidates, key=lambda item: (-item[1], item[0]))[0][0]

    candidates = [
        (node, int(entry.get("available") or 0))
        for node, entry in node_available.items()
        if entry.get("gpu_type") == gpu_type and int(entry.get("available") or 0) >= needed
    ]
    if not candidates:
        return ""
    memory_candidates = [
        (node, available_count)
        for node, available_count in candidates
        if node_satisfies_memory_cap(job, node)
    ]
    if not memory_candidates:
        if strict_memory:
            available_labels = ", ".join(
                f"{node} cap {node_memory_cap_label(node, int(job.get('gpus') or 0))}"
                for node, _ in sorted(candidates)
            )
            raise HarnessError(
                f"Requested memory exceeds available {gpu_type} node caps: "
                f"{available_labels}; requested --mem {job.get('mem')}"
            )
        return ""
    return sorted(memory_candidates, key=lambda item: (-item[1], item[0]))[0][0]


def choose_profile_node_for_memory(job: dict[str, Any], gpu_type: str) -> str:
    profile = GPU_PROFILES[gpu_type]
    nodes = profile.get("nodes") or [profile["node"]]
    candidates = [node for node in nodes if node_satisfies_memory_cap(job, node)]
    if not candidates:
        labels = ", ".join(
            f"{node} cap {node_memory_cap_label(node, int(job.get('gpus') or 0))}"
            for node in nodes
        )
        raise HarnessError(f"Requested --mem {job.get('mem')} exceeds {gpu_type} node caps: {labels}")
    return sorted(candidates, key=lambda node: (-node_memory_cap_mb(node, int(job.get("gpus") or 0)), node))[0]


def plan_jobs(queue: dict[str, Any], status: dict[str, Any]) -> list[dict[str, Any]]:
    remaining = int(status["remaining_user_slots"])
    available = dict(status["available_gpus"])
    node_available = {
        node: dict(entry)
        for node, entry in dict(status.get("available_node_gpus") or {}).items()
    }
    planned: list[dict[str, Any]] = []
    for job in selectable_jobs(queue):
        needed = int(job.get("gpus") or 0)
        requested_gpu_type = str(job.get("gpu_type") or "auto").lower()
        gpu_types = candidate_gpu_types(job, available)
        if not gpu_types:
            continue

        # CPU-only jobs are submitted to the selected GPU-node partition with
        # no --gres flag. They do not consume the user's GPU slot cap.
        if needed == 0:
            for gpu_type in gpu_types:
                profile = GPU_PROFILES[gpu_type]
                node = choose_node_for_gpu_type(
                    job,
                    gpu_type,
                    0,
                    node_available,
                    strict_memory=requested_gpu_type != "auto",
                )
                if not node:
                    try:
                        node = choose_profile_node_for_memory(job, gpu_type)
                    except HarnessError:
                        if requested_gpu_type != "auto":
                            raise
                        continue
                planned_job = dict(job)
                planned_job["gpu_type"] = gpu_type
                planned_job["partition"] = job.get("partition") or profile["partition"]
                planned_job["node"] = job.get("node") or node
                planned_job["nodelist"] = job.get("nodelist") or ""
                planned_job["mem"] = job.get("mem") or profile["mem"]
                planned_job["cpus"] = job.get("cpus") or profile["cpus"]
                validate_job_memory_cap(planned_job)
                planned.append(planned_job)
                break
            continue

        for gpu_type in gpu_types:
            if needed > remaining or needed > available.get(gpu_type, 0):
                continue
            profile = GPU_PROFILES[gpu_type]
            node = choose_node_for_gpu_type(
                job,
                gpu_type,
                needed,
                node_available,
                strict_memory=requested_gpu_type != "auto",
            )
            if not node:
                continue
            planned_job = dict(job)
            planned_job["gpu_type"] = gpu_type
            planned_job["partition"] = job.get("partition") or profile["partition"]
            planned_job["node"] = node
            planned_job["nodelist"] = job.get("nodelist") or ""
            planned_job["mem"] = job.get("mem") or profile["mem"]
            planned_job["cpus"] = job.get("cpus") or profile["cpus"]
            validate_job_memory_cap(planned_job)
            planned.append(planned_job)
            remaining -= needed
            available[gpu_type] -= needed
            node_available[node]["available"] = int(node_available[node].get("available") or 0) - needed
            break
    return planned


def sbatch_args(root: Path, job: dict[str, Any]) -> list[str]:
    validate_job_memory_cap(job)
    log_path = root / str(job.get("log_path") or f"runs/experiments/{job['exp_id']}/run.log")
    body = f"cd {shlex.quote(str(root))} && {str(job['command'])}"
    shell_mode = str(job.get("shell_mode") or "login").lower()
    if shell_mode not in {"login", "plain"}:
        raise HarnessError(f"Unknown shell_mode for {job['id']}: {shell_mode}")
    shell_flag = "-lc" if shell_mode == "login" else "-c"
    args = [
        "sbatch",
        "--parsable",
        f"--job-name={str(job['slurm_job_name'])}",
        f"--partition={str(job['partition'])}",
        f"--mem={str(job['mem'])}",
        f"--cpus-per-task={int(job.get('cpus') or 8)}",
        f"--output={str(log_path)}",
        f"--error={str(log_path)}",
        "--open-mode=append",
    ]
    nodelist = str(job.get("nodelist") or "").strip()
    node = str(job.get("node") or "").strip()
    if nodelist:
        args.append(f"--nodelist={nodelist}")
    elif node:
        args.append(f"--nodelist={node}")
    gpus = int(job.get("gpus") or 0)
    if gpus > 0:
        args.append(f"--gres=gpu:{gpus}")
    args.append(f"--wrap=bash {shell_flag} {shlex.quote(body)}")
    return args


def launch_command(root: Path, job: dict[str, Any]) -> str:
    log_path = root / str(job.get("log_path") or f"runs/experiments/{job['exp_id']}/run.log")
    return " ".join([
        "mkdir",
        "-p",
        shlex.quote(str(log_path.parent)),
        "&&",
        *(shlex.quote(part) for part in sbatch_args(root, job)),
    ])


def mark_launched(root: Path, job: dict[str, Any], slurm_job_id: str) -> None:
    timestamp = now_iso()

    def update_queue(queue: dict[str, Any]) -> None:
        for candidate in queue["jobs"]:
            if candidate.get("id") == job["id"]:
                candidate.update({
                    "status": "running",
                    "launcher": "sbatch",
                    "gpu_type": job["gpu_type"],
                    "partition": job["partition"],
                    "node": job["node"],
                    "nodelist": job.get("nodelist", ""),
                    "mem": job["mem"],
                    "cpus": job["cpus"],
                    "tmux_session": job.get("tmux_session", ""),
                    "slurm_job_name": job["slurm_job_name"],
                    "slurm_job_id": slurm_job_id,
                    "updated_at": timestamp,
                })
                return
        raise HarnessError(f"GPU job not found: {job['id']}")

    mutate_gpu_queue(root, update_queue)

    def update_run_state(data: dict[str, Any]) -> None:
        data["status"] = "running"
        data["owner_agent"] = job.get("owner_agent") or "code_agent"
        data["current_step"] = f"Running GPU job {job['id']}"
        data["tmux_session"] = job.get("tmux_session", "")
        data["slurm_job_name"] = job["slurm_job_name"]
        data["slurm_job_id"] = slurm_job_id
        data["gpu_type"] = job["gpu_type"].upper()
        data["node"] = job["node"]
        if job.get("nodelist"):
            data["nodelist"] = job["nodelist"]
        data["log_path"] = job.get("log_path", "")
        if job.get("result_path"):
            data["result_path"] = job["result_path"]
        data["display_summary"] = f"GPU job {job['id']} is running for {job['exp_id']}."
        data["started_at"] = data.get("started_at") or timestamp
        data["updated_at"] = timestamp
        append_run_history(data, "gpu_launch", f"sbatch_job_id={slurm_job_id} slurm_job={job['slurm_job_name']}")

    mutate_run_state(root, job["exp_id"], update_run_state)
    refresh_report_index(root)


def execute_launch(root: Path, job: dict[str, Any]) -> None:
    if not shutil.which("sbatch"):
        raise HarnessError("sbatch is required for --execute.")
    log_path = root / str(job.get("log_path") or f"runs/experiments/{job['exp_id']}/run.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(sbatch_args(root, job), cwd=root, text=True, capture_output=True)
    if result.returncode != 0:
        raise HarnessError(result.stderr.strip() or f"launch failed for {job['id']}")
    output = result.stdout.strip().splitlines()
    slurm_job_id = output[-1].split(";")[0].strip() if output else ""
    if not slurm_job_id:
        raise HarnessError(f"sbatch did not return a job id for {job['id']}")
    mark_launched(root, job, slurm_job_id)


def print_status(status: dict[str, Any], queue: dict[str, Any]) -> None:
    queued = len([job for job in queue.get("jobs", []) if job.get("status") == "queued"])
    running = len([job for job in queue.get("jobs", []) if job.get("status") == "running"])
    print(f"own_gpus: {status['own_gpus']}/{status['max_user_gpus']}")
    print(f"remaining_user_slots: {status['remaining_user_slots']}")
    print("available_gpus: " + ", ".join(f"{key}={value}" for key, value in status["available_gpus"].items()))
    node_available = status.get("available_node_gpus") or {}
    if node_available:
        node_parts = [
            f"{node}:{entry.get('gpu_type')}={entry.get('available')}/{entry.get('capacity')}"
            for node, entry in sorted(node_available.items())
        ]
        print("available_nodes: " + ", ".join(node_parts))
    print(f"queue: queued={queued}, running={running}")
    for warning in status["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    try:
        root = project_root(args.project)
        if args.action == "init":
            path = gpu_queue_path(root)
            if path.exists():
                queue = load_gpu_queue(root)
            else:
                queue = default_gpu_queue(args.project)
                write_gpu_queue(root, queue)
            refresh_report_index(root)
            print(f"gpu jobs: {len(queue['jobs'])}")
            return 0

        if args.action == "add":
            job = default_job_fields(args)
            if job["gpus"] < 0:
                raise HarnessError("--gpus must be non-negative. Use --gpus 0 for CPU-only Slurm jobs on GPU-node partitions.")
            validate_add_memory_request(job)

            def add_job(queue: dict[str, Any]) -> None:
                if any(existing.get("id") == job["id"] for existing in queue["jobs"]):
                    raise HarnessError(f"GPU job already exists: {job['id']}")
                queue["jobs"].append(job)

            mutate_gpu_queue(root, add_job)
            refresh_report_index(root)
            print(f"added gpu job: {job['id']}")
            return 0

        if args.action == "list":
            queue = load_gpu_queue(root)
            for job in queue["jobs"]:
                if args.status and job.get("status") != args.status:
                    continue
                print(f"{job['id']}\t{job['status']}\t{job.get('priority', '')}\t{job.get('gpu_type', '')}\t{job.get('exp_id', '')}")
            return 0

        if args.action == "update":
            def update_job(queue: dict[str, Any]) -> None:
                for job in queue["jobs"]:
                    if job.get("id") != args.id:
                        continue
                    if args.status:
                        job["status"] = args.status
                    if args.note is not None:
                        job["notes"] = args.note
                    if args.result_path is not None:
                        job["result_path"] = args.result_path
                    job["updated_at"] = now_iso()
                    return
                raise HarnessError(f"GPU job not found: {args.id}")

            mutate_gpu_queue(root, update_job)
            refresh_report_index(root)
            print(f"updated gpu job: {args.id}")
            return 0

        queue = load_gpu_queue(root)
        status = gpu_status(args, queue)
        if args.action == "status":
            print_status(status, queue)
            return 0

        planned = plan_jobs(queue, status)
        if args.action == "plan":
            if args.json:
                print(json.dumps({"status": status, "planned_jobs": planned}, indent=2))
            else:
                print_status(status, queue)
                for job in planned:
                    print(f"plan\t{job['id']}\t{job['exp_id']}\t{job['gpu_type']}\t{job['slurm_job_name']}")
            return 0

        if args.action == "launch":
            for job in planned:
                command = launch_command(root, job)
                if not args.execute:
                    print(command)
                else:
                    execute_launch(root, job)
                    print(f"launched: {job['id']}")
            if not planned:
                print("no launchable GPU jobs")
            return 0

        raise HarnessError(f"Unknown command: {args.action}")
    except (HarnessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
