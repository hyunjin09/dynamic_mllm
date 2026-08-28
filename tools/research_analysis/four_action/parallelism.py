from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class WorkerLayout:
    rank: int
    world_size: int
    gpu_index: int
    replica_index: int
    replicas_per_gpu: int


def worker_layout(rank: int, world_size: int, gpu_count: int = 8) -> WorkerLayout:
    if gpu_count < 1 or world_size < 1 or world_size % gpu_count:
        raise ValueError("worker world size must be a positive multiple of the GPU count")
    if not 0 <= rank < world_size:
        raise ValueError(f"worker rank {rank} is outside world size {world_size}")
    replicas_per_gpu = world_size // gpu_count
    return WorkerLayout(
        rank=rank,
        world_size=world_size,
        gpu_index=rank % gpu_count,
        replica_index=rank // gpu_count,
        replicas_per_gpu=replicas_per_gpu,
    )


def partition_gpu_rows(
    rows: Iterable[dict[str, Any]], replica_index: int, replicas_per_gpu: int
) -> list[dict[str, Any]]:
    if replicas_per_gpu < 1 or not 0 <= replica_index < replicas_per_gpu:
        raise ValueError("replica index must be inside replicas-per-GPU")
    return [
        row
        for index, row in enumerate(rows)
        if index % replicas_per_gpu == replica_index
    ]


def artifact_names(replicas_per_gpu: int, replica_index: int) -> dict[str, str]:
    if replicas_per_gpu < 1 or not 0 <= replica_index < replicas_per_gpu:
        raise ValueError("replica index must be inside replicas-per-GPU")
    suffix = "" if replicas_per_gpu == 1 else f"_replica_{replica_index:02d}"
    return {
        "results": f"results{suffix}.jsonl",
        "failures": f"failures{suffix}.jsonl",
        "runtime": f"runtime{suffix}.json",
    }
