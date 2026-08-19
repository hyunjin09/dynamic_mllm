from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_COUNT_BY_BENCHMARK = {
    "gqa": 2000,
    "chartqa": 1000,
    "docvqa": 1000,
    "textvqa": 1000,
}


def manifest_path(root: Path, benchmark: str, bucket: str) -> Path:
    count = _COUNT_BY_BENCHMARK[benchmark]
    return root / f"{benchmark}_{bucket}_{count}.jsonl"


def _normalized_candidate(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def resolve_local_image(root: Path, record: dict[str, Any]) -> Path:
    source_path = Path(str(record["image_path"]))
    candidate = root / "images" / str(record["benchmark"]) / source_path.name
    if not candidate.is_file():
        raise FileNotFoundError(f"Local image not found for {record.get('id')}: {candidate}")
    return candidate


def select_stage_a_samples(
    root: Path,
    benchmarks: list[str],
    buckets: list[str],
    per_benchmark_bucket: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        for bucket in buckets:
            source = manifest_path(root, benchmark, bucket)
            with source.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if index >= per_benchmark_bucket:
                        break
                    record = json.loads(line)
                    record["local_image_path"] = str(resolve_local_image(root, record))
                    record["manifest_path"] = str(source)
                    record["manifest_index"] = index
                    selected.append(record)
    if len(selected) != len(benchmarks) * len(buckets) * per_benchmark_bucket:
        raise ValueError("Stage A sample selection returned fewer records than requested")

    answers = [str(row["answer"]) for row in selected]
    for index, record in enumerate(selected):
        candidates = [str(record["answer"])]
        prediction = str(record.get("prediction") or "").strip()
        if prediction and _normalized_candidate(prediction) != _normalized_candidate(candidates[0]):
            candidates.append(prediction)
        for offset in range(1, len(selected) + 1):
            alternative = answers[(index + offset) % len(selected)]
            if all(_normalized_candidate(alternative) != _normalized_candidate(item) for item in candidates):
                candidates.append(alternative)
                break
        record["parity_only_candidates"] = candidates[:2]
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
