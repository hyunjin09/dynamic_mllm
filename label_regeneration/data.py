"""Load and normalize the fixed 8K label-regeneration source pool."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable


SOURCE_FILES = (
    ("gqa", "complete_correct", "gqa_complete_correct_2000.jsonl", 2000),
    ("gqa", "complete_wrong", "gqa_complete_wrong_2000.jsonl", 2000),
    ("textvqa", "complete_correct", "textvqa_complete_correct_1000.jsonl", 1000),
    ("textvqa", "complete_wrong", "textvqa_complete_wrong_1000.jsonl", 1000),
    ("chartqa", "complete_correct", "chartqa_complete_correct_1000.jsonl", 1000),
    ("chartqa", "complete_wrong", "chartqa_complete_wrong_1000.jsonl", 1000),
)

CORRECTNESS_THRESHOLDS = {"gqa": 1.0, "textvqa": 0.5, "chartqa": 1.0}
MAX_NEW_TOKENS = {"gqa": 16, "textvqa": 16, "chartqa": 16}


def file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def _local_image_path(root: Path, row: dict) -> Path:
    benchmark = str(row["benchmark"])
    basename = Path(str(row["image_path"])).name
    candidate = root / "images" / benchmark / basename
    if not candidate.is_file():
        raise FileNotFoundError(f"missing local image for {row.get('id')}: {candidate}")
    return candidate.resolve()


def normalize_source_record(root: Path, row: dict, *, source_file: Path, source_index: int) -> dict:
    benchmark = str(row["benchmark"]).lower()
    if benchmark not in CORRECTNESS_THRESHOLDS:
        raise ValueError(f"unsupported benchmark: {benchmark}")
    uid = str(row.get("id") or f"{benchmark}:{row['sample_id']}")
    bucket = str(row["bucket"])
    image_path = _local_image_path(root, row)
    return {
        "uid": uid,
        "sample_id": str(row["sample_id"]),
        "benchmark": benchmark,
        "question": str(row["question"]),
        "prompt": str(row["prompt"]),
        "answer": str(row["answer"]),
        "all_answer_norms": row.get("all_answer_norms"),
        "metric_name": str(row["metric_name"]),
        "correctness_threshold": CORRECTNESS_THRESHOLDS[benchmark],
        "max_new_tokens": MAX_NEW_TOKENS[benchmark],
        "historical_all_on_status": "correct" if bucket == "complete_correct" else "wrong",
        "historical_all_on_prediction": row.get("prediction"),
        "historical_all_on_score": float(row.get("score") or 0.0),
        "source_asset_id": str(row.get("source_asset_id") or ""),
        "image_group_id": str(row.get("source_asset_id") or row.get("original_image_path") or image_path),
        "local_image_path": str(image_path),
        "image_content_sha256": row.get("image_content_sha256"),
        "image_size_bytes": image_path.stat().st_size,
        "source_file": str(source_file.resolve()),
        "source_index": int(source_index),
        "source_row_sha256": sha256(
            json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "max_image_tokens": None,
    }


def load_source_records(root: str | Path) -> list[dict]:
    root = Path(root).resolve()
    records: list[dict] = []
    seen_uids: set[str] = set()
    for benchmark, bucket, filename, expected in SOURCE_FILES:
        path = root / filename
        rows = list(_iter_jsonl(path))
        if len(rows) != expected:
            raise ValueError(f"{path} contains {len(rows)} rows; expected {expected}")
        for index, row in enumerate(rows):
            if row.get("benchmark") != benchmark or row.get("bucket") != bucket:
                raise ValueError(f"source cell mismatch at {path}:{index + 1}")
            normalized = normalize_source_record(root, row, source_file=path, source_index=index)
            if normalized["uid"] in seen_uids:
                raise ValueError(f"duplicate sample uid: {normalized['uid']}")
            seen_uids.add(normalized["uid"])
            records.append(normalized)
    if len(records) != 8000:
        raise ValueError(f"loaded {len(records)} records; expected 8000")
    return records


def deterministic_smoke_records(records: list[dict], *, per_dataset: int, seed: int) -> list[dict]:
    selected: list[dict] = []
    for benchmark in ("gqa", "textvqa", "chartqa"):
        candidates = [row for row in records if row["benchmark"] == benchmark]
        candidates.sort(
            key=lambda row: sha256(f"{seed}:{row['uid']}".encode("utf-8")).hexdigest()
        )
        if len(candidates) < per_dataset:
            raise ValueError(f"insufficient {benchmark} smoke records")
        selected.extend(candidates[:per_dataset])
    return selected


def safe_sample_filename(uid: str) -> str:
    readable = uid.replace(":", "__").replace("/", "_")
    suffix = sha256(uid.encode("utf-8")).hexdigest()[:10]
    return f"{readable}_{suffix}.json"
