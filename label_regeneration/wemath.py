"""Frozen We-Math2.0-Pro manifest contract for binary-route extraction."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


DATASET_ID = "We-Math/We-Math2.0-Pro"
DATASET_REVISION = "c1d9f3ccea7361069f0442362e781d1ae7a28e94"
EXPECTED_ROWS = 4552
MAX_NEW_TOKENS = 96
PROMPT_SUFFIX = (
    "Return only the final answer enclosed in <answer> and </answer>; "
    "do not include reasoning."
)


def technical_invalid_reasons(row: dict) -> list[str]:
    """Return the approved outcome-blind missing-field exclusions."""
    reasons = []
    if not str(row.get("question") or "").strip():
        reasons.append("empty_question")
    if not str(row.get("answer") or "").strip():
        reasons.append("empty_answer")
    return reasons


def build_wemath_record(
    row: dict,
    *,
    source_index: int,
    image_path: str | Path,
    image_sha256: str,
) -> dict:
    """Normalize one official Pro record for the existing MCTS runtime."""
    image_path = Path(image_path).resolve()
    idx = str(row["idx"])
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    if not idx or technical_invalid_reasons(row):
        raise ValueError(f"empty required We-Math field at source index {source_index}")
    return {
        "uid": f"wemath2pro:{idx}",
        "sample_id": idx,
        "benchmark": "wemath2pro",
        "question_id": str(row["question_id"]),
        "question": question,
        "prompt": f"{question}\n{PROMPT_SUFFIX}",
        "answer": answer,
        "all_answer_norms": None,
        "metric_name": "wemath2pro_mathruler_accuracy",
        "correctness_threshold": 1.0,
        "max_new_tokens": MAX_NEW_TOKENS,
        "difficulty": str(row["difficulty"]),
        "knowledge_points": [str(value) for value in row.get("knowledge points", [])],
        "source_dataset": DATASET_ID,
        "source_dataset_revision": DATASET_REVISION,
        "source_split": "pro",
        "source_index": int(source_index),
        "image_group_id": f"sha256:{image_sha256}",
        "local_image_path": str(image_path),
        "image_content_sha256": image_sha256,
        "image_size_bytes": image_path.stat().st_size,
        "max_image_tokens": None,
    }


def deterministic_wemath_smoke_records(
    records: list[dict], *, count: int = 5, seed: int = 20260811
) -> list[dict]:
    if len(records) < count:
        raise ValueError(f"need {count} smoke records, found {len(records)}")
    ranked = sorted(
        records,
        key=lambda row: sha256(f"{seed}:{row['uid']}".encode("utf-8")).hexdigest(),
    )
    alternating = [index % 2 for index in range(28)]
    middle_off = [1] * 28
    for index in range(8, 20):
        middle_off[index] = 0
    selected = []
    for row in ranked[:count]:
        selected.append({**row, "mixed_masks": [alternating, middle_off]})
    return selected
