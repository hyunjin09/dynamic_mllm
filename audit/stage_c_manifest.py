from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence


def normalize_question(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(normalized.split())


def normalize_path(path: str) -> str:
    return str(Path(path).resolve(strict=False))


def record_checksum(row: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in row.items() if key != "record_sha256"}
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selection_hash(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def selection_asset_key(row: Mapping[str, Any]) -> str:
    return str(
        row.get("image_id")
        or row.get("image_sha256")
        or row.get("local_image_path")
        or row["id"]
    )


def select_unique_images(
    rows: Sequence[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("Selection count must be positive")
    ranked = [dict(row) for row in rows]
    for row in ranked:
        row["selection_hash"] = _selection_hash(seed, str(row["id"]))
    ranked.sort(key=lambda row: (row["selection_hash"], str(row["id"])))

    selected: list[dict[str, Any]] = []
    used_assets: set[str] = set()
    for rank, row in enumerate(ranked, start=1):
        asset = selection_asset_key(row)
        if asset in used_assets:
            continue
        row["selection_rank"] = rank
        row["selection_asset_key"] = asset
        selected.append(row)
        used_assets.add(asset)
        if len(selected) == count:
            return selected
    raise ValueError(
        f"Only {len(selected)} unique-image records are available; {count} required"
    )


def overlap_reasons(
    candidate: Mapping[str, Any], discovery: Mapping[str, set[Any]]
) -> list[str]:
    checks = {
        "id": (candidate.get("id"), "ids"),
        "question_id": (candidate.get("question_id"), "question_ids"),
        "annotation_id": (candidate.get("annotation_id"), "annotation_ids"),
        "image_id": (candidate.get("image_id"), "image_ids"),
        "image_sha256": (candidate.get("image_sha256"), "image_hashes"),
        "normalized_image_path": (
            normalize_path(str(candidate["local_image_path"]))
            if candidate.get("local_image_path")
            else None,
            "normalized_image_paths",
        ),
        "normalized_question": (
            candidate.get("normalized_question")
            or normalize_question(str(candidate.get("question") or "")),
            "normalized_questions",
        ),
    }
    reasons = [
        label
        for label, (value, set_name) in checks.items()
        if value is not None and value != "" and value in discovery.get(set_name, set())
    ]
    image_id = candidate.get("image_id")
    question = candidate.get("normalized_question") or normalize_question(
        str(candidate.get("question") or "")
    )
    if image_id and (str(image_id), question) in discovery.get(
        "image_question_pairs", set()
    ):
        reasons.append("image_question_pair")
    return reasons


def blocking_overlap_reasons(reasons: Sequence[str]) -> list[str]:
    """Question text alone is non-unique; retain it as an audit flag only."""
    return [reason for reason in reasons if reason != "normalized_question"]
