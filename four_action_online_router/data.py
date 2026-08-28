"""Checksum-bound manifest and source metadata helpers."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from experiments.train_binary_polar import file_sha256
from four_action_policy.actions import encode_action_route
from .supervision import PrefixTrie


SOURCE_FIELDS = (
    "uid",
    "answer",
    "all_answer_norms",
    "metric_name",
    "correctness_threshold",
    "max_new_tokens",
    "image_content_sha256",
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_verified_manifest(path: str | Path, expected_sha256: str) -> list[dict[str, Any]]:
    current = Path(path)
    if file_sha256(current) != expected_sha256:
        raise RuntimeError("online-router training manifest checksum mismatch")
    rows = load_jsonl(current)
    uids = [str(row.get("uid") or "") for row in rows]
    if any(not uid for uid in uids) or len(uids) != len(set(uids)):
        raise RuntimeError("online-router manifest contains empty or duplicate UIDs")
    return rows


def load_source_metadata(
    path: str | Path, expected_sha256: str, required_uids: set[str]
) -> dict[str, dict[str, Any]]:
    current = Path(path)
    if file_sha256(current) != expected_sha256:
        raise RuntimeError("four-action source manifest checksum mismatch")
    output = {}
    with current.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            uid = str(row.get("uid") or "")
            if uid not in required_uids:
                continue
            if any(field not in row for field in SOURCE_FIELDS):
                raise RuntimeError(f"source evaluator metadata is incomplete for {uid}")
            output[uid] = {field: row[field] for field in SOURCE_FIELDS}
    if set(output) != required_uids:
        raise RuntimeError("source evaluator metadata does not cover the training manifest")
    return output


def manifest_route_tensor(row: dict[str, Any], *, num_layers: int) -> torch.LongTensor:
    routes = row.get("valid_routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError(f"sample {row.get('uid')!r} has no valid routes")
    encoded = [
        encode_action_route(route["actions"], expected_layers=num_layers) for route in routes
    ]
    result = torch.stack(encoded)
    if torch.unique(result, dim=0).shape[0] != result.shape[0]:
        raise ValueError(f"sample {row.get('uid')!r} has duplicate valid routes")
    return result


def manifest_trie(row: dict[str, Any], *, num_layers: int) -> PrefixTrie:
    return PrefixTrie(manifest_route_tensor(row, num_layers=num_layers).tolist())


def choose_smoke_indices(
    rows: Sequence[dict[str, Any]], *, records: int, seed: int
) -> list[int]:
    if records < 6 or records > len(rows):
        raise ValueError("smoke records must cover the six dataset/type cells")

    def key(index: int, purpose: str) -> str:
        return sha256(f"{seed}:{purpose}:{rows[index]['uid']}".encode()).hexdigest()

    selected = []
    for route_type in ("W2C", "C2C"):
        for dataset in ("gqa", "chartqa", "textvqa"):
            candidates = [
                index
                for index, row in enumerate(rows)
                if row.get("route_type") == route_type and row.get("dataset") == dataset
            ]
            if not candidates:
                raise ValueError(f"smoke population lacks {route_type}/{dataset}")
            selected.append(min(candidates, key=lambda index: key(index, "cell")))
    remaining = [index for index in range(len(rows)) if index not in set(selected)]
    remaining.sort(key=lambda index: key(index, "fill"))
    selected.extend(remaining[: records - len(selected)])
    return selected
