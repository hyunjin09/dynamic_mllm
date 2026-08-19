"""Router dataset and cached-feature loading helpers for Phase 5B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


NUM_LAYERS = 28
REQUIRED_RECORD_FIELDS = {
    "id",
    "sample_id",
    "benchmark",
    "split",
    "image",
    "question",
    "visual_on_mask",
    "visual_on_soft",
    "num_visual_tokens",
    "num_visual_on_layers",
    "route_source",
    "label_status",
}


def read_jsonl(path: Path, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    rows = []
    with path.open() as handle:
        for row_idx, line in enumerate(handle):
            if row_idx < offset:
                continue
            if limit is not None and len(rows) >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_router_record(row: dict[str, Any], expected_split: str | None = None) -> None:
    missing = sorted(REQUIRED_RECORD_FIELDS - set(row))
    if missing:
        raise ValueError(f"router record missing fields {missing}: {row.get('id', '<unknown>')}")
    if expected_split is not None and row["split"] != expected_split:
        raise ValueError(f"expected split {expected_split!r}, got {row['split']!r} for {row['id']}")
    if len(row["visual_on_mask"]) != NUM_LAYERS:
        raise ValueError(f"visual_on_mask must have {NUM_LAYERS} layers for {row['id']}")
    if len(row["visual_on_soft"]) != NUM_LAYERS:
        raise ValueError(f"visual_on_soft must have {NUM_LAYERS} layers for {row['id']}")
    if any(value not in (0, 1, False, True) for value in row["visual_on_mask"]):
        raise ValueError(f"visual_on_mask must be binary for {row['id']}")
    if int(sum(int(value) for value in row["visual_on_mask"])) != int(row["num_visual_on_layers"]):
        raise ValueError(f"num_visual_on_layers does not match visual_on_mask for {row['id']}")
    for idx, value in enumerate(row["visual_on_soft"]):
        current = float(value)
        if not 0.0 <= current <= 1.0:
            raise ValueError(f"visual_on_soft[{idx}] outside [0, 1] for {row['id']}")


def load_router_records(
    path: Path,
    *,
    limit: int | None = None,
    offset: int = 0,
    expected_split: str | None = None,
) -> list[dict[str, Any]]:
    rows = read_jsonl(path, limit=limit, offset=offset)
    for row in rows:
        validate_router_record(row, expected_split=expected_split)
    return rows


def route_tensor(record: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    values = torch.tensor(record["visual_on_mask"], dtype=torch.bool)
    if device is not None:
        values = values.to(device)
    return values.unsqueeze(0)


def previous_gate_tensor(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 1 or mask.shape[0] != NUM_LAYERS:
        raise ValueError(f"mask must have shape ({NUM_LAYERS},), got {tuple(mask.shape)}")
    prev = torch.zeros_like(mask, dtype=torch.long)
    prev[1:] = mask[:-1].to(dtype=torch.long)
    return prev


def collate_router_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot collate an empty router batch")
    for row in records:
        validate_router_record(row)
    masks = torch.tensor([row["visual_on_mask"] for row in records], dtype=torch.bool)
    soft = torch.tensor([row["visual_on_soft"] for row in records], dtype=torch.float32)
    return {
        "ids": [row["id"] for row in records],
        "sample_ids": [row["sample_id"] for row in records],
        "benchmarks": [row["benchmark"] for row in records],
        "splits": [row["split"] for row in records],
        "visual_on_mask": masks,
        "visual_on_soft": soft,
        "prev_gates": torch.stack([previous_gate_tensor(mask) for mask in masks], dim=0),
        "num_visual_tokens": torch.tensor([row["num_visual_tokens"] for row in records], dtype=torch.float32),
        "num_visual_on_layers": torch.tensor([row["num_visual_on_layers"] for row in records], dtype=torch.float32),
    }


def validate_cached_feature_sample(sample: dict[str, Any]) -> None:
    required = {
        "id",
        "sample_id",
        "benchmark",
        "split",
        "global_mean",
        "window_mean",
        "last_token",
        "labels",
        "soft_labels",
        "prev_gates",
        "layer_idx",
    }
    missing = sorted(required - set(sample))
    if missing:
        raise ValueError(f"cached feature sample missing fields {missing}")
    labels = sample["labels"]
    if labels.ndim != 1 or labels.shape[0] != NUM_LAYERS:
        raise ValueError(f"labels must have shape ({NUM_LAYERS},), got {tuple(labels.shape)}")
    feature_shape = tuple(sample["global_mean"].shape)
    if len(feature_shape) != 2 or feature_shape[0] != NUM_LAYERS:
        raise ValueError(f"global_mean must have shape ({NUM_LAYERS}, D), got {feature_shape}")
    for key in ["window_mean", "last_token"]:
        if tuple(sample[key].shape) != feature_shape:
            raise ValueError(f"{key} shape {tuple(sample[key].shape)} does not match {feature_shape}")
    if "visual_summaries" in sample:
        visual_summaries = sample["visual_summaries"]
        if visual_summaries.ndim != 3:
            raise ValueError(
                f"visual_summaries must have shape ({NUM_LAYERS}, N, D), got {tuple(visual_summaries.shape)}"
            )
        if int(visual_summaries.shape[0]) != NUM_LAYERS or int(visual_summaries.shape[-1]) != feature_shape[-1]:
            raise ValueError(
                f"visual_summaries shape {tuple(visual_summaries.shape)} is incompatible with {feature_shape}"
            )
        if int(visual_summaries.shape[1]) <= 0:
            raise ValueError("visual_summaries must contain at least one summary")
    for key in ["soft_labels", "prev_gates", "layer_idx"]:
        if tuple(sample[key].shape) != (NUM_LAYERS,):
            raise ValueError(f"{key} must have shape ({NUM_LAYERS},), got {tuple(sample[key].shape)}")


def load_cached_feature_sample(path: Path) -> dict[str, Any]:
    sample = torch.load(path, map_location="cpu")
    validate_cached_feature_sample(sample)
    return sample


def collate_cached_feature_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty cached-feature batch")
    for sample in samples:
        validate_cached_feature_sample(sample)
    has_visual_summaries = ["visual_summaries" in sample for sample in samples]
    if any(has_visual_summaries) and not all(has_visual_summaries):
        raise ValueError("cached-feature samples must either all include visual_summaries or none")
    batch = {
        "ids": [sample["id"] for sample in samples],
        "sample_ids": [sample["sample_id"] for sample in samples],
        "benchmarks": [sample["benchmark"] for sample in samples],
        "splits": [sample["split"] for sample in samples],
        "global_mean": torch.stack([sample["global_mean"] for sample in samples], dim=0),
        "window_mean": torch.stack([sample["window_mean"] for sample in samples], dim=0),
        "last_token": torch.stack([sample["last_token"] for sample in samples], dim=0),
        "labels": torch.stack([sample["labels"] for sample in samples], dim=0),
        "soft_labels": torch.stack([sample["soft_labels"] for sample in samples], dim=0),
        "prev_gates": torch.stack([sample["prev_gates"] for sample in samples], dim=0),
        "layer_idx": torch.stack([sample["layer_idx"] for sample in samples], dim=0),
        "num_visual_tokens": torch.tensor([sample["num_visual_tokens"] for sample in samples], dtype=torch.float32),
    }
    if all(has_visual_summaries):
        batch["visual_summaries"] = torch.stack([sample["visual_summaries"] for sample in samples], dim=0)
    return batch
