"""Adapters for the existing binary-mask MCTS supervision.

The source directory is treated as immutable.  This module validates each
record and emits compact derived manifests; it never executes the model or
changes a reward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .actions import NUM_QWEN_LAYERS, count_transitions, mask_key
from .factorization_audit import factorization_coverage


@dataclass(frozen=True)
class ValidRoute:
    mask: tuple[int, ...]
    key: str
    num_visual_on_layers: int
    weight: float


@dataclass(frozen=True)
class BinaryMCTSExample:
    uid: str
    sample_id: str
    benchmark: str
    difficulty: str
    question: str
    image_path: str
    image_sha256: str | None
    source_asset_id: str | None
    split_group: str
    source_file: str
    root_reward: float
    all_off_reward: float
    valid_routes: tuple[ValidRoute, ...]
    evaluated_route_count: int

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        row["valid_routes"] = [asdict(route) for route in self.valid_routes]
        return row


def _coerce_binary_mask(value: Any, num_layers: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != num_layers:
        raise ValueError(f"route must be a list of {num_layers} binary actions")
    route = tuple(int(item) for item in value)
    if any(item not in (0, 1) for item in route):
        raise ValueError("route contains a non-binary action")
    return route


def _group_id(sample: dict[str, Any]) -> str:
    if sample.get("image_content_sha256"):
        return f"sha256:{sample['image_content_sha256']}"
    if sample.get("source_asset_id"):
        return f"asset:{sample['benchmark']}:{sample['source_asset_id']}"
    return f"uid:{sample.get('uid') or sample.get('sample_id')}"


def deterministic_group_split(
    group_id: str,
    *,
    seed: int = 20260809,
    train_fraction: float = 0.75,
    validation_fraction: float = 0.125,
) -> str:
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must be positive and sum to less than one")
    digest = sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64)
    if unit < train_fraction:
        return "train"
    if unit < train_fraction + validation_fraction:
        return "validation"
    return "test"


def _route_prior_weights(
    masks: list[tuple[int, ...]],
    *,
    all_on_weight: float,
) -> list[float]:
    if not 0 < all_on_weight <= 1:
        raise ValueError("all_on_weight must lie in (0, 1]")
    all_on = (1,) * len(masks[0])
    has_shorter = any(sum(mask) < len(mask) for mask in masks)
    raw = [all_on_weight if has_shorter and mask == all_on else 1.0 for mask in masks]
    total = sum(raw)
    return [value / total for value in raw]


def _deterministic_route_cap(
    masks: list[tuple[int, ...]],
    *,
    limit: int,
    seed: int,
    uid: str,
) -> list[tuple[int, ...]]:
    """Cap routes while retaining the sparsest and valid all-ON anchors."""
    if limit < 1:
        raise ValueError("max_valid_routes must be positive or None")
    if len(masks) <= limit:
        return masks
    anchors = [min(masks, key=lambda mask: (sum(mask), mask_key(mask)))]
    all_on = (1,) * len(masks[0])
    if all_on in masks and all_on not in anchors and len(anchors) < limit:
        anchors.append(all_on)
    remaining = [mask for mask in masks if mask not in anchors]
    remaining.sort(
        key=lambda mask: sha256(f"{seed}:{uid}:{mask_key(mask)}".encode("utf-8")).hexdigest()
    )
    return anchors + remaining[: limit - len(anchors)]


def parse_mcts_record(
    record: dict[str, Any],
    *,
    source_file: str,
    num_layers: int = NUM_QWEN_LAYERS,
    max_valid_routes: int | None = 50,
    route_cap_seed: int = 20260809,
    all_on_weight: float = 0.25,
) -> BinaryMCTSExample:
    if record.get("phase") != "binary_visual_mask_graph_mcts_v2":
        raise ValueError(f"unexpected label phase {record.get('phase')!r}")
    sample = record.get("sample")
    mcts = record.get("mcts")
    if not isinstance(sample, dict) or not isinstance(mcts, dict):
        raise ValueError("record must contain sample and mcts objects")
    uid = str(sample.get("uid") or "")
    question = str(sample.get("question") or "").strip()
    if not uid or not question:
        raise ValueError("record is missing uid or question")
    evaluated = mcts.get("evaluated_masks")
    if not isinstance(evaluated, list) or not evaluated:
        raise ValueError("record has no evaluated_masks")
    threshold = float(sample.get("correctness_threshold", 1.0))
    unique: dict[str, tuple[int, ...]] = {}
    for route in evaluated:
        if not isinstance(route, dict) or float(route.get("reward", 0.0)) < threshold:
            continue
        mask = _coerce_binary_mask(route.get("visual_on_mask"), num_layers)
        unique[mask_key(mask)] = mask
    masks = list(unique.values())
    if max_valid_routes is not None and len(masks) > max_valid_routes:
        masks = _deterministic_route_cap(
            masks,
            limit=max_valid_routes,
            seed=route_cap_seed,
            uid=uid,
        )
    masks.sort(key=lambda mask: (sum(mask), mask_key(mask)))
    weights = _route_prior_weights(masks, all_on_weight=all_on_weight) if masks else []
    routes = tuple(
        ValidRoute(
            mask=mask,
            key=mask_key(mask),
            num_visual_on_layers=sum(mask),
            weight=weight,
        )
        for mask, weight in zip(masks, weights)
    )
    group = _group_id(sample)
    return BinaryMCTSExample(
        uid=uid,
        sample_id=str(sample.get("sample_id") or uid),
        benchmark=str(sample.get("benchmark") or ""),
        difficulty=str(sample.get("mcts_difficulty") or ""),
        question=question,
        image_path=str(sample.get("local_image_path") or ""),
        image_sha256=sample.get("image_content_sha256"),
        source_asset_id=sample.get("source_asset_id"),
        split_group=group,
        source_file=source_file,
        root_reward=float(mcts.get("root_reward", 0.0)),
        all_off_reward=float(mcts.get("all_off_reward", 0.0)),
        valid_routes=routes,
        evaluated_route_count=len(evaluated),
    )


def iter_source_json(root: str | Path) -> Iterator[Path]:
    root = Path(root)
    for path in sorted(root.glob("raw/full_v2/shard_*_of_*/samples/*.json")):
        yield path


def load_mcts_example(path: str | Path, **kwargs: Any) -> BinaryMCTSExample:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    return parse_mcts_record(record, source_file=str(path), **kwargs)


def summarize_label_geometry(examples: Iterable[BinaryMCTSExample]) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    total = 0
    no_success = 0
    for example in examples:
        total += 1
        if not example.valid_routes:
            no_success += 1
        key = f"{example.benchmark}/{example.difficulty}"
        cell = cells.setdefault(
            key,
            {
                "samples": 0,
                "samples_with_valid_route": 0,
                "valid_routes": 0,
                "transition_histogram": {},
                "visual_on_histogram": {},
                "factorization_coverage": {
                    "1": {"direct_hits": 0, "segmented_hits": 0},
                    "5": {"direct_hits": 0, "segmented_hits": 0},
                    "10": {"direct_hits": 0, "segmented_hits": 0},
                },
            },
        )
        cell["samples"] += 1
        cell["samples_with_valid_route"] += int(bool(example.valid_routes))
        if example.valid_routes:
            coverage = factorization_coverage([route.mask for route in example.valid_routes])
            for top_k, result in coverage.items():
                cell["factorization_coverage"][top_k]["direct_hits"] += int(result["direct_hit"])
                cell["factorization_coverage"][top_k]["segmented_hits"] += int(result["segmented_hit"])
        for route in example.valid_routes:
            cell["valid_routes"] += 1
            transitions = str(count_transitions(route.mask))
            on_count = str(route.num_visual_on_layers)
            cell["transition_histogram"][transitions] = cell["transition_histogram"].get(transitions, 0) + 1
            cell["visual_on_histogram"][on_count] = cell["visual_on_histogram"].get(on_count, 0) + 1
    return {"samples": total, "samples_without_valid_route": no_success, "cells": cells}


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], *, overwrite: bool = False) -> int:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count
