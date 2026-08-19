"""Pure deterministic utilities for the bounded P11 routing diagnostic."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import math
from statistics import median
from typing import Iterable


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty value list")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _distribution(values: Iterable[int]) -> dict[str, float]:
    materialized = [float(value) for value in values]
    return {
        "mean": sum(materialized) / len(materialized),
        "minimum": min(materialized),
        "q25": _quantile(materialized, 0.25),
        "median": float(median(materialized)),
        "q75": _quantile(materialized, 0.75),
        "maximum": max(materialized),
    }


def summarize_label_geometry(rows: list[dict]) -> dict:
    """Summarize complete-mask coverage once per input over nonempty valid sets."""
    if not rows:
        raise ValueError("label-geometry rows cannot be empty")
    mask_coverage: Counter[tuple[int, ...]] = Counter()
    set_sizes: list[int] = []
    minimum_on: list[int] = []
    median_on: list[int] = []
    maximum_on: list[int] = []
    all_on_records = 0
    all_off_records = 0
    shortcut_records = 0
    width = None
    for row in rows:
        masks = [tuple(int(value) for value in route["mask"]) for route in row.get("valid_routes", [])]
        if not masks:
            raise ValueError("label geometry requires nonempty valid sets")
        if len(set(masks)) != len(masks):
            raise ValueError(f"duplicate valid mask in {row.get('uid')!r}")
        if width is None:
            width = len(masks[0])
        if any(len(mask) != width for mask in masks):
            raise ValueError("inconsistent mask width")
        all_on = (1,) * width
        all_off = (0,) * width
        current = set(masks)
        mask_coverage.update(current)
        counts = [sum(mask) for mask in masks]
        set_sizes.append(len(masks))
        minimum_on.append(min(counts))
        median_on.append(int(median(counts)) if len(counts) % 2 else median(counts))
        maximum_on.append(max(counts))
        has_all_on = all_on in current
        all_on_records += int(has_all_on)
        all_off_records += int(all_off in current)
        shortcut_records += int(has_all_on and any(count < width for count in counts))

    assert width is not None
    all_on = (1,) * width
    ranked = sorted(mask_coverage.items(), key=lambda item: (-item[1], item[0]))
    non_all_on = [item for item in ranked if item[0] != all_on]

    def mask_record(item: tuple[tuple[int, ...], int]) -> dict:
        mask, count = item
        return {
            "mask": "".join(map(str, mask)),
            "inputs": count,
            "coverage": count / len(rows),
            "visual_on_layers": sum(mask),
        }

    union_coverage = {}
    row_sets = [
        {tuple(int(value) for value in route["mask"]) for route in row["valid_routes"]}
        for row in rows
    ]
    for limit in (1, 5, 10, 25, 50):
        selected = {mask for mask, _ in ranked[:limit]}
        union_coverage[str(limit)] = sum(bool(valid_set & selected) for valid_set in row_sets) / len(rows)

    return {
        "records": len(rows),
        "mask_width": width,
        "unique_valid_masks": len(mask_coverage),
        "all_on_coverage": all_on_records / len(rows),
        "all_off_coverage": all_off_records / len(rows),
        "all_on_with_cheaper_valid_coverage": shortcut_records / len(rows),
        "valid_set_size": _distribution(set_sizes),
        "minimum_visual_on": _distribution(minimum_on),
        "median_visual_on": _distribution(median_on),
        "maximum_visual_on": _distribution(maximum_on),
        "top_10_masks": [mask_record(item) for item in ranked[:10]],
        "best_constant": mask_record(ranked[0]),
        "best_non_all_on_constant": mask_record(non_all_on[0]) if non_all_on else None,
        "top_n_union_coverage": union_coverage,
    }


def deterministic_within_dataset_shuffle(rows: list[dict], *, seed: int) -> dict[str, str]:
    """Return a deterministic within-dataset derangement of question donors."""
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[str(row["benchmark"])].append(str(row["uid"]))
    mapping: dict[str, str] = {}
    for benchmark, uids in sorted(groups.items()):
        if len(uids) < 2:
            raise ValueError(f"within-dataset shuffle requires at least two {benchmark} rows")
        ordered = sorted(
            uids,
            key=lambda uid: (sha256(f"{seed}:{benchmark}:{uid}".encode()).hexdigest(), uid),
        )
        for index, uid in enumerate(ordered):
            mapping[uid] = ordered[(index + 1) % len(ordered)]
    return mapping
