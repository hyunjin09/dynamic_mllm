"""Compress strict local counterfactuals into replayable route trajectories."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrajectorySupervision:
    trajectories: list[dict[str, Any]]
    total_contexts: int
    covered_contexts: int
    covered_off: int
    covered_on: int


def route_covers_context(mask_key: str, row: dict[str, Any]) -> bool:
    """A route exposes the same pre-action state iff its earlier bits match."""
    layer = int(row["layer_index"])
    prefix = str(row["common_prefix_mask"])
    if len(prefix) != layer:
        raise ValueError(f"common prefix length {len(prefix)} != layer {layer}")
    if len(mask_key) <= layer:
        raise ValueError(f"route mask is too short for layer {layer}: {mask_key}")
    return mask_key[:layer] == prefix


def _validate_rows(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    uids = {str(row["uid"]) for row in rows}
    if len(uids) != 1:
        raise ValueError(f"trajectory rows must have one UID, got {sorted(uids)}")
    lengths: set[int] = set()
    for row in rows:
        layer = int(row["layer_index"])
        prefix = str(row["common_prefix_mask"])
        off_mask = str(row["off_mask_key"])
        on_mask = str(row["on_mask_key"])
        lengths.update((len(off_mask), len(on_mask)))
        if len(prefix) != layer or off_mask[:layer] != prefix or on_mask[:layer] != prefix:
            raise ValueError(f"invalid pre-action prefix for {row.get('pair_id')}")
        differences = [index for index, bits in enumerate(zip(off_mask, on_mask)) if bits[0] != bits[1]]
        if differences != [layer] or off_mask[layer] != "0" or on_mask[layer] != "1":
            raise ValueError(f"pair is not an oriented Hamming-1 edge: {row.get('pair_id')}")
        if str(row["label_action"]) not in {"off", "on"}:
            raise ValueError(f"invalid label_action: {row['label_action']!r}")
    if len(lengths) != 1:
        raise ValueError(f"inconsistent route lengths: {sorted(lengths)}")
    return next(iter(lengths))


def _stable_mask_key(mask_key: str) -> str:
    return hashlib.sha256(mask_key.encode("utf-8")).hexdigest()


def _target(row: dict[str, Any]) -> dict[str, Any]:
    action = str(row["label_action"])
    return {
        "pair_id": str(row["pair_id"]),
        "layer_index": int(row["layer_index"]),
        "label_action": action,
        "label": int(action == "on"),
        "weight": float(row.get("recommended_primary_weight", 1.0)),
        "common_prefix_mask": str(row["common_prefix_mask"]),
        "counter_global_layer_prior": bool(row.get("counter_global_layer_prior", False)),
        "counter_benchmark_layer_prior": bool(row.get("counter_benchmark_layer_prior", False)),
        "uid_layer_observable_variable": bool(row.get("uid_layer_observable_variable", False)),
    }


def build_trajectory_supervision(
    rows: list[dict[str, Any]],
    *,
    max_routes: int = 2,
) -> TrajectorySupervision:
    """Greedily select route masks and assign each covered local target once.

    The first route maximizes OFF-label coverage when OFF evidence exists. This
    prevents abundant ON labels from consuming the small replay budget. Later
    routes maximize remaining weighted coverage irrespective of action.
    """
    if max_routes <= 0:
        raise ValueError("max_routes must be positive")
    num_layers = _validate_rows(rows)
    if not rows:
        return TrajectorySupervision([], 0, 0, 0, 0)

    candidates = sorted(
        {str(row[key]) for row in rows for key in ("off_mask_key", "on_mask_key")},
        key=_stable_mask_key,
    )
    uncovered = set(range(len(rows)))
    trajectories: list[dict[str, Any]] = []

    for route_index in range(max_routes):
        off_priority = route_index == 0 and any(
            rows[index]["label_action"] == "off" for index in uncovered
        )
        scored: list[tuple[float, int, str, list[int]]] = []
        for mask_key in candidates:
            covered = [index for index in uncovered if route_covers_context(mask_key, rows[index])]
            eligible = [
                index for index in covered if not off_priority or rows[index]["label_action"] == "off"
            ]
            score = sum(float(rows[index].get("recommended_primary_weight", 1.0)) for index in eligible)
            scored.append((score, len(eligible), mask_key, covered))
        best_score, best_count, best_mask, best_covered = min(
            scored,
            key=lambda item: (-item[0], -item[1], _stable_mask_key(item[2])),
        )
        if best_score <= 0.0 or best_count == 0:
            if off_priority:
                off_priority = False
                scored = []
                for mask_key in candidates:
                    covered = [index for index in uncovered if route_covers_context(mask_key, rows[index])]
                    score = sum(
                        float(rows[index].get("recommended_primary_weight", 1.0)) for index in covered
                    )
                    scored.append((score, len(covered), mask_key, covered))
                best_score, best_count, best_mask, best_covered = min(
                    scored,
                    key=lambda item: (-item[0], -item[1], _stable_mask_key(item[2])),
                )
            if best_score <= 0.0 or best_count == 0:
                break

        assigned = sorted(best_covered, key=lambda index: (int(rows[index]["layer_index"]), str(rows[index]["pair_id"])))
        targets = [_target(rows[index]) for index in assigned]
        trajectories.append(
            {
                "uid": str(rows[0]["uid"]),
                "benchmark": str(rows[0]["benchmark"]),
                "split": str(rows[0]["split"]),
                "trajectory_index": route_index,
                "selection_stage": "off_priority" if off_priority else "total_coverage",
                "mask_key": best_mask,
                "num_layers": num_layers,
                "visual_on_layers": best_mask.count("1"),
                "weighted_selection_score": best_score,
                "targets": targets,
            }
        )
        uncovered.difference_update(assigned)
        if not uncovered:
            break

    all_targets = [target for item in trajectories for target in item["targets"]]
    return TrajectorySupervision(
        trajectories=trajectories,
        total_contexts=len(rows),
        covered_contexts=len(all_targets),
        covered_off=sum(target["label"] == 0 for target in all_targets),
        covered_on=sum(target["label"] == 1 for target in all_targets),
    )
