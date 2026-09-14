"""Deterministic corpus and sampling helpers for shared-union Stage-2 training."""

from __future__ import annotations

from collections import defaultdict
import random
from typing import Any, Mapping, Sequence

from dense_failure_stage2.v1_router import ACTION_NAMES, ACTION_TO_INDEX


OPERATING_POINTS = ("P98", "P95", "P90")


def merge_threshold_routes(
    rows_by_point: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_route_type: str,
) -> list[dict[str, Any]]:
    """Deduplicate threshold-expanded routes and retain exact validity metadata."""
    merged: dict[str, dict[str, Any]] = {}
    for point in OPERATING_POINTS:
        for source in rows_by_point.get(point, ()):
            row = dict(source)
            if str(row.get("operating_point")) != point:
                raise ValueError(f"route operating-point mismatch: {row.get('route_id')}")
            if str(row.get("route_type")) != expected_route_type:
                raise ValueError(f"route-type mismatch: {row.get('route_id')}")
            actions = [str(action) for action in row["actions"]]
            if len(actions) != 28 or any(action not in ACTION_TO_INDEX for action in actions):
                raise ValueError(f"invalid action trajectory: {row.get('route_id')}")
            route_id = str(row["route_id"])
            identity = {
                "uid": str(row["uid"]),
                "dataset": str(row["dataset"]),
                "source_regime": str(row["source_regime"]),
                "route_source": expected_route_type,
                "actions": actions,
            }
            if route_id not in merged:
                merged[route_id] = {
                    "schema_version": "stage2_shared_union_route_v1",
                    "route_id": route_id,
                    **identity,
                    "valid_operating_points": [],
                    "trigger_layers": {},
                }
            existing = merged[route_id]
            for key, value in identity.items():
                if existing[key] != value:
                    raise ValueError(f"threshold-expanded route identity differs: {route_id}:{key}")
            existing["valid_operating_points"].append(point)
            existing["trigger_layers"][point] = int(row["trigger_layer"])

    output = []
    semantic_keys: set[tuple[str, tuple[str, ...]]] = set()
    for route_id in sorted(merged):
        row = merged[route_id]
        row["valid_operating_points"] = [
            point for point in OPERATING_POINTS if point in row["valid_operating_points"]
        ]
        row["activation_layer"] = min(row["trigger_layers"].values())
        non_full = [
            layer
            for layer in range(row["activation_layer"], 28)
            if row["actions"][layer] != "FULL"
        ]
        row["non_full_count"] = len(non_full)
        row["first_non_full_layer"] = non_full[0] if non_full else None
        row["last_non_full_layer"] = non_full[-1] if non_full else None
        if expected_route_type == "preservation_full" and non_full:
            raise ValueError(f"preservation route is not all-FULL: {route_id}")
        if expected_route_type == "single" and len(non_full) != 1:
            raise ValueError(f"single route has {len(non_full)} active non-FULL states: {route_id}")
        if expected_route_type == "mcts" and not non_full:
            raise ValueError(f"MCTS route has no active non-FULL state: {route_id}")
        semantic_key = (row["uid"], tuple(row["actions"]))
        if semantic_key in semantic_keys:
            raise ValueError(f"semantically duplicated route: {route_id}")
        semantic_keys.add(semantic_key)
        row.update(
            {
                "valid_for_P98": "P98" in row["valid_operating_points"],
                "valid_for_P95": "P95" in row["valid_operating_points"],
                "valid_for_P90": "P90" in row["valid_operating_points"],
            }
        )
        output.append(row)
    return output


def balanced_uid_draws(uids: Sequence[str], count: int, rng: random.Random) -> list[str]:
    """Draw UIDs in shuffled cycles, giving every UID near-identical weight."""
    ordered = sorted({str(uid) for uid in uids})
    if not ordered or count < 1:
        raise ValueError("a non-empty UID population and positive draw count are required")
    output: list[str] = []
    while len(output) < int(count):
        cycle = list(ordered)
        rng.shuffle(cycle)
        output.extend(cycle[: int(count) - len(output)])
    return output


def sample_single_layers(route: Mapping[str, Any], rng: random.Random) -> list[int]:
    """Positive-anchored Random-4 over the union-active portion of one single route."""
    actions = [str(action) for action in route["actions"]]
    activation = int(route["activation_layer"])
    correction = [
        layer for layer in range(activation, len(actions)) if actions[layer] != "FULL"
    ]
    if len(correction) != 1:
        raise ValueError("single route must contain exactly one active correction")
    correction_layer = correction[0]
    before = [layer for layer in range(activation, correction_layer) if actions[layer] == "FULL"]
    after = [layer for layer in range(correction_layer + 1, len(actions)) if actions[layer] == "FULL"]
    selected = [correction_layer]
    if before:
        selected.append(rng.choice(before))
    if after:
        selected.append(rng.choice(after))
    remaining = [
        layer
        for layer in range(activation, len(actions))
        if actions[layer] == "FULL" and layer not in selected
    ]
    while remaining and len(selected) < 4:
        chosen = rng.choice(remaining)
        selected.append(chosen)
        remaining.remove(chosen)
    return sorted(selected)


def sample_preservation_layers(route: Mapping[str, Any], rng: random.Random) -> list[int]:
    activation = int(route["activation_layer"])
    eligible = list(range(activation, len(route["actions"])))
    if not eligible:
        raise ValueError("preservation route has no active layers")
    return sorted(rng.sample(eligible, min(4, len(eligible))))


def sample_mcts_layers(
    route: Mapping[str, Any], rng: random.Random, *, state_cap: int = 8
) -> list[int]:
    """Retain every corrective state up to the cap plus bounded FULL context."""
    actions = [str(action) for action in route["actions"]]
    activation = int(route["activation_layer"])
    non_full = [layer for layer in range(activation, len(actions)) if actions[layer] != "FULL"]
    if not non_full:
        raise ValueError("MCTS route must contain at least one active non-FULL state")
    if state_cap < len(non_full):
        raise ValueError("MCTS state cap cannot discard a corrective state")
    selected = list(non_full)
    neighbors = []
    for layer in non_full:
        for candidate in (layer - 1, layer + 1):
            if activation <= candidate < len(actions) and actions[candidate] == "FULL":
                neighbors.append(candidate)
    for candidate in sorted(set(neighbors)):
        if len(selected) < state_cap:
            selected.append(candidate)
    remaining = [
        layer
        for layer in range(activation, len(actions))
        if actions[layer] == "FULL" and layer not in selected
    ]
    while remaining and len(selected) < state_cap:
        chosen = rng.choice(remaining)
        selected.append(chosen)
        remaining.remove(chosen)
    return sorted(selected)


def trigger_layer(score_row: Mapping[str, Any], threshold: float) -> int | None:
    hits = [layer for layer in range(28) if float(score_row[f"p_{layer}"]) > float(threshold)]
    return hits[0] if hits else None


def routes_by_uid(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row["uid"])].append(dict(row))
    for values in output.values():
        values.sort(key=lambda row: str(row["route_id"]))
    return dict(output)


def summarize_schedule(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    kinds: dict[str, int] = defaultdict(int)
    actions: dict[str, int] = defaultdict(int)
    for row in rows:
        kinds[str(row["draw_kind"])] += 1
        for action in row["selected_actions"]:
            actions[str(action)] += 1
    return {"draw_kinds": dict(kinds), "selected_actions": dict(actions)}
