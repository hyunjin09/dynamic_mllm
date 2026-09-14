"""Pure controls for shared corrective search under robust Stage-1 gates."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


POINTS = ("P98", "P95", "P90")
TRIGGER_ORDER = ("P90", "P95", "P98")
PAIR_OUTCOMES = (
    "EXISTING_SINGLE_REUSED",
    "EXISTING_MCTS_REUSED",
    "NEW_SINGLE_FIXABLE",
    "NEW_MCTS_FIXABLE",
    "UNRESOLVED_AT_BUDGET",
)


def trigger_order_status(triggers: Mapping[str, int | None]) -> dict[str, Any]:
    available = {str(point): int(layer) for point, layer in triggers.items() if layer is not None}
    if any(layer not in range(28) for layer in available.values()):
        raise ValueError("trigger layers must lie in 0..27")
    actual = sorted(available, key=lambda point: (available[point], TRIGGER_ORDER.index(point)))
    expected = [point for point in TRIGGER_ORDER if point in available]
    return {
        "monotonic": all(
            available[left] <= available[right]
            for left, right in zip(expected, expected[1:])
        ),
        "expected_order": expected,
        "actual_order": actual,
    }


def build_missing_union_rows(
    trigger_rows: Sequence[Mapping[str, Any]],
    compatibility_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    triggers: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in trigger_rows:
        if not bool(row.get("dense_wrong")) or not bool(row.get("triggered")):
            continue
        uid, point = str(row["uid"]), str(row["threshold_name"])
        if point not in POINTS or point in triggers[uid]:
            raise ValueError(f"invalid/duplicate trigger row: {uid}/{point}")
        triggers[uid][point] = row
    compatibility: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in compatibility_rows:
        key = (str(row["uid"]), str(row["operating_point"]))
        if key in compatibility:
            raise ValueError(f"duplicate compatibility row: {key}")
        compatibility[key] = row

    missing_uids = {
        uid for (uid, _point), row in compatibility.items() if bool(row["requires_new_search"])
    }
    output = []
    for uid in sorted(missing_uids):
        by_point = triggers.get(uid, {})
        needed = [
            point
            for point in POINTS
            if bool(compatibility.get((uid, point), {}).get("requires_new_search", False))
        ]
        if not needed:
            raise RuntimeError(f"missing UID has no missing pair: {uid}")
        if any(point not in by_point for point in needed):
            raise RuntimeError(f"missing pair lacks a trigger: {uid}")
        metadata = next(iter(by_point.values()))
        trigger_values = {
            point: (
                int(by_point[point]["first_trigger_layer"]) if point in by_point else None
            )
            for point in POINTS
        }
        order = trigger_order_status(trigger_values)
        row: dict[str, Any] = {
            "uid": uid,
            "dataset": str(metadata["dataset"]),
            "source_regime": str(metadata["source_regime"]),
            "dense_correct": False,
            "dense_wrong": True,
            "search_root": min(int(trigger_values[point]) for point in needed),
            "trigger_order_monotonic": order["monotonic"],
            "actual_trigger_order": order["actual_order"],
            "existing_reusable_single": {},
            "existing_reusable_mcts": {},
        }
        for point in POINTS:
            pair = compatibility.get((uid, point), {})
            row[f"trigger_{point}"] = trigger_values[point]
            row[f"needs_search_{point}"] = point in needed
            row["existing_reusable_single"][point] = int(
                pair.get("replay_compatible_single_routes", 0)
            )
            row["existing_reusable_mcts"][point] = int(
                pair.get("replay_compatible_mcts_routes", 0)
            )
        output.append(row)
    return output


def route_structural_points(
    triggers: Mapping[str, int | None], actions: Sequence[str]
) -> dict[str, bool]:
    if len(actions) != 28:
        raise ValueError("route must contain exactly 28 actions")
    changed = [index for index, action in enumerate(actions) if action != "FULL"]
    if not changed:
        raise ValueError("corrective route must contain a non-FULL action")
    first = changed[0]
    return {
        point: layer is not None and int(layer) <= first
        for point, layer in triggers.items()
    }


def next_mcts_root(
    remaining_points: set[str],
    triggers: Mapping[str, int],
    *,
    attempted_roots: set[int],
) -> int | None:
    candidates = sorted(
        {
            int(triggers[point])
            for point in remaining_points
            if int(triggers[point]) not in attempted_roots
        }
    )
    return candidates[0] if candidates else None


def classify_pair_outcome(
    *,
    existing_single: bool,
    existing_mcts: bool,
    new_single: bool,
    new_mcts: bool,
) -> str:
    if existing_single:
        return "EXISTING_SINGLE_REUSED"
    if existing_mcts:
        return "EXISTING_MCTS_REUSED"
    if new_single:
        return "NEW_SINGLE_FIXABLE"
    if new_mcts:
        return "NEW_MCTS_FIXABLE"
    return "UNRESOLVED_AT_BUDGET"


def state_slice(
    *, capture_root: int, threshold_trigger: int, tensor_start: int
) -> tuple[int, int]:
    root, trigger, start = int(capture_root), int(threshold_trigger), int(tensor_start)
    if not 0 <= root <= trigger < 28 or start < 0:
        raise ValueError("invalid capture-root/threshold state slice")
    return start + trigger - root, start + (28 - root)
