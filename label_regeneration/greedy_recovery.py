"""Pure, deterministic search logic for WeMath greedy route recovery.

The immutable implementation under ``search/greedy_phase1_phase2_reproduction``
is the algorithmic reference.  This module copies only its search geometry so
the active Transformers 5.3.0 executor can be used without importing the old
runtime.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any


PHASE1_ORDERS = (
    "early_to_late",
    "late_to_early",
    "center_out",
    "outside_in",
    "random:20260714",
    "random:20260715",
    "random:20260716",
    "random:20260717",
    "random:20260718",
    "random:20260719",
)
SCORE_TOLERANCE = 1e-9


def route_key(route: list[int] | tuple[int, ...]) -> str:
    return "".join("1" if int(value) else "0" for value in route)


def route_id(uid: str, route: list[int] | tuple[int, ...]) -> str:
    digest = hashlib.sha256(f"{uid}:{route_key(route)}".encode("utf-8")).hexdigest()[:16]
    return f"{uid}:mask:{digest}"


def layer_order(name: str, num_layers: int, uid: str) -> list[int]:
    if name == "early_to_late":
        return list(range(num_layers))
    if name == "late_to_early":
        return list(range(num_layers - 1, -1, -1))
    if name == "center_out":
        left = (num_layers - 1) // 2
        right = num_layers // 2
        order: list[int] = []
        while left >= 0 or right < num_layers:
            if left >= 0:
                order.append(left)
            if right < num_layers and right != left:
                order.append(right)
            left -= 1
            right += 1
        return order
    if name == "outside_in":
        order = []
        left, right = 0, num_layers - 1
        while left <= right:
            order.append(left)
            if right != left:
                order.append(right)
            left += 1
            right -= 1
        return order
    if name.startswith("random:"):
        seed = int(name.split(":", 1)[1])
        order = list(range(num_layers))
        random.Random(f"{seed}:{uid}").shuffle(order)
        return order
    raise ValueError(f"unsupported order: {name}")


def acceptance_decision(
    candidate_score: float,
    *,
    all_on_score: float,
    current_score: float,
    tolerance: float = SCORE_TOLERANCE,
) -> bool:
    return float(candidate_score) + float(tolerance) >= max(
        float(all_on_score), float(current_score)
    )


def choose_success_bases(payload: dict[str, Any], max_bases: int = 3) -> list[list[int]]:
    execution_by_id = {
        str(row["route_id"]): list(row["visual_on_mask"])
        for row in payload["candidate_executions"]
    }
    by_key = {
        route_key(execution_by_id[str(final["final_route_id"])]): execution_by_id[
            str(final["final_route_id"])
        ]
        for final in payload["permutation_finals"]
        if final["final_correct"]
    }
    candidates = sorted(by_key.values(), key=lambda route: (sum(route), route_key(route)))
    if len(candidates) <= max_bases:
        return candidates
    selected = [candidates[0]]
    while len(selected) < max_bases:
        remaining = [route for route in candidates if route not in selected]
        item = min(
            remaining,
            key=lambda route: (
                -max(sum(a != b for a, b in zip(route, chosen)) for chosen in selected),
                route_key(route),
            ),
        )
        selected.append(item)
    return selected


def _add_candidate(plan: dict[str, dict[str, Any]], route: list[int], origin: dict[str, Any]) -> None:
    key = route_key(route)
    if key not in plan:
        plan[key] = {"route": list(route), "origins": []}
    if origin not in plan[key]["origins"]:
        plan[key]["origins"].append(origin)


def candidate_plan(
    payload: dict[str, Any], budget_center: int, args: Any
) -> dict[str, dict[str, Any]]:
    uid = str(payload["sample"]["uid"])
    num_layers = int(payload["runtime"]["num_layers"])
    rng = random.Random(f"{args.seed}:{uid}")
    plan: dict[str, dict[str, Any]] = {}

    for budget in sorted(
        {max(0, min(num_layers, int(budget_center) + delta)) for delta in (-2, 0, 2)}
    ):
        for index in range(int(args.random_per_budget)):
            chosen = set(rng.sample(range(num_layers), budget))
            route = [1 if layer in chosen else 0 for layer in range(num_layers)]
            _add_candidate(
                plan,
                route,
                {
                    "family": "budget_stratified_random",
                    "budget": budget,
                    "draw": index,
                    "seed": int(args.seed),
                },
            )

    bases = choose_success_bases(payload)
    limit = int(args.local_per_operation)
    for base_index, base in enumerate(bases):
        on = [idx for idx, value in enumerate(base) if value]
        off = [idx for idx, value in enumerate(base) if not value]
        for index in range(min(limit, len(on) * len(off))):
            if not on or not off:
                break
            remove_idx = rng.choice(on)
            add_idx = rng.choice(off)
            route = list(base)
            route[remove_idx] = 0
            route[add_idx] = 1
            _add_candidate(
                plan,
                route,
                {"family": "same_budget_swap", "base_index": base_index, "draw": index},
            )
        for index, add_idx in enumerate(rng.sample(off, min(limit, len(off)))):
            route = list(base)
            route[add_idx] = 1
            _add_candidate(
                plan,
                route,
                {
                    "family": "add_one",
                    "base_index": base_index,
                    "layer_one_based": add_idx + 1,
                    "draw": index,
                },
            )
        for index, remove_idx in enumerate(rng.sample(on, min(limit, len(on)))):
            route = list(base)
            route[remove_idx] = 0
            _add_candidate(
                plan,
                route,
                {
                    "family": "remove_one",
                    "base_index": base_index,
                    "layer_one_based": remove_idx + 1,
                    "draw": index,
                },
            )

    for left in range(len(bases)):
        for right in range(left + 1, len(bases)):
            _add_candidate(
                plan,
                [int(a or b) for a, b in zip(bases[left], bases[right])],
                {"family": "success_union", "base_pair": [left, right]},
            )
            _add_candidate(
                plan,
                [int(a and b) for a, b in zip(bases[left], bases[right])],
                {"family": "success_intersection", "base_pair": [left, right]},
            )
    return plan


def select_diverse_valid_routes(
    routes: list[dict[str, Any]], *, max_routes: int = 50
) -> list[dict[str, Any]]:
    """Build a deterministic diverse training view without truncating raw cache."""
    unique = {
        str(row.get("mask_key") or route_key(row["visual_on_mask"])): row
        for row in routes
        if bool(row.get("result_correct"))
    }
    candidates = sorted(
        unique.values(),
        key=lambda row: (
            int(row.get("num_visual_on_layers", sum(row["visual_on_mask"]))),
            str(row.get("mask_key") or route_key(row["visual_on_mask"])),
        ),
    )
    if len(candidates) <= max_routes:
        return candidates
    selected = [candidates[0]]
    while len(selected) < max_routes:
        remaining = [row for row in candidates if row not in selected]
        next_row = min(
            remaining,
            key=lambda row: (
                -min(
                    sum(
                        int(a) != int(b)
                        for a, b in zip(row["visual_on_mask"], chosen["visual_on_mask"])
                    )
                    for chosen in selected
                ),
                int(row.get("num_visual_on_layers", sum(row["visual_on_mask"]))),
                str(row.get("mask_key") or route_key(row["visual_on_mask"])),
            ),
        )
        selected.append(next_row)
    return selected
