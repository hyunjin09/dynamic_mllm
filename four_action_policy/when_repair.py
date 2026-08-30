"""Deterministic iterative W2C route-cache and WHEN-label repair helpers."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from typing import Any, Callable, Mapping, Sequence

from four_action_policy.actions import FOUR_ACTIONS


Route = tuple[str, ...]


def _route(values: Sequence[str], *, expected_layers: int) -> Route:
    route = tuple(str(value) for value in values)
    if len(route) != expected_layers or any(value not in FOUR_ACTIONS for value in route):
        raise ValueError("route violates the frozen four-action width/action contract")
    return route


def _first_nonfull(route: Route) -> int:
    return next((index for index, action in enumerate(route) if action != "FULL"), len(route))


def maximal_full_boundary(
    routes: Sequence[Sequence[str]], *, expected_layers: int = 28
) -> tuple[int, list[int]]:
    """Return the longest supported all-FULL prefix and its route indices."""

    normalized = [_route(route, expected_layers=expected_layers) for route in routes]
    if not normalized:
        raise ValueError("at least one correct route is required")
    lengths = [_first_nonfull(route) for route in normalized]
    boundary = max(lengths)
    return boundary, [index for index, value in enumerate(lengths) if value == boundary]


def build_known_full_candidates(
    routes: Sequence[Sequence[str]], *, boundary: int, expected_layers: int = 28
) -> list[dict[str, Any]]:
    """Force FULL at the boundary for every maximal-prefix correct route."""

    normalized = [_route(route, expected_layers=expected_layers) for route in routes]
    if not 0 <= boundary < expected_layers:
        raise ValueError("known-suffix insertion requires an in-range boundary")
    grouped: dict[Route, list[int]] = {}
    for index, route in enumerate(normalized):
        if _first_nonfull(route) != boundary:
            continue
        candidate = (*route[:boundary], "FULL", *route[boundary + 1 :])
        grouped.setdefault(candidate, []).append(index)
    if not grouped:
        raise ValueError("no correct route supports the candidate boundary")
    return [
        {
            "actions": list(route),
            "route_key": "|".join(route),
            "source_route_indices": indices,
        }
        for route, indices in sorted(grouped.items())
    ]


def _stable(seed: int, *values: object) -> str:
    return sha256(":".join((str(seed), *(str(value) for value in values))).encode()).hexdigest()


def local_suffix_search_plan(
    base_candidates: Sequence[Mapping[str, Any]],
    *,
    boundary: int,
    uid: str,
    seed: int,
    budget: int,
    excluded_routes: set[Route],
    expected_layers: int = 28,
) -> dict[str, Any]:
    """Select a layer-stratified deterministic one-edit suffix neighborhood."""

    if budget < 0 or not 0 <= boundary < expected_layers:
        raise ValueError("local-search budget or boundary is invalid")
    grouped: dict[Route, list[dict[str, Any]]] = {}
    for base_index, base in enumerate(base_candidates):
        route = _route(base["actions"], expected_layers=expected_layers)
        for layer in range(boundary + 1, expected_layers):
            for action in FOUR_ACTIONS:
                if action == route[layer]:
                    continue
                candidate = (*route[:layer], action, *route[layer + 1 :])
                if candidate in excluded_routes:
                    continue
                grouped.setdefault(candidate, []).append(
                    {
                        "base_candidate_index": base_index,
                        "mutated_layer": layer,
                        "source_action": route[layer],
                        "replacement_action": action,
                    }
                )

    rows = []
    for route, provenance in grouped.items():
        provenance.sort(
            key=lambda row: (
                int(row["mutated_layer"]),
                int(row["base_candidate_index"]),
                FOUR_ACTIONS.index(str(row["replacement_action"])),
            )
        )
        primary = provenance[0]
        rows.append(
            {
                "actions": list(route),
                "route_key": "|".join(route),
                "base_candidate_indices": sorted(
                    {int(row["base_candidate_index"]) for row in provenance}
                ),
                "primary_base_candidate_index": int(
                    primary["base_candidate_index"]
                ),
                "mutated_layer": int(primary["mutated_layer"]),
                "source_action": str(primary["source_action"]),
                "replacement_action": str(primary["replacement_action"]),
                "mutation_provenance": provenance,
            }
        )

    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_layer.setdefault(int(row["mutated_layer"]), []).append(row)
    for layer, values in by_layer.items():
        values.sort(key=lambda row: _stable(seed, "variant", uid, boundary, layer, row["route_key"]))
    layers = sorted(
        by_layer,
        key=lambda layer: _stable(seed, "layer", uid, boundary, layer),
    )
    available_candidates = len(rows)
    selected = []
    cursor = 0
    while len(selected) < budget and by_layer:
        layer = layers[cursor % len(layers)]
        values = by_layer[layer]
        if values:
            selected.append(values.pop(0))
        if not values:
            del by_layer[layer]
            layers.remove(layer)
            cursor = 0
        else:
            cursor += 1
    for index, row in enumerate(selected):
        row["candidate_index"] = index
    return {
        "available_candidates": available_candidates,
        "selected_candidates": len(selected),
        "candidates": selected,
    }


def local_suffix_variants(
    base_candidates: Sequence[Mapping[str, Any]],
    *,
    boundary: int,
    uid: str,
    seed: int,
    budget: int,
    excluded_routes: set[Route],
    expected_layers: int = 28,
) -> list[dict[str, Any]]:
    return local_suffix_search_plan(
        base_candidates,
        boundary=boundary,
        uid=uid,
        seed=seed,
        budget=budget,
        excluded_routes=excluded_routes,
        expected_layers=expected_layers,
    )["candidates"]


def _mechanism(actions: Sequence[str]) -> str:
    values = sorted(set(actions), key=FOUR_ACTIONS.index)
    return values[0] if len(values) == 1 else "MULTI"


def repair_w2c_sample(
    source: Mapping[str, Any],
    evaluate: Callable[[Route], Mapping[str, Any]],
    *,
    search_budget: int,
    seed: int,
    expected_layers: int = 28,
) -> dict[str, Any]:
    """Iteratively advance one W2C boundary under the frozen local budget."""

    if source.get("route_type") != "W2C":
        raise ValueError("WHEN repair accepts W2C samples only")
    uid = str(source["uid"])
    correct_routes: dict[Route, dict[str, Any]] = {}
    for original in source["valid_routes"]:
        route = _route(original["actions"], expected_layers=expected_layers)
        correct_routes.setdefault(
            route,
            {
                "uid": uid,
                "actions": list(route),
                "route_key": "|".join(route),
                "source_of_discovery": "original_cache",
                "candidate_boundary": None,
                "source_route_keys": [str(original.get("route_key", "|".join(route)))],
                "source_binary_route_ids": list(original.get("source_binary_route_ids", [])),
            },
        )
    old_boundary, _ = maximal_full_boundary(
        list(correct_routes), expected_layers=expected_layers
    )
    execution_cache: dict[Route, dict[str, Any]] = {}
    history = []
    continue_states = []

    def execute(
        route: Route,
        *,
        stage: str,
        round_index: int,
        boundary: int,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        if route not in execution_cache:
            result = dict(evaluate(route))
            if "correct" not in result:
                raise ValueError("route evaluator omitted correctness")
            execution_cache[route] = {
                "uid": uid,
                "actions": list(route),
                "route_key": "|".join(route),
                "stage": stage,
                "round": round_index,
                "candidate_boundary": boundary,
                **dict(metadata),
                **result,
            }
        return execution_cache[route]

    for round_index in range(expected_layers + 1):
        ordered_routes = sorted(correct_routes)
        boundary, compatible_indices = maximal_full_boundary(
            ordered_routes, expected_layers=expected_layers
        )
        if boundary == expected_layers:
            return {
                "uid": uid,
                "split": str(source["split"]),
                "dataset": str(source["dataset"]),
                "status": "UNRESOLVED",
                "unresolved_reason": "all_full_prefix_reached_for_w2c",
                "old_boundary": old_boundary,
                "new_boundary": boundary,
                "boundary_shift": boundary - old_boundary,
                "new_correct_route_count": len(correct_routes) - len(source["valid_routes"]),
                "continue_states": continue_states,
                "repaired_when_label": {"label": "UNRESOLVED"},
                "repaired_routes": [correct_routes[key] for key in sorted(correct_routes)],
                "route_execution_cache": [execution_cache[key] for key in sorted(execution_cache)],
                "history": history,
            }

        known = build_known_full_candidates(
            ordered_routes, boundary=boundary, expected_layers=expected_layers
        )
        for candidate in known:
            candidate["source_route_keys"] = [
                "|".join(ordered_routes[index]) for index in candidate["source_route_indices"]
            ]
        known_results = [
            execute(
                _route(candidate["actions"], expected_layers=expected_layers),
                stage="known_suffix_repair",
                round_index=round_index,
                boundary=boundary,
                metadata={
                    "source_route_keys": candidate["source_route_keys"],
                    "search_budget": None,
                },
            )
            for candidate in known
        ]
        known_correct = [row for row in known_results if bool(row["correct"])]
        bounded_plan = {
            "available_candidates": 0,
            "selected_candidates": 0,
            "candidates": [],
        }
        bounded = []
        bounded_results = []
        bounded_correct = []
        if not known_correct:
            bounded_plan = local_suffix_search_plan(
                known,
                boundary=boundary,
                uid=uid,
                seed=seed,
                budget=search_budget,
                excluded_routes=set(correct_routes) | set(execution_cache),
                expected_layers=expected_layers,
            )
            bounded = bounded_plan["candidates"]
            bounded_results = [
                execute(
                    _route(candidate["actions"], expected_layers=expected_layers),
                    stage="bounded_continuation_repair",
                    round_index=round_index,
                    boundary=boundary,
                    metadata={
                        "base_candidate_indices": candidate["base_candidate_indices"],
                        "mutation_provenance": candidate["mutation_provenance"],
                        "search_budget": search_budget,
                    },
                )
                for candidate in bounded
            ]
            bounded_correct = [row for row in bounded_results if bool(row["correct"])]

        successful = known_correct or bounded_correct
        history.append(
            {
                "round": round_index,
                "boundary": boundary,
                "compatible_correct_routes": len(compatible_indices),
                "known_candidates": len(known),
                "known_correct": len(known_correct),
                "bounded_available": bounded_plan["available_candidates"],
                "bounded_selected": len(bounded),
                "bounded_correct": len(bounded_correct),
                "outcome": "FULL_RESCUABLE" if successful else "known_and_bounded_exhausted",
            }
        )
        if not successful:
            valid_actions = sorted(
                {route[boundary] for route in ordered_routes if _first_nonfull(route) == boundary},
                key=FOUR_ACTIONS.index,
            )
            state_id = sha256(f"repaired-when:{uid}:{boundary}".encode()).hexdigest()[:24]
            return {
                "uid": uid,
                "split": str(source["split"]),
                "dataset": str(source["dataset"]),
                "status": "FULL_UNRESCUED_UNDER_BUDGET",
                "unresolved_reason": None,
                "old_boundary": old_boundary,
                "new_boundary": boundary,
                "boundary_shift": boundary - old_boundary,
                "old_mechanism": _mechanism(
                    [
                        route[old_boundary]
                        for route in sorted(
                            _route(row["actions"], expected_layers=expected_layers)
                            for row in source["valid_routes"]
                        )
                        if _first_nonfull(route) == old_boundary
                    ]
                ),
                "new_mechanism": _mechanism(valid_actions),
                "new_correct_route_count": len(correct_routes) - len(source["valid_routes"]),
                "continue_states": continue_states,
                "repaired_when_label": {
                    "state_id": state_id,
                    "uid": uid,
                    "split": str(source["split"]),
                    "dataset": str(source["dataset"]),
                    "label": "DEVIATE_CANDIDATE",
                    "target_layer": boundary,
                    "prefix_actions": ["FULL"] * boundary,
                    "known_valid_nonfull_actions": valid_actions,
                    "known_mechanism": _mechanism(valid_actions),
                    "full_status": "FULL_UNRESCUED_UNDER_BUDGET",
                    "bounded_search_budget": search_budget,
                },
                "repaired_routes": [correct_routes[key] for key in sorted(correct_routes)],
                "route_execution_cache": [execution_cache[key] for key in sorted(execution_cache)],
                "history": history,
            }

        discovery = "known_suffix_repair" if known_correct else "bounded_continuation_repair"
        before = boundary
        for row in successful:
            route = _route(row["actions"], expected_layers=expected_layers)
            correct_routes.setdefault(
                route,
                {
                    "uid": uid,
                    "actions": list(route),
                    "route_key": "|".join(route),
                    "source_of_discovery": discovery,
                    "candidate_boundary": boundary,
                    "source_route_keys": list(row.get("source_route_keys", [])),
                    "execution_result": row,
                },
            )
        advanced, _ = maximal_full_boundary(
            list(correct_routes), expected_layers=expected_layers
        )
        if advanced <= before:
            raise RuntimeError("a FULL rescue did not advance the maximal prefix")
        continue_states.append(
            {
                "uid": uid,
                "split": str(source["split"]),
                "dataset": str(source["dataset"]),
                "label": "CONTINUE",
                "boundary": boundary,
                "target_layer": boundary,
                "prefix_actions": ["FULL"] * boundary,
                "verification_source": discovery,
                "verified_correct_routes": len(successful),
            }
        )
    raise RuntimeError("repair exceeded the maximum possible boundary advances")


def select_repair_smoke(
    states: Sequence[Mapping[str, Any]],
    prior_results: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze 12 validation states spanning the plan's smoke dimensions."""

    state_by_uid = {str(row["uid"]): dict(row) for row in states}
    status_by_uid = {str(row["uid"]): str(row["status"]) for row in prior_results}
    if len(state_by_uid) != len(states) or set(state_by_uid) != set(status_by_uid):
        raise ValueError("smoke inputs must have identical unique UID coverage")
    desired = {
        ("FULL-cache-incomplete", "single"): "early",
        ("FULL-cache-incomplete", "multi"): "middle",
        ("FULL-confirmed-invalid", "single"): "late",
        ("FULL-confirmed-invalid", "multi"): "early",
    }
    selected = []
    for dataset in ("chartqa", "gqa", "textvqa"):
        for (status, suffix_class), depth in desired.items():
            eligible = []
            for uid, state in state_by_uid.items():
                current_suffix = (
                    "single"
                    if int(
                        state.get(
                            "compatible_suffix_count", state["candidate_route_count"]
                        )
                    )
                    == 1
                    else "multi"
                )
                if (
                    str(state["dataset"]) == dataset
                    and status_by_uid[uid] == status
                    and current_suffix == suffix_class
                    and str(state["depth_bin"]) == depth
                ):
                    eligible.append(state)
            if not eligible:
                raise ValueError(
                    f"no smoke candidate for {dataset}/{status}/{suffix_class}/{depth}"
                )
            chosen = min(
                eligible,
                key=lambda row: _stable(
                    seed,
                    "smoke",
                    dataset,
                    status,
                    suffix_class,
                    depth,
                    row["uid"],
                ),
            )
            selected.append(
                {
                    **chosen,
                    "prior_status": status,
                    "suffix_class": suffix_class,
                }
            )
    if len(selected) != 12 or len({row["uid"] for row in selected}) != 12:
        raise RuntimeError("smoke selection is not a 12-UID cohort")
    selected.sort(key=lambda row: str(row["uid"]))
    audit = {
        "records": len(selected),
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in selected).items())),
        "prior_status_counts": dict(
            sorted(Counter(row["prior_status"] for row in selected).items())
        ),
        "suffix_class_counts": dict(
            sorted(Counter(row["suffix_class"] for row in selected).items())
        ),
        "depth_counts": dict(sorted(Counter(row["depth_bin"] for row in selected).items())),
    }
    return selected, audit


def assign_cost_balanced_shards(
    rows: Sequence[Mapping[str, Any]], *, world_size: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically assign records with longest-processing-time sharding."""

    if world_size < 1 or not rows:
        raise ValueError("sharding requires records and a positive world size")
    if len({str(row["uid"]) for row in rows}) != len(rows):
        raise ValueError("sharding rows contain duplicate UIDs")
    ordered = sorted(
        rows,
        key=lambda row: (-int(row["estimated_cost"]), str(row["uid"])),
    )
    rank_cost = [0] * world_size
    rank_count = [0] * world_size
    assigned = []
    for row in ordered:
        rank = min(range(world_size), key=lambda value: (rank_cost[value], rank_count[value], value))
        assigned.append({**dict(row), "rank": rank})
        rank_cost[rank] += int(row["estimated_cost"])
        rank_count[rank] += 1
    assigned.sort(key=lambda row: (int(row["rank"]), -int(row["estimated_cost"]), str(row["uid"])))
    return assigned, {
        "world_size": world_size,
        "records": len(assigned),
        "rank_records": {str(rank): rank_count[rank] for rank in range(world_size)},
        "rank_estimated_cost": {str(rank): rank_cost[rank] for rank in range(world_size)},
    }
