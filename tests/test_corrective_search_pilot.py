from __future__ import annotations

from collections import Counter

import pytest

from dense_failure_stage2.corrective_search import (
    ACTIONS,
    NON_FULL_ACTIONS,
    all_single_routes,
    classify_outcome,
    pilot_cell_weights,
    run_sequential_mcts,
    select_pilot_manifest,
    validate_complete_results,
)


def _candidate(dataset: str, layer: int, index: int) -> dict:
    return {
        "uid": f"{dataset}:{layer}:{index}",
        "dataset": dataset,
        "first_trigger_layer": layer,
        "group_id": f"sha256:{dataset}:{layer}:{index}",
        "split": "train",
        "dense_wrong": True,
    }


def test_select_pilot_manifest_obeys_frozen_cell_allocation_and_groups():
    rows = []
    for dataset in ("gqa", "chartqa", "textvqa"):
        for layer in (0, 4, 12, 22):
            rows.extend(_candidate(dataset, layer, index) for index in range(30))
    allocation = {
        "gqa": {"L0": 3, "L1-8": 12, "L9-18": 12, "L19-27": 13},
        "chartqa": {"L0": 10, "L1-8": 10, "L9-18": 10, "L19-27": 10},
        "textvqa": {"L0": 10, "L1-8": 10, "L9-18": 10, "L19-27": 10},
    }

    selected = select_pilot_manifest(rows, allocation=allocation, seed=20260831)

    assert len(selected) == 120
    assert len({row["uid"] for row in selected}) == 120
    assert len({row["group_id"] for row in selected}) == 120
    actual = Counter((row["dataset"], row["trigger_depth_bin"]) for row in selected)
    assert actual == Counter(
        (dataset, depth_bin)
        for dataset, by_depth in allocation.items()
        for depth_bin, count in by_depth.items()
        for _ in range(count)
    )
    assert selected == select_pilot_manifest(rows, allocation=allocation, seed=20260831)


def test_select_pilot_manifest_fails_when_distinct_groups_are_insufficient():
    rows = [_candidate("gqa", 0, index) for index in range(3)]
    rows[1]["group_id"] = rows[0]["group_id"]
    with pytest.raises(ValueError, match="distinct image groups"):
        select_pilot_manifest(rows, allocation={"gqa": {"L0": 3}}, seed=7)


def test_all_single_routes_is_exhaustive_and_never_changes_dense_prefix():
    routes = all_single_routes(start_layer=25)

    assert len(routes) == 9
    assert len({row["route_key"] for row in routes}) == 9
    assert all(row["actions"][:25] == ["FULL"] * 25 for row in routes)
    assert Counter(row["changed_actions"][0] for row in routes) == Counter(
        {action: 3 for action in NON_FULL_ACTIONS}
    )


def test_sequential_mcts_is_deterministic_cardinality_stratified_and_binary_reward():
    seen: list[tuple[str, ...]] = []

    def evaluate(actions):
        route = tuple(actions)
        seen.append(route)
        return route[8] == "READ_ONLY" and route[11] == "IGNORE"

    first = run_sequential_mcts(
        uid="gqa:sample",
        start_layer=5,
        seed=19,
        max_iterations=40,
        extra_iterations_after_success=7,
        evaluate=evaluate,
    )
    second_seen: list[tuple[str, ...]] = []
    second = run_sequential_mcts(
        uid="gqa:sample",
        start_layer=5,
        seed=19,
        max_iterations=40,
        extra_iterations_after_success=7,
        evaluate=lambda actions: second_seen.append(tuple(actions)) or (
            actions[8] == "READ_ONLY" and actions[11] == "IGNORE"
        ),
    )

    assert first == second
    assert seen == second_seen
    assert first["iterations"] <= 40
    assert all(row["reward"] in (0, 1) for row in first["search_rows"])
    assert all(row["actions"][:5] == ["FULL"] * 5 for row in first["search_rows"])
    assert [row["rollout_target_non_full"] for row in first["search_rows"][:6]] == [
        2,
        3,
        4,
        2,
        3,
        4,
    ]
    assert all(
        sum(action != "FULL" for action in row["actions"]) >= row["rollout_target_non_full"]
        for row in first["search_rows"]
    )
    if first["first_success_iteration"] is not None:
        assert first["iterations"] == min(40, first["first_success_iteration"] + 7)


def test_mcts_tracks_cached_duplicate_routes_without_relabeling_reward():
    physical = Counter()

    def evaluate(actions):
        key = tuple(actions)
        physical[key] += 1
        return False

    result = run_sequential_mcts(
        uid="late",
        start_layer=27,
        seed=2,
        max_iterations=20,
        extra_iterations_after_success=2,
        evaluate=evaluate,
    )

    assert result["iterations"] == 20
    assert result["unique_terminal_routes"] == 4
    assert len(physical) == 4
    assert all(value == 1 for value in physical.values())
    assert sum(row["terminal_cache_hit"] for row in result["search_rows"]) == 16


def test_classification_and_cell_weighting_are_predeclared():
    assert classify_outcome(single_successes=[{"route_key": "a"}], mcts_result=None) == "SINGLE-FIXABLE"
    assert classify_outcome(single_successes=[], mcts_result={"first_success_iteration": 17}) == "MCTS-FIXABLE"
    assert classify_outcome(single_successes=[], mcts_result={"first_success_iteration": None}) == "UNRESOLVED"

    population = Counter({("gqa", "L0"): 3, ("gqa", "L1-8"): 57})
    weights = pilot_cell_weights(population)
    assert weights[("gqa", "L0")] == pytest.approx(0.05)
    assert weights[("gqa", "L1-8")] == pytest.approx(0.95)


def test_global_completion_rejects_missing_duplicate_or_failed_uid():
    expected = ["a", "b"]
    validate_complete_results(expected, [{"uid": "a", "passed": True}, {"uid": "b", "passed": True}])
    with pytest.raises(ValueError, match="coverage"):
        validate_complete_results(expected, [{"uid": "a", "passed": True}])
    with pytest.raises(ValueError, match="coverage"):
        validate_complete_results(expected, [{"uid": "a", "passed": True}, {"uid": "a", "passed": True}])
    with pytest.raises(ValueError, match="failed"):
        validate_complete_results(expected, [{"uid": "a", "passed": True}, {"uid": "b", "passed": False}])


def test_action_order_is_frozen():
    assert ACTIONS == ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")
