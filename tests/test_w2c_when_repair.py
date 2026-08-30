from __future__ import annotations

from four_action_policy.when_repair import (
    assign_cost_balanced_shards,
    build_known_full_candidates,
    local_suffix_search_plan,
    local_suffix_variants,
    maximal_full_boundary,
    repair_w2c_sample,
    select_repair_smoke,
)


def _source(uid: str, routes: list[list[str]]) -> dict:
    return {
        "uid": uid,
        "split": "validation",
        "dataset": "gqa",
        "route_type": "W2C",
        "valid_routes": [
            {
                "actions": route,
                "route_key": "|".join(route),
                "source_binary_route_ids": [f"source-{index}"],
            }
            for index, route in enumerate(routes)
        ],
    }


def test_maximal_boundary_and_known_candidates_use_only_maximal_prefix_routes() -> None:
    routes = [
        ["IGNORE", "FULL", "FULL", "FULL"],
        ["FULL", "READ_ONLY", "FULL", "FULL"],
        ["FULL", "WRITE_ONLY", "FULL", "FULL"],
    ]

    boundary, compatible = maximal_full_boundary(routes, expected_layers=4)
    candidates = build_known_full_candidates(
        routes, boundary=boundary, expected_layers=4
    )

    assert boundary == 1
    assert compatible == [1, 2]
    assert len(candidates) == 1
    assert candidates[0]["actions"] == ["FULL"] * 4
    assert candidates[0]["source_route_indices"] == [1, 2]


def test_local_suffix_variants_are_budgeted_deterministic_and_one_edit() -> None:
    bases = [
        {
            "actions": ["FULL", "FULL", "IGNORE", "FULL"],
            "source_route_indices": [0],
        },
        {
            "actions": ["FULL", "FULL", "WRITE_ONLY", "IGNORE"],
            "source_route_indices": [1],
        },
    ]

    first = local_suffix_variants(
        bases,
        boundary=1,
        uid="u0",
        seed=17,
        budget=4,
        excluded_routes=set(),
        expected_layers=4,
    )
    second = local_suffix_variants(
        bases,
        boundary=1,
        uid="u0",
        seed=17,
        budget=4,
        excluded_routes=set(),
        expected_layers=4,
    )

    assert first == second
    assert len(first) == 4
    assert {row["mutated_layer"] for row in first} == {2, 3}
    for row in first:
        base = bases[row["primary_base_candidate_index"]]["actions"]
        differences = [
            index for index, (left, right) in enumerate(zip(base, row["actions"]))
            if left != right
        ]
        assert differences == [row["mutated_layer"]]
        assert row["mutated_layer"] > 1

    plan = local_suffix_search_plan(
        bases,
        boundary=1,
        uid="u0",
        seed=17,
        budget=4,
        excluded_routes=set(),
        expected_layers=4,
    )
    assert plan["available_candidates"] == 10
    assert plan["selected_candidates"] == 4
    assert plan["candidates"] == first


def test_known_suffix_rescue_advances_boundary_until_bounded_failure() -> None:
    source = _source("u0", [["FULL", "IGNORE", "FULL", "IGNORE"]])
    correct = {
        ("FULL", "IGNORE", "FULL", "IGNORE"),
        ("FULL", "FULL", "FULL", "IGNORE"),
    }

    result = repair_w2c_sample(
        source,
        lambda route: {"correct": tuple(route) in correct, "prediction": "x"},
        search_budget=8,
        seed=19,
        expected_layers=4,
    )

    assert result["status"] == "FULL_UNRESCUED_UNDER_BUDGET"
    assert result["old_boundary"] == 1
    assert result["new_boundary"] == 3
    assert result["boundary_shift"] == 2
    assert result["new_correct_route_count"] == 1
    assert [row["boundary"] for row in result["continue_states"]] == [1]
    assert result["repaired_when_label"]["label"] == "DEVIATE_CANDIDATE"


def test_local_search_rescue_is_added_and_restarts_boundary_iteration() -> None:
    source = _source("u1", [["FULL", "IGNORE", "IGNORE", "FULL"]])
    correct = {
        ("FULL", "IGNORE", "IGNORE", "FULL"),
        ("FULL", "FULL", "READ_ONLY", "FULL"),
    }

    result = repair_w2c_sample(
        source,
        lambda route: {"correct": tuple(route) in correct, "prediction": "x"},
        search_budget=32,
        seed=23,
        expected_layers=4,
    )

    assert result["status"] == "FULL_UNRESCUED_UNDER_BUDGET"
    assert result["old_boundary"] == 1
    assert result["new_boundary"] == 2
    assert result["new_correct_route_count"] == 1
    discovered = [
        row for row in result["repaired_routes"] if row["source_of_discovery"] != "original_cache"
    ]
    assert discovered[0]["source_of_discovery"] == "bounded_continuation_repair"
    assert result["history"][0]["known_correct"] == 0
    assert result["history"][0]["bounded_correct"] == 1


def test_repair_caches_duplicate_route_evaluations_within_sample() -> None:
    source = _source(
        "u2",
        [
            ["FULL", "IGNORE", "FULL", "FULL"],
            ["FULL", "READ_ONLY", "FULL", "FULL"],
        ],
    )
    calls: list[tuple[str, ...]] = []

    def evaluate(route: tuple[str, ...]) -> dict:
        calls.append(route)
        return {"correct": False, "prediction": "x"}

    result = repair_w2c_sample(
        source,
        evaluate,
        search_budget=0,
        seed=29,
        expected_layers=4,
    )

    assert result["status"] == "FULL_UNRESCUED_UNDER_BUDGET"
    assert calls == [("FULL", "FULL", "FULL", "FULL")]
    assert len(result["route_execution_cache"]) == 1


def test_smoke_selection_freezes_dataset_status_suffix_and_depth_coverage() -> None:
    desired = {
        ("FULL-cache-incomplete", "single"): "early",
        ("FULL-cache-incomplete", "multi"): "middle",
        ("FULL-confirmed-invalid", "single"): "late",
        ("FULL-confirmed-invalid", "multi"): "early",
    }
    states = []
    results = []
    for dataset in ("chartqa", "gqa", "textvqa"):
        for index, ((status, suffix_class), depth) in enumerate(desired.items()):
            uid = f"{dataset}-{index}"
            states.append(
                {
                    "uid": uid,
                    "dataset": dataset,
                    "depth_bin": depth,
                    "candidate_route_count": 1 if suffix_class == "single" else 2,
                }
            )
            results.append({"uid": uid, "status": status})

    selected, audit = select_repair_smoke(states, results, seed=31)

    assert len(selected) == 12
    assert audit["dataset_counts"] == {"chartqa": 4, "gqa": 4, "textvqa": 4}
    assert audit["prior_status_counts"] == {
        "FULL-cache-incomplete": 6,
        "FULL-confirmed-invalid": 6,
    }
    assert audit["suffix_class_counts"] == {"multi": 6, "single": 6}
    assert set(audit["depth_counts"]) == {"early", "middle", "late"}


def test_cost_balanced_shards_are_deterministic_and_cover_each_uid_once() -> None:
    rows = [
        {"uid": f"u{index}", "estimated_cost": cost}
        for index, cost in enumerate([10, 9, 8, 7, 6, 5, 4, 3])
    ]

    first, audit = assign_cost_balanced_shards(rows, world_size=4)
    second, _ = assign_cost_balanced_shards(rows, world_size=4)

    assert first == second
    assert {row["uid"] for row in first} == {row["uid"] for row in rows}
    assert len({row["uid"] for row in first}) == len(rows)
    assert audit["rank_records"] == {"0": 2, "1": 2, "2": 2, "3": 2}
    assert max(audit["rank_estimated_cost"].values()) - min(
        audit["rank_estimated_cost"].values()
    ) <= 1
