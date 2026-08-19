import argparse

from label_regeneration.greedy_recovery import (
    acceptance_decision,
    candidate_plan,
    choose_success_bases,
    layer_order,
    select_diverse_valid_routes,
)


def _phase1_payload(masks):
    executions = []
    finals = []
    for index, mask in enumerate(masks):
        route_id = f"r{index}"
        executions.append({"route_id": route_id, "visual_on_mask": mask})
        finals.append(
            {
                "final_route_id": route_id,
                "final_correct": True,
                "final_num_visual_on_layers": sum(mask),
            }
        )
    return {
        "sample": {"uid": "wemath2pro:7"},
        "runtime": {"num_layers": len(masks[0])},
        "candidate_executions": executions,
        "permutation_finals": finals,
    }


def test_frozen_phase1_orders_and_acceptance():
    assert layer_order("early_to_late", 4, "x") == [0, 1, 2, 3]
    assert layer_order("late_to_early", 4, "x") == [3, 2, 1, 0]
    assert layer_order("center_out", 4, "x") == [1, 2, 0, 3]
    assert layer_order("outside_in", 4, "x") == [0, 3, 1, 2]
    assert sorted(layer_order("random:20260714", 28, "wemath2pro:1")) == list(range(28))
    assert layer_order("random:20260714", 28, "wemath2pro:1") == layer_order(
        "random:20260714", 28, "wemath2pro:1"
    )
    assert acceptance_decision(0.0, all_on_score=0.0, current_score=0.0)
    assert not acceptance_decision(0.0, all_on_score=0.0, current_score=1.0)
    assert acceptance_decision(1.0, all_on_score=0.0, current_score=1.0)


def test_phase2_plan_matches_frozen_request_families_and_deduplicates():
    payload = _phase1_payload(
        [
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
        ]
    )
    bases = choose_success_bases(payload)
    assert len(bases) == 3
    args = argparse.Namespace(seed=20260720, random_per_budget=2, local_per_operation=4)
    plan = candidate_plan(payload, budget_center=2, args=args)
    assert len(plan) <= 48
    families = {origin["family"] for item in plan.values() for origin in item["origins"]}
    assert {
        "budget_stratified_random",
        "same_budget_swap",
        "add_one",
        "remove_one",
        "success_union",
        "success_intersection",
    } <= families
    assert len(plan) == len(set(plan))


def test_diverse_valid_route_view_keeps_all_up_to_50_and_caps_deterministically():
    routes = []
    for value in range(64):
        mask = [int(bit) for bit in f"{value:028b}"]
        routes.append({"mask_key": "".join(map(str, mask)), "visual_on_mask": mask, "result_correct": True})
    first = select_diverse_valid_routes(routes, max_routes=50)
    second = select_diverse_valid_routes(list(reversed(routes)), max_routes=50)
    assert len(first) == 50
    assert [row["mask_key"] for row in first] == [row["mask_key"] for row in second]
    assert len(select_diverse_valid_routes(routes[:12], max_routes=50)) == 12
