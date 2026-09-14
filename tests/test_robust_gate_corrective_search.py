from __future__ import annotations

import pytest

from dense_failure_stage2.robust_gate_search import (
    POINTS,
    build_missing_union_rows,
    classify_pair_outcome,
    next_mcts_root,
    route_structural_points,
    state_slice,
    trigger_order_status,
)
from experiments.run_robust_gate_corrective_search import _existing_route, _new_route


def _trigger(uid: str, point: str, layer: int, *, source: str = "historical") -> dict:
    return {
        "uid": uid,
        "dataset": "gqa",
        "source_regime": source,
        "dense_correct": False,
        "dense_wrong": True,
        "threshold_name": point,
        "triggered": True,
        "first_trigger_layer": layer,
    }


def test_missing_union_collapses_pairs_and_preserves_existing_routes() -> None:
    triggers = [_trigger("u1", "P98", 20), _trigger("u1", "P95", 15), _trigger("u1", "P90", 8)]
    compatibility = [
        {"uid": "u1", "operating_point": "P98", "requires_new_search": True,
         "replay_compatible_single_routes": 0, "replay_compatible_mcts_routes": 0},
        {"uid": "u1", "operating_point": "P95", "requires_new_search": False,
         "replay_compatible_single_routes": 2, "replay_compatible_mcts_routes": 1},
        {"uid": "u1", "operating_point": "P90", "requires_new_search": True,
         "replay_compatible_single_routes": 0, "replay_compatible_mcts_routes": 0},
    ]
    rows = build_missing_union_rows(triggers, compatibility)
    assert len(rows) == 1
    assert rows[0]["search_root"] == 8
    assert rows[0]["needs_search_P98"] is True
    assert rows[0]["needs_search_P95"] is False
    assert rows[0]["needs_search_P90"] is True
    assert rows[0]["existing_reusable_single"]["P95"] == 2
    assert rows[0]["existing_reusable_mcts"]["P95"] == 1


def test_trigger_order_is_reported_but_actual_minimum_is_used() -> None:
    good = trigger_order_status({"P98": 20, "P95": 15, "P90": 8})
    assert good["monotonic"] is True
    assert good["actual_order"] == ["P90", "P95", "P98"]
    bad = trigger_order_status({"P98": 12, "P95": 15, "P90": 8})
    assert bad["monotonic"] is False
    assert bad["actual_order"] == ["P90", "P98", "P95"]


def test_route_structural_points_use_first_non_full_layer() -> None:
    actions = ["FULL"] * 28
    actions[17] = "WRITE_ONLY"
    flags = route_structural_points({"P98": 20, "P95": 15, "P90": 8}, actions)
    assert flags == {"P98": False, "P95": True, "P90": True}


def test_next_mcts_root_uses_earliest_unattempted_actual_trigger() -> None:
    remaining = {"P98", "P95", "P90"}
    triggers = {"P98": 20, "P95": 15, "P90": 8}
    assert next_mcts_root(remaining, triggers, attempted_roots=set()) == 8
    assert next_mcts_root(remaining, triggers, attempted_roots={8}) == 15
    assert next_mcts_root({"P90"}, triggers, attempted_roots={8}) is None


@pytest.mark.parametrize(
    ("existing_single", "existing_mcts", "new_single", "new_mcts", "expected"),
    [
        (True, True, True, True, "EXISTING_SINGLE_REUSED"),
        (False, True, True, True, "EXISTING_MCTS_REUSED"),
        (False, False, True, True, "NEW_SINGLE_FIXABLE"),
        (False, False, False, True, "NEW_MCTS_FIXABLE"),
        (False, False, False, False, "UNRESOLVED_AT_BUDGET"),
    ],
)
def test_pair_outcome_precedence(existing_single, existing_mcts, new_single, new_mcts, expected) -> None:
    assert classify_pair_outcome(
        existing_single=existing_single,
        existing_mcts=existing_mcts,
        new_single=new_single,
        new_mcts=new_mcts,
    ) == expected


def test_state_slice_is_explicitly_bound_to_threshold_trigger() -> None:
    assert state_slice(capture_root=8, threshold_trigger=15, tensor_start=100) == (107, 120)
    assert state_slice(capture_root=8, threshold_trigger=8, tensor_start=100) == (100, 120)
    with pytest.raises(ValueError):
        state_slice(capture_root=15, threshold_trigger=8, tensor_start=100)


def test_point_order_is_frozen() -> None:
    assert POINTS == ("P98", "P95", "P90")


def test_new_route_accepts_cached_terminal_record_shape() -> None:
    actions = ["FULL"] * 28
    actions[12] = "READ_ONLY"
    route = _new_route(
        contract={"contract_sha256": "contract"},
        uid="u1",
        dataset="gqa",
        source_regime="canonical",
        origin="new_single",
        route_type="single",
        search_root=8,
        actions=actions,
        state={
            "generated_token_ids": [7, 8],
            "generated_answer": "yes",
            "lmms_metric": "exact_match",
            "lmms_score": 1.0,
            "correct": True,
        },
        discovery={"discovery_order": 1},
    )
    assert route["generated_token_ids"] == [7, 8]
    assert route["correct"] is True


def test_existing_route_identity_comes_from_current_work_row() -> None:
    actions = ["FULL"] * 28
    actions[20] = "IGNORE"
    route = _existing_route(
        {
            "route_id": "r1",
            "route_key": "|".join(actions),
            "route_type": "single",
            "route_origin": "existing_single",
            "actions": actions,
            "expected_generated_token_ids": [9],
            "expected_generated_answer": "2",
            "expected_lmms_metric": "exact_match",
            "expected_lmms_score": 1.0,
            "valid_points": ["P90"],
            "source_contract_sha256": "source",
        },
        {"contract_sha256": "contract"},
        uid="u1",
        dataset="chartqa",
        source_regime="historical",
    )
    assert (route["uid"], route["dataset"], route["source_regime"]) == (
        "u1",
        "chartqa",
        "historical",
    )
