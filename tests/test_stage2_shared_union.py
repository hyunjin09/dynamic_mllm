from __future__ import annotations

import random

import pytest

from dense_failure_stage2.shared_union import (
    balanced_uid_draws,
    merge_threshold_routes,
    sample_mcts_layers,
    sample_single_layers,
    trigger_layer,
)


def _route(point: str, route_id: str = "r", route_type: str = "single"):
    actions = ["FULL"] * 28
    if route_type == "single":
        actions[22] = "IGNORE"
    elif route_type == "mcts":
        actions[20] = "READ_ONLY"
        actions[23] = "WRITE_ONLY"
    return {
        "route_id": route_id,
        "uid": "u",
        "dataset": "gqa",
        "source_regime": "historical",
        "route_type": route_type,
        "operating_point": point,
        "trigger_layer": {"P98": 21, "P95": 18, "P90": 16}[point],
        "actions": actions,
    }


def test_threshold_expansion_is_deduplicated_and_keeps_validity():
    rows = merge_threshold_routes(
        {"P98": [_route("P98")], "P95": [_route("P95")], "P90": [_route("P90")]},
        expected_route_type="single",
    )
    assert len(rows) == 1
    assert rows[0]["activation_layer"] == 16
    assert rows[0]["valid_operating_points"] == ["P98", "P95", "P90"]
    assert rows[0]["valid_for_P98"] and rows[0]["valid_for_P95"] and rows[0]["valid_for_P90"]


def test_route_identity_mismatch_fails_closed():
    bad = _route("P95")
    bad["actions"][21] = "READ_ONLY"
    with pytest.raises(ValueError, match="identity differs"):
        merge_threshold_routes(
            {"P98": [_route("P98")], "P95": [bad]}, expected_route_type="single"
        )


def test_balanced_uid_draws_are_deterministic_and_nearly_equal():
    first = balanced_uid_draws(["a", "b", "c"], 10, random.Random(4))
    second = balanced_uid_draws(["a", "b", "c"], 10, random.Random(4))
    assert first == second
    counts = {uid: first.count(uid) for uid in set(first)}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_single_random4_uses_union_activation_and_mandatory_correction():
    merged = merge_threshold_routes(
        {"P90": [_route("P90")]}, expected_route_type="single"
    )[0]
    layers = sample_single_layers(merged, random.Random(9))
    assert len(layers) == 4
    assert min(layers) >= 16
    assert 22 in layers
    assert any(layer < 22 for layer in layers)
    assert any(layer > 22 for layer in layers)


def test_mcts_sampler_keeps_all_corrective_states_and_caps_context():
    merged = merge_threshold_routes(
        {"P90": [_route("P90", route_type="mcts")]}, expected_route_type="mcts"
    )[0]
    layers = sample_mcts_layers(merged, random.Random(2), state_cap=6)
    assert 20 in layers and 23 in layers
    assert len(layers) == 6
    assert min(layers) >= 16


def test_mcts_sampler_retains_five_action_route_when_below_cap():
    route = _route("P90", route_type="mcts")
    for layer in (17, 19, 21):
        route["actions"][layer] = "IGNORE"
    merged = merge_threshold_routes({"P90": [route]}, expected_route_type="mcts")[0]
    layers = sample_mcts_layers(merged, random.Random(3), state_cap=8)
    assert {17, 19, 20, 21, 23}.issubset(layers)
    assert len(layers) == 8


def test_trigger_uses_strict_greater_than():
    row = {f"p_{layer}": 0.1 for layer in range(28)}
    row[4] = 0.5
    row["p_4"] = 0.5
    row["p_7"] = 0.50001
    assert trigger_layer(row, 0.5) == 7
