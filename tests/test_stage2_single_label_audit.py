from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from dense_failure_stage2.single_label_audit import (
    classify_immediacy,
    sample_weighted_action_distribution,
    same_layer_action_ambiguity,
    semantic_state_id,
    simulate_sampling_schemes,
    validate_single_routes,
)


def _route(
    uid: str,
    *,
    trigger: int,
    intervention: int,
    action: str,
    route_id: str,
) -> dict:
    actions = ["FULL"] * 28
    actions[intervention] = action
    return {
        "uid": uid,
        "dataset": "gqa",
        "trigger_layer": trigger,
        "trigger_depth_bin": "L1-8",
        "intervention_layer": intervention,
        "intervention_action": action,
        "route_id": route_id,
        "route_source": "single",
        "actions": actions,
        "non_full_count": 1,
        "final_lmms_correct": True,
        "replay_token_parity": True,
        "contract_sha256": "contract",
    }


def test_validate_single_routes_rejects_non_single_or_pretrigger_intervention():
    valid = _route(
        "w1", trigger=3, intervention=5, action="READ_ONLY", route_id="r1"
    )
    validate_single_routes([valid], contract_sha256="contract")

    bad_count = {**valid, "non_full_count": 2}
    with pytest.raises(ValueError, match="exactly one non-FULL"):
        validate_single_routes([bad_count], contract_sha256="contract")

    bad_timing = _route(
        "w2", trigger=5, intervention=3, action="IGNORE", route_id="r2"
    )
    with pytest.raises(ValueError, match="before trigger"):
        validate_single_routes([bad_timing], contract_sha256="contract")


def test_sample_weighting_gives_each_sample_total_weight_one():
    routes = [
        _route("many", trigger=0, intervention=0, action="READ_ONLY", route_id="m1"),
        _route("many", trigger=0, intervention=1, action="READ_ONLY", route_id="m2"),
        _route("many", trigger=0, intervention=2, action="WRITE_ONLY", route_id="m3"),
        _route("one", trigger=0, intervention=3, action="IGNORE", route_id="o1"),
    ]

    rows = sample_weighted_action_distribution(routes)
    by_action = {row["action"]: row for row in rows}

    assert by_action["READ_ONLY"]["weighted_count"] == pytest.approx(2 / 3)
    assert by_action["WRITE_ONLY"]["weighted_count"] == pytest.approx(1 / 3)
    assert by_action["IGNORE"]["weighted_count"] == pytest.approx(1)
    assert sum(row["weighted_count"] for row in rows) == pytest.approx(2)


def test_immediate_vs_delayed_is_classified_per_sample():
    routes = [
        _route("immediate", trigger=4, intervention=4, action="IGNORE", route_id="i1"),
        _route("immediate", trigger=4, intervention=9, action="READ_ONLY", route_id="i2"),
        _route("delayed", trigger=4, intervention=6, action="WRITE_ONLY", route_id="d1"),
    ]

    rows = classify_immediacy(routes)

    assert {row["uid"]: row["classification"] for row in rows} == {
        "delayed": "DELAYED_ONLY_FIXABLE",
        "immediate": "IMMEDIATE_FIXABLE",
    }


def test_same_layer_ambiguity_uses_observed_action_sets():
    routes = [
        _route("w1", trigger=2, intervention=5, action="READ_ONLY", route_id="r1"),
        _route("w1", trigger=2, intervention=5, action="IGNORE", route_id="r2"),
        _route("w1", trigger=2, intervention=7, action="WRITE_ONLY", route_id="r3"),
    ]

    rows = same_layer_action_ambiguity(routes)
    by_layer = {row["intervention_layer"]: row for row in rows}

    assert by_layer[5]["observed_successful_action_set"] == "IGNORE|READ_ONLY"
    assert by_layer[5]["ambiguity_class"] == "MULTIPLE_OBSERVED_ACTIONS"
    assert by_layer[7]["ambiguity_class"] == "SINGLE_OBSERVED_ACTION"


def test_sampling_schemes_count_available_prefix_and_suffix_full_states():
    routes = [
        _route("w1", trigger=3, intervention=3, action="READ_ONLY", route_id="r1"),
        _route("w2", trigger=3, intervention=26, action="IGNORE", route_id="r2"),
    ]

    rows = simulate_sampling_schemes(routes)
    by_scheme = {row["scheme"]: row for row in rows}

    # S1: r1 contributes 0 pre + 2 post FULL; r2 contributes 2 pre + 1 post.
    assert by_scheme["S1_ROUTE_BALANCED"]["FULL"] == pytest.approx(5)
    assert by_scheme["S1_ROUTE_BALANCED"]["non_FULL"] == pytest.approx(2)
    # Both routes expose 24 stored FULL states from trigger L3 onward.
    assert by_scheme["S2_ROUTE_BALANCED_K6"]["FULL"] == pytest.approx(12)
    assert by_scheme["S2_ROUTE_BALANCED_K6"]["non_FULL"] == pytest.approx(2)


def test_semantic_state_id_depends_on_pre_layer_prefix_not_future_actions():
    base = {
        "uid": "w1",
        "layer": 5,
        "route_key": "FULL|FULL|FULL|FULL|FULL|READ_ONLY|FULL",
        "contract_sha256": "contract",
        "feature_schema_sha256": "schema",
    }
    same_entering_state = {
        **base,
        "route_key": "FULL|FULL|FULL|FULL|FULL|IGNORE|FULL",
    }
    different_prefix = {
        **base,
        "route_key": "FULL|FULL|FULL|READ_ONLY|FULL|IGNORE|FULL",
    }

    assert semantic_state_id(base) == semantic_state_id(same_entering_state)
    assert semantic_state_id(base) != semantic_state_id(different_prefix)


def test_audit_cli_is_directly_executable_from_repo_root():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "experiments/analyze_stage2_single_label_distribution.py",
            "--help",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
