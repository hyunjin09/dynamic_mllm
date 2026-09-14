from __future__ import annotations

import hashlib
import json

import pytest

from dense_failure_stage2.full_label_generation import (
    assign_workers,
    build_corpus_rows,
    cap_pilot_record,
    choose_preferred_route,
    validate_population_completion,
    verify_artifact_manifest,
)
from experiments.run_full_corrective_label_generation import _route_manifest_row


def _route(key: str, *, non_full: int, stage: str = "mcts") -> dict:
    return {
        "route_key": key,
        "search_stage": stage,
        "non_full_count": non_full,
        "actions": ["FULL"] * 28,
        "correct": True,
    }


def test_cap_pilot_record_excludes_post_cap_success_and_routes():
    sample = {
        "uid": "gqa:late",
        "outcome_class": "MCTS-FIXABLE",
        "successful_single_routes": [],
        "retained_successful_routes": [_route("early", non_full=3), _route("late", non_full=2)],
        "mcts_result": {
            "search_rows": [
                {"iteration": 199, "route_key": "early", "reward": 1, "non_full_count": 3},
                {"iteration": 201, "route_key": "late", "reward": 1, "non_full_count": 2},
            ],
            "first_success_iteration": 199,
        },
    }

    capped = cap_pilot_record(sample, cap=200, retain_successes=8)

    assert capped["outcome_class"] == "MCTS_ONLY_FIXABLE"
    assert capped["first_success_iteration"] == 199
    assert [row["route_key"] for row in capped["retained_successful_routes"]] == ["early"]
    assert capped["excluded_post_cap_route_keys"] == ["late"]


def test_cap_pilot_record_turns_only_post_cap_rescue_into_unresolved():
    sample = {
        "uid": "gqa:post-cap-only",
        "outcome_class": "MCTS-FIXABLE",
        "successful_single_routes": [],
        "retained_successful_routes": [_route("late", non_full=2)],
        "mcts_result": {
            "search_rows": [
                {"iteration": 200, "route_key": "miss", "reward": 0, "non_full_count": 2},
                {"iteration": 294, "route_key": "late", "reward": 1, "non_full_count": 2},
            ],
            "first_success_iteration": 294,
        },
    }

    capped = cap_pilot_record(sample, cap=200, retain_successes=8)

    assert capped["outcome_class"] == "UNRESOLVED"
    assert capped["first_success_iteration"] is None
    assert capped["retained_successful_routes"] == []
    assert capped["mcts_iterations"] == 1


def test_cap_pilot_record_requires_shards_only_for_canonical_retained_routes():
    search_rows = [
        {"iteration": index, "route_key": f"r{index}", "reward": 1, "non_full_count": 2}
        for index in range(1, 10)
    ]
    retained = [_route(f"r{index}", non_full=2) for index in range(1, 9)]
    sample = {
        "uid": "textvqa:many-successes",
        "outcome_class": "MCTS-FIXABLE",
        "successful_single_routes": [],
        "retained_successful_routes": retained,
        "mcts_result": {"search_rows": search_rows, "first_success_iteration": 1},
    }

    capped = cap_pilot_record(sample, cap=200, retain_successes=8)

    assert len(capped["retained_successful_routes"]) == 8
    assert "r9" in capped["excluded_post_cap_route_keys"]


def test_cap_pilot_record_preserves_all_single_successes_without_mcts():
    routes = [_route("single-b", non_full=1, stage="single"), _route("single-a", non_full=1, stage="single")]
    sample = {
        "uid": "chartqa:single",
        "outcome_class": "SINGLE-FIXABLE",
        "successful_single_routes": routes,
        "retained_successful_routes": routes,
        "mcts_result": None,
    }

    capped = cap_pilot_record(sample, cap=200, retain_successes=8)

    assert capped["outcome_class"] == "SINGLE_FIXABLE"
    assert {row["route_key"] for row in capped["retained_successful_routes"]} == {
        "single-a",
        "single-b",
    }
    assert capped["mcts_iterations"] == 0


def test_preferred_route_uses_non_full_then_discovery_then_key():
    routes = [
        {**_route("z", non_full=2), "discovery_iteration": 4},
        {**_route("b", non_full=1), "discovery_iteration": 8},
        {**_route("a", non_full=1), "discovery_iteration": 8},
        {**_route("c", non_full=1), "discovery_iteration": 9},
    ]

    assert choose_preferred_route(routes)["route_key"] == "a"


def test_worker_assignment_is_deterministic_and_cost_balanced():
    rows = [
        {"uid": "a", "estimated_terminal_evaluations": 100},
        {"uid": "b", "estimated_terminal_evaluations": 70},
        {"uid": "c", "estimated_terminal_evaluations": 40},
        {"uid": "d", "estimated_terminal_evaluations": 30},
        {"uid": "e", "estimated_terminal_evaluations": 10},
    ]
    first = assign_workers(rows, world_size=2)
    second = assign_workers(list(reversed(rows)), world_size=2)

    assert first == second
    loads = [sum(row["estimated_terminal_evaluations"] for row in first if row["worker_rank"] == rank) for rank in range(2)]
    assert max(loads) - min(loads) <= 30


def test_population_completion_rejects_missing_duplicate_and_mixed_task_types():
    rows = [
        {"uid": "w1", "task_type": "triggered_wrong", "passed": True},
        {"uid": "w2", "task_type": "triggered_wrong", "passed": True},
        {"uid": "c1", "task_type": "triggered_correct_preservation", "passed": True},
    ]
    validate_population_completion(["w1", "w2"], ["c1"], rows)
    with pytest.raises(ValueError, match="coverage"):
        validate_population_completion(["w1", "w2"], ["c1"], rows[:-1])
    with pytest.raises(ValueError, match="coverage"):
        validate_population_completion(["w1", "w2"], ["c1"], [rows[0], rows[0], rows[2]])
    with pytest.raises(ValueError, match="task type"):
        validate_population_completion(["w1", "w2"], ["c1"], [{**row, "task_type": "triggered_wrong"} for row in rows])


def test_corpora_remain_separate_and_reference_only_correct_replayed_routes():
    routes = [
        {"uid": "c", "route_source": "preservation_full", "final_lmms_correct": True, "replay_token_parity": True},
        {"uid": "s", "route_source": "single", "final_lmms_correct": True, "replay_token_parity": True},
        {"uid": "m", "route_source": "mcts", "final_lmms_correct": True, "replay_token_parity": True},
    ]
    corpora = build_corpus_rows(routes)

    assert [row["uid"] for row in corpora["A"]] == ["c"]
    assert [row["uid"] for row in corpora["B"]] == ["s"]
    assert [row["uid"] for row in corpora["C"]] == ["m"]
    with pytest.raises(ValueError, match="correct replay"):
        build_corpus_rows([{**routes[1], "replay_token_parity": False}])


def test_artifact_manifest_verification_fails_closed(tmp_path):
    payload = tmp_path / "evidence.json"
    payload.write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
    expected = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = {"passed": True, "files": {"evidence.json": expected}}

    verify_artifact_manifest(tmp_path, manifest)
    payload.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_artifact_manifest(tmp_path, manifest)


@pytest.mark.parametrize("trigger_field", ["trigger_layer", "first_trigger_layer"])
def test_route_manifest_accepts_imported_and_fresh_trigger_field_names(trigger_field):
    route = {
        **_route("route", non_full=1, stage="single"),
        "actions": ["READ_ONLY", *(["FULL"] * 27)],
        "generated_ids": [1],
        "generated_answer": "answer",
        "lmms_metric": "accuracy",
        "lmms_score": 1.0,
    }
    sample = {
        "uid": "gqa:row",
        "dataset": "gqa",
        "trigger_depth_bin": "L0",
        trigger_field: 0,
    }
    contract = {"contract_sha256": "contract"}

    row = _route_manifest_row(
        route=route,
        sample=sample,
        route_source="single",
        feature_file="states/row.pt",
        feature_hash="hash",
        feature_rows=28,
        contract=contract,
        provenance={},
    )

    assert row["trigger_layer"] == 0
