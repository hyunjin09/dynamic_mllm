from __future__ import annotations

import math

import pytest

from dense_failure_stage2.abstention_margin import (
    artifact_hash,
    assign_group_folds,
    build_margin_grid,
    choose_action_from_logits,
    select_margin,
    summarize_margin_rollout,
    validate_result_matrix,
)


def test_artifact_hash_ignores_only_its_declared_self_hash_field():
    value = {"payload": 3, "binding_sha256": "placeholder", "contract_sha256": "parent"}
    expected = artifact_hash({"payload": 3, "contract_sha256": "parent"}, hash_field="binding_sha256")
    assert artifact_hash(value, hash_field="binding_sha256") == expected
    assert artifact_hash(value, hash_field="contract_sha256") != expected


def test_action_uses_best_nonfull_only_when_strict_margin_exceeds_delta():
    logits = [1.0, 1.25, 2.0, -3.0]
    chosen = choose_action_from_logits(logits, delta=0.5)
    assert chosen["action_index"] == 2
    assert chosen["action"] == "WRITE_ONLY"
    assert chosen["raw_margin"] == pytest.approx(1.0)

    tied = choose_action_from_logits(logits, delta=1.0)
    assert tied["action"] == "FULL"
    assert choose_action_from_logits(logits, delta=math.inf)["action"] == "FULL"


def test_margin_grid_is_frozen_from_positive_training_margins_and_deduplicated():
    grid = build_margin_grid(
        [-2.0, 0.0, 1.0, 1.0, 3.0, 5.0],
        quantiles=(0.1, 0.5, 0.9),
    )
    assert grid[0]["label"] == "zero"
    assert grid[-1]["label"] == "infinity"
    assert math.isinf(grid[-1]["delta"])
    finite = [row["delta"] for row in grid[:-1]]
    assert finite == sorted(set(finite))
    assert finite == pytest.approx([0.0, 1.0, 2.0, 4.4])


def test_rollout_summary_counts_transitions_and_preservation():
    rows = [
        {"dense_correct": True, "routed_correct": True, "triggered": False, "any_non_full": False, "non_full_count": 0, "post_trigger_action_counts": {}},
        {"dense_correct": True, "routed_correct": False, "triggered": True, "any_non_full": True, "non_full_count": 2, "post_trigger_action_counts": {"FULL": 2, "IGNORE": 2}},
        {"dense_correct": False, "routed_correct": True, "triggered": True, "any_non_full": True, "non_full_count": 1, "post_trigger_action_counts": {"FULL": 3, "WRITE_ONLY": 1}},
        {"dense_correct": False, "routed_correct": False, "triggered": True, "any_non_full": False, "non_full_count": 0, "post_trigger_action_counts": {"FULL": 4}},
    ]
    summary = summarize_margin_rollout(rows)
    assert summary["w_to_c"] == 1
    assert summary["c_to_w"] == 1
    assert summary["net_corrections"] == 0
    assert summary["c_to_c_preservation_rate"] == pytest.approx(0.5)
    assert summary["intervened_samples"] == 2
    assert summary["stage1_triggered_count"] == 3
    assert summary["total_non_full_count"] == 3
    assert summary["post_trigger_full_fraction"] == pytest.approx(0.75)


def test_margin_selection_requires_positive_margin_to_improve_over_zero():
    baseline = {
        "delta": 0.0,
        "net_corrections": 3,
        "c_to_w": 1,
        "w_to_c": 4,
        "intervened_samples": 21,
        "c_to_c_preservation_rate": 0.9975,
    }
    tied = dict(
        baseline,
        delta=0.4,
        c_to_w=0,
        w_to_c=3,
        intervened_samples=12,
        c_to_c_preservation_rate=1.0,
    )
    assert select_margin([baseline, tied])["delta"] == 0.0

    improved = dict(
        baseline,
        delta=0.7,
        net_corrections=4,
        c_to_w=0,
        w_to_c=4,
        intervened_samples=9,
        c_to_c_preservation_rate=1.0,
    )
    assert select_margin([baseline, tied, improved])["delta"] == 0.7


def test_group_folds_are_deterministic_and_group_disjoint():
    rows = [
        {"uid": "a1", "image_group_id": "a"},
        {"uid": "a2", "image_group_id": "a"},
        {"uid": "b", "image_group_id": "b"},
        {"uid": "c", "image_group_id": "c"},
        {"uid": "d", "image_group_id": "d"},
        {"uid": "e", "image_group_id": "e"},
    ]
    first = assign_group_folds(rows, folds=3, seed=17)
    second = assign_group_folds(list(reversed(rows)), folds=3, seed=17)
    assert first == second
    assert first["a1"] == first["a2"]
    groups_by_fold = {
        fold: {row["image_group_id"] for row in rows if first[row["uid"]] == fold}
        for fold in range(3)
    }
    assert all(
        groups_by_fold[left].isdisjoint(groups_by_fold[right])
        for left in groups_by_fold
        for right in groups_by_fold
        if left < right
    )


def test_result_matrix_rejects_missing_duplicate_or_wrong_contract_rows():
    rows = [
        {"uid": "a", "margin_id": "m0", "contract_sha256": "ok"},
        {"uid": "a", "margin_id": "m1", "contract_sha256": "ok"},
        {"uid": "b", "margin_id": "m0", "contract_sha256": "ok"},
        {"uid": "b", "margin_id": "m1", "contract_sha256": "ok"},
    ]
    validate_result_matrix(["a", "b"], ["m0", "m1"], rows, contract_sha256="ok")
    with pytest.raises(ValueError, match="matrix"):
        validate_result_matrix(["a", "b"], ["m0", "m1"], rows[:-1], contract_sha256="ok")
    with pytest.raises(ValueError, match="matrix"):
        validate_result_matrix(["a", "b"], ["m0", "m1"], rows + [rows[0]], contract_sha256="ok")
    bad = [dict(row) for row in rows]
    bad[0]["contract_sha256"] = "wrong"
    with pytest.raises(ValueError, match="contract"):
        validate_result_matrix(["a", "b"], ["m0", "m1"], bad, contract_sha256="ok")
