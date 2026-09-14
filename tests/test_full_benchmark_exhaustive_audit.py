from __future__ import annotations

from copy import deepcopy

import pytest

from dense_failure_stage2.full_benchmark_exhaustive_audit import (
    answer_change_record,
    classify_bottleneck,
    summarize_action_behavior,
    summarize_funnel,
    validate_audit_rows,
)


def _row(
    uid: str,
    *,
    dense_correct: bool,
    routed_correct: bool,
    triggered: bool = False,
    trigger_layer: int | None = None,
    nonfull: tuple[tuple[int, str], ...] = (),
    max_score: float = 0.4,
) -> dict:
    actions = ["FULL"] * 28
    for layer, action in nonfull:
        actions[layer] = action
    scores = [0.1] * 28
    if triggered:
        assert trigger_layer is not None
        scores[trigger_layer] = max_score
    first_nonfull = min((layer for layer, _ in nonfull), default=None)
    transition = ("C" if dense_correct else "W") + "→" + (
        "C" if routed_correct else "W"
    )
    return {
        "uid": uid,
        "benchmark": "chartqa",
        "benchmark_family": "chartqa",
        "contract_sha256": "contract",
        "dense_correct": dense_correct,
        "routed_correct": routed_correct,
        "transition": transition,
        "dense_generated_answer": "dense",
        "routed_generated_answer": "routed",
        "answer": "gt",
        "dense_score": float(dense_correct),
        "routed_score": float(routed_correct),
        "stage1_threshold": 0.9,
        "stage1_comparison": "strict_greater_than",
        "stage1_scores": scores,
        "stage1_max_score": max_score,
        "triggered": triggered,
        "trigger_layer": trigger_layer,
        "actions": actions,
        "action_rows": [
            {"layer": layer, "action": action, "active": triggered and layer >= trigger_layer}
            for layer, action in enumerate(actions)
        ],
        "any_non_full": bool(nonfull),
        "non_full_count": len(nonfull),
        "first_non_full_layer": first_nonfull,
        "trigger_to_first_non_full_delay": (
            None if first_nonfull is None else first_nonfull - int(trigger_layer)
        ),
        "post_trigger_actions": 0 if trigger_layer is None else 28 - trigger_layer,
        "post_trigger_action_counts": {
            action: actions[int(trigger_layer) :].count(action)
            for action in ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")
        }
        if triggered
        else {},
    }


def test_funnel_conditions_outcomes_on_dense_state_trigger_and_intervention():
    rows = [
        _row("wc", dense_correct=False, routed_correct=True, triggered=True, trigger_layer=5, nonfull=((5, "WRITE_ONLY"),)),
        _row("ww-trigger-full", dense_correct=False, routed_correct=False, triggered=True, trigger_layer=7),
        _row("ww", dense_correct=False, routed_correct=False),
        _row("cw", dense_correct=True, routed_correct=False, triggered=True, trigger_layer=9, nonfull=((10, "IGNORE"),)),
        _row("cc-trigger", dense_correct=True, routed_correct=True, triggered=True, trigger_layer=8, nonfull=((12, "READ_ONLY"),)),
        _row("cc", dense_correct=True, routed_correct=True),
    ]

    summary = summarize_funnel(rows)

    assert summary["dense_w"] == 3
    assert summary["triggered_w"] == 2
    assert summary["triggered_w_nonfull"] == 1
    assert summary["w_to_c"] == 1
    assert summary["p_w_to_c_given_trigger_nonfull_w"] == 1.0
    assert summary["dense_c"] == 3
    assert summary["triggered_c"] == 2
    assert summary["triggered_c_nonfull"] == 2
    assert summary["c_to_w"] == 1
    assert summary["p_c_to_w_given_trigger_nonfull_c"] == 0.5


def test_action_behavior_counts_sample_level_use_and_first_action():
    rows = [
        _row("w-mixed", dense_correct=False, routed_correct=False, triggered=True, trigger_layer=5, nonfull=((7, "WRITE_ONLY"), (9, "IGNORE"))),
        _row("w-full", dense_correct=False, routed_correct=False, triggered=True, trigger_layer=6),
        _row("c-read", dense_correct=True, routed_correct=True, triggered=True, trigger_layer=8, nonfull=((8, "READ_ONLY"),)),
    ]

    behavior = summarize_action_behavior(rows, dense_correct=False)

    assert behavior["triggered"] == 2
    assert behavior["never_nonfull"] == 1
    assert behavior["write_only_used"] == 1
    assert behavior["ignore_used"] == 1
    assert behavior["multiple_action_types"] == 1
    assert behavior["first_write_only"] == 1
    assert behavior["first_ignore"] == 0


def test_answer_change_record_uses_fixed_overlapping_regression_rules():
    row = _row(
        "regression",
        dense_correct=True,
        routed_correct=False,
        triggered=True,
        trigger_layer=10,
        nonfull=((10, "WRITE_ONLY"), (11, "IGNORE"), (12, "IGNORE")),
        max_score=0.97,
    )

    record = answer_change_record(row)

    assert record["first_non_full_action"] == "WRITE_ONLY"
    assert record["trigger_to_first_non_full_delay"] == 0
    assert record["non_full_actions"] == [
        {"layer": 10, "action": "WRITE_ONLY"},
        {"layer": 11, "action": "IGNORE"},
        {"layer": 12, "action": "IGNORE"},
    ]
    assert record["regression_classes"] == [
        "R2_MULTIPLE_INTERVENTIONS",
        "R3_IMMEDIATE_INTERVENTION",
        "R6_WRITE_RELATED",
        "R7_IGNORE_DOMINATED",
        "R8_OTHER_OR_MIXED",
    ]


def test_validation_rejects_trace_inconsistency():
    row = _row("valid", dense_correct=False, routed_correct=True, triggered=True, trigger_layer=4, nonfull=((6, "IGNORE"),), max_score=0.95)
    validate_audit_rows([row], expected_contract="contract")

    invalid = dict(row, any_non_full=False)
    with pytest.raises(ValueError, match="any_non_full"):
        validate_audit_rows([invalid], expected_contract="contract")

    invalid_scores = dict(row, stage1_scores=[0.1] * 27)
    with pytest.raises(ValueError, match="28 Stage-1 scores"):
        validate_audit_rows([invalid_scores], expected_contract="contract")

    invalid_max = dict(row, stage1_max_score=0.94)
    with pytest.raises(ValueError, match="max score"):
        validate_audit_rows([invalid_max], expected_contract="contract")

    invalid_action_rows = deepcopy(row)
    invalid_action_rows["action_rows"][6]["action"] = "FULL"
    with pytest.raises(ValueError, match="action_rows"):
        validate_audit_rows([invalid_action_rows], expected_contract="contract")

    invalid_counts = deepcopy(row)
    invalid_counts["post_trigger_action_counts"] = {"FULL": 22}
    with pytest.raises(ValueError, match="post-trigger action counts"):
        validate_audit_rows([invalid_counts], expected_contract="contract")


def test_bottleneck_rules_distinguish_inactive_treatment_and_preservation():
    assert classify_bottleneck({"triggered_w": 0, "triggered_c": 0}) == "INACTIVE"
    assert classify_bottleneck(
        {
            "triggered_w": 50,
            "triggered_c": 10,
            "triggered_w_nonfull": 20,
            "triggered_c_nonfull": 5,
            "w_to_c": 0,
            "c_to_w": 2,
        }
    ) == "TREATMENT_QUALITY_LIMITED"
    assert classify_bottleneck(
        {
            "triggered_w": 50,
            "triggered_c": 50,
            "triggered_w_nonfull": 10,
            "triggered_c_nonfull": 10,
            "w_to_c": 2,
            "c_to_w": 8,
        }
    ) == "PRESERVATION_LIMITED"
