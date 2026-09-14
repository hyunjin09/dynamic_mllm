from __future__ import annotations

from collections import Counter

import pytest

from dense_failure_stage2.treatment_label_completeness import (
    ACTIONS,
    AUDITED_INTERVENE,
    AUDITED_KEEP,
    AUDITED_MIXED,
    classify_audited_actions,
    deterministic_state_sample,
    mcts_budget_saturation,
    unobserved_actions,
    validate_complete_state_audits,
)


def _rows() -> list[dict]:
    rows = []
    labels = (("KEEP_REQUIRED", 18), ("INTERVENE_REQUIRED", 18), ("MIXED", 12))
    for label, count in labels:
        for index in range(count):
            rows.append(
                {
                    "state_id": f"{label}:{index:02d}",
                    "uid": f"uid:{label}:{index // 2:02d}",
                    "image_group_id": f"group:{label}:{index // 2:02d}",
                    "state_label": label,
                    "dataset": ("gqa", "chartqa", "textvqa")[index % 3],
                    "source_regime": ("historical", "canonical")[index % 2],
                    "layer_bin": ("early_0_8", "middle_9_18", "late_19_27")[index % 3],
                    "route_source_signature": ("single", "mcts", "mcts+single")[index % 3],
                }
            )
    return rows


def test_deterministic_state_sample_meets_class_targets_and_uid_cap():
    rows = _rows()
    targets = {"KEEP_REQUIRED": 10, "INTERVENE_REQUIRED": 10, "MIXED": 6}
    first = deterministic_state_sample(rows, targets=targets, seed=73, uid_cap=4)
    second = deterministic_state_sample(list(reversed(rows)), targets=targets, seed=73, uid_cap=4)

    assert [row["state_id"] for row in first] == [row["state_id"] for row in second]
    assert Counter(row["state_label"] for row in first) == Counter(targets)
    assert max(Counter(row["uid"] for row in first).values()) <= 4
    assert len({row["state_id"] for row in first}) == len(first)
    for label in targets:
        selected = [row for row in first if row["state_label"] == label]
        assert {row["dataset"] for row in selected} == {"gqa", "chartqa", "textvqa"}
        assert {row["source_regime"] for row in selected} == {"historical", "canonical"}


def test_deterministic_state_sample_fails_when_uid_cap_makes_target_impossible():
    rows = [{**row, "uid": "one"} for row in _rows() if row["state_label"] == "KEEP_REQUIRED"]
    with pytest.raises(ValueError, match="uid cap"):
        deterministic_state_sample(
            rows,
            targets={"KEEP_REQUIRED": 5},
            seed=73,
            uid_cap=4,
        )


def test_unobserved_actions_and_audited_label_contract():
    assert unobserved_actions(["FULL"]) == ("READ_ONLY", "WRITE_ONLY", "IGNORE")
    assert unobserved_actions(["READ_ONLY", "IGNORE"]) == ("FULL", "WRITE_ONLY")
    assert classify_audited_actions(["FULL"]) == AUDITED_KEEP
    assert classify_audited_actions(["READ_ONLY", "IGNORE"]) == AUDITED_INTERVENE
    assert classify_audited_actions(["FULL", "WRITE_ONLY"]) == AUDITED_MIXED
    with pytest.raises(ValueError, match="unsupported"):
        unobserved_actions(["BAD"])


def test_global_audit_validation_rejects_missing_duplicates_and_quarantine():
    expected = ["s0", "s1"]
    good = [
        {"state_id": "s0", "passed": True, "quarantined": False},
        {"state_id": "s1", "passed": True, "quarantined": False},
    ]
    assert validate_complete_state_audits(expected, good) == {
        "expected": 2,
        "completed": 2,
        "duplicates": 0,
        "missing": 0,
        "quarantined": 0,
    }
    with pytest.raises(ValueError, match="coverage"):
        validate_complete_state_audits(expected, good[:1])
    with pytest.raises(ValueError, match="coverage"):
        validate_complete_state_audits(expected, [good[0], good[0]])
    with pytest.raises(ValueError, match="quarantined"):
        validate_complete_state_audits(
            expected,
            [good[0], {"state_id": "s1", "passed": False, "quarantined": True}],
        )


def test_mcts_saturation_uses_prospectively_fixed_late_discovery_rule():
    saturated = mcts_budget_saturation([5, 40, 80, 100, 120, 140, 150, 151, 149, 130])
    assert saturated["saturated"] is True
    assert saturated["late_151_200_fraction"] == pytest.approx(0.1)

    unsaturated = mcts_budget_saturation([10, 30, 80, 100, 120, 140, 151, 170, 180, 190])
    assert unsaturated["saturated"] is False
    assert unsaturated["late_151_200_fraction"] > 0.1

    unsupported = mcts_budget_saturation([10, 20, 30, 40])
    assert unsupported["saturated"] is None
    assert unsupported["reason"] == "fewer_than_10_mcts_discoveries"


def test_action_order_is_frozen():
    assert ACTIONS == ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")
