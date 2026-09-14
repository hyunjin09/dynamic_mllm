import math

import pytest

from dense_failure_stage2.predictability_measurement import (
    accepted_answer_specs,
    build_p90_trigger_rows,
    derive_utility_row,
    validate_complete_state_results,
)


def test_accepted_answers_preserve_dataset_evaluator_structure():
    assert accepted_answer_specs({"dataset": "gqa", "answer": "The, DOG!"}) == [
        {"text": "the dog", "weight": 1.0}
    ]
    assert accepted_answer_specs({"dataset": "chartqa", "answer": "69.79"}) == [
        {"text": "69.79", "weight": 1.0}
    ]
    textvqa = accepted_answer_specs(
        {
            "dataset": "textvqa",
            "answer": "Two",
            "all_answer_norms": ["Two", "two", "2", "cars"],
        }
    )
    assert textvqa == [
        {"text": "2", "weight": 0.75},
        {"text": "cars", "weight": 0.25},
    ]


def test_p90_manifest_is_recomputed_from_scores_and_is_unique():
    rows = [
        {
            "uid": "a",
            "threshold_name": "P98",
            "dataset": "gqa",
            "source_regime": "historical",
            "dense_correct": False,
            **{f"p_{layer}": 0.1 for layer in range(28)},
        },
        {
            "uid": "a",
            "threshold_name": "P90",
            "dataset": "gqa",
            "source_regime": "historical",
            "dense_correct": False,
            **{f"p_{layer}": 0.95 if layer >= 11 else 0.1 for layer in range(28)},
        },
        {
            "uid": "b",
            "threshold_name": "P90",
            "dataset": "chartqa",
            "source_regime": "canonical",
            "dense_correct": True,
            **{f"p_{layer}": 0.2 for layer in range(28)},
        },
    ]
    result = build_p90_trigger_rows(rows, threshold=0.9061332901863008)
    assert [row["uid"] for row in result] == ["a", "b"]
    assert result[0]["triggered"] is True
    assert result[0]["first_trigger_layer"] == 11
    assert result[0]["post_trigger_state_count"] == 17
    assert result[1]["triggered"] is False
    assert result[1]["first_trigger_layer"] is None


def test_utility_decomposition_and_flip_labels_are_exact():
    branches = {
        "FULL": {"mean_logprob": -2.0, "correct": False},
        "READ_ONLY": {"mean_logprob": -1.0, "correct": True},
        "WRITE_ONLY": {"mean_logprob": -3.0, "correct": True},
        "IGNORE": {"mean_logprob": -4.0, "correct": False},
    }
    row = derive_utility_row(branches)
    assert row["u_read_w1"] == pytest.approx(1.0)
    assert row["u_read_w0"] == pytest.approx(3.0)
    assert row["u_write_r1"] == pytest.approx(-1.0)
    assert row["u_write_r0"] == pytest.approx(1.0)
    assert row["u_read"] == pytest.approx(2.0)
    assert row["u_write"] == pytest.approx(0.0)
    assert row["u_interaction"] == pytest.approx(-2.0)
    assert row["best_action_by_q"] == "READ_ONLY"
    assert row["full_gap"] == pytest.approx(1.0)
    assert row["local_rescue_exists"] is True
    assert row["local_regression_exists"] is False
    assert row["read_harmful_flip_w1"] is True
    assert row["read_beneficial_flip_w0"] is True
    assert row["write_harmful_flip_r1"] is True
    assert row["write_beneficial_flip_r0"] is True
    assert row["correct_action_count"] == 2
    assert math.exp(row["u_read_w1"]) == pytest.approx(row["exp_u_read_w1"])


def test_global_completeness_rejects_missing_duplicate_and_partial_states():
    expected = ["s0", "s1"]
    valid = [
        {"state_id": "s0", "branches": ["FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE"]},
        {"state_id": "s1", "branches": ["FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE"]},
    ]
    validate_complete_state_results(expected, valid)
    with pytest.raises(ValueError, match="missing"):
        validate_complete_state_results(expected, valid[:1])
    with pytest.raises(ValueError, match="duplicate"):
        validate_complete_state_results(expected, [valid[0], valid[0], valid[1]])
    with pytest.raises(ValueError, match="four branches"):
        validate_complete_state_results(
            expected,
            [valid[0], {"state_id": "s1", "branches": ["FULL"]}],
        )
