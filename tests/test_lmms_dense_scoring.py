from __future__ import annotations

import pytest

from dense_failure_stage1.lmms_scoring import score_lmms_sample


def test_gqa_uses_lmms_exact_match_case_and_punctuation_contract():
    result = score_lmms_sample(
        dataset="gqa",
        prediction="Blue.",
        answer="blue",
        answers=None,
        uid="gqa:1",
    )

    assert result.metric_name == "exact_match"
    assert result.raw_score == 1.0
    assert result.correctness_threshold == 1.0
    assert result.correct is True


@pytest.mark.parametrize(
    ("prediction", "answer", "expected"),
    [("104", "100", 1.0), ("106", "100", 0.0), ("Blue", "blue", 1.0)],
)
def test_chartqa_uses_lmms_relaxed_overall(prediction, answer, expected):
    result = score_lmms_sample(
        dataset="chartqa",
        prediction=prediction,
        answer=answer,
        answers=None,
        uid="chartqa:1",
    )

    assert result.metric_name == "relaxed_overall"
    assert result.raw_score == expected
    assert result.correctness_threshold == 1.0
    assert result.correct is bool(expected)


def test_textvqa_preserves_fractional_lmms_score_and_existing_binary_threshold():
    answers = ["one"] + ["other"] * 9
    result = score_lmms_sample(
        dataset="textvqa",
        prediction="one",
        answer="one",
        answers=answers,
        uid="textvqa:1",
    )

    # LMMS TextVQA uses leave-one-out EvalAI consensus, so one of ten matching
    # references scores 0.3 rather than the simplified 1/3 approximation.
    assert result.metric_name == "exact_match"
    assert result.raw_score == pytest.approx(0.3)
    assert result.normalized_prediction == "1"
    assert result.correctness_threshold == 0.5
    assert result.correct is False


def test_textvqa_requires_the_full_reference_list():
    with pytest.raises(ValueError, match="reference answers"):
        score_lmms_sample(
            dataset="textvqa",
            prediction="answer",
            answer="answer",
            answers=None,
            uid="textvqa:missing",
        )

