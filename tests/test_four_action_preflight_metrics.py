import torch
import pytest

from experiments.run_four_action_answer_alignment import (
    external_semantic_comparison,
    logit_difference_stats,
    native_unified_full_diagnostic,
)


def test_logit_difference_stats_exact_identity():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    stats = logit_difference_stats(logits, logits.clone())
    assert stats["max_abs"] == 0.0
    assert stats["relative_rmse"] == 0.0
    assert stats["logprob_max_abs"] == 0.0
    assert stats["argmax_match"] is True
    assert stats["top10_overlap_fraction"] == 1.0


def test_logit_difference_stats_reports_scale_aware_difference():
    reference = torch.tensor([[1.0, 2.0, 3.0]])
    candidate = torch.tensor([[1.1, 2.0, 2.9]])
    stats = logit_difference_stats(candidate, reference)
    assert 0.0 < stats["mean_abs"] < stats["max_abs"]
    assert stats["relative_rmse"] > 0.0
    assert stats["logprob_rmse"] > 0.0
    assert stats["argmax_match"] is True


def semantic_state(correct_score, wrong_score, ids=(1, 2), answer="answer", correct=False):
    return {
        "S_correct": correct_score,
        "S_full_wrong": wrong_score,
        "margin": correct_score - wrong_score,
        "generated_ids": list(ids),
        "generated_answer": answer,
        "correctness_score": float(correct),
        "correct": correct,
    }


def test_native_unified_drift_is_external_and_signed():
    unified = semantic_state(-1.0, -0.5)
    native = semantic_state(-1.2, -0.6)
    diagnostic = native_unified_full_diagnostic(unified, native)
    assert diagnostic["signed_drift"] == pytest.approx({
        "S_correct": 0.2,
        "margin": 0.1,
        "S_full_wrong": 0.1,
    })
    assert diagnostic["generated_ids_match"] is True


def test_external_semantic_comparison_detects_generation_disagreement():
    unified = semantic_state(-1.0, -0.5)
    external = semantic_state(-1.0, -0.5, ids=(1, 3), answer="other")
    comparison = external_semantic_comparison(unified, external)
    assert comparison["generated_ids_match"] is False
    assert comparison["generated_answer_match"] is False
    assert comparison["correctness_match"] is True
