import pytest

from tools.research_analysis.four_action.eligibility import (
    expected_unified_full_correct,
    summarize_eligibility,
)


def test_expected_unified_full_correct_matches_cohort_definitions():
    assert not expected_unified_full_correct("primary_a_plus")
    assert not expected_unified_full_correct("control_no_correction_found")
    assert expected_unified_full_correct("control_full_correct_all_off_wrong")
    with pytest.raises(ValueError):
        expected_unified_full_correct("other")


def test_eligibility_summary_freezes_candidate_and_exclusion_counts():
    rows = [
        {"uid": "a", "cohort": "primary_a_plus", "eligible": True},
        {"uid": "b", "cohort": "primary_a_plus", "eligible": False},
    ]
    summary = summarize_eligibility(rows, {"primary_a_plus": 2})
    assert summary["candidate_count"] == 2
    assert summary["eligible_counts"] == {"primary_a_plus": 1}
    assert summary["excluded_counts"] == {"primary_a_plus": 1}
