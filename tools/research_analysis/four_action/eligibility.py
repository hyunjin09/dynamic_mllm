from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


EXPECTED_FULL_CORRECT = {
    "primary_a_plus": False,
    "control_no_correction_found": False,
    "control_full_correct_all_off_wrong": True,
}


def expected_unified_full_correct(cohort: str) -> bool:
    try:
        return EXPECTED_FULL_CORRECT[cohort]
    except KeyError as exc:
        raise ValueError(f"unsupported four-action cohort: {cohort}") from exc


def summarize_eligibility(
    rows: Iterable[dict[str, Any]], candidate_counts: dict[str, int]
) -> dict[str, Any]:
    records = list(rows)
    ids = [row["uid"] for row in records]
    observed = Counter(row["cohort"] for row in records)
    if len(ids) != len(set(ids)):
        raise ValueError("unified-FULL eligibility rows contain duplicate UIDs")
    if dict(observed) != candidate_counts:
        raise ValueError(
            f"eligibility candidate counts differ: observed={dict(observed)} "
            f"expected={candidate_counts}"
        )
    eligible = Counter(row["cohort"] for row in records if row["eligible"])
    excluded = Counter(row["cohort"] for row in records if not row["eligible"])
    return {
        "schema_version": "four_action_unified_full_eligibility_summary_v1",
        "candidate_counts": dict(observed),
        "eligible_counts": {cohort: int(eligible[cohort]) for cohort in candidate_counts},
        "excluded_counts": {cohort: int(excluded[cohort]) for cohort in candidate_counts},
        "candidate_count": len(records),
        "eligible_count": sum(eligible.values()),
        "excluded_count": sum(excluded.values()),
        "rule": (
            "matched-cache cohort supplies candidates/routes; current unified FULL "
            "correctness must satisfy the cohort FULL-wrong/FULL-correct condition"
        ),
    }
