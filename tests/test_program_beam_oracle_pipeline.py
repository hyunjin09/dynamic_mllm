from __future__ import annotations

from pathlib import Path

import pytest

from experiments.run_program_beam_oracle_audit import (
    _assign_uid_workers,
    _decision_case,
    _validate_execution_coverage,
    _validate_parity_rows,
)


def test_assign_uid_workers_keeps_each_uid_on_one_rank():
    rows = [
        {"uid": "a", "program_id": f"a{i}", "actions": ["FULL"] * 10}
        for i in range(8)
    ] + [
        {"uid": "b", "program_id": f"b{i}", "actions": ["FULL"] * 2}
        for i in range(3)
    ] + [
        {"uid": "c", "program_id": "c0", "actions": ["FULL"] * 6}
    ]
    assignments, loads = _assign_uid_workers(rows, world_size=2)
    uid_ranks = {}
    for row in assignments:
        uid_ranks.setdefault(row["uid"], set()).add(row["worker_rank"])
    assert all(len(ranks) == 1 for ranks in uid_ranks.values())
    assert len(loads) == 2
    assert sum(loads) == sum(len(row["actions"]) for row in rows)


def test_validate_execution_coverage_rejects_missing_and_duplicate_programs():
    expected = [{"program_id": "p1"}, {"program_id": "p2"}]
    with pytest.raises(RuntimeError, match="coverage"):
        _validate_execution_coverage(expected, [{"program_id": "p1"}])
    with pytest.raises(RuntimeError, match="coverage"):
        _validate_execution_coverage(
            expected,
            [{"program_id": "p1"}, {"program_id": "p1"}, {"program_id": "p2"}],
        )
    _validate_execution_coverage(
        expected, [{"program_id": "p2"}, {"program_id": "p1"}]
    )


def test_decision_case_uses_quantitatively_dominant_w_failure_mode():
    assert _decision_case(ranking_failures=20, generation_failures=5) == (
        "ranking",
        "training-side program preference/ranking objective",
    )
    assert _decision_case(ranking_failures=3, generation_failures=400) == (
        "generation/representation",
        "one minimal trigger-state representation enrichment experiment",
    )
    assert _decision_case(ranking_failures=7, generation_failures=7) == (
        "mixed",
        "one minimal trigger-state representation enrichment experiment",
    )


def test_validate_parity_rows_requires_all_uids_and_expected_transitions():
    expected = [{"uid": "w"}, {"uid": "c"}]
    rows = [
        {
            "uid": "w",
            "dense_correct": False,
            "program_correct": True,
            "parity": {"tokens": True},
        },
        {
            "uid": "c",
            "dense_correct": True,
            "program_correct": False,
            "parity": {"tokens": True},
        },
    ]
    summary = _validate_parity_rows(
        expected, rows, expected_w_to_c=1, expected_c_to_w=1, expected_net=0
    )
    assert summary["rows"] == 2
    assert summary["w_to_c"] == 1
    assert summary["c_to_w"] == 1

    rows[0]["parity"]["tokens"] = False
    with pytest.raises(RuntimeError, match="parity"):
        _validate_parity_rows(
            expected, rows, expected_w_to_c=1, expected_c_to_w=1, expected_net=0
        )


def test_script_binds_command_helpers_before_invoking_main():
    source = Path("experiments/run_program_beam_oracle_audit.py").read_text()
    main_guard = source.rindex('if __name__ == "__main__":')
    assert source.index("def _assign_uid_workers") < main_guard
    assert source.index("def _validate_execution_coverage") < main_guard
    assert source.index("def _decision_case") < main_guard
