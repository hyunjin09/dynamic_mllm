from __future__ import annotations

import numpy as np
import pytest

from dense_failure_stage1.treatment_correctability import (
    bounded_route_panel,
    depth_bin,
    summarize_regime,
    trigger_manifest_rows,
    validate_execution_coverage,
)
from experiments.analyze_stage1_gate_treatment_correctability import _run_regime


def test_route_panel_has_all_singles_and_constant_pair_panel():
    early = bounded_route_panel(uid="u", start_layer=0, seed=7)
    middle = bounded_route_panel(uid="u", start_layer=20, seed=7)
    final = bounded_route_panel(uid="u", start_layer=27, seed=7)
    assert len(early) == 84 + 12
    assert len(middle) == 24 + 12
    assert len(final) == 3
    assert [row["search_stage"] for row in early[:3]] == ["immediate_single"] * 3
    assert all(action == "FULL" for row in middle for action in row["actions"][:20])
    assert early == bounded_route_panel(uid="u", start_layer=0, seed=7)
    assert early != bounded_route_panel(uid="other", start_layer=0, seed=7)


def test_trigger_manifest_freezes_dynamic_and_replay_start_layers():
    records = [
        {"uid": "a", "dataset": "gqa", "image_group_id": "i1", "current_dense_wrong": True},
        {"uid": "b", "dataset": "chartqa", "image_group_id": "i2", "current_dense_wrong": False},
    ]
    scores = np.arange(56, dtype=float).reshape(2, 28)
    dynamic = trigger_manifest_rows(records, scores, [4, -1], gate="shared_random4", split="val")
    fixed = trigger_manifest_rows(records, scores, [27, -1], gate="fixed_l27", split="val")
    assert dynamic[0]["treatment_start_layer"] == 4
    assert dynamic[0]["cohort"] == "triggered_dense_wrong"
    assert dynamic[1]["cohort"] == "non_triggered_dense_correct"
    assert fixed[0]["trigger_layer"] == 27
    assert fixed[0]["treatment_start_layer"] == 0


def test_depth_bins_are_frozen():
    assert depth_bin(0) == depth_bin(8) == "early"
    assert depth_bin(9) == depth_bin(18) == "middle"
    assert depth_bin(19) == depth_bin(27) == "late"
    with pytest.raises(ValueError):
        depth_bin(28)


def test_summary_separates_triggered_and_population_denominators():
    rows = [
        {"current_dense_wrong": True, "correctable": True, "single_intervention_success": False, "route_evaluations": 5, "elapsed_seconds": 2},
        {"current_dense_wrong": True, "correctable": False, "single_intervention_success": False, "route_evaluations": 10, "elapsed_seconds": 4},
        {"current_dense_wrong": False, "correctable": True, "single_intervention_success": True, "route_evaluations": 1, "elapsed_seconds": 1},
    ]
    summary = summarize_regime(rows, total_dense_wrong=4)
    assert summary["triggered_wrong_correctability"] == 0.5
    assert summary["population_oracle_rescue"] == 0.25
    assert summary["triggered_correct_preservability"] == 1.0


def test_coverage_requires_one_passed_record_per_expected_key():
    rows = [{"split": "val", "gate": "g", "uid": "u", "passed": True}]
    validate_execution_coverage([("val", "g", "u")], rows)
    with pytest.raises(ValueError, match="coverage"):
        validate_execution_coverage([("val", "g", "x")], rows)


def test_regime_producer_marks_successful_final_record_passed():
    panel = [{"route_key": "route", "search_stage": "immediate_single"}]

    result = _run_regime(
        panel=panel,
        evaluate=lambda candidate: ({**candidate, "correct": False}, True),
        smoke_limit=None,
    )

    assert result["passed"] is True
    validate_execution_coverage(
        [("val", "gate", "uid")],
        [{"split": "val", "gate": "gate", "uid": "uid", **result}],
    )
