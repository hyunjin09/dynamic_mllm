from __future__ import annotations

import pytest

from dense_failure_stage2.program_beam_oracle import (
    compare_top1_replay,
    expected_beam_cardinality,
    freeze_beam_manifests,
    mean_pairwise_hamming,
    summarize_oracle_sample,
)


ACTIONS = ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")


def _beam(actions: list[str], score: float) -> dict:
    return {
        "action_indices": [ACTIONS.index(action) for action in actions],
        "actions": actions,
        "score": score,
    }


def _triggered_row(
    *,
    uid: str = "u1",
    dense_correct: bool = False,
    trigger_layer: int = 27,
    beams: list[dict] | None = None,
) -> dict:
    beams = beams or [
        _beam(["FULL"], -0.1),
        _beam(["READ_ONLY"], -0.2),
        _beam(["WRITE_ONLY"], -0.3),
        _beam(["IGNORE"], -0.4),
    ]
    return {
        "uid": uid,
        "sample_id": uid,
        "benchmark": "chartqa",
        "benchmark_family": "chartqa",
        "image_group_id": f"image:{uid}",
        "dense_correct": dense_correct,
        "dense_generated_token_ids": [1, 2],
        "dense_generated_answer": "dense",
        "dense_score": 1.0 if dense_correct else 0.0,
        "triggered": True,
        "trigger_layer": trigger_layer,
        "program_suffix_actions": beams[0]["actions"],
        "program_generated_token_ids": [3, 4],
        "program_generated_answer": "program",
        "program_score": 0.0,
        "program_correct": False,
        "beam_rows": beams,
        "consumed_image_sha256s": ["abc"],
    }


def test_expected_beam_cardinality_uses_complete_support_for_late_trigger():
    assert expected_beam_cardinality(27) == 4
    assert expected_beam_cardinality(26) == 8
    assert expected_beam_cardinality(17) == 8


def test_freeze_beam_manifests_preserves_ranks_and_deduplicates_execution():
    beams = [
        _beam(["FULL"], -0.1),
        _beam(["READ_ONLY"], -0.2),
        _beam(["READ_ONLY"], -0.3),
        _beam(["IGNORE"], -0.4),
    ]
    ranked, unique, summary = freeze_beam_manifests(
        [_triggered_row(beams=beams)], require_complete_cardinality=True
    )

    assert [row["rank"] for row in ranked] == [1, 2, 3, 4]
    assert len(unique) == 3
    read_only = next(row for row in unique if row["actions"] == ["READ_ONLY"])
    assert read_only["source_ranks"] == [2, 3]
    assert summary == {
        "triggered_samples": 1,
        "ranked_entries": 4,
        "unique_programs": 3,
        "duplicate_ranked_entries": 1,
        "late_trigger_exhaustive_four": 1,
    }


def test_freeze_beam_manifests_rejects_out_of_order_scores():
    beams = [
        _beam(["FULL"], -0.2),
        _beam(["READ_ONLY"], -0.1),
        _beam(["WRITE_ONLY"], -0.3),
        _beam(["IGNORE"], -0.4),
    ]
    with pytest.raises(ValueError, match="score order"):
        freeze_beam_manifests([_triggered_row(beams=beams)])


def test_compare_top1_replay_requires_exact_behavior_but_tolerates_tiny_score_error():
    expected = _triggered_row()
    actual = dict(expected)
    actual["beam_rows"] = [dict(row) for row in expected["beam_rows"]]
    actual["beam_rows"][0]["score"] += 5e-7

    parity = compare_top1_replay(expected, actual, score_abs_tolerance=1e-6)
    assert all(parity.values())

    actual["program_generated_token_ids"] = [99]
    parity = compare_top1_replay(expected, actual, score_abs_tolerance=1e-6)
    assert parity["generated_token_ids"] is False


def test_mean_pairwise_hamming_handles_single_and_multiple_programs():
    assert mean_pairwise_hamming([["FULL"]]) == 0.0
    assert mean_pairwise_hamming(
        [["FULL", "FULL"], ["FULL", "IGNORE"], ["IGNORE", "IGNORE"]]
    ) == pytest.approx(4 / 3)


def test_summarize_w_sample_separates_ranking_and_generation_failures():
    expected = _triggered_row()
    ranked, unique, _ = freeze_beam_manifests([expected])
    results = {
        row["program_id"]: {"program_id": row["program_id"], "correct": row["rank"] == 3}
        for row in ranked
    }
    summary = summarize_oracle_sample(expected, ranked, results)
    assert summary["first_correct_rank"] == 3
    assert summary["w_failure_class"] == "RANKING_FAILURE_W"
    assert summary["rescue_at_1"] is False
    assert summary["rescue_at_2"] is False
    assert summary["rescue_at_4"] is True
    assert summary["rescue_at_8"] is True

    results = {
        row["program_id"]: {"program_id": row["program_id"], "correct": False}
        for row in unique
    }
    summary = summarize_oracle_sample(expected, ranked, results)
    assert summary["first_correct_rank"] is None
    assert summary["w_failure_class"] == "GENERATION_FAILURE_W"


def test_summarize_c_regression_tracks_all_full_candidate_support():
    expected = _triggered_row(dense_correct=True)
    ranked, _, _ = freeze_beam_manifests([expected])
    results = {
        row["program_id"]: {
            "program_id": row["program_id"],
            "correct": row["actions"] == ["FULL"],
        }
        for row in ranked
    }
    summary = summarize_oracle_sample(expected, ranked, results)
    assert summary["c_failure_class"] == "TOP1_SUCCESS_C"
    assert summary["all_full_rank"] == 1

    wrong_top1 = _triggered_row(
        dense_correct=True,
        beams=[
            _beam(["IGNORE"], -0.1),
            _beam(["FULL"], -0.2),
            _beam(["READ_ONLY"], -0.3),
            _beam(["WRITE_ONLY"], -0.4),
        ],
    )
    ranked, _, _ = freeze_beam_manifests([wrong_top1])
    results = {
        row["program_id"]: {
            "program_id": row["program_id"],
            "correct": row["actions"] == ["FULL"],
        }
        for row in ranked
    }
    summary = summarize_oracle_sample(wrong_top1, ranked, results)
    assert summary["c_failure_class"] == "RANKING_FAILURE_C"
    assert summary["c_all_full_class"] == "C1_ALL_FULL_RANKED_BELOW_WRONG_TOP1"
