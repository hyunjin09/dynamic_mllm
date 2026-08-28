from __future__ import annotations

from collections import Counter

import pytest

from tools.research_analysis.four_action.route_conditioned import (
    balance_work_units,
    build_anchor_candidate_rows,
    classify_route_conditioned_cell,
    evaluate_until_current_correct,
    finalize_anchor_rows,
    flatten_route_conditioned_samples,
    ordered_anchor_candidates,
    select_current_anchor,
    select_execution_rows,
    select_stratified_pilot,
    choose_pilot_configuration,
    summarize_gpu_metrics,
)


def test_candidate_manifest_uses_only_frozen_current_eligible_primary_rows():
    route = {
        "route_id": "r1",
        "mask": [1, 0, 1],
        "hamming_distance_to_full": 1,
        "score": 1.0,
    }
    cohort = [
        {"uid": "keep", "dataset": "gqa", "cohort": "primary_a_plus", "binary_routes": {"correcting_routes": [route]}},
        {"uid": "stale", "dataset": "gqa", "cohort": "primary_a_plus", "binary_routes": {"correcting_routes": [route]}},
        {"uid": "control", "dataset": "gqa", "cohort": "control", "binary_routes": None},
    ]
    eligibility = [
        {"uid": "keep", "eligible": True},
        {"uid": "stale", "eligible": False},
        {"uid": "control", "eligible": True},
    ]

    rows = build_anchor_candidate_rows(cohort, eligibility, layer_count=3)

    assert [row["uid"] for row in rows] == ["keep"]
    assert rows[0]["schema_version"] == "route_conditioned_anchor_candidates_v1"
    assert rows[0]["anchor_candidates"][0]["route_id"] == "r1"
    assert rows[0]["historical_minimum_off_count"] == 1


def test_anchor_merger_requires_complete_unique_validation_and_preserves_exclusions():
    route = {
        "route_id": "r1",
        "mask": [1, 0, 1],
        "hamming_distance_to_full": 1,
        "score": 1.0,
        "candidate_rank": 0,
        "minimum_distance_tie": True,
        "fallback_count": 0,
        "current_state": {"correct": True, "margin": 1.25},
    }
    candidates = [
        {"uid": "keep", "dataset": "gqa", "anchor_candidates": [route]},
        {"uid": "exclude", "dataset": "textvqa", "anchor_candidates": [route]},
    ]
    validations = [
        {
            "uid": "keep",
            "passed": True,
            "analyzable": True,
            "anchor": route,
            "anchor_off_layers": [1],
            "candidate_evaluations": [{"route_id": "r1", "correct": True}],
        },
        {
            "uid": "exclude",
            "passed": True,
            "analyzable": False,
            "exclusion_reason": "no_cached_correcting_route_current_correct",
            "candidate_evaluations": [{"route_id": "r1", "correct": False}],
        },
    ]

    anchors, exclusions = finalize_anchor_rows(candidates, validations)

    assert [row["uid"] for row in anchors] == ["keep"]
    assert anchors[0]["anchor_route_id"] == "r1"
    assert anchors[0]["anchor_off_layers"] == [1]
    assert anchors[0]["anchor_off_count"] == 1
    assert anchors[0]["anchor_current_state"]["margin"] == 1.25
    assert anchors[0]["minimum_distance_tied_candidates"][0]["route_id"] == "r1"
    assert exclusions[0]["uid"] == "exclude"
    assert exclusions[0]["exclusion_reason"] == "no_cached_correcting_route_current_correct"

    with pytest.raises(ValueError, match="duplicate validation"):
        finalize_anchor_rows(candidates, [validations[0], validations[0]])
    with pytest.raises(ValueError, match="missing validation"):
        finalize_anchor_rows(candidates, validations[:1])


def test_anchor_candidates_use_distance_score_then_stable_identity():
    routes = [
        {"route_id": "z", "mask": [1, 0, 1, 0], "hamming_distance_to_full": 2, "score": 0.6},
        {"route_id": "b", "mask": [0, 1, 1, 0], "hamming_distance_to_full": 2, "score": 1.0},
        {"route_id": "a", "mask": [1, 1, 0, 0], "hamming_distance_to_full": 2, "score": 1.0},
        {"route_id": "nearer", "mask": [1, 1, 1, 0], "hamming_distance_to_full": 1, "score": 0.5},
        {"route_id": "far", "mask": [0, 0, 1, 0], "hamming_distance_to_full": 3, "score": 1.0},
    ]

    ordered = ordered_anchor_candidates(routes, layer_count=4)

    assert [row["route_id"] for row in ordered] == ["nearer", "a", "b", "z", "far"]
    assert [row["candidate_rank"] for row in ordered] == [0, 1, 2, 3, 4]
    assert [row["minimum_distance_tie"] for row in ordered] == [True, False, False, False, False]


def test_anchor_candidate_validation_falls_back_without_inventing_a_route():
    candidates = ordered_anchor_candidates(
        [
            {"route_id": "first", "mask": [1, 0, 1], "hamming_distance_to_full": 1, "score": 1.0},
            {"route_id": "second", "mask": [0, 1, 1], "hamming_distance_to_full": 1, "score": 0.5},
        ],
        layer_count=3,
    )
    evaluations = {
        "first": {"correct": False, "generated_answer": "wrong"},
        "second": {"correct": True, "generated_answer": "right"},
    }

    selected = select_current_anchor(candidates, evaluations)

    assert selected["route_id"] == "second"
    assert selected["candidate_rank"] == 1
    assert selected["fallback_count"] == 1
    assert selected["current_state"]["generated_answer"] == "right"
    assert select_current_anchor(candidates, {key: {"correct": False} for key in evaluations}) is None


def test_candidate_runtime_evaluation_stops_at_first_current_correct_route():
    candidates = ordered_anchor_candidates(
        [
            {"route_id": "first", "mask": [1, 0, 1], "hamming_distance_to_full": 1, "score": 1.0},
            {"route_id": "second", "mask": [0, 1, 1], "hamming_distance_to_full": 1, "score": 0.5},
            {"route_id": "third", "mask": [1, 1, 0], "hamming_distance_to_full": 1, "score": 0.4},
        ],
        layer_count=3,
    )
    seen = []

    def evaluate(candidate):
        seen.append(candidate["route_id"])
        return {"correct": candidate["route_id"] == "second"}

    result = evaluate_until_current_correct(candidates, evaluate)

    assert seen == ["first", "second"]
    assert result["selected"]["route_id"] == "second"
    assert result["selected"]["fallback_count"] == 1
    assert [row["route_id"] for row in result["evaluations"]] == ["first", "second"]


def test_stratified_pilot_selects_both_datasets_and_three_off_count_strata():
    rows = [
        {
            "uid": f"{dataset}:{index:03d}",
            "dataset": dataset,
            "anchor_off_count": (index % 18) + 2,
        }
        for dataset in ("gqa", "textvqa")
        for index in range(90)
    ]

    selected = select_stratified_pilot(rows, total=56)

    assert len(selected) == 56
    assert len({row["uid"] for row in selected}) == 56
    assert Counter(row["dataset"] for row in selected) == {"gqa": 28, "textvqa": 28}
    assert Counter((row["dataset"], row["pilot_off_count_stratum"]) for row in selected) == {
        ("gqa", "small"): 10,
        ("gqa", "medium"): 9,
        ("gqa", "large"): 9,
        ("textvqa", "small"): 10,
        ("textvqa", "medium"): 9,
        ("textvqa", "large"): 9,
    }


def test_work_units_are_complete_deterministic_and_cost_balanced():
    rows = [
        {"uid": f"u{index}", "anchor_off_count": cost}
        for index, cost in enumerate([22, 18, 14, 12, 11, 9, 8, 7, 6, 4, 3, 2])
    ]

    first = balance_work_units(rows, work_unit_count=5)
    second = balance_work_units(list(reversed(rows)), work_unit_count=5)

    assert first == second
    assigned = [row["uid"] for unit in first for row in unit["samples"]]
    assert sorted(assigned) == sorted(row["uid"] for row in rows)
    assert len(assigned) == len(set(assigned))
    costs = [unit["expected_new_cells"] for unit in first]
    assert max(costs) - min(costs) <= 3 * max(row["anchor_off_count"] for row in rows)
    assert all(row["work_unit_id"] == unit["work_unit_id"] for unit in first for row in unit["samples"])


def test_execution_rows_preserve_cost_units_and_split_gpu_replicas_without_overlap():
    full_rows = [
        {"uid": f"u{index}", "work_unit_id": f"work_unit_{index:03d}"}
        for index in range(20)
    ]
    gpu_zero = [
        select_execution_rows(
            full_rows,
            mode="full",
            gpu_index=0,
            replica_index=replica,
            replicas_per_gpu=2,
        )
        for replica in range(2)
    ]
    expected = {"u0", "u8", "u16"}
    assert {row["uid"] for part in gpu_zero for row in part} == expected
    assert sum(len(part) for part in gpu_zero) == len(expected)

    pilot_rows = [
        {"uid": f"p{index}", "pilot_worker_index": index % 8}
        for index in range(24)
    ]
    selected = select_execution_rows(
        pilot_rows,
        mode="pilot",
        gpu_index=3,
        replica_index=0,
        replicas_per_gpu=1,
    )
    assert [row["uid"] for row in selected] == ["p3", "p11", "p19"]


@pytest.mark.parametrize(
    ("correctness", "expected"),
    [
        ({"IGNORE": True, "FULL": True, "WRITE_ONLY": False, "READ_ONLY": False}, "redundant"),
        ({"IGNORE": True, "FULL": False, "WRITE_ONLY": True, "READ_ONLY": False}, "read_mediated"),
        ({"IGNORE": True, "FULL": False, "WRITE_ONLY": False, "READ_ONLY": True}, "write_mediated"),
        ({"IGNORE": True, "FULL": False, "WRITE_ONLY": True, "READ_ONLY": True}, "either_removal_sufficient"),
        ({"IGNORE": True, "FULL": False, "WRITE_ONLY": False, "READ_ONLY": False}, "both_required"),
        ({"IGNORE": False, "FULL": False, "WRITE_ONLY": False, "READ_ONLY": False}, "inconsistent_anchor"),
    ],
)
def test_route_conditioned_taxonomy_is_mutually_exclusive(correctness, expected):
    assert classify_route_conditioned_cell(correctness) == expected


def test_route_conditioned_flattening_emits_all_four_named_action_rows():
    state = {
        "generated_answer": "answer",
        "generated_ids": [1, 2],
        "correctness_score": 1.0,
        "correct": True,
        "S_correct": -1.0,
        "S_full_wrong": -2.0,
        "margin": 1.0,
    }
    sample = {
        "uid": "gqa:one",
        "dataset": "gqa",
        "image_id": "image",
        "image_group_id": "group",
        "work_unit_id": "work_unit_000",
        "anchor_route_id": "route",
        "anchor_route_mask": [1, 0, 1],
        "anchor_off_count": 1,
        "anchor_hamming_distance_from_full": 1,
        "fixed_correct_target_text": "answer",
        "fixed_wrong_target_text": "wrong",
        "worker": {"rank": 0, "gpu_index": 0, "replica_index": 0},
        "cells": [
            {
                "target_layer": 1,
                "taxonomy": "both_required",
                "effects": {
                    "read_w0": 0.1,
                    "read_w1": 0.2,
                    "write_r0": 0.3,
                    "write_r1": 0.4,
                    "interaction": 0.5,
                },
                "states": {action: dict(state) for action in ("IGNORE", "READ_ONLY", "WRITE_ONLY", "FULL")},
                "elapsed_seconds": 1.5,
            }
        ],
    }

    rows = flatten_route_conditioned_samples([sample])

    assert len(rows) == 4
    assert {(row["action"], row["route_action_name"], row["factorial_state"]) for row in rows} == {
        ("IGNORE", "BOTH_OFF", "M00"),
        ("READ_ONLY", "WRITE_OFF", "M10"),
        ("WRITE_ONLY", "READ_OFF", "M01"),
        ("FULL", "FULL_RESTORE", "M11"),
    }
    assert all(row["taxonomy"] == "both_required" for row in rows)
    assert all(row["S_original_full_wrong"] == -2.0 for row in rows)
    assert sum(row["new_evaluation"] for row in rows) == 3


def test_pilot_configuration_uses_valid_cells_per_second_not_replica_count():
    selected = choose_pilot_configuration(
        [
            {"name": "one", "passed": True, "replicas_per_gpu": 1, "useful_new_cells_per_second": 10.0},
            {"name": "two", "passed": True, "replicas_per_gpu": 2, "useful_new_cells_per_second": 8.0},
        ]
    )
    assert selected["name"] == "one"
    with pytest.raises(ValueError, match="no passing"):
        choose_pilot_configuration([{"name": "bad", "passed": False}])


def test_gpu_metric_summary_reports_peak_memory_and_utilization_distribution():
    rows = [
        {"gpu_index": "0", "memory_used_mib": "100", "utilization_gpu_percent": "10"},
        {"gpu_index": "0", "memory_used_mib": "200", "utilization_gpu_percent": "90"},
        {"gpu_index": "1", "memory_used_mib": "150", "utilization_gpu_percent": "50"},
    ]
    summary = summarize_gpu_metrics(rows)
    assert summary["peak_memory_used_mib"] == 200.0
    assert summary["mean_gpu_utilization_percent"] == 50.0
    assert summary["per_gpu_peak_memory_used_mib"] == {"0": 200.0, "1": 150.0}
