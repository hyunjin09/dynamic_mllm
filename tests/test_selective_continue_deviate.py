from __future__ import annotations

import pytest

from four_action_policy.selective_continue_deviate import (
    bootstrap_uid_rescue_rate,
    build_full_insertion_subset,
    summarize_full_insertion_audit,
)


def test_full_insertion_subset_uses_every_compatible_suffix_and_deduplicates() -> None:
    rows = [
        {
            "uid": "u0",
            "split": "validation",
            "dataset": "gqa",
            "route_type": "W2C",
            "valid_routes": [
                {"route_key": "a", "actions": ["FULL", "IGNORE"] + ["FULL"] * 26},
                {"route_key": "b", "actions": ["FULL", "READ_ONLY"] + ["FULL"] * 26},
                {"route_key": "c", "actions": ["FULL", "IGNORE"] + ["FULL"] * 26},
                {"route_key": "not-compatible", "actions": ["IGNORE"] + ["FULL"] * 27},
            ],
        }
    ]
    boundaries = [
        {
            "uid": "u0",
            "boundary_layer": 1,
            "all_full_prefix_length": 1,
            "valid_nonfull_actions": ["IGNORE", "READ_ONLY"],
            "boundary_route_indices": [0, 1, 2],
        }
    ]

    subset, audit = build_full_insertion_subset(rows, boundaries, split="validation")

    assert audit["states"] == 1
    assert audit["candidate_routes"] == 1
    assert audit["deduplicated_source_routes"] == 2
    state = subset[0]
    assert state["known_mechanism"] == "MULTI"
    assert state["depth_bin"] == "early"
    assert state["compatible_suffix_count"] == 3
    assert state["candidate_route_count"] == 1
    assert state["candidate_routes"][0]["actions"] == ["FULL"] * 28
    assert state["candidate_routes"][0]["source_route_indices"] == [0, 1, 2]


def test_full_insertion_subset_rejects_incomplete_boundary_route_coverage() -> None:
    rows = [
        {
            "uid": "u0",
            "split": "validation",
            "dataset": "chartqa",
            "route_type": "W2C",
            "valid_routes": [
                {"actions": ["IGNORE", "FULL"]},
                {"actions": ["READ_ONLY", "FULL"]},
            ],
        }
    ]
    boundaries = [
        {
            "uid": "u0",
            "boundary_layer": 0,
            "all_full_prefix_length": 0,
            "valid_nonfull_actions": ["IGNORE", "READ_ONLY"],
            "boundary_route_indices": [0],
        }
    ]

    with pytest.raises(ValueError, match="exactly enumerate compatible"):
        build_full_insertion_subset(rows, boundaries, split="validation")


def test_full_insertion_summary_classifies_at_state_level() -> None:
    subset = [
        {
            "state_id": "s0",
            "uid": "u0",
            "dataset": "gqa",
            "depth_bin": "early",
            "known_mechanism": "IGNORE",
            "suffix_set_complete": True,
            "candidate_routes": [{"candidate_index": 0}, {"candidate_index": 1}],
        },
        {
            "state_id": "s1",
            "uid": "u1",
            "dataset": "textvqa",
            "depth_bin": "late",
            "known_mechanism": "WRITE_ONLY",
            "suffix_set_complete": True,
            "candidate_routes": [{"candidate_index": 0}],
        },
    ]
    executions = [
        {"state_id": "s0", "candidate_index": 0, "correct": False},
        {"state_id": "s0", "candidate_index": 1, "correct": True},
        {"state_id": "s1", "candidate_index": 0, "correct": False},
    ]

    result = summarize_full_insertion_audit(
        subset, executions, bootstrap_draws=100, bootstrap_seed=11
    )

    assert result["states"] == 2
    assert result["candidate_executions"] == 3
    assert result["status_counts"] == {
        "FULL-cache-incomplete": 1,
        "FULL-confirmed-invalid": 1,
    }
    assert result["overall"]["rescue_rate"] == pytest.approx(0.5)
    rescued = next(row for row in result["state_results"] if row["uid"] == "u0")
    assert rescued["status"] == "FULL-cache-incomplete"
    assert rescued["successful_candidate_indices"] == [1]
    assert result["by_dataset"]["gqa"]["rescue_rate"] == pytest.approx(1.0)


def test_uid_bootstrap_is_deterministic_and_groups_duplicate_uid_rows() -> None:
    rows = [
        {"uid": "u0", "rescued": True},
        {"uid": "u0", "rescued": True},
        {"uid": "u1", "rescued": False},
    ]

    first = bootstrap_uid_rescue_rate(rows, draws=1000, seed=7)
    second = bootstrap_uid_rescue_rate(rows, draws=1000, seed=7)

    assert first == second
    assert first["uids"] == 2
    assert first["estimate"] == pytest.approx(0.5)
    assert first["draws"] == 1000
