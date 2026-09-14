from __future__ import annotations

import math

import pytest
import torch

from dense_failure_stage2.teacher_forced_free_run import (
    action_diagnostics,
    classify_bottleneck,
    first_non_full_offset,
    prefix_support_trace,
    score_cached_uid,
    select_reference_programs,
    support_survival,
)


def _program(program_id: str, actions: list[str]) -> dict:
    return {"program_id": program_id, "actions": actions}


def test_action_diagnostics_reports_probability_rank_margin_and_top1():
    row = action_diagnostics([2.0, 1.0, -1.0, 0.0], target_action="READ_ONLY")

    normalizer = math.exp(2.0) + math.exp(1.0) + math.exp(-1.0) + math.exp(0.0)
    assert row["top1_action"] == "FULL"
    assert row["target_rank"] == 2
    assert row["target_probability"] == pytest.approx(math.exp(1.0) / normalizer)
    assert row["target_vs_full_margin"] == pytest.approx(-1.0)
    assert row["probabilities"]["FULL"] == pytest.approx(math.exp(2.0) / normalizer)


def test_select_reference_programs_uses_highest_responsibility_and_stable_tie_break():
    rows = [
        {"uid": "w0", "program_id": "p2", "responsibility": 0.7},
        {"uid": "w0", "program_id": "p1", "responsibility": 0.7},
        {"uid": "w0", "program_id": "p0", "responsibility": 0.1},
        {"uid": "w1", "program_id": "q0", "responsibility": 1.0},
    ]

    selected = select_reference_programs(rows)

    assert selected == {"w0": "p1", "w1": "q0"}


def test_prefix_support_trace_respects_union_of_compatible_successful_routes():
    programs = [
        _program("p0", ["FULL", "READ_ONLY", "FULL"]),
        _program("p1", ["FULL", "WRITE_ONLY", "IGNORE"]),
    ]

    trace, first = prefix_support_trace(
        ["FULL", "WRITE_ONLY", "FULL"],
        programs,
        trigger_layer=20,
        probabilities=[
            {"FULL": 0.8, "READ_ONLY": 0.1, "WRITE_ONLY": 0.05, "IGNORE": 0.05},
            {"FULL": 0.2, "READ_ONLY": 0.3, "WRITE_ONLY": 0.4, "IGNORE": 0.1},
            {"FULL": 0.7, "READ_ONLY": 0.1, "WRITE_ONLY": 0.1, "IGNORE": 0.1},
        ],
    )

    assert trace[0]["compatible_program_ids_before"] == ["p0", "p1"]
    assert trace[1]["supported_actions"] == ["READ_ONLY", "WRITE_ONLY"]
    assert trace[1]["supported_probability_mass"] == pytest.approx(0.7)
    assert first["layer"] == 22
    assert first["chosen_action"] == "FULL"
    assert first["supported_actions"] == ["IGNORE"]
    assert first["supported_probability_mass"] == pytest.approx(0.1)
    assert first["compatible_route_count_before"] == 1


def test_prefix_support_trace_reports_none_when_one_route_survives_to_end():
    programs = [
        _program("p0", ["FULL", "READ_ONLY"]),
        _program("p1", ["WRITE_ONLY", "FULL"]),
    ]
    trace, first = prefix_support_trace(
        ["WRITE_ONLY", "FULL"], programs, trigger_layer=26
    )

    assert first is None
    assert trace[-1]["supported"] is True
    assert trace[-1]["compatible_program_ids_after"] == ["p1"]


def test_support_survival_counts_zero_depth_and_permanent_off_support():
    traces = [
        [{"depth_after_trigger": 0, "supported": True}, {"depth_after_trigger": 1, "supported": False}],
        [{"depth_after_trigger": 0, "supported": True}, {"depth_after_trigger": 1, "supported": True}],
    ]
    assert support_survival(traces) == [
        {"depth_after_trigger": 0, "eligible_uids": 2, "supported_uids": 2, "survival": 1.0},
        {"depth_after_trigger": 1, "eligible_uids": 2, "supported_uids": 1, "survival": 0.5},
    ]


def test_score_cached_uid_forwards_unique_states_once_and_scores_every_route_occurrence():
    class CountingRouter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forwarded = 0

        def forward(self, text_states, visual_states, *, text_mask, visual_mask):
            self.forwarded += int(text_states.shape[0])
            value = text_states[:, 0, 0].float()
            return torch.stack((value, -value, value * 0.5, torch.zeros_like(value)), dim=-1)

    def state(value: float, layer: int):
        return {
            "text_states": torch.tensor([[[value]]], dtype=torch.bfloat16),
            "visual_states": torch.tensor([[[1.0]]], dtype=torch.bfloat16),
            "text_mask": torch.ones(1, 1, dtype=torch.bool),
            "visual_mask": torch.ones(1, 1, dtype=torch.bool),
            "layer": layer,
            "prefix_actions": [],
        }

    payload = {
        "uid": "w0",
        "trigger_layer": 26,
        "dense_outcome": "W",
        "dataset": "gqa",
        "source_regime": "canonical",
        "image_group_id": "image0",
        "states": {"s0": state(1.0, 26), "s1": state(-1.0, 27)},
        "programs": [
            {"program_id": "p0", "state_ids": ["s0", "s1"],
             "actions": ["FULL", "IGNORE"], "provenance": ["single"]},
            {"program_id": "p1", "state_ids": ["s0", "s1"],
             "actions": ["READ_ONLY", "FULL"], "provenance": ["original_mcts"]},
        ],
    }
    router = CountingRouter()

    result = score_cached_uid(router, payload, device=torch.device("cpu"), state_microbatch=1)

    assert router.forwarded == 2
    assert len(result["state_logits"]) == 2
    assert len(result["occurrences"]) == 4
    assert len(result["programs"]) == 2
    assert sum(row["responsibility"] for row in result["programs"]) == pytest.approx(1.0)
    assert result["occurrences"][0]["uid"] == "w0"
    assert {row["target_action"] for row in result["occurrences"]} == {
        "FULL", "IGNORE", "READ_ONLY"
    }

def test_first_non_full_offset_and_release_boundaries_are_unambiguous():
    assert first_non_full_offset(["FULL", "IGNORE", "FULL"]) == 1
    with pytest.raises(ValueError, match="no non-FULL"):
        first_non_full_offset(["FULL", "FULL"])


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        (
            {"seen_first_nonfull_recall": 0.2, "r0_success": 0.0, "r1_success": 0.01,
             "r2_success": 0.2, "heldout_first_nonfull_recall": 0.1,
             "pre_intervention_off_support_fraction": 0.1},
            "objective_action_learning",
        ),
        (
            {"seen_first_nonfull_recall": 0.8, "r0_success": 0.1, "r1_success": 0.12,
             "r2_success": 0.14, "heldout_first_nonfull_recall": 0.5,
             "pre_intervention_off_support_fraction": 0.1},
            "generalization",
        ),
        (
            {"seen_first_nonfull_recall": 0.8, "r0_success": 0.1, "r1_success": 0.3,
             "r2_success": 0.31, "heldout_first_nonfull_recall": 0.75,
             "pre_intervention_off_support_fraction": 0.7},
            "pre_intervention_exposure",
        ),
        (
            {"seen_first_nonfull_recall": 0.8, "r0_success": 0.1, "r1_success": 0.12,
             "r2_success": 0.15, "heldout_first_nonfull_recall": 0.75,
             "pre_intervention_off_support_fraction": 0.2,
             "post_intervention_off_support_fraction": 0.7},
            "post_intervention_exposure",
        ),
    ],
)
def test_classify_bottleneck_uses_fixed_prospective_thresholds(metrics, expected):
    thresholds = {
        "high_first_nonfull_recall": 0.5,
        "material_release_delta": 0.05,
        "material_generalization_drop": 0.15,
        "majority_fraction": 0.5,
        "adequate_release_success": 0.5,
    }
    assert classify_bottleneck(metrics, thresholds)["case"] == expected
