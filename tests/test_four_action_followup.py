from tools.research_analysis.four_action.followup import (
    select_followups,
    trajectory_reference_from_state,
)
from experiments.summarize_four_action_trajectory_rescue import (
    trajectory_result_semantically_passes,
)


def state(correct, margin=0.0):
    return {"correct": correct, "margin": margin, "generated_ids": [1]}


def sample(read=-0.1, write=-0.1, ignore=False, read_only=False, write_only=False):
    return {
        "uid": "gqa:one",
        "dataset": "gqa",
        "layers": [
            {
                "layer": 3,
                "effects": {"read_w1": read, "write_r1": write},
                "states": {
                    "IGNORE": state(ignore),
                    "READ_ONLY": state(read_only),
                    "WRITE_ONLY": state(write_only),
                    "FULL": state(False),
                },
            }
        ],
    }


def test_followup_selects_discrete_and_strong_candidates_without_duplicates():
    rows = select_followups(
        [sample(read=-2.0, write=-0.1, write_only=True)],
        {"read_w1_q90_absolute": 1.0, "write_r1_q90_absolute": 1.0},
    )
    assert len(rows) == 1
    assert rows[0]["suppressed_action"] == "WRITE_ONLY"
    assert rows[0]["culprit_operation"] == "READ"
    assert rows[0]["reasons"] == ["discrete_read_removal_rescue", "read_w1_negative_q90"]


def test_followup_selects_both_branches_for_either_removal():
    rows = select_followups(
        [sample(read_only=True, write_only=True)],
        {"read_w1_q90_absolute": 1.0, "write_r1_q90_absolute": 1.0},
    )
    assert {row["suppressed_action"] for row in rows} == {"READ_ONLY", "WRITE_ONLY"}


def test_rescue_trajectory_identity_can_be_rechecked_at_current_bf16_tolerance():
    row = {
        "passed": False,
        "checks": {
            "final_margin_matches_primary": True,
            "generated_ids_match_primary": True,
            "correctness_matches_primary": True,
            "trajectory_final_margin_matches_state": False,
        },
        "state": {"margin": 0.30497098},
        "suppressed_trajectory": {"final_margin": 0.30502860},
    }

    assert not trajectory_result_semantically_passes(row, trajectory_atol=1e-5)
    assert trajectory_result_semantically_passes(row, trajectory_atol=1e-4)


def test_trajectory_reference_keeps_baseline_target_when_valid_phrase_switches():
    state = {
        "margin": -5.050833441666327,
        "correct_target_scores": {
            "selected": {
                "text": "yes",
                "token_ids": [9693],
                "mean_logprob": -6.5240278244018555,
            },
            "candidates": [
                {
                    "text": "not question",
                    "token_ids": [1921, 3405],
                    "mean_logprob": -7.015852451324463,
                },
                {
                    "text": "yes",
                    "token_ids": [9693],
                    "mean_logprob": -6.5240278244018555,
                },
            ],
        },
        "full_wrong_target_score": {
            "text": "Sparkling Black Cherry",
            "token_ids": [67483, 2718, 5235, 44705],
            "mean_logprob": -1.473194382735528,
        },
    }
    trajectory = {
        "correct_target_text": "not question",
        "correct_target_token_ids": [1921, 3405],
        "wrong_target_text": "Sparkling Black Cherry",
        "wrong_target_token_ids": [67483, 2718, 5235, 44705],
        "final_margin": -5.542658068588935,
    }

    reference = trajectory_reference_from_state(state, trajectory)

    assert reference["correct_target_switched"] is True
    assert reference["state_selected_correct_target_text"] == "yes"
    assert reference["fixed_correct_target_text"] == "not question"
    assert reference["fixed_target_state_margin"] == -5.542658068588935
