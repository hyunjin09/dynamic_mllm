from __future__ import annotations

from tools.research_analysis.four_action.sequential_label_analysis import aggregate_records


def test_aggregate_separates_w2c_mechanisms_from_c2c_preservation():
    correct = {"correct": True, "generated_answer": "yes", "answer_alignment_margin": 1.0}
    records = [
        {
            "dataset": "gqa",
            "route_type": "W2C",
            "source_positive_route_count": 1,
            "source_route_replay_valid_count": 1,
            "source_route_replay_failure_count": 0,
            "raw_conversions": [
                {
                    "status": "converted",
                    "source_binary_route": [0, 0, 1],
                    "source_off_count": 2,
                    "all_off_seed": False,
                    "maximum_branch_count": 2,
                    "steps": [
                        {"both_partial_correct_count": 1},
                        {"full_restored_count": 2},
                    ],
                    "final_branches": [
                        {"route": ["READ_ONLY", "FULL", "FULL"], "evaluation": correct},
                        {"route": ["WRITE_ONLY", "FULL", "FULL"], "evaluation": correct},
                    ],
                }
            ],
            "unique_valid_four_action_routes": [
                {"label_semantics": "corrective_w2c", "four_action_route": ["READ_ONLY", "FULL", "FULL"]},
                {"label_semantics": "corrective_w2c", "four_action_route": ["WRITE_ONLY", "FULL", "FULL"]},
            ],
        },
        {
            "dataset": "textvqa",
            "route_type": "C2C",
            "source_positive_route_count": 1,
            "source_route_replay_valid_count": 1,
            "source_route_replay_failure_count": 0,
            "raw_conversions": [
                {
                    "status": "converted",
                    "source_binary_route": [0, 1, 1],
                    "source_off_count": 1,
                    "all_off_seed": False,
                    "maximum_branch_count": 1,
                    "steps": [],
                    "final_branches": [
                        {"route": ["IGNORE", "FULL", "FULL"], "evaluation": correct}
                    ],
                }
            ],
            "unique_valid_four_action_routes": [
                {"label_semantics": "preserving_c2c", "four_action_route": ["IGNORE", "FULL", "FULL"]}
            ],
        },
    ]

    result = aggregate_records(records, layer_count=3)
    combined = result["combined"]

    assert combined["counts"]["w2c_source_routes"] == 1
    assert combined["counts"]["c2c_source_routes"] == 1
    assert combined["w2c"]["off_position_final_actions"] == {
        "FULL": 2,
        "READ_ONLY": 1,
        "WRITE_ONLY": 1,
        "IGNORE": 0,
    }
    assert combined["w2c"]["source_routes_with_branching"] == 1
    assert combined["w2c"]["both_partial_branch_events"] == 1
    assert combined["c2c"]["action_counts"] == {
        "FULL": 2,
        "READ_ONLY": 0,
        "WRITE_ONLY": 0,
        "IGNORE": 1,
    }
    assert result["by_dataset"]["gqa"]["counts"]["c2c_source_routes"] == 0
    assert result["by_dataset"]["textvqa"]["counts"]["w2c_source_routes"] == 0
