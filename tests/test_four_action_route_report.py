from __future__ import annotations

from experiments.write_route_conditioned_final_report import render_report


def test_final_report_explicitly_answers_all_required_questions_and_context_boundary():
    anchor = {
        "frozen_a_plus_count": 10,
        "validated_anchor_count": 9,
        "excluded_no_current_correct_anchor_count": 1,
        "dataset_counts": {"gqa": 6, "textvqa": 3},
        "fallback_sample_count": 2,
        "expected_new_cells_3k": 90,
    }
    pilot = {
        "selected_configuration": "one_replica",
        "selected_replicas_per_gpu": 1,
        "configurations": [
            {
                "name": "one_replica",
                "useful_new_cells_per_second": 2.0,
                "useful_samples_per_second": 0.2,
                "gpu_metrics": {
                    "peak_memory_used_mib": 20000,
                    "mean_gpu_utilization_percent": 50.0,
                },
            }
        ],
    }
    stage = {"sample_count": 9, "anchor_off_position_count": 30, "flat_action_row_count": 120}
    aggregate = {
        "taxonomy": {
            "individually_necessary": {"count": 12, "fraction": 0.4, "ci_low": 0.3, "ci_high": 0.5},
            "redundant": {"count": 18, "fraction": 0.6, "ci_low": 0.5, "ci_high": 0.7},
            "read_mediated": {"count": 3, "conditional_among_necessary_fraction": 0.25, "conditional_among_necessary_ci_low": 0.1, "conditional_among_necessary_ci_high": 0.4},
            "write_mediated": {"count": 3, "conditional_among_necessary_fraction": 0.25, "conditional_among_necessary_ci_low": 0.1, "conditional_among_necessary_ci_high": 0.4},
            "either_removal_sufficient": {"count": 2, "conditional_among_necessary_fraction": 1 / 6, "conditional_among_necessary_ci_low": 0.05, "conditional_among_necessary_ci_high": 0.3},
            "both_required": {"count": 4, "conditional_among_necessary_fraction": 1 / 3, "conditional_among_necessary_ci_low": 0.15, "conditional_among_necessary_ci_high": 0.5},
        },
        "category_depth_comparison": {
            "joint": {
                "read_mediated_mean_layer": 8.0,
                "write_mediated_mean_layer": 16.0,
                "read_minus_write_mean_layer": -8.0,
                "read_minus_write_ci_low": -10.0,
                "read_minus_write_ci_high": -5.0,
            }
        },
        "sample_structure_counts": {"mixed_read_write": 4},
        "context_comparison": {
            "joint": {
                "matched_cell_count": 30,
                "discrete_context_agreement_fraction": 0.7,
                "route_necessity_recall_from_full_context": 0.5,
                "full_context_rescue_precision_for_route_necessity": 0.6,
                "route_necessary_full_context_missed_count": 6,
                "route_necessary_full_context_missed_fraction": 0.5,
                "read_harm_sign_agreement_fraction": 0.55,
                "write_harm_sign_agreement_fraction": 0.65,
                "read_effect_spearman": 0.1,
                "write_effect_spearman": 0.2,
            },
            "within_sample": {
                "eligible_sample_count": 8,
                "median_read_spearman": 0.0,
                "median_write_spearman": 0.1,
            },
        },
    }
    route_size = [
        {"dataset": "joint", "route_size_stratum": "2-4", "sample_count": 3, "off_position_count": 8, "necessary_fraction": 0.75, "redundant_fraction": 0.25},
        {"dataset": "joint", "route_size_stratum": ">16", "sample_count": 2, "off_position_count": 20, "necessary_fraction": 0.25, "redundant_fraction": 0.75},
    ]
    continuous = [
        {"dataset": "joint", "taxonomy": "all", "effect": "read_w0", "estimate": -0.1, "median": -0.05, "negative_fraction": 0.6},
    ]
    estimate = {"expected_wall_hours": 1.0, "expected_gpu_hours": 8.0}

    report = render_report(anchor, pilot, stage, aggregate, route_size, continuous, estimate)

    for number in range(1, 10):
        assert f"### {number}." in report
    assert "FULL-context" in report
    assert "route-conditioned" in report
    assert "12/30" in report
    assert "Do not launch" in report
