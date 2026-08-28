from experiments.analyze_four_action_results import (
    distance_association_summary,
    drift_summary,
    per_sample_effect_summary,
    validation_semantic_summary,
)


def test_validation_semantic_summary_counts_both_external_comparisons():
    comparison = {
        "generated_ids_match": True,
        "generated_answer_match": True,
        "evaluator_score_match": True,
        "correctness_match": True,
    }
    samples = [
        {
            "native_full_external": {"diagnostic": comparison},
            "layers": [
                {"old_binary_ignore_external": comparison},
                {"old_binary_ignore_external": comparison},
            ],
        }
    ]
    summary = validation_semantic_summary(samples)
    assert summary["validation_sample_count"] == 1
    assert summary["unified_full_vs_native"]["generated_ids_match_count"] == 1
    ignore = summary["unified_ignore_vs_old_binary_single_off"]
    assert ignore["comparisons"] == 2
    assert ignore["correctness_match_count"] == 2


def test_per_sample_effect_summary_counts_harmful_and_rescue_layers():
    rows = [
        {
            "uid": "gqa:one",
            "dataset": "gqa",
            "image_group_id": "image",
            "nearest_correcting_route_distance": 2,
            "hamming_stratum": "2",
            "read_w1": -2.0,
            "write_r1": 1.0,
            "interaction": 0.5,
            "rescue_category": "read_removal_only",
        },
        {
            "uid": "gqa:one",
            "dataset": "gqa",
            "image_group_id": "image",
            "nearest_correcting_route_distance": 2,
            "hamming_stratum": "2",
            "read_w1": 0.5,
            "write_r1": -1.0,
            "interaction": -0.25,
            "rescue_category": "no_local_rescue",
        },
    ]
    summary = per_sample_effect_summary(rows)[0]
    assert summary["negative_read_layer_count"] == 1
    assert summary["negative_write_layer_count"] == 1
    assert summary["negative_either_layer_count"] == 2
    assert summary["rescue_layer_count"] == 1
    assert summary["strongest_negative_component_magnitude"] == 2.0
    associations = distance_association_summary([summary])
    assert {row["dataset"] for row in associations} == {"gqa", "joint"}


def test_drift_summary_handles_mixed_wrong_target_availability():
    rows = [
        {
            "analysis_set": "production",
            "cohort": "primary_a_plus",
            "dataset": "gqa",
            "S_correct_signed_drift": 0.1,
            "S_full_wrong_signed_drift": 0.2,
            "margin_signed_drift": -0.1,
        },
        {
            "analysis_set": "production",
            "cohort": "vision_required",
            "dataset": "gqa",
            "S_correct_signed_drift": 0.3,
            "margin_signed_drift": 0.3,
        },
    ]
    summary = drift_summary(rows)
    wrong_all = next(
        row
        for row in summary
        if row["cohort"] == "all"
        and row["dataset"] == "joint"
        and row["quantity"] == "S_full_wrong"
    )
    assert wrong_all["signed"]["count"] == 1
