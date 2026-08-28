from __future__ import annotations

import pandas as pd

from experiments.analyze_route_conditioned_results import (
    _category_depth_comparison,
    _taxonomy_summary,
    cluster_bootstrap_mean,
    context_comparison_table,
    route_size_stratum,
    sample_structure_category,
)


def test_route_size_strata_cover_observed_anchor_range():
    assert [route_size_stratum(value) for value in (2, 4, 5, 8, 9, 12, 13, 16, 17)] == [
        "2-4",
        "2-4",
        "5-8",
        "5-8",
        "9-12",
        "9-12",
        "13-16",
        "13-16",
        ">16",
    ]


def test_sample_structure_category_distinguishes_simple_mixed_and_joint_routes():
    assert sample_structure_category({"read_mediated": 1}) == "one_dominant_operation"
    assert sample_structure_category({"read_mediated": 2}) == "multiple_read_mediated"
    assert sample_structure_category({"write_mediated": 2}) == "multiple_write_mediated"
    assert sample_structure_category({"read_mediated": 1, "write_mediated": 1}) == "mixed_read_write"
    assert sample_structure_category({"both_required": 2}) == "joint_both_suppression"
    assert sample_structure_category({"both_required": 1}) == "joint_both_suppression"
    assert sample_structure_category({"either_removal_sufficient": 1}) == "either_only_ambiguous"
    assert sample_structure_category({"redundant": 4}) == "no_essential_off"


def test_taxonomy_summary_reports_all_off_and_necessary_only_denominators():
    cells = pd.DataFrame(
        [
            {"dataset": "gqa", "image_group_id": "a", "taxonomy": "redundant"},
            {"dataset": "gqa", "image_group_id": "b", "taxonomy": "read_mediated"},
            {"dataset": "gqa", "image_group_id": "c", "taxonomy": "write_mediated"},
        ]
    )

    summary = _taxonomy_summary(cells)
    joint = summary[summary.dataset == "joint"].set_index("metric")

    assert joint.loc["individually_necessary", "estimate"] == 2 / 3
    assert joint.loc["read_mediated", "estimate"] == 1 / 3
    assert joint.loc["read_mediated", "conditional_necessary_estimate"] == 1 / 2
    assert joint.loc["write_mediated", "conditional_necessary_estimate"] == 1 / 2
    assert pd.isna(joint.loc["redundant", "conditional_necessary_estimate"])


def test_cluster_bootstrap_mean_uses_cluster_resampling_and_is_deterministic():
    frame = pd.DataFrame(
        {
            "image_group_id": ["a", "a", "b", "b"],
            "value": [0.0, 0.0, 1.0, 1.0],
        }
    )
    first = cluster_bootstrap_mean(frame, "value", replicates=200, seed=7)
    second = cluster_bootstrap_mean(frame, "value", replicates=200, seed=7)
    assert first == second
    assert first["estimate"] == 0.5
    assert first["ci_low"] <= 0.5 <= first["ci_high"]
    assert first["cluster_count"] == 2


def test_context_comparison_identifies_route_necessary_positions_missed_in_full_context():
    route = pd.DataFrame(
        [
            {
                "uid": "u1",
                "dataset": "gqa",
                "image_group_id": "i1",
                "target_layer": 3,
                "taxonomy": "read_mediated",
                "read_w1": -1.0,
                "write_r1": 0.2,
            },
            {
                "uid": "u2",
                "dataset": "gqa",
                "image_group_id": "i2",
                "target_layer": 4,
                "taxonomy": "redundant",
                "read_w1": 0.1,
                "write_r1": 0.1,
            },
        ]
    )
    full = pd.DataFrame(
        [
            {
                "uid": "u1",
                "layer": 3,
                "read_w1": 0.2,
                "write_r1": 0.1,
                "rescue_category": "no_local_rescue",
                "M00": -0.2,
                "M11": -0.3,
            },
            {
                "uid": "u2",
                "layer": 4,
                "read_w1": -0.1,
                "write_r1": -0.1,
                "rescue_category": "read_removal_only",
                "M00": 0.1,
                "M11": -0.1,
            },
        ]
    )

    joined, summary = context_comparison_table(route, full)

    assert joined.loc[joined.uid == "u1", "route_necessary_full_context_missed"].item()
    assert summary["route_necessary_count"] == 1
    assert summary["route_necessary_full_context_missed_count"] == 1
    assert summary["full_context_rescue_route_redundant_count"] == 1
    assert summary["discrete_context_agreement_count"] == 0
    assert summary["discrete_context_agreement_fraction"] == 0.0
    assert summary["route_necessity_recall_from_full_context"] == 0.0
    assert summary["full_context_rescue_precision_for_route_necessity"] == 0.0


def test_category_depth_comparison_bootstraps_read_minus_write_by_image_group():
    cells = pd.DataFrame(
        [
            {"image_group_id": "a", "taxonomy": "read_mediated", "target_layer": 1},
            {"image_group_id": "a", "taxonomy": "write_mediated", "target_layer": 5},
            {"image_group_id": "b", "taxonomy": "read_mediated", "target_layer": 1},
            {"image_group_id": "b", "taxonomy": "write_mediated", "target_layer": 5},
        ]
    )

    summary = _category_depth_comparison(cells, replicates=100, seed=3)

    assert summary["read_mediated_mean_layer"] == 1.0
    assert summary["write_mediated_mean_layer"] == 5.0
    assert summary["read_minus_write_mean_layer"] == -4.0
    assert summary["read_minus_write_ci_low"] == -4.0
    assert summary["read_minus_write_ci_high"] == -4.0
