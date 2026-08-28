import math

from tools.research_analysis.four_action.aggregate import (
    classify_rescue,
    flatten_samples,
    hamming_stratum,
    route_overlap_table,
    route_metadata,
    spearman,
)


def test_rescue_taxonomy_is_mutually_exclusive():
    assert classify_rescue(False, True, False) == "write_removal_only"
    assert classify_rescue(False, False, True) == "read_removal_only"
    assert classify_rescue(False, True, True) == "either_removal_sufficient"
    assert classify_rescue(True, False, False) == "joint_removal_only"
    assert classify_rescue(False, False, False) == "no_local_rescue"


def test_hamming_strata_follow_approved_bins():
    assert hamming_stratum(1) == "1"
    assert hamming_stratum(2) == "2"
    assert hamming_stratum(4) == "3-4"
    assert hamming_stratum(8) == "5-8"
    assert hamming_stratum(9) == ">8"
    assert hamming_stratum(None) == "not_available"


def test_route_metadata_uses_nearest_route_and_all_correcting_routes():
    routes = {
        "nearest_correcting_route_distance": 1,
        "nearest_correcting_routes": [
            {"mask": [1, 0, 1], "route_id": "nearest-a"},
            {"mask": [0, 1, 1], "route_id": "nearest-b"},
        ],
        "correcting_routes": [
            {"mask": [1, 0, 1], "visual_on_count": 2},
            {"mask": [0, 0, 1], "visual_on_count": 1},
        ],
        "minimum_correcting_visual_on_count": 1,
    }
    metadata = route_metadata(routes, 3)
    assert metadata["nearest_distance"] == 1
    assert metadata["nearest_route_id"] == "nearest-a"
    assert metadata["nearest_mask"] == [1, 0, 1]
    assert metadata["correcting_route_count"] == 2
    assert metadata["off_frequency"] == [0.5, 1.0, 0.0]
    assert metadata["minimum_visual_on_count"] == 1
    assert metadata["minimum_on_route_count"] == 1
    assert metadata["minimum_on_off_frequency"] == [1.0, 1.0, 0.0]


def test_spearman_handles_ties_and_constant_inputs():
    assert math.isclose(spearman([1, 2, 3], [10, 20, 30]), 1.0)
    assert math.isclose(spearman([1, 2, 3], [30, 20, 10]), -1.0)
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_flatten_samples_preserves_factorial_mapping_and_route_metadata():
    def state(margin, correct):
        return {
            "S_correct": margin,
            "S_full_wrong": 0.0,
            "margin": margin,
            "generated_answer": "answer",
            "correctness_score": float(correct),
            "correct": correct,
        }

    sample = {
        "uid": "gqa:one",
        "dataset": "gqa",
        "cohort": "primary_a_plus",
        "sample_id": "one",
        "image_id": "image",
        "image_group_id": "image",
        "visual_token_count": 4,
        "binary_routes": {
            "nearest_correcting_route_distance": 1,
            "nearest_correcting_routes": [{"route_id": "r", "mask": [0, 1]}],
            "correcting_routes": [
                {"route_id": "r", "mask": [0, 1], "visual_on_count": 1}
            ],
            "minimum_correcting_visual_on_count": 1,
        },
        "layers": [
            {
                "layer": 0,
                "states": {
                    "IGNORE": state(1.0, False),
                    "READ_ONLY": state(2.0, True),
                    "WRITE_ONLY": state(3.0, False),
                    "FULL": state(4.0, False),
                },
                "effects": {
                    "read_w0": 1.0,
                    "read_w1": 1.0,
                    "write_r0": 2.0,
                    "write_r1": 2.0,
                    "interaction": -1.0,
                },
            }
        ],
    }
    rows = flatten_samples([sample], layer_count=2)
    assert len(rows) == 4
    assert {row["action"] for row in rows} == {"IGNORE", "READ_ONLY", "WRITE_ONLY", "FULL"}
    assert rows[0]["M00"] == 1.0
    assert rows[0]["M10"] == 2.0
    assert rows[0]["M01"] == 3.0
    assert rows[0]["M11"] == 4.0
    assert rows[0]["rescue_category"] == "write_removal_only"
    assert rows[0]["nearest_route_layer_off"] is True
    assert rows[0]["correcting_route_off_frequency"] == 1.0
    assert rows[0]["minimum_on_route_off_frequency"] == 1.0


def test_route_overlap_compares_each_requested_metric_off_vs_on():
    rows = []
    for layer, is_off, read, write, ignore_gain, frequency in (
        (0, True, -3.0, -1.0, 2.0, 1.0),
        (1, False, -0.5, 0.0, 0.25, 0.0),
    ):
        rows.append(
            {
                "uid": "gqa:one",
                "dataset": "gqa",
                "layer": layer,
                "read_w1": read,
                "write_r1": write,
                "M00": ignore_gain,
                "M11": 0.0,
                "nearest_route_layer_off": is_off,
                "correcting_route_off_frequency": frequency,
                "nearest_correcting_route_distance": 1,
                "minimum_correcting_visual_on_count": 1,
                "minimum_on_route_count": 1,
            }
        )
    result = route_overlap_table(rows)
    sample = result["per_sample"][0]
    assert sample["nearest_off_minus_on_read_w1"] == -2.5
    assert sample["nearest_off_minus_on_write_r1"] == -1.0
    assert sample["nearest_off_minus_on_ignore_gain_m00_minus_m11"] == 1.75
    assert sample["nearest_off_minus_on_strongest_local_harmfulness"] == 2.5
