from experiments.analyze_cross_dataset_visual_access import (
    ALL_OFF,
    ALL_ON,
    classify,
    matched_candidates,
    profile_distance,
    sample_placement,
)


def _route(mask, correct=False):
    return {"visual_on_mask": list(mask), "result_correct": correct}


def test_matched_candidates_follow_simulation_prefix_and_deduplicate():
    a = (1, 0) + (1,) * 26
    b = (0, 1) + (1,) * 26
    record = {
        "candidate_executions": [_route(ALL_ON), _route(ALL_OFF), _route(a), _route(b)],
        "mcts": {"simulations": [{"evaluated_mask": list(a)}, {"evaluated_mask": list(a)},
                                  {"evaluated_mask": list(b)}]},
    }
    selected = matched_candidates(record, 2)
    assert [tuple(row["visual_on_mask"]) for row in selected] == [ALL_ON, ALL_OFF, a]


def test_taxonomy_is_anchor_first_and_positive_route_aware():
    route = [(1,) + (0,) * 27]
    assert classify(True, True, route) == "V0"
    assert classify(True, False, route) == "V+"
    assert classify(False, True, route) == "A0"
    assert classify(False, False, route) == "A+"
    assert classify(False, False, []) == "D"


def test_sample_placement_averages_all_minimum_routes():
    early = (1, 1) + (0,) * 26
    late = (0,) * 26 + (1, 1)
    row = {"dataset": "gqa", "uid": "x", "image_group_id": "i", "actual_visual_tokens": 10,
           "actual_text_tokens": 5}
    summary = sample_placement(row, [early, late, ALL_ON], 0)
    assert summary["selected_route_count"] == 2
    assert summary["min_positive_on"] == 2
    assert summary["normalized_centroid"] == 0.5
    assert summary["early_fraction"] == 0.5
    assert summary["late_fraction"] == 0.5


def test_profile_distance_identity_and_separation():
    identity = profile_distance([1, 0], [1, 0])
    separated = profile_distance([1, 0], [0, 1])
    assert identity["l1_distance"] == 0
    assert identity["cosine_similarity"] == 1
    assert separated["l1_distance"] == 2
    assert separated["cosine_similarity"] == 0
