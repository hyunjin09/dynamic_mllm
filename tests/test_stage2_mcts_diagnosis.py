from dense_failure_stage2.mcts_diagnosis import (
    action_metrics,
    entering_state_key,
    first_deviation_class,
    forcing_boundary,
    intervention_index,
    intervention_index_bin,
    matched_mode_uid_pairs,
    observed_successful_action_sets,
    paired_uid_bootstrap,
    route_complexity_bin,
)


def test_entering_state_key_depends_only_on_prefix():
    one = ["FULL", "READ_ONLY", "FULL"]
    two = ["FULL", "WRITE_ONLY", "IGNORE"]
    assert entering_state_key("u", one, 1) == entering_state_key("u", two, 1)
    assert entering_state_key("u", one, 2) != entering_state_key("u", two, 2)


def test_observed_successful_action_sets_find_exact_prefix_ambiguity():
    routes = [
        {"uid": "u", "activation_layer": 0, "actions": ["FULL", "READ_ONLY"]},
        {"uid": "u", "activation_layer": 0, "actions": ["FULL", "IGNORE"]},
    ]
    observed = observed_successful_action_sets(routes)
    assert observed[entering_state_key("u", routes[0]["actions"], 0)] == ("FULL",)
    assert observed[entering_state_key("u", routes[0]["actions"], 1)] == (
        "READ_ONLY",
        "IGNORE",
    )


def test_forcing_boundary_and_intervention_indices():
    actions = ["FULL", "READ_ONLY", "FULL", "IGNORE"]
    assert forcing_boundary(actions, 0, 0) == -1
    assert forcing_boundary(actions, 0, 1) == 1
    assert forcing_boundary(actions, 0, 2) == 3
    assert intervention_index(actions, 1, 0) == 1
    assert intervention_index(actions, 2, 0) is None
    assert intervention_index(actions, 3, 0) == 2
    assert intervention_index_bin(1) == "first"
    assert intervention_index_bin(2) == "second"
    assert intervention_index_bin(4) == "third+"
    assert route_complexity_bin(4) == "4+"


def test_first_deviation_categories():
    actions = ["FULL", "FULL", "READ_ONLY", "FULL", "IGNORE"]
    assert first_deviation_class(actions, actions, 0) == ("NO_DEVIATION", None)
    assert first_deviation_class(actions, ["IGNORE", *actions[1:]], 0) == (
        "BEFORE_FIRST_INTERVENTION",
        0,
    )
    assert first_deviation_class(actions, [*actions[:2], "FULL", *actions[3:]], 0) == (
        "AT_FIRST_INTERVENTION",
        2,
    )
    assert first_deviation_class(actions, [*actions[:3], "IGNORE", actions[4]], 0) == (
        "BETWEEN_INTERVENTIONS",
        3,
    )
    assert first_deviation_class(actions, [*actions[:4], "FULL"], 0) == (
        "AT_SECOND_OR_LATER_INTERVENTION",
        4,
    )


def test_action_metrics_and_paired_bootstrap_are_deterministic():
    rows = [
        {
            "target_action": "FULL",
            "predicted_action": "FULL",
            "target_probability": 0.8,
            "target_vs_full_margin": 0.0,
            "full_probability": 0.8,
            "best_non_full_probability": 0.1,
        },
        {
            "target_action": "IGNORE",
            "predicted_action": "FULL",
            "target_probability": 0.2,
            "target_vs_full_margin": -0.5,
            "full_probability": 0.7,
            "best_non_full_probability": 0.2,
        },
    ]
    metrics = action_metrics(rows)
    assert metrics["action_accuracy"] == 0.5
    assert metrics["full_recall"] == 1.0
    assert metrics["non_full_recall"] == 0.0
    pairs = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
    first = paired_uid_bootstrap(pairs, seed=7, replicates=100)
    second = paired_uid_bootstrap(pairs, seed=7, replicates=100)
    assert first == second
    assert first["mean_delta_B_minus_A"] == 1.0


def test_matched_mode_uid_pairs_excludes_ineligible_baseline_routes():
    rows = [
        {"mode": "C0", "route_id": "r1", "uid": "u1", "correct": False},
        {"mode": "C0", "route_id": "r2", "uid": "u1", "correct": True},
        {"mode": "C0", "route_id": "r3", "uid": "u2", "correct": False},
        {"mode": "C2", "route_id": "r1", "uid": "u1", "correct": True},
        {"mode": "C2", "route_id": "r3", "uid": "u2", "correct": False},
    ]
    assert matched_mode_uid_pairs(rows, "C0", "C2") == {
        "u1": (0.0, 1.0),
        "u2": (0.0, 0.0),
    }
