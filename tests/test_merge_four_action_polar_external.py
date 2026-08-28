from __future__ import annotations

from experiments.merge_four_action_polar_external import summarize_rows


def _row(uid, cluster, baseline, predicted, baseline_score, predicted_score, actions):
    return {
        "uid": uid,
        "cluster_key": cluster,
        "baseline_correct": baseline,
        "baseline_score": baseline_score,
        "baseline_generated_ids": [1],
        "predicted": {
            "correct": predicted,
            "score": predicted_score,
            "generated_ids": [2],
            "route_key": "|".join(actions),
            "action_counts": {
                action: actions.count(action)
                for action in ("IGNORE", "READ_ONLY", "WRITE_ONLY", "FULL")
            },
            "non_full_layers": sum(action != "FULL" for action in actions),
            "transition_count": 1,
        },
    }


def test_summary_reports_paired_accuracy_and_action_usage() -> None:
    rows = [
        _row("a", "i1", False, True, 0.0, 1.0, ["FULL"] * 27 + ["IGNORE"]),
        _row("b", "i2", True, False, 1.0, 0.0, ["READ_ONLY"] * 28),
        _row("c", "i2", True, True, 0.5, 0.75, ["WRITE_ONLY"] * 28),
    ]

    summary = summarize_rows(rows, bootstrap_draws=100, seed=7)

    assert summary["records"] == 3
    assert summary["full_wrong_to_predicted_correct"] == 1
    assert summary["full_correct_to_predicted_wrong"] == 1
    assert summary["behavior_changing_executions"] == 3
    assert summary["mean_action_layers"]["READ_ONLY"] == 28 / 3
    assert summary["cluster_count"] == 2


def test_four_action_cluster_bootstrap_is_deterministic() -> None:
    rows = [
        _row("a", "i1", False, True, 0.0, 1.0, ["FULL"] * 28),
        _row("b", "i2", True, True, 1.0, 1.0, ["IGNORE"] * 28),
    ]
    left = summarize_rows(rows, bootstrap_draws=100, seed=11)
    right = summarize_rows(rows, bootstrap_draws=100, seed=11)
    assert left["correctness_delta_clustered_95ci"] == right["correctness_delta_clustered_95ci"]
