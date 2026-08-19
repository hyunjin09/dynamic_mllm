"""Deterministic summary contracts for external binary-router evaluation."""

from experiments.merge_binary_polar_external_eval import summarize_rows


def _row(uid, cluster, baseline, predicted, baseline_score, predicted_score, on):
    return {
        "uid": uid,
        "cluster_key": cluster,
        "baseline_correct": baseline,
        "baseline_score": baseline_score,
        "question": {
            "correct": predicted,
            "score": predicted_score,
            "num_visual_on_layers": on,
            "transition_count": 2,
            "mask_key": "1" * on + "0" * (28 - on),
        },
    }


def test_summary_reports_paired_behavior_and_compute():
    rows = [
        _row("a", "i1", False, True, 0.0, 1.0, 10),
        _row("b", "i2", True, False, 1.0, 0.0, 20),
        _row("c", "i2", True, True, 0.5, 0.75, 20),
    ]
    result = summarize_rows(rows, "question", bootstrap_draws=100, seed=7)
    assert result["records"] == 3
    assert result["full_wrong_to_predicted_correct"] == 1
    assert result["full_correct_to_predicted_wrong"] == 1
    assert result["unchanged_correct"] == 1
    assert result["unchanged_wrong"] == 0
    assert result["mean_visual_on_layers"] == 50 / 3
    assert result["unique_predicted_masks"] == 2
    assert result["cluster_count"] == 2


def test_clustered_bootstrap_is_deterministic():
    rows = [
        _row("a", "i1", False, True, 0.0, 1.0, 10),
        _row("b", "i2", True, True, 1.0, 1.0, 28),
    ]
    left = summarize_rows(rows, "question", bootstrap_draws=100, seed=19)
    right = summarize_rows(rows, "question", bootstrap_draws=100, seed=19)
    assert left["correctness_delta_clustered_95ci"] == right["correctness_delta_clustered_95ci"]
    assert left["score_delta_clustered_95ci"] == right["score_delta_clustered_95ci"]
