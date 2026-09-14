from __future__ import annotations

import numpy as np

from dense_failure_stage1.historical_shortcut_audit import (
    average_precision,
    binary_auroc,
    deterministic_group_folds,
    exact_stratum_match,
    fit_centroid_probe,
    spearman_correlation,
)


def test_binary_metrics_are_exact_for_perfect_ranking() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    assert binary_auroc(labels, scores) == 1.0
    assert average_precision(labels, scores) == 1.0


def test_average_precision_groups_tied_scores() -> None:
    assert average_precision([1, 0], [0.5, 0.5]) == 0.5


def test_centroid_probe_uses_training_statistics_only() -> None:
    x = np.asarray([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    y = np.asarray([0, 0, 1, 1])
    probe = fit_centroid_probe(x, y)
    scores = probe.score(x)
    assert binary_auroc(y, scores) == 1.0
    assert np.isfinite(scores).all()


def test_group_folds_never_split_a_group() -> None:
    groups = ["a", "a", "b", "c", "c", "d"]
    folds = deterministic_group_folds(groups, seed=17, n_folds=3)
    for group in set(groups):
        assigned = {folds[index] for index, value in enumerate(groups) if value == group}
        assert len(assigned) == 1


def test_exact_matching_balances_labels_inside_strata() -> None:
    rows = [
        {"uid": "a", "label": 0, "stratum": "x"},
        {"uid": "b", "label": 0, "stratum": "x"},
        {"uid": "c", "label": 1, "stratum": "x"},
        {"uid": "d", "label": 0, "stratum": "y"},
        {"uid": "e", "label": 1, "stratum": "y"},
        {"uid": "f", "label": 1, "stratum": "y"},
    ]
    matched = exact_stratum_match(rows, label_key="label", stratum_key="stratum", seed=5)
    assert len(matched) == 4
    for stratum in {row["stratum"] for row in matched}:
        labels = [row["label"] for row in matched if row["stratum"] == stratum]
        assert labels.count(0) == labels.count(1)


def test_spearman_handles_ties_and_monotone_values() -> None:
    assert spearman_correlation([1, 2, 2, 4], [10, 20, 20, 40]) == 1.0
    assert np.isnan(spearman_correlation([1, 1], [2, 3]))
