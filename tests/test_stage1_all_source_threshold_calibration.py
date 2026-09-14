from __future__ import annotations

import numpy as np
import pytest

from dense_failure_stage1.threshold_calibration import (
    choose_freeze_decision,
    first_trigger_layers,
    group_bootstrap_metrics,
    most_permissive_at_preservation,
    operating_metrics,
    robust_threshold_sweep,
    select_primary_threshold,
    wilson_interval,
)


def _scores(maxima: list[float]) -> np.ndarray:
    values = np.zeros((len(maxima), 28), dtype=np.float64)
    values[:, 0] = maxima
    values[:, 7] = np.asarray(maxima) * 0.9
    return values


def test_strict_threshold_and_first_layer_semantics() -> None:
    scores = np.zeros((3, 28), dtype=np.float64)
    scores[0, 0] = 0.5
    scores[1, 4] = 0.5000001
    scores[2, 3] = 0.4
    scores[2, 9] = 0.7
    assert first_trigger_layers(scores, 0.5).tolist() == [-1, 4, 9]
    metrics = operating_metrics(scores, np.asarray([0, 1, 1]), 0.5)
    assert metrics["correct_false_admissions"] == 0
    assert metrics["wrong_detected"] == 2
    assert metrics["median_first_trigger_layer"] == pytest.approx(6.5)


def test_reference_and_primary_use_worst_source_preservation() -> None:
    historical_scores = _scores([0.91, 0.80, 0.70, 0.60, 0.95, 0.85])
    historical_labels = np.asarray([0, 0, 0, 0, 1, 1])
    canonical_scores = _scores([0.99, 0.40, 0.30, 0.20, 0.98, 0.50])
    canonical_labels = np.asarray([0, 0, 0, 0, 1, 1])
    sweep = robust_threshold_sweep(
        historical_scores,
        historical_labels,
        canonical_scores,
        canonical_labels,
        include_depth=True,
    )
    reference = most_permissive_at_preservation(sweep, 0.75)
    primary = select_primary_threshold(sweep, minimum_preservation=0.75)
    assert reference["threshold"] == pytest.approx(0.80)
    assert primary["threshold"] == pytest.approx(0.80)
    assert reference["historical_c_preservation"] >= 0.75
    assert reference["canonical_c_preservation"] >= 0.75


def test_group_bootstrap_is_deterministic_and_group_clustered() -> None:
    rows = []
    for source in ("historical", "canonical"):
        for group in range(8):
            for label in (0, 1):
                rows.append(
                    {
                        "source_regime": source,
                        "image_group_id": f"{source}:{group}",
                        "label": label,
                        "score_max": 0.2 if label == 0 else 0.8,
                    }
                )
    first = group_bootstrap_metrics(rows, threshold=0.5, draws=100, seed=9)
    second = group_bootstrap_metrics(rows, threshold=0.5, draws=100, seed=9)
    assert first == second
    assert all(row["historical_c_preservation"] == 1.0 for row in first)
    assert all(row["canonical_w_recall"] == 1.0 for row in first)
    assert all(row["trigger_precision"] == 1.0 for row in first)


def test_wilson_interval_handles_sparse_support_without_thresholding() -> None:
    low, high = wilson_interval(1, 19)
    assert 0.0 < low < 1 / 19 < high < 1.0
    empty = wilson_interval(0, 0)
    assert np.isnan(empty[0]) and np.isnan(empty[1])


@pytest.mark.parametrize(
    ("useful", "stable", "catastrophic", "prefix"),
    [
        (True, True, 0, "A"),
        (False, True, 0, "B"),
        (True, False, 0, "C"),
        (True, True, 1, "C"),
    ],
)
def test_freeze_decision_is_fail_closed(
    useful: bool, stable: bool, catastrophic: int, prefix: str
) -> None:
    result = choose_freeze_decision(
        primary_metrics={"worst_source_c_preservation": 0.98},
        useful_w_detection=useful,
        crossfit_stable=stable,
        catastrophic_cells=catastrophic,
    )
    assert result.startswith(prefix)


def test_freeze_decision_rejects_invalid_primary_metrics() -> None:
    with pytest.raises(ValueError, match="preservation"):
        choose_freeze_decision(
            primary_metrics={"worst_source_c_preservation": 0.979},
            useful_w_detection=True,
            crossfit_stable=True,
            catastrophic_cells=0,
        )
