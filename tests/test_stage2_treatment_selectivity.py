from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pytest
import torch

from dense_failure_stage2.treatment_selectivity import (
    assign_group_folds,
    binary_metrics,
    classify_observed_actions,
    exact_router_representations,
    fit_binary_probe,
    matched_evaluation_weights,
    predict_binary_probe,
    selective_operating_points,
    standardize_fold,
    training_weights,
    validate_extracted_state_rows,
)
from dense_failure_stage2.v1_router import SharedReadWriteRouter


def test_clean_label_contract_never_calls_search_miss_keep() -> None:
    assert classify_observed_actions(["FULL"]) == "KEEP_REQUIRED"
    assert classify_observed_actions(["READ_ONLY"]) == "INTERVENE_REQUIRED"
    assert classify_observed_actions(["WRITE_ONLY", "IGNORE"]) == "INTERVENE_REQUIRED"
    assert classify_observed_actions(["FULL", "IGNORE"]) == "MIXED"
    assert classify_observed_actions([]) == "UNRESOLVED"
    with pytest.raises(ValueError):
        classify_observed_actions(["NOT_AN_ACTION"])


def test_extracted_branch_path_exactly_matches_frozen_router_forward() -> None:
    torch.manual_seed(7)
    router = SharedReadWriteRouter(hidden_size=12, router_size=8, num_heads=2, dropout=0.0).eval()
    text = torch.randn(3, 5, 12)
    visual = torch.randn(3, 7, 12)
    text_mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True], [True, False, False, False, False]]
    )
    visual_mask = torch.tensor(
        [[True, True, True, True, False, False, False], [True] * 7, [True, True, False, False, False, False, False]]
    )
    with torch.inference_mode():
        expected = router(text, visual, text_mask=text_mask, visual_mask=visual_mask)
        read, write, logits = exact_router_representations(
            router, text, visual, text_mask=text_mask, visual_mask=visual_mask
        )
    assert read.shape == (3, 8)
    assert write.shape == (3, 8)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


def _fold_rows() -> list[dict[str, object]]:
    rows = []
    for group_index in range(30):
        dataset = ("gqa", "chartqa", "textvqa")[group_index % 3]
        source = ("historical", "canonical")[group_index % 2]
        label = group_index % 2
        for state_index in range(1 + group_index % 3):
            rows.append(
                {
                    "state_id": f"s-{group_index}-{state_index}",
                    "uid": f"u-{group_index}",
                    "image_group_id": f"g-{group_index // 2}",
                    "dataset": dataset,
                    "source_regime": source,
                    "binary_target": label,
                }
            )
    return rows


def test_group_folds_are_repeatable_and_group_disjoint() -> None:
    rows = _fold_rows()
    first = assign_group_folds(rows, folds=5, seed=19)
    second = assign_group_folds(rows, folds=5, seed=19)
    assert first == second
    state_fold = {row["state_id"]: row["fold"] for row in first}
    group_folds: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        group_folds[str(row["image_group_id"])].add(state_fold[str(row["state_id"])])
    assert all(len(values) == 1 for values in group_folds.values())
    assert set(state_fold.values()) == set(range(5))


def test_group_folds_handle_route_rich_group_size_skew() -> None:
    rows = []
    for group_index, size in enumerate([80, 60, 40, 30, 20, *([2] * 30)]):
        for state_index in range(size):
            rows.append(
                {
                    "state_id": f"skew-{group_index}-{state_index}",
                    "uid": f"u-{group_index}",
                    "image_group_id": f"g-{group_index}",
                    "dataset": ("gqa", "chartqa", "textvqa")[group_index % 3],
                    "source_regime": ("historical", "canonical")[group_index % 2],
                    "binary_target": (group_index + state_index) % 2,
                }
            )
    assigned = assign_group_folds(rows, folds=5, seed=20260905)
    counts = Counter(int(row["fold"]) for row in assigned)
    assert set(counts) == set(range(5))
    assert min(counts.values()) > 0


def test_group_folds_do_not_pack_homogeneous_groups_into_only_near_target_folds() -> None:
    rows = [
        {
            "state_id": f"homogeneous-{index}",
            "uid": f"u-{index}",
            "image_group_id": f"g-{index}",
            "dataset": "gqa",
            "source_regime": "historical",
            "binary_target": 0,
        }
        for index in range(100)
    ]
    assigned = assign_group_folds(rows, folds=5, seed=20260905)
    assert Counter(int(row["fold"]) for row in assigned) == Counter({fold: 20 for fold in range(5)})


def test_training_weights_equalize_uids_then_classes() -> None:
    rows = [
        {"uid": "a", "binary_target": 0},
        {"uid": "a", "binary_target": 0},
        {"uid": "b", "binary_target": 0},
        {"uid": "c", "binary_target": 1},
        {"uid": "c", "binary_target": 1},
        {"uid": "c", "binary_target": 1},
    ]
    weights = training_weights(rows)
    totals_by_uid: dict[str, float] = defaultdict(float)
    totals_by_class: dict[int, float] = defaultdict(float)
    for row, weight in zip(rows, weights, strict=True):
        totals_by_uid[str(row["uid"])] += float(weight)
        totals_by_class[int(row["binary_target"])] += float(weight)
    assert totals_by_uid["a"] == pytest.approx(totals_by_uid["b"])
    assert totals_by_class[0] == pytest.approx(totals_by_class[1])


def test_metrics_and_matched_weights_are_fail_closed() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    perfect = binary_metrics(labels, np.asarray([0.1, 0.2, 0.8, 0.9]))
    assert perfect["auroc"] == pytest.approx(1.0)
    assert perfect["auprc"] == pytest.approx(1.0)
    rows = [
        {"binary_target": 0, "match_cell": "a"},
        {"binary_target": 0, "match_cell": "a"},
        {"binary_target": 1, "match_cell": "a"},
        {"binary_target": 0, "match_cell": "b"},
    ]
    weights, support = matched_evaluation_weights(rows)
    assert weights.tolist() == pytest.approx([0.5, 0.5, 1.0, 0.0])
    assert support == {"supported_cells": 1, "supported_states": 3, "excluded_states": 1}
    with pytest.raises(ValueError):
        binary_metrics(np.asarray([0, 0]), np.asarray([0.1, 0.2]))


def test_extraction_aggregation_rejects_missing_and_duplicate_states() -> None:
    expected = ["a", "b"]
    valid = [{"state_id": "a"}, {"state_id": "b"}]
    assert validate_extracted_state_rows(expected, valid) == {"states": 2, "duplicates": 0, "missing": 0}
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_extracted_state_rows(expected, [{"state_id": "a"}, {"state_id": "a"}, {"state_id": "b"}])
    with pytest.raises(RuntimeError, match="missing"):
        validate_extracted_state_rows(expected, [{"state_id": "a"}])


def test_fold_local_standardization_and_linear_probe_are_deterministic() -> None:
    train = np.asarray([[-2.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float32)
    heldout = np.asarray([[-3.0, 1.0], [3.0, 1.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    normalized_train, normalized_heldout, normalization = standardize_fold(
        train, heldout, np.ones(4)
    )
    assert np.isfinite(normalized_train).all()
    assert normalization["std"][1] == pytest.approx(1.0)
    first = fit_binary_probe(
        normalized_train,
        labels,
        np.ones(4),
        kind="linear",
        seed=11,
        epochs=200,
        learning_rate=0.1,
        weight_decay=0.0,
        device="cpu",
    )
    second = fit_binary_probe(
        normalized_train,
        labels,
        np.ones(4),
        kind="linear",
        seed=11,
        epochs=200,
        learning_rate=0.1,
        weight_decay=0.0,
        device="cpu",
    )
    assert first["state_dict"] == second["state_dict"]
    scores = predict_binary_probe(first, normalized_heldout, device="cpu")
    assert scores[0] < 0.5 < scores[1]


def test_selective_operating_points_use_complete_score_ties() -> None:
    labels = np.asarray([1, 0, 1, 0, 0], dtype=np.int64)
    scores = np.asarray([0.9, 0.8, 0.8, 0.2, 0.1], dtype=np.float64)
    rows = selective_operating_points(
        labels, scores, coverage_fractions=[0.4], precision_targets=[0.6, 0.9]
    )
    coverage = next(row for row in rows if row["operating_point"] == "top_40pct")
    assert coverage["selected"] == 3
    assert coverage["precision"] == pytest.approx(2 / 3)
    at_60 = next(row for row in rows if row["operating_point"] == "recall_at_60pct_precision")
    assert at_60["recall"] == pytest.approx(1.0)
    at_90 = next(row for row in rows if row["operating_point"] == "recall_at_90pct_precision")
    assert at_90["recall"] == pytest.approx(0.5)
    assert len({tuple(row.keys()) for row in rows}) == 1
