from __future__ import annotations

import numpy as np

from dense_failure_stage1.canonical_refit import (
    assign_group_folds,
    balanced_epoch_indices,
    inner_validation_uids,
    paired_bootstrap_auc_difference,
)


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, correct, wrong in (("gqa", 17, 8), ("chartqa", 13, 7), ("textvqa", 11, 6)):
        for label, count in ((0, correct), (1, wrong)):
            for index in range(count):
                uid = f"{dataset}:{label}:{index}"
                rows.append(
                    {
                        "uid": uid,
                        "dataset": dataset,
                        "current_dense_wrong": bool(label),
                        "image_group_id": f"sha256:{uid}",
                    }
                )
    return rows


def test_group_folds_are_deterministic_disjoint_and_stratified() -> None:
    rows = _rows()
    first = assign_group_folds(rows, folds=5, seed=19)
    second = assign_group_folds(list(reversed(rows)), folds=5, seed=19)
    assert first == second
    assert {row["uid"] for row in first} == {row["uid"] for row in rows}
    assert {row["fold"] for row in first} == set(range(5))

    group_folds: dict[str, set[int]] = {}
    for row in first:
        group_folds.setdefault(str(row["image_group_id"]), set()).add(int(row["fold"]))
    assert all(len(value) == 1 for value in group_folds.values())

    for dataset in ("gqa", "chartqa", "textvqa"):
        for label in (False, True):
            counts = [
                sum(
                    row["dataset"] == dataset
                    and row["current_dense_wrong"] is label
                    and row["fold"] == fold
                    for row in first
                )
                for fold in range(5)
            ]
            assert max(counts) - min(counts) <= 1


def test_inner_validation_is_training_only_and_group_disjoint() -> None:
    folded = assign_group_folds(_rows(), folds=5, seed=3)
    outer_train = [row for row in folded if row["fold"] != 2]
    validation = inner_validation_uids(
        outer_train, fraction=0.125, seed=29, outer_fold=2
    )
    assert validation
    assert validation < {str(row["uid"]) for row in outer_train}
    assert not validation & {
        str(row["uid"]) for row in folded if row["fold"] == 2
    }
    validation_groups = {
        str(row["image_group_id"])
        for row in outer_train
        if str(row["uid"]) in validation
    }
    training_groups = {
        str(row["image_group_id"])
        for row in outer_train
        if str(row["uid"]) not in validation
    }
    assert not validation_groups & training_groups


def test_balanced_epoch_indices_are_deterministic_and_balanced() -> None:
    labels = np.asarray([0] * 23 + [1] * 7, dtype=np.int64)
    first = balanced_epoch_indices(labels, seed=41, epoch=0)
    repeated = balanced_epoch_indices(labels, seed=41, epoch=0)
    next_epoch = balanced_epoch_indices(labels, seed=41, epoch=1)
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, next_epoch)
    assert len(first) == 14
    assert int(labels[first].sum()) == 7
    assert len(set(first.tolist())) == len(first)


def test_paired_bootstrap_is_deterministic_and_directional() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    old = np.asarray([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    refit = 1.0 - old
    first = paired_bootstrap_auc_difference(
        labels, old, refit, draws=500, seed=17
    )
    repeated = paired_bootstrap_auc_difference(
        labels, old, refit, draws=500, seed=17
    )
    assert first == repeated
    assert first["observed_difference"] == 1.0
    assert first["ci_low"] > 0.0
    assert first["valid_draws"] == 500
