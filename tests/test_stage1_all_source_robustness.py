from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from dense_failure_stage1.all_source_robustness import (
    bootstrap_auc_interval,
    choose_robustness_decision,
    four_cell_epoch_indices,
    source_stratified_validation_uids,
    validate_lodo_training_rows,
)
from experiments.run_stage1_all_source_robustness import _validation_bce


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    counts = {
        ("historical", False): 19,
        ("historical", True): 17,
        ("canonical", False): 13,
        ("canonical", True): 5,
    }
    datasets = ("gqa", "chartqa", "textvqa")
    for (source, wrong), count in counts.items():
        for index in range(count):
            dataset = datasets[index % len(datasets)]
            uid = f"{source}:{int(wrong)}:{index}"
            rows.append(
                {
                    "uid": uid,
                    "source_regime": source,
                    "dataset": dataset,
                    "current_dense_wrong": wrong,
                    "image_group_id": f"sha256:{uid}",
                }
            )
    return rows


def test_four_cell_sampler_is_exact_deterministic_and_uses_no_other_rows() -> None:
    rows = _rows()
    first = four_cell_epoch_indices(rows, records=80, seed=11, epoch=2)
    repeated = four_cell_epoch_indices(rows, records=80, seed=11, epoch=2)
    changed = four_cell_epoch_indices(rows, records=80, seed=11, epoch=3)
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, changed)
    assert len(first) == 80
    cells = Counter(
        (rows[index]["source_regime"], rows[index]["current_dense_wrong"])
        for index in first
    )
    assert cells == {
        ("historical", False): 20,
        ("historical", True): 20,
        ("canonical", False): 20,
        ("canonical", True): 20,
    }
    assert all(0 <= int(index) < len(rows) for index in first)


def test_source_stratified_validation_is_group_disjoint_and_supported() -> None:
    rows = _rows()
    selected = source_stratified_validation_uids(
        rows, fraction=0.2, seed=23, run_id="main_fold_0"
    )
    assert selected
    selected_groups = {
        row["image_group_id"] for row in rows if row["uid"] in selected
    }
    training_groups = {
        row["image_group_id"] for row in rows if row["uid"] not in selected
    }
    assert not selected_groups & training_groups
    selected_strata = Counter(
        (row["source_regime"], row["dataset"], row["current_dense_wrong"])
        for row in rows
        if row["uid"] in selected
    )
    assert all(value >= 1 for value in selected_strata.values())


def test_source_stratified_validation_keeps_mixed_label_image_group_whole() -> None:
    rows = _rows()
    rows[0]["image_group_id"] = "sha256:shared"
    rows[3]["image_group_id"] = "sha256:shared"
    rows[3]["current_dense_wrong"] = True
    selected = source_stratified_validation_uids(
        rows, fraction=0.2, seed=31, run_id="mixed_group"
    )
    assert (str(rows[0]["uid"]) in selected) == (str(rows[3]["uid"]) in selected)


def test_lodo_validation_rejects_target_dataset_in_training() -> None:
    rows = _rows()
    non_target = [row for row in rows if row["dataset"] != "chartqa"]
    validate_lodo_training_rows(non_target, target_dataset="chartqa")
    with pytest.raises(ValueError, match="target dataset"):
        validate_lodo_training_rows(rows, target_dataset="chartqa")


def test_bootstrap_interval_is_deterministic_and_contains_perfect_auc() -> None:
    labels = np.asarray([0] * 8 + [1] * 8)
    scores = np.asarray(list(range(16)), dtype=np.float64)
    first = bootstrap_auc_interval(labels, scores, draws=300, seed=7)
    second = bootstrap_auc_interval(labels, scores, draws=300, seed=7)
    assert first == second
    assert first["auroc"] == 1.0
    assert first["ci_low"] == 1.0
    assert first["ci_high"] == 1.0


def test_validation_bce_matches_historical_logit_formulation() -> None:
    logits = np.asarray([[-20.0, -1.0], [1.0, 20.0]], dtype=np.float64)
    labels = np.asarray([0, 1], dtype=np.int64)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(logits.reshape(-1)),
        torch.tensor(np.repeat(labels, 2), dtype=torch.float64),
    )
    assert _validation_bce(logits, labels) == pytest.approx(float(expected), abs=1e-12)


@pytest.mark.parametrize(
    ("historical", "canonical", "lodo", "expected"),
    [
        (0.82, 0.78, [0.65, 0.66, 0.67], "A"),
        (0.82, 0.78, [0.65, 0.59, 0.67], "B"),
        (0.82, 0.61, [0.70, 0.70, 0.70], "C"),
    ],
)
def test_robustness_decision_is_prospective(
    historical: float, canonical: float, lodo: list[float], expected: str
) -> None:
    result = choose_robustness_decision(
        historical_auroc=historical,
        canonical_auroc=canonical,
        lodo_worst_source_aurocs=lodo,
        historical_lodo_mean=0.70,
        phase49_historical_mean=0.5966,
    )
    assert result["decision"].startswith(expected)
