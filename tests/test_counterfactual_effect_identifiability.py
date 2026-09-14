from __future__ import annotations

import numpy as np
import pytest
import torch

from dense_failure_stage2.counterfactual_identifiability import (
    PairedTokenComparator,
    classify_case,
    construct_summary_feature,
    feature_width,
    matched_random_pair_indices,
    matched_random_pair_assignment,
    pool_branch,
    validate_prediction_census,
)


def _summaries() -> torch.Tensor:
    values = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(2, 4, 6)
    return values


def test_primary_summary_features_use_fixed_read_write_off_branches():
    values = _summaries()
    read_delta = construct_summary_feature(values, target="read", condition="delta")
    write_delta = construct_summary_feature(values, target="write", condition="delta")
    assert torch.equal(read_delta, values[:, 1] - values[:, 2])
    assert torch.equal(write_delta, values[:, 1] - values[:, 3])
    pair = construct_summary_feature(values, target="read", condition="pair")
    swapped = construct_summary_feature(values, target="read", condition="pair", swapped=True)
    assert torch.equal(pair, torch.cat((values[:, 1], values[:, 2]), dim=-1))
    assert torch.equal(swapped, torch.cat((values[:, 2], values[:, 1]), dim=-1))


def test_pair_plus_delta_and_text_visual_ablation_are_algebraically_exact():
    values = _summaries()
    full, off = values[:, 1], values[:, 2]
    combined = construct_summary_feature(values, target="read", condition="pair_plus_delta")
    assert torch.equal(combined, torch.cat((full, off, full - off), dim=-1))
    assert torch.equal(
        construct_summary_feature(values, target="read", condition="text_delta"),
        (full - off)[:, :3],
    )
    assert torch.equal(
        construct_summary_feature(values, target="read", condition="visual_delta"),
        (full - off)[:, 3:],
    )
    assert feature_width("pair_plus_delta", 3) == 18


def test_random_pair_control_matches_cells_and_never_uses_same_uid():
    rows = []
    for dataset in ("gqa", "chartqa"):
        for layer in (18, 19):
            for wrong in (False, True):
                for uid_index in range(3):
                    rows.append(
                        {
                            "state_id": f"{dataset}-{layer}-{wrong}-{uid_index}",
                            "uid": f"{dataset}-u{uid_index}",
                            "dataset": dataset,
                            "layer": layer,
                            "dense_wrong": wrong,
                        }
                    )
    pairs = matched_random_pair_indices(rows, seed=7)
    assert len(pairs) == len(rows)
    for index, paired in enumerate(pairs):
        assert rows[index]["uid"] != rows[int(paired)]["uid"]
        assert rows[index]["dataset"] == rows[int(paired)]["dataset"]
        assert rows[index]["layer"] == rows[int(paired)]["layer"]
        assert rows[index]["dense_wrong"] == rows[int(paired)]["dense_wrong"]


def test_random_pair_control_relaxes_outcome_before_layer():
    rows = [
        {"state_id": "a", "uid": "u1", "dataset": "gqa", "source_regime": "x", "layer": 1, "dense_wrong": True},
        {"state_id": "b", "uid": "u2", "dataset": "gqa", "source_regime": "x", "layer": 1, "dense_wrong": False},
    ]
    pairs, tiers = matched_random_pair_assignment(rows, seed=1)
    assert pairs.tolist() == [1, 0]
    assert tiers == ["layer_dataset", "layer_dataset"]


def test_random_pair_control_rejects_single_uid_population():
    with pytest.raises(ValueError, match="another UID"):
        matched_random_pair_indices(
            [{"state_id": "s", "uid": "u", "dataset": "gqa", "layer": 1, "dense_wrong": True}],
            seed=1,
        )


def test_pool_branch_uses_last_text_and_mean_visual():
    text = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    visual = torch.tensor([[[2.0, 4.0], [4.0, 8.0]]])
    pooled = pool_branch(text, visual).float()
    assert torch.equal(pooled, torch.tensor([3.0, 4.0, 3.0, 6.0]))


def test_token_comparator_is_pair_order_sensitive_and_shape_safe():
    torch.manual_seed(3)
    model = PairedTokenComparator(
        hidden_size=4,
        projection_size=4,
        attention_heads=1,
        readout_hidden_size=5,
        dropout=0.0,
    ).eval()
    full_text = torch.randn(2, 1, 4)
    off_text = torch.randn(2, 1, 4)
    full_visual = torch.randn(2, 3, 4)
    off_visual = torch.randn(2, 3, 4)
    text_mask = torch.ones(2, 1, dtype=torch.bool)
    visual_mask = torch.ones(2, 3, dtype=torch.bool)
    output = model(
        full_text,
        full_visual,
        off_text,
        off_visual,
        text_mask=text_mask,
        visual_mask=visual_mask,
    )
    swapped = model(
        off_text,
        off_visual,
        full_text,
        full_visual,
        text_mask=text_mask,
        visual_mask=visual_mask,
    )
    assert output.shape == (2,)
    assert not torch.equal(output, swapped)


def test_case_rules_apply_frozen_priority():
    thresholds = {
        "weak_absolute_spearman_below": 0.1,
        "strong_spearman_at_least": 0.2,
        "material_gain_over_pre": 0.1,
        "unique_pair_gain_over_best_single": 0.05,
        "token_gain_over_best_pooled": 0.1,
        "random_pair_collapse_gap": 0.05,
    }
    assert classify_case(
        pre=0.03, full_post=0.06, off_post=0.07, pair=0.28, delta=0.31,
        pair_plus_delta=0.3, token=0.32, random_pair_best=0.04, thresholds=thresholds,
    )[0] == "A"
    assert classify_case(
        pre=0.03, full_post=0.31, off_post=0.3, pair=0.33, delta=0.32,
        pair_plus_delta=0.34, token=0.35, random_pair_best=0.02, thresholds=thresholds,
    )[0] == "B"
    assert classify_case(
        pre=0.03, full_post=0.04, off_post=0.05, pair=0.07, delta=0.08,
        pair_plus_delta=0.09, token=0.25, random_pair_best=0.01, thresholds=thresholds,
    )[0] == "C"
    assert classify_case(
        pre=0.03, full_post=0.04, off_post=0.05, pair=0.07, delta=0.08,
        pair_plus_delta=0.09, token=0.1, random_pair_best=0.01, thresholds=thresholds,
    )[0] == "D"


def test_prediction_census_rejects_missing_and_duplicates():
    validate_prediction_census(["a", "b"], [{"state_id": "a"}, {"state_id": "b"}])
    with pytest.raises(ValueError):
        validate_prediction_census(["a", "b"], [{"state_id": "a"}])
    with pytest.raises(ValueError):
        validate_prediction_census(["a", "b"], [{"state_id": "a"}, {"state_id": "a"}])
