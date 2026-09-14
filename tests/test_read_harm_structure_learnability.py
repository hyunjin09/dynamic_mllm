from __future__ import annotations

import numpy as np
import pytest
import torch
import experiments.analyze_read_harm_structure_learnability as read_analysis

from dense_failure_stage2.read_harm_learnability import (
    FEATURE_GROUP_ORDER,
    adjacent_transition_counts,
    classify_read_behavior,
    contiguous_run_lengths,
    exact_nuisance_matches,
    fit_predict_fold,
    select_dense_winner,
    summarize_read_attention,
    summarize_read_update,
    validate_attention_reconstruction,
    validate_feature_census,
)
from experiments.analyze_read_harm_structure_learnability import (
    _flat_sequence_metrics,
    _holdout_roles,
    _matched_role_indices,
    _sequence_metrics,
)


def test_read_behavior_uses_full_and_write_only_pair_only():
    assert classify_read_behavior(False, True) == "read_harmful_flip"
    assert classify_read_behavior(True, False) == "read_beneficial_flip"
    assert classify_read_behavior(False, False) == "stable_wrong"
    assert classify_read_behavior(True, True) == "stable_correct"


def test_adjacent_transitions_treat_zero_as_a_separate_state():
    sequence = np.asarray([1.0, 2.0, -1.0, 0.0, -3.0])
    counts = adjacent_transition_counts(sequence)
    assert counts == {
        ("harmful", "harmful"): 1,
        ("harmful", "beneficial"): 1,
        ("beneficial", "zero"): 1,
        ("zero", "beneficial"): 1,
    }


def test_contiguous_runs_are_maximal_and_sign_specific():
    values = np.asarray([1.0, 2.0, -1.0, 3.0, 0.0, 4.0, 5.0, 6.0])
    assert contiguous_run_lengths(values, sign="harmful") == [2, 1, 3]
    assert contiguous_run_lengths(values, sign="beneficial") == [1]


def test_vectorized_flat_sequence_metrics_match_reference_sequences():
    sequences = [np.asarray([1.0, 1.0, -1.0]), np.asarray([-1.0, 0.0, -1.0, 1.0])]
    signs = np.sign(np.concatenate(sequences)).astype(np.int8)
    starts = np.asarray([True, False, False, True, False, False, False])
    assert _flat_sequence_metrics(signs, starts) == pytest.approx(_sequence_metrics(sequences))


def test_update_summary_is_scale_and_concentration_explicit():
    pre = np.asarray([3.0, 4.0])
    off = np.asarray([4.0, 3.0])
    full = np.asarray([7.0, 7.0])
    token_deltas = np.asarray([[3.0, 4.0], [0.0, 0.0]])
    row = summarize_read_update(pre, off, full, token_deltas, top_k=5)
    assert row["f1_read_update_norm"] == pytest.approx(5.0)
    assert row["f1_update_over_pre"] == pytest.approx(1.0)
    assert row["f3_top1_update_share"] == pytest.approx(1.0)
    assert row["f3_topk_update_share"] == pytest.approx(1.0)
    assert row["f3_update_entropy"] == pytest.approx(0.0)


def test_exact_matching_never_crosses_preregistered_nuisance_cells():
    rows = [
        {"state_id": "h1", "cohort": "read_harmful_flip", "cell": "a"},
        {"state_id": "h2", "cohort": "read_harmful_flip", "cell": "a"},
        {"state_id": "b1", "cohort": "read_beneficial_flip", "cell": "a"},
        {"state_id": "b2", "cohort": "read_beneficial_flip", "cell": "b"},
    ]
    matches = exact_nuisance_matches(
        rows,
        treated="read_harmful_flip",
        control="read_beneficial_flip",
        cell_key="cell",
        seed=9,
    )
    assert len(matches) == 1
    assert matches[0]["treated_state_id"] in {"h1", "h2"}
    assert matches[0]["control_state_id"] == "b1"
    assert matches[0]["cell"] == "a"


def test_dense_winner_rule_is_deterministic_with_fixed_tiebreaks():
    candidates = [
        {"feature_group": "F2", "model": "mlp", "spearman": 0.2, "harmful_auroc": 0.6},
        {"feature_group": "F_ALL", "model": "linear", "spearman": 0.2, "harmful_auroc": 0.6},
        {"feature_group": "F1", "model": "linear", "spearman": 0.1, "harmful_auroc": 0.9},
    ]
    winner = select_dense_winner(candidates)
    assert FEATURE_GROUP_ORDER[0] == "F_ALL"
    assert winner["feature_group"] == "F_ALL"
    assert winner["model"] == "linear"


def test_feature_census_rejects_missing_duplicates_and_nonfinite_values():
    validate_feature_census(
        ["a", "b"],
        [
            {"state_id": "a", "features": {"x": 1.0}},
            {"state_id": "b", "features": {"x": 2.0}},
        ],
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_feature_census(
            ["a", "b"],
            [
                {"state_id": "a", "features": {"x": 1.0}},
                {"state_id": "a", "features": {"x": 2.0}},
            ],
        )
    with pytest.raises(ValueError, match="non-finite"):
        validate_feature_census(
            ["a"], [{"state_id": "a", "features": {"x": float("nan")}}]
        )


def test_read_attention_summary_uses_visual_mass_and_text_only_counterfactual():
    query = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    keys = torch.tensor(
        [
            [[0.0, 1.0], [2.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0], [2.0, 0.0]],
        ]
    )
    values = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 2.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
        ]
    )
    output_weight = torch.eye(4)
    visual_mask = torch.tensor([False, True, True])
    positions = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    row = summarize_read_attention(
        query,
        keys,
        values,
        visual_mask=visual_mask,
        output_projection_weight=output_weight,
        residual=torch.ones(4),
        visual_positions=positions[visual_mask],
    )
    assert 0.0 < row["f4_total_visual_attention_mass"] < 1.0
    assert row["f4_top5_attention_mass"] == pytest.approx(
        row["f4_total_visual_attention_mass"]
    )
    assert row["f5_max_qk"] > row["f5_mean_topk_qk"]
    assert row["f6_read_output_norm"] > 0.0
    assert row["f6_visual_value_aggregation_norm"] > 0.0
    assert row["f7_tokens_for_50pct"] == pytest.approx(1.0)
    assert row["f7_tokens_for_80pct"] <= 2.0


def test_attention_reconstruction_requires_all_three_frozen_bounds():
    bounds = {"max_abs_tolerance": 0.0625, "mean_abs_tolerance": 0.002, "min_cosine": 0.9999}
    assert validate_attention_reconstruction(max_abs=0.03125, mean_abs=0.000805, cosine=0.999999, **bounds)
    assert not validate_attention_reconstruction(max_abs=0.125, mean_abs=0.000805, cosine=0.999999, **bounds)
    assert not validate_attention_reconstruction(max_abs=0.03125, mean_abs=0.003, cosine=0.999999, **bounds)
    assert not validate_attention_reconstruction(max_abs=0.03125, mean_abs=0.000805, cosine=0.99, **bounds)


def test_fold_trainer_learns_a_simple_held_out_linear_signal():
    torch.manual_seed(0)
    x = torch.linspace(-2.0, 2.0, 80).unsqueeze(1)
    features = torch.cat((x, x.square()), dim=1)
    target = (2.5 * x[:, 0] - 0.1).numpy()
    result = fit_predict_fold(
        features,
        target,
        fit_indices=np.arange(0, 55),
        calibration_indices=np.arange(55, 65),
        test_indices=np.arange(65, 80),
        fit_weights=np.ones(55),
        calibration_weights=np.ones(10),
        kind="linear",
        spec={
            "learning_rate": 0.03,
            "weight_decay": 0.0,
            "batch_size": 16,
            "minimum_epochs": 20,
            "maximum_epochs": 120,
            "early_stopping_patience": 15,
        },
        hidden_size=8,
        dropout=0.0,
        target_scale_floor=1e-4,
        gradient_clip_norm=1.0,
        seed=4,
        device=torch.device("cpu"),
        classification=False,
    )
    correlation = np.corrcoef(target[65:80], result["test_prediction"])[0, 1]
    assert correlation > 0.98
    assert result["best_epoch"] >= 19


def test_source_holdout_is_group_disjoint_and_uses_only_named_training_source():
    rows = [
        {"uid": "a", "image_group_id": "shared", "source_regime": "historical", "dataset": "gqa"},
        {"uid": "b", "image_group_id": "shared", "source_regime": "canonical", "dataset": "gqa"},
        {"uid": "c", "image_group_id": "train1", "source_regime": "historical", "dataset": "gqa"},
        {"uid": "d", "image_group_id": "train2", "source_regime": "historical", "dataset": "gqa"},
        {"uid": "e", "image_group_id": "unused", "source_regime": "other", "dataset": "gqa"},
        {"uid": "f", "image_group_id": "test", "source_regime": "canonical", "dataset": "gqa"},
    ]
    roles = _holdout_roles(
        {"family": "source", "name": "historical_to_canonical", "train_value": "historical", "test_value": "canonical"},
        rows,
        {"seed": 4, "generalization": {"minimum_test_states": 1}},
    )
    train_indices = set(roles["fit"]) | set(roles["calibration"])
    assert all(rows[index]["source_regime"] == "historical" for index in train_indices)
    assert all(rows[index]["source_regime"] == "canonical" for index in roles["outer_test"])
    assert not ({rows[index]["image_group_id"] for index in train_indices} & {rows[index]["image_group_id"] for index in roles["outer_test"]})
    assert 4 not in train_indices


def test_sparse_matched_probe_derives_group_disjoint_calibration(monkeypatch):
    rows = [
        {"uid": f"u{index}", "image_group_id": f"g{index}", "state_id": f"s{index}"}
        for index in range(10)
    ]
    role_rows = [
        {"uid": row["uid"], "role": "outer_test" if index < 2 else "fit"}
        for index, row in enumerate(rows)
    ]
    monkeypatch.setattr(read_analysis, "read_jsonl", lambda _: role_rows)
    config = {
        "split": {"inner_roles_pattern": "unused_roles{fold}.jsonl"},
        "matching": {"probe_calibration_seed": 9, "probe_calibration_fraction": 0.2},
    }
    roles = _matched_role_indices(config, rows, 0)
    groups = {
        role: {rows[int(index)]["image_group_id"] for index in indices}
        for role, indices in roles.items()
    }
    assert len(roles["outer_test"]) == 2
    assert len(roles["calibration"]) >= 2
    assert not groups["fit"] & groups["calibration"]
    assert not groups["fit"] & groups["outer_test"]
    assert not groups["calibration"] & groups["outer_test"]
