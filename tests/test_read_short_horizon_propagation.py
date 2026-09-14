from __future__ import annotations

import numpy as np
import pytest
import torch

from dense_failure_stage2.read_short_horizon import (
    HORIZONS,
    classify_h_read,
    construct_horizon_feature,
    eligible_horizons,
    expected_action_trace,
    matched_random_pair_indices,
    monotonic_nondecreasing,
    pool_horizon_state,
    validate_horizon_census,
    validate_order_invariant_hashes,
)
from experiments.analyze_read_short_horizon_propagation import (
    _canonical_json_hash,
    build_task_grid,
)


def test_canonical_json_hash_accepts_ordered_state_id_sequence() -> None:
    ordered = ["state-a", "state-b"]
    assert _canonical_json_hash(ordered) == _canonical_json_hash(list(ordered))
    assert _canonical_json_hash(ordered) != _canonical_json_hash(list(reversed(ordered)))


def test_eligible_horizons_respect_decoder_boundary() -> None:
    assert eligible_horizons(20, 28) == (1, 2, 4, 8)
    assert eligible_horizons(24, 28) == (1, 2, 4)
    assert eligible_horizons(26, 28) == (1, 2)
    assert eligible_horizons(27, 28) == (1,)


def test_action_trace_differs_only_at_intervention_layer() -> None:
    assert expected_action_trace(4, "FULL") == ("FULL", "FULL", "FULL", "FULL")
    assert expected_action_trace(4, "WRITE_ONLY") == (
        "WRITE_ONLY",
        "FULL",
        "FULL",
        "FULL",
    )
    with pytest.raises(ValueError, match="branch action"):
        expected_action_trace(2, "READ_ONLY")


def test_pool_horizon_state_is_last_text_plus_mean_visual() -> None:
    text = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    visual = torch.tensor([[[2.0, 0.0], [4.0, 2.0]]])
    assert torch.equal(pool_horizon_state(text, visual), torch.tensor([3.0, 4.0, 3.0, 1.0]))


def test_construct_horizon_features_preserves_on_minus_off_sign() -> None:
    on = torch.tensor([[4.0, 2.0]])
    off = torch.tensor([[1.0, 3.0]])
    assert torch.equal(construct_horizon_feature(on, off, "on"), on)
    assert torch.equal(construct_horizon_feature(on, off, "off"), off)
    assert torch.equal(construct_horizon_feature(on, off, "pair"), torch.tensor([[4.0, 2.0, 1.0, 3.0]]))
    assert torch.equal(construct_horizon_feature(on, off, "delta"), torch.tensor([[3.0, -1.0]]))
    assert torch.equal(
        construct_horizon_feature(on, off, "pair_plus_delta"),
        torch.tensor([[4.0, 2.0, 1.0, 3.0, 3.0, -1.0]]),
    )


def test_validate_horizon_census_rejects_missing_or_duplicate_branch() -> None:
    expected = [{"state_id": "a", "layer": 26}, {"state_id": "b", "layer": 27}]
    complete = [
        {"state_id": "a", "horizon": h, "branch": branch}
        for h in (1, 2)
        for branch in ("ON", "OFF")
    ] + [
        {"state_id": "b", "horizon": 1, "branch": branch}
        for branch in ("ON", "OFF")
    ]
    validate_horizon_census(expected, complete, num_layers=28)
    with pytest.raises(ValueError, match="census"):
        validate_horizon_census(expected, complete[:-1], num_layers=28)
    with pytest.raises(ValueError, match="duplicated"):
        validate_horizon_census(expected, complete + [complete[0]], num_layers=28)


def test_swapped_order_validation_requires_exact_per_horizon_hashes() -> None:
    forward = {
        ("ON", 1): "a",
        ("OFF", 1): "b",
        ("ON", 8): "c",
        ("OFF", 8): "d",
    }
    validate_order_invariant_hashes(forward, dict(forward))
    changed = dict(forward)
    changed[("OFF", 8)] = "different"
    with pytest.raises(ValueError, match="order"):
        validate_order_invariant_hashes(forward, changed)


def test_random_pairing_is_deterministic_and_never_same_uid() -> None:
    rows = [
        {"state_id": f"s{i}", "uid": f"u{i}", "layer": 20, "dataset": "gqa", "source_regime": "canonical", "dense_wrong": bool(i % 2)}
        for i in range(8)
    ]
    first = matched_random_pair_indices(rows, seed=7)
    second = matched_random_pair_indices(rows, seed=7)
    assert np.array_equal(first, second)
    assert all(rows[i]["uid"] != rows[int(j)]["uid"] for i, j in enumerate(first))


def test_permuted_target_pairing_is_uid_level() -> None:
    rows = [
        {"state_id": "a0", "uid": "a", "layer": 20, "dataset": "gqa", "source_regime": "canonical", "dense_wrong": True},
        {"state_id": "a1", "uid": "a", "layer": 21, "dataset": "gqa", "source_regime": "canonical", "dense_wrong": True},
        {"state_id": "b0", "uid": "b", "layer": 20, "dataset": "gqa", "source_regime": "canonical", "dense_wrong": False},
        {"state_id": "c0", "uid": "c", "layer": 20, "dataset": "gqa", "source_regime": "canonical", "dense_wrong": True},
    ]
    indices = matched_random_pair_indices(rows, seed=11, uid_permutation=True)
    assert rows[int(indices[0])]["uid"] == rows[int(indices[1])]["uid"]
    assert rows[int(indices[0])]["uid"] != "a"


def test_monotonicity_uses_fixed_horizon_order() -> None:
    assert monotonic_nondecreasing({1: 0.01, 2: 0.02, 4: 0.02, 8: 0.04})
    assert not monotonic_nondecreasing({1: 0.01, 2: 0.03, 4: 0.02, 8: 0.04})


def test_classification_prefers_counterfactual_emergence_when_all_gates_pass() -> None:
    result = classify_h_read(
        h1_spearman=0.05,
        h1_auroc=0.52,
        horizon_spearman={2: 0.08, 4: 0.18, 8: 0.21},
        horizon_auroc={2: 0.54, 4: 0.62, 8: 0.64},
        best_single_spearman={4: 0.06, 8: 0.07},
        random_pair_spearman={4: 0.01, 8: 0.02},
        token_spearman={1: 0.04, 2: 0.06, 4: 0.09, 8: 0.10},
        ci_lower_spearman={2: -0.01, 4: 0.04, 8: 0.08},
        ci_lower_auroc={2: -0.01, 4: 0.02, 8: 0.04},
        ci_lower_token_spearman={2: -0.02, 4: -0.01, 8: -0.01},
        precision_top10={1: 0.52, 2: 0.55, 4: 0.70, 8: 0.72},
        prevalence=0.48,
        thresholds={"spearman_gain": 0.10, "auroc_gain": 0.08, "pair_over_single": 0.05, "random_pair_gap": 0.05, "useful_precision_gain": 0.10},
    )
    assert result["category"] == "H-READ-A"
    assert result["smallest_material_horizon"] == 4


def test_classification_delayed_single_state_and_token_categories() -> None:
    common = dict(
        h1_spearman=0.02,
        h1_auroc=0.51,
        horizon_auroc={2: 0.52, 4: 0.58, 8: 0.60},
        random_pair_spearman={4: 0.01, 8: 0.01},
        ci_lower_spearman={2: -0.01, 4: 0.03, 8: 0.08},
        ci_lower_auroc={2: -0.01, 4: 0.01, 8: 0.02},
        ci_lower_token_spearman={2: -0.01, 4: 0.01, 8: 0.03},
        precision_top10={1: 0.5, 2: 0.52, 4: 0.62, 8: 0.65},
        prevalence=0.48,
        thresholds={"spearman_gain": 0.10, "auroc_gain": 0.08, "pair_over_single": 0.05, "random_pair_gap": 0.05, "useful_precision_gain": 0.10},
    )
    delayed = classify_h_read(
        horizon_spearman={2: 0.06, 4: 0.13, 8: 0.18},
        best_single_spearman={4: 0.12, 8: 0.17},
        token_spearman={1: 0.03, 2: 0.05, 4: 0.08, 8: 0.09},
        **common,
    )
    assert delayed["category"] == "H-READ-B"
    token_common = dict(common)
    token_common["horizon_auroc"] = {2: 0.52, 4: 0.53, 8: 0.54}
    token = classify_h_read(
        horizon_spearman={2: 0.03, 4: 0.05, 8: 0.06},
        best_single_spearman={4: 0.04, 8: 0.05},
        token_spearman={1: 0.02, 2: 0.05, 4: 0.10, 8: 0.16},
        **token_common,
    )
    assert token["category"] == "H-READ-C"


def test_classification_defaults_to_no_short_horizon_identifiability() -> None:
    result = classify_h_read(
        h1_spearman=0.05,
        h1_auroc=0.52,
        horizon_spearman={2: 0.06, 4: 0.07, 8: 0.08},
        horizon_auroc={2: 0.53, 4: 0.54, 8: 0.55},
        best_single_spearman={4: 0.06, 8: 0.07},
        random_pair_spearman={4: 0.04, 8: 0.05},
        token_spearman={1: 0.04, 2: 0.05, 4: 0.06, 8: 0.07},
        ci_lower_spearman={2: -0.01, 4: -0.01, 8: -0.01},
        ci_lower_auroc={2: -0.01, 4: -0.01, 8: -0.01},
        ci_lower_token_spearman={2: -0.01, 4: -0.01, 8: -0.01},
        precision_top10={1: 0.50, 2: 0.51, 4: 0.52, 8: 0.53},
        prevalence=0.48,
        thresholds={"spearman_gain": 0.10, "auroc_gain": 0.08, "pair_over_single": 0.05, "random_pair_gap": 0.05, "useful_precision_gain": 0.10},
    )
    assert result["category"] == "H-READ-D"


def test_horizon_constant_is_frozen() -> None:
    assert HORIZONS == (1, 2, 4, 8)


def test_training_grid_freezes_all_horizons_conditions_and_token_comparator() -> None:
    config = {
        "features": {
            "pooled_conditions": ["on", "off", "pair", "delta", "pair_plus_delta"],
            "delta_conditions": ["text_delta", "visual_delta", "text_visual_delta"],
        },
        "training": {"seeds": [1, 2, 3]},
        "split": {"folds": 5},
        "controls": {
            "random_pair_horizons": [1, 4, 8],
            "random_pair_condition": "delta",
            "permuted_target_horizons": [1, 8],
            "permuted_target_condition": "delta",
        },
    }
    tasks = build_task_grid(config)
    assert len(tasks) == 2115
    assert len({task["task_id"] for task in tasks}) == len(tasks)
    token_cells = {
        (task["support"], task["horizon"])
        for task in tasks
        if task["model"] == "token_comparator" and task["family"] == "standard"
    }
    assert token_cells == {(support, horizon) for support in ("native", "common_h8") for horizon in HORIZONS}


def test_controls_are_common_support_delta_mlp_only() -> None:
    config = {
        "features": {
            "pooled_conditions": ["on", "off", "pair", "delta", "pair_plus_delta"],
            "delta_conditions": ["text_delta", "visual_delta", "text_visual_delta"],
        },
        "training": {"seeds": [1]},
        "split": {"folds": 2},
        "controls": {
            "random_pair_horizons": [1, 4, 8],
            "random_pair_condition": "delta",
            "permuted_target_horizons": [1, 8],
            "permuted_target_condition": "delta",
        },
    }
    controls = [task for task in build_task_grid(config) if task["family"] != "standard"]
    assert {(task["family"], task["horizon"]) for task in controls} == {
        ("random_pair", 1), ("random_pair", 4), ("random_pair", 8),
        ("permuted_target", 1), ("permuted_target", 8),
    }
    assert all(task["support"] == "common_h8" for task in controls)
    assert all(task["condition"] == "delta" and task["model"] == "mlp" for task in controls)
