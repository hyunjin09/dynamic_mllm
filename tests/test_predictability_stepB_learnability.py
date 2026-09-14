from __future__ import annotations

import numpy as np
import pytest
import torch
import json
from pathlib import Path

from dense_failure_stage2.predictability_learnability import (
    RouterStyleUtilityRegressor,
    SummaryScalarPredictor,
    assign_inner_group_roles,
    assign_shared_group_folds,
    binary_classification_metrics,
    fit_standardizer,
    harmful_ranking_metrics,
    high_precision_harmful_metrics,
    primary_utility_targets,
    regression_metrics,
    robust_target_scale,
    select_preservation_threshold,
    uid_equal_state_weights,
    uid_macro_regression_metrics,
    validate_prediction_roundtrip,
    validate_oof_completeness,
)
from experiments.aggregate_predictability_stepB_learnability import (
    _seed_metric_summary,
    _uid_sequential_operating_point,
    _weighted_binary,
)
from experiments.run_predictability_stepB_learnability import (
    PackedStateStore,
    _smoke_tasks,
    _transfer_tasks,
    iter_prefetched_packed_batches,
    combined_state_hash,
    resolve_stepa_artifact_path,
)
from analysis.predictability_generalization.stepB_id_learnability.aggregation_patch_runner import (
    _apply_stage2_schema_patch,
    _weighted_binary_vectorized,
)


def _base_rows() -> list[dict[str, object]]:
    rows = []
    for index in range(60):
        rows.append(
            {
                "uid": f"uid-{index}",
                "image_group_id": f"group-{index // 2}",
                "dataset": ("gqa", "chartqa", "textvqa")[index % 3],
                "source_regime": ("historical", "canonical")[index % 2],
                "dense_wrong": bool(index % 2),
                "p90_triggered": bool(index % 5 == 0),
            }
        )
    return rows


def test_shared_fold_registry_is_deterministic_group_disjoint_and_complete():
    rows = _base_rows()
    first = assign_shared_group_folds(rows, folds=5, seed=17)
    second = assign_shared_group_folds(list(reversed(rows)), folds=5, seed=17)
    assert first == second
    assert {row["uid"] for row in first} == {row["uid"] for row in rows}
    assert {row["fold"] for row in first} == set(range(5))

    group_folds: dict[str, set[int]] = {}
    for row in first:
        group_folds.setdefault(str(row["image_group_id"]), set()).add(int(row["fold"]))
    assert all(len(values) == 1 for values in group_folds.values())


def test_shared_fold_registry_spreads_rare_supported_strata():
    rows = []
    for index in range(100):
        rows.append(
            {
                "uid": f"rare-{index}",
                "image_group_id": f"rare-group-{index}",
                "dataset": "gqa",
                "source_regime": "historical",
                "dense_wrong": bool(index < 5),
                "p90_triggered": bool(index < 5),
            }
        )
    registry = assign_shared_group_folds(rows, folds=5, seed=29)
    rare_folds = {
        int(row["fold"]) for row in registry
        if bool(row["dense_wrong"]) and bool(row["p90_triggered"])
    }
    assert rare_folds == set(range(5))


def test_inner_roles_never_use_outer_test_and_keep_groups_intact():
    registry = assign_shared_group_folds(_base_rows(), folds=5, seed=19)
    roles = assign_inner_group_roles(
        registry, outer_fold=2, calibration_fraction=0.2, seed=23
    )
    by_uid = {str(row["uid"]): str(row["role"]) for row in roles}
    assert set(by_uid) == {str(row["uid"]) for row in registry}
    for row in registry:
        expected = "outer_test" if int(row["fold"]) == 2 else None
        if expected:
            assert by_uid[str(row["uid"])] == expected

    group_roles: dict[str, set[str]] = {}
    for row in roles:
        group_roles.setdefault(str(row["image_group_id"]), set()).add(str(row["role"]))
    assert all(len(values) == 1 for values in group_roles.values())
    assert {"fit", "calibration", "outer_test"} == set(by_uid.values())


def test_uid_equal_weights_give_each_uid_equal_total_mass():
    rows = [
        {"state_id": "a0", "uid": "a"},
        {"state_id": "a1", "uid": "a"},
        {"state_id": "a2", "uid": "a"},
        {"state_id": "b0", "uid": "b"},
    ]
    weights = uid_equal_state_weights(rows)
    assert weights.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 1.0])
    assert weights[:3].sum() == pytest.approx(weights[3:].sum())


def test_primary_targets_use_dense_full_one_bit_deviations():
    row = {
        "q_full": 0.4,
        "q_read_only": 0.1,
        "q_write_only": 0.7,
        "q_ignore": -0.2,
        "u_read_w1": -0.3,
        "u_write_r1": 0.3,
    }
    targets = primary_utility_targets(row)
    assert targets == {"read": pytest.approx(-0.3), "write": pytest.approx(0.3)}


def test_robust_target_scaling_uses_only_provided_training_values():
    scale = robust_target_scale(np.asarray([-2.0, -1.0, 0.0, 1.0, 100.0]), floor=0.1)
    assert scale.center == pytest.approx(0.0)
    assert scale.scale == pytest.approx(1.4826)
    transformed = scale.transform(np.asarray([0.0, 1.4826]))
    assert transformed.tolist() == pytest.approx([0.0, 1.0])
    assert scale.inverse(transformed).tolist() == pytest.approx([0.0, 1.4826])


def test_router_style_regressor_supports_read_write_and_joint_views():
    text = torch.randn(3, 2, 8)
    visual = torch.randn(3, 5, 8)
    text_mask = torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.bool)
    visual_mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool
    )
    for representation in ("z_R", "z_W", "z_RW"):
        model = RouterStyleUtilityRegressor(
            hidden_size=8,
            router_size=4,
            num_heads=2,
            dropout=0.0,
            readout_hidden_size=4,
            representation=representation,
        )
        prediction, details = model(
            text, visual, text_mask=text_mask, visual_mask=visual_mask, return_details=True
        )
        assert prediction.shape == (3,)
        assert details["z_R"].shape == (3, 4)
        assert details["z_W"].shape == (3, 4)
        assert torch.isfinite(prediction).all()


def test_harmful_metrics_exclude_exact_zero_targets():
    metrics = harmful_ranking_metrics(
        truth=np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]),
        prediction=np.asarray([-1.5, -0.5, -100.0, 0.5, 1.5]),
    )
    assert metrics["support"] == 4
    assert metrics["neutral_excluded"] == 1
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)


def test_oof_completeness_rejects_missing_duplicate_and_wrong_fold_rows():
    expected = {
        "s0": {"fold": 0},
        "s1": {"fold": 1},
    }
    valid = [
        {"state_id": "s0", "fold": 0},
        {"state_id": "s1", "fold": 1},
    ]
    validate_oof_completeness(expected, valid)
    with pytest.raises(ValueError, match="missing"):
        validate_oof_completeness(expected, valid[:1])
    with pytest.raises(ValueError, match="duplicate"):
        validate_oof_completeness(expected, [valid[0], valid[0], valid[1]])
    with pytest.raises(ValueError, match="fold"):
        validate_oof_completeness(expected, [valid[0], {"state_id": "s1", "fold": 0}])


def test_fold_standardizer_uses_only_selected_fit_rows():
    features = torch.tensor([[0.0, 0.0], [2.0, 4.0], [100.0, 100.0]])
    standardizer = fit_standardizer(features, np.asarray([0, 1]), floor=1e-6)
    assert standardizer.mean.tolist() == pytest.approx([1.0, 2.0])
    assert standardizer.std.tolist() == pytest.approx([1.0, 2.0])
    assert standardizer.transform(features[:2]).mean(dim=0).tolist() == pytest.approx([0.0, 0.0])


def test_summary_predictor_capacity_contracts_are_scalar():
    values = torch.randn(7, 6)
    linear = SummaryScalarPredictor(kind="linear", input_size=6, hidden_size=4, dropout=0.0)
    mlp = SummaryScalarPredictor(kind="mlp", input_size=6, hidden_size=4, dropout=0.0)
    assert linear(values).shape == (7,)
    assert mlp(values).shape == (7,)


def test_regression_and_high_precision_metrics_follow_prediction_order():
    truth = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    prediction = truth.copy()
    metrics = regression_metrics(truth=truth, prediction=prediction)
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(0.0)
    precision = high_precision_harmful_metrics(
        truth=truth, prediction=prediction, coverages=(0.5,), precision_targets=(0.9,)
    )
    assert precision["precision_at_0.5"] == pytest.approx(1.0)
    assert precision["recall_at_precision_0.9"] == pytest.approx(1.0)


def test_preservation_threshold_is_selected_without_heldout_labels():
    calibration_scores = np.asarray([0.1, 0.2, 0.3, 0.8, 0.9])
    calibration_wrong = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    threshold = select_preservation_threshold(
        calibration_scores, calibration_wrong, target_correct_preservation=2 / 3
    )
    # strict score > threshold triggers; threshold 0.2 preserves exactly 2/3 correct rows.
    assert threshold == pytest.approx(0.2)


def test_public_binary_metrics_report_support_and_prevalence():
    result = binary_classification_metrics(
        truth=np.asarray([0, 0, 1, 1]), prediction=np.asarray([0.1, 0.2, 0.8, 0.9])
    )
    assert result["support"] == 4
    assert result["positives"] == 2
    assert result["prevalence"] == pytest.approx(0.5)
    assert result["auroc"] == pytest.approx(1.0)
    assert result["auprc"] == pytest.approx(1.0)


def test_uid_macro_metrics_average_valid_per_uid_correlations():
    truth = np.asarray([0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 7.0, 7.0, 7.0])
    prediction = np.asarray([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0])
    uids = np.asarray(["a"] * 3 + ["b"] * 3 + ["c"] * 3)
    result = uid_macro_regression_metrics(
        truth=truth, prediction=prediction, uids=uids, minimum_nonconstant_states=3
    )
    assert result["uids_total"] == 3
    assert result["uids_included"] == 2
    assert result["spearman"] == pytest.approx(0.0)


def test_smoke_and_transfer_task_registries_cover_four_workers():
    config = json.loads(
        (Path(__file__).parents[1] / "configs/predictability_stepB_id_learnability_v1.json").read_text()
    )
    smoke = _smoke_tasks(config)
    transfer = _transfer_tasks(config)
    assert len(smoke) == 20
    assert len(transfer) == 140
    assert {int(row["worker_rank"]) for row in smoke} == {0, 1, 2, 3}
    assert {int(row["worker_rank"]) for row in transfer} == {0, 1, 2, 3}
    assert len({str(row["task_id"]) for row in transfer}) == len(transfer)


def test_weighted_binary_matches_unweighted_and_handles_ties():
    labels = np.asarray([0, 1, 0, 1])
    scores = np.asarray([0.0, 0.5, 0.5, 1.0])
    auroc, _ = _weighted_binary(labels, scores, np.ones(4))
    assert auroc == pytest.approx(0.875)


def test_uid_sequential_operating_point_calibrates_whole_sample_preservation():
    calibration_rows = [
        {"uid": "c1", "layer": 0},
        {"uid": "c1", "layer": 1},
        {"uid": "c2", "layer": 0},
        {"uid": "c2", "layer": 1},
        {"uid": "w1", "layer": 0},
        {"uid": "w1", "layer": 1},
    ]
    calibration_scores = np.asarray([0.1, 0.9, 0.2, 0.3, 0.4, 0.8])
    calibration_wrong = np.asarray([0, 0, 0, 0, 1, 1])
    test_rows = [
        {"uid": "c3", "layer": 0},
        {"uid": "c3", "layer": 1},
        {"uid": "w2", "layer": 0},
        {"uid": "w2", "layer": 1},
    ]
    result = _uid_sequential_operating_point(
        calibration_rows=calibration_rows,
        calibration_scores=calibration_scores,
        calibration_wrong=calibration_wrong,
        test_rows=test_rows,
        test_scores=np.asarray([0.2, 0.95, 0.91, 0.1]),
        test_wrong=np.asarray([0, 0, 1, 1]),
        target_correct_preservation=0.5,
    )
    # Correct calibration UID maxima are 0.9 and 0.3, so strict-greater
    # triggering at 0.3 preserves exactly one of two calibration samples.
    assert result["threshold"] == pytest.approx(0.3)
    assert result["heldout_correct_preservation"] == pytest.approx(0.0)
    assert result["heldout_wrong_recall_uid"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(0.5)
    assert result["median_trigger_layer"] == pytest.approx(0.5)


def test_seed_metric_summary_reports_mean_and_sample_standard_deviation():
    rows = [
        {"domain": "stage1", "target": "dense_wrong", "model": "m2", "input": "state", "seed": 1, "auroc": 0.6},
        {"domain": "stage1", "target": "dense_wrong", "model": "m2", "input": "state", "seed": 2, "auroc": 0.8},
        {"domain": "stage1", "target": "dense_wrong", "model": "m1", "input": "state", "seed": 1, "auroc": 0.7},
    ]
    summary = _seed_metric_summary(rows)
    m2 = next(row for row in summary if row["model"] == "m2")
    m1 = next(row for row in summary if row["model"] == "m1")
    assert m2["seeds"] == 2
    assert m2["auroc_mean"] == pytest.approx(0.7)
    assert m2["auroc_std"] == pytest.approx(np.sqrt(0.02))
    assert m1["seeds"] == 1
    assert m1["auroc_std"] == pytest.approx(0.0)


def test_aggregation_patch_exposes_expected_stage2_oof_keys_without_mutation():
    payload = {
        "oof_read": [{"state_id": "read-0"}],
        "oof_write": [{"state_id": "write-0"}],
        "continuous": [],
    }
    patched = _apply_stage2_schema_patch(payload)
    assert patched["read"] is payload["oof_read"]
    assert patched["write"] is payload["oof_write"]
    assert patched["continuous"] == []
    assert "read" not in payload and "write" not in payload


def test_vectorized_weighted_binary_matches_frozen_reference_with_ties():
    from experiments.aggregate_predictability_stepB_learnability import _weighted_binary

    labels = np.asarray([0, 1, 0, 1, 1, 0, 1], dtype=np.int64)
    scores = np.asarray([0.2, 0.8, 0.5, 0.5, 0.9, 0.2, 0.5], dtype=np.float64)
    for weights in (
        np.ones(7, dtype=np.float64),
        np.asarray([0.0, 2.0, 3.0, 1.0, 4.0, 5.0, 2.0]),
    ):
        expected = _weighted_binary(labels, scores, weights)
        observed = _weighted_binary_vectorized(labels, scores, weights)
        assert observed == pytest.approx(expected, rel=1e-14, abs=1e-14)


def test_relative_state_paths_are_bound_to_stepa_artifact_root():
    config = {
        "sources": {
            "stepA_contract": "analysis/predictability_generalization/stepA_measurement/frozen_contract.json"
        }
    }
    resolved = resolve_stepa_artifact_path(config, "stage2_dense/states/example.pt")
    assert resolved == (
        Path(__file__).resolve().parents[1]
        / "analysis/predictability_generalization/stepA_measurement/stage2_dense/states/example.pt"
    )


def test_combined_state_hash_matches_stepa_canonical_json_contract():
    import hashlib

    hashes = {"visual_states": "b", "text_states": "a"}
    expected = hashlib.sha256(b'{"text_states":"a","visual_states":"b"}').hexdigest()
    assert combined_state_hash(hashes) == expected


def test_checkpoint_prediction_roundtrip_uses_fixed_float32_ulp_bound():
    result = validate_prediction_roundtrip(
        np.asarray([0.0, 1.0]),
        np.asarray([5e-8, 1.0 - 5e-8]),
        float32_ulp_multiplier=2.0,
    )
    assert result["maximum_absolute_difference"] == pytest.approx(5e-8)
    assert result["maximum_scaled_float32_ulps"] < 1.0
    with pytest.raises(ValueError, match="exceeds float32 ULP bound"):
        validate_prediction_roundtrip(
            np.asarray([0.0]), np.asarray([3e-7]), float32_ulp_multiplier=2.0
        )


def test_packed_collation_pinned_and_unpinned_are_bit_identical():
    store = PackedStateStore.__new__(PackedStateStore)
    store.hidden_size = 2
    store.rows = [
        {"text_tokens": 1, "visual_tokens": 2, "text_offset": 0, "visual_offset": 0},
        {"text_tokens": 1, "visual_tokens": 1, "text_offset": 1, "visual_offset": 2},
    ]
    text = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    visual = torch.tensor(
        [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=torch.bfloat16
    )
    store.text = text.view(torch.uint16).numpy()
    store.visual = visual.view(torch.uint16).numpy()
    store.text_mask = np.asarray([1, 1], dtype=np.uint8)
    store.visual_mask = np.asarray([1, 1, 1], dtype=np.uint8)
    plain = store.collate(np.asarray([0, 1]), pin_memory=False)
    pinned = store.collate(np.asarray([0, 1]), pin_memory=True)
    assert all(torch.equal(left, right) for left, right in zip(plain, pinned, strict=True))
    assert all(value.is_pinned() for value in pinned)


def test_prefetched_packed_batches_preserve_exact_batch_order():
    class FakeStore:
        def collate(self, indices, *, pin_memory):
            assert pin_memory is False
            values = torch.as_tensor(indices, dtype=torch.int64)
            return values, values + 1, values >= 0, values >= 0

    batches = [np.asarray([3, 1]), np.asarray([4]), np.asarray([2, 0])]
    observed = list(
        iter_prefetched_packed_batches(FakeStore(), batches, pin_memory=False)
    )
    assert [indices.tolist() for indices, _ in observed] == [
        [3, 1], [4], [2, 0]
    ]
    assert [packed[0].tolist() for _, packed in observed] == [[3, 1], [4], [2, 0]]
