from __future__ import annotations

import numpy as np
import pytest
import torch

from dense_failure_stage1.layerwise_probe import (
    FEATURE_NAMES,
    binary_metrics,
    compose_layer_features,
    fit_linear_probe,
    preservation_operating_point,
    score_linear_probe,
    validate_checkpoint_provenance,
    validate_complete_layer_records,
)


def test_binary_metrics_report_rank_and_fixed_threshold_metrics() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])

    assert metrics["auroc"] == pytest.approx(0.75)
    assert metrics["auprc"] == pytest.approx(0.8333333333333333)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["wrong_recall"] == pytest.approx(0.5)
    assert metrics["correct_preservation"] == pytest.approx(0.5)


def test_preservation_threshold_is_validation_only_and_tie_safe() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.3, 0.4, 0.25, 0.35, 0.8, 0.9])

    strict = preservation_operating_point(labels, scores, target_preservation=0.99)
    relaxed = preservation_operating_point(labels, scores, target_preservation=0.75)

    assert strict["allowed_false_deviations"] == 0
    assert strict["correct_preservation"] == pytest.approx(1.0)
    assert strict["wrong_recall"] == pytest.approx(0.5)
    assert strict["threshold"] > 0.4
    assert relaxed["allowed_false_deviations"] == 1
    assert relaxed["correct_preservation"] == pytest.approx(0.75)
    assert relaxed["wrong_recall"] == pytest.approx(0.75)
    assert relaxed["threshold"] > 0.3
    assert relaxed["threshold"] <= 0.4

    tied = preservation_operating_point(
        [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.9], target_preservation=0.5
    )
    assert tied["correct_preservation"] == pytest.approx(1.0)
    assert tied["wrong_recall"] == pytest.approx(0.5)


def test_compose_layer_features_uses_all_frozen_summaries_in_fixed_order() -> None:
    shard = {
        "text_final": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        "text_mean": torch.arange(24, 48, dtype=torch.float32).reshape(2, 3, 4),
        "visual_mean": torch.arange(48, 72, dtype=torch.float32).reshape(2, 3, 4),
    }

    values = compose_layer_features(shard, row_indices=[1, 0], layer=2)

    assert FEATURE_NAMES == ("text_final", "text_mean", "visual_mean")
    assert values.shape == (2, 12)
    assert torch.equal(values[0, :4], shard["text_final"][1, 2])
    assert torch.equal(values[0, 4:8], shard["text_mean"][1, 2])
    assert torch.equal(values[0, 8:], shard["visual_mean"][1, 2])


def test_complete_layer_validation_rejects_missing_and_duplicate_results() -> None:
    complete = [{"layer": layer} for layer in range(28)]
    assert validate_complete_layer_records(complete) == complete

    with pytest.raises(ValueError, match="exactly one result"):
        validate_complete_layer_records(complete[:-1])
    with pytest.raises(ValueError, match="exactly one result"):
        validate_complete_layer_records(complete + [{"layer": 27}])


def test_checkpoint_provenance_rejects_any_contract_or_layer_mismatch() -> None:
    payload = {
        "schema_version": "layerwise_dense_failure_probe_checkpoint_v1",
        "layer": 7,
        "contract_sha256": "contract",
        "split_manifest_sha256": "split",
        "feature_schema_sha256": "schema",
        "feature_index_sha256": "index",
        "dense_outputs_sha256": "outputs",
        "input_features": list(FEATURE_NAMES),
    }
    expected = {
        "contract_sha256": "contract",
        "split_manifest_sha256": "split",
        "feature_schema_sha256": "schema",
        "feature_index_sha256": "index",
        "dense_outputs_sha256": "outputs",
    }

    validate_checkpoint_provenance(payload, layer=7, expected=expected)

    changed = dict(payload)
    changed["feature_index_sha256"] = "different"
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_checkpoint_provenance(changed, layer=7, expected=expected)
    with pytest.raises(ValueError, match="layer mismatch"):
        validate_checkpoint_provenance(payload, layer=8, expected=expected)


def test_linear_probe_fit_is_deterministic_and_uses_train_only_normalization() -> None:
    train_x = torch.tensor(
        [[-3.0, 0.0], [-2.0, 0.2], [-1.0, -0.1], [1.0, 0.1], [2.0, -0.2], [3.0, 0.0]]
    )
    train_y = torch.tensor([0, 0, 0, 1, 1, 1])
    val_x = torch.tensor([[-4.0, 0.0], [-1.5, 0.1], [1.5, -0.1], [4.0, 0.0]])
    val_y = torch.tensor([0, 0, 1, 1])
    config = {
        "learning_rate": 0.05,
        "weight_decay": 0.01,
        "batch_size": 3,
        "maximum_epochs": 60,
        "minimum_epochs": 10,
        "early_stopping_patience": 8,
        "normalization_std_floor": 1e-6,
    }

    first = fit_linear_probe(
        train_x, train_y, val_x, val_y, config=config, seed=17, device=torch.device("cpu")
    )
    second = fit_linear_probe(
        train_x, train_y, val_x, val_y, config=config, seed=17, device=torch.device("cpu")
    )

    assert torch.equal(first["mean"], train_x.mean(dim=0))
    assert torch.equal(first["weight"], second["weight"])
    assert torch.equal(first["bias"], second["bias"])
    scores = score_linear_probe(val_x, first, device=torch.device("cpu"))
    assert binary_metrics(val_y.numpy(), scores.numpy())["auroc"] == pytest.approx(1.0)
