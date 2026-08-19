"""Pure diagnostic contracts for Pareto checkpoint fitting analysis."""

import torch

from experiments.analyze_binary_pareto_training_fit import (
    batch_probability_diagnostics,
    finalize_nonempty,
    validation_reproduction_passes,
    weighted_bce_oracle,
)


def test_weighted_bce_oracle_ignores_padding_and_uses_threshold_ties_as_on():
    masks = torch.tensor(
        [
            [[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]],
            [[1, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    valid = torch.tensor([[True, True, False], [True, False, False]])
    weights = torch.tensor([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])

    oracle = weighted_bce_oracle(masks, valid, weights)

    assert oracle.tolist() == [[1, 1, 1, 1], [1, 0, 1, 0]]


def test_probability_diagnostics_reports_route_mass_and_bit_confidence():
    logits = torch.tensor([[3.0, -3.0], [0.0, 0.0]])
    masks = torch.tensor(
        [
            [[1, 0], [0, 1]],
            [[1, 1], [0, 0]],
        ],
        dtype=torch.float32,
    )
    valid = torch.ones(2, 2, dtype=torch.bool)
    weights = torch.full((2, 2), 0.5)

    result = batch_probability_diagnostics(logits, masks, valid, weights)

    # The first row is confident; the second has every bit at the threshold.
    assert result["bit_count"] == 4
    assert result["near_half_count"] == 2
    assert result["oracle_exact_count"] == 1
    assert result["oracle_hamming_sum"] == 1
    assert result["pareto_hit_count"] == 2
    assert result["max_route_probability_sum"] > 0.9
    assert result["total_pareto_probability_sum"] > result["max_route_probability_sum"]
    assert result["weighted_pareto_mass_sum"] < result["total_pareto_probability_sum"]


def test_probability_diagnostics_singleton_target_confidence_is_separate():
    logits = torch.tensor([[4.0, -4.0], [-4.0, 4.0]])
    masks = torch.tensor([[[1, 0]], [[1, 0]]], dtype=torch.float32)
    valid = torch.ones(2, 1, dtype=torch.bool)
    weights = torch.ones(2, 1)

    result = batch_probability_diagnostics(logits, masks, valid, weights)

    assert result["singleton_sample_count"] == 2
    assert result["singleton_bit_count"] == 4
    assert result["singleton_correct_bit_count"] == 2
    assert result["singleton_confident_correct_bit_count"] == 2


def test_finalize_nonempty_skips_absent_smoke_strata():
    class FakeAccumulator:
        def __init__(self, examples):
            self.examples = examples

        def finalize(self):
            return {"examples": self.examples}

    assert finalize_nonempty(
        {"present": FakeAccumulator(3), "absent": FakeAccumulator(0)}
    ) == {"present": {"examples": 3}}


def test_ampere_diagnostic_reproduction_has_narrow_metric_specific_tolerances():
    acceptable = {
        "pareto_valid_hit_at_1": 1 / 874,
        "original_valid_hit_at_1": 0.0,
        "nearest_valid_hamming": 0.01,
        "average_predicted_visual_on": 0.01,
        "fraction_top1_all_on": 0.0,
        "fraction_top1_all_off": 1 / 874,
        "top1_mask_entropy_nats": 0.01,
        "objective_loss": 0.0005,
    }
    rejected = dict(acceptable, nearest_valid_hamming=0.021)
    rejected_loss = dict(acceptable, objective_loss=0.001)
    reference = dict(acceptable, objective_loss=17.0)

    assert validation_reproduction_passes(acceptable, "ampere_diagnostic", reference)
    assert not validation_reproduction_passes(rejected, "ampere_diagnostic", reference)
    assert not validation_reproduction_passes(rejected_loss, "ampere_diagnostic", reference)
    assert not validation_reproduction_passes(acceptable, "exact", reference)
