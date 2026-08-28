from __future__ import annotations

import math

import pytest
import torch

from four_action_policy.actions import (
    ACTION_TO_INDEX,
    FOUR_ACTIONS,
    decode_action_indices,
    encode_action_route,
)
from four_action_policy.decode import decode_argmax, topk_factorized_routes
from four_action_policy.losses import (
    categorical_route_log_probability,
    exact_valid_set_nll,
    polar_action_bce,
    polar_action_bce_per_route,
)
from four_action_policy.predictor import FourActionPolarBackbone


def test_action_order_matches_the_unified_executor() -> None:
    from binary_policy.executor.four_action import FOUR_ACTIONS as EXECUTOR_ACTIONS

    assert FOUR_ACTIONS == EXECUTOR_ACTIONS
    assert ACTION_TO_INDEX == {
        "IGNORE": 0,
        "READ_ONLY": 1,
        "WRITE_ONLY": 2,
        "FULL": 3,
    }


def test_action_route_round_trip_is_strict() -> None:
    route = ["FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE"]
    encoded = encode_action_route(route, expected_layers=4)
    assert encoded.tolist() == [3, 1, 2, 0]
    assert decode_action_indices(encoded) == tuple(route)

    with pytest.raises(ValueError, match="exactly 4"):
        encode_action_route(route[:3], expected_layers=4)
    with pytest.raises(ValueError, match="unknown four-action value"):
        encode_action_route(["FULL", "OFF"], expected_layers=2)


def test_four_action_predictor_emits_one_logit_per_layer_and_action() -> None:
    model = FourActionPolarBackbone(
        num_layers=5,
        input_dim=7,
        image_dim=11,
        d_model=8,
        num_heads=2,
        num_layer_blocks=1,
        dropout=0.0,
    )
    question = torch.randn(3, 4, 7)
    question_mask = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool
    )
    image = torch.randn(3, 6, 11)
    image_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    logits = model(question, question_mask, image, image_mask)

    assert logits.shape == (3, 5, 4)


def test_categorical_route_log_probability_matches_manual_softmax() -> None:
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, -1.0, 1.0],
                [-1.0, 3.0, 0.5, 0.0],
            ]
        ]
    )
    routes = torch.tensor([[[0, 1], [3, 2]]])

    actual = categorical_route_log_probability(logits, routes)
    expected_log_probs = torch.log_softmax(logits, dim=-1)
    expected = torch.stack(
        [
            expected_log_probs[0, 0, 0] + expected_log_probs[0, 1, 1],
            expected_log_probs[0, 0, 3] + expected_log_probs[0, 1, 2],
        ]
    )

    assert actual.shape == (1, 2)
    assert torch.allclose(actual[0], expected)


def test_exact_valid_set_nll_uses_weighted_complete_route_mass() -> None:
    logits = torch.tensor(
        [
            [
                [4.0, -2.0, -2.0, -2.0],
                [-2.0, 4.0, -2.0, -2.0],
            ]
        ]
    )
    routes = torch.tensor([[[0, 1], [3, 3], [0, 0]]])
    valid = torch.tensor([[True, True, False]])
    weights = torch.tensor([[0.75, 0.25, 0.0]])

    actual = exact_valid_set_nll(
        logits,
        routes,
        valid_mask=valid,
        route_weights=weights,
    )
    log_prob = categorical_route_log_probability(logits, routes)
    expected = -torch.logsumexp(
        torch.stack([log_prob[0, 0] + math.log(0.75), log_prob[0, 1] + math.log(0.25)]),
        dim=0,
    )

    assert torch.allclose(actual, expected)


def test_polar_action_bce_is_one_hot_bce_normalized_by_input_route_weights() -> None:
    logits = torch.zeros(3, 2, 4)
    targets = torch.tensor([[0, 1], [3, 2], [1, 1]])
    weights = torch.tensor([0.25, 0.75, 1.0])

    loss = polar_action_bce(logits, targets, route_weights=weights)

    assert torch.allclose(loss, torch.tensor(math.log(2.0)))


def test_polar_action_bce_per_route_supports_exact_microbatch_accumulation() -> None:
    logits = torch.zeros(3, 2, 4)
    targets = torch.tensor([[0, 1], [3, 2], [1, 1]])

    per_route = polar_action_bce_per_route(logits, targets)

    assert per_route.shape == (3,)
    assert torch.allclose(per_route, torch.full((3,), math.log(2.0)))


def test_argmax_and_topk_decode_complete_four_action_routes() -> None:
    logits = torch.tensor(
        [
            [
                [5.0, 4.0, 0.0, -1.0],
                [-2.0, -1.0, 3.0, 2.0],
            ]
        ]
    )

    top1 = decode_argmax(logits)
    candidates = topk_factorized_routes(logits, top_k=3)[0]

    assert top1.tolist() == [[0, 2]]
    assert candidates[0].actions == ("IGNORE", "WRITE_ONLY")
    assert [candidate.action_indices for candidate in candidates] == [
        (0, 2),
        (0, 3),
        (1, 2),
    ]
    assert candidates[0].log_probability >= candidates[1].log_probability
