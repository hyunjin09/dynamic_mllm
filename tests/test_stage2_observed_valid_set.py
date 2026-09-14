import torch
import torch.nn.functional as F

from dense_failure_stage2.observed_valid_set import (
    build_exact_valid_sets,
    exact_state_id,
    observed_valid_metrics,
    observed_valid_set_loss,
    uid_paired_values,
    valid_action_mask,
)


def test_exact_state_identity_uses_complete_entering_prefix():
    one = ["FULL", "READ_ONLY", "FULL"]
    two = ["FULL", "WRITE_ONLY", "FULL"]
    assert exact_state_id("u", 1, one) == exact_state_id("u", 1, two)
    assert exact_state_id("u", 2, one) != exact_state_id("u", 2, two)


def test_build_exact_valid_sets_merges_only_exact_prefixes():
    routes = [
        {"route_id": "r1", "route_source": "single", "uid": "u", "activation_layer": 0, "actions": ["FULL", "READ_ONLY", "FULL"] + ["FULL"] * 25},
        {"route_id": "r2", "route_source": "mcts", "uid": "u", "activation_layer": 0, "actions": ["FULL", "IGNORE", "FULL"] + ["FULL"] * 25},
    ]
    states, occurrences = build_exact_valid_sets(routes)
    by_layer = {row["layer"]: row for row in states if row["layer"] < 2}
    assert by_layer[0]["observed_valid_actions"] == ["FULL"]
    assert by_layer[1]["observed_valid_actions"] == ["READ_ONLY", "IGNORE"]
    assert len(occurrences) == 56


def test_single_valid_set_loss_equals_cross_entropy():
    logits = torch.tensor([[1.2, -0.3, 2.1, 0.4], [-1.0, 0.2, 0.5, 1.7]], dtype=torch.float64)
    targets = torch.tensor([2, 3])
    mask = valid_action_mask([[2], [3]])
    actual = observed_valid_set_loss(logits, mask)
    expected = F.cross_entropy(logits.float(), targets)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_valid_set_loss_rewards_total_valid_mass_and_is_finite():
    logits = torch.tensor([[0.0, 2.0, -1.0, 1.0]], requires_grad=True)
    mask = valid_action_mask([[1, 3]])
    loss = observed_valid_set_loss(logits, mask)
    metrics = observed_valid_metrics(logits, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert metrics["observed_valid_top1"] == 1
    assert metrics["valid_probability_mass"][0] > 0.8


def test_invalid_valid_set_is_rejected():
    try:
        valid_action_mask([[]])
    except ValueError as error:
        assert "invalid or empty" in str(error)
    else:
        raise AssertionError("empty valid set was accepted")


def test_uid_paired_values_uses_uid_means():
    rows = [
        {"uid": "u", "checkpoint": "B", "score": 0.0},
        {"uid": "u", "checkpoint": "B", "score": 1.0},
        {"uid": "u", "checkpoint": "C", "score": 1.0},
        {"uid": "u", "checkpoint": "C", "score": 1.0},
    ]
    assert uid_paired_values(rows, "score", "B", "C") == {"u": (0.5, 1.0)}
