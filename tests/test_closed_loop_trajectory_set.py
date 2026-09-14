from __future__ import annotations

import math

import pytest
import torch

from dense_failure_stage2.closed_loop_trajectory_set import (
    build_trajectory_sets,
    compact_router_state,
    compute_uid_trajectory_loss,
    prefix_action_hash,
    trajectory_set_marginal_loss,
    validate_state_references,
)
from dense_failure_stage2.v1_router import SharedReadWriteRouter
from experiments.run_closed_loop_trajectory_set import (
    _fixed_rank_uids,
    load_config,
    read_jsonl,
)


def _row(uid: str, program: str, outcome: str, suffix: list[str], *, trigger: int = 25):
    return {
        "uid": uid,
        "program_id": program,
        "dataset": "gqa",
        "source_regime": "canonical",
        "dense_outcome": outcome,
        "trigger_layer": trigger,
        "suffix_actions": suffix,
        "full_actions": ["FULL"] * trigger + suffix,
        "image_group_id": f"image:{uid}",
        "provenance": ["preservation" if outcome == "C" else "single"],
    }


def test_build_trajectory_sets_keeps_one_uid_loss_unit_and_all_w_routes():
    rows = [
        _row("c", "c0", "C", ["FULL", "FULL", "FULL"]),
        _row("w", "w0", "W", ["READ_ONLY", "FULL", "FULL"]),
        _row("w", "w1", "W", ["FULL", "WRITE_ONLY", "FULL"]),
    ]

    grouped, summary = build_trajectory_sets(rows, total_layers=28)

    assert sorted(grouped) == ["c", "w"]
    assert [row["program_id"] for row in grouped["w"]] == ["w0", "w1"]
    assert summary == {
        "uids": 2,
        "programs": 3,
        "dense_c_uids": 1,
        "dense_w_uids": 1,
        "route_state_occurrences": 9,
        "unique_prefix_states": 8,
    }


def test_build_trajectory_sets_rejects_nonfull_dense_c_and_duplicate_programs():
    with pytest.raises(ValueError, match="Dense-C"):
        build_trajectory_sets(
            [_row("c", "c0", "C", ["FULL", "IGNORE", "FULL"])], total_layers=28
        )
    duplicate = _row("w", "w0", "W", ["READ_ONLY", "FULL", "FULL"])
    with pytest.raises(ValueError, match="duplicate program"):
        build_trajectory_sets([duplicate, dict(duplicate)], total_layers=28)


def test_prefix_action_hash_is_stable_and_prefix_sensitive():
    first = prefix_action_hash("u", 7, 9, ["READ_ONLY", "FULL"])
    assert first == prefix_action_hash("u", 7, 9, ["READ_ONLY", "FULL"])
    assert first != prefix_action_hash("u", 7, 9, ["FULL", "READ_ONLY"])
    assert first != prefix_action_hash("u", 7, 10, ["READ_ONLY", "FULL"])


def test_trajectory_set_marginal_loss_matches_manual_logmeanexp_and_gradients():
    route_logps = torch.tensor(
        [math.log(0.8) + math.log(0.5), math.log(0.2) + math.log(0.9)],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss, responsibilities = trajectory_set_marginal_loss(route_logps, suffix_length=2)
    expected = -math.log((0.8 * 0.5 + 0.2 * 0.9) / 2.0) / 2.0
    assert float(loss) == pytest.approx(expected, abs=1e-12)
    assert responsibilities.tolist() == pytest.approx([0.4 / 0.58, 0.18 / 0.58])
    loss.backward()
    observed = route_logps.grad.detach().clone()

    brute = route_logps.detach().clone().requires_grad_(True)
    manual = -torch.log(torch.exp(brute).mean()) / 2.0
    manual.backward()
    assert torch.equal(observed, brute.grad)


def test_trajectory_set_marginal_loss_is_stable_for_long_low_probability_routes():
    route_logps = torch.tensor([-3000.0, -3001.0, -3100.0], requires_grad=True)
    loss, responsibilities = trajectory_set_marginal_loss(route_logps, suffix_length=27)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(route_logps.grad).all()
    assert float(responsibilities.sum()) == pytest.approx(1.0)


def test_compact_router_state_preserves_exact_router_logits():
    torch.manual_seed(4)
    router = SharedReadWriteRouter(hidden_size=8, router_size=8, num_heads=2, dropout=0.0)
    router.eval()
    text = torch.randn(1, 5, 8)
    visual = torch.randn(1, 4, 8)
    text_mask = torch.tensor([[True, True, True, False, False]])
    visual_mask = torch.tensor([[True, True, False, False]])

    expected = router(text, visual, text_mask=text_mask, visual_mask=visual_mask)
    compact = compact_router_state(text, visual, text_mask, visual_mask)
    observed = router(
        compact["text_states"],
        compact["visual_states"],
        text_mask=compact["text_mask"],
        visual_mask=compact["visual_mask"],
    )

    assert compact["text_states"].shape == (1, 1, 8)
    assert compact["visual_states"].shape == (1, 4, 8)
    assert torch.equal(compact["visual_mask"], visual_mask)
    assert torch.equal(expected, observed)


def test_validate_state_references_requires_every_expected_state_exactly_once():
    expected = {
        "p0": ["s0", "s1"],
        "p1": ["s0", "s2"],
    }
    rows = [
        {"state_id": "s0", "state_sha256": "a" * 64},
        {"state_id": "s1", "state_sha256": "b" * 64},
        {"state_id": "s2", "state_sha256": "c" * 64},
    ]
    assert validate_state_references(expected, rows) == {
        "programs": 2,
        "state_references": 4,
        "unique_states": 3,
    }
    with pytest.raises(ValueError, match="missing state"):
        validate_state_references(expected, rows[:-1])
    with pytest.raises(ValueError, match="duplicate state"):
        validate_state_references(expected, rows + [dict(rows[0])])


def test_compute_uid_trajectory_loss_uses_every_route_and_backpropagates_exactly():
    class TinyRouter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.7))

        def forward(self, text, visual, *, text_mask, visual_mask):
            value = text[:, 0, 0] * self.scale
            return torch.stack((value, -value, value * 0.5, value * 0.0), dim=-1)

    def state(value: float):
        return {
            "text_states": torch.tensor([[[value]]]),
            "visual_states": torch.tensor([[[1.0]]]),
            "text_mask": torch.ones(1, 1, dtype=torch.bool),
            "visual_mask": torch.ones(1, 1, dtype=torch.bool),
        }

    payload = {
        "states": {"s0": state(1.0), "s1": state(2.0), "s2": state(-1.0)},
        "programs": [
            {"program_id": "p0", "state_ids": ["s0", "s1"], "action_indices": [0, 1]},
            {"program_id": "p1", "state_ids": ["s0", "s2"], "action_indices": [2, 3]},
        ],
        "trigger_layer": 26,
    }
    router = TinyRouter()

    loss, stats = compute_uid_trajectory_loss(
        router, payload, device=torch.device("cpu"), state_microbatch=1
    )
    loss.backward()

    logits = {key: router(**{
        "text": value["text_states"],
        "visual": value["visual_states"],
        "text_mask": value["text_mask"],
        "visual_mask": value["visual_mask"],
    })[0].log_softmax(-1) for key, value in payload["states"].items()}
    route_logps = torch.stack((logits["s0"][0] + logits["s1"][1], logits["s0"][2] + logits["s2"][3]))
    expected, _ = trajectory_set_marginal_loss(route_logps, suffix_length=2)

    assert torch.allclose(loss.detach(), expected.detach(), atol=0, rtol=0)
    assert stats["route_count"] == 2
    assert stats["state_count"] == 3
    assert 0.0 <= stats["top_responsibility"] <= 1.0
    assert router.scale.grad is not None and torch.isfinite(router.scale.grad)


def test_fixed_rank_schedule_covers_each_uid_once_with_equal_rank_steps():
    uids = [f"u{index}" for index in range(23)]
    assignments = [
        _fixed_rank_uids(uids, rank=rank, world_size=4, seed=17, epoch=3)[0]
        for rank in range(4)
    ]
    assert len({len(values) for values in assignments}) == 1
    observed = [uid for values in assignments for uid in values if uid is not None]
    assert sorted(observed) == sorted(uids)
    for rank, values in enumerate(assignments):
        assert all(
            int(__import__("hashlib").sha256(uid.encode()).hexdigest(), 16) % 4 == rank
            for uid in values
            if uid is not None
        )


def test_real_phase74_corpus_matches_phase76_frozen_population_contract():
    config = load_config("configs/closed_loop_trajectory_set_v1.json")
    rows = read_jsonl(config["corpus"]["program_manifest"])
    _grouped, summary = build_trajectory_sets(rows, total_layers=28)
    assert summary == {
        "uids": 569,
        "programs": 4948,
        "dense_c_uids": 106,
        "dense_w_uids": 463,
        "route_state_occurrences": 69178,
        "unique_prefix_states": 35565,
    }
