from __future__ import annotations

import copy

import pytest
import torch

from dense_failure_stage2.polar_suffix_program import (
    ACTION_NAMES,
    PolarSuffixProgramPredictor,
    assign_group_disjoint_dev,
    build_program_corpus,
    canonical_suffix,
    hierarchical_program_weights,
    initialize_from_stage2a,
    weighted_program_loss,
)
from dense_failure_stage2.v1_router import SharedReadWriteRouter


def _program_row(uid: str, group: str, outcome: str, route: int = 0) -> dict:
    return {
        "uid": uid,
        "image_group_id": group,
        "dataset": "gqa",
        "source_regime": "historical",
        "dense_outcome": outcome,
        "program_id": f"{uid}:{route}",
    }


def _model() -> PolarSuffixProgramPredictor:
    torch.manual_seed(3)
    return PolarSuffixProgramPredictor(
        hidden_size=12,
        router_size=8,
        num_heads=2,
        decoder_layers=2,
        feedforward_size=24,
        dropout=0.0,
        total_layers=6,
    ).eval()


def test_canonical_suffix_requires_full_prefix_and_complete_route():
    actions = ["FULL"] * 6
    actions[3] = "IGNORE"
    assert canonical_suffix(actions, trigger_layer=2, total_layers=6) == [
        "FULL",
        "IGNORE",
        "FULL",
        "FULL",
    ]
    actions[1] = "READ_ONLY"
    with pytest.raises(ValueError, match="pre-trigger"):
        canonical_suffix(actions, trigger_layer=2, total_layers=6)
    with pytest.raises(ValueError, match="exactly 6"):
        canonical_suffix(["FULL"] * 5, trigger_layer=2, total_layers=6)


def test_group_split_is_deterministic_and_has_zero_group_overlap():
    rows = []
    for index in range(30):
        group = f"group-{index // 2}"
        rows.append(_program_row(f"uid-{index}", group, "W" if index % 3 else "C"))
    first = assign_group_disjoint_dev(rows, dev_fraction=0.2, seed=74)
    second = assign_group_disjoint_dev(list(reversed(rows)), dev_fraction=0.2, seed=74)
    assert first == second
    train_groups = {row["image_group_id"] for row in rows if first[row["uid"]] == "train"}
    dev_groups = {row["image_group_id"] for row in rows if first[row["uid"]] == "dev"}
    assert not train_groups & dev_groups
    assert train_groups and dev_groups


def test_hierarchical_weights_equalize_uids_and_apply_one_over_k():
    rows = [
        _program_row("w-many", "g1", "W", route) for route in range(4)
    ] + [
        _program_row("w-one", "g2", "W"),
        _program_row("c-one", "g3", "C"),
    ]
    weights = hierarchical_program_weights(rows, minimum_cell_uids=1)
    uid_weight = {}
    for row, weight in zip(rows, weights):
        uid_weight[row["uid"]] = uid_weight.get(row["uid"], 0.0) + weight
    assert uid_weight["w-many"] == pytest.approx(uid_weight["w-one"])
    assert weights[0] == pytest.approx(weights[1])
    assert sum(weights) == pytest.approx(1.0)


def test_stage2a_initialization_copies_branches_and_context_exactly():
    torch.manual_seed(9)
    old = SharedReadWriteRouter(hidden_size=12, router_size=8, num_heads=2, dropout=0.0)
    new = _model()
    report = initialize_from_stage2a(new, old.state_dict())
    assert report["all_exact"]

    text = torch.randn(2, 4, 12)
    visual = torch.randn(2, 3, 12)
    text_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    visual_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    with torch.no_grad():
        old_logits_input = old.action_head[:3](
            torch.cat(new.compute_branches(text, visual, text_mask, visual_mask), dim=-1)
        )
        context = new.compute_context(text, visual, text_mask, visual_mask)
    assert torch.equal(context, old_logits_input)


def test_teacher_forcing_is_causal_and_has_expected_shape():
    model = _model()
    context = torch.randn(2, 8)
    trigger = torch.tensor([2, 2])
    first = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    changed_future = first.clone()
    changed_future[:, 2:] = torch.tensor([3, 0])
    with torch.no_grad():
        logits_a = model.decode_teacher_forced(context, trigger, first)
        logits_b = model.decode_teacher_forced(context, trigger, changed_future)
    assert logits_a.shape == (2, 4, len(ACTION_NAMES))
    assert torch.equal(logits_a[:, :3], logits_b[:, :3])


def test_weighted_loss_normalizes_each_program_by_suffix_length():
    logits = torch.zeros(2, 4, 4)
    targets = torch.tensor([[0, 1, -100, -100], [0, 1, 2, 3]])
    weights = torch.tensor([0.5, 0.5])
    loss, per_program = weighted_program_loss(logits, targets, weights)
    expected = torch.log(torch.tensor(4.0))
    assert torch.allclose(per_program, torch.tensor([expected, expected]))
    assert torch.allclose(loss, expected)


def test_beam_decode_returns_complete_sorted_programs():
    model = _model()
    context = torch.randn(1, 8)
    beams = model.beam_decode(context, trigger_layer=3, beam_width=3)
    assert len(beams) == 3
    assert all(len(row["action_indices"]) == 3 for row in beams)
    assert [row["score"] for row in beams] == sorted(
        (row["score"] for row in beams), reverse=True
    )
    assert model.greedy_decode(context, trigger_layer=3) == beams[0]["action_indices"] or len(
        model.greedy_decode(context, trigger_layer=3)
    ) == 3


def test_cached_trigger_round_trip_preserves_initial_outputs_exactly():
    model = _model()
    text = torch.randn(1, 5, 12, dtype=torch.bfloat16)
    visual = torch.randn(1, 4, 12, dtype=torch.bfloat16)
    text_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool)
    visual_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    cached = copy.deepcopy(
        {
            "text_states": text.cpu(),
            "visual_states": visual.cpu(),
            "text_mask": text_mask.cpu(),
            "visual_mask": visual_mask.cpu(),
        }
    )
    with torch.no_grad():
        live_context = model.compute_context(text, visual, text_mask, visual_mask)
        cached_context = model.compute_context(**cached)
        live_logits = model.decode_teacher_forced(
            live_context, torch.tensor([2]), torch.tensor([[0, 1, 2, 3]])
        )
        cached_logits = model.decode_teacher_forced(
            cached_context, torch.tensor([2]), torch.tensor([[0, 1, 2, 3]])
        )
    assert torch.equal(text, cached["text_states"])
    assert torch.equal(visual, cached["visual_states"])
    assert torch.equal(live_context, cached_context)
    assert torch.equal(live_logits, cached_logits)


def test_program_corpus_enforces_dense_c_and_imports_dense_w_discoveries():
    full = ["FULL"] * 6
    corrected = full.copy()
    corrected[4] = "IGNORE"
    work_rows = [
        {
            "uid": "c",
            "dataset": "gqa",
            "source_regime": "historical",
            "triggers": {"P90": 3},
            "sample": {"image_group_id": "gc"},
            "dense_output": {"current_dense_correct": True, "generated_token_ids": [1, 2]},
        },
        {
            "uid": "w",
            "dataset": "gqa",
            "source_regime": "historical",
            "triggers": {"P90": 3},
            "sample": {"image_group_id": "gw"},
            "dense_output": {"current_dense_correct": False, "generated_token_ids": [8, 2]},
        },
    ]
    base = [
        {
            "uid": "c",
            "route_id": "keep",
            "route_type": "preservation_full",
            "route_origin": "preservation_full",
            "actions": full,
            "trigger_layer": 3,
            "exact_replay_valid": True,
            "final_lmms_correct": True,
        },
        {
            "uid": "c",
            "route_id": "unsafe-c",
            "route_type": "single",
            "route_origin": "existing_single",
            "actions": corrected,
            "trigger_layer": 3,
            "exact_replay_valid": True,
            "final_lmms_correct": True,
        },
        {
            "uid": "w",
            "route_id": "single-w",
            "route_type": "single",
            "route_origin": "existing_single",
            "actions": corrected,
            "trigger_layer": 3,
            "exact_replay_valid": True,
            "final_lmms_correct": True,
        },
    ]
    route_store = [
        {"route_id": "keep", "uid": "c", "generated_token_ids": [1, 2]},
        {"route_id": "single-w", "uid": "w", "generated_token_ids": [3, 2]},
    ]
    audit = [
        {
            "kind": "new_discovery",
            "uid": "w",
            "route_key": "|".join(corrected),
            "correct": True,
            "exact_token_parity": True,
            "generated_token_ids": [3, 2],
            "search_stage": "mcts",
            "state_id": "s",
        },
        {
            "kind": "new_discovery",
            "uid": "c",
            "route_key": "|".join(corrected),
            "correct": True,
            "exact_token_parity": True,
            "generated_token_ids": [1, 2],
            "search_stage": "mcts",
            "state_id": "s2",
        },
    ]
    rows = build_program_corpus(
        base,
        audit,
        work_rows,
        route_store,
        total_layers=6,
    )
    assert len(rows) == 2
    by_uid = {row["uid"]: row for row in rows}
    assert by_uid["c"]["full_actions"] == full
    assert by_uid["c"]["provenance"] == ["preservation"]
    assert by_uid["w"]["provenance"] == ["completeness_audit", "single"]
    assert by_uid["w"]["expected_generated_token_ids"] == [3, 2]


def test_program_corpus_rejects_invalid_replay_evidence():
    route = {
        "uid": "w",
        "route_id": "bad",
        "route_type": "single",
        "route_origin": "existing_single",
        "actions": ["FULL"] * 6,
        "trigger_layer": 3,
        "exact_replay_valid": False,
        "final_lmms_correct": True,
    }
    work = [{
        "uid": "w", "dataset": "gqa", "source_regime": "historical",
        "triggers": {"P90": 3}, "sample": {"image_group_id": "g"},
        "dense_output": {"current_dense_correct": False, "generated_token_ids": [1]},
    }]
    with pytest.raises(ValueError, match="replay-valid"):
        build_program_corpus([route], [], work, [], total_layers=6)
