from __future__ import annotations

import pytest
import torch
from torch import nn

from binary_policy.executor.cache import BinaryRouteCache
from binary_policy.executor.four_action import (
    FOUR_ACTIONS,
    capture_four_action_route,
    capture_full_baseline,
    capture_online_four_action_route,
    capture_route_baseline,
    four_action_layer,
    full_baseline_post_layer_text_states,
    greedy_generate_from_local_forward,
    local_four_action_forward,
    layerwise_token_scores_from_cached_prompt,
    normalize_four_action,
    route_conditioned_four_action_forward,
    score_token_ids_from_local_forward,
    score_token_ids_from_cached_prompt,
    unified_target_four_action_layer,
)
from binary_policy.executor.inputs import BinaryInputs


class FakeRotary(nn.Module):
    def forward(self, hidden, position_ids):
        shape = (position_ids.shape[1], position_ids.shape[2], hidden.shape[-1])
        return torch.ones(shape, device=hidden.device), torch.zeros(shape, device=hidden.device)


class FakeLayer(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls: list[int] = []

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
    ):
        del attention_mask, position_embeddings, cache_position
        rows = hidden_states.shape[1]
        self.calls.append(rows)
        if past_key_values is not None and use_cache:
            batch = hidden_states.shape[0]
            kv = hidden_states.new_zeros(batch, 1, rows, 1)
            past_key_values.update(kv, kv, self.layer_idx)
        # Row-count dependence makes compact and full text execution distinct.
        return (hidden_states + float((self.layer_idx + 1) * rows),)


class FakeNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, hidden_states):
        return hidden_states


class FakeDecoder(nn.Module):
    def __init__(self, layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(FakeLayer(index) for index in range(layers))
        self.rotary_emb = FakeRotary()
        self.norm = FakeNorm()
        self.embed_tokens = nn.Embedding(32, 3)

    def get_input_embeddings(self):
        return self.embed_tokens


class FakeModel(nn.Module):
    def __init__(self, layers: int = 3):
        super().__init__()
        self.model = FakeDecoder(layers)
        self.lm_head = nn.Linear(3, 7, bias=False)


def fixture_inputs() -> BinaryInputs:
    full = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    text_indices = torch.tensor([[0, 2]])
    visual_indices = torch.tensor([[1, 3]])
    positions = torch.arange(4).view(1, 1, 4).expand(3, 1, 4)
    return BinaryInputs(
        full_inputs_embeds=full,
        text_states=full[:, [0, 2]],
        visual_states=full[:, [1, 3]],
        text_indices=text_indices,
        visual_indices=visual_indices,
        text_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        visual_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        full_position_ids=positions,
        text_position_ids=positions[:, :, [0, 2]],
        visual_position_ids=positions[:, :, [1, 3]],
        full_attention_mask=torch.ones(1, 4, dtype=torch.long),
        full_prompt_len=torch.tensor([4]),
        rope_deltas=torch.tensor([[0]]),
    )


def test_four_action_names_are_exact_and_normalized():
    assert FOUR_ACTIONS == ("IGNORE", "READ_ONLY", "WRITE_ONLY", "FULL")
    assert normalize_four_action("read_only") == "READ_ONLY"
    with pytest.raises(ValueError):
        normalize_four_action("VISUAL_OFF")


@pytest.mark.parametrize(
    ("action", "expected_text_delta", "expected_visual_delta", "expected_cache_rows", "calls"),
    [
        ("IGNORE", 2.0, 0.0, 2, [2]),
        ("READ_ONLY", 4.0, 0.0, 4, [4]),
        ("WRITE_ONLY", 2.0, 4.0, 2, [4, 2]),
        ("FULL", 4.0, 4.0, 4, [4]),
    ],
)
def test_four_action_layer_has_exact_read_write_and_cache_semantics(
    action, expected_text_delta, expected_visual_delta, expected_cache_rows, calls
):
    model = FakeModel(layers=1)
    layer = model.model.layers[0]
    meta = fixture_inputs()
    cache = BinaryRouteCache(1)
    text, visual, execution = four_action_layer(
        model,
        layer,
        meta.text_states,
        meta.visual_states,
        meta,
        action=action,
        layer_index=0,
        cache=cache,
        use_cache=True,
        native_causal=True,
    )
    assert torch.equal(text, meta.text_states + expected_text_delta)
    assert torch.equal(visual, meta.visual_states + expected_visual_delta)
    assert cache.get_seq_length(0) == expected_cache_rows
    assert layer.calls == calls
    assert execution.read_on is (action in {"READ_ONLY", "FULL"})
    assert execution.write_on is (action in {"WRITE_ONLY", "FULL"})


def test_ignore_exactly_reproduces_binary_off_outputs():
    from binary_policy.executor.layers import visual_off_layer

    meta = fixture_inputs()
    model = FakeModel(layers=1)
    layer = model.model.layers[0]
    expected_text, expected_visual, _ = visual_off_layer(
        model, layer, meta.text_states, meta.visual_states, meta, layer_index=0
    )
    actual_text, actual_visual, _ = four_action_layer(
        model,
        layer,
        meta.text_states,
        meta.visual_states,
        meta,
        action="IGNORE",
        layer_index=0,
    )
    assert torch.equal(actual_text, expected_text)
    assert torch.equal(actual_visual, expected_visual)


@pytest.mark.parametrize("action", FOUR_ACTIONS)
def test_unified_target_runs_identical_full_then_compact_machinery(action):
    meta = fixture_inputs()
    model = FakeModel(layers=1)
    layer = model.model.layers[0]
    cache = BinaryRouteCache(1)
    text, visual, selected_cache, execution = unified_target_four_action_layer(
        model,
        layer,
        meta.text_states,
        meta.visual_states,
        meta,
        action=action,
        layer_index=0,
        prefix_cache=cache,
    )
    assert layer.calls == [4, 2]
    assert execution.decoder_calls == 2
    assert selected_cache is not None
    assert selected_cache.get_seq_length(0) == (4 if action in {"READ_ONLY", "FULL"} else 2)
    assert torch.equal(
        text,
        meta.text_states + (4.0 if action in {"READ_ONLY", "FULL"} else 2.0),
    )
    assert torch.equal(
        visual,
        meta.visual_states + (4.0 if action in {"WRITE_ONLY", "FULL"} else 0.0),
    )


def test_local_ignore_exactly_reproduces_binary_single_layer_off_route():
    from binary_policy.executor.generation import _last_text_logits, binary_prefill, binary_route_forward

    model = FakeModel(layers=3)
    meta = fixture_inputs()
    baseline = capture_full_baseline(model, {}, prepared_inputs=meta, use_cache=True)
    local = local_four_action_forward(model, baseline, 1, "IGNORE")
    binary = binary_route_forward(model, {}, [1, 0, 1], prepared_inputs=meta)
    binary_cached = binary_prefill(model, {}, [1, 0, 1], prepared_inputs=meta, use_cache=True)
    assert torch.equal(local.full_hidden_state, binary.full_hidden_state)
    assert torch.equal(local.prompt_logits, _last_text_logits(model, binary_cached))


def test_local_branches_share_one_captured_pre_layer_state_and_restore_full_suffix():
    model = FakeModel(layers=3)
    meta = fixture_inputs()
    baseline = capture_full_baseline(model, {}, prepared_inputs=meta, use_cache=True)
    target = 1

    full = local_four_action_forward(model, baseline, target, "FULL")
    ignore = local_four_action_forward(model, baseline, target, "IGNORE")
    read_only = local_four_action_forward(model, baseline, target, "READ_ONLY")
    write_only = local_four_action_forward(model, baseline, target, "WRITE_ONLY")

    expected_pre_text, expected_pre_visual = baseline.pre_layer_states[target]
    for result in (full, ignore, read_only, write_only):
        assert torch.equal(result.prefill.target_pre_text_state, expected_pre_text)
        assert torch.equal(result.prefill.target_pre_visual_state, expected_pre_visual)
        assert [row.action for row in result.prefill.layer_stats] == ["FULL", result.prefill.action, "FULL"]

    assert torch.equal(full.full_hidden_state, baseline.full_hidden_state)
    assert torch.equal(full.prompt_logits, baseline.prompt_logits)
    assert full.prefill.cache.lengths() == [4, 4, 4]
    assert ignore.prefill.cache.lengths() == [4, 2, 4]
    assert read_only.prefill.cache.lengths() == [4, 4, 4]
    assert write_only.prefill.cache.lengths() == [4, 2, 4]


def test_baseline_cache_is_not_mutated_by_local_branches():
    model = FakeModel(layers=3)
    baseline = capture_full_baseline(model, {}, prepared_inputs=fixture_inputs(), use_cache=True)
    before = baseline.cache.lengths()
    local_four_action_forward(model, baseline, 1, "WRITE_ONLY")
    assert baseline.cache.lengths() == before == [4, 4, 4]


def test_online_action_selector_uses_routed_prefix_and_matches_fixed_route():
    meta = fixture_inputs()
    actions = ("FULL", "IGNORE", "READ_ONLY")
    observed = []

    def selector(layer_index, text_states, visual_states, inputs):
        observed.append(
            (
                layer_index,
                text_states.detach().clone(),
                visual_states.detach().clone(),
                inputs,
            )
        )
        return actions[layer_index]

    online_model = FakeModel(layers=3)
    online = capture_online_four_action_route(
        online_model, {}, selector, prepared_inputs=meta, use_cache=True
    )
    fixed_model = FakeModel(layers=3)
    fixed_model.load_state_dict(online_model.state_dict())
    fixed = capture_four_action_route(
        fixed_model, {}, actions, prepared_inputs=meta, use_cache=True
    )

    assert online.layer_actions == actions
    assert torch.equal(online.full_hidden_state, fixed.full_hidden_state)
    assert torch.equal(online.prompt_logits, fixed.prompt_logits)
    assert online.cache is not None and fixed.cache is not None
    assert online.cache.lengths() == fixed.cache.lengths()
    assert len(observed) == 3
    for index, (_, text_states, visual_states, inputs) in enumerate(observed):
        assert torch.equal(text_states, online.pre_layer_states[index][0])
        assert torch.equal(visual_states, online.pre_layer_states[index][1])
        assert inputs is meta


def test_local_hybrid_target_states_match_their_exact_factorial_components():
    model = FakeModel(layers=3)
    baseline = capture_full_baseline(model, {}, prepared_inputs=fixture_inputs(), use_cache=True)
    full = local_four_action_forward(model, baseline, 1, "FULL").prefill
    ignore = local_four_action_forward(model, baseline, 1, "IGNORE").prefill
    read_only = local_four_action_forward(model, baseline, 1, "READ_ONLY").prefill
    write_only = local_four_action_forward(model, baseline, 1, "WRITE_ONLY").prefill
    assert torch.equal(read_only.target_post_text_state, full.target_post_text_state)
    assert torch.equal(read_only.target_post_visual_state, read_only.target_pre_visual_state)
    assert torch.equal(write_only.target_post_text_state, ignore.target_post_text_state)
    assert torch.equal(write_only.target_post_visual_state, full.target_post_visual_state)


def test_route_conditioned_branches_preserve_the_anchor_schedule_outside_target():
    model = FakeModel(layers=4)
    meta = fixture_inputs()
    route = [1, 0, 0, 1]
    baseline = capture_route_baseline(
        model,
        {},
        route,
        prepared_inputs=meta,
        use_cache=True,
    )

    outputs = {
        action: route_conditioned_four_action_forward(model, baseline, 1, action)
        for action in FOUR_ACTIONS
    }
    expected_pre_text, expected_pre_visual = baseline.pre_layer_states[1]
    for action, output in outputs.items():
        assert torch.equal(output.prefill.target_pre_text_state, expected_pre_text)
        assert torch.equal(output.prefill.target_pre_visual_state, expected_pre_visual)
        assert [row.action for row in output.prefill.layer_stats] == [
            "FULL",
            action,
            "IGNORE",
            "FULL",
        ]

    assert torch.equal(outputs["IGNORE"].full_hidden_state, baseline.full_hidden_state)
    assert torch.equal(outputs["IGNORE"].prompt_logits, baseline.prompt_logits)
    assert outputs["IGNORE"].prefill.cache.lengths() == [4, 2, 2, 4]
    assert outputs["READ_ONLY"].prefill.cache.lengths() == [4, 4, 2, 4]
    assert outputs["WRITE_ONLY"].prefill.cache.lengths() == [4, 2, 2, 4]
    assert outputs["FULL"].prefill.cache.lengths() == [4, 4, 2, 4]


def test_route_baseline_exactly_reproduces_mixed_binary_route_execution():
    from binary_policy.executor.generation import _last_text_logits, binary_prefill, binary_route_forward

    model = FakeModel(layers=4)
    meta = fixture_inputs()
    route = [1, 0, 0, 1]
    baseline = capture_route_baseline(
        model,
        {},
        route,
        prepared_inputs=meta,
        use_cache=True,
    )
    binary = binary_route_forward(model, {}, route, prepared_inputs=meta)
    binary_cached = binary_prefill(model, {}, route, prepared_inputs=meta, use_cache=True)

    assert torch.equal(baseline.full_hidden_state, binary.full_hidden_state)
    assert torch.equal(baseline.prompt_logits, _last_text_logits(model, binary_cached))
    assert [row.action for row in baseline.layer_stats] == ["FULL", "IGNORE", "IGNORE", "FULL"]
    assert baseline.cache.lengths() == [4, 2, 2, 4]


def test_complete_four_action_route_executes_every_layer_and_keeps_route_cache():
    model = FakeModel(layers=3)
    meta = fixture_inputs()
    route = ["FULL", "READ_ONLY", "WRITE_ONLY"]

    output = capture_four_action_route(
        model,
        {},
        route,
        prepared_inputs=meta,
        use_cache=True,
    )

    assert output.layer_actions == tuple(route)
    assert [row.action for row in output.layer_stats] == route
    assert output.cache is not None
    assert output.cache.lengths() == [4, 4, 2]
    assert [layer.calls for layer in model.model.layers] == [[4], [4], [4, 2]]
    assert torch.equal(output.text_hidden_state, meta.text_states + 18.0)
    assert torch.equal(output.visual_hidden_state, meta.visual_states + 16.0)


def test_route_conditioned_target_must_be_off_in_anchor_route():
    model = FakeModel(layers=4)
    baseline = capture_route_baseline(
        model,
        {},
        [1, 0, 0, 1],
        prepared_inputs=fixture_inputs(),
        use_cache=True,
    )
    with pytest.raises(ValueError, match="OFF in the anchor route"):
        route_conditioned_four_action_forward(model, baseline, 0, "IGNORE")


def test_generation_and_teacher_forcing_clone_the_local_prompt_cache():
    model = FakeModel(layers=3)
    baseline = capture_full_baseline(model, {}, prepared_inputs=fixture_inputs(), use_cache=True)
    output = local_four_action_forward(model, baseline, 1, "WRITE_ONLY")
    before = output.prefill.cache.lengths()

    generation = greedy_generate_from_local_forward(
        model,
        output,
        torch.tensor([[1, 2, 3, 4]]),
        max_new_tokens=2,
        eos_token_ids=[],
        repetition_penalty=1.0,
    )
    score = score_token_ids_from_local_forward(model, output, torch.tensor([1, 2]))

    assert generation.generated_ids.shape == (1, 2)
    assert generation.decode_cache.lengths() == [6, 4, 6]
    assert output.prefill.cache.lengths() == before == [4, 2, 4]
    assert score.token_ids == [1, 2]
    assert len(score.token_logprobs) == 2
    assert torch.isfinite(torch.tensor(score.mean_logprob))


def test_layerwise_logit_lens_final_score_matches_unified_baseline_score():
    model = FakeModel(layers=3)
    baseline = capture_full_baseline(
        model, {}, prepared_inputs=fixture_inputs(), use_cache=True, native_causal=False
    )
    assert baseline.cache is not None
    target = torch.tensor([1, 2])
    trajectory = layerwise_token_scores_from_cached_prompt(
        model,
        full_baseline_post_layer_text_states(baseline),
        baseline.inputs,
        baseline.cache,
        target,
    )
    final = score_token_ids_from_cached_prompt(
        model,
        baseline.prompt_logits,
        baseline.inputs,
        baseline.cache,
        target,
    )
    assert len(trajectory) == 3
    assert trajectory[-1] == pytest.approx(final.mean_logprob, abs=1e-6)
    assert baseline.cache.lengths() == [4, 4, 4]
