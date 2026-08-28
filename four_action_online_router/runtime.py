"""Exact routed-state replay and online action selection."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from binary_policy.executor.four_action import (
    FullBaseline,
    capture_online_four_action_route,
    four_action_layer,
)
from binary_policy.executor.inputs import BinaryInputs, resolve_decoder
from four_action_policy.actions import decode_action_indices
from .supervision import PrefixTrie


@dataclass(frozen=True)
class TeacherForcedTrajectory:
    text_queries: torch.Tensor
    visual_states: torch.Tensor
    visual_valid_mask: torch.BoolTensor
    layer_indices: torch.LongTensor
    valid_action_mask: torch.BoolTensor
    action_indices: torch.LongTensor


def select_last_text_state(
    text_states: torch.Tensor, text_valid_mask: torch.BoolTensor
) -> torch.Tensor:
    """Select the final valid compact text/control row for every sample."""

    if text_states.ndim != 3 or text_valid_mask.shape != text_states.shape[:2]:
        raise ValueError("text states and valid mask have incompatible shapes")
    if text_valid_mask.dtype != torch.bool:
        raise TypeError("text valid mask must be boolean")
    counts = text_valid_mask.long().sum(dim=1)
    if not bool((counts > 0).all().item()):
        raise ValueError("every sample requires at least one valid text/control row")
    return text_states[
        torch.arange(text_states.shape[0], device=text_states.device),
        (counts - 1).to(text_states.device),
    ]


def replay_teacher_forced_states(
    model,
    prepared_inputs: BinaryInputs,
    action_indices: Sequence[int] | torch.Tensor,
    trie: PrefixTrie,
) -> TeacherForcedTrajectory:
    """Replay one valid route and retain the exact pre-layer router inputs."""

    route = torch.as_tensor(action_indices, dtype=torch.long).reshape(-1)
    decoder = resolve_decoder(model)
    if route.numel() != len(decoder.layers) or trie.num_layers != len(decoder.layers):
        raise ValueError("teacher route, trie, and decoder layer counts differ")
    route_tuple = tuple(int(value) for value in route.tolist())
    valid_actions = trie.valid_action_masks_for_route(route_tuple)
    actions = decode_action_indices(route)
    text_states = prepared_inputs.text_states
    visual_states = prepared_inputs.visual_states
    text_queries = []
    visual_trajectory = []
    with torch.no_grad():
        for layer_index, (layer, action) in enumerate(zip(decoder.layers, actions)):
            text_queries.append(
                select_last_text_state(
                    text_states, prepared_inputs.text_valid_mask.to(text_states.device)
                ).detach()
            )
            visual_trajectory.append(visual_states.detach())
            text_states, visual_states, _ = four_action_layer(
                model,
                layer,
                text_states,
                visual_states,
                prepared_inputs,
                action=action,
                layer_index=layer_index,
                cache=None,
                use_cache=False,
                native_causal=False,
            )
    stacked_visual = torch.cat(visual_trajectory, dim=0)
    visual_mask = prepared_inputs.visual_valid_mask.to(stacked_visual.device).expand(
        len(decoder.layers), -1
    )
    return TeacherForcedTrajectory(
        text_queries=torch.cat(text_queries, dim=0),
        visual_states=stacked_visual,
        visual_valid_mask=visual_mask,
        layer_indices=torch.arange(len(decoder.layers), device=stacked_visual.device),
        valid_action_mask=valid_actions.to(stacked_visual.device),
        action_indices=route.to(stacked_visual.device),
    )


def router_logits_for_trajectory(router, trajectory: TeacherForcedTrajectory) -> torch.Tensor:
    return router(
        trajectory.text_queries,
        trajectory.visual_states,
        trajectory.visual_valid_mask,
        trajectory.layer_indices,
    )


def _autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is not None and device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


@torch.inference_mode()
def capture_online_router_route(
    model,
    inputs: Mapping[str, Any],
    router,
    *,
    prepared_inputs: BinaryInputs | None = None,
    amp_dtype: torch.dtype | None = torch.bfloat16,
    use_cache: bool = True,
) -> FullBaseline:
    """Execute argmax router actions from each exact routed-prefix state."""

    router_device = next(router.parameters()).device

    def choose(layer_index, text_states, visual_states, meta):
        query = select_last_text_state(
            text_states, meta.text_valid_mask.to(text_states.device)
        ).to(router_device)
        visual = visual_states.to(router_device)
        mask = meta.visual_valid_mask.to(router_device)
        with _autocast(router_device, amp_dtype):
            logits = router(query, visual, mask, layer_index)
        return decode_action_indices(logits.argmax(dim=-1))[0]

    return capture_online_four_action_route(
        model,
        inputs,
        choose,
        prepared_inputs=prepared_inputs,
        use_cache=use_cache,
    )
