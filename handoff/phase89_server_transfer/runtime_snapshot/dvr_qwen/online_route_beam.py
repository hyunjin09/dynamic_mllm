"""Trajectory-conditioned beam search for binary visual routes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable

import torch

from dvr_qwen.binary_generate import _compute_online_layer_route_logit
from dvr_qwen.binary_layer import (
    BinaryDVRCStaticInputCache,
    forward_text_only_layer,
    forward_visual_on_layer,
)
from dvr_qwen.modeling_dvr_qwen2_5_vl import qwen_num_hidden_layers, qwen_text_model
from dvr_qwen.router_features import summarize_text_states, summarize_visual_states
from dvr_qwen.split_scatter import BinaryDVRCIndexCache, BinaryDVRCInputs


@dataclass(frozen=True)
class BinaryBeamCandidate:
    route: tuple[int, ...]
    log_probability: float
    state: Any
    route_logits: tuple[float, ...] = ()


def _log_sigmoid(value: float) -> float:
    if value >= 0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def advance_binary_beam(
    beam: Iterable[BinaryBeamCandidate],
    *,
    beam_size: int,
    logit_fn: Callable[[BinaryBeamCandidate], float],
    transition_fn: Callable[[BinaryBeamCandidate, int], Any],
) -> list[BinaryBeamCandidate]:
    """Expand ON/OFF actions once and retain the highest probability prefixes."""

    if beam_size <= 0:
        raise ValueError("beam_size must be positive")
    expanded: list[BinaryBeamCandidate] = []
    for candidate in beam:
        logit = float(logit_fn(candidate))
        for action, action_log_probability in (
            (1, _log_sigmoid(logit)),
            (0, _log_sigmoid(-logit)),
        ):
            expanded.append(
                BinaryBeamCandidate(
                    route=(*candidate.route, action),
                    log_probability=candidate.log_probability + action_log_probability,
                    state=transition_fn(candidate, action),
                    route_logits=(*candidate.route_logits, logit),
                )
            )
    expanded.sort(key=lambda item: (-item.log_probability, item.route))
    return expanded[:beam_size]


@torch.inference_mode()
def online_visual_route_beam_search(
    model: Any,
    prepared_binary_inputs: BinaryDVRCInputs,
    visual_on_router: torch.nn.Module,
    *,
    beam_size: int = 5,
    scalar_features: torch.Tensor | None = None,
    visual_summary_mode: str = "none",
    text_summary_mode: str = "all_text",
) -> list[BinaryBeamCandidate]:
    """Search routes while replaying each route prefix's own hidden trajectory.

    This differs from top-k decoding of logits captured from one greedy route:
    every retained prefix executes its chosen action before the next router logit
    is computed.
    """

    if visual_summary_mode not in {"none", "mean_abs"}:
        raise ValueError("visual_summary_mode must be 'none' or 'mean_abs'")
    if text_summary_mode not in {"all_text", "instruction_only"}:
        raise ValueError("text_summary_mode must be 'all_text' or 'instruction_only'")
    if prepared_binary_inputs.text_states.shape[0] != 1:
        raise NotImplementedError("online route beam search supports batch size 1")

    text_model = qwen_text_model(model)
    num_layers = qwen_num_hidden_layers(model.config)
    if len(text_model.layers) != num_layers:
        raise ValueError("model layer count does not match the Qwen configuration")
    summary_valid_mask = prepared_binary_inputs.text_valid_mask
    if text_summary_mode == "instruction_only":
        if prepared_binary_inputs.instruction_valid_mask is None:
            raise ValueError("instruction_only summaries require instruction_token_mask")
        summary_valid_mask = prepared_binary_inputs.instruction_valid_mask

    static_input_cache = BinaryDVRCStaticInputCache(index_cache=BinaryDVRCIndexCache())
    beam = [
        BinaryBeamCandidate(
            route=(),
            log_probability=0.0,
            state=(prepared_binary_inputs.text_states, prepared_binary_inputs.visual_states),
        )
    ]

    for layer_idx, layer in enumerate(text_model.layers):
        cached_logits: dict[tuple[int, ...], float] = {}

        def route_logit(candidate: BinaryBeamCandidate) -> float:
            if candidate.route not in cached_logits:
                text_states, visual_states = candidate.state
                summary = summarize_text_states(
                    text_states.detach(), summary_valid_mask.to(text_states.device)
                )
                visual_summary = None
                if visual_summary_mode == "mean_abs":
                    visual_summary = summarize_visual_states(
                        visual_states.detach(),
                        prepared_binary_inputs.visual_valid_mask.to(visual_states.device),
                    )
                prev_gate = candidate.route[-1] if candidate.route else 0
                value = _compute_online_layer_route_logit(
                    visual_on_router,
                    summary,
                    layer_idx=layer_idx,
                    prev_gate=prev_gate,
                    scalar_features=scalar_features,
                    visual_summary=visual_summary,
                    text_states=text_states,
                    text_valid_mask=summary_valid_mask,
                )
                cached_logits[candidate.route] = float(value.detach().view(-1)[0].item())
            return cached_logits[candidate.route]

        def transition(candidate: BinaryBeamCandidate, action: int) -> tuple[torch.Tensor, torch.Tensor]:
            text_states, visual_states = candidate.state
            forward = forward_visual_on_layer if action else forward_text_only_layer
            next_text, next_visual, _ = forward(
                text_model,
                layer,
                text_states,
                visual_states,
                prepared_binary_inputs,
                layer_idx=layer_idx,
                cache=None,
                use_cache=False,
                static_input_cache=static_input_cache,
            )
            return next_text, next_visual

        beam = advance_binary_beam(
            beam,
            beam_size=beam_size,
            logit_fn=route_logit,
            transition_fn=transition,
        )

    return beam


def summarize_pass_at_k(rows: list[dict[str, Any]], beam_size: int) -> dict[str, Any]:
    """Summarize oracle candidate coverage separately from deployable top-1."""

    if beam_size <= 0:
        raise ValueError("beam_size must be positive")
    valid = [
        row
        for row in rows
        if row.get("evaluation_error") is None and len(row.get("beam_paths", [])) >= beam_size
    ]
    n = len(valid)
    if n == 0:
        return {
            "samples": 0,
            "all_on_accuracy": None,
            "top1_accuracy": None,
            "pass_at_k": {str(k): None for k in range(1, beam_size + 1)},
            "all_on_union_pass_at_k": {str(k): None for k in range(1, beam_size + 1)},
        }

    pass_at_k: dict[str, float] = {}
    all_on_union: dict[str, float] = {}
    for k in range(1, beam_size + 1):
        pass_at_k[str(k)] = sum(
            int(any(path["correct"] for path in row["beam_paths"][:k])) for row in valid
        ) / n
        all_on_union[str(k)] = sum(
            int(
                row["all_on"]["correct"]
                or any(path["correct"] for path in row["beam_paths"][:k])
            )
            for row in valid
        ) / n
    return {
        "samples": n,
        "all_on_accuracy": sum(int(row["all_on"]["correct"]) for row in valid) / n,
        "top1_accuracy": sum(int(row["beam_paths"][0]["correct"]) for row in valid) / n,
        "pass_at_k": pass_at_k,
        "all_on_union_pass_at_k": all_on_union,
    }
