"""Text-summary feature extraction for the Phase 5B binary router."""

from __future__ import annotations

from typing import Any

import torch

from dvr_qwen.binary_layer import (
    BinaryDVRCStaticInputCache,
    forward_text_only_layer,
    forward_visual_on_layer,
    normalize_visual_on_mask,
)
from dvr_qwen.modeling_dvr_qwen2_5_vl import (
    DVRQwen2_5_VLForConditionalGeneration,
    qwen_num_hidden_layers,
    qwen_text_model,
)
from dvr_qwen.router_data import NUM_LAYERS, previous_gate_tensor
from dvr_qwen.split_scatter import build_binary_dvrc_inputs


def masked_mean(states: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    mask = valid_mask.to(device=states.device, dtype=states.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (states * mask.unsqueeze(-1)).sum(dim=1) / denom


def last_valid_token(states: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    lengths = valid_mask.to(device=states.device).long().sum(dim=1).clamp_min(1)
    idx = lengths - 1
    batch_idx = torch.arange(states.shape[0], device=states.device)
    return states[batch_idx, idx]


def window_mean(states: torch.Tensor, valid_mask: torch.Tensor, num_windows: int = 8) -> torch.Tensor:
    if states.shape[1] == 0:
        return states.new_zeros(states.shape[0], states.shape[-1])
    windows = min(max(int(num_windows), 1), states.shape[1])
    pooled = []
    for idx in torch.arange(states.shape[1], device=states.device).chunk(windows):
        pooled.append(masked_mean(states[:, idx], valid_mask[:, idx]))
    return torch.stack(pooled, dim=1).mean(dim=1)


def summarize_text_states(
    text_states: torch.Tensor,
    text_valid_mask: torch.Tensor,
    *,
    num_windows: int = 8,
) -> dict[str, torch.Tensor]:
    return {
        "global_mean": masked_mean(text_states, text_valid_mask),
        "window_mean": window_mean(text_states, text_valid_mask, num_windows=num_windows),
        "last_token": last_valid_token(text_states, text_valid_mask),
    }


def summarize_visual_states(
    visual_states: torch.Tensor,
    visual_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return cheap visual summaries `[B, 2, D]`: mean(V_l), mean(abs(V_l))."""
    return torch.stack(
        [
            masked_mean(visual_states, visual_valid_mask),
            masked_mean(visual_states.abs(), visual_valid_mask),
        ],
        dim=1,
    )


@torch.inference_mode()
def collect_teacher_forced_router_features(
    model: DVRQwen2_5_VLForConditionalGeneration,
    inputs: dict[str, Any],
    visual_on_mask: torch.Tensor,
    *,
    num_windows: int = 8,
    visual_summary_mode: str = "none",
) -> dict[str, torch.Tensor]:
    """Collect layer-entry text summaries while executing the teacher route."""
    if visual_summary_mode not in {"none", "mean_abs"}:
        raise ValueError("visual_summary_mode must be 'none' or 'mean_abs'")
    binary_inputs = build_binary_dvrc_inputs(
        model.base_model,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        mm_token_type_ids=inputs["mm_token_type_ids"],
        pixel_values=inputs.get("pixel_values"),
        pixel_values_videos=inputs.get("pixel_values_videos"),
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        second_per_grid_ts=inputs.get("second_per_grid_ts"),
    )
    batch_size = binary_inputs.text_states.shape[0]
    if batch_size != 1:
        raise NotImplementedError("Phase 5B feature pilot supports batch size 1")

    text_model = qwen_text_model(model)
    num_layers = qwen_num_hidden_layers(model.config)
    if num_layers != NUM_LAYERS:
        raise ValueError(f"expected {NUM_LAYERS} layers, got {num_layers}")
    route = normalize_visual_on_mask(
        visual_on_mask,
        batch_size=batch_size,
        num_layers=num_layers,
        device=binary_inputs.text_states.device,
    )
    static_input_cache = BinaryDVRCStaticInputCache()
    text_states = binary_inputs.text_states
    visual_states = binary_inputs.visual_states

    summaries: dict[str, list[torch.Tensor]] = {
        "global_mean": [],
        "window_mean": [],
        "last_token": [],
    }
    visual_summaries: list[torch.Tensor] = []
    for layer_idx, layer in enumerate(text_model.layers):
        current = summarize_text_states(
            text_states.detach(),
            binary_inputs.text_valid_mask.to(text_states.device),
            num_windows=num_windows,
        )
        for key, value in current.items():
            summaries[key].append(value[0].detach().float().cpu())
        if visual_summary_mode == "mean_abs":
            visual_summary = summarize_visual_states(
                visual_states.detach(),
                binary_inputs.visual_valid_mask.to(visual_states.device),
            )
            visual_summaries.append(visual_summary[0].detach().float().cpu())

        if bool(route[0, layer_idx].item()):
            text_states, visual_states, _ = forward_visual_on_layer(
                text_model,
                layer,
                text_states,
                visual_states,
                binary_inputs,
                layer_idx=layer_idx,
                cache=None,
                use_cache=False,
                static_input_cache=static_input_cache,
            )
        else:
            text_states, visual_states, _ = forward_text_only_layer(
                text_model,
                layer,
                text_states,
                visual_states,
                binary_inputs,
                layer_idx=layer_idx,
                cache=None,
                use_cache=False,
                static_input_cache=static_input_cache,
            )

    route_cpu = route[0].detach().cpu()
    output = {
        "global_mean": torch.stack(summaries["global_mean"], dim=0),
        "window_mean": torch.stack(summaries["window_mean"], dim=0),
        "last_token": torch.stack(summaries["last_token"], dim=0),
        "labels": route_cpu.to(dtype=torch.float32),
        "prev_gates": previous_gate_tensor(route_cpu),
        "layer_idx": torch.arange(num_layers, dtype=torch.long),
        "num_text_tokens": binary_inputs.text_valid_mask.long().sum(dim=1).cpu(),
        "num_visual_tokens": binary_inputs.visual_valid_mask.long().sum(dim=1).cpu(),
        "full_prompt_tokens": binary_inputs.full_attention_mask.long().sum(dim=1).cpu(),
    }
    if visual_summary_mode == "mean_abs":
        output["visual_summaries"] = torch.stack(visual_summaries, dim=0)
    return output
