"""DVR-Qwen2.5-VL static-route execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration

from dvr_qwen.attention import forward_text_only, forward_visual_read
from dvr_qwen.input_builder import DVRInputs, build_dvr_inputs
from dvr_qwen.masks import make_text_causal_mask, make_visual_read_mask


def qwen_text_model(qwen_or_dvr_model):
    qwen_model = qwen_or_dvr_model.model if hasattr(qwen_or_dvr_model, "model") else qwen_or_dvr_model
    return qwen_model.language_model if hasattr(qwen_model, "language_model") else qwen_model


def qwen_num_hidden_layers(config) -> int:
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, "num_hidden_layers"):
        return int(text_config.num_hidden_layers)
    if hasattr(config, "num_hidden_layers"):
        return int(config.num_hidden_layers)
    resolved = config.get_text_config() if hasattr(config, "get_text_config") else None
    if resolved is not None and hasattr(resolved, "num_hidden_layers"):
        return int(resolved.num_hidden_layers)
    raise AttributeError(f"{type(config).__name__} does not expose num_hidden_layers")


@dataclass
class DVRLayerStats:
    layer_idx: int
    visual_read_on: bool
    key_value_length: int
    residual_sequence_length: int
    visual_kv_projected: bool


@dataclass
class DVRStaticOutput:
    logits: torch.Tensor
    last_hidden_state: torch.Tensor
    route_mask: torch.Tensor
    dvr_inputs: DVRInputs
    layer_stats: list[DVRLayerStats]


class DVRQwen2_5_VLForConditionalGeneration(nn.Module):
    """Local static-route DVR wrapper around an unmodified Qwen2.5-VL checkpoint."""

    def __init__(self, base_model: Qwen2_5_VLForConditionalGeneration):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(*args, **kwargs)
        return cls(base_model)

    @property
    def model(self):
        return self.base_model.model

    @property
    def lm_head(self):
        return self.base_model.lm_head

    def _normalize_static_route(
        self,
        visual_route_mask: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        num_layers = qwen_num_hidden_layers(self.config)
        route = visual_route_mask.to(device=device, dtype=torch.bool)
        if route.ndim == 1:
            route = route.unsqueeze(0)
        if route.shape != (batch_size, num_layers):
            raise ValueError(f"visual_route_mask must have shape {(batch_size, num_layers)}, got {tuple(route.shape)}")
        return route

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        mm_token_type_ids: torch.IntTensor,
        visual_route_mask: torch.Tensor,
        route_mode: str = "static_route",
        return_route_logits: bool = False,
        visual_attention_bias: float = 0.0,
        visual_value_scale: float = 1.0,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> DVRStaticOutput:
        if route_mode != "static_route":
            raise ValueError("Phase 2 supports only route_mode='static_route'")
        if return_route_logits:
            raise ValueError("Phase 2 static-route execution does not produce router logits")

        dvr_inputs = build_dvr_inputs(
            self.base_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )

        batch_size = dvr_inputs.text_hidden.shape[0]
        if batch_size != 1:
            raise NotImplementedError("Phase 2 static DVR execution is validated for batch size 1 only")

        hidden_states = dvr_inputs.text_hidden
        route = self._normalize_static_route(visual_route_mask, batch_size, hidden_states.device)
        text_model = qwen_text_model(self)
        layer_stats: list[DVRLayerStats] = []

        for layer_idx, layer in enumerate(text_model.layers):
            layer_device = next(layer.parameters()).device
            hidden_states = hidden_states.to(layer_device)
            text_position_ids = dvr_inputs.text_position_ids.to(layer_device)
            visual_position_ids = dvr_inputs.visual_position_ids.to(layer_device)
            text_valid_mask = dvr_inputs.text_valid_mask.to(layer_device)
            visual_valid_mask = dvr_inputs.visual_valid_mask.to(layer_device)
            text_orig_idx = dvr_inputs.text_orig_idx.to(layer_device)
            visual_orig_idx = dvr_inputs.visual_orig_idx.to(layer_device)

            residual = hidden_states
            text_norm = layer.input_layernorm(hidden_states)
            text_position_embeddings = text_model.rotary_emb(text_norm, text_position_ids)

            gate = bool(route[0, layer_idx].item())
            if gate:
                visual_memory = dvr_inputs.visual_memory.to(layer_device)
                visual_norm = layer.input_layernorm(visual_memory)
                visual_position_embeddings = text_model.rotary_emb(visual_norm, visual_position_ids)
                attn_mask = make_visual_read_mask(
                    text_orig_idx,
                    visual_orig_idx,
                    text_valid_mask,
                    visual_valid_mask,
                    dtype=text_norm.dtype,
                    device=layer_device,
                )
                attn_out, attn_stats = forward_visual_read(
                    layer.self_attn,
                    text_norm,
                    visual_norm,
                    attn_mask,
                    text_position_embeddings,
                    visual_position_embeddings,
                    visual_attention_bias=visual_attention_bias,
                    visual_value_scale=visual_value_scale,
                )
            else:
                attn_mask = make_text_causal_mask(
                    text_orig_idx,
                    text_valid_mask,
                    dtype=text_norm.dtype,
                    device=layer_device,
                )
                attn_out, attn_stats = forward_text_only(
                    layer.self_attn,
                    text_norm,
                    attn_mask,
                    text_position_embeddings,
                )

            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            layer_stats.append(
                DVRLayerStats(
                    layer_idx=layer_idx,
                    visual_read_on=gate,
                    key_value_length=attn_stats.key_value_length,
                    residual_sequence_length=hidden_states.shape[1],
                    visual_kv_projected=attn_stats.visual_kv_projected,
                )
            )

        norm_device = next(text_model.norm.parameters()).device
        hidden_states = text_model.norm(hidden_states.to(norm_device))
        lm_device = next(self.lm_head.parameters()).device
        logits = self.lm_head(hidden_states.to(lm_device))

        return DVRStaticOutput(
            logits=logits,
            last_hidden_state=hidden_states,
            route_mask=route,
            dvr_inputs=dvr_inputs,
            layer_stats=layer_stats,
        )
