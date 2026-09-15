"""Per-layer cache for DVR-Qwen generation."""

from __future__ import annotations

import torch


class DVRCache:
    """Tiny K/V cache that permits different sequence lengths per decoder layer."""

    def __init__(self, num_layers: int):
        self.key_cache: list[torch.Tensor | None] = [None] * num_layers
        self.value_cache: list[torch.Tensor | None] = [None] * num_layers
        self.has_visual: list[bool] = [False] * num_layers
        self.num_visual_tokens: list[int] = [0] * num_layers

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        has_visual: bool = False,
        num_visual_tokens: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(has_visual, dict):
            has_visual = False
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            self.has_visual[layer_idx] = has_visual
            self.num_visual_tokens[layer_idx] = num_visual_tokens
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int) -> int:
        key_states = self.key_cache[layer_idx]
        return 0 if key_states is None else int(key_states.shape[-2])

    def lengths(self) -> list[int]:
        return [self.get_seq_length(layer_idx) for layer_idx in range(len(self.key_cache))]
