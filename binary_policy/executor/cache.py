"""Per-layer K/V cache permitting route-dependent prompt lengths."""

from __future__ import annotations

import torch


class BinaryRouteCache:
    def __init__(self, num_layers: int):
        self.key_cache: list[torch.Tensor | None] = [None] * num_layers
        self.value_cache: list[torch.Tensor | None] = [None] * num_layers

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs, kwargs
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], key_states), dim=-2)
            self.value_cache[layer_idx] = torch.cat((self.value_cache[layer_idx], value_states), dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        value = self.key_cache[layer_idx]
        return 0 if value is None else int(value.shape[-2])

    def lengths(self) -> list[int]:
        return [self.get_seq_length(index) for index in range(len(self.key_cache))]
