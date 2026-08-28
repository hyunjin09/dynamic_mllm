"""Function-specific online router over current Qwen text and visual states."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class RouterFeatures:
    read_visual: torch.Tensor
    write_visual: torch.Tensor
    read_state: torch.Tensor
    write_state: torch.Tensor


class OnlineFourActionRouter(nn.Module):
    """Shared READ/WRITE branches with separate learned per-layer queries."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        d_router: int = 256,
        num_heads: int = 4,
        mlp_hidden_size: int = 512,
        dropout: float = 0.1,
        interaction_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_layers < 1 or d_router < 1 or mlp_hidden_size < 1:
            raise ValueError("router dimensions must be positive")
        if num_heads < 1 or d_router % num_heads:
            raise ValueError("d_router must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if interaction_scale < 0:
            raise ValueError("interaction_scale must be nonnegative")
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.d_router = int(d_router)
        self.num_heads = int(num_heads)
        self.interaction_scale = float(interaction_scale)

        self.read_layer_queries = nn.Embedding(num_layers, d_router)
        self.write_layer_queries = nn.Embedding(num_layers, d_router)
        self.read_query = nn.Linear(hidden_size, d_router)
        self.read_key = nn.Linear(hidden_size, d_router)
        self.read_value = nn.Linear(hidden_size, d_router)
        self.write_key = nn.Linear(hidden_size, d_router)
        self.write_value = nn.Linear(hidden_size, d_router)
        self.read_text = nn.Linear(hidden_size, d_router)
        self.write_text = nn.Linear(hidden_size, d_router)
        self.read_mlp = nn.Sequential(
            nn.Linear(3 * d_router, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, d_router),
        )
        self.write_mlp = nn.Sequential(
            nn.Linear(3 * d_router, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, d_router),
        )
        self.read_head = nn.Linear(d_router, 1)
        self.write_head = nn.Linear(d_router, 1)
        self.interaction_head = nn.Sequential(
            nn.Linear(2 * d_router, d_router),
            nn.GELU(),
            nn.Linear(d_router, 4),
        )

    def _attention(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        batch, visual_tokens, width = keys.shape
        if width != self.d_router or values.shape != keys.shape:
            raise ValueError("projected visual keys and values have the wrong shape")
        if query.shape != (batch, self.d_router):
            raise ValueError("router query has the wrong shape")
        if valid_mask.shape != (batch, visual_tokens) or valid_mask.dtype != torch.bool:
            raise ValueError("visual valid mask has the wrong shape or dtype")
        if not bool(valid_mask.any(dim=1).all().item()):
            raise ValueError("every router state requires at least one visual token")
        head_width = self.d_router // self.num_heads
        q = query.view(batch, self.num_heads, head_width)
        k = keys.view(batch, visual_tokens, self.num_heads, head_width).transpose(1, 2)
        v = values.view(batch, visual_tokens, self.num_heads, head_width).transpose(1, 2)
        scores = torch.einsum("bhd,bhvd->bhv", q, k) / math.sqrt(head_width)
        scores = scores.masked_fill(~valid_mask[:, None, :].to(scores.device), float("-inf"))
        weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        attended = torch.einsum("bhv,bhvd->bhd", weights, v)
        return attended.reshape(batch, self.d_router)

    def _layer_indices(self, layer_indices: int | torch.Tensor, batch: int, device) -> torch.Tensor:
        values = torch.as_tensor(layer_indices, dtype=torch.long, device=device).reshape(-1)
        if values.numel() == 1:
            values = values.expand(batch)
        if values.shape != (batch,) or bool(((values < 0) | (values >= self.num_layers)).any()):
            raise ValueError("layer indices must supply one valid index per state")
        return values

    def forward_features(
        self,
        text_state: torch.Tensor,
        visual_states: torch.Tensor,
        visual_valid_mask: torch.BoolTensor,
        layer_indices: int | torch.Tensor,
    ) -> RouterFeatures:
        if text_state.ndim != 2 or text_state.shape[-1] != self.hidden_size:
            raise ValueError("text_state must have shape [B, hidden_size]")
        if visual_states.ndim != 3 or visual_states.shape[0] != text_state.shape[0] or visual_states.shape[-1] != self.hidden_size:
            raise ValueError("visual_states must have shape [B, V, hidden_size]")
        layers = self._layer_indices(layer_indices, text_state.shape[0], text_state.device)
        read_layer = self.read_layer_queries(layers)
        write_layer = self.write_layer_queries(layers)
        read_visual = self._attention(
            self.read_query(text_state) + read_layer,
            self.read_key(visual_states),
            self.read_value(visual_states),
            visual_valid_mask,
        )
        write_visual = self._attention(
            write_layer,
            self.write_key(visual_states),
            self.write_value(visual_states),
            visual_valid_mask,
        )
        read_state = self.read_mlp(
            torch.cat([self.read_text(text_state), read_visual, read_layer], dim=-1)
        )
        write_state = self.write_mlp(
            torch.cat([write_visual, self.write_text(text_state), write_layer], dim=-1)
        )
        return RouterFeatures(read_visual, write_visual, read_state, write_state)

    @staticmethod
    def structured_logits(
        read_score: torch.Tensor,
        write_score: torch.Tensor,
        interaction_residual: torch.Tensor,
        *,
        interaction_scale: float,
    ) -> torch.Tensor:
        if read_score.shape != write_score.shape or interaction_residual.shape != (*read_score.shape, 4):
            raise ValueError("structured score shapes are inconsistent")
        base = torch.stack(
            [
                -read_score - write_score,
                read_score - write_score,
                -read_score + write_score,
                read_score + write_score,
            ],
            dim=-1,
        )
        return base + float(interaction_scale) * interaction_residual

    def forward(
        self,
        text_state: torch.Tensor,
        visual_states: torch.Tensor,
        visual_valid_mask: torch.BoolTensor,
        layer_indices: int | torch.Tensor,
    ) -> torch.Tensor:
        features = self.forward_features(
            text_state, visual_states, visual_valid_mask, layer_indices
        )
        read_score = self.read_head(features.read_state).squeeze(-1)
        write_score = self.write_head(features.write_state).squeeze(-1)
        interaction = self.interaction_head(
            torch.cat([features.read_state, features.write_state], dim=-1)
        )
        return self.structured_logits(
            read_score,
            write_score,
            interaction,
            interaction_scale=self.interaction_scale,
        )
