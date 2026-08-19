"""Binary VISUAL_ON router for cached Phase 5B text summaries."""

from __future__ import annotations

import torch
from torch import nn


class BinaryVisualOnRouter(nn.Module):
    """Predict one VISUAL_ON logit for each Qwen language layer.

    The module consumes the cached text/control summaries produced before each
    decoder layer. It does not read or mutate Qwen hidden states directly.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        hidden_dim: int = 256,
        use_prev_gate: bool = True,
        scalar_features_dim: int = 0,
        visual_summary_count: int = 0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if scalar_features_dim < 0:
            raise ValueError("scalar_features_dim must be non-negative")
        if visual_summary_count < 0:
            raise ValueError("visual_summary_count must be non-negative")

        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.use_prev_gate = bool(use_prev_gate)
        self.scalar_features_dim = int(scalar_features_dim)
        self.visual_summary_count = int(visual_summary_count)
        self.layer_emb = nn.Embedding(self.num_layers, self.d_model)
        if self.use_prev_gate:
            self.prev_gate_emb = nn.Embedding(2, self.d_model)
        if self.scalar_features_dim:
            self.scalar_proj = nn.Linear(self.scalar_features_dim, self.d_model)
        proj_input_dim = self.d_model * (4 + (1 if self.scalar_features_dim else 0) + self.visual_summary_count)
        self.proj = nn.Sequential(
            nn.Linear(proj_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _validate_feature_tensor(self, name: str, value: torch.Tensor) -> tuple[int, int, int]:
        if value.ndim != 3:
            raise ValueError(f"{name} must have shape [B, L, D], got {tuple(value.shape)}")
        batch_size, num_layers, d_model = value.shape
        if num_layers != self.num_layers:
            raise ValueError(f"{name} has {num_layers} layers, expected {self.num_layers}")
        if d_model != self.d_model:
            raise ValueError(f"{name} has d_model={d_model}, expected {self.d_model}")
        return int(batch_size), int(num_layers), int(d_model)

    def _normalize_layer_idx(self, layer_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
        if layer_idx.ndim == 1:
            if layer_idx.shape[0] != self.num_layers:
                raise ValueError(f"layer_idx must have {self.num_layers} entries, got {tuple(layer_idx.shape)}")
            layer_idx = layer_idx.unsqueeze(0).expand(batch_size, -1)
        if layer_idx.ndim != 2 or tuple(layer_idx.shape) != (batch_size, self.num_layers):
            raise ValueError(
                f"layer_idx must have shape [{batch_size}, {self.num_layers}], got {tuple(layer_idx.shape)}"
            )
        return layer_idx.to(dtype=torch.long)

    def _normalize_prev_gate(self, prev_gate: torch.Tensor, batch_size: int) -> torch.Tensor:
        if prev_gate.ndim != 2 or tuple(prev_gate.shape) != (batch_size, self.num_layers):
            raise ValueError(
                f"prev_gate must have shape [{batch_size}, {self.num_layers}], got {tuple(prev_gate.shape)}"
            )
        prev_gate = prev_gate.to(dtype=torch.long)
        if bool(((prev_gate < 0) | (prev_gate > 1)).any().item()):
            raise ValueError("prev_gate must contain only 0/1 values")
        return prev_gate

    def _normalize_scalar_features(
        self,
        scalar_features: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor | None:
        if self.scalar_features_dim == 0:
            if scalar_features is not None:
                raise ValueError("scalar_features were provided but scalar_features_dim is 0")
            return None
        if scalar_features is None:
            raise ValueError("scalar_features are required when scalar_features_dim is positive")
        if scalar_features.ndim == 2:
            if tuple(scalar_features.shape) != (batch_size, self.scalar_features_dim):
                raise ValueError(
                    "scalar_features must have shape "
                    f"[{batch_size}, {self.scalar_features_dim}], got {tuple(scalar_features.shape)}"
                )
            projected = self.scalar_proj(scalar_features.float())
            return projected.unsqueeze(1).expand(-1, self.num_layers, -1)
        if scalar_features.ndim == 3:
            if tuple(scalar_features.shape) != (batch_size, self.num_layers, self.scalar_features_dim):
                raise ValueError(
                    "scalar_features must have shape "
                    f"[{batch_size}, {self.num_layers}, {self.scalar_features_dim}], "
                    f"got {tuple(scalar_features.shape)}"
                )
            return self.scalar_proj(scalar_features.float())
        raise ValueError(f"scalar_features must have rank 2 or 3, got {scalar_features.ndim}")

    def _normalize_visual_summaries(
        self,
        visual_summaries: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor | None:
        if self.visual_summary_count == 0:
            if visual_summaries is not None:
                raise ValueError("visual_summaries were provided but visual_summary_count is 0")
            return None
        if visual_summaries is None:
            raise ValueError("visual_summaries are required when visual_summary_count is positive")
        if visual_summaries.ndim == 3:
            if self.visual_summary_count != 1:
                raise ValueError(
                    "rank-3 visual_summaries are only valid when visual_summary_count is 1; "
                    f"got {self.visual_summary_count}"
                )
            visual_summaries = visual_summaries.unsqueeze(2)
        expected = (batch_size, self.num_layers, self.visual_summary_count, self.d_model)
        if visual_summaries.ndim != 4 or tuple(visual_summaries.shape) != expected:
            raise ValueError(f"visual_summaries must have shape {expected}, got {tuple(visual_summaries.shape)}")
        return visual_summaries.float()

    def forward(
        self,
        global_mean: torch.Tensor,
        window_mean: torch.Tensor,
        last_token: torch.Tensor,
        layer_idx: torch.Tensor,
        prev_gate: torch.Tensor,
        scalar_features: torch.Tensor | None = None,
        visual_summaries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, _, _ = self._validate_feature_tensor("global_mean", global_mean)
        self._validate_feature_tensor("window_mean", window_mean)
        self._validate_feature_tensor("last_token", last_token)
        layer_idx = self._normalize_layer_idx(layer_idx.to(device=global_mean.device), batch_size)
        prev_gate = self._normalize_prev_gate(prev_gate.to(device=global_mean.device), batch_size)
        scalar = self._normalize_scalar_features(
            None if scalar_features is None else scalar_features.to(device=global_mean.device),
            batch_size,
        )
        visual = self._normalize_visual_summaries(
            None if visual_summaries is None else visual_summaries.to(device=global_mean.device),
            batch_size,
        )

        layer = self.layer_emb(layer_idx)
        if self.use_prev_gate:
            prev = self.prev_gate_emb(prev_gate)
        else:
            prev = torch.zeros_like(layer)
        parts = [
            global_mean + layer,
            window_mean + layer,
            last_token + layer,
            prev,
        ]
        if scalar is not None:
            parts.append(scalar)
        if visual is not None:
            for summary_idx in range(self.visual_summary_count):
                parts.append(visual[:, :, summary_idx, :] + layer)
        features = torch.cat(parts, dim=-1)
        return self.proj(features).squeeze(-1)
