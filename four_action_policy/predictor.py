"""Image+Question POLAR backbone with one categorical action head per layer."""

from __future__ import annotations

import torch
from torch import nn

from binary_policy.predictor import PolarLayerEncoder


class FourActionPolarBackbone(nn.Module):
    """Frozen-input token features to `[B, L, 4]` action logits."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.encoder = PolarLayerEncoder(**kwargs)
        self.num_layers = self.encoder.num_layers
        width = self.encoder.layer_embedding.embedding_dim
        self.route_head = nn.Linear(width, 4)

    def forward(
        self,
        token_features: torch.Tensor,
        token_attention_mask: torch.Tensor,
        image_features: torch.Tensor | None = None,
        image_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.encoder(
            token_features,
            token_attention_mask,
            image_features,
            image_attention_mask,
        )
        return self.route_head(encoded)
