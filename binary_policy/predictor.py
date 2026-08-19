"""Lightweight POLAR-style binary route predictor.

The backbone consumes token embeddings rather than instantiating a foundation
model.  A caller may obtain those embeddings from the frozen POLAR text encoder
or a prospectively approved pre-action multimodal encoder.  This keeps model
loading out of unit tests and prevents accidental MLLM fine-tuning.
"""

from __future__ import annotations

import torch
from torch import nn


class PolarLayerEncoder(nn.Module):
    """POLAR's token-to-layer cross-attention and cross-layer context stack."""

    def __init__(
        self,
        *,
        num_layers: int = 28,
        input_dim: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layer_blocks: int = 2,
        dropout: float = 0.1,
        image_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 1 or input_dim < 1 or d_model < 1:
            raise ValueError("num_layers, input_dim, and d_model must be positive")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_layers = int(num_layers)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.image_projection = nn.Linear(image_dim, d_model) if image_dim is not None else None
        self.layer_embedding = nn.Embedding(num_layers, d_model)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer_blocks)

    def forward(
        self,
        token_features: torch.Tensor,
        token_attention_mask: torch.Tensor,
        image_features: torch.Tensor | None = None,
        image_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_features.ndim != 3:
            raise ValueError("token_features must have shape [B, T, D]")
        if token_attention_mask.shape != token_features.shape[:2]:
            raise ValueError("token_attention_mask must have shape [B, T]")
        tokens = self.input_projection(token_features)
        if self.image_projection is not None:
            if image_features is None:
                if image_attention_mask is not None:
                    raise ValueError("image_attention_mask requires image_features")
            else:
                if image_features.ndim == 2:
                    image_features = image_features.unsqueeze(1)
                if image_features.ndim != 3 or image_features.shape[0] != tokens.shape[0]:
                    raise ValueError("image_features must have shape [B,D] or [B,V,D]")
                if image_attention_mask is None:
                    image_attention_mask = torch.ones(
                        image_features.shape[:2], dtype=torch.bool, device=image_features.device
                    )
                if image_attention_mask.shape != image_features.shape[:2]:
                    raise ValueError("image_attention_mask must have shape [B,V]")
                image_tokens = self.image_projection(image_features)
                tokens = torch.cat([tokens, image_tokens], dim=1)
                token_attention_mask = torch.cat(
                    [
                        token_attention_mask,
                        image_attention_mask.to(
                            device=token_attention_mask.device,
                            dtype=token_attention_mask.dtype,
                        ),
                    ],
                    dim=1,
                )
        elif image_features is not None:
            raise ValueError("image_features were supplied to a question-only predictor")
        if bool((token_attention_mask.sum(dim=1) == 0).any().item()):
            raise ValueError("every input must contain at least one valid question or image token")
        layer_ids = torch.arange(self.num_layers, device=tokens.device)
        layer_queries = self.layer_embedding(layer_ids).unsqueeze(0).expand(tokens.shape[0], -1, -1)
        attended, _ = self.cross_attention(
            query=layer_queries,
            key=tokens,
            value=tokens,
            key_padding_mask=~token_attention_mask.bool(),
            need_weights=False,
        )
        return self.layer_encoder(attended + layer_queries)


class BinaryPolarBackbone(nn.Module):
    """Token features -> layer queries -> direct binary VISUAL_ON logits."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        # P13 adds only a visual input projection. Build every P11-shared
        # tensor first so the same seed gives bit-identical common parameters.
        image_dim = kwargs.pop("image_dim", None)
        self.encoder = PolarLayerEncoder(**kwargs, image_dim=None)
        self.num_layers = self.encoder.num_layers
        self.route_head = nn.Linear(self.encoder.layer_embedding.embedding_dim, 1)
        if image_dim is not None:
            self.encoder.image_projection = nn.Linear(
                image_dim, self.encoder.layer_embedding.embedding_dim
            )

    def forward(
        self,
        token_features: torch.Tensor,
        token_attention_mask: torch.Tensor,
        image_features: torch.Tensor | None = None,
        image_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.route_head(
            self.encoder(
                token_features,
                token_attention_mask,
                image_features,
                image_attention_mask,
            )
        ).squeeze(-1)


class BiasOnlyBinaryPredictor(nn.Module):
    """Input-independent Bernoulli route prior, optionally conditioned on dataset."""

    def __init__(self, *, num_layers: int = 28, num_datasets: int = 1) -> None:
        super().__init__()
        if num_layers < 1 or num_datasets < 1:
            raise ValueError("num_layers and num_datasets must be positive")
        self.logits = nn.Parameter(torch.zeros(num_datasets, num_layers))

    def forward(self, dataset_ids: torch.Tensor) -> torch.Tensor:
        if dataset_ids.ndim != 1:
            raise ValueError("dataset_ids must have shape [B]")
        if self.logits.shape[0] == 1:
            return self.logits.expand(dataset_ids.shape[0], -1)
        return self.logits.index_select(0, dataset_ids.to(device=self.logits.device, dtype=torch.long))


class SegmentedBinaryPolarBackbone(nn.Module):
    """Canonical run-boundary and binary operation baseline."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.encoder = PolarLayerEncoder(**kwargs)
        self.num_layers = self.encoder.num_layers
        width = self.encoder.layer_embedding.embedding_dim
        self.boundary_head = nn.Linear(width, 1)
        self.operation_head = nn.Linear(width, 2)

    def forward(
        self,
        token_features: torch.Tensor,
        token_attention_mask: torch.Tensor,
        image_features: torch.Tensor | None = None,
        image_attention_mask: torch.Tensor | None = None,
    ):
        encoded = self.encoder(
            token_features,
            token_attention_mask,
            image_features,
            image_attention_mask,
        )
        return self.boundary_head(encoded).squeeze(-1), self.operation_head(encoded)


class FrozenHFTokenEncoder(nn.Module):
    """Thin frozen Hugging Face encoder used by the question-only POLAR path."""

    def __init__(self, model_name_or_path: str, *, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name_or_path, dtype=dtype, local_files_only=True)
        self.model.requires_grad_(False)
        self.model.eval()

    @property
    def output_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
