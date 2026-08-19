"""Binary VISUAL_ON router for cached Phase 5B text summaries."""

from __future__ import annotations

import math

import torch
from torch import nn


LEGACY_OPTIONAL_GATE_STATE_KEYS = frozenset(
    {
        "gate_layer_emb.weight",
        "gate_proj.0.weight",
        "gate_proj.0.bias",
        "gate_proj.2.weight",
        "gate_proj.2.bias",
    }
)


def load_binary_router_state_dict(router: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load legacy routers while allowing only the later optional gate head."""

    incompatible = router.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    disallowed_missing = missing - LEGACY_OPTIONAL_GATE_STATE_KEYS
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "incompatible BinaryVisualOnRouter checkpoint: "
            f"missing={sorted(disallowed_missing)}, unexpected={sorted(unexpected)}"
        )


class InputFallbackGate(nn.Module):
    """Conservatively decide whether to invoke a sparse layer router.

    The gate is evaluated *before* the first Qwen language layer.  It therefore
    cannot use route-conditioned hidden states, a benchmark identifier, an
    answer, or a search outcome.  Its only inputs are summaries of the rendered
    instruction tokens and the initial image-token embeddings.

    A positive output means "run the sparse router".  A non-positive output
    means "fall back to the exact all-VISUAL_ON route".  This separation is
    intentional: the gate decides whether sparse routing is admissible, while
    :class:`BinaryVisualOnRouter` decides the individual layer actions only
    after sparse routing has been admitted.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 256,
        use_last_token: bool = True,
        visual_summary_count: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if visual_summary_count < 0:
            raise ValueError("visual_summary_count must be non-negative")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.use_last_token = bool(use_last_token)
        self.visual_summary_count = int(visual_summary_count)
        self.dropout = float(dropout)
        field_count = 1 + int(self.use_last_token) + self.visual_summary_count
        self.proj = nn.Sequential(
            nn.Linear(field_count * self.d_model, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    @staticmethod
    def _rms_normalize(value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2:
            raise ValueError(f"input feature must have shape [B, D], got {tuple(value.shape)}")
        scale = torch.rsqrt(value.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return value.float() * scale

    def _validate_feature(self, name: str, value: torch.Tensor) -> tuple[int, int]:
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [B, D], got {tuple(value.shape)}")
        batch_size, d_model = value.shape
        if d_model != self.d_model:
            raise ValueError(f"{name} has d_model={d_model}, expected {self.d_model}")
        return int(batch_size), int(d_model)

    def forward(
        self,
        instruction_mean: torch.Tensor,
        instruction_last: torch.Tensor | None = None,
        visual_summaries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one sparse-routing-admission logit per sample.

        ``visual_summaries`` has shape ``[B, S, D]`` and conventionally contains
        mean(V) and mean(abs(V)).  The explicit tensor contract makes it hard to
        accidentally pass a post-routing or per-layer hidden trajectory here.
        """

        batch_size, _ = self._validate_feature("instruction_mean", instruction_mean)
        features = [self._rms_normalize(instruction_mean)]
        if self.use_last_token:
            if instruction_last is None:
                raise ValueError("instruction_last is required when use_last_token is true")
            last_batch, _ = self._validate_feature("instruction_last", instruction_last)
            if last_batch != batch_size:
                raise ValueError("instruction_mean and instruction_last batch sizes differ")
            features.append(self._rms_normalize(instruction_last))
        elif instruction_last is not None:
            raise ValueError("instruction_last was provided but use_last_token is false")

        if self.visual_summary_count == 0:
            if visual_summaries is not None:
                raise ValueError("visual_summaries were provided but visual_summary_count is 0")
        else:
            if visual_summaries is None:
                raise ValueError("visual_summaries are required when visual_summary_count is positive")
            expected = (batch_size, self.visual_summary_count, self.d_model)
            if tuple(visual_summaries.shape) != expected:
                raise ValueError(f"visual_summaries must have shape {expected}, got {tuple(visual_summaries.shape)}")
            features.extend(
                self._rms_normalize(visual_summaries[:, summary_index, :])
                for summary_index in range(self.visual_summary_count)
            )
        return self.proj(torch.cat(features, dim=-1)).squeeze(-1)

    def initialize_output_prior(self, sparse_probability: float) -> None:
        """Initialize to an explicit conservative sparse-routing prior."""

        probability = float(sparse_probability)
        if not 0.0 < probability < 1.0:
            raise ValueError("sparse_probability must be strictly between 0 and 1")
        output = self.proj[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("fallback output projection must be linear")
        with torch.no_grad():
            output.weight.zero_()
            output.bias.fill_(math.log(probability / (1.0 - probability)))


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
        feature_fusion_mode: str = "legacy_layer_add",
        layer_embedding_dim: int = 32,
        text_feature_mode: str = "full",
        gate_hidden_dim: int = 128,
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
        if feature_fusion_mode not in {"legacy_layer_add", "content_residual"}:
            raise ValueError(f"unsupported feature_fusion_mode: {feature_fusion_mode!r}")
        if layer_embedding_dim <= 0:
            raise ValueError("layer_embedding_dim must be positive")
        if text_feature_mode not in {"full", "mean_last", "mean_only"}:
            raise ValueError(f"unsupported text_feature_mode: {text_feature_mode!r}")
        if gate_hidden_dim <= 0:
            raise ValueError("gate_hidden_dim must be positive")

        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.use_prev_gate = bool(use_prev_gate)
        self.scalar_features_dim = int(scalar_features_dim)
        self.visual_summary_count = int(visual_summary_count)
        self.feature_fusion_mode = str(feature_fusion_mode)
        self.layer_embedding_dim = int(layer_embedding_dim)
        self.text_feature_mode = str(text_feature_mode)
        self.gate_hidden_dim = int(gate_hidden_dim)
        text_field_count = {"full": 3, "mean_last": 2, "mean_only": 1}[self.text_feature_mode]
        embedding_dim = self.d_model if self.feature_fusion_mode == "legacy_layer_add" else self.layer_embedding_dim
        self.layer_emb = nn.Embedding(self.num_layers, embedding_dim)
        self.gate_layer_emb = nn.Embedding(self.num_layers, embedding_dim)
        if self.use_prev_gate:
            self.prev_gate_emb = nn.Embedding(2, embedding_dim)
        if self.scalar_features_dim:
            self.scalar_proj = nn.Linear(self.scalar_features_dim, embedding_dim)
        if self.feature_fusion_mode == "legacy_layer_add":
            proj_input_dim = self.d_model * (
                text_field_count + 1 + (1 if self.scalar_features_dim else 0) + self.visual_summary_count
            )
        else:
            content_fields = text_field_count + self.visual_summary_count
            control_fields = 2 + (1 if self.scalar_features_dim else 0)
            proj_input_dim = self.d_model * content_fields + embedding_dim * control_fields
        self.proj = nn.Sequential(
            nn.Linear(proj_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        if self.feature_fusion_mode == "legacy_layer_add":
            gate_input_dim = self.d_model * (
                text_field_count + self.visual_summary_count
            ) + embedding_dim * (1 + (1 if self.scalar_features_dim else 0))
        else:
            gate_input_dim = self.d_model * (
                text_field_count + self.visual_summary_count
            ) + embedding_dim * (1 + (1 if self.scalar_features_dim else 0))
        self.gate_proj = nn.Sequential(
            nn.Linear(gate_input_dim, self.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(self.gate_hidden_dim, 1),
        )

    def initialize_output_prior(self, on_probability: float) -> None:
        """Start from a content-neutral Bernoulli policy with full action support."""
        probability = float(on_probability)
        if not 0.0 < probability < 1.0:
            raise ValueError("on_probability must be strictly between 0 and 1")
        output = self.proj[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("router output projection must be linear")
        with torch.no_grad():
            output.weight.zero_()
            output.bias.fill_(math.log(probability / (1.0 - probability)))

    def initialize_gate_prior(self, gate_probability: float) -> None:
        """Initialize conservative gate logits with a shared Bernoulli prior."""
        probability = float(gate_probability)
        if not 0.0 < probability < 1.0:
            raise ValueError("gate_probability must be strictly between 0 and 1")
        gate_output = self.gate_proj[-1]
        if not isinstance(gate_output, nn.Linear):
            raise TypeError("router gate output projection must be linear")
        with torch.no_grad():
            gate_output.weight.zero_()
            gate_output.bias.fill_(math.log(probability / (1.0 - probability)))

    def forward_gate(
        self,
        global_mean: torch.Tensor,
        window_mean: torch.Tensor,
        last_token: torch.Tensor,
        scalar_features: torch.Tensor | None = None,
        visual_summaries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict a sample-level sparse-routing gate logit.

        The gate is sample-level: it observes all per-layer summary context for the
        sample under candidate route mode and returns a scalar score.
        """
        batch_size, _, _ = self._validate_feature_tensor("global_mean", global_mean)
        self._validate_feature_tensor("window_mean", window_mean)
        self._validate_feature_tensor("last_token", last_token)

        layer_idx = torch.arange(self.num_layers, dtype=torch.long, device=global_mean.device).unsqueeze(0).expand(
            batch_size, -1
        )
        prev_gate = torch.zeros(batch_size, self.num_layers, dtype=torch.long, device=global_mean.device)
        scalar = self._normalize_scalar_features(
            None if scalar_features is None else scalar_features.to(device=global_mean.device),
            batch_size,
        )
        visual = self._normalize_visual_summaries(
            None if visual_summaries is None else visual_summaries.to(device=global_mean.device),
            batch_size,
        )

        features = self._compose_features(
            global_mean.float(),
            window_mean.float(),
            last_token.float(),
            layer_idx,
            prev_gate,
            scalar,
            visual,
        )

        return self.gate_proj(features).squeeze(-1).mean(dim=1)

    @staticmethod
    def _rms_normalize(value: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(value.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return value.float() * scale.to(dtype=value.dtype)

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

    def _compose_features(
        self,
        global_mean: torch.Tensor,
        window_mean: torch.Tensor,
        last_token: torch.Tensor,
        layer_idx: torch.Tensor,
        prev_gate: torch.Tensor,
        scalar: torch.Tensor | None,
        visual: torch.Tensor | None,
    ) -> torch.Tensor:
        layer = self.layer_emb(layer_idx)
        if self.use_prev_gate:
            prev = self.prev_gate_emb(prev_gate)
        else:
            prev = torch.zeros_like(layer)
        if self.feature_fusion_mode == "content_residual":
            text_parts = [self._rms_normalize(global_mean)]
            if self.text_feature_mode == "full":
                text_parts.append(self._rms_normalize(window_mean))
            if self.text_feature_mode in {"full", "mean_last"}:
                text_parts.append(self._rms_normalize(last_token))
            parts = [*text_parts, layer, prev]
            if scalar is not None:
                parts.append(scalar)
            if visual is not None:
                for summary_idx in range(self.visual_summary_count):
                    parts.append(self._rms_normalize(visual[:, :, summary_idx, :]))
            return torch.cat(parts, dim=-1)
        parts = [global_mean + layer]
        if self.text_feature_mode == "full":
            parts.append(window_mean + layer)
        if self.text_feature_mode in {"full", "mean_last"}:
            parts.append(last_token + layer)
        parts.append(prev)
        if scalar is not None:
            parts.append(scalar)
        if visual is not None:
            for summary_idx in range(self.visual_summary_count):
                parts.append(visual[:, :, summary_idx, :] + layer)
        return torch.cat(parts, dim=-1)

    def forward_layer(
        self,
        global_mean: torch.Tensor,
        window_mean: torch.Tensor,
        last_token: torch.Tensor,
        layer_idx: int | torch.Tensor,
        prev_gate: int | torch.Tensor,
        scalar_features: torch.Tensor | None = None,
        visual_summaries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict one VISUAL_ON logit for a single layer during live prefill.

        The cached-feature ``forward`` method consumes all layer summaries at
        once. This method keeps the same MLP and embeddings but accepts only the
        current layer-entry summary, so the caller can decide whether to execute
        that layer with visual tokens before computing later hidden states.
        """

        if global_mean.ndim == 2:
            global_mean = global_mean.unsqueeze(1)
        if window_mean.ndim == 2:
            window_mean = window_mean.unsqueeze(1)
        if last_token.ndim == 2:
            last_token = last_token.unsqueeze(1)
        for name, value in {
            "global_mean": global_mean,
            "window_mean": window_mean,
            "last_token": last_token,
        }.items():
            if value.ndim != 3 or value.shape[1] != 1 or value.shape[2] != self.d_model:
                raise ValueError(f"{name} must have shape [B, D] or [B, 1, D], got {tuple(value.shape)}")
        batch_size = int(global_mean.shape[0])
        if window_mean.shape[0] != batch_size or last_token.shape[0] != batch_size:
            raise ValueError("all text summary tensors must have the same batch size")

        device = global_mean.device
        if isinstance(layer_idx, int):
            layer_tensor = torch.full((batch_size, 1), layer_idx, dtype=torch.long, device=device)
        else:
            layer_tensor = layer_idx.to(device=device, dtype=torch.long)
            if layer_tensor.ndim == 0:
                layer_tensor = layer_tensor.view(1, 1).expand(batch_size, 1)
            elif layer_tensor.ndim == 1:
                if layer_tensor.numel() == 1:
                    layer_tensor = layer_tensor.view(1, 1).expand(batch_size, 1)
                elif layer_tensor.shape[0] == batch_size:
                    layer_tensor = layer_tensor.view(batch_size, 1)
                else:
                    raise ValueError(f"layer_idx must have 1 or {batch_size} entries, got {tuple(layer_tensor.shape)}")
            elif layer_tensor.ndim != 2 or tuple(layer_tensor.shape) != (batch_size, 1):
                raise ValueError(f"layer_idx must have shape [{batch_size}, 1], got {tuple(layer_tensor.shape)}")
        if bool(((layer_tensor < 0) | (layer_tensor >= self.num_layers)).any().item()):
            raise ValueError(f"layer_idx must be in [0, {self.num_layers})")

        if isinstance(prev_gate, int):
            prev_tensor = torch.full((batch_size, 1), prev_gate, dtype=torch.long, device=device)
        else:
            prev_tensor = prev_gate.to(device=device, dtype=torch.long)
            if prev_tensor.ndim == 0:
                prev_tensor = prev_tensor.view(1, 1).expand(batch_size, 1)
            elif prev_tensor.ndim == 1:
                if prev_tensor.numel() == 1:
                    prev_tensor = prev_tensor.view(1, 1).expand(batch_size, 1)
                elif prev_tensor.shape[0] == batch_size:
                    prev_tensor = prev_tensor.view(batch_size, 1)
                else:
                    raise ValueError(f"prev_gate must have 1 or {batch_size} entries, got {tuple(prev_tensor.shape)}")
            elif prev_tensor.ndim != 2 or tuple(prev_tensor.shape) != (batch_size, 1):
                raise ValueError(f"prev_gate must have shape [{batch_size}, 1], got {tuple(prev_tensor.shape)}")
        if bool(((prev_tensor < 0) | (prev_tensor > 1)).any().item()):
            raise ValueError("prev_gate must contain only 0/1 values")

        scalar = None
        if self.scalar_features_dim == 0:
            if scalar_features is not None:
                raise ValueError("scalar_features were provided but scalar_features_dim is 0")
        else:
            if scalar_features is None:
                raise ValueError("scalar_features are required when scalar_features_dim is positive")
            scalar_input = scalar_features.to(device=device, dtype=torch.float32)
            if scalar_input.ndim == 2:
                if tuple(scalar_input.shape) != (batch_size, self.scalar_features_dim):
                    raise ValueError(
                        "scalar_features must have shape "
                        f"[{batch_size}, {self.scalar_features_dim}], got {tuple(scalar_input.shape)}"
                    )
                scalar = self.scalar_proj(scalar_input).unsqueeze(1)
            elif scalar_input.ndim == 3:
                if tuple(scalar_input.shape) != (batch_size, 1, self.scalar_features_dim):
                    raise ValueError(
                        "single-layer scalar_features must have shape "
                        f"[{batch_size}, 1, {self.scalar_features_dim}], got {tuple(scalar_input.shape)}"
                    )
                scalar = self.scalar_proj(scalar_input)
            else:
                raise ValueError(f"scalar_features must have rank 2 or 3, got {scalar_input.ndim}")

        visual = None
        if self.visual_summary_count == 0:
            if visual_summaries is not None:
                raise ValueError("visual_summaries were provided but visual_summary_count is 0")
        else:
            if visual_summaries is None:
                raise ValueError("visual_summaries are required when visual_summary_count is positive")
            visual = visual_summaries.to(device=device, dtype=torch.float32)
            if visual.ndim == 3:
                expected = (batch_size, self.visual_summary_count, self.d_model)
                if tuple(visual.shape) != expected:
                    raise ValueError(f"visual_summaries must have shape {expected}, got {tuple(visual.shape)}")
                visual = visual.unsqueeze(1)
            elif visual.ndim != 4 or tuple(visual.shape) != (
                batch_size,
                1,
                self.visual_summary_count,
                self.d_model,
            ):
                raise ValueError(
                    "single-layer visual_summaries must have shape "
                    f"[{batch_size}, {self.visual_summary_count}, {self.d_model}] or "
                    f"[{batch_size}, 1, {self.visual_summary_count}, {self.d_model}], "
                    f"got {tuple(visual.shape)}"
                )

        features = self._compose_features(
            global_mean.float(),
            window_mean.float(),
            last_token.float(),
            layer_tensor,
            prev_tensor,
            scalar,
            visual,
        )
        return self.proj(features).squeeze(-1).squeeze(-1)

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

        features = self._compose_features(
            global_mean.float(),
            window_mean.float(),
            last_token.float(),
            layer_idx,
            prev_gate,
            scalar,
            visual,
        )
        return self.proj(features).squeeze(-1)
