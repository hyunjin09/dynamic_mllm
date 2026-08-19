from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)


@dataclass
class ReadDecomposition:
    visual_contribution: torch.Tensor
    nonvisual_contribution: torch.Tensor
    full_recomputed: torch.Tensor
    decomposition_max_abs: float
    reference_actual_rms_ratio: float
    quantization_adjustment_max_abs: float
    quantization_half_ulp_ratio_max: float
    quantization_adjustment_rms: float
    ideal_visual_delta_rms: float
    quantization_adjustment_to_ideal_rms: float
    text_visual_attention_mass_mean: float
    visual_future_attention_mass_max: float


@dataclass
class ReadInterventionCache:
    actual_output: torch.Tensor | None = None
    off_output: torch.Tensor | None = None
    reconstructed: torch.Tensor | None = None
    decomposition: ReadDecomposition | None = None
    hook_identity_max_abs: float | None = None


def apply_reference_attention_mask(
    weights: torch.Tensor,
    attention_mask: torch.Tensor | None,
    query_start: int,
    key_length: int,
) -> torch.Tensor:
    """Match Qwen SDPA masking for a cache-free prompt chunk."""
    query_stop = query_start + weights.shape[-2]
    if attention_mask is not None:
        return weights + attention_mask[:, :, query_start:query_stop, :key_length]
    query_positions = torch.arange(query_start, query_stop, device=weights.device)
    key_positions = torch.arange(key_length, device=weights.device)
    causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    return weights.masked_fill(
        ~causal.unsqueeze(0).unsqueeze(0),
        torch.finfo(weights.dtype).min,
    )


def quantized_path_subtraction(
    full_output: torch.Tensor,
    ideal_path_contribution: torch.Tensor,
    affected_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Remove a path at runtime dtype and retain an exact add-back residual.

    The ideal path uses the original fixed attention weights. The executed OFF
    state must have the model dtype, so its one unavoidable quantization is
    explicit. The effective residual is the exact representable difference
    between FULL and that executed OFF state.
    """
    ideal_delta = torch.where(
        affected_rows.unsqueeze(-1),
        ideal_path_contribution.float(),
        torch.zeros_like(full_output, dtype=torch.float32),
    )
    off_output = (full_output.float() - ideal_delta).to(full_output.dtype)
    effective_delta = full_output.float() - off_output.float()
    reconstructed = (off_output.float() + effective_delta).to(full_output.dtype)
    adjustment = effective_delta - ideal_delta

    exact_off = full_output.float() - ideal_delta
    rounded_off = off_output.float()
    lower = torch.nextafter(
        off_output,
        torch.full_like(off_output, -torch.inf),
    ).float()
    upper = torch.nextafter(
        off_output,
        torch.full_like(off_output, torch.inf),
    ).float()
    spacing_toward_exact = torch.where(exact_off >= rounded_off, upper - rounded_off, rounded_off - lower)
    half_ulp = spacing_toward_exact * 0.5

    selected = affected_rows.unsqueeze(-1).expand_as(full_output)
    selected_adjustment = adjustment[selected]
    selected_ideal = ideal_delta[selected]
    selected_half_ulp = half_ulp[selected]
    nonzero_bound = selected_half_ulp > 0
    half_ulp_ratio = torch.zeros_like(selected_adjustment)
    half_ulp_ratio[nonzero_bound] = (
        selected_adjustment[nonzero_bound].abs() / selected_half_ulp[nonzero_bound]
    )
    adjustment_rms = float(selected_adjustment.float().square().mean().sqrt().item())
    ideal_rms = float(selected_ideal.float().square().mean().sqrt().item())
    metrics = {
        "adjustment_max_abs": float(selected_adjustment.abs().max().item()),
        "half_ulp_ratio_max": float(half_ulp_ratio.max().item()),
        "adjustment_rms": adjustment_rms,
        "ideal_visual_delta_rms": ideal_rms,
        "adjustment_to_ideal_rms": adjustment_rms / ideal_rms if ideal_rms > 0 else 0.0,
    }
    return off_output, effective_delta, reconstructed, metrics


def decompose_visual_value_path(
    attention_module,
    normalized_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    visual_token_mask: torch.Tensor,
) -> ReadDecomposition:
    """Decompose eager attention with fixed weights into visual/non-visual value paths."""
    batch_size, query_length, _ = normalized_hidden_states.shape
    query_states = attention_module.q_proj(normalized_hidden_states)
    key_states = attention_module.k_proj(normalized_hidden_states)
    value_states = attention_module.v_proj(normalized_hidden_states)

    query_states = query_states.view(
        batch_size, query_length, attention_module.num_heads, attention_module.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        batch_size, query_length, attention_module.num_key_value_heads, attention_module.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        batch_size, query_length, attention_module.num_key_value_heads, attention_module.head_dim
    ).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        attention_module.rope_scaling["mrope_section"],
    )
    key_states = repeat_kv(key_states, attention_module.num_key_value_groups)
    value_states = repeat_kv(value_states, attention_module.num_key_value_groups)

    visual_keys = visual_token_mask[:, None, :, None].to(value_states.dtype)
    nonvisual_keys = 1.0 - visual_keys
    visual_values = value_states * visual_keys
    nonvisual_values = value_states * nonvisual_keys

    visual_context_chunks: list[torch.Tensor] = []
    nonvisual_context_chunks: list[torch.Tensor] = []
    visual_mass_sum = 0.0
    visual_mass_count = 0
    future_max = 0.0
    key_positions = torch.arange(query_length, device=query_states.device)
    visual_ranges: list[tuple[int, int]] = []
    for batch_index in range(batch_size):
        visual_indices = torch.where(visual_token_mask[batch_index])[0]
        if visual_indices.numel() == 0:
            raise ValueError("No visual-token rows for READ decomposition")
        first_visual = int(visual_indices[0].item())
        last_visual = int(visual_indices[-1].item())
        if visual_indices.numel() != last_visual - first_visual + 1:
            raise ValueError("READ decomposition requires contiguous visual-token rows")
        visual_ranges.append((first_visual, last_visual))

    query_chunk_size = int(getattr(attention_module, "stage_a_query_chunk_size", 1024))
    for query_start in range(0, query_length, query_chunk_size):
        query_stop = min(query_start + query_chunk_size, query_length)
        query_chunk = query_states[:, :, query_start:query_stop, :]
        weights = torch.matmul(query_chunk, key_states.transpose(2, 3)) / math.sqrt(
            attention_module.head_dim
        )
        weights = apply_reference_attention_mask(
            weights,
            attention_mask,
            query_start,
            key_states.shape[-2],
        )
        if query_states.dtype == torch.float16:
            weights = torch.where(torch.isinf(weights), torch.zeros_like(weights), weights)
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        visual_context_chunks.append(torch.matmul(weights, visual_values))
        nonvisual_context_chunks.append(torch.matmul(weights, nonvisual_values))

        query_positions = torch.arange(query_start, query_stop, device=weights.device)
        future_keys = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        for batch_index, (first_visual, last_visual) in enumerate(visual_ranges):
            visual_mass = weights[
                batch_index, :, :, first_visual : last_visual + 1
            ].sum(dim=-1)
            text_rows = ~visual_token_mask[batch_index, query_start:query_stop]
            selected_mass = visual_mass[:, text_rows]
            visual_mass_sum += float(selected_mass.float().sum().item())
            visual_mass_count += selected_mass.numel()

            visual_queries = visual_token_mask[batch_index, query_start:query_stop]
            if visual_queries.any():
                future_weights = torch.where(
                    future_keys[visual_queries].unsqueeze(0),
                    weights[batch_index, :, visual_queries, :].abs(),
                    torch.zeros((), dtype=weights.dtype, device=weights.device),
                )
                future_max = max(future_max, float(future_weights.max().item()))

    visual_context = torch.cat(visual_context_chunks, dim=2)
    nonvisual_context = torch.cat(nonvisual_context_chunks, dim=2)

    def project(context: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        context = context.transpose(1, 2).contiguous().reshape(batch_size, query_length, -1)
        return F.linear(context, attention_module.o_proj.weight, bias)

    visual_output = project(visual_context, None)
    nonvisual_output = project(nonvisual_context, attention_module.o_proj.bias)
    recomputed = visual_output + nonvisual_output

    text_visual_mass = visual_mass_sum / visual_mass_count if visual_mass_count else 0.0

    return ReadDecomposition(
        visual_contribution=visual_output,
        nonvisual_contribution=nonvisual_output,
        full_recomputed=recomputed,
        decomposition_max_abs=0.0,
        reference_actual_rms_ratio=0.0,
        quantization_adjustment_max_abs=0.0,
        quantization_half_ulp_ratio_max=0.0,
        quantization_adjustment_rms=0.0,
        ideal_visual_delta_rms=0.0,
        quantization_adjustment_to_ideal_rms=0.0,
        text_visual_attention_mass_mean=text_visual_mass,
        visual_future_attention_mass_max=float(future_max),
    )


def _subtract_read_delta(
    actual_output: torch.Tensor,
    removal_delta: torch.Tensor,
    visual_token_mask: torch.Tensor,
) -> torch.Tensor:
    if removal_delta.shape != actual_output.shape:
        raise ValueError("Removal READ delta must match the attention output shape")
    return torch.where(
        visual_token_mask.unsqueeze(-1),
        actual_output,
        (actual_output.float() - removal_delta.float()).to(actual_output.dtype),
    )


class ReadPathController:
    """Subtract or reconstruct the exact visual-value path at self-attention output."""

    def __init__(
        self,
        attention_module,
        visual_token_mask: torch.Tensor,
        mode: str,
        cache: ReadInterventionCache | None = None,
        replacement_delta: torch.Tensor | None = None,
    ):
        if mode not in {"full", "off", "reconstruct", "reference", "replace", "subtract"}:
            raise ValueError(f"Unsupported READ mode: {mode}")
        if mode in {"replace", "subtract"} and replacement_delta is None:
            raise ValueError(f"READ {mode} mode requires a replacement delta")
        self.attention_module = attention_module
        self.visual_token_mask = visual_token_mask.bool()
        self.mode = mode
        self.cache = cache
        self.replacement_delta = replacement_delta
        self._inputs: tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor]] | None = None
        self.decomposition: ReadDecomposition | None = None
        self.hook_identity_max_abs: float | None = None
        self._pre_handle = None
        self._post_handle = None

    def __enter__(self):
        self._pre_handle = self.attention_module.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        self._post_handle = self.attention_module.register_forward_hook(self._post_hook, with_kwargs=True)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._pre_handle.remove()
        self._post_handle.remove()

    def _pre_hook(self, module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        if hidden_states is None:
            raise RuntimeError("Could not capture attention hidden states")
        past_key_value = kwargs.get("past_key_value")
        if past_key_value is not None and past_key_value.get_seq_length(module.layer_idx) != 0:
            raise RuntimeError("Primary READ decomposition requires an empty target-layer prompt cache")
        position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is None:
            raise RuntimeError("Qwen2.5-VL position embeddings were not supplied to attention")
        self._inputs = (hidden_states, kwargs.get("attention_mask"), position_embeddings)

    def _post_hook(self, module, args, kwargs, output):
        if self._inputs is None:
            raise RuntimeError("READ pre-hook did not capture inputs")
        actual_output = output[0]
        if self.mode == "full":
            self.hook_identity_max_abs = 0.0
            self.decomposition = self.cache.decomposition if self.cache is not None else None
            return output
        if self.cache is not None and self.cache.decomposition is not None:
            if self.cache.actual_output is None or not torch.equal(actual_output, self.cache.actual_output):
                raise RuntimeError("READ cache was reused with a different attention output")
            decomposition = self.cache.decomposition
            off = self.cache.off_output
            reconstructed = self.cache.reconstructed
            self.hook_identity_max_abs = self.cache.hook_identity_max_abs
            if off is None or reconstructed is None or self.hook_identity_max_abs is None:
                raise RuntimeError("READ cache is incomplete")
            self.decomposition = decomposition
            if self.mode == "off":
                replacement = off
            elif self.mode == "reconstruct":
                replacement = reconstructed
            elif self.mode == "replace":
                if self.replacement_delta is None or self.replacement_delta.shape != actual_output.shape:
                    raise ValueError("Replacement READ delta must match the attention output shape")
                replacement = torch.where(
                    self.visual_token_mask.unsqueeze(-1),
                    actual_output,
                    (off.float() + self.replacement_delta.float()).to(actual_output.dtype),
                )
            elif self.mode == "subtract":
                assert self.replacement_delta is not None
                replacement = _subtract_read_delta(
                    actual_output, self.replacement_delta, self.visual_token_mask
                )
            else:
                replacement = decomposition.full_recomputed
            return (replacement,) + tuple(output[1:])

        hidden_states, attention_mask, position_embeddings = self._inputs
        decomposition = decompose_visual_value_path(
            module,
            hidden_states,
            attention_mask,
            position_embeddings,
            self.visual_token_mask,
        )
        decomposition.decomposition_max_abs = float(
            (decomposition.full_recomputed.float() - actual_output.float()).abs().max().item()
        )
        reference_difference = decomposition.full_recomputed.float() - actual_output.float()
        actual_rms = actual_output.float().square().mean().sqrt().item()
        decomposition.reference_actual_rms_ratio = float(
            reference_difference.square().mean().sqrt().item() / max(actual_rms, 1e-12)
        )
        off, effective_delta, reconstructed, precision = quantized_path_subtraction(
            actual_output,
            decomposition.visual_contribution,
            ~self.visual_token_mask,
        )
        decomposition.quantization_adjustment_max_abs = precision["adjustment_max_abs"]
        decomposition.quantization_half_ulp_ratio_max = precision["half_ulp_ratio_max"]
        decomposition.quantization_adjustment_rms = precision["adjustment_rms"]
        decomposition.ideal_visual_delta_rms = precision["ideal_visual_delta_rms"]
        decomposition.quantization_adjustment_to_ideal_rms = precision[
            "adjustment_to_ideal_rms"
        ]
        self.hook_identity_max_abs = float((reconstructed.float() - actual_output.float()).abs().max().item())
        self.decomposition = decomposition
        if self.cache is not None:
            self.cache.actual_output = actual_output.detach().clone()
            self.cache.off_output = off.detach().clone()
            self.cache.reconstructed = reconstructed.detach().clone()
            self.cache.decomposition = decomposition
            self.cache.hook_identity_max_abs = self.hook_identity_max_abs

        if self.mode == "off":
            replacement = off
        elif self.mode == "reconstruct":
            replacement = reconstructed
        elif self.mode == "replace":
            if self.replacement_delta is None or self.replacement_delta.shape != actual_output.shape:
                raise ValueError("Replacement READ delta must match the attention output shape")
            replacement = torch.where(
                self.visual_token_mask.unsqueeze(-1),
                actual_output,
                (off.float() + self.replacement_delta.float()).to(actual_output.dtype),
            )
        elif self.mode == "subtract":
            assert self.replacement_delta is not None
            replacement = _subtract_read_delta(
                actual_output, self.replacement_delta, self.visual_token_mask
            )
        elif self.mode == "reference":
            replacement = decomposition.full_recomputed
        else:
            replacement = actual_output
        return (replacement,) + tuple(output[1:])
