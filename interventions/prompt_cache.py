from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from interventions.four_state import LayerContext, _write_state
from interventions.read_path import ReadDecomposition, ReadInterventionCache, ReadPathController


def _layer_tensors(cache: DynamicCache, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_index]
        if not layer.is_initialized:
            return torch.tensor([]), torch.tensor([])
        return layer.keys, layer.values
    return cache.key_cache[layer_index], cache.value_cache[layer_index]


def _layer_count(cache: DynamicCache) -> int:
    return len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)


def _append_empty_layer(cache: DynamicCache) -> None:
    if hasattr(cache, "layers"):
        cache.layers.append(DynamicLayer())
    else:
        cache.key_cache.append(torch.tensor([]))
        cache.value_cache.append(torch.tensor([]))


# Transformers 5 moved DynamicCache storage from key_cache/value_cache lists to
# DynamicLayer objects. Preserve the repository's read-only compatibility
# surface while the implementation below uses the version-neutral helpers.
if not hasattr(DynamicCache, "key_cache"):
    DynamicCache.key_cache = property(  # type: ignore[attr-defined]
        lambda cache: [_layer_tensors(cache, index)[0] for index in range(_layer_count(cache))]
    )
    DynamicCache.value_cache = property(  # type: ignore[attr-defined]
        lambda cache: [_layer_tensors(cache, index)[1] for index in range(_layer_count(cache))]
    )


@dataclass
class PromptStateResult:
    name: str
    target_output: torch.Tensor
    prompt_logits: torch.Tensor
    past_key_values: DynamicCache
    read_decomposition: ReadDecomposition | None
    read_hook_identity_max_abs: float
    write_hook_identity_max_abs: float
    injected_prestate_max_abs: float


def clone_dynamic_cache(
    source: DynamicCache, through_layer_exclusive: int | None = None
) -> DynamicCache:
    source_layers = _layer_count(source)
    stop = source_layers if through_layer_exclusive is None else through_layer_exclusive
    if stop < 0 or stop > source_layers:
        raise ValueError("Invalid cache layer boundary")
    cloned = DynamicCache()
    for layer_index in range(stop):
        key, value = _layer_tensors(source, layer_index)
        if key.numel():
            cloned.update(key.detach().clone(), value.detach().clone(), layer_index)
        else:
            while _layer_count(cloned) <= layer_index:
                _append_empty_layer(cloned)
    return cloned


def truncate_dynamic_cache(source: DynamicCache, sequence_length: int) -> DynamicCache:
    """Clone a prompt cache while dropping right-padding token positions.

    Common-padded prompt execution keeps visual computation shape-identical
    across same-image questions. Accepted-answer scoring must nevertheless
    start at each question's original prompt boundary, so padding rows are
    removed from every layer cache before the frozen continuation scorer runs.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    truncated = DynamicCache()
    for layer_index in range(_layer_count(source)):
        key, value = _layer_tensors(source, layer_index)
        if key.numel() == 0:
            while _layer_count(truncated) <= layer_index:
                _append_empty_layer(truncated)
            continue
        if key.shape[-2] < sequence_length or value.shape[-2] < sequence_length:
            raise ValueError("Cannot truncate cache beyond its sequence length")
        truncated.update(
            key[..., :sequence_length, :].detach().clone(),
            value[..., :sequence_length, :].detach().clone(),
            layer_index,
        )
    return truncated


def run_cached_prompt_state(
    causal_lm,
    context: LayerContext,
    baseline_cache: DynamicCache,
    visual_token_mask: torch.Tensor,
    name: str,
    read_mode: str,
    write_mode: str,
    read_cache: ReadInterventionCache | None = None,
    read_replacement_delta: torch.Tensor | None = None,
    write_replacement_delta: torch.Tensor | None = None,
) -> PromptStateResult:
    """Run a prompt state from one identical cached pre-layer activation.

    Prefix-layer K/V are cloned from the unmodified prompt. The target and all
    suffix layers rebuild their prompt K/V under the requested state, yielding a
    state-specific cache for unchanged answer scoring and greedy decoding.
    """
    decoder = causal_lm.model
    layer = decoder.layers[context.layer_index]
    state_cache = clone_dynamic_cache(
        baseline_cache, through_layer_exclusive=context.layer_index
    )
    injected = context.pre_layer_state.detach().clone()
    injected_error = float(
        (injected.float() - context.pre_layer_state.float()).abs().max().item()
    )
    kwargs = dict(context.layer_kwargs)
    kwargs["past_key_value"] = state_cache
    kwargs["use_cache"] = True
    kwargs["output_attentions"] = False

    with ReadPathController(
        layer.self_attn,
        visual_token_mask,
        read_mode,
        read_cache,
        replacement_delta=read_replacement_delta,
    ) as controller:
        target_output = layer(injected, **kwargs)[0]
    target_output, write_identity = _write_state(
        target_output,
        context,
        visual_token_mask,
        write_mode,
        write_replacement_delta,
    )

    hidden_states = target_output
    for suffix_layer in decoder.layers[context.layer_index + 1 :]:
        hidden_states = suffix_layer(hidden_states, **kwargs)[0]
    hidden_states = decoder.norm(hidden_states)
    prompt_logits = causal_lm.lm_head(hidden_states)
    if controller.hook_identity_max_abs is None:
        raise RuntimeError("READ controller did not record its identity")
    if len(state_cache.key_cache) != len(decoder.layers):
        raise RuntimeError("Prompt state did not populate every decoder cache layer")
    return PromptStateResult(
        name=name,
        target_output=target_output,
        prompt_logits=prompt_logits,
        past_key_values=state_cache,
        read_decomposition=controller.decomposition,
        read_hook_identity_max_abs=controller.hook_identity_max_abs,
        write_hook_identity_max_abs=write_identity,
        injected_prestate_max_abs=injected_error,
    )
