from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from interventions.four_state import LayerContext
from interventions.prompt_cache import clone_dynamic_cache


@dataclass
class VisualReplay:
    refined_visual_state: torch.Tensor
    native_visual_state: torch.Tensor
    visual_reconstruction_max_abs: float
    conditioning_edge_count: int


@dataclass
class RefinedPromptState:
    prompt_logits: torch.Tensor
    past_key_values: DynamicCache
    refined_visual_state: torch.Tensor
    native_visual_state: torch.Tensor
    visual_reconstruction_max_abs: float
    inserted_visual_max_abs: float
    conditioning_edge_count: int


def minimal_contextual_question_span(
    *,
    prompt_text: str,
    question: str,
    fast_tokenizer,
    slow_tokenizer,
    actual_input_ids: torch.Tensor,
    image_token_id: int,
) -> dict[str, Any]:
    """Map the literal question to the minimal contextual token cover.

    The pinned slow Qwen tokenizer has no offset mapping. A fast tokenizer is
    used only to obtain offsets after verifying exact token-ID equality with
    the pinned tokenizer. The processor expands one image placeholder into the
    native visual-token run; this expansion is reconstructed and checked
    exactly before indices are mapped to the actual prompt.
    """
    starts = []
    cursor = 0
    while True:
        found = prompt_text.find(question, cursor)
        if found < 0:
            break
        starts.append(found)
        cursor = found + 1
    if len(starts) != 1:
        raise ValueError(f"Question must occur exactly once in prompt text, found {len(starts)}")
    char_start = starts[0]
    char_stop = char_start + len(question)

    encoded = fast_tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prompt_ids = [int(value) for value in encoded["input_ids"]]
    slow_ids = [
        int(value)
        for value in slow_tokenizer(prompt_text, add_special_tokens=False).input_ids
    ]
    if prompt_ids != slow_ids:
        raise RuntimeError("Fast offset tokenizer does not reproduce pinned slow-tokenizer IDs")
    offsets = [(int(start), int(stop)) for start, stop in encoded["offset_mapping"]]
    span = [
        index
        for index, (start, stop) in enumerate(offsets)
        if stop > char_start and start < char_stop
    ]
    if not span or span != list(range(span[0], span[-1] + 1)):
        raise RuntimeError("Question does not map to a nonempty contiguous token span")

    image_positions = [index for index, value in enumerate(prompt_ids) if value == image_token_id]
    if len(image_positions) != 1:
        raise RuntimeError("Chat-template text must contain exactly one image placeholder")
    actual_ids = [int(value) for value in actual_input_ids.reshape(-1).tolist()]
    visual_count = sum(value == image_token_id for value in actual_ids)
    if visual_count < 1:
        raise RuntimeError("Processed prompt contains no image tokens")
    placeholder = image_positions[0]
    expanded = (
        prompt_ids[:placeholder]
        + [image_token_id] * visual_count
        + prompt_ids[placeholder + 1 :]
    )
    if expanded != actual_ids:
        raise RuntimeError("Image-placeholder expansion does not reproduce processed input IDs")
    shift = visual_count - 1
    mapped = [index if index <= placeholder else index + shift for index in span]
    if mapped[0] <= placeholder + visual_count - 1:
        raise RuntimeError("Question span does not follow the visual-token run")

    covered_start = offsets[span[0]][0]
    covered_stop = offsets[span[-1]][1]
    covered_text = prompt_text[covered_start:covered_stop]
    prefix_extra = prompt_text[covered_start:char_start]
    suffix_extra = prompt_text[char_stop:covered_stop]
    if prefix_extra.strip() or suffix_extra.strip():
        raise RuntimeError("Contextual question-token cover includes non-whitespace text")
    if question not in covered_text:
        raise RuntimeError("Contextual token cover does not contain the literal question")

    mask = torch.zeros_like(actual_input_ids, dtype=torch.bool)
    mask[:, mapped] = True
    return {
        "mask": mask,
        "token_first": mapped[0],
        "token_last": mapped[-1],
        "token_count": len(mapped),
        "char_start": char_start,
        "char_stop": char_stop,
        "covered_text": covered_text,
        "boundary_prefix": prefix_extra,
        "boundary_suffix": suffix_extra,
        "exact_literal_contained": True,
        "only_whitespace_outside_literal": True,
        "fast_slow_token_ids_identical": True,
        "processor_expansion_identical": True,
    }


def conditioned_visual_attention_mask(
    native_attention_mask: torch.Tensor,
    visual_token_mask: torch.Tensor,
    question_token_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Permit visual queries to attend only the frozen literal-question span."""
    if native_attention_mask.ndim != 4 or native_attention_mask.shape[1] != 1:
        raise ValueError("Replay requires a four-dimensional additive attention mask")
    if visual_token_mask.shape != question_token_mask.shape:
        raise ValueError("Visual and question masks must have identical shapes")
    if visual_token_mask.shape[0] != native_attention_mask.shape[0]:
        raise ValueError("Token-mask batch does not match attention-mask batch")
    if not bool(visual_token_mask.any().item()) or not bool(question_token_mask.any().item()):
        raise ValueError("Replay masks must select visual and question tokens")
    if bool((visual_token_mask & question_token_mask).any().item()):
        raise ValueError("Visual and question masks overlap")

    result = native_attention_mask.detach().clone()
    edge_count = 0
    for batch_index in range(result.shape[0]):
        visual = torch.where(visual_token_mask[batch_index])[0]
        question = torch.where(question_token_mask[batch_index])[0]
        if int(question[0]) <= int(visual[-1]):
            raise ValueError("Question tokens must occur after visual tokens")
        result[batch_index, 0, visual[:, None], question[None, :]] = 0
        edge_count += int(visual.numel() * question.numel())
    return result, edge_count


def replay_visual_rows(
    layer,
    context: LayerContext,
    visual_token_mask: torch.Tensor,
    question_token_mask: torch.Tensor | None,
) -> VisualReplay:
    """Run one native frozen layer and retain only its visual output rows."""
    kwargs = dict(context.layer_kwargs)
    kwargs["past_key_value"] = None
    kwargs["use_cache"] = False
    kwargs["output_attentions"] = False
    edge_count = 0
    if question_token_mask is not None:
        conditioned, edge_count = conditioned_visual_attention_mask(
            kwargs["attention_mask"], visual_token_mask, question_token_mask
        )
        kwargs["attention_mask"] = conditioned
    replay_output = layer(context.pre_layer_state.detach().clone(), **kwargs)[0]
    selected = visual_token_mask.unsqueeze(-1).expand_as(replay_output)
    refined = replay_output[selected].reshape(
        replay_output.shape[0], -1, replay_output.shape[-1]
    )
    native = context.full_layer_output[selected].reshape(
        replay_output.shape[0], -1, replay_output.shape[-1]
    )
    reconstruction = float((refined.float() - native.float()).abs().max().item())
    return VisualReplay(
        refined_visual_state=refined,
        native_visual_state=native,
        visual_reconstruction_max_abs=reconstruction,
        conditioning_edge_count=edge_count,
    )


def run_refined_prompt_state(
    causal_lm,
    *,
    target_context: LayerContext,
    conditioning_context: LayerContext,
    baseline_cache: DynamicCache,
    target_visual_token_mask: torch.Tensor,
    conditioning_visual_token_mask: torch.Tensor,
    conditioning_question_token_mask: torch.Tensor | None,
) -> RefinedPromptState:
    """Insert replayed visual rows at H_(l+1), then run the unchanged suffix."""
    if target_context.layer_index != conditioning_context.layer_index:
        raise ValueError("Target and conditioning contexts must use the same layer")
    if target_context.pre_layer_state.shape != conditioning_context.pre_layer_state.shape:
        raise ValueError("Common-padded target and conditioning states must have equal shape")
    if target_visual_token_mask.shape != conditioning_visual_token_mask.shape:
        raise ValueError("Target and conditioning visual masks must have equal shape")

    decoder = causal_lm.model
    layer_index = target_context.layer_index
    replay = replay_visual_rows(
        decoder.layers[layer_index],
        conditioning_context,
        conditioning_visual_token_mask,
        conditioning_question_token_mask,
    )
    target_visual = target_context.pre_layer_state[
        target_visual_token_mask.unsqueeze(-1).expand_as(target_context.pre_layer_state)
    ].reshape_as(replay.refined_visual_state)
    conditioning_visual = conditioning_context.pre_layer_state[
        conditioning_visual_token_mask.unsqueeze(-1).expand_as(
            conditioning_context.pre_layer_state
        )
    ].reshape_as(replay.refined_visual_state)
    if not torch.equal(target_visual, conditioning_visual):
        raise RuntimeError("Replay variants do not begin from identical visual states")

    hidden_states = target_context.full_layer_output.detach().clone()
    selected = target_visual_token_mask.unsqueeze(-1).expand_as(hidden_states)
    hidden_states[selected] = replay.refined_visual_state.reshape(-1)
    inserted = hidden_states[selected].reshape_as(replay.refined_visual_state)
    inserted_error = float(
        (inserted.float() - replay.refined_visual_state.float()).abs().max().item()
    )

    state_cache = clone_dynamic_cache(baseline_cache, through_layer_exclusive=layer_index + 1)
    kwargs = dict(target_context.layer_kwargs)
    kwargs["past_key_value"] = state_cache
    kwargs["use_cache"] = True
    kwargs["output_attentions"] = False
    for suffix_layer in decoder.layers[layer_index + 1 :]:
        hidden_states = suffix_layer(hidden_states, **kwargs)[0]
    hidden_states = decoder.norm(hidden_states)
    prompt_logits = causal_lm.lm_head(hidden_states)
    if len(state_cache.key_cache) != len(decoder.layers):
        raise RuntimeError("Refined prompt state did not populate every decoder cache layer")
    return RefinedPromptState(
        prompt_logits=prompt_logits,
        past_key_values=state_cache,
        refined_visual_state=replay.refined_visual_state,
        native_visual_state=replay.native_visual_state,
        visual_reconstruction_max_abs=replay.visual_reconstruction_max_abs,
        inserted_visual_max_abs=inserted_error,
        conditioning_edge_count=replay.conditioning_edge_count,
    )


def replay_compute_macs(
    *,
    sequence_length: int,
    hidden_size: int,
    intermediate_size: int,
    num_visual_tokens: int,
    num_key_value_heads: int,
    num_attention_heads: int,
) -> dict[str, int]:
    """Dense multiply-accumulate accounting for the frozen replay call.

    Q/K/V and attention are evaluated for the full common-padded prompt because
    the native layer call is reused unchanged. The layer MLP is likewise
    evaluated for every row; only visual output rows are retained.
    """
    if min(sequence_length, hidden_size, intermediate_size, num_visual_tokens) <= 0:
        raise ValueError("Replay dimensions must be positive")
    if num_attention_heads <= 0 or num_key_value_heads <= 0:
        raise ValueError("Attention head counts must be positive")
    if hidden_size % num_attention_heads:
        raise ValueError("Hidden size must be divisible by attention heads")
    head_dim = hidden_size // num_attention_heads
    kv_width = num_key_value_heads * head_dim
    q_projection = sequence_length * hidden_size * hidden_size
    kv_projection = 2 * sequence_length * hidden_size * kv_width
    output_projection = sequence_length * hidden_size * hidden_size
    attention_qk = sequence_length * sequence_length * hidden_size
    attention_av = sequence_length * sequence_length * hidden_size
    mlp = 3 * sequence_length * hidden_size * intermediate_size
    total = q_projection + kv_projection + output_projection + attention_qk + attention_av + mlp
    return {
        "q_projection": q_projection,
        "kv_projection": kv_projection,
        "output_projection": output_projection,
        "attention_qk": attention_qk,
        "attention_av": attention_av,
        "mlp": mlp,
        "total_macs": total,
        "total_flops_two_per_mac": 2 * total,
        "sequence_length": sequence_length,
        "visual_output_rows_retained": num_visual_tokens,
    }
