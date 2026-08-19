from __future__ import annotations

from dataclasses import dataclass

import torch

from interventions.prompt_cache import clone_dynamic_cache
from scoring.reference_likelihood import AnswerTokenScore, score_answer_token_logits


@dataclass(frozen=True)
class ContextualContinuation:
    prompt_text: str
    target_text: str
    literal_text: str
    prompt_token_ids: list[int]
    target_token_ids: list[int]
    target_token_count: int
    prompt_is_combined_prefix: bool
    decoded_text_exact: bool
    prompt_positions_contributing_to_score: int


def _token_ids(tokenizer, text: str) -> list[int]:
    raw = tokenizer(text, add_special_tokens=False).input_ids
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    return [int(token_id) for token_id in raw]


def _decode(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def contextual_continuation(
    tokenizer,
    prompt_text: str,
    continuation_text: str,
    expected_literal_text: str,
) -> ContextualContinuation:
    """Derive target IDs from the combined literal text at a stable boundary."""
    if prompt_text + continuation_text != expected_literal_text:
        raise ValueError("Prompt and continuation do not reconstruct the expected literal text")
    prompt_ids = _token_ids(tokenizer, prompt_text)
    combined_ids = _token_ids(tokenizer, expected_literal_text)
    if combined_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Contextual prompt is not a token prefix of the combined literal text")
    target_ids = combined_ids[len(prompt_ids) :]
    if not target_ids:
        raise ValueError("Contextual continuation tokenized to an empty target span")
    decoded_exact = _decode(tokenizer, combined_ids) == expected_literal_text
    if not decoded_exact:
        raise ValueError("Contextual token IDs do not decode to the exact literal text")
    return ContextualContinuation(
        prompt_text=prompt_text,
        target_text=continuation_text,
        literal_text=expected_literal_text,
        prompt_token_ids=prompt_ids,
        target_token_ids=target_ids,
        target_token_count=len(target_ids),
        prompt_is_combined_prefix=True,
        decoded_text_exact=True,
        prompt_positions_contributing_to_score=0,
    )


def score_reference_token_ids_from_prompt(
    causal_lm,
    prompt_logits: torch.Tensor,
    prompt_cache,
    prompt_attention_mask: torch.Tensor,
    target_token_ids: list[int],
) -> AnswerTokenScore:
    """Teacher-force explicit contextual target IDs after a cached prompt."""
    answer_ids = torch.tensor(
        target_token_ids,
        dtype=torch.long,
        device=prompt_logits.device,
    )
    if answer_ids.numel() < 1:
        raise ValueError("Contextual target token span is empty")

    continuation_logits = prompt_logits.new_empty((0, prompt_logits.shape[-1]))
    if answer_ids.numel() > 1:
        cache = clone_dynamic_cache(prompt_cache)
        continuation_ids = answer_ids[:-1].unsqueeze(0)
        prompt_length = int(prompt_attention_mask.shape[1])
        continuation_attention = torch.cat(
            [
                prompt_attention_mask,
                torch.ones_like(
                    continuation_ids,
                    device=prompt_attention_mask.device,
                ),
            ],
            dim=1,
        )
        cache_position = torch.arange(
            prompt_length,
            prompt_length + continuation_ids.shape[1],
            device=prompt_logits.device,
        )
        continuation = causal_lm(
            input_ids=continuation_ids,
            attention_mask=continuation_attention,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        continuation_logits = continuation.logits[0]
    return score_answer_token_logits(
        prompt_next_logits=prompt_logits[0, -1],
        continuation_logits=continuation_logits,
        answer_ids=answer_ids,
    )
