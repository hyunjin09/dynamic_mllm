from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from scoring.benchmark_metrics import normalize_exact, normalize_textvqa
from interventions.prompt_cache import clone_dynamic_cache


@dataclass(frozen=True)
class AcceptedAnswer:
    text: str
    weight: float


@dataclass(frozen=True)
class AnswerTokenScore:
    token_ids: list[int]
    token_logprobs: list[float]
    sequence_logprob: float
    mean_logprob: float


def accepted_answers(record: Mapping) -> list[AcceptedAnswer]:
    benchmark = str(record["benchmark"])
    if benchmark == "gqa":
        canonical = normalize_exact(str(record["answer"]))
        if not canonical:
            raise ValueError("GQA canonical answer is empty after normalization")
        return [AcceptedAnswer(canonical, 1.0)]
    if benchmark != "textvqa":
        raise ValueError(f"Unsupported Stage B benchmark: {benchmark}")

    raw_answers = list(record.get("all_answer_norms") or [record["answer"]])
    normalized = [normalize_textvqa(str(answer)) for answer in raw_answers]
    counts = Counter(answer for answer in normalized if answer)
    if not counts:
        raise ValueError("TextVQA accepted-answer set is empty after normalization")
    total = sum(counts.values())
    return [
        AcceptedAnswer(text=answer, weight=count / total)
        for answer, count in counts.items()
    ]


def weighted_logsumexp(scores: list[float], weights: list[float]) -> float:
    if not scores or len(scores) != len(weights):
        raise ValueError("Scores and weights must be non-empty and have equal length")
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("Every accepted-answer weight must be positive")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Accepted-answer weights must sum to one")
    terms = [score + math.log(weight) for score, weight in zip(scores, weights)]
    maximum = max(terms)
    return maximum + math.log(sum(math.exp(term - maximum) for term in terms))


def score_answer_token_logits(
    prompt_next_logits: torch.Tensor,
    continuation_logits: torch.Tensor,
    answer_ids: torch.Tensor,
) -> AnswerTokenScore:
    if prompt_next_logits.ndim != 1 or continuation_logits.ndim != 2 or answer_ids.ndim != 1:
        raise ValueError("Expected prompt logits [vocab], continuation [T-1,vocab], answer IDs [T]")
    if answer_ids.numel() < 1:
        raise ValueError("Reference answer must contain at least one token")
    if continuation_logits.shape[0] != answer_ids.numel() - 1:
        raise ValueError("Continuation logits must contain exactly T-1 rows")

    rows = torch.cat([prompt_next_logits.unsqueeze(0), continuation_logits], dim=0)
    token_logprobs = F.log_softmax(rows.float(), dim=-1).gather(
        dim=-1, index=answer_ids.to(rows.device).unsqueeze(-1)
    ).squeeze(-1)
    sequence = float(token_logprobs.sum().item())
    return AnswerTokenScore(
        token_ids=[int(value) for value in answer_ids.detach().cpu().tolist()],
        token_logprobs=[float(value) for value in token_logprobs.detach().cpu().tolist()],
        sequence_logprob=sequence,
        mean_logprob=sequence / int(answer_ids.numel()),
    )


def factorial_effects(state_scores: Mapping[str, float]) -> dict[str, float]:
    required = {"IGNORE", "READ_ONLY", "WRITE_ONLY", "FULL"}
    if set(state_scores) != required:
        raise ValueError(f"State scores must be exactly {sorted(required)}")
    ignore = float(state_scores["IGNORE"])
    read_only = float(state_scores["READ_ONLY"])
    write_only = float(state_scores["WRITE_ONLY"])
    full = float(state_scores["FULL"])
    return {
        "read_w0": read_only - ignore,
        "read_w1": full - write_only,
        "write_r0": write_only - ignore,
        "write_r1": full - read_only,
        "interaction": full - read_only - write_only + ignore,
    }


def score_reference_from_prompt(
    causal_lm,
    tokenizer,
    prompt_logits: torch.Tensor,
    prompt_cache,
    prompt_attention_mask: torch.Tensor,
    answer: str,
) -> AnswerTokenScore:
    answer_ids = tokenizer(
        answer, add_special_tokens=False, return_tensors="pt"
    ).input_ids[0].to(prompt_logits.device)
    if answer_ids.numel() < 1:
        raise ValueError(f"Reference answer tokenized to empty content: {answer!r}")

    continuation_logits = prompt_logits.new_empty((0, prompt_logits.shape[-1]))
    if answer_ids.numel() > 1:
        cache = clone_dynamic_cache(prompt_cache)
        continuation_ids = answer_ids[:-1].unsqueeze(0)
        prompt_length = int(prompt_attention_mask.shape[1])
        continuation_attention = torch.cat(
            [
                prompt_attention_mask,
                torch.ones_like(continuation_ids, device=prompt_attention_mask.device),
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


def aggregate_accepted_scores(
    answers: list[AcceptedAnswer], scores: list[AnswerTokenScore]
) -> dict[str, float]:
    if len(answers) != len(scores):
        raise ValueError("Accepted answers and scores must align")
    weights = [answer.weight for answer in answers]
    return {
        "sequence_logprob": weighted_logsumexp(
            [score.sequence_logprob for score in scores], weights
        ),
        "mean_logprob": weighted_logsumexp(
            [score.mean_logprob for score in scores], weights
        ),
    }
