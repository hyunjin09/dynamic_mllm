from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ContentScore:
    token_ids: list[int]
    token_logprobs: list[float]
    mean_logprob: float


def score_appended_content(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_length: int,
    answer_length: int,
) -> ContentScore:
    if answer_length < 1:
        raise ValueError("answer_length must be positive")
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("Expected logits [batch, sequence, vocab] and input_ids [batch, sequence]")
    start = prompt_length - 1
    stop = start + answer_length
    target_ids = input_ids[0, prompt_length : prompt_length + answer_length]
    selected_logits = logits[0, start:stop]
    if selected_logits.shape[0] != answer_length or target_ids.shape[0] != answer_length:
        raise ValueError("Logit/input lengths do not cover the requested answer span")
    token_logprobs = F.log_softmax(selected_logits.float(), dim=-1).gather(
        dim=-1, index=target_ids.unsqueeze(-1)
    ).squeeze(-1)
    return ContentScore(
        token_ids=[int(value) for value in target_ids.detach().cpu().tolist()],
        token_logprobs=[float(value) for value in token_logprobs.detach().cpu().tolist()],
        mean_logprob=float(token_logprobs.mean().item()),
    )


def answer_margin(correct_score: float, distractor_scores: list[float]) -> float:
    if not distractor_scores:
        raise ValueError("At least one distractor score is required")
    return float(correct_score - max(distractor_scores))
