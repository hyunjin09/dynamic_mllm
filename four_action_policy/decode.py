"""Deterministic decoding for factorized four-action policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .actions import decode_action_indices


@dataclass(frozen=True)
class RouteCandidate:
    action_indices: tuple[int, ...]
    actions: tuple[str, ...]
    log_probability: float


def decode_argmax(logits: torch.Tensor) -> torch.LongTensor:
    if logits.ndim != 3 or logits.shape[-1] != 4:
        raise ValueError("logits must have shape [B, L, 4]")
    return logits.argmax(dim=-1)


def topk_factorized_routes(
    logits: torch.Tensor, *, top_k: int = 5
) -> list[list[RouteCandidate]]:
    """Return the exact top-k complete routes under layerwise categoricals."""
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if logits.ndim != 3 or logits.shape[-1] != 4:
        raise ValueError("logits must have shape [B, L, 4]")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    log_probs = F.log_softmax(logits, dim=-1).detach().cpu()
    output: list[list[RouteCandidate]] = []
    for batch_index in range(logits.shape[0]):
        beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
        for layer_index in range(logits.shape[1]):
            expanded = [
                (
                    prefix + (action_index,),
                    score + float(log_probs[batch_index, layer_index, action_index]),
                )
                for prefix, score in beams
                for action_index in range(4)
            ]
            expanded.sort(key=lambda item: (-item[1], item[0]))
            beams = expanded[:top_k]
        output.append(
            [
                RouteCandidate(
                    action_indices=indices,
                    actions=decode_action_indices(indices),
                    log_probability=score,
                )
                for indices, score in beams
            ]
        )
    return output
