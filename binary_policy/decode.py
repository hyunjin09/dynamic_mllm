"""Deterministic decoding for direct and run-length binary policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaskCandidate:
    mask: tuple[int, ...]
    log_probability: float


def topk_factorized_masks(logits: torch.Tensor, top_k: int = 5) -> list[list[MaskCandidate]]:
    """Exact top-k masks under independent Bernoulli layer logits.

    Beam pruning is exact because each remaining bit contributes at most its
    locally better log-probability and every beam has the same suffix bound.
    Ties are resolved lexicographically for reproducibility.
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, L], got {tuple(logits.shape)}")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    log_on = torch.nn.functional.logsigmoid(logits).detach().cpu()
    log_off = torch.nn.functional.logsigmoid(-logits).detach().cpu()
    output: list[list[MaskCandidate]] = []
    for batch_idx in range(logits.shape[0]):
        beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
        for layer_idx in range(logits.shape[1]):
            expanded = []
            for prefix, score in beams:
                expanded.append((prefix + (0,), score + float(log_off[batch_idx, layer_idx])))
                expanded.append((prefix + (1,), score + float(log_on[batch_idx, layer_idx])))
            expanded.sort(key=lambda item: (-item[1], item[0]))
            beams = expanded[:top_k]
        output.append([MaskCandidate(mask=mask, log_probability=score) for mask, score in beams])
    return output


def decode_threshold(logits: torch.Tensor, threshold: float = 0.5) -> torch.BoolTensor:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly between zero and one")
    return torch.sigmoid(logits) >= threshold
