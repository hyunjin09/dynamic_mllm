"""Causal masks for full and compacted-text layer execution."""

from __future__ import annotations

import torch


def additive_causal_mask(
    valid_mask: torch.Tensor,
    original_indices: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    valid = valid_mask.to(device=device, dtype=torch.bool)
    indices = original_indices.to(device=device)
    allowed = valid[:, :, None] & valid[:, None, :] & (indices[:, None, :] <= indices[:, :, None])
    output = torch.zeros((valid.shape[0], 1, valid.shape[1], valid.shape[1]), dtype=dtype, device=device)
    return output.masked_fill(~allowed[:, None], torch.finfo(dtype).min)


def full_causal_mask(attention_mask: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    valid = attention_mask.to(device=device, dtype=torch.bool)
    seq_len = valid.shape[1]
    indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(valid.shape[0], -1)
    return additive_causal_mask(valid, indices, dtype=dtype, device=device)
