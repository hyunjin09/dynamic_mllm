"""Attention masks for DVR static visual-read execution."""

from __future__ import annotations

import torch


def _additive_mask(allowed: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
    return mask.masked_fill(~allowed.to(device=device), torch.finfo(dtype).min)


def make_text_causal_mask(
    text_orig_idx: torch.Tensor,
    text_valid_mask: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return additive `[B, 1, T, T]` text-only causal mask."""
    text_orig_idx = text_orig_idx.to(device)
    text_valid_mask = text_valid_mask.to(device)

    query_idx = text_orig_idx[:, :, None]
    key_idx = text_orig_idx[:, None, :]
    query_valid = text_valid_mask[:, :, None]
    key_valid = text_valid_mask[:, None, :]
    allowed = query_valid & key_valid & (key_idx <= query_idx)
    return _additive_mask(allowed[:, None, :, :], dtype=dtype, device=device)


def make_visual_read_mask(
    text_orig_idx: torch.Tensor,
    visual_orig_idx: torch.Tensor,
    text_valid_mask: torch.Tensor,
    visual_valid_mask: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return additive `[B, 1, T, V + T]` mask for key order `[visual, text]`."""
    text_orig_idx = text_orig_idx.to(device)
    visual_orig_idx = visual_orig_idx.to(device)
    text_valid_mask = text_valid_mask.to(device)
    visual_valid_mask = visual_valid_mask.to(device)

    query_idx = text_orig_idx[:, :, None]
    query_valid = text_valid_mask[:, :, None]

    visual_allowed = (
        query_valid
        & visual_valid_mask[:, None, :]
        & (visual_orig_idx[:, None, :] <= query_idx)
    )
    text_allowed = (
        query_valid
        & text_valid_mask[:, None, :]
        & (text_orig_idx[:, None, :] <= query_idx)
    )
    allowed = torch.cat([visual_allowed, text_allowed], dim=-1)
    return _additive_mask(allowed[:, None, :, :], dtype=dtype, device=device)
