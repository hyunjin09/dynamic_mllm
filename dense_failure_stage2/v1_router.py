"""Small shared Stage-2 READ/WRITE router and frozen V1 sampling contracts."""

from __future__ import annotations

from collections import Counter
import random
from typing import Any, Mapping, Sequence

import torch
from torch import nn


ACTION_NAMES = ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")
ACTION_TO_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}


class SharedReadWriteRouter(nn.Module):
    """Shared, depth-agnostic router over the current routed token states."""

    def __init__(
        self,
        *,
        hidden_size: int,
        router_size: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or router_size < 1:
            raise ValueError("hidden and router sizes must be positive")
        if router_size % num_heads:
            raise ValueError("router size must be divisible by the number of heads")
        self.hidden_size = int(hidden_size)
        self.router_size = int(router_size)
        self.text_query_projection = nn.Linear(hidden_size, router_size)
        self.visual_projection = nn.Linear(hidden_size, router_size)
        self.read_attention = nn.MultiheadAttention(
            router_size, num_heads, dropout=dropout, batch_first=True
        )
        self.write_attention = nn.MultiheadAttention(
            router_size, num_heads, dropout=dropout, batch_first=True
        )
        self.write_query = nn.Parameter(torch.empty(1, 1, router_size))
        nn.init.normal_(self.write_query, std=router_size**-0.5)
        self.action_head = nn.Sequential(
            nn.Linear(2 * router_size, router_size),
            nn.GELU(),
            nn.LayerNorm(router_size),
            nn.Dropout(dropout),
            nn.Linear(router_size, len(ACTION_NAMES)),
        )

    def forward(
        self,
        text_states: torch.Tensor,
        visual_states: torch.Tensor,
        *,
        text_mask: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> torch.Tensor:
        if text_states.ndim != 3 or visual_states.ndim != 3:
            raise ValueError("text and visual states must have [batch, tokens, hidden] shape")
        if text_states.shape[0] != visual_states.shape[0]:
            raise ValueError("text and visual batch sizes differ")
        if text_states.shape[-1] != self.hidden_size or visual_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden-state width differs from the frozen router contract")
        if text_mask.shape != text_states.shape[:2] or visual_mask.shape != visual_states.shape[:2]:
            raise ValueError("mask shape differs from its token-state shape")
        text_mask = text_mask.bool()
        visual_mask = visual_mask.bool()
        if not bool(text_mask.any(dim=1).all()):
            raise ValueError("every sample must contain a valid text/control token")
        if not bool(visual_mask.any(dim=1).all()):
            raise ValueError("every sample must contain a valid visual mask row")

        # Qwen stays frozen in BF16.  The small trainable router is explicitly
        # FP32 so rare high-magnitude routed states cannot silently overflow an
        # autocast attention/backward path.
        compute_dtype = self.text_query_projection.weight.dtype
        text_states = text_states.to(dtype=compute_dtype)
        visual_states = visual_states.to(dtype=compute_dtype)

        batch = text_states.shape[0]
        last = text_mask.long().sum(dim=1) - 1
        q = text_states[torch.arange(batch, device=text_states.device), last]
        q = self.text_query_projection(q).unsqueeze(1)
        visual = self.visual_projection(visual_states)
        padding = ~visual_mask
        read, _ = self.read_attention(q, visual, visual, key_padding_mask=padding, need_weights=False)
        write_query = self.write_query.expand(batch, -1, -1)
        write, _ = self.write_attention(
            write_query, visual, visual, key_padding_mask=padding, need_weights=False
        )
        return self.action_head(torch.cat((read[:, 0], write[:, 0]), dim=-1))


def _validate_route(actions: Sequence[str], trigger_layer: int) -> tuple[list[str], int]:
    normalized = [str(action).upper() for action in actions]
    if not normalized or any(action not in ACTION_TO_INDEX for action in normalized):
        raise ValueError("route contains an unsupported action")
    trigger = int(trigger_layer)
    if trigger < 0 or trigger >= len(normalized):
        raise ValueError("trigger layer is outside the route")
    non_full = [index for index in range(trigger, len(normalized)) if normalized[index] != "FULL"]
    if len(non_full) != 1:
        raise ValueError("Corpus-B V1 routes must contain exactly one post-trigger correction")
    return normalized, non_full[0]


def sample_w_layers(
    actions: Sequence[str], *, trigger_layer: int, rng: random.Random
) -> list[int]:
    """Positive-anchored Random-4: correction, pre-FULL, post-FULL, other FULL."""
    normalized, correction = _validate_route(actions, trigger_layer)
    trigger = int(trigger_layer)
    before = [index for index in range(trigger, correction) if normalized[index] == "FULL"]
    after = [index for index in range(correction + 1, len(normalized)) if normalized[index] == "FULL"]
    selected = [correction]
    if before:
        selected.append(rng.choice(before))
    if after:
        selected.append(rng.choice(after))
    remaining = [
        index
        for index in range(trigger, len(normalized))
        if normalized[index] == "FULL" and index not in selected
    ]
    while remaining and len(selected) < 4:
        chosen = rng.choice(remaining)
        selected.append(chosen)
        remaining.remove(chosen)
    return sorted(selected)


def sample_c_layers(
    *, trigger_layer: int, num_layers: int, rng: random.Random
) -> list[int]:
    trigger, layers = int(trigger_layer), int(num_layers)
    if trigger < 0 or trigger >= layers:
        raise ValueError("trigger layer is outside the model")
    eligible = list(range(trigger, layers))
    return sorted(rng.sample(eligible, k=min(4, len(eligible))))


def build_epoch_draws(
    w_routes_by_uid: Mapping[str, Sequence[Mapping[str, Any]]],
    c_uids: Sequence[str],
    *,
    c_draws: int,
    seed: int,
    epoch: int,
) -> list[dict[str, Any]]:
    """Freeze one sample-balanced epoch with one route per W UID."""
    if not w_routes_by_uid or not c_uids or c_draws < 1:
        raise ValueError("both corpora and a positive C draw count are required")
    rng = random.Random((int(seed) << 16) + int(epoch))
    draws: list[dict[str, Any]] = []
    for uid in sorted(w_routes_by_uid):
        routes = list(w_routes_by_uid[uid])
        if not routes:
            raise ValueError(f"W UID has no successful routes: {uid}")
        draws.append({"kind": "W", "uid": uid, "route": dict(rng.choice(routes))})
    ordered_c = sorted(map(str, c_uids))
    for index in range(int(c_draws)):
        if index % len(ordered_c) == 0:
            rng.shuffle(ordered_c)
        draws.append({"kind": "C", "uid": ordered_c[index % len(ordered_c)]})
    rng.shuffle(draws)
    return draws


def summarize_action_predictions(
    *, targets: Sequence[int], predictions: Sequence[int]
) -> dict[str, Any]:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("targets and predictions must be equally sized and nonempty")
    target_counts, predicted_counts = Counter(map(int, targets)), Counter(map(int, predictions))
    correct = Counter(
        target for target, prediction in zip(targets, predictions) if int(target) == int(prediction)
    )
    recalls = {
        action: (correct[index] / target_counts[index] if target_counts[index] else None)
        for index, action in enumerate(ACTION_NAMES)
    }
    non_full_total = sum(target_counts[index] for index in range(1, len(ACTION_NAMES)))
    non_full_correct = sum(correct[index] for index in range(1, len(ACTION_NAMES)))
    total = len(targets)
    return {
        "accuracy": sum(correct.values()) / total,
        "recall": recalls,
        "non_full_recall": non_full_correct / non_full_total if non_full_total else None,
        "predicted_distribution": {
            action: predicted_counts[index] / total for index, action in enumerate(ACTION_NAMES)
        },
        "target_distribution": {
            action: target_counts[index] / total for index, action in enumerate(ACTION_NAMES)
        },
    }
