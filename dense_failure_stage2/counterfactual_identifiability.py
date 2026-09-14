"""Pure contracts for one-step READ/WRITE effect identifiability."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


ACTIONS = ("FULL", "WRITE_ONLY", "READ_ONLY")
CONDITIONS = ("pre", "full_post", "off_post", "pair", "delta", "pair_plus_delta")


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    payload = tensor.view(torch.uint8).numpy().tobytes()
    metadata = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode()
    return sha256(metadata + payload).hexdigest()


def state_sha256(text: torch.Tensor, visual: torch.Tensor) -> str:
    return sha256(
        f"text:{tensor_sha256(text)}|visual:{tensor_sha256(visual)}".encode()
    ).hexdigest()


def pool_branch(text: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
    """Return [last text ; mean visual] for compact, valid token tensors."""

    if text.ndim != 3 or visual.ndim != 3 or text.shape[0] != 1 or visual.shape[0] != 1:
        raise ValueError("branch tensors must be [1,tokens,hidden]")
    if text.shape[-1] != visual.shape[-1] or text.shape[1] < 1 or visual.shape[1] < 1:
        raise ValueError("branch tensors have incompatible or empty token axes")
    return torch.cat((text[0, -1].float(), visual[0].float().mean(dim=0))).to(torch.bfloat16)


def feature_width(condition: str, hidden_size: int) -> int:
    base = 2 * int(hidden_size)
    if condition in {"pre", "full_post", "off_post", "delta"}:
        return base
    if condition == "pair":
        return 2 * base
    if condition == "pair_plus_delta":
        return 3 * base
    if condition in {"text_delta", "visual_delta"}:
        return int(hidden_size)
    if condition == "text_visual_delta":
        return base
    raise ValueError(f"unsupported feature condition: {condition}")


def construct_summary_feature(
    summaries: torch.Tensor,
    *,
    target: str,
    condition: str,
    paired_off_summaries: torch.Tensor | None = None,
    swapped: bool = False,
) -> torch.Tensor:
    """Construct one prospectively named condition from [N,4,2H] summaries.

    Summary slots are PRE, FULL, WRITE_ONLY, READ_ONLY.  A random-pair control
    supplies ``paired_off_summaries`` while preserving the row's FULL state.
    """

    if summaries.ndim != 3 or summaries.shape[1] != 4 or summaries.shape[2] % 2:
        raise ValueError("summaries must be [rows,4,2*hidden]")
    target = str(target)
    if target not in {"read", "write"}:
        raise ValueError("target must be read or write")
    pre = summaries[:, 0].float()
    full = summaries[:, 1].float()
    off_index = 2 if target == "read" else 3
    off = summaries[:, off_index].float()
    if paired_off_summaries is not None:
        if paired_off_summaries.shape != off.shape:
            raise ValueError("random-pair OFF summaries do not align")
        off = paired_off_summaries.float()
    delta = full - off
    if condition == "pre":
        return pre
    if condition == "full_post":
        return full
    if condition == "off_post":
        return off
    if condition == "pair":
        return torch.cat((off, full), dim=-1) if swapped else torch.cat((full, off), dim=-1)
    if condition == "delta":
        return -delta if swapped else delta
    if condition == "pair_plus_delta":
        return (
            torch.cat((off, full, -delta), dim=-1)
            if swapped
            else torch.cat((full, off, delta), dim=-1)
        )
    hidden = summaries.shape[-1] // 2
    if condition == "text_delta":
        return delta[:, :hidden]
    if condition == "visual_delta":
        return delta[:, hidden:]
    if condition == "text_visual_delta":
        return delta
    raise ValueError(f"unsupported feature condition: {condition}")


def matched_random_pair_assignment(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> tuple[np.ndarray, list[str]]:
    """Pair with another UID using the strongest feasible nuisance match."""

    cells: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    keys_by_index: list[list[tuple[Any, ...]]] = []
    tier_names = (
        "layer_dataset_source_outcome",
        "layer_dataset_outcome",
        "layer_dataset",
        "layer",
        "dataset_source_outcome",
        "dataset_outcome",
        "dataset",
        "global",
    )
    for index, row in enumerate(rows):
        layer = int(row["layer"])
        dataset = str(row["dataset"])
        source = str(row.get("source_regime", ""))
        wrong = bool(row["dense_wrong"])
        keys = [
            (0, layer, dataset, source, wrong),
            (1, layer, dataset, wrong),
            (2, layer, dataset),
            (3, layer),
            (4, dataset, source, wrong),
            (5, dataset, wrong),
            (6, dataset),
            (7,),
        ]
        keys_by_index.append(keys)
        for key in keys:
            cells[key].append(index)
    output = np.full(len(rows), -1, dtype=np.int64)
    tiers: list[str] = [""] * len(rows)
    for index, keys in enumerate(keys_by_index):
        for tier, key in enumerate(keys):
            candidates = [
                candidate for candidate in cells[key]
                if str(rows[candidate]["uid"]) != str(rows[index]["uid"])
            ]
            if not candidates:
                continue
            output[index] = min(
                candidates,
                key=lambda candidate: sha256(
                    f"{int(seed)}|{rows[index]['state_id']}|{rows[candidate]['state_id']}".encode()
                ).hexdigest()
                if tier < 4
                else (
                    f"{abs(int(rows[candidate]['layer']) - int(rows[index]['layer'])):03d}|"
                    + sha256(
                        f"{int(seed)}|{rows[index]['state_id']}|{rows[candidate]['state_id']}".encode()
                    ).hexdigest()
                ),
            )
            tiers[index] = tier_names[tier]
            break
    if np.any(output < 0):
        raise ValueError("random-pair construction lacks another UID")
    return output, tiers


def matched_random_pair_indices(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> np.ndarray:
    return matched_random_pair_assignment(rows, seed=seed)[0]


def validate_prediction_census(expected_state_ids: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    expected = [str(value) for value in expected_state_ids]
    observed = [str(row["state_id"]) for row in rows]
    if len(expected) != len(set(expected)):
        raise ValueError("expected state IDs are duplicated")
    if len(observed) != len(set(observed)):
        raise ValueError("prediction state IDs are duplicated")
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise ValueError(f"prediction census differs: missing={missing[:3]} extra={extra[:3]}")


class PairedTokenComparator(nn.Module):
    """Lightweight shared token-aware encoder for FULL/OFF post-state pairs."""

    def __init__(
        self,
        *,
        hidden_size: int,
        projection_size: int,
        attention_heads: int,
        readout_hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if projection_size % attention_heads:
            raise ValueError("projection size must be divisible by attention heads")
        self.hidden_size = int(hidden_size)
        self.projection_size = int(projection_size)
        self.text_projection = nn.Linear(hidden_size, projection_size)
        self.visual_projection = nn.Linear(hidden_size, projection_size)
        self.attention = nn.MultiheadAttention(
            projection_size, attention_heads, dropout=dropout, batch_first=True
        )
        self.readout = nn.Sequential(
            nn.Linear(6 * projection_size, readout_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(readout_hidden_size, 1),
        )

    def _encode(
        self,
        text: torch.Tensor,
        visual: torch.Tensor,
        text_mask: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> torch.Tensor:
        if text.ndim != 3 or visual.ndim != 3:
            raise ValueError("token comparator inputs must be rank three")
        if text.shape[-1] != self.hidden_size or visual.shape[-1] != self.hidden_size:
            raise ValueError("token comparator hidden width differs")
        if text_mask.shape != text.shape[:2] or visual_mask.shape != visual.shape[:2]:
            raise ValueError("token comparator masks differ")
        if not bool(text_mask.any(dim=1).all()) or not bool(visual_mask.any(dim=1).all()):
            raise ValueError("every comparator row needs text and visual tokens")
        dtype = self.text_projection.weight.dtype
        text = text.to(dtype=dtype)
        visual = visual.to(dtype=dtype)
        last = text_mask.long().sum(dim=1) - 1
        query = text[torch.arange(len(text), device=text.device), last]
        query = self.text_projection(query).unsqueeze(1)
        values = self.visual_projection(visual)
        attended, _ = self.attention(
            query, values, values, key_padding_mask=~visual_mask.bool(), need_weights=False
        )
        return torch.cat((query[:, 0], attended[:, 0]), dim=-1)

    def forward(
        self,
        full_text: torch.Tensor,
        full_visual: torch.Tensor,
        off_text: torch.Tensor,
        off_visual: torch.Tensor,
        *,
        text_mask: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> torch.Tensor:
        full = self._encode(full_text, full_visual, text_mask, visual_mask)
        off = self._encode(off_text, off_visual, text_mask, visual_mask)
        return self.readout(torch.cat((full, off, full - off), dim=-1)).squeeze(-1)


def classify_case(
    *,
    pre: float,
    full_post: float,
    off_post: float,
    pair: float,
    delta: float,
    pair_plus_delta: float,
    token: float,
    random_pair_best: float,
    thresholds: Mapping[str, float],
) -> tuple[str, str]:
    """Apply the frozen A/B/C/D decision hierarchy."""

    weak = float(thresholds["weak_absolute_spearman_below"])
    strong = float(thresholds["strong_spearman_at_least"])
    gain_pre = float(thresholds["material_gain_over_pre"])
    gain_single = float(thresholds["unique_pair_gain_over_best_single"])
    gain_token = float(thresholds["token_gain_over_best_pooled"])
    collapse = float(thresholds["random_pair_collapse_gap"])
    best_single = max(full_post, off_post)
    best_counterfactual = max(pair, delta, pair_plus_delta)
    best_pooled = max(pre, best_single, best_counterfactual)
    if token >= strong and token - best_pooled >= gain_token:
        return "C", "token-aware pair materially exceeds every pooled condition"
    if (
        pre < weak
        and best_single < strong
        and best_counterfactual >= strong
        and best_counterfactual - pre >= gain_pre
        and best_counterfactual - best_single >= gain_single
        and best_counterfactual - random_pair_best >= collapse
    ):
        return "A", "within-state pair/delta uniquely and materially identifies utility"
    if best_single >= strong and best_single - pre >= gain_pre and best_counterfactual - best_single < gain_single:
        return "B", "one post-state explains essentially all of the counterfactual gain"
    return "D", "one-step conditions remain weak or the positive pattern is incomplete"
