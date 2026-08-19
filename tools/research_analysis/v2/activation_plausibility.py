from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F


def deterministic_token_sample(states: torch.Tensor, maximum: int = 16) -> torch.Tensor:
    rows = states[0].float().detach().cpu()
    if rows.shape[0] <= maximum:
        return rows
    indices = torch.linspace(0, rows.shape[0] - 1, steps=maximum).round().long()
    return rows[indices]


def basic_activation_row(
    sample_id: str,
    benchmark: str,
    state_name: str,
    states: torch.Tensor,
    full_states: torch.Tensor,
    visual_mask: torch.Tensor,
) -> dict[str, Any]:
    current = states[0].float()
    full = full_states[0].float()
    visual = visual_mask[0]
    text = ~visual
    cosine = 1.0 - F.cosine_similarity(current, full, dim=-1)
    return {
        "sample_id": sample_id,
        "benchmark": benchmark,
        "state": state_name,
        "visual_token_norm_mean": float(current[visual].norm(dim=-1).mean().item()),
        "text_token_norm_mean": float(current[text].norm(dim=-1).mean().item()),
        "visual_rms_mean": float(current[visual].pow(2).mean(dim=-1).sqrt().mean().item()),
        "text_rms_mean": float(current[text].pow(2).mean(dim=-1).sqrt().mean().item()),
        "cosine_distance_to_full_mean": float(cosine.mean().item()),
        "cosine_distance_to_full_max": float(cosine.max().item()),
    }


def add_geometry_metrics(
    rows: list[dict[str, Any]],
    sampled_states: list[dict[str, Any]],
    rank: int = 16,
) -> None:
    natural_by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sampled_states:
        if item["state"] == "FULL":
            natural_by_benchmark[item["benchmark"]].append(item)

    bases: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    pooled_full: dict[str, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for benchmark, items in natural_by_benchmark.items():
        bank = torch.cat([item["tokens"] for item in items], dim=0).float()
        center = bank.mean(dim=0, keepdim=True)
        centered = bank - center
        q = min(rank, centered.shape[0] - 1, centered.shape[1])
        _, _, right = torch.pca_lowrank(centered, q=max(q, 1), center=False)
        bases[benchmark] = (center, right)
        pooled_full[benchmark] = [(item["sample_id"], item["tokens"].mean(dim=0)) for item in items]

    geometry: dict[tuple[str, str], tuple[float, float]] = {}
    for item in sampled_states:
        benchmark = item["benchmark"]
        tokens = item["tokens"].float()
        center, basis = bases[benchmark]
        centered = tokens - center
        projected = (centered @ basis) @ basis.T
        residual = (centered - projected).norm(dim=-1)
        denominator = centered.norm(dim=-1).clamp_min(1e-8)
        subspace_distance = float((residual / denominator).mean().item())

        pooled = tokens.mean(dim=0)
        neighbors = [
            vector
            for sample_id, vector in pooled_full[benchmark]
            if sample_id != item["sample_id"] or item["state"] != "FULL"
        ]
        if not neighbors:
            nearest = 0.0
        else:
            nearest = float(
                min((pooled - vector).norm().item() / (pooled.numel() ** 0.5) for vector in neighbors)
            )
        geometry[(item["sample_id"], item["state"])] = (subspace_distance, nearest)

    for row in rows:
        subspace, nearest = geometry[(row["sample_id"], row["state"])]
        row["pca_subspace_residual_ratio"] = subspace
        row["nearest_natural_full_rms_distance"] = nearest
