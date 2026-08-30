"""Shared supervision helpers for the matched persistent-corrective study."""

from __future__ import annotations

from hashlib import sha256
import random
from typing import Any, Sequence

import torch

from four_action_online_router.supervision import set_valued_action_loss


def _seed_value(*parts: object) -> int:
    return int.from_bytes(
        sha256(":".join(str(part) for part in parts).encode()).digest()[:8]
    )


def matched_epoch_indices(
    rows: Sequence[dict[str, Any]], *, seed: int, epoch: int
) -> list[int]:
    """Visit every fixed-subset row once, alternating W2C and C2C."""

    pools = {
        route_type: [
            index
            for index, row in enumerate(rows)
            if str(row.get("route_type")) == route_type
        ]
        for route_type in ("W2C", "C2C")
    }
    if not pools["W2C"] or len(pools["W2C"]) != len(pools["C2C"]):
        raise ValueError("matched epoch sampling requires equal nonempty W2C and C2C pools")
    for route_type, pool in pools.items():
        random.Random(_seed_value(seed, epoch, route_type)).shuffle(pool)
    order = []
    for w2c, c2c in zip(pools["W2C"], pools["C2C"]):
        order.extend((w2c, c2c))
    if len(order) != len(rows) or len(set(order)) != len(rows):
        raise RuntimeError("matched epoch order does not cover the fixed subset exactly")
    return order


def persistent_boundary_loss(
    logits: torch.Tensor,
    *,
    boundary_layers: torch.Tensor,
    valid_actions: torch.BoolTensor,
    present: torch.BoolTensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Apply set-valued loss at each selected W2C mandatory boundary."""

    if logits.ndim != 3 or logits.shape[-1] != 4:
        raise ValueError("boundary logits must have shape [batch, layers, 4]")
    batch_size, num_layers, _ = logits.shape
    layers = boundary_layers.to(device=logits.device, dtype=torch.long)
    masks = valid_actions.to(device=logits.device, dtype=torch.bool)
    selected = present.to(device=logits.device, dtype=torch.bool)
    if layers.shape != (batch_size,) or masks.shape != (batch_size, 4):
        raise ValueError("boundary layers/masks must have shape [batch] and [batch, 4]")
    if selected.shape != (batch_size,) or not bool(selected.any().item()):
        raise ValueError("persistent boundary loss requires at least one selected W2C row")
    if bool(((layers[selected] < 0) | (layers[selected] >= num_layers)).any().item()):
        raise ValueError("selected boundary layer lies outside the policy")
    if bool(masks[selected, 3].any().item()):
        raise ValueError("FULL cannot be valid at a mandatory boundary")
    if not bool(masks[selected].any(dim=-1).all().item()):
        raise ValueError("every selected boundary needs a nonempty valid-action set")
    selected_logits = logits[selected]
    selected_layers = layers[selected]
    positions = torch.arange(selected_logits.shape[0], device=logits.device)
    return set_valued_action_loss(
        selected_logits[positions, selected_layers], masks[selected], reduction=reduction
    )


def _behavior_key(row: dict[str, Any]) -> tuple[float, float, float, int, float, int]:
    execution = row["execution"]
    return (
        float(execution["w2c_rescue_rate"]),
        float(execution["c2c_preservation_rate"]),
        float(execution["overall_routed_accuracy"]),
        -int(execution["c2c_regressions"]),
        -float(execution["mean_action_layers"]["FULL"]),
        -int(row["epoch"]),
    )


def select_behavioral_checkpoint(
    history: Sequence[dict[str, Any]], *, c2c_threshold: float = 0.95
) -> dict[str, Any]:
    """Apply the prospective rescue-under-preservation checkpoint rule."""

    if not history or not 0.0 <= c2c_threshold <= 1.0:
        raise ValueError("behavioral selection needs history and a valid C2C threshold")
    eligible = [
        row
        for row in history
        if float(row["execution"]["c2c_preservation_rate"]) >= c2c_threshold
    ]
    selected = max(eligible, key=_behavior_key) if eligible else None
    frontier = []
    for candidate in history:
        rescue = float(candidate["execution"]["w2c_rescue_rate"])
        preservation = float(candidate["execution"]["c2c_preservation_rate"])
        dominated = any(
            (
                float(other["execution"]["w2c_rescue_rate"]) >= rescue
                and float(other["execution"]["c2c_preservation_rate"])
                >= preservation
                and (
                    float(other["execution"]["w2c_rescue_rate"]) > rescue
                    or float(other["execution"]["c2c_preservation_rate"])
                    > preservation
                )
            )
            for other in history
        )
        if not dominated:
            frontier.append(candidate)
    frontier.sort(key=lambda row: int(row["epoch"]))
    return {
        "c2c_preservation_threshold": float(c2c_threshold),
        "eligible_epochs": [int(row["epoch"]) for row in eligible],
        "selected_epoch": int(selected["epoch"]) if selected is not None else None,
        "pareto_frontier_epochs": [int(row["epoch"]) for row in frontier],
        "pareto_frontier": [
            {
                "epoch": int(row["epoch"]),
                "w2c_rescue_rate": float(row["execution"]["w2c_rescue_rate"]),
                "c2c_preservation_rate": float(
                    row["execution"]["c2c_preservation_rate"]
                ),
            }
            for row in frontier
        ],
    }


def paired_bootstrap_rate_difference(
    left: Sequence[bool],
    right: Sequence[bool],
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap the paired right-minus-left binary success-rate difference."""

    if not left or len(left) != len(right) or draws < 1:
        raise ValueError("paired bootstrap requires equal nonempty rows and positive draws")
    differences = [float(r) - float(l) for l, r in zip(left, right)]
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        samples.append(
            sum(differences[rng.randrange(len(differences))] for _ in differences)
            / len(differences)
        )
    samples.sort()
    low_index = int((draws - 1) * 0.025)
    high_index = int((draws - 1) * 0.975)
    return {
        "records": len(differences),
        "draws": int(draws),
        "seed": int(seed),
        "right_minus_left": sum(differences) / len(differences),
        "ci_low": samples[low_index],
        "ci_high": samples[high_index],
    }
