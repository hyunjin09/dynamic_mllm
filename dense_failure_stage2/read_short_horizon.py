"""Pure contracts for short-horizon READ counterfactual propagation."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dense_failure_stage2.counterfactual_identifiability import (
    matched_random_pair_indices as _matched_state_pairs,
)


HORIZONS = (1, 2, 4, 8)
BRANCHES = ("ON", "OFF")
POOLED_CONDITIONS = ("on", "off", "pair", "delta", "pair_plus_delta")
DELTA_CONDITIONS = ("text_delta", "visual_delta", "text_visual_delta")


def eligible_horizons(
    layer: int,
    num_layers: int,
    horizons: Sequence[int] = HORIZONS,
) -> tuple[int, ...]:
    """Return frozen horizons whose final executed layer is in bounds."""

    layer = int(layer)
    num_layers = int(num_layers)
    if layer < 0 or layer >= num_layers:
        raise ValueError("intervention layer is outside decoder bounds")
    result = tuple(int(horizon) for horizon in horizons if layer + int(horizon) <= num_layers)
    if not result or result[0] != 1:
        raise ValueError("every valid state must support H=1")
    return result


def expected_action_trace(horizon: int, first_action: str) -> tuple[str, ...]:
    """Return the only allowed branch trace through one frozen horizon."""

    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    action = str(first_action).upper()
    if action not in {"FULL", "WRITE_ONLY"}:
        raise ValueError("branch action must be FULL or WRITE_ONLY")
    return (action, *("FULL" for _ in range(horizon - 1)))


def pool_horizon_state(text: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
    """Return `[last text/control; mean visual]` for compact valid states."""

    if text.ndim != 3 or visual.ndim != 3 or text.shape[0] != 1 or visual.shape[0] != 1:
        raise ValueError("horizon tensors must be [1,tokens,hidden]")
    if text.shape[-1] != visual.shape[-1] or text.shape[1] < 1 or visual.shape[1] < 1:
        raise ValueError("horizon tensors have incompatible or empty token axes")
    return torch.cat((text[0, -1].float(), visual[0].float().mean(dim=0)))


def construct_horizon_feature(
    on: torch.Tensor,
    off: torch.Tensor,
    condition: str,
) -> torch.Tensor:
    """Construct a pooled condition with the frozen ON-minus-OFF sign."""

    if on.shape != off.shape or on.ndim != 2:
        raise ValueError("ON/OFF pooled features must be aligned matrices")
    condition = str(condition)
    delta = on - off
    if condition == "on":
        return on
    if condition == "off":
        return off
    if condition == "pair":
        return torch.cat((on, off), dim=-1)
    if condition == "delta":
        return delta
    if condition == "pair_plus_delta":
        return torch.cat((on, off, delta), dim=-1)
    hidden = on.shape[-1] // 2
    if on.shape[-1] % 2:
        raise ValueError("pooled state must contain equal text and visual blocks")
    if condition == "text_delta":
        return delta[:, :hidden]
    if condition == "visual_delta":
        return delta[:, hidden:]
    if condition == "text_visual_delta":
        return delta
    raise ValueError(f"unsupported horizon feature condition: {condition}")


def validate_horizon_census(
    expected_states: Sequence[Mapping[str, Any]],
    observed: Sequence[Mapping[str, Any]],
    *,
    num_layers: int,
) -> None:
    """Require exactly one ON and OFF record for every eligible state/horizon."""

    expected_ids = [str(row["state_id"]) for row in expected_states]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("expected state IDs are duplicated")
    keys = [
        (str(row["state_id"]), int(row["horizon"]), str(row["branch"]).upper())
        for row in observed
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("observed horizon records are duplicated")
    expected = {
        (str(row["state_id"]), horizon, branch)
        for row in expected_states
        for horizon in eligible_horizons(int(row["layer"]), int(num_layers))
        for branch in BRANCHES
    }
    if set(keys) != expected:
        missing = sorted(expected - set(keys))
        extra = sorted(set(keys) - expected)
        raise ValueError(f"horizon census differs: missing={missing[:3]} extra={extra[:3]}")


def validate_order_invariant_hashes(
    forward: Mapping[tuple[str, int], str],
    reversed_order: Mapping[tuple[str, int], str],
) -> None:
    """Reject any fresh-cache branch-order dependence."""

    if dict(forward) != dict(reversed_order):
        keys = sorted(set(forward).union(reversed_order))
        changed = [key for key in keys if forward.get(key) != reversed_order.get(key)]
        raise ValueError(f"rollout order changed horizon hashes: {changed[:3]}")


def _stable(seed: int, *parts: object) -> str:
    return sha256("|".join((str(seed), *(str(part) for part in parts))).encode()).hexdigest()


def matched_random_pair_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    uid_permutation: bool = False,
) -> np.ndarray:
    """Create deterministic different-UID state pairs or UID-level target donors."""

    if not uid_permutation:
        return _matched_state_pairs(rows, seed=int(seed))
    by_uid: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_uid[str(row["uid"])].append(index)
    uids = sorted(by_uid)
    if len(uids) < 2:
        raise ValueError("UID permutation requires at least two UIDs")
    ordered = sorted(uids, key=lambda uid: _stable(seed, "uid", uid))
    donors = {uid: ordered[(position + 1) % len(ordered)] for position, uid in enumerate(ordered)}
    output = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        candidates = by_uid[donors[str(row["uid"])]]
        layer = int(row["layer"])
        output[index] = min(
            candidates,
            key=lambda candidate: (
                abs(int(rows[candidate]["layer"]) - layer),
                _stable(seed, row["state_id"], rows[candidate]["state_id"]),
            ),
        )
    return output


def monotonic_nondecreasing(metrics: Mapping[int, float]) -> bool:
    """Check non-decreasing performance in the frozen horizon order."""

    values = [float(metrics[horizon]) for horizon in HORIZONS]
    return all(right >= left for left, right in zip(values, values[1:]))


def classify_h_read(
    *,
    h1_spearman: float,
    h1_auroc: float,
    horizon_spearman: Mapping[int, float],
    horizon_auroc: Mapping[int, float],
    best_single_spearman: Mapping[int, float],
    random_pair_spearman: Mapping[int, float],
    token_spearman: Mapping[int, float],
    ci_lower_spearman: Mapping[int, float],
    ci_lower_auroc: Mapping[int, float],
    ci_lower_token_spearman: Mapping[int, float],
    precision_top10: Mapping[int, float],
    prevalence: float,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Apply the prospectively frozen H-READ-A/B/C/D hierarchy."""

    material = []
    for horizon in (2, 4, 8):
        spearman_ok = (
            float(horizon_spearman[horizon]) - float(h1_spearman)
            >= float(thresholds["spearman_gain"])
            and float(ci_lower_spearman[horizon]) > 0
        )
        auroc_ok = (
            float(horizon_auroc[horizon]) - float(h1_auroc)
            >= float(thresholds["auroc_gain"])
            and float(ci_lower_auroc[horizon]) > 0
        )
        if spearman_ok or auroc_ok:
            material.append(horizon)
    useful_precision = any(
        float(precision_top10[horizon]) - float(prevalence)
        >= float(thresholds["useful_precision_gain"])
        for horizon in HORIZONS
    )
    for horizon in material:
        pair_gap = float(horizon_spearman[horizon]) - float(best_single_spearman[horizon])
        if horizon not in random_pair_spearman:
            continue
        random_gap = float(horizon_spearman[horizon]) - float(random_pair_spearman[horizon])
        if pair_gap >= float(thresholds["pair_over_single"]) and random_gap >= float(thresholds["random_pair_gap"]):
            return {
                "category": "H-READ-A",
                "smallest_material_horizon": horizon,
                "reason": "counterfactual pair/delta materially exceeds H=1, both singles, and random pairs",
                "material_horizons": material,
                "useful_high_precision_subset": useful_precision,
            }
    for horizon in material:
        single_gain = float(best_single_spearman[horizon]) - float(h1_spearman)
        pair_gap = float(horizon_spearman[horizon]) - float(best_single_spearman[horizon])
        if single_gain >= float(thresholds["spearman_gain"]) and pair_gap < float(thresholds["pair_over_single"]):
            return {
                "category": "H-READ-B",
                "smallest_material_horizon": horizon,
                "reason": "delayed single-branch state explains the material horizon gain",
                "material_horizons": material,
                "useful_high_precision_subset": useful_precision,
            }
    token_material = [
        horizon
        for horizon in (2, 4, 8)
        if float(token_spearman[horizon]) - float(token_spearman[1])
        >= float(thresholds["spearman_gain"])
        and float(ci_lower_token_spearman[horizon]) > 0
    ]
    if not material and token_material:
        return {
            "category": "H-READ-C",
            "smallest_material_horizon": token_material[0],
            "reason": "only the frozen token comparator shows material propagation gain",
            "material_horizons": token_material,
            "useful_high_precision_subset": useful_precision,
        }
    return {
        "category": "H-READ-D",
        "smallest_material_horizon": None,
        "reason": "no short horizon satisfies the complete prospective emergence pattern",
        "material_horizons": material,
        "useful_high_precision_subset": useful_precision,
    }
