"""Exact-prefix observed-valid supervision for matched Stage-2 training."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from dense_failure_stage2.v1_router import ACTION_NAMES, ACTION_TO_INDEX


def exact_state_id(uid: str, layer: int, actions: Sequence[str]) -> str:
    """Hash the exact entering route state: UID, layer, and complete prefix."""
    index = int(layer)
    if index < 0 or index >= len(actions):
        raise ValueError("layer is outside the action trajectory")
    normalized = tuple(str(action) for action in actions)
    if any(action not in ACTION_TO_INDEX for action in normalized):
        raise ValueError("trajectory contains an unsupported action")
    prefix = "|".join(normalized[:index])
    return sha256(f"{uid}\n{index}\n{prefix}".encode()).hexdigest()


def build_exact_valid_sets(
    routes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return unique exact states and all route-state occurrences."""
    states: dict[str, dict[str, Any]] = {}
    occurrences: list[dict[str, Any]] = []
    identity_by_id: dict[str, tuple[str, int, tuple[str, ...]]] = {}
    for route in routes:
        uid = str(route["uid"])
        actions = tuple(str(action) for action in route["actions"])
        activation = int(route["activation_layer"])
        if len(actions) != 28 or activation < 0 or activation >= len(actions):
            raise ValueError(f"invalid route geometry: {route.get('route_id')}")
        for layer in range(activation, len(actions)):
            state_id = exact_state_id(uid, layer, actions)
            identity = (uid, layer, actions[:layer])
            if state_id in identity_by_id and identity_by_id[state_id] != identity:
                raise RuntimeError(f"exact state hash collision: {state_id}")
            identity_by_id[state_id] = identity
            if state_id not in states:
                states[state_id] = {
                    "schema_version": "stage2_exact_observed_valid_state_v1",
                    "state_id": state_id,
                    "uid": uid,
                    "layer": layer,
                    "prefix_actions": list(actions[:layer]),
                    "observed_valid_actions": set(),
                    "route_sources": set(),
                    "route_ids": set(),
                    "representative_route_id": str(route["route_id"]),
                    "representative_actions": list(actions),
                    "representative_activation_layer": activation,
                }
            state = states[state_id]
            state["observed_valid_actions"].add(actions[layer])
            state["route_sources"].add(str(route["route_source"]))
            state["route_ids"].add(str(route["route_id"]))
            occurrences.append(
                {
                    "state_id": state_id,
                    "route_id": str(route["route_id"]),
                    "route_source": str(route["route_source"]),
                    "uid": uid,
                    "layer": layer,
                    "target_action": actions[layer],
                }
            )
    output = []
    for state_id in sorted(states):
        state = states[state_id]
        valid = [action for action in ACTION_NAMES if action in state["observed_valid_actions"]]
        if not valid:
            raise RuntimeError(f"empty observed-valid set: {state_id}")
        output.append(
            {
                **state,
                "observed_valid_actions": valid,
                "observed_valid_action_indices": [ACTION_TO_INDEX[action] for action in valid],
                "valid_action_count": len(valid),
                "contains_FULL": "FULL" in valid,
                "contains_nonFULL": any(action != "FULL" for action in valid),
                "route_sources": sorted(state["route_sources"]),
                "route_ids": sorted(state["route_ids"]),
            }
        )
    return output, occurrences


def valid_action_mask(
    valid_indices: Sequence[Sequence[int]], *, device: torch.device | str | None = None
) -> torch.Tensor:
    if not valid_indices:
        raise ValueError("valid-action batch cannot be empty")
    mask = torch.zeros((len(valid_indices), len(ACTION_NAMES)), dtype=torch.bool, device=device)
    for row_index, indices in enumerate(valid_indices):
        normalized = sorted({int(index) for index in indices})
        if not normalized or normalized[0] < 0 or normalized[-1] >= len(ACTION_NAMES):
            raise ValueError(f"invalid or empty action set at row {row_index}")
        mask[row_index, normalized] = True
    return mask


def observed_valid_set_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[-1] != len(ACTION_NAMES):
        raise ValueError("logits must have [batch, 4] shape")
    if mask.shape != logits.shape:
        raise ValueError("valid-action mask shape differs from logits")
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("every row must have at least one observed-valid action")
    stable = logits.float()
    if not bool(torch.isfinite(stable).all()):
        raise ValueError("logits contain NaN or Inf")
    valid_logits = stable.masked_fill(~mask.bool(), float("-inf"))
    loss = -(torch.logsumexp(valid_logits, dim=-1) - torch.logsumexp(stable, dim=-1)).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("observed-valid-set loss is non-finite")
    return loss


def observed_valid_metrics(logits: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    stable = logits.detach().float()
    if mask.shape != stable.shape or not bool(mask.any(dim=-1).all()):
        raise ValueError("invalid observed-valid metric inputs")
    probabilities = stable.softmax(dim=-1)
    predictions = probabilities.argmax(dim=-1)
    top1_valid = mask.gather(1, predictions[:, None]).squeeze(1)
    valid_mass = (probabilities * mask).sum(dim=-1)
    non_full_mask = mask.clone()
    non_full_mask[:, 0] = False
    non_full_mass = (probabilities * non_full_mask).sum(dim=-1)
    return {
        "rows": int(stable.shape[0]),
        "observed_valid_top1": int(top1_valid.sum().item()),
        "valid_probability_mass_sum": float(valid_mass.sum().item()),
        "non_full_valid_mass_sum": float(non_full_mass.sum().item()),
        "contains_non_full_rows": int(non_full_mask.any(dim=-1).sum().item()),
        "predictions": predictions.cpu().tolist(),
        "valid_probability_mass": valid_mass.cpu().tolist(),
        "non_full_valid_mass": non_full_mass.cpu().tolist(),
    }


def uid_paired_values(
    rows: Sequence[Mapping[str, Any]], metric: str, left: str, right: str
) -> dict[str, tuple[float, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        checkpoint = str(row["checkpoint"])
        if checkpoint in {left, right}:
            grouped[str(row["uid"])][checkpoint].append(float(row[metric]))
    pairs = {}
    for uid, values in grouped.items():
        if left not in values or right not in values:
            continue
        pairs[uid] = (
            sum(values[left]) / len(values[left]),
            sum(values[right]) / len(values[right]),
        )
    return pairs
