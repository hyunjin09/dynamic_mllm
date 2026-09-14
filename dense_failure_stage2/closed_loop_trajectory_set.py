"""Exact per-UID trajectory-set supervision for the closed-loop Stage-2 router."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

import torch

from .v1_router import ACTION_NAMES


def prefix_action_hash(
    uid: str,
    trigger_layer: int,
    layer: int,
    prefix_actions: Sequence[str],
) -> str:
    """Identify the exact routed state before ``layer`` by its action prefix."""
    payload = {
        "uid": str(uid),
        "trigger_layer": int(trigger_layer),
        "layer": int(layer),
        "prefix_actions": [str(action).upper() for action in prefix_actions],
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_trajectory_sets(
    rows: Sequence[Mapping[str, Any]], *, total_layers: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Validate the frozen corpus and group complete trajectories by UID."""
    if total_layers < 1 or not rows:
        raise ValueError("a nonempty corpus and positive decoder depth are required")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    program_ids: set[str] = set()
    unique_prefixes: set[tuple[str, int, tuple[str, ...]]] = set()
    route_state_occurrences = 0
    for source in rows:
        row = dict(source)
        uid = str(row["uid"])
        program_id = str(row["program_id"])
        if program_id in program_ids:
            raise ValueError(f"duplicate program ID: {program_id}")
        program_ids.add(program_id)
        trigger = int(row["trigger_layer"])
        suffix = [str(action).upper() for action in row["suffix_actions"]]
        if trigger < 0 or trigger >= total_layers:
            raise ValueError(f"trigger layer is outside the decoder: {uid}")
        if len(suffix) != total_layers - trigger:
            raise ValueError(f"suffix length differs from trigger layer: {program_id}")
        if any(action not in ACTION_NAMES for action in suffix):
            raise ValueError(f"unsupported action in program: {program_id}")
        full_actions = [str(action).upper() for action in row["full_actions"]]
        if len(full_actions) != total_layers or full_actions[:trigger] != ["FULL"] * trigger:
            raise ValueError(f"program does not preserve the all-FULL prefix: {program_id}")
        if full_actions[trigger:] != suffix:
            raise ValueError(f"full route and suffix disagree: {program_id}")
        outcome = str(row["dense_outcome"])
        if outcome not in {"C", "W"}:
            raise ValueError(f"unsupported Dense outcome: {outcome}")
        if outcome == "C" and any(action != "FULL" for action in suffix):
            raise ValueError(f"Dense-C trajectory must be all-FULL: {program_id}")
        prefix: list[str] = []
        for offset, action in enumerate(suffix):
            layer = trigger + offset
            unique_prefixes.add((uid, layer, tuple(prefix)))
            route_state_occurrences += 1
            prefix.append(action)
        grouped[uid].append(row)

    dense_c_uids = 0
    for uid, items in grouped.items():
        items.sort(key=lambda row: str(row["program_id"]))
        identity = {
            (
                row["dataset"],
                row["source_regime"],
                row["dense_outcome"],
                int(row["trigger_layer"]),
                row["image_group_id"],
            )
            for row in items
        }
        if len(identity) != 1:
            raise ValueError(f"UID metadata differs across trajectories: {uid}")
        if items[0]["dense_outcome"] == "C":
            dense_c_uids += 1
            if len(items) != 1:
                raise ValueError(f"Dense-C UID must have one preservation trajectory: {uid}")
    ordered = {uid: grouped[uid] for uid in sorted(grouped)}
    return ordered, {
        "uids": len(ordered),
        "programs": len(rows),
        "dense_c_uids": dense_c_uids,
        "dense_w_uids": len(ordered) - dense_c_uids,
        "route_state_occurrences": route_state_occurrences,
        "unique_prefix_states": len(unique_prefixes),
    }


def trajectory_set_marginal_loss(
    route_logps: torch.Tensor, *, suffix_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact length-normalized negative log-mean route probability."""
    if route_logps.ndim != 1 or route_logps.numel() < 1:
        raise ValueError("route_logps must be a nonempty vector")
    if suffix_length < 1:
        raise ValueError("suffix_length must be positive")
    if not torch.isfinite(route_logps).all():
        raise ValueError("route_logps contains a non-finite value")
    log_mean = torch.logsumexp(route_logps, dim=0) - math.log(route_logps.numel())
    loss = -log_mean / int(suffix_length)
    responsibilities = torch.softmax(route_logps.detach(), dim=0)
    return loss, responsibilities


def compact_router_state(
    text_states: torch.Tensor,
    visual_states: torch.Tensor,
    text_mask: torch.Tensor,
    visual_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Keep only the frozen router's actual query and routed visual inputs."""
    if text_states.ndim != 3 or visual_states.ndim != 3:
        raise ValueError("router states must have [batch, tokens, hidden] shape")
    if text_states.shape[0] != 1 or visual_states.shape[0] != 1:
        raise ValueError("routed-state caching is validated only for batch size one")
    if text_mask.shape != text_states.shape[:2] or visual_mask.shape != visual_states.shape[:2]:
        raise ValueError("router masks do not match state shapes")
    text_mask = text_mask.bool()
    visual_mask = visual_mask.bool()
    if not bool(text_mask.any()) or not bool(visual_mask.any()):
        raise ValueError("router state is missing text or visual tokens")
    last = int(text_mask.long().sum(dim=1)[0].item()) - 1
    query = text_states[:, last : last + 1].detach().contiguous()
    # Keep the visual tensor and its mask byte-for-byte.  Removing masked rows
    # is mathematically equivalent, but changes the attention kernel shape and
    # can change FP32 logits by a few ulps, violating the exact parity contract.
    visual = visual_states.detach().contiguous()
    return {
        "text_states": query,
        "visual_states": visual,
        "text_mask": torch.ones((1, 1), dtype=torch.bool, device=query.device),
        "visual_mask": visual_mask.detach().contiguous(),
    }


def validate_state_references(
    program_to_states: Mapping[str, Sequence[str]],
    state_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Fail closed unless the cache is a one-to-one cover of referenced states."""
    seen: dict[str, str] = {}
    for row in state_rows:
        state_id = str(row["state_id"])
        if state_id in seen:
            raise ValueError(f"duplicate state row: {state_id}")
        digest = str(row["state_sha256"])
        if len(digest) != 64:
            raise ValueError(f"invalid state hash: {state_id}")
        seen[state_id] = digest
    referenced = [str(state) for states in program_to_states.values() for state in states]
    missing = sorted(set(referenced).difference(seen))
    if missing:
        raise ValueError(f"missing state rows: {missing[:3]}")
    extra = sorted(set(seen).difference(referenced))
    if extra:
        raise ValueError(f"unreferenced state rows: {extra[:3]}")
    return {
        "programs": len(program_to_states),
        "state_references": len(referenced),
        "unique_states": len(seen),
    }


def hierarchical_uid_weights(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, float]:
    """Give equal mass to supported dataset/source/outcome cells, then UIDs."""
    cells: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for uid, rows in grouped.items():
        if not rows:
            raise ValueError(f"UID has no trajectories: {uid}")
        first = rows[0]
        cells[(str(first["dataset"]), str(first["source_regime"]), str(first["dense_outcome"]))].append(uid)
    weights: dict[str, float] = {}
    for uids in cells.values():
        for uid in uids:
            weights[uid] = 1.0 / len(cells) / len(uids)
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-12):
        raise AssertionError("UID weights do not sum to one")
    return weights


def _collate_cached_states(
    rows: Sequence[Mapping[str, torch.Tensor]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("cannot collate an empty state batch")
    hidden = int(rows[0]["text_states"].shape[-1])
    max_text = max(int(row["text_states"].shape[1]) for row in rows)
    max_visual = max(int(row["visual_states"].shape[1]) for row in rows)
    text_dtype = rows[0]["text_states"].dtype
    visual_dtype = rows[0]["visual_states"].dtype
    text = torch.zeros(len(rows), max_text, hidden, dtype=text_dtype, device=device)
    visual = torch.zeros(len(rows), max_visual, hidden, dtype=visual_dtype, device=device)
    text_mask = torch.zeros(len(rows), max_text, dtype=torch.bool, device=device)
    visual_mask = torch.zeros(len(rows), max_visual, dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        nt = int(row["text_states"].shape[1])
        nv = int(row["visual_states"].shape[1])
        if int(row["text_states"].shape[-1]) != hidden or int(row["visual_states"].shape[-1]) != hidden:
            raise ValueError("cached states use inconsistent hidden widths")
        text[index, :nt] = row["text_states"][0].to(device)
        visual[index, :nv] = row["visual_states"][0].to(device)
        text_mask[index, :nt] = row["text_mask"][0].to(device).bool()
        visual_mask[index, :nv] = row["visual_mask"][0].to(device).bool()
    return text, visual, text_mask, visual_mask


def compute_uid_trajectory_loss(
    router: torch.nn.Module,
    payload: Mapping[str, Any],
    *,
    device: torch.device,
    state_microbatch: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Compute one exact UID loss while forwarding each unique prefix state once."""
    if state_microbatch < 1:
        raise ValueError("state_microbatch must be positive")
    states = payload.get("states")
    programs = payload.get("programs")
    if not isinstance(states, Mapping) or not states or not isinstance(programs, Sequence) or not programs:
        raise ValueError("UID payload requires nonempty states and programs")
    state_ids = sorted(map(str, states))
    logits_by_state: dict[str, torch.Tensor] = {}
    for start in range(0, len(state_ids), int(state_microbatch)):
        current = state_ids[start : start + int(state_microbatch)]
        text, visual, text_mask, visual_mask = _collate_cached_states(
            [states[state_id] for state_id in current], device
        )
        logits = router(
            text,
            visual,
            text_mask=text_mask,
            visual_mask=visual_mask,
        )
        if tuple(logits.shape) != (len(current), len(ACTION_NAMES)) or not torch.isfinite(logits).all():
            raise RuntimeError("router produced invalid cached-state logits")
        for index, state_id in enumerate(current):
            logits_by_state[state_id] = logits[index]

    route_logps = []
    suffix_lengths = set()
    for program in programs:
        refs = [str(value) for value in program["state_ids"]]
        actions = [int(value) for value in program["action_indices"]]
        if not refs or len(refs) != len(actions):
            raise ValueError(f"program state/action lengths differ: {program.get('program_id')}")
        if any(state_id not in logits_by_state for state_id in refs):
            raise ValueError(f"program references an absent cached state: {program.get('program_id')}")
        if any(action < 0 or action >= len(ACTION_NAMES) for action in actions):
            raise ValueError(f"program contains an invalid action index: {program.get('program_id')}")
        terms = [
            torch.log_softmax(logits_by_state[state_id].float(), dim=-1)[action]
            for state_id, action in zip(refs, actions)
        ]
        route_logps.append(torch.stack(terms).sum())
        suffix_lengths.add(len(refs))
    if len(suffix_lengths) != 1:
        raise ValueError("one UID contains trajectories with different suffix lengths")
    stacked = torch.stack(route_logps)
    loss, responsibilities = trajectory_set_marginal_loss(
        stacked, suffix_length=next(iter(suffix_lengths))
    )
    entropy = -(
        responsibilities.double()
        * responsibilities.double().clamp_min(torch.finfo(torch.float64).tiny).log()
    ).sum()
    return loss, {
        "route_count": len(programs),
        "state_count": len(states),
        "responsibility_entropy": float(entropy.item()),
        "top_responsibility": float(responsibilities.max().item()),
        "mean_route_logp": float(stacked.detach().mean().item()),
        "best_route_logp": float(stacked.detach().max().item()),
        "program_ids": [str(program["program_id"]) for program in programs],
        "route_logps": [float(value) for value in stacked.detach().cpu().tolist()],
        "responsibilities": [
            float(value) for value in responsibilities.detach().cpu().tolist()
        ],
    }
