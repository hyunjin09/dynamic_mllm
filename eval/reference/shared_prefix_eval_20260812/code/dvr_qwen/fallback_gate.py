"""Input-only all-on fallback utilities for the online visual router."""

from __future__ import annotations

from typing import Any

import torch

from dvr_qwen.preference_gt import NUM_LAYERS
from dvr_qwen.router_features import summarize_text_states, summarize_visual_states
from dvr_qwen.split_scatter import BinaryDVRCInputs


RESCUE_ONLY_POLICY = "rescue_only"
SUPPORTED_SPARSE_POLICY = "supported_sparse"
GATE_LABEL_POLICIES = frozenset({RESCUE_ONLY_POLICY, SUPPORTED_SPARSE_POLICY})


def sparse_correct_route_supported(row: dict[str, Any], *, num_layers: int = NUM_LAYERS) -> bool:
    """Whether the finite candidate pool contains a non-trivial correct sparse route."""

    sparse_count = int(row.get("sparse_correct_route_count") or 0)
    minimum_budget = row.get("minimum_sparse_correct_budget")
    if sparse_count <= 0 or minimum_budget is None:
        return False
    budget = int(minimum_budget)
    return 0 < budget < int(num_layers) and not bool(row.get("null_visual_optimal", False))


def sparse_router_gate_label(
    row: dict[str, Any],
    *,
    policy: str = RESCUE_ONLY_POLICY,
    num_layers: int = NUM_LAYERS,
) -> int:
    """Derive an all-on-relative admission target from V3.1 outcomes.

    ``rescue_only`` is the default conservative setting.  It admits a sparse
    router only where all-on was incorrect and the candidate pool observed a
    correct non-trivial sparse route.  All all-on-correct rows, all-off-optimal
    rows, and rows with no observed correct sparse route explicitly supervise
    the all-on fallback.

    ``supported_sparse`` is retained solely for later efficiency ablations; it
    admits any observed correct sparse route and is not the safe default.
    """

    if policy not in GATE_LABEL_POLICIES:
        raise ValueError(f"unknown fallback-gate policy {policy!r}")
    supported = sparse_correct_route_supported(row, num_layers=num_layers)
    if policy == RESCUE_ONLY_POLICY:
        return int(supported and not bool(row.get("all_on_correct", False)))
    return int(supported)


def initial_input_gate_features(
    binary_inputs: BinaryDVRCInputs,
    *,
    text_summary_mode: str = "instruction_only",
    visual_summary_count: int = 2,
) -> dict[str, torch.Tensor]:
    """Extract pre-language-layer gate features from immutable model inputs."""

    if text_summary_mode not in {"all_text", "instruction_only"}:
        raise ValueError("text_summary_mode must be 'all_text' or 'instruction_only'")
    if visual_summary_count not in {0, 2}:
        raise ValueError("visual_summary_count must be 0 or 2")
    valid_mask = binary_inputs.text_valid_mask
    if text_summary_mode == "instruction_only":
        if binary_inputs.instruction_valid_mask is None:
            raise ValueError("instruction_only fallback gate requires instruction_token_mask")
        valid_mask = binary_inputs.instruction_valid_mask
    summary = summarize_text_states(binary_inputs.text_states.detach(), valid_mask.to(binary_inputs.text_states.device))
    result = {
        "instruction_mean": summary["global_mean"].detach(),
        "instruction_last": summary["last_token"].detach(),
    }
    if visual_summary_count:
        result["visual_summaries"] = summarize_visual_states(
            binary_inputs.visual_states.detach(),
            binary_inputs.visual_valid_mask.to(binary_inputs.visual_states.device),
        ).detach()
    return result

