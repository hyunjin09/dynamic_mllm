"""Pure metrics for the teacher-forced versus free-run Stage-2 audit."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence

import torch

from dense_failure_stage2.closed_loop_trajectory_set import _collate_cached_states
from dense_failure_stage2.v1_router import ACTION_NAMES


def action_diagnostics(
    logits: Sequence[float], *, target_action: str
) -> dict[str, Any]:
    """Summarize one frozen router decision without executing the action."""
    values = [float(value) for value in logits]
    if len(values) != len(ACTION_NAMES) or not all(math.isfinite(value) for value in values):
        raise ValueError("action logits must be four finite values")
    target = str(target_action).upper()
    if target not in ACTION_NAMES:
        raise ValueError(f"unsupported target action: {target_action}")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    normalizer = sum(weights)
    probabilities = {
        action: weights[index] / normalizer for index, action in enumerate(ACTION_NAMES)
    }
    target_index = ACTION_NAMES.index(target)
    top_index = max(range(len(values)), key=lambda index: (values[index], -index))
    target_rank = 1 + sum(value > values[target_index] for value in values)
    best_non_full = max(values[1:])
    return {
        "logits": {action: values[index] for index, action in enumerate(ACTION_NAMES)},
        "probabilities": probabilities,
        "top1_action": ACTION_NAMES[top_index],
        "target_action": target,
        "target_top1": top_index == target_index,
        "target_probability": probabilities[target],
        "target_rank": target_rank,
        "target_vs_full_margin": values[target_index] - values[0],
        "full_vs_best_nonfull_margin": values[0] - best_non_full,
    }


def select_reference_programs(
    responsibility_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Select each UID's highest-responsibility route with a stable tie-break."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in responsibility_rows:
        uid = str(row["uid"])
        probability = float(row["responsibility"])
        if not math.isfinite(probability) or probability < 0:
            raise ValueError(f"invalid responsibility for {uid}")
        grouped[uid].append(row)
    if not grouped:
        raise ValueError("reference selection requires responsibility rows")
    return {
        uid: str(
            min(
                rows,
                key=lambda row: (-float(row["responsibility"]), str(row["program_id"])),
            )["program_id"]
        )
        for uid, rows in sorted(grouped.items())
    }


@torch.inference_mode()
def score_cached_uid(
    router: torch.nn.Module,
    payload: Mapping[str, Any],
    *,
    device: torch.device,
    state_microbatch: int,
) -> dict[str, Any]:
    """Score every unique cached state once, then expand to route occurrences."""
    if state_microbatch < 1:
        raise ValueError("state microbatch must be positive")
    states = payload.get("states")
    programs = payload.get("programs")
    if not isinstance(states, Mapping) or not states:
        raise ValueError("cached UID payload has no states")
    if not isinstance(programs, Sequence) or not programs:
        raise ValueError("cached UID payload has no programs")
    state_ids = sorted(map(str, states))
    logits_by_state: dict[str, list[float]] = {}
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
        ).float()
        if tuple(logits.shape) != (len(current), len(ACTION_NAMES)):
            raise RuntimeError("router returned invalid cached-state logits")
        if not torch.isfinite(logits).all():
            raise RuntimeError("router returned non-finite cached-state logits")
        for index, state_id in enumerate(current):
            logits_by_state[state_id] = [
                float(value) for value in logits[index].detach().cpu().tolist()
            ]

    occurrence_rows: list[dict[str, Any]] = []
    route_logps: list[float] = []
    program_rows: list[dict[str, Any]] = []
    trigger = int(payload["trigger_layer"])
    for program in programs:
        references = [str(value) for value in program["state_ids"]]
        actions = [str(value).upper() for value in program["actions"]]
        if not references or len(references) != len(actions):
            raise ValueError(f"program references/actions differ: {program.get('program_id')}")
        terms = []
        for offset, (state_id, action) in enumerate(zip(references, actions)):
            if state_id not in logits_by_state:
                raise ValueError(f"program references absent state: {program.get('program_id')}")
            diagnostics = action_diagnostics(logits_by_state[state_id], target_action=action)
            terms.append(math.log(max(float(diagnostics["target_probability"]), 1e-45)))
            state = states[state_id]
            occurrence_rows.append({
                "uid": str(payload["uid"]),
                "dataset": str(payload["dataset"]),
                "source_regime": str(payload["source_regime"]),
                "dense_outcome": str(payload["dense_outcome"]),
                "image_group_id": str(payload["image_group_id"]),
                "program_id": str(program["program_id"]),
                "provenance": list(program.get("provenance", [])),
                "state_id": state_id,
                "layer": int(state.get("layer", trigger + offset)),
                "trigger_layer": trigger,
                "depth_after_trigger": offset,
                "prefix_actions": list(state.get("prefix_actions", actions[:offset])),
                **diagnostics,
            })
        route_logp = float(sum(terms))
        route_logps.append(route_logp)
        program_rows.append({
            "uid": str(payload["uid"]),
            "program_id": str(program["program_id"]),
            "actions": actions,
            "provenance": list(program.get("provenance", [])),
            "route_logp": route_logp,
            "first_nonfull_offset": (
                first_non_full_offset(actions)
                if str(payload["dense_outcome"]) == "W"
                else None
            ),
        })
    route_tensor = torch.tensor(route_logps, dtype=torch.float64)
    responsibilities = torch.softmax(route_tensor, dim=0).tolist()
    for row, responsibility in zip(program_rows, responsibilities):
        row["responsibility"] = float(responsibility)
    return {
        "uid": str(payload["uid"]),
        "state_logits": logits_by_state,
        "occurrences": occurrence_rows,
        "programs": program_rows,
    }


def first_non_full_offset(actions: Sequence[str]) -> int:
    for index, action in enumerate(actions):
        if str(action).upper() != "FULL":
            return index
    raise ValueError("route has no non-FULL action")


def prefix_support_trace(
    free_actions: Sequence[str],
    programs: Sequence[Mapping[str, Any]],
    *,
    trigger_layer: int,
    probabilities: Sequence[Mapping[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Track exact compatibility with the union of observed successful routes."""
    actions = [str(action).upper() for action in free_actions]
    if not actions or any(action not in ACTION_NAMES for action in actions):
        raise ValueError("free rollout contains an unsupported or empty action sequence")
    normalized_programs = []
    for program in programs:
        route = [str(action).upper() for action in program["actions"]]
        if len(route) != len(actions) or any(action not in ACTION_NAMES for action in route):
            raise ValueError("successful routes must match the free suffix length")
        normalized_programs.append((str(program["program_id"]), route))
    if not normalized_programs:
        raise ValueError("support tracking requires at least one successful route")
    if probabilities is not None and len(probabilities) != len(actions):
        raise ValueError("probability rows must align with free actions")

    compatible = list(normalized_programs)
    trace: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    for offset, chosen in enumerate(actions):
        before = list(compatible)
        supported_actions = [
            action for action in ACTION_NAMES if any(route[offset] == action for _, route in before)
        ]
        current_probabilities = probabilities[offset] if probabilities is not None else None
        if current_probabilities is not None:
            if set(current_probabilities) != set(ACTION_NAMES):
                raise ValueError("probability row does not cover the four actions")
            supported_mass = sum(float(current_probabilities[action]) for action in supported_actions)
            full_probability = float(current_probabilities["FULL"])
            best_supported = (
                max(supported_actions, key=lambda action: float(current_probabilities[action]))
                if supported_actions
                else None
            )
            best_supported_probability = (
                float(current_probabilities[best_supported]) if best_supported is not None else None
            )
            chosen_probability = float(current_probabilities[chosen])
            chosen_margin = (
                math.log(max(chosen_probability, 1e-45))
                - math.log(max(best_supported_probability, 1e-45))
                if best_supported_probability is not None
                else None
            )
        else:
            supported_mass = full_probability = best_supported_probability = chosen_margin = None
            best_supported = None
        compatible = [(program_id, route) for program_id, route in before if route[offset] == chosen]
        row = {
            "layer": int(trigger_layer) + offset,
            "depth_after_trigger": offset,
            "chosen_action": chosen,
            "supported_actions": supported_actions,
            "supported": bool(compatible),
            "compatible_program_ids_before": sorted(program_id for program_id, _ in before),
            "compatible_program_ids_after": sorted(program_id for program_id, _ in compatible),
            "compatible_route_count_before": len(before),
            "compatible_route_count_after": len(compatible),
            "supported_probability_mass": supported_mass,
            "full_probability": full_probability,
            "best_supported_action": best_supported,
            "best_supported_probability": best_supported_probability,
            "chosen_vs_best_supported_logit_margin": chosen_margin,
        }
        trace.append(row)
        if first is None and not compatible:
            first = dict(row)
    return trace, first


def support_survival(
    traces: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, int | float]]:
    if not traces:
        raise ValueError("support survival requires at least one trace")
    depths = sorted({int(row["depth_after_trigger"]) for trace in traces for row in trace})
    output = []
    for depth in depths:
        eligible = [
            next((row for row in trace if int(row["depth_after_trigger"]) == depth), None)
            for trace in traces
        ]
        eligible = [row for row in eligible if row is not None]
        supported = sum(bool(row["supported"]) for row in eligible)
        output.append({
            "depth_after_trigger": depth,
            "eligible_uids": len(eligible),
            "supported_uids": supported,
            "survival": supported / len(eligible),
        })
    return output


def classify_bottleneck(
    metrics: Mapping[str, float | None], thresholds: Mapping[str, float]
) -> dict[str, Any]:
    """Apply the prospectively frozen, earliest-failure decision order."""
    required = (
        "high_first_nonfull_recall",
        "material_release_delta",
        "material_generalization_drop",
        "majority_fraction",
        "adequate_release_success",
    )
    if any(key not in thresholds for key in required):
        raise ValueError("decision thresholds are incomplete")
    seen = float(metrics["seen_first_nonfull_recall"])
    r0 = float(metrics["r0_success"])
    r1 = float(metrics["r1_success"])
    r2 = float(metrics["r2_success"])
    heldout = metrics.get("heldout_first_nonfull_recall")
    pre_off = float(metrics.get("pre_intervention_off_support_fraction") or 0.0)
    post_off = float(metrics.get("post_intervention_off_support_fraction") or 0.0)
    high = float(thresholds["high_first_nonfull_recall"])
    delta = float(thresholds["material_release_delta"])
    generalization = float(thresholds["material_generalization_drop"])
    majority = float(thresholds["majority_fraction"])
    adequate = float(thresholds["adequate_release_success"])

    evidence = {
        "seen_first_nonfull_recall": seen,
        "r1_minus_r0": r1 - r0,
        "r2_minus_r1": r2 - r1,
        "seen_minus_heldout": None if heldout is None else seen - float(heldout),
        "pre_intervention_off_support_fraction": pre_off,
        "post_intervention_off_support_fraction": post_off,
    }
    if seen < high or r2 - r1 >= delta:
        case = "objective_action_learning"
    elif heldout is not None and seen - float(heldout) >= generalization:
        case = "generalization"
    elif r1 - r0 >= delta and pre_off >= majority:
        case = "pre_intervention_exposure"
    elif r2 < adequate and post_off >= majority:
        case = "post_intervention_exposure"
    else:
        case = "mixed"
    return {"case": case, "evidence": evidence, "thresholds": dict(thresholds)}
