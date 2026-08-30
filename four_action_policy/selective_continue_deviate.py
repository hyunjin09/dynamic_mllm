"""Pure helpers for the selective CONTINUE/DEVIATE Phase-1 audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np

from four_action_policy.actions import FOUR_ACTIONS


def _depth_bin(layer: int) -> str:
    if not 0 <= layer < 28:
        raise ValueError("boundary layer must be in [0, 27]")
    if layer <= 9:
        return "early"
    if layer <= 18:
        return "middle"
    return "late"


def build_full_insertion_subset(
    manifest_rows: Sequence[dict[str, Any]],
    boundaries: Sequence[dict[str, Any]],
    *,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a census of W2C boundaries with every known compatible suffix.

    Compatible routes are exactly the correcting routes that reach the frozen
    all-FULL prefix and use a non-FULL action at the mandatory boundary.  The
    boundary action is replaced by FULL and identical resulting routes are
    deduplicated without losing their source-route provenance.
    """

    row_by_uid = {str(row["uid"]): row for row in manifest_rows}
    if len(row_by_uid) != len(manifest_rows):
        raise ValueError("source manifest contains duplicate UIDs")
    boundary_by_uid = {str(row["uid"]): row for row in boundaries}
    if len(boundary_by_uid) != len(boundaries):
        raise ValueError("boundary manifest contains duplicate UIDs")

    selected_rows = [
        row
        for row in manifest_rows
        if row.get("split") == split and row.get("route_type") == "W2C"
    ]
    selected_uids = {str(row["uid"]) for row in selected_rows}
    if not selected_rows or not selected_uids.issubset(boundary_by_uid):
        raise ValueError("boundaries do not cover the requested W2C split")

    subset: list[dict[str, Any]] = []
    source_suffixes = 0
    for row in sorted(selected_rows, key=lambda value: str(value["uid"])):
        uid = str(row["uid"])
        boundary = boundary_by_uid[uid]
        layer = int(boundary["boundary_layer"])
        if int(boundary["all_full_prefix_length"]) != layer:
            raise ValueError("mandatory boundary all-FULL prefix is inconsistent")
        prefix = ("FULL",) * layer
        valid_actions = sorted(
            {str(value) for value in boundary["valid_nonfull_actions"]},
            key=FOUR_ACTIONS.index,
        )
        if not valid_actions or "FULL" in valid_actions:
            raise ValueError("mandatory boundary must have non-FULL valid actions")

        compatible_indices = []
        for route_index, source in enumerate(row["valid_routes"]):
            actions = tuple(str(value) for value in source["actions"])
            if (
                len(actions) == 28
                and actions[:layer] == prefix
                and actions[layer] != "FULL"
            ):
                compatible_indices.append(route_index)
        frozen_indices = sorted(
            {int(value) for value in boundary["boundary_route_indices"]}
        )
        if frozen_indices != compatible_indices:
            raise ValueError(
                "boundary indices must exactly enumerate compatible known suffixes"
            )
        observed_actions = sorted(
            {
                str(row["valid_routes"][route_index]["actions"][layer])
                for route_index in frozen_indices
            },
            key=FOUR_ACTIONS.index,
        )
        if observed_actions != valid_actions:
            raise ValueError("boundary valid actions disagree with compatible routes")

        grouped: dict[tuple[str, ...], dict[str, Any]] = {}
        for route_index in frozen_indices:
            source = row["valid_routes"][route_index]
            actions = tuple(str(value) for value in source["actions"])
            candidate = (*prefix, "FULL", *actions[layer + 1 :])
            group = grouped.setdefault(
                candidate,
                {
                    "source_route_indices": [],
                    "source_route_keys": [],
                },
            )
            group["source_route_indices"].append(route_index)
            group["source_route_keys"].append(
                str(source.get("route_key", f"route_index_{route_index}"))
            )
        if not grouped:
            raise ValueError("mandatory boundary has no compatible known suffix")

        candidates = []
        for candidate_actions, provenance in sorted(grouped.items()):
            candidates.append(
                {
                    "candidate_index": len(candidates),
                    "actions": list(candidate_actions),
                    "source_route_indices": provenance["source_route_indices"],
                    "source_route_keys": provenance["source_route_keys"],
                }
            )
        state_id = sha256(f"full-insertion:{uid}:{layer}".encode()).hexdigest()[:24]
        subset.append(
            {
                "state_id": state_id,
                "uid": uid,
                "split": split,
                "dataset": str(row["dataset"]),
                "route_type": "W2C",
                "target_layer": layer,
                "depth_bin": _depth_bin(layer),
                "known_valid_actions": valid_actions,
                "known_mechanism": (
                    valid_actions[0] if len(valid_actions) == 1 else "MULTI"
                ),
                "prefix_actions": list(prefix),
                "compatible_suffix_count": len(frozen_indices),
                "candidate_route_count": len(candidates),
                "suffix_set_complete": True,
                "suffix_set_basis": "all_frozen_boundary_route_indices",
                "candidate_routes": candidates,
            }
        )
        source_suffixes += len(frozen_indices)

    candidate_routes = sum(row["candidate_route_count"] for row in subset)
    audit = {
        "schema_version": "selective_continue_deviate_full_insertion_subset_audit_v1",
        "split": split,
        "states": len(subset),
        "uids": len({row["uid"] for row in subset}),
        "source_compatible_suffixes": source_suffixes,
        "candidate_routes": candidate_routes,
        "deduplicated_source_routes": source_suffixes - candidate_routes,
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in subset).items())),
        "depth_counts": dict(sorted(Counter(row["depth_bin"] for row in subset).items())),
        "mechanism_counts": dict(
            sorted(Counter(row["known_mechanism"] for row in subset).items())
        ),
        "candidate_count_histogram": dict(
            sorted(Counter(row["candidate_route_count"] for row in subset).items())
        ),
        "all_suffix_sets_complete": all(row["suffix_set_complete"] for row in subset),
    }
    return subset, audit


def bootstrap_uid_rescue_rate(
    rows: Sequence[Mapping[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    """Return a deterministic UID-group percentile interval for rescue rate."""

    if draws < 1 or not rows:
        raise ValueError("bootstrap requires nonempty rows and positive draws")
    grouped: dict[str, set[bool]] = defaultdict(set)
    for row in rows:
        grouped[str(row["uid"])].add(bool(row["rescued"]))
    if any(len(values) != 1 for values in grouped.values()):
        raise ValueError("rescue labels must be constant within each UID")
    values = np.asarray(
        [float(next(iter(grouped[uid]))) for uid in sorted(grouped)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    samples = values[indices].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "uids": int(len(values)),
        "draws": int(draws),
        "seed": int(seed),
        "estimate": float(values.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


def summarize_full_insertion_audit(
    subset: Sequence[dict[str, Any]],
    executions: Sequence[dict[str, Any]],
    *,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Validate exhaustive execution coverage and classify each boundary."""

    state_by_id: dict[str, dict[str, Any]] = {}
    expected: set[tuple[str, int]] = set()
    for state in subset:
        state_id = str(state["state_id"])
        if state_id in state_by_id:
            raise ValueError("audit subset contains duplicate states")
        state_by_id[state_id] = state
        candidates = state.get("candidate_routes")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("audit state has no candidate routes")
        for candidate in candidates:
            key = (state_id, int(candidate["candidate_index"]))
            if key in expected:
                raise ValueError("audit subset contains duplicate candidates")
            expected.add(key)

    observed: dict[tuple[str, int], dict[str, Any]] = {}
    for execution in executions:
        key = (str(execution["state_id"]), int(execution["candidate_index"]))
        if key in observed:
            raise ValueError("audit executions contain duplicate candidates")
        observed[key] = execution
    if set(observed) != expected:
        raise ValueError("audit executions must exactly cover frozen candidates")

    state_results = []
    for state_id in sorted(state_by_id):
        state = state_by_id[state_id]
        candidates = [
            observed[(state_id, int(candidate["candidate_index"]))]
            for candidate in state["candidate_routes"]
        ]
        successful = [
            int(row["candidate_index"]) for row in candidates if bool(row["correct"])
        ]
        rescued = bool(successful)
        if rescued:
            status = "FULL-cache-incomplete"
        elif bool(state.get("suffix_set_complete", False)):
            status = "FULL-confirmed-invalid"
        else:
            status = "unresolved"
        state_results.append(
            {
                "state_id": state_id,
                "uid": str(state["uid"]),
                "dataset": str(state["dataset"]),
                "target_layer": int(state.get("target_layer", -1)),
                "depth_bin": str(state["depth_bin"]),
                "known_mechanism": str(state["known_mechanism"]),
                "compatible_suffix_count": int(
                    state.get("compatible_suffix_count", len(candidates))
                ),
                "candidate_route_count": len(candidates),
                "successful_candidate_indices": successful,
                "rescued": rescued,
                "status": status,
                "candidate_executions": candidates,
            }
        )

    def group_summary(key: str) -> dict[str, Any]:
        output = {}
        groups = sorted({str(row[key]) for row in state_results})
        for index, value in enumerate(groups):
            rows = [row for row in state_results if str(row[key]) == value]
            interval = bootstrap_uid_rescue_rate(
                rows,
                draws=bootstrap_draws,
                seed=bootstrap_seed + index + 1,
            )
            output[value] = {
                "states": len(rows),
                "rescued": sum(int(row["rescued"]) for row in rows),
                "rescue_rate": interval["estimate"],
                "ci95_lower": interval["ci95_lower"],
                "ci95_upper": interval["ci95_upper"],
                "bootstrap_draws": bootstrap_draws,
                "bootstrap_seed": interval["seed"],
            }
        return output

    overall_interval = bootstrap_uid_rescue_rate(
        state_results, draws=bootstrap_draws, seed=bootstrap_seed
    )
    return {
        "schema_version": "selective_continue_deviate_when_label_audit_v1",
        "states": len(state_results),
        "uids": len({row["uid"] for row in state_results}),
        "candidate_executions": len(executions),
        "status_counts": dict(sorted(Counter(row["status"] for row in state_results).items())),
        "overall": {
            "states": len(state_results),
            "rescued": sum(int(row["rescued"]) for row in state_results),
            "rescue_rate": overall_interval["estimate"],
            "ci95_lower": overall_interval["ci95_lower"],
            "ci95_upper": overall_interval["ci95_upper"],
            "bootstrap_draws": bootstrap_draws,
            "bootstrap_seed": bootstrap_seed,
        },
        "by_dataset": group_summary("dataset"),
        "by_depth_bin": group_summary("depth_bin"),
        "by_known_mechanism": group_summary("known_mechanism"),
        "suffix_count_histogram": dict(
            sorted(Counter(row["candidate_route_count"] for row in state_results).items())
        ),
        "state_results": state_results,
    }


def evaluate_phase1_gate(
    summary: Mapping[str, Any], decision_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the frozen sample-contract gate to a Phase-1 audit summary."""

    states = int(summary["states"])
    counts = summary["status_counts"]
    rescued = int(counts.get("FULL-cache-incomplete", 0))
    unresolved = int(counts.get("unresolved", 0))
    trusted = int(counts.get("FULL-confirmed-invalid", 0))
    if rescued + unresolved + trusted != states:
        raise ValueError("Phase-1 status counts do not cover every state")
    required = int(decision_config["required_trusted_validation_positives"])
    passed = (
        rescued <= int(decision_config["maximum_rescued_states"])
        and unresolved <= int(decision_config["maximum_unresolved_states"])
        and trusted >= required
    )
    return {
        "passed": passed,
        "outcome": (
            "case_b_proceed_selective_gate"
            if passed
            else "case_a_stop_label_incompleteness"
        ),
        "states": states,
        "trusted_validation_deviate_positives": trusted,
        "rescued_states": rescued,
        "unresolved_states": unresolved,
        "required_trusted_validation_positives": required,
        "next_stage": (
            "linear_and_mlp_gate_training"
            if passed
            else "stop_before_gate_training"
        ),
    }
