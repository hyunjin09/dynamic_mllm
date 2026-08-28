from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .aggregate import classify_rescue


def trajectory_reference_from_state(
    state: dict[str, Any], trajectory: dict[str, Any]
) -> dict[str, Any]:
    """Reconstruct the endpoint margin for the trajectory's fixed answer target."""
    target_ids = trajectory.get("correct_target_token_ids")
    target_text = trajectory["correct_target_text"]
    candidates = state["correct_target_scores"]["candidates"]
    if target_ids is not None:
        matches = [row for row in candidates if row.get("token_ids") == target_ids]
    else:
        matches = [row for row in candidates if row.get("text") == target_text]
    if len(matches) != 1:
        raise ValueError(
            f"expected one state candidate for trajectory target {target_text!r}, "
            f"found {len(matches)}"
        )
    fixed = matches[0]
    wrong = state.get("full_wrong_target_score")
    fixed_margin = float(fixed["mean_logprob"])
    if wrong is not None:
        wrong_ids = trajectory.get("wrong_target_token_ids")
        wrong_text = trajectory.get("wrong_target_text")
        if wrong_ids is not None and wrong.get("token_ids") != wrong_ids:
            raise ValueError("trajectory and state use different wrong-target token ids")
        if wrong_ids is None and wrong_text is not None and wrong.get("text") != wrong_text:
            raise ValueError("trajectory and state use different wrong-target texts")
        fixed_margin -= float(wrong["mean_logprob"])
    selected = state["correct_target_scores"]["selected"]
    return {
        "definition": "fixed_baseline_correct_target_against_fixed_full_wrong_target",
        "fixed_correct_target_text": fixed["text"],
        "fixed_correct_target_token_ids": fixed.get("token_ids"),
        "state_selected_correct_target_text": selected["text"],
        "state_selected_correct_target_token_ids": selected.get("token_ids"),
        "correct_target_switched": fixed.get("token_ids") != selected.get("token_ids"),
        "fixed_target_state_S_correct": float(fixed["mean_logprob"]),
        "fixed_target_state_S_full_wrong": None
        if wrong is None
        else float(wrong["mean_logprob"]),
        "fixed_target_state_margin": fixed_margin,
        "evaluator_best_state_margin": float(state["margin"]),
    }


def followup_thresholds(samples: Iterable[dict[str, Any]]) -> dict[str, float]:
    read = []
    write = []
    for sample in samples:
        for layer in sample["layers"]:
            read.append(abs(float(layer["effects"]["read_w1"])))
            write.append(abs(float(layer["effects"]["write_r1"])))
    return {
        "read_w1_q90_absolute": float(np.quantile(read, 0.90)),
        "write_r1_q90_absolute": float(np.quantile(write, 0.90)),
    }


def select_followups(
    samples: Iterable[dict[str, Any]], thresholds: dict[str, float]
) -> list[dict[str, Any]]:
    selected: dict[tuple[str, int, str], dict[str, Any]] = {}

    def add(sample, layer, action, operation, reason):
        key = (sample["uid"], int(layer["layer"]), action)
        if key not in selected:
            state = layer["states"][action]
            selected[key] = {
                "uid": sample["uid"],
                "dataset": sample["dataset"],
                "layer": int(layer["layer"]),
                "suppressed_action": action,
                "culprit_operation": operation,
                "reasons": [],
                "expected_final_margin": float(state["margin"]),
                "expected_generated_ids": state["generated_ids"],
                "expected_correct": bool(state["correct"]),
            }
        selected[key]["reasons"].append(reason)

    for sample in samples:
        for layer in sample["layers"]:
            states = layer["states"]
            category = classify_rescue(
                bool(states["IGNORE"]["correct"]),
                bool(states["READ_ONLY"]["correct"]),
                bool(states["WRITE_ONLY"]["correct"]),
            )
            if category == "write_removal_only":
                add(sample, layer, "READ_ONLY", "WRITE", "discrete_write_removal_rescue")
            elif category == "read_removal_only":
                add(sample, layer, "WRITE_ONLY", "READ", "discrete_read_removal_rescue")
            elif category == "either_removal_sufficient":
                add(sample, layer, "READ_ONLY", "WRITE", "discrete_either_removal_rescue")
                add(sample, layer, "WRITE_ONLY", "READ", "discrete_either_removal_rescue")
            elif category == "joint_removal_only":
                add(sample, layer, "IGNORE", "JOINT", "discrete_joint_removal_rescue")

            read_effect = float(layer["effects"]["read_w1"])
            write_effect = float(layer["effects"]["write_r1"])
            if read_effect <= -thresholds["read_w1_q90_absolute"]:
                add(sample, layer, "WRITE_ONLY", "READ", "read_w1_negative_q90")
            if write_effect <= -thresholds["write_r1_q90_absolute"]:
                add(sample, layer, "READ_ONLY", "WRITE", "write_r1_negative_q90")

    rows = list(selected.values())
    for row in rows:
        row["reasons"] = sorted(set(row["reasons"]))
    return sorted(rows, key=lambda row: (row["dataset"], row["uid"], row["layer"], row["suppressed_action"]))
