"""Deterministic offline audit helpers for the frozen full-benchmark run."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence


ACTION_NAMES = ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")
NONFULL_ACTIONS = ACTION_NAMES[1:]


def safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _nonfull_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"layer": layer, "action": str(action)}
        for layer, action in enumerate(row["actions"])
        if str(action) != "FULL"
    ]


def validate_audit_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_contract: str
) -> dict[str, int]:
    """Fail closed on fields needed to reconstruct the frozen pipeline trace."""
    if not rows:
        raise ValueError("audit population is empty")
    uids = [str(row.get("uid")) for row in rows]
    duplicates = [uid for uid, count in Counter(uids).items() if count != 1]
    if duplicates:
        raise ValueError(f"audit population contains duplicate UIDs: {duplicates[:5]}")

    families: Counter[str] = Counter()
    for row in rows:
        uid = str(row["uid"])
        if str(row.get("contract_sha256")) != str(expected_contract):
            raise ValueError(f"{uid}: contract mismatch")
        scores = list(row.get("stage1_scores", ()))
        if len(scores) != 28:
            raise ValueError(f"{uid}: expected 28 Stage-1 scores")
        if not math.isclose(
            float(row.get("stage1_max_score")),
            max(float(value) for value in scores),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{uid}: Stage-1 max score is inconsistent")
        actions = [str(action) for action in row.get("actions", ())]
        if len(actions) != 28 or any(action not in ACTION_NAMES for action in actions):
            raise ValueError(f"{uid}: expected 28 valid Stage-2 actions")
        action_rows = list(row.get("action_rows", ()))
        if len(action_rows) != 28:
            raise ValueError(f"{uid}: expected 28 action_rows")

        dense = bool(row["dense_correct"])
        routed = bool(row["routed_correct"])
        expected_transition = ("C" if dense else "W") + "→" + ("C" if routed else "W")
        if str(row.get("transition")) != expected_transition:
            raise ValueError(f"{uid}: transition is inconsistent with correctness")

        threshold = float(row["stage1_threshold"])
        expected_trigger = next(
            (index for index, value in enumerate(scores) if float(value) > threshold), None
        )
        actual_trigger = row.get("trigger_layer")
        actual_trigger = None if actual_trigger is None else int(actual_trigger)
        if bool(row.get("triggered")) != (expected_trigger is not None):
            raise ValueError(f"{uid}: triggered flag is inconsistent with Stage-1 scores")
        if actual_trigger != expected_trigger:
            raise ValueError(f"{uid}: trigger layer is inconsistent with Stage-1 scores")
        for layer, action_row in enumerate(action_rows):
            expected_active = expected_trigger is not None and layer >= expected_trigger
            if (
                int(action_row.get("layer", -1)) != layer
                or str(action_row.get("action")) != actions[layer]
                or bool(action_row.get("active")) != expected_active
            ):
                raise ValueError(f"{uid}: action_rows are inconsistent at layer {layer}")

        expected_post_actions = 0 if expected_trigger is None else 28 - expected_trigger
        if int(row.get("post_trigger_actions", -1)) != expected_post_actions:
            raise ValueError(f"{uid}: post-trigger action count is inconsistent")
        expected_counts = Counter(actions[expected_trigger:]) if expected_trigger is not None else Counter()
        actual_counts = Counter(
            {
                str(action): int(count)
                for action, count in dict(row.get("post_trigger_action_counts", {})).items()
                if int(count)
            }
        )
        if actual_counts != expected_counts:
            raise ValueError(f"{uid}: post-trigger action counts are inconsistent")

        nonfull = _nonfull_rows(row)
        if bool(row.get("any_non_full")) != bool(nonfull):
            raise ValueError(f"{uid}: any_non_full is inconsistent with action trace")
        if int(row.get("non_full_count", -1)) != len(nonfull):
            raise ValueError(f"{uid}: non_full_count is inconsistent with action trace")
        expected_first = nonfull[0]["layer"] if nonfull else None
        if row.get("first_non_full_layer") != expected_first:
            raise ValueError(f"{uid}: first_non_full_layer is inconsistent")
        if nonfull and expected_trigger is None:
            raise ValueError(f"{uid}: non-FULL action exists without a trigger")
        expected_delay = None if expected_first is None else expected_first - int(expected_trigger)
        if row.get("trigger_to_first_non_full_delay") != expected_delay:
            raise ValueError(f"{uid}: trigger-to-intervention delay is inconsistent")
        if expected_delay is not None and expected_delay < 0:
            raise ValueError(f"{uid}: non-FULL action precedes Stage-1 trigger")
        if expected_transition in {"W→C", "C→W"} and not nonfull:
            raise ValueError(f"{uid}: correctness changed without a non-FULL action")
        families[str(row["benchmark_family"])] += 1
    return dict(families)


def summarize_funnel(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dense_w = [row for row in rows if not bool(row["dense_correct"])]
    dense_c = [row for row in rows if bool(row["dense_correct"])]
    triggered_w = [row for row in dense_w if bool(row["triggered"])]
    triggered_c = [row for row in dense_c if bool(row["triggered"])]
    nonfull_w = [row for row in triggered_w if bool(row["any_non_full"])]
    nonfull_c = [row for row in triggered_c if bool(row["any_non_full"])]
    w_to_c = sum(str(row["transition"]) == "W→C" for row in rows)
    c_to_w = sum(str(row["transition"]) == "C→W" for row in rows)
    return {
        "n": len(rows),
        "dense_w": len(dense_w),
        "triggered_w": len(triggered_w),
        "p_trigger_given_w": safe_rate(len(triggered_w), len(dense_w)),
        "triggered_w_nonfull": len(nonfull_w),
        "p_nonfull_given_trigger_w": safe_rate(len(nonfull_w), len(triggered_w)),
        "w_to_c": w_to_c,
        "p_w_to_c_given_w": safe_rate(w_to_c, len(dense_w)),
        "p_w_to_c_given_trigger_w": safe_rate(w_to_c, len(triggered_w)),
        "p_w_to_c_given_trigger_nonfull_w": safe_rate(w_to_c, len(nonfull_w)),
        "dense_c": len(dense_c),
        "triggered_c": len(triggered_c),
        "p_trigger_given_c": safe_rate(len(triggered_c), len(dense_c)),
        "triggered_c_nonfull": len(nonfull_c),
        "p_nonfull_given_trigger_c": safe_rate(len(nonfull_c), len(triggered_c)),
        "c_to_w": c_to_w,
        "p_c_to_w_given_c": safe_rate(c_to_w, len(dense_c)),
        "p_c_to_w_given_trigger_c": safe_rate(c_to_w, len(triggered_c)),
        "p_c_to_w_given_trigger_nonfull_c": safe_rate(c_to_w, len(nonfull_c)),
        "net": w_to_c - c_to_w,
    }


def summarize_action_behavior(
    rows: Sequence[Mapping[str, Any]], *, dense_correct: bool
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if bool(row["triggered"]) and bool(row["dense_correct"]) is dense_correct
    ]
    result: dict[str, Any] = {
        "dense_state": "C" if dense_correct else "W",
        "triggered": len(selected),
        "never_nonfull": sum(not bool(row["any_non_full"]) for row in selected),
        "any_nonfull": sum(bool(row["any_non_full"]) for row in selected),
    }
    for action in NONFULL_ACTIONS:
        key = action.lower()
        result[f"{key}_used"] = sum(action in row["actions"] for row in selected)
        result[f"first_{key}"] = sum(
            row.get("first_non_full_layer") is not None
            and str(row["actions"][int(row["first_non_full_layer"])]) == action
            for row in selected
        )
    result["multiple_action_types"] = sum(
        len({str(action) for action in row["actions"] if str(action) != "FULL"}) > 1
        for row in selected
    )
    return result


def answer_change_record(row: Mapping[str, Any]) -> dict[str, Any]:
    if str(row["transition"]) not in {"W→C", "C→W"}:
        raise ValueError("answer-change record requires W→C or C→W")
    nonfull = _nonfull_rows(row)
    first = nonfull[0]
    record = {
        "uid": str(row["uid"]),
        "benchmark": str(row["benchmark"]),
        "benchmark_family": str(row["benchmark_family"]),
        "dense_correct": bool(row["dense_correct"]),
        "dense_answer": row["dense_generated_answer"],
        "dense_score": row["dense_score"],
        "routed_correct": bool(row["routed_correct"]),
        "routed_answer": row["routed_generated_answer"],
        "routed_score": row["routed_score"],
        "gt_answer": row["answer"],
        "transition": str(row["transition"]),
        "trigger_layer": int(row["trigger_layer"]),
        "trigger_score": float(row["stage1_scores"][int(row["trigger_layer"])]),
        "stage1_max_score": float(row["stage1_max_score"]),
        "stage1_score_trajectory": [float(value) for value in row["stage1_scores"]],
        "first_non_full_layer": int(first["layer"]),
        "first_non_full_action": str(first["action"]),
        "non_full_actions": nonfull,
        "non_full_count": len(nonfull),
        "post_trigger_action_sequence": [
            {"layer": layer, "action": str(row["actions"][layer])}
            for layer in range(int(row["trigger_layer"]), 28)
        ],
        "trigger_to_first_non_full_delay": int(row["trigger_to_first_non_full_delay"]),
    }
    if str(row["transition"]) == "C→W":
        actions = Counter(item["action"] for item in nonfull)
        action_types = set(actions)
        classes = [
            "R1_SINGLE_INTERVENTION" if len(nonfull) == 1 else "R2_MULTIPLE_INTERVENTIONS",
            "R3_IMMEDIATE_INTERVENTION"
            if record["trigger_to_first_non_full_delay"] == 0
            else "R4_DELAYED_INTERVENTION",
        ]
        if "READ_ONLY" in action_types:
            classes.append("R5_READ_RELATED")
        if "WRITE_ONLY" in action_types:
            classes.append("R6_WRITE_RELATED")
        if actions["IGNORE"] > sum(actions[action] for action in ("READ_ONLY", "WRITE_ONLY")):
            classes.append("R7_IGNORE_DOMINATED")
        if len(action_types) > 1:
            classes.append("R8_OTHER_OR_MIXED")
        record["regression_classes"] = classes
    else:
        record["regression_classes"] = []
    return record


def classify_bottleneck(funnel: Mapping[str, Any]) -> str:
    """Apply fixed descriptive rules; this is not a causal classifier."""
    if int(funnel.get("triggered_w", 0)) + int(funnel.get("triggered_c", 0)) == 0:
        return "INACTIVE"
    nonfull_w = int(funnel.get("triggered_w_nonfull", 0))
    if nonfull_w >= 20 and int(funnel.get("w_to_c", 0)) == 0:
        return "TREATMENT_QUALITY_LIMITED"
    if int(funnel.get("c_to_w", 0)) >= 2 * max(1, int(funnel.get("w_to_c", 0))):
        return "PRESERVATION_LIMITED"
    if safe_rate(nonfull_w, int(funnel.get("triggered_w", 0))) is not None and safe_rate(
        nonfull_w, int(funnel.get("triggered_w", 0))
    ) < 0.1:
        return "INTERVENTION_LIMITED"
    if safe_rate(int(funnel.get("triggered_w", 0)), int(funnel.get("dense_w", 0))) is not None and safe_rate(
        int(funnel.get("triggered_w", 0)), int(funnel.get("dense_w", 0))
    ) < 0.05:
        return "ADMISSION_LIMITED"
    return "MIXED"
