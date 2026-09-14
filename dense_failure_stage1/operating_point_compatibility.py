"""Pure controls for robust Stage-1 operating-point and label compatibility audits."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from dense_failure_stage1.threshold_calibration import first_trigger_layers


TRANSITIONS = (
    "SAME_TRIGGER",
    "NEW_EARLIER",
    "NEW_LATER",
    "OLD_TRIGGER_ONLY",
    "NEW_TRIGGER_ONLY",
    "NEITHER_TRIGGER",
)


def transition_category(old_layer: int | None, new_layer: int | None) -> tuple[str, int | None]:
    if old_layer is None and new_layer is None:
        return "NEITHER_TRIGGER", None
    if old_layer is None:
        return "NEW_TRIGGER_ONLY", None
    if new_layer is None:
        return "OLD_TRIGGER_ONLY", None
    delta = int(new_layer) - int(old_layer)
    if delta == 0:
        return "SAME_TRIGGER", 0
    return ("NEW_EARLIER" if delta < 0 else "NEW_LATER"), delta


def structural_route_status(
    *, new_trigger_layer: int | None, first_non_full_layer: int
) -> str:
    if first_non_full_layer not in range(28):
        raise ValueError("first non-FULL layer must lie in 0..27")
    if new_trigger_layer is None:
        return "NOT_NEW_TRIGGERED"
    if int(new_trigger_layer) not in range(28):
        raise ValueError("new trigger layer must lie in 0..27")
    return (
        "STRUCTURALLY_COMPATIBLE"
        if int(new_trigger_layer) <= int(first_non_full_layer)
        else "STRUCTURALLY_INCOMPATIBLE"
    )


def build_trigger_rows(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[Sequence[float]] | np.ndarray,
    operating_points: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(rows), 28) or not np.isfinite(values).all():
        raise ValueError("score matrix must be finite [N, 28]")
    if len({str(row["uid"]) for row in rows}) != len(rows):
        raise ValueError("trigger source contains duplicate UIDs")
    output = []
    for point in operating_points:
        name = str(point["operating_point"])
        threshold = float(point["threshold"])
        first = first_trigger_layers(values, threshold)
        for index, row in enumerate(rows):
            layer = int(first[index])
            output.append(
                {
                    "uid": str(row["uid"]),
                    "dataset": str(row["dataset"]),
                    "source_regime": str(row["source_regime"]),
                    "dense_correct": not bool(row["current_dense_wrong"]),
                    "dense_wrong": bool(row["current_dense_wrong"]),
                    "image_group_id": str(row["image_group_id"]),
                    "threshold_name": name,
                    "threshold_value": threshold,
                    "triggered": layer >= 0,
                    "first_trigger_layer": layer if layer >= 0 else None,
                    "score_at_trigger": float(values[index, layer]) if layer >= 0 else None,
                    "max_score": float(values[index].max()),
                }
            )
    return output


def select_retained_operating_points(
    named_points: Sequence[Mapping[str, Any]],
    *,
    default_names: Sequence[str],
    minimum_recall_increment: float,
    maximum_points: int,
) -> tuple[list[str], str]:
    by_name = {str(row["operating_point"]): row for row in named_points}
    if len(by_name) != len(named_points) or "P98" not in by_name:
        raise ValueError("named operating points must be unique and include P98")
    if maximum_points < 1 or not 0 <= minimum_recall_increment <= 1:
        raise ValueError("invalid retention controls")

    def stable(name: str) -> bool:
        return bool(by_name[name]["fold_stable"])

    def recall(name: str) -> float:
        return float(by_name[name]["pooled_w_recall"])

    default = list(default_names)
    default_ok = (
        len(default) <= maximum_points
        and all(name in by_name and stable(name) for name in default)
        and len({float(by_name[name]["threshold"]) for name in default}) == len(default)
        and all(
            recall(right) - recall(left) >= minimum_recall_increment
            for left, right in zip(default, default[1:])
        )
    )
    if default_ok:
        return default, "prospective_default_passed_stability_distinctness_and_recall_spacing"

    stable_names = [
        name
        for name, row in by_name.items()
        if stable(name) and float(row["preservation_target"]) <= 0.98
    ]
    if "P98" not in stable_names:
        raise RuntimeError("P98 is not fold-stable; cannot retain the frozen reference")
    ordered = sorted(
        stable_names, key=lambda name: float(by_name[name]["preservation_target"]), reverse=True
    )
    retained = ["P98"]
    permissive = next(
        (
            name
            for name in reversed(ordered)
            if name != "P98" and recall(name) - recall("P98") >= minimum_recall_increment
        ),
        None,
    )
    if permissive is not None and maximum_points > 1:
        retained.append(permissive)
    if permissive is not None and maximum_points > 2:
        midpoint = (recall("P98") + recall(permissive)) / 2.0
        middle = [
            name
            for name in ordered
            if name not in retained
            and recall(name) - recall("P98") >= minimum_recall_increment
            and recall(permissive) - recall(name) >= minimum_recall_increment
        ]
        if middle:
            retained.append(min(middle, key=lambda name: (abs(recall(name) - midpoint), name)))
    retained.sort(
        key=lambda name: float(by_name[name]["preservation_target"]), reverse=True
    )
    return retained, "deterministic_stable_recall_spaced_fallback"


def compatibility_category(
    *,
    has_replay_single: bool,
    has_replay_mcts: bool,
    prior_label: str | None,
) -> str:
    if has_replay_single:
        return "EXISTING_SINGLE_REUSABLE"
    if has_replay_mcts:
        return "EXISTING_MCTS_REUSABLE"
    if prior_label in {"SINGLE_FIXABLE", "MCTS_ONLY_FIXABLE"}:
        return "EXISTING_LABEL_BUT_INCOMPATIBLE"
    if prior_label == "UNRESOLVED":
        return "EXISTING_UNRESOLVED"
    if prior_label is None:
        return "NEW_TRIGGER_NO_EXISTING_LABEL"
    raise ValueError(f"unsupported prior label: {prior_label}")
