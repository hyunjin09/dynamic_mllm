"""Pure contracts for bounded Stage-1 gate treatment-correctability analysis."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np


TREATMENT_ACTIONS = ("READ_ONLY", "WRITE_ONLY", "IGNORE")
GATES = ("shared_random4", "independent_sequential", "fixed_l27")


def depth_bin(layer: int) -> str:
    value = int(layer)
    if 0 <= value <= 8:
        return "early"
    if 9 <= value <= 18:
        return "middle"
    if 19 <= value <= 27:
        return "late"
    raise ValueError("trigger layer must lie in 0..27")


def _stable_key(seed: int, uid: str, start_layer: int, route: Sequence[str]) -> str:
    payload = f"{seed}:{uid}:{start_layer}:{'|'.join(route)}".encode()
    return sha256(payload).hexdigest()


def bounded_route_panel(
    *,
    uid: str,
    start_layer: int,
    seed: int,
    pair_panel_size: int = 12,
    layer_count: int = 28,
) -> list[dict[str, Any]]:
    """Return all single interventions plus a fixed-size seeded pair panel."""

    start = int(start_layer)
    if not uid or not 0 <= start < layer_count or pair_panel_size < 0:
        raise ValueError("invalid bounded route-panel contract")
    full = ("FULL",) * layer_count
    rows = []
    for layer in range(start, layer_count):
        for action in TREATMENT_ACTIONS:
            route = list(full)
            route[layer] = action
            rows.append(
                {
                    "candidate_index": len(rows),
                    "search_stage": "immediate_single" if layer == start else "suffix_single",
                    "actions": route,
                    "route_key": "|".join(route),
                    "changed_layers": [layer],
                    "changed_actions": [action],
                }
            )

    pairs = []
    for left in range(start, layer_count):
        for right in range(left + 1, layer_count):
            for left_action in TREATMENT_ACTIONS:
                for right_action in TREATMENT_ACTIONS:
                    route = list(full)
                    route[left] = left_action
                    route[right] = right_action
                    pairs.append(
                        {
                            "search_stage": "seeded_pair",
                            "actions": route,
                            "route_key": "|".join(route),
                            "changed_layers": [left, right],
                            "changed_actions": [left_action, right_action],
                        }
                    )
    pairs.sort(
        key=lambda row: (
            _stable_key(seed, uid, start, row["actions"]),
            row["route_key"],
        )
    )
    for row in pairs[:pair_panel_size]:
        rows.append({"candidate_index": len(rows), **row})
    if len({row["route_key"] for row in rows}) != len(rows):
        raise RuntimeError("bounded route panel contains duplicates")
    if any(any(action != "FULL" for action in row["actions"][:start]) for row in rows):
        raise RuntimeError("bounded route panel modified the frozen dense prefix")
    return rows


def trigger_manifest_rows(
    records: Sequence[Mapping[str, Any]],
    scores: Sequence[Sequence[float]] | np.ndarray,
    first_layers: Sequence[int] | np.ndarray,
    *,
    gate: str,
    split: str,
) -> list[dict[str, Any]]:
    values = np.asarray(scores, dtype=np.float64)
    first = np.asarray(first_layers, dtype=np.int64)
    if gate not in GATES or split not in {"val", "test"}:
        raise ValueError("invalid gate trigger-manifest identity")
    if values.shape != (len(records), 28) or first.shape != (len(records),):
        raise ValueError("trigger-manifest score shapes are invalid")
    if not np.isfinite(values).all() or not np.isin(first, np.arange(-1, 28)).all():
        raise ValueError("trigger-manifest scores/layers are invalid")
    output = []
    for record, trajectory, layer in zip(records, values, first):
        triggered = int(layer) >= 0
        wrong = bool(record["current_dense_wrong"])
        output.append(
            {
                "uid": str(record["uid"]),
                "split": split,
                "dataset": str(record["dataset"]),
                "image_group_id": str(record["image_group_id"]),
                "current_dense_wrong": wrong,
                "gate": gate,
                "triggered": triggered,
                "trigger_layer": int(layer) if triggered else None,
                "treatment_start_layer": 0 if gate == "fixed_l27" and triggered else (int(layer) if triggered else None),
                "trigger_depth_bin": depth_bin(int(layer)) if triggered else None,
                "risk_at_trigger": float(trajectory[int(layer)]) if triggered else None,
                "risk_trajectory": trajectory.tolist(),
                "cohort": (
                    "triggered_dense_wrong"
                    if triggered and wrong
                    else "triggered_dense_correct"
                    if triggered
                    else "non_triggered_dense_wrong"
                    if wrong
                    else "non_triggered_dense_correct"
                ),
            }
        )
    if len({row["uid"] for row in output}) != len(output):
        raise ValueError("trigger manifest contains duplicate UIDs")
    return output


def summarize_regime(
    rows: Sequence[Mapping[str, Any]],
    *,
    total_dense_wrong: int,
) -> dict[str, Any]:
    if not rows or total_dense_wrong < 1:
        raise ValueError("correctability summary requires nonempty rows")
    wrong = [row for row in rows if bool(row["current_dense_wrong"])]
    correct = [row for row in rows if not bool(row["current_dense_wrong"])]
    wrong_correctable = sum(bool(row["correctable"]) for row in wrong)
    correct_preservable = sum(bool(row["correctable"]) for row in correct)
    single = sum(bool(row["single_intervention_success"]) for row in wrong)
    evaluations = [int(row["route_evaluations"]) for row in rows]
    return {
        "triggered_records": len(rows),
        "triggered_wrong": len(wrong),
        "triggered_correct": len(correct),
        "wrong_correctable_single": single,
        "wrong_correctable_bounded": wrong_correctable,
        "correct_preservable_bounded": correct_preservable,
        "correctable_at_single_intervention": single / len(wrong) if wrong else None,
        "triggered_wrong_correctability": wrong_correctable / len(wrong) if wrong else None,
        "triggered_correct_preservability": correct_preservable / len(correct) if correct else None,
        "population_oracle_rescue": wrong_correctable / total_dense_wrong,
        "mean_route_evaluations": float(np.mean(evaluations)),
        "median_route_evaluations": float(np.median(evaluations)),
        "total_route_evaluations": sum(evaluations),
        "mean_elapsed_seconds": float(np.mean([float(row["elapsed_seconds"]) for row in rows])),
    }


def validate_execution_coverage(
    expected_keys: Sequence[tuple[str, str, str]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    expected = Counter(expected_keys)
    actual = Counter((str(row["split"]), str(row["gate"]), str(row["uid"])) for row in rows)
    if expected != actual:
        missing = list((expected - actual).elements())[:10]
        extra = list((actual - expected).elements())[:10]
        raise ValueError(f"execution coverage mismatch: missing={missing} extra={extra}")
    if any(not bool(row.get("passed")) for row in rows):
        raise ValueError("execution coverage contains failed final records")
