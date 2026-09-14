"""Pure helpers for the frozen Stage-2 MCTS failure diagnosis."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import random
from typing import Any, Mapping, Sequence


ACTIONS = ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")


def entering_state_key(uid: str, actions: Sequence[str], layer: int) -> str:
    """Identify an exact entering state by UID and its action prefix."""
    index = int(layer)
    if index < 0 or index >= len(actions):
        raise ValueError("layer is outside the route")
    prefix = "|".join(str(action) for action in actions[:index])
    return hashlib.sha256(f"{uid}\n{index}\n{prefix}".encode()).hexdigest()


def observed_successful_action_sets(
    routes: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    observed: dict[str, set[str]] = defaultdict(set)
    for route in routes:
        actions = [str(action) for action in route["actions"]]
        start = int(route["activation_layer"])
        for layer in range(start, len(actions)):
            observed[entering_state_key(str(route["uid"]), actions, layer)].add(actions[layer])
    return {key: tuple(action for action in ACTIONS if action in values) for key, values in observed.items()}


def route_complexity_bin(non_full_count: int) -> str:
    count = int(non_full_count)
    if count < 1:
        raise ValueError("route complexity requires at least one non-FULL action")
    return str(count) if count < 4 else "4+"


def intervention_index(actions: Sequence[str], layer: int, start: int) -> int | None:
    if str(actions[int(layer)]) == "FULL":
        return None
    interventions = [
        index for index in range(int(start), int(layer) + 1) if str(actions[index]) != "FULL"
    ]
    return len(interventions)


def intervention_index_bin(index: int) -> str:
    value = int(index)
    if value < 1:
        raise ValueError("intervention index must be positive")
    return "first" if value == 1 else ("second" if value == 2 else "third+")


def forcing_boundary(actions: Sequence[str], trigger_layer: int, forced_interventions: int) -> int:
    trigger = int(trigger_layer)
    count = int(forced_interventions)
    if count < 0:
        raise ValueError("forced intervention count cannot be negative")
    if count == 0:
        return trigger - 1
    layers = [index for index in range(trigger, len(actions)) if str(actions[index]) != "FULL"]
    if count > len(layers):
        raise ValueError("forcing depth exceeds route interventions")
    return layers[count - 1]


def first_deviation_class(
    actions: Sequence[str], predictions: Sequence[str], start: int
) -> tuple[str, int | None]:
    if len(actions) != len(predictions):
        raise ValueError("action and prediction trajectories differ in length")
    begin = int(start)
    non_full = [index for index in range(begin, len(actions)) if str(actions[index]) != "FULL"]
    if not non_full:
        first = None
    else:
        first = non_full[0]
    deviation = next(
        (
            index
            for index in range(begin, len(actions))
            if str(actions[index]) != str(predictions[index])
        ),
        None,
    )
    if deviation is None:
        return "NO_DEVIATION", None
    if first is None or deviation < first:
        return "BEFORE_FIRST_INTERVENTION", deviation
    if deviation == first:
        return "AT_FIRST_INTERVENTION", deviation
    if str(actions[deviation]) != "FULL":
        return "AT_SECOND_OR_LATER_INTERVENTION", deviation
    return "BETWEEN_INTERVENTIONS", deviation


def action_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"states": 0}
    targets = [str(row["target_action"]) for row in rows]
    predictions = [str(row["predicted_action"]) for row in rows]
    target_counts = Counter(targets)
    correct = Counter(target for target, pred in zip(targets, predictions) if target == pred)
    non_full_total = sum(target_counts[action] for action in ACTIONS[1:])
    non_full_correct = sum(correct[action] for action in ACTIONS[1:])
    return {
        "states": len(rows),
        "action_accuracy": sum(target == pred for target, pred in zip(targets, predictions))
        / len(rows),
        "full_recall": correct["FULL"] / target_counts["FULL"] if target_counts["FULL"] else None,
        "non_full_recall": non_full_correct / non_full_total if non_full_total else None,
        **{
            f"{action.lower()}_recall": correct[action] / target_counts[action]
            if target_counts[action]
            else None
            for action in ACTIONS[1:]
        },
        "mean_target_probability": sum(float(row["target_probability"]) for row in rows)
        / len(rows),
        "mean_target_vs_full_margin": sum(float(row["target_vs_full_margin"]) for row in rows)
        / len(rows),
        "mean_full_probability": sum(float(row["full_probability"]) for row in rows)
        / len(rows),
        "mean_best_non_full_probability": sum(
            float(row["best_non_full_probability"]) for row in rows
        )
        / len(rows),
    }


def paired_uid_bootstrap(
    pairs: Mapping[str, tuple[float, float]], *, seed: int, replicates: int
) -> dict[str, float | int]:
    if not pairs:
        raise ValueError("paired bootstrap requires at least one UID")
    ordered = sorted(pairs)
    deltas = [float(pairs[uid][1]) - float(pairs[uid][0]) for uid in ordered]
    rng = random.Random(int(seed))
    draws = []
    for _ in range(int(replicates)):
        draws.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    draws.sort()

    def quantile(probability: float) -> float:
        position = probability * (len(draws) - 1)
        lower = int(position)
        upper = min(lower + 1, len(draws) - 1)
        fraction = position - lower
        return draws[lower] * (1.0 - fraction) + draws[upper] * fraction

    return {
        "uids": len(ordered),
        "replicates": int(replicates),
        "mean_delta_B_minus_A": sum(deltas) / len(deltas),
        "ci95_low": quantile(0.025),
        "ci95_high": quantile(0.975),
    }


def matched_mode_uid_pairs(
    rows: Sequence[Mapping[str, Any]], baseline_mode: str, comparison_mode: str
) -> dict[str, tuple[float, float]]:
    """Build UID pairs using only routes present in both forcing modes."""
    by_mode_route: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        mode = str(row["mode"])
        route_id = str(row["route_id"])
        if route_id in by_mode_route[mode]:
            raise ValueError(f"duplicate {mode} result for route {route_id}")
        by_mode_route[mode][route_id] = row
    baseline = by_mode_route[str(baseline_mode)]
    comparison = by_mode_route[str(comparison_mode)]
    common_routes = sorted(set(baseline) & set(comparison))
    if not common_routes:
        raise ValueError("forcing-mode comparison has no common routes")
    by_uid: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for route_id in common_routes:
        baseline_row = baseline[route_id]
        comparison_row = comparison[route_id]
        if str(baseline_row["uid"]) != str(comparison_row["uid"]):
            raise ValueError(f"UID mismatch for route {route_id}")
        by_uid[str(baseline_row["uid"])].append(
            (float(bool(baseline_row["correct"])), float(bool(comparison_row["correct"])))
        )
    return {
        uid: (
            sum(value[0] for value in values) / len(values),
            sum(value[1] for value in values) / len(values),
        )
        for uid, values in sorted(by_uid.items())
    }
