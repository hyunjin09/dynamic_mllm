from __future__ import annotations

from collections import Counter
import math
import statistics
from typing import Any, Iterable


ACTIONS = ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")


def _percentile(values: list[int | float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: list[int | float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def _new_accumulator(layer_count: int) -> dict[str, Any]:
    return {
        "layer_count": layer_count,
        "counts": Counter(),
        "w2c_actions": Counter(),
        "w2c_layer_actions": [Counter() for _ in range(layer_count)],
        "w2c_branch_counts": [],
        "w2c_max_branch_counts": [],
        "w2c_source_off_counts": [],
        "w2c_both_events": 0,
        "w2c_branching_routes": 0,
        "w2c_all_off_routes": 0,
        "w2c_all_off_actions": Counter(),
        "c2c_actions": Counter(),
        "c2c_layer_actions": [Counter() for _ in range(layer_count)],
        "c2c_source_off_counts": [],
    }


def _add_record(acc: dict[str, Any], record: dict[str, Any]) -> None:
    counts = acc["counts"]
    route_type = str(record["route_type"])
    counts["samples"] += 1
    counts[f"{route_type.lower()}_samples"] += 1
    counts["source_routes"] += int(record.get("source_positive_route_count", 0))
    counts["source_replay_valid_routes"] += int(
        record.get("source_route_replay_valid_count", 0)
    )
    counts["source_replay_failure_routes"] += int(
        record.get("source_route_replay_failure_count", 0)
    )
    unique = record.get("unique_valid_four_action_routes", [])
    counts["unique_valid_routes"] += len(unique)
    counts[f"{route_type.lower()}_unique_valid_routes"] += len(unique)

    for conversion in record.get("raw_conversions", []):
        if conversion.get("status") != "converted":
            continue
        counts[f"{route_type.lower()}_source_routes"] += 1
        branches = conversion.get("final_branches", [])
        counts[f"{route_type.lower()}_final_branch_occurrences"] += len(branches)
        if route_type == "W2C":
            maximum = int(conversion.get("maximum_branch_count", len(branches)))
            acc["w2c_branch_counts"].append(len(branches))
            acc["w2c_max_branch_counts"].append(maximum)
            acc["w2c_source_off_counts"].append(int(conversion["source_off_count"]))
            if maximum > 1:
                acc["w2c_branching_routes"] += 1
            acc["w2c_both_events"] += sum(
                int(step.get("both_partial_correct_count", 0))
                for step in conversion.get("steps", [])
            )
            is_all_off = bool(conversion.get("all_off_seed"))
            if is_all_off:
                acc["w2c_all_off_routes"] += 1
            off_layers = [
                index
                for index, value in enumerate(conversion["source_binary_route"])
                if not bool(value)
            ]
            for branch in branches:
                route = branch["route"]
                for layer in off_layers:
                    action = str(route[layer])
                    acc["w2c_actions"][action] += 1
                    acc["w2c_layer_actions"][layer][action] += 1
                    if is_all_off:
                        acc["w2c_all_off_actions"][action] += 1
        elif route_type == "C2C":
            acc["c2c_source_off_counts"].append(int(conversion["source_off_count"]))
            for branch in branches:
                for layer, action in enumerate(branch["route"]):
                    acc["c2c_actions"][str(action)] += 1
                    acc["c2c_layer_actions"][layer][str(action)] += 1
        else:
            raise ValueError(f"unexpected route type: {route_type}")


def _action_counts(counter: Counter) -> dict[str, int]:
    return {action: int(counter[action]) for action in ACTIONS}


def _finalize(acc: dict[str, Any]) -> dict[str, Any]:
    counts = {key: int(value) for key, value in acc["counts"].items()}
    for key in (
        "samples",
        "w2c_samples",
        "c2c_samples",
        "source_routes",
        "source_replay_valid_routes",
        "source_replay_failure_routes",
        "unique_valid_routes",
        "w2c_unique_valid_routes",
        "c2c_unique_valid_routes",
        "w2c_source_routes",
        "c2c_source_routes",
        "w2c_final_branch_occurrences",
        "c2c_final_branch_occurrences",
    ):
        counts.setdefault(key, 0)
    w2c_occurrences = counts["w2c_final_branch_occurrences"]
    return {
        "counts": counts,
        "w2c": {
            "off_position_final_actions": _action_counts(acc["w2c_actions"]),
            "off_position_final_action_fractions": {
                action: (
                    acc["w2c_actions"][action] / sum(acc["w2c_actions"].values())
                    if sum(acc["w2c_actions"].values())
                    else None
                )
                for action in ACTIONS
            },
            "off_position_actions_by_layer": [
                _action_counts(counter) for counter in acc["w2c_layer_actions"]
            ],
            "final_branches_per_source_route": _distribution(acc["w2c_branch_counts"]),
            "maximum_active_branches_per_source_route": _distribution(
                acc["w2c_max_branch_counts"]
            ),
            "source_off_count": _distribution(acc["w2c_source_off_counts"]),
            "source_routes_with_branching": int(acc["w2c_branching_routes"]),
            "both_partial_branch_events": int(acc["w2c_both_events"]),
            "all_off_seed_source_routes": int(acc["w2c_all_off_routes"]),
            "all_off_seed_final_actions": _action_counts(acc["w2c_all_off_actions"]),
            "deduplication_ratio_unique_per_branch_occurrence": (
                counts["w2c_unique_valid_routes"] / w2c_occurrences
                if w2c_occurrences
                else None
            ),
        },
        "c2c": {
            "action_counts": _action_counts(acc["c2c_actions"]),
            "actions_by_layer": [
                _action_counts(counter) for counter in acc["c2c_layer_actions"]
            ],
            "source_off_count": _distribution(acc["c2c_source_off_counts"]),
        },
    }


def aggregate_records(
    records: Iterable[dict[str, Any]], *, layer_count: int
) -> dict[str, Any]:
    combined = _new_accumulator(layer_count)
    datasets: dict[str, dict[str, Any]] = {}
    for record in records:
        dataset = str(record["dataset"])
        _add_record(combined, record)
        _add_record(datasets.setdefault(dataset, _new_accumulator(layer_count)), record)
    return {
        "schema_version": "exact_sequential_four_action_aggregate_v1",
        "combined": _finalize(combined),
        "by_dataset": {
            dataset: _finalize(accumulator)
            for dataset, accumulator in sorted(datasets.items())
        },
    }
