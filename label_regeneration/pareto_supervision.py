"""Pure helpers for filtering compute-dominated binary routes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from .bce_geometry import as_mask, polar_route_weights


@dataclass(frozen=True)
class ParetoFilterResult:
    retained: list[dict[str, Any]]
    removed_witnesses: dict[str, str]


def _validated_route(route: dict[str, Any], expected_width: int) -> tuple[str, int, float]:
    mask = route.get("mask")
    if (
        not isinstance(mask, list)
        or len(mask) != expected_width
        or any(bit not in (0, 1) for bit in mask)
    ):
        raise ValueError(f"route must contain an exact {expected_width}-bit binary mask")
    key = "".join(str(int(bit)) for bit in mask)
    if route.get("key", key) != key:
        raise ValueError("route key does not match its mask")
    on_count = sum(int(bit) for bit in mask)
    if int(route.get("num_visual_on_layers", on_count)) != on_count:
        raise ValueError("stored visual ON count does not match the route mask")
    score = float(route["score"])
    if not math.isfinite(score):
        raise ValueError("route score must be finite")
    return key, on_count, score


def filter_pareto_routes(
    routes: Sequence[dict[str, Any]], *, expected_width: int = 28
) -> ParetoFilterResult:
    """Retain routes not dominated by an equal/higher-score, lower-ON route.

    Parent-manifest order is preserved. Every removed route is paired with a
    deterministic retained witness, preferring lower cost, higher score, then
    lexicographically smaller mask keys.
    """
    if expected_width < 1:
        raise ValueError("expected_width must be positive")
    validated = [_validated_route(route, expected_width) for route in routes]
    keys = [item[0] for item in validated]
    if len(keys) != len(set(keys)):
        raise ValueError("route set contains a duplicate complete mask")

    retained_indices = []
    for index, (_, cost, score) in enumerate(validated):
        dominated = any(
            other_index != index and other_cost < cost and other_score >= score
            for other_index, (_, other_cost, other_score) in enumerate(validated)
        )
        if not dominated:
            retained_indices.append(index)

    retained_index_set = set(retained_indices)
    witnesses: dict[str, str] = {}
    for index, (key, cost, score) in enumerate(validated):
        if index in retained_index_set:
            continue
        candidates = [
            (other_cost, -other_score, other_key, other_index)
            for other_index, (other_key, other_cost, other_score) in enumerate(validated)
            if other_index in retained_index_set
            and other_cost < cost
            and other_score >= score
        ]
        if not candidates:
            raise RuntimeError(f"removed route {key} has no retained dominance witness")
        witnesses[key] = min(candidates)[2]

    return ParetoFilterResult(
        retained=[dict(routes[index]) for index in retained_indices],
        removed_witnesses=witnesses,
    )


def build_pareto_record(
    source: dict[str, Any], *, expected_width: int = 28
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build one population-preserving Pareto supervision record."""
    routes = source.get("valid_routes")
    if not isinstance(routes, list):
        raise ValueError("source record valid_routes must be a list")
    if int(source.get("selected_valid_route_count", len(routes))) != len(routes):
        raise ValueError("selected valid-route count does not match the route list")
    for route in routes:
        _, _, score = _validated_route(route, expected_width)
        threshold = float(route.get("correctness_threshold", 0.0))
        if score < threshold:
            raise ValueError("source valid route does not satisfy its correctness threshold")

    result = filter_pareto_routes(routes, expected_width=expected_width)
    retained = []
    retained_masks = [as_mask(route["mask"]) for route in result.retained]
    retained_weights = polar_route_weights(retained_masks) if retained_masks else []
    for route, weight in zip(result.retained, retained_weights):
        current = dict(route)
        if "weight" in current:
            current["source_selected_weight"] = float(current["weight"])
        current["pareto_polar_weight"] = weight
        retained.append(current)

    status = str(source.get("current_all_on_status") or "")
    if status not in {"correct", "wrong"}:
        raise ValueError("current_all_on_status must be correct or wrong")
    if not retained:
        group = "D"
    elif status == "wrong":
        group = "A"
    elif any(sum(mask) < expected_width for mask in retained_masks):
        group = "B"
    else:
        group = "C"

    output = dict(source)
    output.update(
        {
            "schema_version": "binary_pareto_predictor_manifest_v1",
            "original_selected_valid_route_count": len(routes),
            "pareto_efficient_route_count": len(retained),
            "selected_valid_route_count": len(retained),
            "original_valid_mask_keys": [item[0] for item in map(
                lambda route: _validated_route(route, expected_width), routes
            )],
            "supervision_group": group,
            "valid_routes": retained,
        }
    )
    return output, result.removed_witnesses
