"""Deterministic absolute VISUAL_ON-cap supervision transforms."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence


def filter_routes_by_cap(
    routes: Sequence[dict[str, Any]], *, cap: int, expected_width: int = 28
) -> list[dict[str, Any]]:
    """Return all parent routes at or below ``cap``, preserving parent order."""
    if not 0 <= cap <= expected_width:
        raise ValueError("cap must lie within the mask width")
    seen: set[str] = set()
    retained: list[dict[str, Any]] = []
    for route in routes:
        mask = route.get("mask")
        if not isinstance(mask, list) or len(mask) != expected_width or any(bit not in (0, 1) for bit in mask):
            raise ValueError("every route must contain one exact-width binary mask")
        key = str(route.get("key") or "".join(str(int(bit)) for bit in mask))
        if key in seen:
            raise ValueError("duplicate route mask")
        seen.add(key)
        on_count = sum(int(bit) for bit in mask)
        if int(route.get("num_visual_on_layers", on_count)) != on_count:
            raise ValueError("route VISUAL_ON count does not match its mask")
        if on_count <= cap:
            retained.append(deepcopy(route))
    return retained


def build_cap_record(
    source: dict[str, Any],
    *,
    cap: int,
    common_eligible: bool,
    expected_width: int = 28,
) -> dict[str, Any]:
    """Build one matched-population record without inventing fallback routes."""
    original = list(source.get("valid_routes") or [])
    native = filter_routes_by_cap(original, cap=cap, expected_width=expected_width)
    selected = native if common_eligible else []
    record = deepcopy(source)
    record["schema_version"] = "binary_cap_predictor_manifest_v1"
    record["cap_visual_on_layers"] = cap
    record["common_eligible_cap18"] = bool(common_eligible)
    record["original_selected_valid_route_count"] = len(original)
    record["original_valid_mask_keys"] = [str(route["key"]) for route in original]
    record["cap_surviving_route_count_native"] = len(native)
    record["supervision_route_count"] = len(selected)
    record["valid_routes"] = selected
    if not selected:
        group = "D"
    elif str(source.get("current_all_on_status")) == "wrong":
        group = "A"
    else:
        group = "B"
    record["supervision_group"] = group
    return record
