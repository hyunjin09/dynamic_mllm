"""Pure aggregation contracts for the Stage-2 single-label audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence


ACTIONS = ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")
CORRECTIVE_ACTIONS = ACTIONS[1:]


def validate_single_routes(
    routes: Sequence[Mapping[str, Any]], *, contract_sha256: str
) -> None:
    """Fail closed unless every row is a valid replayed single intervention."""

    seen_route_ids: set[str] = set()
    for row in routes:
        route_id = str(row["route_id"])
        if route_id in seen_route_ids:
            raise ValueError(f"duplicate route ID: {route_id}")
        seen_route_ids.add(route_id)
        if str(row.get("contract_sha256")) != contract_sha256:
            raise ValueError(f"contract mismatch for route {route_id}")
        if row.get("route_source") != "single":
            raise ValueError(f"non-single route in Corpus B: {route_id}")
        if int(row.get("non_full_count", -1)) != 1:
            raise ValueError(f"route must contain exactly one non-FULL action: {route_id}")
        trigger = int(row["trigger_layer"])
        intervention = int(row["intervention_layer"])
        if intervention < trigger:
            raise ValueError(f"intervention occurs before trigger: {route_id}")
        action = str(row["intervention_action"])
        actions = list(map(str, row["actions"]))
        if len(actions) != 28 or action not in CORRECTIVE_ACTIONS:
            raise ValueError(f"invalid four-action route: {route_id}")
        non_full = [(layer, value) for layer, value in enumerate(actions) if value != "FULL"]
        if non_full != [(intervention, action)]:
            raise ValueError(f"route/action metadata mismatch: {route_id}")
        if not bool(row.get("final_lmms_correct")) or not bool(
            row.get("replay_token_parity")
        ):
            raise ValueError(f"route is not a correct exact replay: {route_id}")


def _routes_by_uid(
    routes: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in routes:
        grouped[str(row["uid"])].append(row)
    return dict(grouped)


def sample_weighted_action_distribution(
    routes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Give every sample total weight one, divided uniformly over its routes."""

    grouped = _routes_by_uid(routes)
    counts = Counter({action: 0.0 for action in CORRECTIVE_ACTIONS})
    for sample_routes in grouped.values():
        weight = 1.0 / len(sample_routes)
        for row in sample_routes:
            counts[str(row["intervention_action"])] += weight
    total = float(len(grouped))
    return [
        {
            "action": action,
            "weighted_count": float(counts[action]),
            "weighted_fraction": float(counts[action] / total) if total else 0.0,
            "weighting_unit": "sample",
        }
        for action in CORRECTIVE_ACTIONS
    ]


def classify_immediacy(
    routes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for uid, sample_routes in sorted(_routes_by_uid(routes).items()):
        trigger_layers = {int(row["trigger_layer"]) for row in sample_routes}
        if len(trigger_layers) != 1:
            raise ValueError(f"sample has inconsistent trigger layers: {uid}")
        trigger = next(iter(trigger_layers))
        interventions = sorted(int(row["intervention_layer"]) for row in sample_routes)
        immediate = trigger in interventions
        rows.append(
            {
                "uid": uid,
                "dataset": str(sample_routes[0]["dataset"]),
                "trigger_layer": trigger,
                "trigger_depth_bin": str(sample_routes[0]["trigger_depth_bin"]),
                "classification": (
                    "IMMEDIATE_FIXABLE" if immediate else "DELAYED_ONLY_FIXABLE"
                ),
                "successful_routes": len(sample_routes),
                "minimum_delay": min(layer - trigger for layer in interventions),
            }
        )
    return rows


def same_layer_action_ambiguity(
    routes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in routes:
        grouped[(str(row["uid"]), int(row["intervention_layer"]))].append(row)
    rows = []
    for (uid, layer), layer_routes in sorted(grouped.items()):
        actions = sorted({str(row["intervention_action"]) for row in layer_routes})
        rows.append(
            {
                "uid": uid,
                "dataset": str(layer_routes[0]["dataset"]),
                "trigger_layer": int(layer_routes[0]["trigger_layer"]),
                "trigger_depth_bin": str(layer_routes[0]["trigger_depth_bin"]),
                "intervention_layer": layer,
                "observed_successful_action_set": "|".join(actions),
                "observed_successful_action_count": len(actions),
                "ambiguity_class": (
                    "MULTIPLE_OBSERVED_ACTIONS"
                    if len(actions) > 1
                    else "SINGLE_OBSERVED_ACTION"
                ),
            }
        )
    return rows


def _scheme_counts_for_route(
    row: Mapping[str, Any], scheme: str
) -> Counter[str]:
    trigger = int(row["trigger_layer"])
    intervention = int(row["intervention_layer"])
    action = str(row["intervention_action"])
    if scheme == "S0":
        full = 27 - trigger
    elif scheme == "S1":
        full = min(2, intervention - trigger) + min(2, 27 - intervention)
    elif scheme.startswith("S2_K"):
        k = int(scheme.removeprefix("S2_K"))
        full = min(k, 27 - trigger)
    else:
        raise ValueError(f"unsupported sampling scheme: {scheme}")
    return Counter({"FULL": full, action: 1})


def _sampling_row(
    name: str,
    counts: Mapping[str, float],
    *,
    weighting_unit: str,
) -> dict[str, Any]:
    values = {action: float(counts.get(action, 0.0)) for action in ACTIONS}
    non_full = sum(values[action] for action in CORRECTIVE_ACTIONS)
    total = values["FULL"] + non_full
    return {
        "scheme": name,
        "weighting_unit": weighting_unit,
        **values,
        "non_FULL": non_full,
        "total_states": total,
        "FULL_fraction": values["FULL"] / total if total else 0.0,
        "non_FULL_fraction": non_full / total if total else 0.0,
        "FULL_to_non_FULL_ratio": values["FULL"] / non_full if non_full else None,
    }


def simulate_sampling_schemes(
    routes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return exact route-expanded and expected sample-balanced class counts."""

    schemes = ("S0", "S1", "S2_K2", "S2_K4", "S2_K6")
    rows: list[dict[str, Any]] = []
    for scheme in schemes:
        counts: Counter[str] = Counter()
        for route in routes:
            counts.update(_scheme_counts_for_route(route, scheme))
        name = {
            "S0": "S0_NAIVE_ALL_STATE",
            "S1": "S1_ROUTE_BALANCED",
        }.get(scheme, f"S2_ROUTE_BALANCED_{scheme.removeprefix('S2_')}")
        rows.append(_sampling_row(name, counts, weighting_unit="route"))

    grouped = _routes_by_uid(routes)
    for scheme in schemes:
        expected = Counter({action: 0.0 for action in ACTIONS})
        for sample_routes in grouped.values():
            route_weight = 1.0 / len(sample_routes)
            for route in sample_routes:
                for action, count in _scheme_counts_for_route(route, scheme).items():
                    expected[action] += route_weight * count
        suffix = {
            "S0": "S0",
            "S1": "S1",
        }.get(scheme, scheme.removeprefix("S2_"))
        rows.append(
            _sampling_row(
                f"S3_SAMPLE_BALANCED_{suffix}",
                expected,
                weighting_unit="sample_expected_uniform_route",
            )
        )
    return rows


def semantic_state_id(row: Mapping[str, Any]) -> str:
    """Hash the exact deterministic pre-layer routed-state identity.

    State tensors are captured before applying the action at ``layer``.  Thus
    the current state is fixed by sample identity, contract/schema, layer, and
    the action prefix strictly before that layer; future/current actions do not
    change the entering state.
    """

    layer = int(row["layer"])
    actions = str(row["route_key"]).split("|")
    if not 0 <= layer < len(actions):
        raise ValueError("state layer is outside the route")
    payload = {
        "contract_sha256": str(row["contract_sha256"]),
        "feature_schema_sha256": str(row["feature_schema_sha256"]),
        "uid": str(row["uid"]),
        "layer": layer,
        "action_prefix": actions[:layer],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()
