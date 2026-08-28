from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import inf
from typing import Any


FOUR_ACTIONS = ("IGNORE", "READ_ONLY", "WRITE_ONLY", "FULL")
ACTION_COST = {
    "FULL": 0,
    "READ_ONLY": 1,
    "WRITE_ONLY": 1,
    "IGNORE": 2,
}
_ACTION_TIE_ORDER = {
    "FULL": 0,
    "READ_ONLY": 1,
    "WRITE_ONLY": 2,
    "IGNORE": 3,
}


def _normalize_route(route: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(action).strip().upper() for action in route)
    invalid = sorted(set(normalized) - set(FOUR_ACTIONS))
    if invalid:
        raise ValueError(f"invalid four-action values: {invalid}")
    if not normalized:
        raise ValueError("four-action route must not be empty")
    return normalized


def binary_to_four_action(route: Sequence[int | bool]) -> tuple[str, ...]:
    values = tuple(route)
    if not values or any(value not in {0, 1, False, True} for value in values):
        raise ValueError("binary route must contain only 0/1 actions")
    return tuple("FULL" if bool(value) else "IGNORE" for value in values)


def action_route_cost(route: Sequence[str]) -> int:
    return sum(ACTION_COST[action] for action in _normalize_route(route))


def _margin(evaluation: Mapping[str, Any]) -> float | None:
    value = evaluation.get("answer_alignment_margin", evaluation.get("margin"))
    return None if value is None else float(value)


def _candidate_key(route: tuple[str, ...], evaluation: Mapping[str, Any]) -> tuple[Any, ...]:
    margin = _margin(evaluation)
    return (
        action_route_cost(route),
        inf if margin is None else -margin,
        tuple(_ACTION_TIE_ORDER[action] for action in route),
    )


class CachedRouteEvaluator:
    """Memoize exact complete-route evaluations within one sample."""

    def __init__(self, evaluate: Callable[[tuple[str, ...]], Mapping[str, Any]]):
        self._evaluate = evaluate
        self.cache: dict[tuple[str, ...], dict[str, Any]] = {}

    def __call__(self, route: Sequence[str]) -> dict[str, Any]:
        key = _normalize_route(route)
        if key not in self.cache:
            self.cache[key] = dict(self._evaluate(key))
        return self.cache[key]


@dataclass(frozen=True)
class PurificationCandidate:
    route: tuple[str, ...]
    evaluation: dict[str, Any]
    order: str
    passes: int

    @property
    def cost(self) -> int:
        return action_route_cost(self.route)


@dataclass(frozen=True)
class PurificationResult:
    route: tuple[str, ...]
    evaluation: dict[str, Any]
    order: str
    candidates: dict[str, PurificationCandidate]

    @property
    def cost(self) -> int:
        return action_route_cost(self.route)


@dataclass(frozen=True)
class RefinementResult:
    route: tuple[str, ...]
    evaluation: dict[str, Any]
    cost: int
    beam_width: int
    search_rounds: int
    evaluated_route_count: int
    valid_route_count: int
    first_round_candidate_count: int
    first_round_correct_count: int
    composite_candidate_count: int
    composite_correct_count: int
    independently_supported_composite_count: int
    independent_composition_failure_count: int


@dataclass(frozen=True)
class SourceRouteConversion:
    route: tuple[str, ...]
    evaluation: dict[str, Any]
    label_semantics: str
    source_route: tuple[str, ...]
    purification: PurificationResult | None
    refinement: RefinementResult | None


def _purify_in_order(
    anchor: tuple[str, ...],
    evaluate: CachedRouteEvaluator,
    *,
    indices: Sequence[int],
    name: str,
) -> PurificationCandidate:
    current = anchor
    passes = 0
    while True:
        passes += 1
        changed = False
        for layer_index in indices:
            if current[layer_index] != "IGNORE":
                continue
            candidate = list(current)
            candidate[layer_index] = "FULL"
            candidate_route = tuple(candidate)
            if bool(evaluate(candidate_route).get("correct")):
                current = candidate_route
                changed = True
        if not changed:
            break
    return PurificationCandidate(
        route=current,
        evaluation=evaluate(current),
        order=name,
        passes=passes,
    )


def purify_w2c_anchor(
    anchor: Sequence[str],
    evaluate: CachedRouteEvaluator,
) -> PurificationResult:
    """Restore W2C IGNORE positions to FULL in both deterministic orders."""
    normalized = _normalize_route(anchor)
    initial = evaluate(normalized)
    if not bool(initial.get("correct")):
        raise ValueError("W2C anchor must replay as evaluator-correct before purification")
    suppressed = [index for index, action in enumerate(normalized) if action == "IGNORE"]
    if any(action not in {"FULL", "IGNORE"} for action in normalized):
        raise ValueError("purification requires a mechanically mapped binary route")
    candidates = {
        "early_to_late": _purify_in_order(
            normalized,
            evaluate,
            indices=suppressed,
            name="early_to_late",
        ),
        "late_to_early": _purify_in_order(
            normalized,
            evaluate,
            indices=list(reversed(suppressed)),
            name="late_to_early",
        ),
    }
    selected = min(
        candidates.values(),
        key=lambda candidate: _candidate_key(candidate.route, candidate.evaluation),
    )
    return PurificationResult(
        route=selected.route,
        evaluation=selected.evaluation,
        order=selected.order,
        candidates=candidates,
    )


def _relaxations(route: tuple[str, ...], variable_layers: Sequence[int]):
    for layer_index in variable_layers:
        action = route[layer_index]
        if action == "IGNORE":
            replacements = ("READ_ONLY", "WRITE_ONLY", "FULL")
        elif action in {"READ_ONLY", "WRITE_ONLY"}:
            replacements = ("FULL",)
        else:
            replacements = ()
        for replacement in replacements:
            candidate = list(route)
            candidate[layer_index] = replacement
            yield tuple(candidate)


def refine_w2c_anchor(
    anchor: Sequence[str],
    evaluate: CachedRouteEvaluator,
    *,
    beam_width: int = 8,
) -> RefinementResult:
    """Run bounded monotone joint refinement from a correct purified anchor."""
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    normalized = _normalize_route(anchor)
    if any(action not in {"FULL", "IGNORE"} for action in normalized):
        raise ValueError("refinement anchor must contain only FULL and IGNORE")
    anchor_evaluation = evaluate(normalized)
    if not bool(anchor_evaluation.get("correct")):
        raise ValueError("refinement anchor must be evaluator-correct")

    variable_layers = [index for index, action in enumerate(normalized) if action == "IGNORE"]
    frontier = [normalized]
    visited = {normalized}
    valid = {normalized: anchor_evaluation}
    rounds = 0
    first_round_candidates = 0
    first_round_correct = 0
    single_valid_actions: dict[int, set[str]] = {}
    composite_candidates = 0
    composite_correct = 0
    independently_supported_composites = 0
    independent_composition_failures = 0
    while frontier:
        candidates = sorted(
            {
                candidate
                for route in frontier
                for candidate in _relaxations(route, variable_layers)
                if candidate not in visited
            },
            key=lambda route: tuple(_ACTION_TIE_ORDER[action] for action in route),
        )
        if not candidates:
            break
        rounds += 1
        if rounds == 1:
            first_round_candidates = len(candidates)
        correct_candidates = []
        for candidate in candidates:
            visited.add(candidate)
            evaluation = evaluate(candidate)
            changed = [
                index for index, (left, right) in enumerate(zip(normalized, candidate))
                if left != right
            ]
            if rounds > 1 and len(changed) > 1:
                composite_candidates += 1
                independently_supported = all(
                    candidate[index] in single_valid_actions.get(index, set())
                    for index in changed
                )
                if independently_supported:
                    independently_supported_composites += 1
                    if not bool(evaluation.get("correct")):
                        independent_composition_failures += 1
            if bool(evaluation.get("correct")):
                valid[candidate] = evaluation
                correct_candidates.append(candidate)
                if rounds == 1:
                    if len(changed) != 1:
                        raise RuntimeError("first refinement round must change exactly one layer")
                    single_valid_actions.setdefault(changed[0], set()).add(candidate[changed[0]])
                    first_round_correct += 1
                elif len(changed) > 1:
                    composite_correct += 1
        if not correct_candidates:
            break
        correct_candidates.sort(key=lambda route: _candidate_key(route, valid[route]))
        frontier = correct_candidates[:beam_width]

    selected = min(valid, key=lambda route: _candidate_key(route, valid[route]))
    return RefinementResult(
        route=selected,
        evaluation=valid[selected],
        cost=action_route_cost(selected),
        beam_width=beam_width,
        search_rounds=rounds,
        evaluated_route_count=len(visited),
        valid_route_count=len(valid),
        first_round_candidate_count=first_round_candidates,
        first_round_correct_count=first_round_correct,
        composite_candidate_count=composite_candidates,
        composite_correct_count=composite_correct,
        independently_supported_composite_count=independently_supported_composites,
        independent_composition_failure_count=independent_composition_failures,
    )


def convert_valid_source_route(
    binary_route: Sequence[int | bool],
    *,
    full_correct: bool,
    evaluate: CachedRouteEvaluator,
    beam_width: int = 8,
) -> SourceRouteConversion:
    """Convert one current-valid source route under W2C or C2C semantics."""
    source_route = binary_to_four_action(binary_route)
    source_evaluation = evaluate(source_route)
    if not bool(source_evaluation.get("correct")):
        raise ValueError("source binary route failed current-runtime replay")
    if full_correct:
        return SourceRouteConversion(
            route=source_route,
            evaluation=source_evaluation,
            label_semantics="preserving_c2c",
            source_route=source_route,
            purification=None,
            refinement=None,
        )

    purification = purify_w2c_anchor(source_route, evaluate)
    refinement = refine_w2c_anchor(
        purification.route,
        evaluate,
        beam_width=beam_width,
    )
    return SourceRouteConversion(
        route=refinement.route,
        evaluation=refinement.evaluation,
        label_semantics="corrective_w2c",
        source_route=source_route,
        purification=purification,
        refinement=refinement,
    )


def _route_metadata(route: tuple[str, ...]) -> dict[str, Any]:
    counts = {action: route.count(action) for action in FOUR_ACTIONS}
    return {
        "route": list(route),
        "route_key": "|".join(route),
        "full_count": counts["FULL"],
        "read_only_count": counts["READ_ONLY"],
        "write_only_count": counts["WRITE_ONLY"],
        "ignore_count": counts["IGNORE"],
        "read_suppression_count": counts["WRITE_ONLY"] + counts["IGNORE"],
        "write_suppression_count": counts["READ_ONLY"] + counts["IGNORE"],
        "suppression_component_cost": action_route_cost(route),
    }


def deduplicate_final_routes(
    conversion_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse identical final routes without losing source-route provenance."""
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in conversion_rows:
        if row.get("status") != "converted":
            continue
        route = _normalize_route(row["final_route"])
        grouped.setdefault(route, []).append(row)
    output = []
    for route, rows in grouped.items():
        semantics = {str(row["label_semantics"]) for row in rows}
        if len(semantics) != 1:
            raise ValueError("one final route cannot mix W2C and C2C semantics")
        evaluations = [dict(row["final_evaluation"]) for row in rows]
        if not all(bool(evaluation.get("correct")) for evaluation in evaluations):
            raise ValueError("deduplicated final route must be evaluator-correct")
        reference = evaluations[0]
        for evaluation in evaluations[1:]:
            for field in ("generated_ids", "correct", "answer_alignment_margin"):
                if field in reference or field in evaluation:
                    if reference.get(field) != evaluation.get(field):
                        raise ValueError(f"cached final-route evaluation drift in {field}")
        provenance = []
        for row in sorted(rows, key=lambda value: str(value["source_binary_route_id"])):
            provenance.append(
                {
                    key: row.get(key)
                    for key in (
                        "source_binary_route_id",
                        "source_route_id",
                        "source_binary_route",
                        "source_off_count",
                        "all_off_seed",
                        "purification",
                        "refinement",
                    )
                    if key in row
                }
            )
        output.append(
            {
                "schema_version": "unique_valid_four_action_route_v1",
                "label_semantics": semantics.pop(),
                **_route_metadata(route),
                "evaluation": reference,
                "source_binary_route_ids": sorted(
                    str(row["source_binary_route_id"]) for row in rows
                ),
                "conversion_provenance": provenance,
            }
        )
    output.sort(key=lambda row: row["route_key"])
    return output


def canonical_route(
    unique_routes: Sequence[Mapping[str, Any]],
    *,
    label_semantics: str,
) -> dict[str, Any]:
    """Choose the documented corrective or preserving canonical route."""
    rows = [row for row in unique_routes if row["label_semantics"] == label_semantics]
    if not rows:
        raise ValueError(f"no {label_semantics} route is available")

    def margin(row: Mapping[str, Any]) -> float:
        value = row["evaluation"].get("answer_alignment_margin")
        return -inf if value is None else float(value)

    if label_semantics == "corrective_w2c":
        key = lambda row: (
            int(row["suppression_component_cost"]),
            -margin(row),
            str(row["route_key"]),
        )
        selection_rule = "minimum suppression cost, maximum margin, stable route key"
    elif label_semantics == "preserving_c2c":
        key = lambda row: (
            -int(row["suppression_component_cost"]),
            -margin(row),
            str(row["route_key"]),
        )
        selection_rule = "maximum preserved source efficiency, maximum S_correct, stable route key"
    else:
        raise ValueError(f"unsupported label semantics: {label_semantics}")
    selected = dict(min(rows, key=key))
    selected["canonical_selection_rule"] = selection_rule
    return selected


def select_diverse_four_action_routes(
    routes: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    seed: int,
    uid: str,
    canonical_route_key: str,
) -> list[dict[str, Any]]:
    """Build the deterministic diverse max-K four-action training view."""
    if limit < 1:
        raise ValueError("route limit must be positive")
    by_key: dict[str, dict[str, Any]] = {}
    widths = set()
    for source in routes:
        row = dict(source)
        route = _normalize_route(row["route"])
        widths.add(len(route))
        route_key = "|".join(route)
        if row.get("route_key") != route_key:
            raise ValueError(f"route key mismatch for {uid}: {row.get('route_key')}")
        if route_key in by_key:
            raise ValueError(f"duplicate four-action route for {uid}: {route_key}")
        by_key[route_key] = row
    if len(widths) > 1:
        raise ValueError(f"inconsistent four-action route widths for {uid}")
    if canonical_route_key not in by_key:
        raise ValueError(f"canonical route is absent for {uid}")
    ordered = sorted(by_key.values(), key=lambda row: str(row["route_key"]))
    if len(ordered) <= limit:
        return ordered

    selected: list[dict[str, Any]] = []
    selected_keys = set()

    def add(row: Mapping[str, Any] | None) -> None:
        if row is None or len(selected) >= limit:
            return
        key = str(row["route_key"])
        if key not in selected_keys:
            selected.append(dict(row))
            selected_keys.add(key)

    add(by_key[canonical_route_key])
    add(min(ordered, key=lambda row: (int(row["suppression_component_cost"]), row["route_key"])))
    add(max(ordered, key=lambda row: (int(row["suppression_component_cost"]), row["route_key"])))
    width = next(iter(widths))
    add(by_key.get("|".join(["FULL"] * width)))
    add(by_key.get("|".join(["IGNORE"] * width)))
    for field in ("read_only_count", "write_only_count"):
        add(max(ordered, key=lambda row: (int(row[field]), row["route_key"])))

    while len(selected) < limit:
        cost_counts = Counter(int(row["suppression_component_cost"]) for row in selected)
        signature_counts = Counter(
            (
                int(row["read_only_count"]),
                int(row["write_only_count"]),
                int(row["ignore_count"]),
            )
            for row in selected
        )
        selected_routes = [tuple(row["route"]) for row in selected]
        selected_costs = tuple(cost_counts)

        def score(row: Mapping[str, Any]):
            route = tuple(row["route"])
            cost = int(row["suppression_component_cost"])
            signature = (
                int(row["read_only_count"]),
                int(row["write_only_count"]),
                int(row["ignore_count"]),
            )
            min_hamming = min(
                sum(left != right for left, right in zip(route, chosen))
                for chosen in selected_routes
            )
            min_cost_gap = min(abs(cost - chosen) for chosen in selected_costs)
            tie = int.from_bytes(
                sha256(f"{seed}:{uid}:{row['route_key']}".encode()).digest()[:8],
                "big",
            )
            return (
                -cost_counts[cost],
                min_hamming,
                -signature_counts[signature],
                min_cost_gap,
                tie,
            )

        candidates = [row for row in ordered if row["route_key"] not in selected_keys]
        if not candidates:
            raise RuntimeError(f"four-action selection exhausted early for {uid}")
        add(max(candidates, key=score))
    return selected
