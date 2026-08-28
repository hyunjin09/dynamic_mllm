from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import floor, inf
from statistics import mean, median, pstdev
from typing import Any


THREE_ACTION_VALUES = ("READ_OFF", "WRITE_OFF", "BOTH_OFF")
ROUTE_VALUES = ("FULL", *THREE_ACTION_VALUES)
EXECUTOR_ACTION = {
    "FULL": "FULL",
    "READ_OFF": "WRITE_ONLY",
    "WRITE_OFF": "READ_ONLY",
    "BOTH_OFF": "IGNORE",
}
SUPPRESSION_COST = {
    "FULL": 0,
    "READ_OFF": 1,
    "WRITE_OFF": 1,
    "BOTH_OFF": 2,
}
_ACTION_ORDER = {action: index for index, action in enumerate(ROUTE_VALUES)}
_RETAINED_CLASSIFICATIONS = {
    "HARD_NECESSARY",
    "SOFT_ALIGNMENT_HELPFUL",
    "CONTEXT_DEPENDENT_NECESSARY",
}


def _normalize_route(route: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(action).strip().upper() for action in route)
    invalid = sorted(set(normalized) - set(ROUTE_VALUES))
    if invalid:
        raise ValueError(f"invalid three-action route values: {invalid}")
    if not normalized:
        raise ValueError("three-action route must not be empty")
    return normalized


def binary_to_three_action(route: Sequence[int | bool]) -> tuple[str, ...]:
    values = tuple(route)
    if not values or any(value not in {0, 1, False, True} for value in values):
        raise ValueError("binary route must contain only 0/1 actions")
    return tuple("FULL" if bool(value) else "BOTH_OFF" for value in values)


def three_action_to_executor(route: Sequence[str]) -> tuple[str, ...]:
    return tuple(EXECUTOR_ACTION[action] for action in _normalize_route(route))


def suppression_cost(route: Sequence[str]) -> int:
    return sum(SUPPRESSION_COST[action] for action in _normalize_route(route))


def _route_key(route: Sequence[str]) -> str:
    return "|".join(_normalize_route(route))


def _replace(route: tuple[str, ...], layer: int, action: str) -> tuple[str, ...]:
    candidate = list(route)
    candidate[layer] = action
    return tuple(candidate)


class CachedThreeActionEvaluator:
    """Cache semantic routes while executing the already-validated action path."""

    def __init__(self, evaluate_executor: Callable[[tuple[str, ...]], Mapping[str, Any]]):
        self._evaluate_executor = evaluate_executor
        self.cache: dict[tuple[str, ...], dict[str, Any]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def __call__(self, route: Sequence[str]) -> dict[str, Any]:
        key = _normalize_route(route)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.cache_misses += 1
        self.cache[key] = dict(self._evaluate_executor(three_action_to_executor(key)))
        return self.cache[key]


@dataclass(frozen=True)
class ScreeningPosition:
    layer: int
    classification: str
    route_context: tuple[str, ...]
    both_off_evaluation: dict[str, Any]
    full_reference_route: tuple[str, ...]
    full_reference_evaluation: dict[str, Any]
    score_quantity: str
    both_off_minus_full: float
    pass_index: int


@dataclass(frozen=True)
class ScreeningResult:
    route_type: str
    source_route: tuple[str, ...]
    route: tuple[str, ...]
    route_evaluation: dict[str, Any]
    positions: tuple[ScreeningPosition, ...]
    history: tuple[ScreeningPosition, ...]
    passes: int
    epsilon: float

    @property
    def candidate_layers(self) -> tuple[int, ...]:
        return tuple(
            row.layer for row in self.positions
            if row.classification in _RETAINED_CLASSIFICATIONS
        )


@dataclass(frozen=True)
class DecompositionPosition:
    layer: int
    screening_classification: str
    action_classification: str
    score_quantity: str
    full_reference_evaluation: dict[str, Any]
    actions: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class RouteEvaluation:
    route: tuple[str, ...]
    evaluation: dict[str, Any]
    suppression_cost: int


@dataclass(frozen=True)
class RefinementResult:
    route_type: str
    beam_width: int
    candidate_layers: tuple[int, ...]
    positive_routes: tuple[RouteEvaluation, ...]
    pareto_routes: tuple[RouteEvaluation, ...]
    max_margin_route: RouteEvaluation | None
    corrective_partial_candidates: tuple[RouteEvaluation, ...]
    evaluated_route_count: int
    beam_rounds: int


def _score(evaluation: Mapping[str, Any], route_type: str) -> float:
    if route_type == "W2C":
        value = evaluation.get("answer_alignment_margin")
        if value is None:
            raise ValueError("W2C evaluation lacks answer_alignment_margin")
        return float(value)
    if route_type == "C2C":
        value = evaluation.get("S_correct")
        if value is None:
            raise ValueError("C2C evaluation lacks S_correct")
        return float(value)
    raise ValueError(f"unsupported route_type: {route_type}")


def _score_quantity(route_type: str) -> str:
    if route_type == "W2C":
        return "answer_alignment_margin"
    if route_type == "C2C":
        return "S_correct"
    raise ValueError(f"unsupported route_type: {route_type}")


def _screen_classification(
    *,
    route_type: str,
    both_off: Mapping[str, Any],
    full: Mapping[str, Any],
    epsilon: float,
) -> str:
    if not bool(both_off.get("correct")):
        raise ValueError("screening anchor must remain evaluator-correct")
    delta = _score(both_off, route_type) - _score(full, route_type)
    if not bool(full.get("correct")):
        return "HARD_NECESSARY" if route_type == "W2C" else "CONTEXT_DEPENDENT_NECESSARY"
    if delta > epsilon:
        return "SOFT_ALIGNMENT_HELPFUL"
    return "REDUNDANT"


def screen_binary_off_positions(
    anchor: Sequence[str],
    *,
    route_type: str,
    evaluate: CachedThreeActionEvaluator,
    epsilon: float,
) -> ScreeningResult:
    """Restore answer-alignment-redundant BOTH_OFF positions to a fixed point."""
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    source = _normalize_route(anchor)
    if any(action not in {"FULL", "BOTH_OFF"} for action in source):
        raise ValueError("screening requires a mechanically mapped binary route")
    if not bool(evaluate(source).get("correct")):
        raise ValueError("source binary route failed current-runtime replay")
    original_off = tuple(index for index, action in enumerate(source) if action == "BOTH_OFF")
    current = source
    history: list[ScreeningPosition] = []
    latest: dict[int, ScreeningPosition] = {}
    passes = 0
    while True:
        passes += 1
        changed = False
        for layer in original_off:
            if current[layer] != "BOTH_OFF":
                continue
            both_off_evaluation = dict(evaluate(current))
            full_route = _replace(current, layer, "FULL")
            full_evaluation = dict(evaluate(full_route))
            row = ScreeningPosition(
                layer=layer,
                classification=_screen_classification(
                    route_type=route_type,
                    both_off=both_off_evaluation,
                    full=full_evaluation,
                    epsilon=epsilon,
                ),
                route_context=current,
                both_off_evaluation=both_off_evaluation,
                full_reference_route=full_route,
                full_reference_evaluation=full_evaluation,
                score_quantity=_score_quantity(route_type),
                both_off_minus_full=(
                    _score(both_off_evaluation, route_type)
                    - _score(full_evaluation, route_type)
                ),
                pass_index=passes,
            )
            history.append(row)
            latest[layer] = row
            if row.classification == "REDUNDANT":
                current = full_route
                changed = True
        if not changed:
            break
    return ScreeningResult(
        route_type=route_type,
        source_route=source,
        route=current,
        route_evaluation=dict(evaluate(current)),
        positions=tuple(latest[layer] for layer in original_off),
        history=tuple(history),
        passes=passes,
        epsilon=float(epsilon),
    )


def _action_classification(
    route_type: str,
    states: Mapping[str, Mapping[str, Any]],
    epsilon: float,
) -> str:
    full = states["FULL"]

    def works(action: str) -> bool:
        state = states[action]
        if not bool(state.get("correct")):
            return False
        if route_type == "W2C":
            return True
        return _score(state, route_type) - _score(full, route_type) > epsilon

    read = works("READ_OFF")
    write = works("WRITE_OFF")
    both = works("BOTH_OFF")
    if read and write:
        return "EITHER_SUPPRESSION"
    if read:
        return "READ_SUPPRESSION"
    if write:
        return "WRITE_SUPPRESSION"
    if both:
        return "BOTH_SUPPRESSION"
    return "NO_MEANINGFUL_GAIN"


def decompose_screened_positions(
    screening: ScreeningResult,
    *,
    evaluate: CachedThreeActionEvaluator,
    epsilon: float,
) -> tuple[DecompositionPosition, ...]:
    output = []
    anchor = screening.route
    for position in screening.positions:
        if position.classification not in _RETAINED_CLASSIFICATIONS:
            continue
        layer = position.layer
        if anchor[layer] != "BOTH_OFF":
            raise RuntimeError("retained screening position is not BOTH_OFF in final anchor")
        routes = {
            "FULL": _replace(anchor, layer, "FULL"),
            "READ_OFF": _replace(anchor, layer, "READ_OFF"),
            "WRITE_OFF": _replace(anchor, layer, "WRITE_OFF"),
            "BOTH_OFF": anchor,
        }
        states = {action: dict(evaluate(route)) for action, route in routes.items()}
        full_score = _score(states["FULL"], screening.route_type)
        action_rows = {}
        for action in ("READ_OFF", "WRITE_OFF", "BOTH_OFF"):
            state = states[action]
            action_rows[action] = {
                "route": list(routes[action]),
                "evaluation": state,
                "delta_vs_full_reference": _score(state, screening.route_type) - full_score,
                "delta_vs_both_off": (
                    _score(state, screening.route_type)
                    - _score(states["BOTH_OFF"], screening.route_type)
                ),
            }
        output.append(
            DecompositionPosition(
                layer=layer,
                screening_classification=position.classification,
                action_classification=_action_classification(
                    screening.route_type, states, epsilon
                ),
                score_quantity=_score_quantity(screening.route_type),
                full_reference_evaluation=states["FULL"],
                actions=action_rows,
            )
        )
    return tuple(output)


def evaluate_independent_composition(
    screening: ScreeningResult,
    decomposition: Sequence[DecompositionPosition],
    *,
    evaluate: CachedThreeActionEvaluator,
    epsilon: float,
    unified_full_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute the route implied by independently best supported local actions."""
    route = list(screening.route)
    selections = []
    all_supported = True
    for position in decomposition:
        supported = []
        for action, row in position.actions.items():
            state = row["evaluation"]
            if screening.route_type == "W2C":
                qualifies = bool(state.get("correct")) or float(
                    row["delta_vs_full_reference"]
                ) > epsilon
            else:
                qualifies = bool(state.get("correct")) and float(
                    row["delta_vs_full_reference"]
                ) > epsilon
            if qualifies:
                supported.append((action, row))
        if not supported:
            all_supported = False
            action = "BOTH_OFF"
            selected = position.actions[action]
        else:
            action, selected = min(
                supported,
                key=lambda item: (
                    -float(item[1]["delta_vs_full_reference"]),
                    SUPPRESSION_COST[item[0]],
                    _ACTION_ORDER[item[0]],
                ),
            )
        route[position.layer] = action
        selections.append(
            {
                "layer": position.layer,
                "action": action,
                "locally_supported": bool(supported),
                "delta_vs_full_reference": float(selected["delta_vs_full_reference"]),
                "local_evaluation": selected["evaluation"],
            }
        )
    route_tuple = tuple(route)
    evaluation = dict(evaluate(route_tuple))
    if screening.route_type == "W2C":
        joint_positive = bool(evaluation.get("correct"))
    else:
        joint_positive = (
            bool(evaluation.get("correct"))
            and _score(evaluation, "C2C")
            > _score(unified_full_evaluation, "C2C") + epsilon
        )
    return {
        "route": list(route_tuple),
        "executor_route": list(three_action_to_executor(route_tuple)),
        "evaluation": evaluation,
        "locally_selected_actions": selections,
        "all_local_actions_supported": all_supported,
        "joint_positive": joint_positive,
        "independent_composition_failure": bool(selections) and all_supported and not joint_positive,
    }


def _route_state(route: tuple[str, ...], evaluation: Mapping[str, Any]) -> RouteEvaluation:
    return RouteEvaluation(route, dict(evaluation), suppression_cost(route))


def _better_score_key(row: RouteEvaluation, route_type: str) -> tuple[Any, ...]:
    return (
        -_score(row.evaluation, route_type),
        not bool(row.evaluation.get("correct")),
        row.suppression_cost,
        tuple(_ACTION_ORDER[action] for action in row.route),
    )


def _beam_select(
    rows: Sequence[RouteEvaluation],
    *,
    route_type: str,
    beam_width: int,
) -> list[RouteEvaluation]:
    unique = {row.route: row for row in rows}
    ordered = sorted(unique.values(), key=lambda row: _better_score_key(row, route_type))
    selected: list[RouteEvaluation] = []

    def add(row: RouteEvaluation | None) -> None:
        if row is not None and row.route not in {value.route for value in selected}:
            selected.append(row)

    correct = [row for row in ordered if bool(row.evaluation.get("correct"))]
    add(
        min(
            correct,
            key=lambda row: (
                row.suppression_cost,
                -_score(row.evaluation, route_type),
                _route_key(row.route),
            ),
        )
        if correct else None
    )
    add(ordered[0] if ordered else None)
    for row in ordered:
        if len(selected) >= beam_width:
            break
        add(row)
    return selected[:beam_width]


def _pareto(rows: Sequence[RouteEvaluation], route_type: str) -> tuple[RouteEvaluation, ...]:
    output = []
    for row in rows:
        score = _score(row.evaluation, route_type)
        dominated = any(
            other.route != row.route
            and other.suppression_cost <= row.suppression_cost
            and _score(other.evaluation, route_type) >= score
            and (
                other.suppression_cost < row.suppression_cost
                or _score(other.evaluation, route_type) > score
            )
            for other in rows
        )
        if not dominated:
            output.append(row)
    return tuple(sorted(output, key=lambda row: (row.suppression_cost, -_score(row.evaluation, route_type), _route_key(row.route))))


def refine_three_action_route(
    anchor: Sequence[str],
    *,
    candidate_layers: Sequence[int],
    route_type: str,
    evaluate: CachedThreeActionEvaluator,
    epsilon: float,
    beam_width: int,
    unified_full_evaluation: Mapping[str, Any],
) -> RefinementResult:
    """Bounded coordinate beam over three suppressions; FULL is never expanded."""
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    normalized = _normalize_route(anchor)
    layers = tuple(sorted(set(int(layer) for layer in candidate_layers)))
    if any(not 0 <= layer < len(normalized) for layer in layers):
        raise ValueError("candidate layer is outside the route")
    if any(normalized[layer] != "BOTH_OFF" for layer in layers):
        raise ValueError("candidate layers must begin at BOTH_OFF")
    initial = _route_state(normalized, evaluate(normalized))
    if not bool(initial.evaluation.get("correct")):
        raise ValueError("refinement anchor must be evaluator-correct")

    visited: dict[tuple[str, ...], RouteEvaluation] = {normalized: initial}
    positives: dict[tuple[str, ...], RouteEvaluation] = {}
    partial: dict[tuple[str, ...], RouteEvaluation] = {}

    def classify(row: RouteEvaluation) -> None:
        if route_type == "W2C":
            if bool(row.evaluation.get("correct")):
                positives[row.route] = row
            elif (
                _score(row.evaluation, route_type)
                > _score(unified_full_evaluation, route_type) + epsilon
            ):
                partial[row.route] = row
        else:
            if (
                bool(row.evaluation.get("correct"))
                and _score(row.evaluation, route_type)
                > _score(unified_full_evaluation, route_type) + epsilon
            ):
                positives[row.route] = row

    classify(initial)
    frontier = [initial]
    rounds = 0
    for layer in layers:
        rounds += 1
        candidates = []
        for row in frontier:
            for action in THREE_ACTION_VALUES:
                route = _replace(row.route, layer, action)
                if route not in visited:
                    visited[route] = _route_state(route, evaluate(route))
                candidate = visited[route]
                classify(candidate)
                candidates.append(candidate)
        frontier = _beam_select(candidates, route_type=route_type, beam_width=beam_width)
        if not frontier:
            break

    positive_rows = tuple(
        sorted(
            positives.values(),
            key=lambda row: (row.suppression_cost, -_score(row.evaluation, route_type), _route_key(row.route)),
        )
    )
    max_route = (
        min(positive_rows, key=lambda row: _better_score_key(row, route_type))
        if positive_rows else None
    )
    partial_rows = tuple(
        sorted(partial.values(), key=lambda row: _better_score_key(row, route_type))
    )
    return RefinementResult(
        route_type=route_type,
        beam_width=beam_width,
        candidate_layers=layers,
        positive_routes=positive_rows,
        pareto_routes=_pareto(positive_rows, route_type),
        max_margin_route=max_route,
        corrective_partial_candidates=partial_rows,
        evaluated_route_count=len(visited),
        beam_rounds=rounds,
    )


def select_canonical_w2c_route(
    routes: Sequence[Mapping[str, Any]],
    *,
    best_seed_margin: float,
    epsilon: float,
) -> dict[str, Any]:
    correct = [dict(row) for row in routes if bool(row["evaluation"].get("correct"))]
    eligible = [
        dict(row) for row in routes
        if bool(row["evaluation"].get("correct"))
        and float(row["evaluation"]["answer_alignment_margin"])
        >= float(best_seed_margin) - float(epsilon)
    ]
    if not eligible:
        if not correct:
            raise ValueError("no correct W2C route is available")
        selected = min(
            correct,
            key=lambda row: (
                -float(row["evaluation"]["answer_alignment_margin"]),
                int(row["suppression_cost"]),
                _route_key(row["route"]),
            ),
        )
        selected["canonical_within_seed_epsilon"] = False
        selected["canonical_selection_rule"] = (
            "maximum refined margin fallback because no refined route is within seed epsilon"
        )
        return selected
    selected = min(
        eligible,
        key=lambda row: (
            int(row["suppression_cost"]),
            -float(row["evaluation"]["answer_alignment_margin"]),
            _route_key(row["route"]),
        ),
    )
    selected["canonical_within_seed_epsilon"] = True
    selected["canonical_selection_rule"] = (
        "minimum suppression cost within epsilon of best seed margin, then maximum margin"
    )
    return selected


def _route_metadata(route: Sequence[str]) -> dict[str, Any]:
    normalized = _normalize_route(route)
    return {
        "route": list(normalized),
        "route_key": _route_key(normalized),
        "full_count": normalized.count("FULL"),
        "read_off_count": normalized.count("READ_OFF"),
        "write_off_count": normalized.count("WRITE_OFF"),
        "both_off_count": normalized.count("BOTH_OFF"),
        "suppression_cost": suppression_cost(normalized),
        "executor_route": list(three_action_to_executor(normalized)),
    }


def deduplicate_positive_routes(
    conversion_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate correct positive routes without duplicating training weight."""
    grouped: dict[tuple[str, ...], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for conversion in conversion_rows:
        if conversion.get("status") != "converted":
            continue
        for source in conversion.get("positive_routes", []):
            route = _normalize_route(source["route"])
            grouped.setdefault(route, []).append((conversion, source))
    output = []
    for route, members in grouped.items():
        semantics = {str(conversion["label_semantics"]) for conversion, _ in members}
        if len(semantics) != 1:
            raise ValueError("one positive route cannot mix W2C and C2C semantics")
        reference = dict(members[0][1]["evaluation"])
        if not bool(reference.get("correct")):
            raise ValueError("positive route must be evaluator-correct")
        for _, source in members[1:]:
            evaluation = source["evaluation"]
            if not bool(evaluation.get("correct")):
                raise ValueError("positive route must be evaluator-correct")
            for field in ("generated_ids", "correct", "S_correct", "answer_alignment_margin"):
                if reference.get(field) != evaluation.get(field):
                    raise ValueError(f"cached positive-route evaluation drift in {field}")
        source_ids = sorted(
            {str(conversion["source_binary_route_id"]) for conversion, _ in members}
        )
        output.append(
            {
                "schema_version": "unique_valid_three_action_route_v1",
                "label_semantics": semantics.pop(),
                **_route_metadata(route),
                "evaluation": reference,
                "source_binary_route_ids": source_ids,
                "source_conversion_count": len(members),
            }
        )
    output.sort(key=lambda row: row["route_key"])
    return output


def select_canonical_c2c_route(
    routes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    eligible = [dict(row) for row in routes if bool(row["evaluation"].get("correct"))]
    if not eligible:
        raise ValueError("no correct C2C alignment-improving route is available")
    return min(
        eligible,
        key=lambda row: (
            int(row["suppression_cost"]),
            -float(row["evaluation"]["S_correct"]),
            _route_key(row["route"]),
        ),
    )


def select_diverse_three_action_routes(
    routes: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    seed: int,
    uid: str,
    canonical_route_key: str,
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("route limit must be positive")
    by_key: dict[str, dict[str, Any]] = {}
    for source in routes:
        row = dict(source)
        route = _normalize_route(row["route"])
        key = _route_key(route)
        if row.get("route_key") != key:
            raise ValueError(f"route key mismatch for {uid}: {row.get('route_key')}")
        if key in by_key:
            raise ValueError(f"duplicate three-action route for {uid}: {key}")
        by_key[key] = row
    if canonical_route_key not in by_key:
        raise ValueError(f"canonical route is absent for {uid}")
    ordered = sorted(by_key.values(), key=lambda row: str(row["route_key"]))
    if len(ordered) <= limit:
        return ordered
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()

    def add(row: Mapping[str, Any]) -> None:
        key = str(row["route_key"])
        if len(selected) < limit and key not in selected_keys:
            selected.append(dict(row))
            selected_keys.add(key)

    add(by_key[canonical_route_key])
    add(min(ordered, key=lambda row: (int(row["suppression_cost"]), row["route_key"])))
    add(max(ordered, key=lambda row: (int(row["suppression_cost"]), row["route_key"])))
    while len(selected) < limit:
        chosen_routes = [tuple(row["route"]) for row in selected]
        chosen_costs = [int(row["suppression_cost"]) for row in selected]

        def score(row: Mapping[str, Any]) -> tuple[int, int, int]:
            route = tuple(row["route"])
            cost = int(row["suppression_cost"])
            min_hamming = min(
                sum(left != right for left, right in zip(route, chosen))
                for chosen in chosen_routes
            )
            min_cost_gap = min(abs(cost - chosen) for chosen in chosen_costs)
            tie = int.from_bytes(
                sha256(f"{seed}:{uid}:{row['route_key']}".encode()).digest()[:8],
                "big",
            )
            return min_hamming, min_cost_gap, tie

        candidates = [row for row in ordered if row["route_key"] not in selected_keys]
        if not candidates:
            raise RuntimeError(f"three-action selection exhausted early for {uid}")
        add(max(candidates, key=score))
    return selected


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0 <= probability <= 1:
        raise ValueError("quantile probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def calibrate_repeatability_epsilon(
    *,
    signed_differences: Sequence[float],
    floor: float,
    quantile: float = 0.99,
) -> dict[str, Any]:
    if floor < 0:
        raise ValueError("repeatability floor must be non-negative")
    signed = [float(value) for value in signed_differences]
    if not signed:
        raise ValueError("repeatability calibration requires differences")
    absolute = [abs(value) for value in signed]
    percentile = int(round(100 * quantile))
    empirical = _quantile(absolute, quantile)
    return {
        "count": len(signed),
        "signed_mean": mean(signed),
        "signed_median": median(signed),
        "signed_std": pstdev(signed),
        "absolute_mean": mean(absolute),
        "absolute_median": median(absolute),
        "absolute_std": pstdev(absolute),
        f"absolute_p{percentile}": empirical,
        "predeclared_floor": float(floor),
        "epsilon": max(float(floor), empirical),
        "selection_rule": (
            f"max(predeclared_floor, empirical_absolute_repeat_difference_p{percentile})"
        ),
    }
