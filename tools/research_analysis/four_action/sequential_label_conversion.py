from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from binary_policy.executor.four_action import FOUR_ACTIONS, normalize_four_action


def normalize_route(route: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(normalize_four_action(action) for action in route)
    if not normalized:
        raise ValueError("four-action route must not be empty")
    return normalized


def binary_to_four_action(route: Sequence[int | bool]) -> tuple[str, ...]:
    values = tuple(route)
    if not values or any(value not in {0, 1, False, True} for value in values):
        raise ValueError("binary route must contain only 0/1 actions")
    return tuple("FULL" if bool(value) else "IGNORE" for value in values)


class ExactRouteEvaluator:
    """Memoize exact complete-route evaluations within one sample."""

    def __init__(self, evaluate: Callable[[tuple[str, ...]], Mapping[str, Any]]):
        self._evaluate = evaluate
        self.cache: dict[tuple[str, ...], dict[str, Any]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def __call__(self, route: Sequence[str]) -> dict[str, Any]:
        key = normalize_route(route)
        if key in self.cache:
            self.cache_hits += 1
        else:
            self.cache[key] = dict(self._evaluate(key))
            self.cache_misses += 1
        return self.cache[key]


@dataclass(frozen=True)
class BranchDecision:
    layer: int
    action: str
    full_correct: bool
    read_only_correct: bool | None
    write_only_correct: bool | None


@dataclass(frozen=True)
class FinalBranch:
    route: tuple[str, ...]
    evaluation: dict[str, Any]
    decisions: tuple[BranchDecision, ...]


@dataclass(frozen=True)
class SequentialStep:
    layer: int
    incoming_branch_count: int
    full_restored_count: int
    read_only_only_count: int
    write_only_only_count: int
    both_partial_correct_count: int
    ignore_fallback_count: int
    outgoing_branch_count: int
    new_route_evaluations: int


@dataclass(frozen=True)
class SequentialRefinement:
    source_route: tuple[str, ...]
    final_branches: tuple[FinalBranch, ...]
    steps: tuple[SequentialStep, ...]
    maximum_branch_count: int
    new_route_evaluations: int


@dataclass(frozen=True)
class SourceRouteConversion:
    source_route: tuple[str, ...]
    label_semantics: str
    final_branches: tuple[FinalBranch, ...]
    steps: tuple[SequentialStep, ...]
    maximum_branch_count: int
    new_route_evaluations: int


def _replace(route: tuple[str, ...], layer: int, action: str) -> tuple[str, ...]:
    candidate = list(route)
    candidate[layer] = action
    return tuple(candidate)


def sequentially_refine_w2c(
    source_route: Sequence[str],
    evaluate: ExactRouteEvaluator,
) -> SequentialRefinement:
    """Retain every correct branch under fixed early-to-late OFF refinement."""
    anchor = normalize_route(source_route)
    if any(action not in {"FULL", "IGNORE"} for action in anchor):
        raise ValueError("W2C source route must be a mechanical FULL/IGNORE mapping")
    initial = evaluate(anchor)
    if not bool(initial.get("correct")):
        raise ValueError("W2C source route failed current-runtime replay")

    starting_misses = evaluate.cache_misses
    active: dict[tuple[str, ...], FinalBranch] = {
        anchor: FinalBranch(anchor, initial, ())
    }
    maximum_branch_count = 1
    steps = []
    for layer in (index for index, action in enumerate(anchor) if action == "IGNORE"):
        incoming = tuple(active[key] for key in sorted(active))
        outgoing: dict[tuple[str, ...], FinalBranch] = {}
        misses_before = evaluate.cache_misses
        full_restored = 0
        read_only_only = 0
        write_only_only = 0
        both_partial = 0
        ignore_fallback = 0

        for branch in incoming:
            full_route = _replace(branch.route, layer, "FULL")
            full_evaluation = evaluate(full_route)
            if bool(full_evaluation.get("correct")):
                decision = BranchDecision(layer, "FULL", True, None, None)
                outgoing[full_route] = FinalBranch(
                    full_route,
                    full_evaluation,
                    branch.decisions + (decision,),
                )
                full_restored += 1
                continue

            read_route = _replace(branch.route, layer, "READ_ONLY")
            write_route = _replace(branch.route, layer, "WRITE_ONLY")
            read_evaluation = evaluate(read_route)
            write_evaluation = evaluate(write_route)
            read_correct = bool(read_evaluation.get("correct"))
            write_correct = bool(write_evaluation.get("correct"))

            if read_correct:
                decision = BranchDecision(
                    layer,
                    "READ_ONLY",
                    False,
                    True,
                    write_correct,
                )
                outgoing[read_route] = FinalBranch(
                    read_route,
                    read_evaluation,
                    branch.decisions + (decision,),
                )
            if write_correct:
                decision = BranchDecision(
                    layer,
                    "WRITE_ONLY",
                    False,
                    read_correct,
                    True,
                )
                outgoing[write_route] = FinalBranch(
                    write_route,
                    write_evaluation,
                    branch.decisions + (decision,),
                )

            if read_correct and write_correct:
                both_partial += 1
            elif read_correct:
                read_only_only += 1
            elif write_correct:
                write_only_only += 1
            else:
                decision = BranchDecision(layer, "IGNORE", False, False, False)
                outgoing[branch.route] = FinalBranch(
                    branch.route,
                    branch.evaluation,
                    branch.decisions + (decision,),
                )
                ignore_fallback += 1

        if not outgoing or not all(
            bool(branch.evaluation.get("correct")) for branch in outgoing.values()
        ):
            raise RuntimeError("sequential refinement lost its correct-branch invariant")
        active = outgoing
        maximum_branch_count = max(maximum_branch_count, len(active))
        steps.append(
            SequentialStep(
                layer=layer,
                incoming_branch_count=len(incoming),
                full_restored_count=full_restored,
                read_only_only_count=read_only_only,
                write_only_only_count=write_only_only,
                both_partial_correct_count=both_partial,
                ignore_fallback_count=ignore_fallback,
                outgoing_branch_count=len(active),
                new_route_evaluations=evaluate.cache_misses - misses_before,
            )
        )

    return SequentialRefinement(
        source_route=anchor,
        final_branches=tuple(active[key] for key in sorted(active)),
        steps=tuple(steps),
        maximum_branch_count=maximum_branch_count,
        new_route_evaluations=evaluate.cache_misses - starting_misses,
    )


def convert_replay_valid_source_route(
    binary_route: Sequence[int | bool],
    *,
    full_correct: bool,
    evaluate: ExactRouteEvaluator,
) -> SourceRouteConversion:
    source_route = binary_to_four_action(binary_route)
    source_evaluation = evaluate(source_route)
    if not bool(source_evaluation.get("correct")):
        raise ValueError("source binary route failed current-runtime replay")
    if full_correct:
        branch = FinalBranch(source_route, source_evaluation, ())
        return SourceRouteConversion(
            source_route=source_route,
            label_semantics="preserving_c2c",
            final_branches=(branch,),
            steps=(),
            maximum_branch_count=1,
            new_route_evaluations=0,
        )

    refinement = sequentially_refine_w2c(source_route, evaluate)
    return SourceRouteConversion(
        source_route=source_route,
        label_semantics="corrective_w2c",
        final_branches=refinement.final_branches,
        steps=refinement.steps,
        maximum_branch_count=refinement.maximum_branch_count,
        new_route_evaluations=refinement.new_route_evaluations,
    )


def _route_metadata(route: tuple[str, ...]) -> dict[str, Any]:
    counts = {action: route.count(action) for action in FOUR_ACTIONS}
    return {
        "four_action_route": list(route),
        "route_key": "|".join(route),
        "num_FULL": counts["FULL"],
        "num_READ_ONLY": counts["READ_ONLY"],
        "num_WRITE_ONLY": counts["WRITE_ONLY"],
        "num_IGNORE": counts["IGNORE"],
        "read_suppression_count": counts["WRITE_ONLY"] + counts["IGNORE"],
        "write_suppression_count": counts["READ_ONLY"] + counts["IGNORE"],
    }


def deduplicate_sequential_routes(
    conversion_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for row in conversion_rows:
        if row.get("status") != "converted":
            continue
        for branch in row.get("final_branches", []):
            route = normalize_route(branch["route"])
            evaluation = branch["evaluation"]
            if not bool(evaluation.get("correct")):
                raise ValueError("every final sequential branch must be evaluator-correct")
            grouped.setdefault(route, []).append((row, branch))

    output = []
    for route, occurrences in grouped.items():
        semantics = {str(row["label_semantics"]) for row, _ in occurrences}
        if len(semantics) != 1:
            raise ValueError("one final route cannot mix W2C and C2C semantics")
        reference = dict(occurrences[0][1]["evaluation"])
        for _, branch in occurrences[1:]:
            evaluation = branch["evaluation"]
            for field in ("generated_ids", "generated_answer", "correct", "answer_alignment_margin"):
                if field in reference or field in evaluation:
                    if reference.get(field) != evaluation.get(field):
                        raise ValueError(f"cached final-route evaluation drift in {field}")
        provenance = []
        for row, branch in sorted(
            occurrences,
            key=lambda occurrence: str(occurrence[0]["source_binary_route_id"]),
        ):
            provenance.append(
                {
                    "source_binary_route_id": row["source_binary_route_id"],
                    "source_route_id": row.get("source_route_id"),
                    "source_binary_route": row.get("source_binary_route"),
                    "source_off_count": row.get("source_off_count"),
                    "all_off_seed": row.get("all_off_seed"),
                    "decisions": branch.get("decisions", []),
                    "steps": row.get("steps", []),
                    "maximum_branch_count": row.get("maximum_branch_count"),
                    "new_route_evaluations": row.get("new_route_evaluations"),
                }
            )
        output.append(
            {
                "schema_version": "exact_sequential_four_action_route_v1",
                "label_semantics": semantics.pop(),
                **_route_metadata(route),
                "evaluation": reference,
                "source_binary_route_ids": sorted(
                    str(row["source_binary_route_id"]) for row, _ in occurrences
                ),
                "conversion_provenance": provenance,
            }
        )
    output.sort(key=lambda row: row["route_key"])
    return output
