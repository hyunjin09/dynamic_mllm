from __future__ import annotations

from collections import Counter
import inspect

import pytest

from tools.research_analysis.four_action.sequential_label_conversion import (
    ExactRouteEvaluator,
    convert_replay_valid_source_route,
    deduplicate_sequential_routes,
    sequentially_refine_w2c,
)


def _evaluator(correct_routes):
    correct = {tuple(route) for route in correct_routes}
    calls = Counter()

    def evaluate(route):
        key = tuple(route)
        calls[key] += 1
        return {
            "correct": key in correct,
            "generated_answer": "correct" if key in correct else "wrong",
            "answer_alignment_margin": 0.5 if key in correct else -0.5,
        }

    return ExactRouteEvaluator(evaluate), calls


def test_full_restoration_short_circuits_partial_actions():
    anchor = ("IGNORE", "IGNORE")
    evaluator, calls = _evaluator(
        {
            anchor,
            ("FULL", "IGNORE"),
            ("FULL", "FULL"),
        }
    )

    result = sequentially_refine_w2c(anchor, evaluator)

    assert [branch.route for branch in result.final_branches] == [("FULL", "FULL")]
    assert result.steps[0].full_restored_count == 1
    assert result.steps[1].full_restored_count == 1
    assert ("READ_ONLY", "IGNORE") not in calls
    assert ("WRITE_ONLY", "IGNORE") not in calls
    assert ("FULL", "READ_ONLY") not in calls
    assert ("FULL", "WRITE_ONLY") not in calls


@pytest.mark.parametrize(
    ("correct_partial", "expected"),
    [
        ("READ_ONLY", ("READ_ONLY",)),
        ("WRITE_ONLY", ("WRITE_ONLY",)),
    ],
)
def test_one_sided_partial_restoration_keeps_only_the_correct_action(
    correct_partial,
    expected,
):
    anchor = ("IGNORE",)
    evaluator, _ = _evaluator({anchor, (correct_partial,)})

    result = sequentially_refine_w2c(anchor, evaluator)

    assert [branch.route for branch in result.final_branches] == [expected]


def test_ignore_fallback_reuses_the_known_correct_branch():
    anchor = ("IGNORE",)
    evaluator, calls = _evaluator({anchor})

    result = sequentially_refine_w2c(anchor, evaluator)

    assert [branch.route for branch in result.final_branches] == [anchor]
    assert result.steps[0].ignore_fallback_count == 1
    assert calls[anchor] == 1


def test_both_partial_actions_branch_and_later_layers_use_each_branch_context():
    anchor = ("IGNORE", "IGNORE")
    evaluator, calls = _evaluator(
        {
            anchor,
            ("READ_ONLY", "IGNORE"),
            ("WRITE_ONLY", "IGNORE"),
            ("READ_ONLY", "FULL"),
            ("WRITE_ONLY", "READ_ONLY"),
        }
    )

    result = sequentially_refine_w2c(anchor, evaluator)

    assert [branch.route for branch in result.final_branches] == [
        ("READ_ONLY", "FULL"),
        ("WRITE_ONLY", "READ_ONLY"),
    ]
    assert result.steps[0].both_partial_correct_count == 1
    assert result.steps[0].outgoing_branch_count == 2
    assert result.steps[1].incoming_branch_count == 2
    assert ("READ_ONLY", "FULL") in calls
    assert ("WRITE_ONLY", "FULL") in calls
    assert ("WRITE_ONLY", "READ_ONLY") in calls
    assert ("READ_ONLY", "READ_ONLY") not in calls


def test_c2c_preserves_the_mechanical_full_ignore_route_without_refinement():
    mapped = ("FULL", "IGNORE", "FULL")
    evaluator, calls = _evaluator({mapped})

    result = convert_replay_valid_source_route(
        [1, 0, 1],
        full_correct=True,
        evaluate=evaluator,
    )

    assert result.label_semantics == "preserving_c2c"
    assert [branch.route for branch in result.final_branches] == [mapped]
    assert result.steps == ()
    assert calls == Counter({mapped: 1})


def test_w2c_requires_a_replay_valid_source_route():
    evaluator, _ = _evaluator(set())

    with pytest.raises(ValueError, match="replay"):
        convert_replay_valid_source_route(
            [1, 0, 1],
            full_correct=False,
            evaluate=evaluator,
        )


def test_exact_route_evaluator_caches_complete_routes_and_counts_hits():
    evaluator, calls = _evaluator({("IGNORE",)})

    evaluator(("IGNORE",))
    evaluator(("IGNORE",))

    assert calls == Counter({("IGNORE",): 1})
    assert evaluator.cache_misses == 1
    assert evaluator.cache_hits == 1


def test_deduplication_retains_every_source_route_as_provenance():
    rows = [
        {
            "status": "converted",
            "source_binary_route_id": "u::r2",
            "source_route_id": "r2",
            "source_binary_route": [1, 0],
            "source_off_count": 1,
            "all_off_seed": False,
            "label_semantics": "corrective_w2c",
            "final_branches": [
                {
                    "route": ["FULL", "READ_ONLY"],
                    "evaluation": {"correct": True, "generated_answer": "yes"},
                }
            ],
        },
        {
            "status": "converted",
            "source_binary_route_id": "u::r1",
            "source_route_id": "r1",
            "source_binary_route": [0, 1],
            "source_off_count": 1,
            "all_off_seed": False,
            "label_semantics": "corrective_w2c",
            "final_branches": [
                {
                    "route": ["FULL", "READ_ONLY"],
                    "evaluation": {"correct": True, "generated_answer": "yes"},
                }
            ],
        },
    ]

    unique = deduplicate_sequential_routes(rows)

    assert len(unique) == 1
    assert unique[0]["source_binary_route_ids"] == ["u::r1", "u::r2"]
    assert unique[0]["num_FULL"] == 1
    assert unique[0]["num_READ_ONLY"] == 1
    assert unique[0]["read_suppression_count"] == 0
    assert unique[0]["write_suppression_count"] == 1


def test_public_refinement_contract_has_no_beam_or_branch_cap_parameter():
    parameters = inspect.signature(sequentially_refine_w2c).parameters

    assert "beam_width" not in parameters
    assert "branch_cap" not in parameters
