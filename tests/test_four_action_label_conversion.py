from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from experiments.analyze_four_action_label_conversion import build_final_decisions
import experiments.run_four_action_label_conversion as conversion_runner

from tools.research_analysis.four_action.label_conversion import (
    CachedRouteEvaluator,
    action_route_cost,
    binary_to_four_action,
    canonical_route,
    convert_valid_source_route,
    deduplicate_final_routes,
    purify_w2c_anchor,
    refine_w2c_anchor,
    select_diverse_four_action_routes,
)
from tools.research_analysis.four_action.label_sources import (
    normalize_math_record,
    normalize_vqa_record,
)
from tools.research_analysis.four_action.label_jobs import (
    AtomicSampleQueue,
    CONVERSION_CODE_PATHS,
    balanced_worker_rows,
    build_conversion_execution_contract,
    empirical_full_run_cost,
    select_conversion_pilot,
)


def _evaluation(correct_routes, margins=None):
    correct_routes = {tuple(route) for route in correct_routes}
    margins = {tuple(route): value for route, value in (margins or {}).items()}
    calls = Counter()

    def evaluate(route):
        key = tuple(route)
        calls[key] += 1
        return {
            "correct": key in correct_routes,
            "margin": margins.get(key),
            "generated_answer": "correct" if key in correct_routes else "wrong",
        }

    return evaluate, calls


def test_binary_mapping_and_suppression_cost_are_exact():
    route = binary_to_four_action([1, 0, True, False])
    assert route == ("FULL", "IGNORE", "FULL", "IGNORE")
    assert action_route_cost(route) == 4
    with pytest.raises(ValueError, match="binary"):
        binary_to_four_action([1, 2])


def test_purification_runs_early_and_late_to_fixed_point_then_uses_margin_tie_break():
    ignore = ("IGNORE", "IGNORE", "IGNORE")
    early = ("FULL", "IGNORE", "FULL")
    late = ("IGNORE", "FULL", "FULL")
    correct = {
        ignore,
        ("FULL", "IGNORE", "IGNORE"),
        early,
        ("IGNORE", "IGNORE", "FULL"),
        late,
    }
    evaluate, calls = _evaluation(correct, {early: 0.2, late: 0.7})
    cached = CachedRouteEvaluator(evaluate)

    result = purify_w2c_anchor(ignore, cached)

    assert result.route == late
    assert result.order == "late_to_early"
    assert result.candidates["early_to_late"].route == early
    assert result.candidates["late_to_early"].route == late
    assert result.candidates["early_to_late"].passes >= 2
    assert all(count == 1 for count in calls.values())


def test_monotone_beam_refinement_uses_joint_routes_and_returns_lowest_cost_correct_state():
    anchor = ("IGNORE", "IGNORE")
    best = ("WRITE_ONLY", "FULL")
    correct = {
        anchor,
        ("READ_ONLY", "IGNORE"),
        ("WRITE_ONLY", "IGNORE"),
        ("IGNORE", "READ_ONLY"),
        ("WRITE_ONLY", "READ_ONLY"),
        best,
    }
    evaluate, calls = _evaluation(correct, {best: 0.8})
    cached = CachedRouteEvaluator(evaluate)

    result = refine_w2c_anchor(anchor, cached, beam_width=8)

    assert result.route == best
    assert result.cost == 1
    assert result.evaluation["correct"] is True
    assert result.evaluated_route_count == len(calls)
    assert all(count == 1 for count in calls.values())


def test_refinement_counts_failures_of_individually_valid_relaxation_composition():
    anchor = ("IGNORE", "IGNORE")
    correct = {
        anchor,
        ("READ_ONLY", "IGNORE"),
        ("IGNORE", "READ_ONLY"),
    }
    evaluate, _ = _evaluation(correct)

    result = refine_w2c_anchor(anchor, CachedRouteEvaluator(evaluate), beam_width=8)

    assert result.first_round_correct_count == 2
    assert result.independently_supported_composite_count >= 1
    assert result.independent_composition_failure_count >= 1


def test_c2c_conversion_preserves_the_binary_route_without_purification():
    mapped = ("FULL", "IGNORE", "FULL")
    evaluate, calls = _evaluation({mapped})
    cached = CachedRouteEvaluator(evaluate)

    result = convert_valid_source_route(
        [1, 0, 1],
        full_correct=True,
        evaluate=cached,
    )

    assert result.label_semantics == "preserving_c2c"
    assert result.route == mapped
    assert result.purification is None
    assert result.refinement is None
    assert calls == Counter({mapped: 1})


def test_w2c_conversion_rejects_a_source_route_that_no_longer_replays_correct():
    evaluate, _ = _evaluation(set())
    with pytest.raises(ValueError, match="replay"):
        convert_valid_source_route(
            [1, 0, 1],
            full_correct=False,
            evaluate=CachedRouteEvaluator(evaluate),
        )


def test_vqa_source_normalization_uses_only_selected_positive_routes(tmp_path):
    image_root = tmp_path / "images"
    image_path = image_root / "gqa" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    source = {
        "uid": "gqa:1",
        "sample_id": "1",
        "benchmark": "gqa",
        "question": "q",
        "prompt": "p",
        "answer": "a",
        "all_answer_norms": None,
        "metric_name": "exact_match_ignore_case_punctuation",
        "correctness_threshold": 1.0,
        "max_new_tokens": 16,
        "local_image_path": "/old/sample.jpg",
        "image_group_id": "image:1",
        "image_size_bytes": 5,
    }
    predictor = {
        "uid": "gqa:1",
        "benchmark": "gqa",
        "split": "train",
        "current_all_on_status": "wrong",
        "selected_valid_route_count": 1,
        "valid_routes": [
            {"route_id": "r1", "mask": [1, 0, 1], "score": 1.0, "reward": 1.0}
        ],
    }

    row = normalize_vqa_record(
        predictor,
        source,
        image_root=image_root,
        layer_count=3,
        source_artifact="selected.jsonl",
    )

    assert row["dataset"] == "gqa"
    assert row["image_path"] == str(image_path.resolve())
    assert row["source_positive_route_count"] == 1
    assert row["source_positive_routes"][0]["source_binary_route_id"] == "gqa:1::r1"

    predictor["selected_valid_route_count"] = 0
    predictor["valid_routes"] = []
    assert (
        normalize_vqa_record(
            predictor,
            source,
            image_root=image_root,
            layer_count=3,
            source_artifact="selected.jsonl",
        )
        is None
    )


def test_math_source_normalization_rejects_declared_success_without_correct_candidate(tmp_path):
    image_path = tmp_path / "x.png"
    image_path.write_bytes(b"image")
    raw = {
        "sample": {
            "uid": "wemath2pro:1",
            "sample_id": "1",
            "benchmark": "wemath2pro",
            "source_split": "pro",
            "question": "q",
            "prompt": "p",
            "answer": "1",
            "all_answer_norms": None,
            "metric_name": "wemath2pro_mathruler_accuracy",
            "correctness_threshold": 1.0,
            "max_new_tokens": 96,
            "max_image_tokens": None,
            "local_image_path": "/old/x.png",
            "image_group_id": "image:1",
            "image_size_bytes": 5,
        },
        "successful_route_ids": ["r1"],
        "candidate_executions": [
            {"route_id": "r1", "visual_on_mask": [1, 0, 1], "result_correct": False}
        ],
    }
    with pytest.raises(ValueError, match="evaluator-correct"):
        normalize_math_record(
            raw,
            image_path=image_path,
            record_path=tmp_path / "record.json",
            record_sha256="abc",
            layer_count=3,
        )


def test_cost_balanced_worker_rows_are_disjoint_complete_and_lpt_balanced():
    rows = [
        {"uid": f"u{index}", "estimated_conversion_cost": cost}
        for index, cost in enumerate([9, 8, 7, 6, 5, 4])
    ]
    bins = [balanced_worker_rows(rows, rank, 3) for rank in range(3)]
    assert {row["uid"] for part in bins for row in part} == {row["uid"] for row in rows}
    assert sum(len(part) for part in bins) == len(rows)
    costs = [sum(row["estimated_conversion_cost"] for row in part) for part in bins]
    assert max(costs) - min(costs) <= 1


def test_atomic_sample_queue_dynamically_claims_each_pending_sample_once(tmp_path):
    rows = [
        {
            "uid": "correct-heavy",
            "source_current_all_on_status": "correct",
            "source_positive_route_count": 20,
            "estimated_conversion_cost": 100,
        },
        {
            "uid": "wrong-medium",
            "source_current_all_on_status": "wrong",
            "source_positive_route_count": 1,
            "estimated_conversion_cost": 20,
        },
        {
            "uid": "already-done",
            "source_current_all_on_status": "wrong",
            "source_positive_route_count": 1,
            "estimated_conversion_cost": 1000,
        },
        {
            "uid": "correct-small",
            "source_current_all_on_status": "correct",
            "source_positive_route_count": 1,
            "estimated_conversion_cost": 2,
        },
    ]
    queue_a = AtomicSampleQueue(
        rows,
        claim_root=tmp_path / "claims",
        completed_uids={"already-done"},
        claimant="rank-0",
    )
    queue_b = AtomicSampleQueue(
        rows,
        claim_root=tmp_path / "claims",
        completed_uids={"already-done"},
        claimant="rank-1",
    )

    claimed = [
        queue_a.claim_next(),
        queue_b.claim_next(),
        queue_a.claim_next(),
    ]

    assert [row["uid"] for row in claimed if row is not None] == [
        "wrong-medium",
        "correct-heavy",
        "correct-small",
    ]
    assert queue_b.claim_next() is None
    assert len(list((tmp_path / "claims").glob("*.json"))) == 3


def test_empirical_full_run_cost_uses_pilot_w2c_multiplier():
    base = {
        "source_positive_route_count": 5,
        "estimated_conversion_cost": 20,
    }
    assert empirical_full_run_cost({**base, "source_current_all_on_status": "wrong"}) == 360
    assert empirical_full_run_cost({**base, "source_current_all_on_status": "correct"}) == 20


def test_conversion_execution_contract_hashes_manifest_config_and_code(tmp_path):
    for relative in CONVERSION_CODE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    manifest_path = tmp_path / "manifest.jsonl"
    config_path.write_text("model: {}\n", encoding="utf-8")
    manifest_path.write_text('{"uid":"u"}\n', encoding="utf-8")

    contract = build_conversion_execution_contract(
        project_root=tmp_path,
        config_path=config_path,
        manifest_path=manifest_path,
        config={
            "model": {
                "snapshot_path": "models/qwen",
                "revision": "revision-1",
                "attention_implementation": "sdpa",
            },
            "seed": 7,
            "beam_width": 8,
            "layer_count": 28,
        },
        git_commit="abc123",
        torch_version="2.6.0",
        transformers_version="5.3.0",
    )

    assert contract["git_commit"] == "abc123"
    assert contract["model_revision"] == "revision-1"
    assert contract["source_manifest_sha256"]
    assert contract["config_sha256"]
    assert len(contract["contract_sha256"]) == 64
    assert set(contract["code_sha256"]) == set(CONVERSION_CODE_PATHS)


def test_sample_runner_records_and_reuses_current_unified_all_off(monkeypatch):
    class FakeRuntime:
        def __init__(self, **_kwargs):
            self.calls = Counter()
            self.full_evaluation = {
                "generated_ids": [1],
                "generated_answer": "yes",
                "correct": True,
                "answer_alignment_margin": 0.5,
            }
            self.input_metadata = {"prompt_sha256": "prompt"}
            self.geometry = {"text_tokens": 2, "visual_tokens": 1, "full_prompt_tokens": 3}

        def initialize_full(self):
            return SimpleNamespace(evaluation=dict(self.full_evaluation))

        def evaluate(self, route):
            route = tuple(route)
            self.calls[route] += 1
            if route == ("FULL", "FULL"):
                return dict(self.full_evaluation)
            return {
                "generated_ids": [1],
                "generated_answer": "yes",
                "correct": True,
                "answer_alignment_margin": 0.5,
            }

    runtime = FakeRuntime()
    monkeypatch.setattr(
        conversion_runner,
        "FourActionSampleRuntime",
        lambda **_kwargs: runtime,
    )
    monkeypatch.setattr(conversion_runner.torch.cuda, "reset_peak_memory_stats", lambda _d: None)
    monkeypatch.setattr(conversion_runner.torch.cuda, "max_memory_allocated", lambda _d: 10)

    result = conversion_runner.process_sample(
        processor=None,
        model=None,
        record={
            "uid": "gqa:u",
            "dataset": "gqa",
            "sample_id": "u",
            "source_split": "train",
            "source_current_all_on_status": "correct",
            "source_current_all_on_prediction": "yes",
            "source_positive_route_count": 1,
            "source_positive_routes": [
                {
                    "source_binary_route_id": "gqa:u::r1",
                    "route_id": "r1",
                    "mask": [0, 0],
                    "source_off_count": 2,
                    "source_all_off": True,
                }
            ],
        },
        device=SimpleNamespace(index=0),
        config={"layer_count": 2, "beam_width": 8, "scoring_timeout_seconds": 1.0},
        rank=0,
        mode="full",
        execution_contract={"contract_sha256": "hash"},
    )

    assert result["current_unified_all_off"]["correct"] is True
    assert runtime.calls[("FULL", "FULL")] == 1
    assert runtime.calls[("IGNORE", "IGNORE")] == 1


def test_pilot_selection_covers_each_dataset_and_available_semantic_extremes():
    rows = []
    for dataset in ("gqa", "textvqa", "chartqa", "wemath20_standard", "wemath2pro"):
        rows.extend(
            [
                {
                    "uid": f"{dataset}:w2c_all_off",
                    "dataset": dataset,
                    "source_current_all_on_status": "wrong",
                    "source_positive_routes": [
                        {"source_all_off": True, "source_off_count": 3}
                    ],
                    "source_positive_route_count": 1,
                    "estimated_conversion_cost": 4,
                },
                {
                    "uid": f"{dataset}:c2c_multi",
                    "dataset": dataset,
                    "source_current_all_on_status": "correct",
                    "source_positive_routes": [
                        {"source_all_off": False, "source_off_count": 1},
                        {"source_all_off": False, "source_off_count": 2},
                    ],
                    "source_positive_route_count": 2,
                    "estimated_conversion_cost": 5,
                },
            ]
        )

    selected, coverage = select_conversion_pilot(rows, total=10)

    assert len(selected) == 10
    assert {row["dataset"] for row in selected} == {
        "gqa", "textvqa", "chartqa", "wemath20_standard", "wemath2pro"
    }
    assert coverage["source_w2c_proxy"] == 5
    assert coverage["source_c2c_proxy"] == 5
    assert coverage["all_off_w2c_proxy"] == 5
    assert coverage["multi_route_samples"] == 5


def test_final_route_dedup_preserves_all_source_route_ids_and_canonical_semantics():
    rows = [
        {
            "status": "converted",
            "source_binary_route_id": "u::r2",
            "label_semantics": "corrective_w2c",
            "final_route": ["FULL", "READ_ONLY"],
            "final_evaluation": {"correct": True, "answer_alignment_margin": 0.4},
        },
        {
            "status": "converted",
            "source_binary_route_id": "u::r1",
            "label_semantics": "corrective_w2c",
            "final_route": ["FULL", "READ_ONLY"],
            "final_evaluation": {"correct": True, "answer_alignment_margin": 0.4},
        },
        {
            "status": "converted",
            "source_binary_route_id": "u::r3",
            "label_semantics": "corrective_w2c",
            "final_route": ["IGNORE", "FULL"],
            "final_evaluation": {"correct": True, "answer_alignment_margin": 0.8},
        },
    ]

    unique = deduplicate_final_routes(rows)

    assert len(unique) == 2
    first = next(row for row in unique if row["route"] == ["FULL", "READ_ONLY"])
    assert first["source_binary_route_ids"] == ["u::r1", "u::r2"]
    assert first["suppression_component_cost"] == 1
    assert canonical_route(unique, label_semantics="corrective_w2c")["route"] == [
        "FULL", "READ_ONLY"
    ]
    c2c = [{**row, "label_semantics": "preserving_c2c"} for row in unique]
    assert canonical_route(c2c, label_semantics="preserving_c2c")["route"] == [
        "IGNORE", "FULL"
    ]


def test_four_action_training_cap_is_deterministic_diverse_and_keeps_anchors():
    actions = ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")
    routes = []
    for left in actions:
        for middle in actions:
            for right in actions:
                route = (left, middle, right)
                routes.append(
                    {
                        "route": list(route),
                        "route_key": "|".join(route),
                        "suppression_component_cost": action_route_cost(route),
                        "full_count": route.count("FULL"),
                        "read_only_count": route.count("READ_ONLY"),
                        "write_only_count": route.count("WRITE_ONLY"),
                        "ignore_count": route.count("IGNORE"),
                    }
                )

    first = select_diverse_four_action_routes(
        routes,
        limit=12,
        seed=7,
        uid="u",
        canonical_route_key="FULL|READ_ONLY|WRITE_ONLY",
    )
    second = select_diverse_four_action_routes(
        list(reversed(routes)),
        limit=12,
        seed=7,
        uid="u",
        canonical_route_key="FULL|READ_ONLY|WRITE_ONLY",
    )

    assert [row["route_key"] for row in first] == [row["route_key"] for row in second]
    assert len(first) == 12
    assert "FULL|READ_ONLY|WRITE_ONLY" in {row["route_key"] for row in first}
    assert "FULL|FULL|FULL" in {row["route_key"] for row in first}
    assert "IGNORE|IGNORE|IGNORE" in {row["route_key"] for row in first}


def test_final_decisions_make_plan_required_training_and_search_answers_explicit():
    combined = {
        "counts": {
            "source_samples": 100,
            "source_replay_valid_routes": 900,
            "source_replay_failure_routes": 10,
            "unique_valid_routes": 250,
            "corrective_w2c_unique_routes": 200,
            "corrective_w2c_unique_routes_using_read_only": 80,
            "corrective_w2c_unique_routes_using_write_only": 70,
            "corrective_w2c_unique_routes_using_read_or_write_only": 120,
            "samples_without_current_valid_route": 0,
        }
    }
    audit = {
        "passed": True,
        "unresolved_worker_failure_rows": [],
    }

    decisions = build_final_decisions(combined, audit)

    assert decisions["successfully_converted_binary_labels"] == 900
    assert decisions["unique_valid_four_action_labels"] == 250
    assert decisions["read_write_structure"]["substantial"] is True
    assert decisions["read_write_structure"]["using_either_fraction"] == pytest.approx(0.6)
    assert decisions["keep_w2c_and_c2c_separate"] is True
    assert decisions["training_readiness"]["decision"] == "ready"
    assert decisions["fresh_four_action_search"]["decision"] == "not_needed"
