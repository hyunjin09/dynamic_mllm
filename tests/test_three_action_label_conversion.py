from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.run_three_action_label_conversion as three_runner
from experiments.finalize_three_action_noise_calibration import build_calibration_report
from experiments.audit_three_action_label_pilot import build_pilot_audit
from experiments.audit_three_action_label_full import build_full_audit
from experiments.finalize_three_action_label_conversion import build_final_views
from experiments.analyze_three_action_label_conversion import analyze_records
from experiments.estimate_three_action_label_conversion import build_compute_estimate
from experiments.write_three_action_label_checksum_ledger import build_checksum_ledger
import tools.research_analysis.four_action.label_runtime as label_runtime
from tools.research_analysis.four_action.label_runtime import FourActionSampleRuntime
from tools.research_analysis.four_action.three_action_jobs import (
    THREE_ACTION_CODE_PATHS,
    build_three_action_execution_contract,
)

from tools.research_analysis.four_action.three_action_labels import (
    CachedThreeActionEvaluator,
    binary_to_three_action,
    calibrate_repeatability_epsilon,
    deduplicate_positive_routes,
    decompose_screened_positions,
    evaluate_independent_composition,
    refine_three_action_route,
    screen_binary_off_positions,
    select_canonical_w2c_route,
    select_canonical_c2c_route,
    select_diverse_three_action_routes,
    three_action_to_executor,
)


def _evaluator(states):
    calls = Counter()

    def evaluate(executor_route):
        semantic = tuple(
            {
                "FULL": "FULL",
                "WRITE_ONLY": "READ_OFF",
                "READ_ONLY": "WRITE_OFF",
                "IGNORE": "BOTH_OFF",
            }[action]
            for action in executor_route
        )
        calls[semantic] += 1
        state = states.get(semantic, {})
        return {
            "correct": bool(state.get("correct", False)),
            "answer_alignment_margin": float(state.get("margin", -10.0)),
            "S_correct": float(state.get("support", state.get("margin", -10.0))),
            "generated_answer": "correct" if state.get("correct", False) else "wrong",
        }

    return evaluate, calls


def test_three_action_aliases_reuse_the_existing_executor_exactly():
    assert binary_to_three_action([1, 0, 1, 0]) == (
        "FULL",
        "BOTH_OFF",
        "FULL",
        "BOTH_OFF",
    )
    assert three_action_to_executor(
        ("FULL", "READ_OFF", "WRITE_OFF", "BOTH_OFF")
    ) == ("FULL", "WRITE_ONLY", "READ_ONLY", "IGNORE")


def test_w2c_screening_retains_hard_and_soft_but_restores_redundant_positions():
    anchor = ("BOTH_OFF", "BOTH_OFF", "BOTH_OFF")
    states = {
        anchor: {"correct": True, "margin": 1.0},
        ("FULL", "BOTH_OFF", "BOTH_OFF"): {"correct": False, "margin": -0.4},
        ("BOTH_OFF", "FULL", "BOTH_OFF"): {"correct": True, "margin": 0.4},
        ("BOTH_OFF", "BOTH_OFF", "FULL"): {"correct": True, "margin": 1.0},
        ("FULL", "BOTH_OFF", "FULL"): {"correct": False, "margin": -0.2},
        ("BOTH_OFF", "FULL", "FULL"): {"correct": True, "margin": 0.4},
    }
    evaluate, _ = _evaluator(states)
    cached = CachedThreeActionEvaluator(evaluate)

    result = screen_binary_off_positions(
        anchor,
        route_type="W2C",
        evaluate=cached,
        epsilon=0.1,
    )

    assert result.route == ("BOTH_OFF", "BOTH_OFF", "FULL")
    classifications = {row.layer: row.classification for row in result.positions}
    assert classifications == {
        0: "HARD_NECESSARY",
        1: "SOFT_ALIGNMENT_HELPFUL",
        2: "REDUNDANT",
    }
    assert result.route_evaluation["correct"] is True


def test_c2c_screening_keeps_support_gain_and_context_dependence_only():
    anchor = ("BOTH_OFF", "BOTH_OFF", "BOTH_OFF")
    states = {
        anchor: {"correct": True, "support": 0.9},
        ("FULL", "BOTH_OFF", "BOTH_OFF"): {"correct": True, "support": 0.4},
        ("BOTH_OFF", "FULL", "BOTH_OFF"): {"correct": True, "support": 0.9},
        ("BOTH_OFF", "BOTH_OFF", "FULL"): {"correct": False, "support": 0.2},
        ("FULL", "FULL", "BOTH_OFF"): {"correct": True, "support": 0.4},
        ("BOTH_OFF", "FULL", "FULL"): {"correct": False, "support": 0.1},
    }
    evaluate, _ = _evaluator(states)

    result = screen_binary_off_positions(
        anchor,
        route_type="C2C",
        evaluate=CachedThreeActionEvaluator(evaluate),
        epsilon=0.1,
    )

    assert result.route == ("BOTH_OFF", "FULL", "BOTH_OFF")
    classifications = {row.layer: row.classification for row in result.positions}
    assert classifications == {
        0: "SOFT_ALIGNMENT_HELPFUL",
        1: "REDUNDANT",
        2: "CONTEXT_DEPENDENT_NECESSARY",
    }


def test_decomposition_reuses_both_off_and_full_reference_and_adds_only_two_forwards():
    anchor = ("BOTH_OFF", "FULL")
    states = {
        anchor: {"correct": True, "margin": 1.0},
        ("FULL", "FULL"): {"correct": False, "margin": -1.0},
        ("READ_OFF", "FULL"): {"correct": True, "margin": 0.8},
        ("WRITE_OFF", "FULL"): {"correct": False, "margin": 0.2},
    }
    evaluate, calls = _evaluator(states)
    cached = CachedThreeActionEvaluator(evaluate)
    screening = screen_binary_off_positions(
        anchor,
        route_type="W2C",
        evaluate=cached,
        epsilon=0.1,
    )
    misses_before = cached.cache_misses

    rows = decompose_screened_positions(screening, evaluate=cached, epsilon=0.1)

    assert cached.cache_misses - misses_before == 2
    assert calls[anchor] == 1
    assert calls[("FULL", "FULL")] == 1
    assert rows[0].action_classification == "READ_SUPPRESSION"


def test_w2c_beam_keeps_high_margin_wrong_partial_state_that_enables_joint_rescue():
    anchor = ("BOTH_OFF", "BOTH_OFF")
    states = {
        anchor: {"correct": True, "margin": 0.0},
        ("READ_OFF", "BOTH_OFF"): {"correct": False, "margin": 2.0},
        ("WRITE_OFF", "BOTH_OFF"): {"correct": False, "margin": -1.0},
        ("BOTH_OFF", "READ_OFF"): {"correct": True, "margin": 0.1},
        ("BOTH_OFF", "WRITE_OFF"): {"correct": True, "margin": 0.2},
        ("READ_OFF", "READ_OFF"): {"correct": False, "margin": 2.1},
        ("READ_OFF", "WRITE_OFF"): {"correct": True, "margin": 3.0},
    }
    evaluate, _ = _evaluator(states)
    cached = CachedThreeActionEvaluator(evaluate)

    result = refine_three_action_route(
        anchor,
        candidate_layers=(0, 1),
        route_type="W2C",
        evaluate=cached,
        epsilon=0.1,
        beam_width=2,
        unified_full_evaluation={"correct": False, "answer_alignment_margin": -1.0},
    )

    assert result.max_margin_route.route == ("READ_OFF", "WRITE_OFF")
    assert result.max_margin_route.evaluation["correct"] is True
    assert any(
        row.route == ("READ_OFF", "BOTH_OFF")
        for row in result.corrective_partial_candidates
    )
    assert all(row.evaluation["correct"] for row in result.positive_routes)


def test_c2c_positive_routes_require_global_support_gain_above_epsilon():
    anchor = ("BOTH_OFF",)
    states = {
        anchor: {"correct": True, "support": 0.9},
        ("READ_OFF",): {"correct": True, "support": 1.2},
        ("WRITE_OFF",): {"correct": True, "support": 1.05},
    }
    evaluate, _ = _evaluator(states)
    result = refine_three_action_route(
        anchor,
        candidate_layers=(0,),
        route_type="C2C",
        evaluate=CachedThreeActionEvaluator(evaluate),
        epsilon=0.1,
        beam_width=8,
        unified_full_evaluation={"correct": True, "S_correct": 1.0},
    )

    assert [row.route for row in result.positive_routes] == [("READ_OFF",)]


def test_independently_best_local_suppressions_are_executed_jointly_and_failure_is_saved():
    anchor = ("BOTH_OFF", "BOTH_OFF")
    states = {
        anchor: {"correct": True, "margin": 1.0},
        ("FULL", "BOTH_OFF"): {"correct": False, "margin": -1.0},
        ("BOTH_OFF", "FULL"): {"correct": False, "margin": -1.0},
        ("READ_OFF", "BOTH_OFF"): {"correct": True, "margin": 2.0},
        ("WRITE_OFF", "BOTH_OFF"): {"correct": True, "margin": 0.0},
        ("BOTH_OFF", "READ_OFF"): {"correct": True, "margin": 2.0},
        ("BOTH_OFF", "WRITE_OFF"): {"correct": True, "margin": 0.0},
        ("READ_OFF", "READ_OFF"): {"correct": False, "margin": 3.0},
    }
    evaluate, _ = _evaluator(states)
    cached = CachedThreeActionEvaluator(evaluate)
    screening = screen_binary_off_positions(
        anchor, route_type="W2C", evaluate=cached, epsilon=0.1
    )
    decomposition = decompose_screened_positions(
        screening, evaluate=cached, epsilon=0.1
    )

    result = evaluate_independent_composition(
        screening,
        decomposition,
        evaluate=cached,
        epsilon=0.1,
        unified_full_evaluation={"correct": False, "answer_alignment_margin": -1.0},
    )

    assert result["route"] == ["READ_OFF", "READ_OFF"]
    assert result["evaluation"]["correct"] is False
    assert result["all_local_actions_supported"] is True
    assert result["joint_positive"] is False
    assert result["independent_composition_failure"] is True


def test_repeatability_epsilon_uses_predeclared_floor_or_empirical_p99():
    report = calibrate_repeatability_epsilon(
        signed_differences=[0.0, 0.01, -0.02, 0.03],
        floor=0.005,
        quantile=0.75,
    )

    assert report["absolute_p75"] == pytest.approx(0.0225)
    assert report["epsilon"] == pytest.approx(0.0225)
    assert report["selection_rule"] == "max(predeclared_floor, empirical_absolute_repeat_difference_p75)"


def test_w2c_canonical_prefers_low_cost_only_within_epsilon_of_best_seed_margin():
    routes = [
        {"route": ["BOTH_OFF"], "suppression_cost": 2, "evaluation": {"correct": True, "answer_alignment_margin": 1.0}},
        {"route": ["READ_OFF"], "suppression_cost": 1, "evaluation": {"correct": True, "answer_alignment_margin": 0.96}},
        {"route": ["FULL"], "suppression_cost": 0, "evaluation": {"correct": True, "answer_alignment_margin": 0.7}},
    ]

    selected = select_canonical_w2c_route(routes, best_seed_margin=1.0, epsilon=0.05)

    assert selected["route"] == ["READ_OFF"]


def test_w2c_canonical_falls_back_to_best_refined_margin_when_all_exceed_seed_tolerance():
    routes = [
        {"route": ["READ_OFF"], "suppression_cost": 1, "evaluation": {"correct": True, "answer_alignment_margin": 0.8}},
        {"route": ["BOTH_OFF"], "suppression_cost": 2, "evaluation": {"correct": True, "answer_alignment_margin": 0.9}},
    ]

    selected = select_canonical_w2c_route(routes, best_seed_margin=1.2, epsilon=0.05)

    assert selected["route"] == ["BOTH_OFF"]
    assert selected["canonical_within_seed_epsilon"] is False


def test_uncached_runtime_evaluation_reexecutes_identical_unified_route(monkeypatch):
    captures = []

    def capture(_model, _inputs, route, **_kwargs):
        captures.append(tuple(route))
        return SimpleNamespace(cache=object())

    monkeypatch.setattr(label_runtime, "capture_four_action_route", capture)
    runtime = object.__new__(FourActionSampleRuntime)
    runtime.model = object()
    runtime.inputs = {"input_ids": object()}
    runtime.prepared = object()
    runtime.initialize_full = lambda: SimpleNamespace(
        correct_targets=("correct",), wrong_target=("wrong",)
    )
    runtime._score_output = lambda _output, **_kwargs: {
        "correct": True,
        "S_correct": 1.0,
        "answer_alignment_margin": 2.0,
    }

    first = runtime.evaluate_uncached(("FULL", "IGNORE"))
    second = runtime.evaluate_uncached(("FULL", "IGNORE"))

    assert first == second
    assert captures == [("FULL", "IGNORE"), ("FULL", "IGNORE")]


def test_positive_route_dedup_preserves_source_provenance_and_c2c_canonical_rule():
    conversions = [
        {
            "status": "converted",
            "source_binary_route_id": "u::r2",
            "label_semantics": "C2C_COMPENSATED_ALIGNMENT",
            "positive_routes": [
                {
                    "route": ["READ_OFF", "FULL"],
                    "evaluation": {"correct": True, "S_correct": 1.2, "generated_ids": [1]},
                    "suppression_cost": 1,
                }
            ],
        },
        {
            "status": "converted",
            "source_binary_route_id": "u::r1",
            "label_semantics": "C2C_COMPENSATED_ALIGNMENT",
            "positive_routes": [
                {
                    "route": ["READ_OFF", "FULL"],
                    "evaluation": {"correct": True, "S_correct": 1.2, "generated_ids": [1]},
                    "suppression_cost": 1,
                },
                {
                    "route": ["BOTH_OFF", "FULL"],
                    "evaluation": {"correct": True, "S_correct": 1.4, "generated_ids": [2]},
                    "suppression_cost": 2,
                },
            ],
        },
    ]

    unique = deduplicate_positive_routes(conversions)
    canonical = select_canonical_c2c_route(unique)

    assert len(unique) == 2
    read = next(row for row in unique if row["route"] == ["READ_OFF", "FULL"])
    assert read["source_binary_route_ids"] == ["u::r1", "u::r2"]
    assert canonical["route"] == ["READ_OFF", "FULL"]


def test_three_action_training_cap_is_deterministic_and_keeps_canonical_route():
    routes = []
    for left in ("FULL", "READ_OFF", "WRITE_OFF", "BOTH_OFF"):
        for right in ("FULL", "READ_OFF", "WRITE_OFF", "BOTH_OFF"):
            route = [left, right]
            routes.append({
                "route": route,
                "route_key": "|".join(route),
                "suppression_cost": {"FULL": 0, "READ_OFF": 1, "WRITE_OFF": 1, "BOTH_OFF": 2}[left]
                + {"FULL": 0, "READ_OFF": 1, "WRITE_OFF": 1, "BOTH_OFF": 2}[right],
            })

    first = select_diverse_three_action_routes(
        routes, limit=6, seed=7, uid="u", canonical_route_key="READ_OFF|FULL"
    )
    second = select_diverse_three_action_routes(
        list(reversed(routes)), limit=6, seed=7, uid="u", canonical_route_key="READ_OFF|FULL"
    )

    assert [row["route_key"] for row in first] == [row["route_key"] for row in second]
    assert "READ_OFF|FULL" in {row["route_key"] for row in first}


def test_three_action_contract_binds_frozen_epsilon_and_compatibility_runtime(tmp_path):
    for relative in THREE_ACTION_CODE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    manifest_path = tmp_path / "manifest.jsonl"
    epsilon_path = tmp_path / "epsilon.json"
    config_path.write_text("three: action\n", encoding="utf-8")
    manifest_path.write_text('{"uid":"u"}\n', encoding="utf-8")
    epsilon_path.write_text(json.dumps({"epsilon": 0.01}), encoding="utf-8")

    contract = build_three_action_execution_contract(
        project_root=tmp_path,
        config_path=config_path,
        manifest_path=manifest_path,
        epsilon_path=epsilon_path,
        config={
            "model": {
                "snapshot_path": "model",
                "revision": "r",
                "attention_implementation": "sdpa",
            },
            "seed": 7,
            "beam_width": 8,
            "beam_validation_width": 16,
            "layer_count": 28,
        },
        git_commit="abc",
        torch_version="2.6.0",
        transformers_version="5.3.0",
        mode="pilot",
    )

    assert contract["epsilon"] == pytest.approx(0.01)
    assert contract["epsilon_sha256"]
    assert "label_regeneration/runtime.py" in contract["code_sha256"]
    assert "experiments/finalize_three_action_noise_calibration.py" in contract["code_sha256"]
    assert len(contract["contract_sha256"]) == 64


def test_calibration_sample_reexecutes_full_and_source_routes_without_route_cache(monkeypatch):
    class FakeRuntime:
        def __init__(self, **_kwargs):
            self.calls = Counter()
            self.input_metadata = {"prompt": "frozen"}
            self.geometry = {"text_tokens": 2, "visual_tokens": 1, "full_prompt_tokens": 3}

        def initialize_full(self):
            return SimpleNamespace(
                evaluation={
                    "correct": True,
                    "generated_ids": [1],
                    "generated_answer": "yes",
                    "S_correct": 1.0,
                    "answer_alignment_margin": 1.0,
                }
            )

        def evaluate_uncached(self, route):
            route = tuple(route)
            self.calls[route] += 1
            offset = (0.0, 0.01, -0.01)[self.calls[route] - 1]
            return {
                "correct": True,
                "generated_ids": [1],
                "generated_answer": "yes",
                "S_correct": 1.0 + offset,
                "answer_alignment_margin": 1.0 + offset,
            }

    runtime = FakeRuntime()
    monkeypatch.setattr(three_runner, "FourActionSampleRuntime", lambda **_kwargs: runtime)
    monkeypatch.setattr(three_runner.torch.cuda, "reset_peak_memory_stats", lambda _d: None)
    monkeypatch.setattr(three_runner.torch.cuda, "max_memory_allocated", lambda _d: 10)

    result = three_runner.calibrate_sample(
        processor=None,
        model=None,
        record={
            "uid": "gqa:u",
            "dataset": "gqa",
            "sample_id": "u",
            "source_split": "train",
            "source_positive_routes": [{"mask": [0, 1], "source_binary_route_id": "u::r1"}],
        },
        device=SimpleNamespace(index=0),
        config={"scoring_timeout_seconds": 1.0, "layer_count": 2, "noise_calibration": {"repetitions": 3}},
        rank=0,
        execution_contract={"contract_sha256": "calibration"},
    )

    assert len(result["repeatability_controls"]) == 2
    assert all(len(row["evaluations"]) == 3 for row in result["repeatability_controls"])
    assert runtime.calls[("FULL", "FULL")] == 3
    assert runtime.calls[("IGNORE", "FULL")] == 3


def test_modified_runner_emits_answer_aligned_w2c_routes_and_beam_stability(monkeypatch):
    semantic_states = {
        ("FULL", "FULL"): {"correct": False, "margin": -1.0},
        ("BOTH_OFF", "BOTH_OFF"): {"correct": True, "margin": 1.0},
        ("FULL", "BOTH_OFF"): {"correct": False, "margin": -0.5},
        ("BOTH_OFF", "FULL"): {"correct": True, "margin": 1.0},
        ("READ_OFF", "FULL"): {"correct": True, "margin": 0.8},
        ("WRITE_OFF", "FULL"): {"correct": False, "margin": 0.2},
    }

    class FakeRuntime:
        input_metadata = {"prompt": "frozen"}
        geometry = {"text_tokens": 2, "visual_tokens": 1, "full_prompt_tokens": 3}

        def __init__(self, **_kwargs):
            pass

        def initialize_full(self):
            return SimpleNamespace(evaluation=self.evaluate(("FULL", "FULL")))

        def evaluate(self, executor_route):
            semantic = tuple(
                {"FULL": "FULL", "WRITE_ONLY": "READ_OFF", "READ_ONLY": "WRITE_OFF", "IGNORE": "BOTH_OFF"}[action]
                for action in executor_route
            )
            state = semantic_states[semantic]
            return {
                "correct": state["correct"],
                "generated_ids": [1] if state["correct"] else [2],
                "generated_answer": "yes" if state["correct"] else "no",
                "S_correct": state["margin"],
                "S_full_wrong": 0.0,
                "answer_alignment_margin": state["margin"],
            }

        def evaluate_old_binary(self, _mask):
            return self.evaluate(("IGNORE", "IGNORE"))

    monkeypatch.setattr(three_runner, "FourActionSampleRuntime", FakeRuntime)
    monkeypatch.setattr(three_runner.torch.cuda, "reset_peak_memory_stats", lambda _d: None)
    monkeypatch.setattr(three_runner.torch.cuda, "max_memory_allocated", lambda _d: 10)

    result = three_runner.process_sample(
        processor=None,
        model=None,
        record={
            "uid": "gqa:u",
            "dataset": "gqa",
            "sample_id": "u",
            "source_split": "train",
            "source_current_all_on_status": "wrong",
            "source_current_all_on_prediction": "no",
            "source_positive_route_count": 1,
            "source_positive_routes": [{
                "mask": [0, 0],
                "source_binary_route_id": "u::r1",
                "route_id": "r1",
                "source_off_count": 2,
                "source_all_off": True,
            }],
        },
        device=SimpleNamespace(index=0),
        config={
            "scoring_timeout_seconds": 1.0,
            "layer_count": 2,
            "beam_width": 8,
            "beam_validation_width": 16,
        },
        epsilon=0.1,
        rank=0,
        mode="pilot",
        execution_contract={"contract_sha256": "conversion"},
    )

    conversion = result["raw_conversions"][0]
    assert result["route_type"] == "W2C"
    assert conversion["label_semantics"] == "W2C_HARD_CORRECTIVE"
    assert conversion["screening"]["positions"][0]["classification"] == "HARD_NECESSARY"
    assert conversion["screening"]["positions"][1]["classification"] == "REDUNDANT"
    assert all(row["evaluation"]["correct"] for row in conversion["positive_routes"])
    assert conversion["pilot_beam_stability"]["canonical_route_match"] is True
    assert conversion["execution_efficiency"]["decomposition_new_cache_misses"] == 2
    assert conversion["execution_efficiency"]["theoretical_four_state_evaluations_avoided"] == 2


def test_calibration_report_freezes_within_unified_epsilon_and_rejects_native_drift():
    records = [
        {
            "uid": "gqa:u",
            "dataset": "gqa",
            "passed": True,
            "route_type": "W2C",
            "execution_contract": {"contract_sha256": "one"},
            "repeatability_controls": [
                {
                    "control": "unified_full",
                    "score_quantity": "answer_alignment_margin",
                    "signed_differences_from_first": [0.0, 0.02],
                    "generated_ids_identical": True,
                    "correctness_identical": True,
                }
            ],
        }
    ]

    report = build_calibration_report(
        records,
        expected_uids={"gqa:u"},
        floor=0.005,
        quantile=0.5,
        failure_rows=[{"uid": "gqa:u", "exception": "resolved transient failure"}],
    )

    assert report["passed"] is True
    assert report["epsilon"] == pytest.approx(0.01)
    assert report["threshold_source"] == "within_unified_identical_route_repeatability"
    assert "native" not in report["selection_rule"]
    assert report["historical_failure_count"] == 1
    assert report["unresolved_failure_count"] == 0


def test_calibration_report_rejects_a_missing_execution_contract_hash():
    base = {
        "dataset": "gqa",
        "passed": True,
        "route_type": "C2C",
        "repeatability_controls": [{
            "score_quantity": "S_correct",
            "signed_differences_from_first": [0.0],
            "generated_ids_identical": True,
            "correctness_identical": True,
        }],
    }
    report = build_calibration_report(
        [
            {**base, "uid": "gqa:a", "execution_contract": {"contract_sha256": "one"}},
            {**base, "uid": "gqa:b", "execution_contract": {}},
        ],
        expected_uids={"gqa:a", "gqa:b"},
        floor=0.0,
        quantile=0.99,
        failure_rows=[],
    )

    assert report["passed"] is False
    assert report["checks"]["single_execution_contract"] is False


def test_new_semantics_pilot_audit_requires_hard_soft_c2c_beam_and_16_workers():
    datasets = ("gqa", "textvqa", "chartqa", "wemath20_standard", "wemath2pro")
    records = []
    for index, dataset in enumerate(datasets):
        route_type = "W2C" if index < 2 else "C2C"
        classification = (
            "HARD_NECESSARY" if index == 0
            else "SOFT_ALIGNMENT_HELPFUL" if index == 1
            else "SOFT_ALIGNMENT_HELPFUL"
        )
        label = (
            "W2C_HARD_CORRECTIVE" if index == 0
            else "W2C_SOFT_ALIGNMENT" if index == 1
            else "C2C_COMPENSATED_ALIGNMENT"
        )
        full_support = 0.0 if route_type == "W2C" else 1.0
        records.append(
            {
                "uid": f"{dataset}:u",
                "dataset": dataset,
                "passed": True,
                "route_type": route_type,
                "epsilon": 0.01,
                "execution_contract": {"contract_sha256": "one", "epsilon_sha256": "eps"},
                "current_unified_full": {
                    "correct": route_type == "C2C",
                    "S_correct": full_support,
                    "score_quantity": "S_correct" if route_type == "C2C" else "S_correct_minus_S_full_wrong",
                    "correct_target_scores": [{"text": "yes", "evaluator_score": 1.0}],
                    "selected_correct_target": {"text": "yes", "evaluator_score": 1.0},
                },
                "source_positive_route_count": 1,
                "source_route_replay_valid_count": 1,
                "source_route_replay_failure_count": 0,
                "pilot_old_binary_semantic_checks": [{
                    "generated_ids_match": True,
                    "generated_answer_match": True,
                    "correctness_match": True,
                }],
                "raw_conversions": [{
                    "status": "converted",
                    "label_semantics": label,
                    "screening": {"positions": [{"classification": classification}]},
                    "positive_routes": [{
                        "route": ["READ_OFF"],
                        "evaluation": {"correct": True, "S_correct": 1.2, "answer_alignment_margin": 1.0},
                    }],
                    "corrective_partial_candidates": [],
                    "pilot_beam_stability": {
                        "canonical_route_match": True,
                        "both_have_no_positive_route": False,
                        "positive_route_jaccard": 1.0,
                    },
                    "execution_efficiency": {
                        "candidate_positions": 1,
                        "decomposition_new_cache_misses": 2,
                        "theoretical_four_state_evaluations_avoided": 2,
                    },
                }],
                "unique_valid_three_action_routes": [{
                    "route": ["READ_OFF"],
                    "evaluation": {"correct": True, "S_correct": 1.2},
                }],
            }
        )
    progress = [
        {"event": "worker_start", "rank": rank, "gpu_index": rank // 2, "replica_index": rank % 2}
        for rank in range(16)
    ] + [{"event": "worker_complete", "rank": rank} for rank in range(16)]

    report = build_pilot_audit(
        records,
        expected_uids={row["uid"] for row in records},
        failure_rows=[{"uid": "gqa:u", "exception": "resolved retry"}],
        progress=progress,
        epsilon_sha256="eps",
        minimum_jaccard=0.5,
        checksum_errors=[],
        slurm_jobs=[{"state": "COMPLETED"}],
    )

    assert report["passed"] is True
    assert report["checks"]["w2c_hard_path_exercised"] is True
    assert report["checks"]["w2c_soft_path_exercised"] is True
    assert report["checks"]["c2c_alignment_path_exercised"] is True
    assert report["checks"]["evaluator_compatible_target_policy_present"] is True


def test_full_audit_reconciles_every_source_route_and_positive_route_contract():
    manifest = [{
        "uid": "gqa:u",
        "dataset": "gqa",
        "source_positive_route_count": 1,
        "source_positive_routes": [{"source_binary_route_id": "gqa:u::r1"}],
    }]
    records = [{
        "uid": "gqa:u",
        "dataset": "gqa",
        "passed": True,
        "route_type": "W2C",
        "epsilon": 0.01,
        "execution_contract": {"contract_sha256": "one", "epsilon_sha256": "eps"},
        "source_positive_route_count": 1,
        "source_route_replay_valid_count": 1,
        "source_route_replay_failure_count": 0,
        "raw_conversions": [{
            "source_binary_route_id": "gqa:u::r1",
            "status": "converted",
            "positive_routes": [{
                "route": ["READ_OFF"],
                "evaluation": {"correct": True, "answer_alignment_margin": 1.0},
            }],
        }],
        "unique_valid_three_action_routes": [{
            "route": ["READ_OFF"],
            "evaluation": {"correct": True, "answer_alignment_margin": 1.0},
        }],
        "canonical_three_action_route": {
            "route": ["READ_OFF"],
            "evaluation": {"correct": True, "answer_alignment_margin": 1.0},
        },
    }]

    report = build_full_audit(
        records,
        manifest=manifest,
        failure_rows=[{"uid": "gqa:u", "exception": "resolved retry"}],
        checksum_errors=[],
        epsilon_sha256="eps",
        slurm_jobs=[
            {"state": "FAILED", "exit_code": "1:0", "elapsed_seconds": 3},
            {"state": "COMPLETED", "exit_code": "0:0", "elapsed_seconds": 7},
        ],
        progress_rows=[
            {
                "event": "worker_start",
                "rank": rank,
                "gpu_index": rank // 2,
                "replica_index": rank % 2,
                "work_assignment": "atomic_dynamic",
            }
            for rank in range(16)
        ],
        telemetry={"samples": 8, "gpu_indices": list(range(8))},
    )

    assert report["passed"] is True
    assert report["source_routes"] == 1
    assert report["checks"]["source_route_ids_reconcile_exactly"] is True
    assert report["checks"]["final_declared_slurm_job_completed"] is True
    assert report["checks"]["all_sixteen_dynamic_workers_observed"] is True


def test_final_views_keep_semantics_canonical_and_correct_training_rows():
    route = {
        "route": ["READ_OFF"],
        "route_key": "READ_OFF",
        "suppression_cost": 1,
        "evaluation": {"correct": True, "answer_alignment_margin": 1.0},
        "label_semantics": "W2C_HARD_CORRECTIVE",
        "source_binary_route_ids": ["u::r1"],
    }
    records = [{
        "uid": "gqa:u",
        "dataset": "gqa",
        "sample_id": "u",
        "image_id": "i",
        "image_group_id": "i",
        "source_split": "train",
        "route_type": "W2C",
        "epsilon": 0.01,
        "execution_contract": {"contract_sha256": "one", "epsilon_sha256": "eps"},
        "raw_conversions": [{
            "source_binary_route_id": "u::r1",
            "all_off_seed": True,
            "status": "converted",
            "positive_routes": [route],
            "corrective_partial_candidates": [],
        }],
        "unique_valid_three_action_routes": [route],
        "canonical_three_action_route": route,
    }]

    views, summary = build_final_views(records, route_cap=50, diversity_seed=7)

    assert summary["passed"] is True
    assert len(views["source_conversion_view"]) == 1
    assert len(views["w2c_corrective_training_view"]) == 1
    assert views["combined_training_manifest"][0]["route_key"] == "READ_OFF"
    assert views["combined_training_manifest"][0]["all_off_seed"] is True
    assert views["canonical_routes"][0]["all_off_source_binary_route_ids"] == ["u::r1"]


def test_analysis_counts_soft_effects_missed_by_correctness_only_and_joint_failures():
    records = [{
        "uid": "gqa:u",
        "dataset": "gqa",
        "route_type": "W2C",
        "source_positive_route_count": 1,
        "source_route_replay_valid_count": 1,
        "source_route_replay_failure_count": 0,
        "unique_valid_three_action_routes": [{"route": ["READ_OFF"]}],
        "route_evaluation_cache": {"cache_hits": 4, "cache_misses": 6},
        "raw_conversions": [{
            "status": "converted",
            "screening": {"positions": [{
                "layer": 3,
                "classification": "SOFT_ALIGNMENT_HELPFUL",
                "both_off_minus_full": 0.2,
            }]},
            "decomposition": [{
                "layer": 3,
                "action_classification": "READ_SUPPRESSION",
                "actions": {
                    "READ_OFF": {"delta_vs_full_reference": 0.3, "evaluation": {"correct": True}},
                    "WRITE_OFF": {"delta_vs_full_reference": 0.1, "evaluation": {"correct": True}},
                    "BOTH_OFF": {"delta_vs_full_reference": 0.2, "evaluation": {"correct": True}},
                },
            }],
            "independent_composition": {"independent_composition_failure": True},
            "positive_routes": [{"route": ["READ_OFF"], "evaluation": {"correct": True}}],
            "execution_efficiency": {
                "candidate_positions": 1,
                "decomposition_new_cache_misses": 2,
                "theoretical_four_state_evaluations_avoided": 2,
            },
        }],
    }]

    analysis = analyze_records(records)
    combined = analysis["combined"]

    assert combined["w2c"]["soft_alignment_helpful_positions"] == 1
    assert combined["w2c"]["useful_positions_missed_by_correctness_only"] == 1
    assert combined["w2c"]["correctness_only_miss_fraction_among_useful_screening_positions"] == 1.0
    assert combined["w2c"]["strongest_local_gain_distribution"]["median"] == pytest.approx(0.3)
    assert combined["w2c"]["useful_operation_layers_per_sample"]["median"] == 1
    assert combined["joint_validation"]["independent_composition_failures"] == 1
    assert combined["w2c_vs_c2c"]["c2c_strongest_local_gain_weaker_by_median"] is None


def test_checksum_ledger_binds_artifacts_and_record_sidecars(tmp_path: Path):
    audit = tmp_path / "audit.json"
    artifact = tmp_path / "artifact.jsonl"
    record = tmp_path / "record.json"
    sidecar = tmp_path / "record.json.sha256"
    output = tmp_path / "ledger.json"
    audit.write_text('{"passed": true}\n', encoding="utf-8")
    artifact.write_text('{"route": ["READ_OFF"]}\n', encoding="utf-8")
    record.write_text('{"passed": true}\n', encoding="utf-8")
    record_digest = hashlib.sha256(record.read_bytes()).hexdigest()
    sidecar.write_text(f"{record_digest}  {record.name}\n", encoding="utf-8")

    ledger = build_checksum_ledger(
        [artifact, audit, artifact],
        record_sidecars=[sidecar],
        full_audit_path=audit,
        output_path=output,
    )

    assert ledger["artifact_count"] == 2
    assert ledger["full_record_count"] == 1
    assert len(ledger["full_record_sidecar_manifest_sha256"]) == 64


def test_compute_estimate_projects_stratified_pilot_rates():
    pilot_manifest = [
        {"uid": "a", "source_current_all_on_status": "wrong", "estimated_conversion_cost": 10},
        {"uid": "b", "source_current_all_on_status": "correct", "estimated_conversion_cost": 20},
    ]
    pilot_records = [
        {"uid": "a", "runtime": {"elapsed_seconds": 10.0}, "route_evaluation_cache": {"cache_misses": 5}},
        {"uid": "b", "runtime": {"elapsed_seconds": 10.0}, "route_evaluation_cache": {"cache_misses": 15}},
    ]
    full_manifest = [
        {"uid": "x", "source_current_all_on_status": "wrong", "estimated_conversion_cost": 100},
        {"uid": "y", "source_current_all_on_status": "correct", "estimated_conversion_cost": 200},
    ]

    report = build_compute_estimate(
        pilot_records,
        pilot_manifest=pilot_manifest,
        full_manifest=full_manifest,
        workers=2,
        gpus=1,
    )

    assert report["pilot_observed_unique_route_evaluations"] == 20
    assert report["estimates"]["central"]["worker_hours"] == pytest.approx(200 / 3600)
    assert report["estimates"]["central"]["ideal_wall_hours"] == pytest.approx(100 / 3600)
