from __future__ import annotations

import pytest

from dense_failure_stage2.full_benchmark_eval import (
    benchmark_family,
    feature_positions_from_masks,
    method_decision,
    paired_bootstrap,
    stage1_summary,
    stage2_summary,
    summarize_pairs,
    transition,
    validate_complete_rows,
)


def test_feature_positions_allow_trailing_visual_span():
    instruction, visual = feature_positions_from_masks(
        instruction_mask=[False, True, True, False, False, True, True, False, False],
        multimodal_token_types=[0, 0, 0, 1, 1, 0, 0, 1, 1],
    )
    assert instruction == (1, 2, 5, 6)
    assert visual == (3, 4, 7, 8)


def _row(uid, dense, routed, *, triggered=False, layer=None, actions=()):
    counts = {name: list(actions).count(name) for name in ("FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE")}
    nonfull = [index for index, action in enumerate(actions) if action != "FULL"]
    return {
        "uid": uid,
        "contract_sha256": "contract",
        "dense_correct": dense,
        "routed_correct": routed,
        "transition": transition(dense, routed),
        "triggered": triggered,
        "trigger_layer": layer,
        "post_trigger_action_counts": counts,
        "non_full_count": len(nonfull),
        "any_non_full": bool(nonfull),
        "first_non_full_layer": None if not nonfull else (layer or 0) + nonfull[0],
        "trigger_to_first_non_full_delay": None if not nonfull else nonfull[0],
    }


def test_family_mapping_is_closed():
    assert benchmark_family("chartqa") == "chartqa"
    assert benchmark_family("textvqa") == "textvqa"
    assert benchmark_family("mmmu_pro_vision_test") == "mmmu_pro"
    assert benchmark_family("pope_random") == "pope"
    with pytest.raises(ValueError):
        benchmark_family("gqa")


def test_pair_summary_counts_net_and_rates():
    rows = [
        _row("cc", True, True),
        _row("cw", True, False),
        _row("wc", False, True),
        _row("ww", False, False),
        _row("wc2", False, True),
    ]
    result = summarize_pairs(rows)
    assert result["dense_correct"] == 2
    assert result["routed_correct"] == 3
    assert result["net_correction"] == 1
    assert result["delta_accuracy"] == pytest.approx(0.2)
    assert result["w_to_c_rate_among_dense_wrong"] == pytest.approx(2 / 3)
    assert result["c_to_c_preservation"] == pytest.approx(0.5)


def test_bootstrap_is_deterministic_and_paired():
    rows = [_row(str(i), i % 2 == 0, i % 3 == 0) for i in range(20)]
    first = paired_bootstrap(rows, draws=200, seed=17)
    second = paired_bootstrap(rows, draws=200, seed=17)
    assert first == second
    delta = next(row for row in first if row["metric"] == "delta_accuracy")
    assert delta["estimate"] == summarize_pairs(rows)["delta_accuracy"]
    assert delta["ci_low"] <= delta["estimate"] <= delta["ci_high"]


def test_stage_summaries_use_triggered_subset():
    rows = [
        _row("a", True, True),
        _row("b", False, True, triggered=True, layer=4, actions=("FULL", "READ_ONLY")),
        _row("c", True, False, triggered=True, layer=20, actions=("FULL", "FULL")),
    ]
    gate = stage1_summary(rows)
    router = stage2_summary(rows)
    assert gate["triggered"] == 2
    assert gate["p_trigger_given_dense_w"] == 1.0
    assert gate["early_triggers"] == 1 and gate["late_triggers"] == 1
    assert router["triggered_any_non_full"] == 1
    assert router["read_only_count"] == 1
    assert router["w_to_c"] == 1 and router["c_to_w"] == 1


def test_global_completeness_rejects_missing_duplicate_and_wrong_contract():
    rows = [_row("a", True, True), _row("b", False, False)]
    validate_complete_rows(["a", "b"], rows, contract_sha256="contract")
    with pytest.raises(ValueError, match="coverage differs"):
        validate_complete_rows(["a", "b"], rows[:1], contract_sha256="contract")
    duplicate = rows + [_row("a", True, True)]
    with pytest.raises(ValueError, match="coverage differs"):
        validate_complete_rows(["a", "b"], duplicate, contract_sha256="contract")
    rows[0]["contract_sha256"] = "other"
    with pytest.raises(ValueError, match="another frozen contract"):
        validate_complete_rows(["a", "b"], rows, contract_sha256="contract")


def test_method_decision_rules_are_prospective():
    values = {name: {"net_correction": 0} for name in ("chartqa", "textvqa", "mmmu_pro", "pope")}
    values["overall"] = {"net_correction": 2}
    values["chartqa"]["net_correction"] = 2
    assert method_decision(values) == "B"
    values["textvqa"]["net_correction"] = 1
    assert method_decision(values) == "A"
    values["overall"]["net_correction"] = -1
    assert method_decision(values) == "D"
    values["overall"]["net_correction"] = 0
    assert method_decision(values) == "C"
