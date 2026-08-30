from __future__ import annotations

import pytest

from experiments.analyze_four_action_generalization_diagnostics import atomic_csv
from four_action_policy.generalization_diagnostics import (
    binary_metrics,
    bit_requirement_labels,
    build_label_incompleteness_subset,
    build_matched_state_manifest,
    candidate_suffix_routes,
    compact_knn_label_consistency,
    first_deviation_bucket,
    knn_label_consistency,
    layer_only_binary_scores,
    multiclass_metrics,
    summarize_label_incompleteness_audit,
)


def _row(uid: str, actions: list[str]) -> dict:
    return {
        "uid": uid,
        "split": "train",
        "dataset": "gqa",
        "route_type": "W2C",
        "valid_routes": [{"actions": actions, "route_key": f"route:{uid}"}],
    }


def test_portable_csv_writer_uses_lf_line_endings(tmp_path) -> None:
    path = tmp_path / "table.csv"

    atomic_csv(path, [{"metric": "when", "value": 0.5}])

    assert b"\r\n" not in path.read_bytes()


def test_state_manifest_exactly_matches_full_unique_negatives_by_cell() -> None:
    rows = [
        _row("early-a", ["IGNORE", "FULL", "FULL"]),
        _row("early-b", ["READ_ONLY", "FULL", "FULL"]),
        _row("late-a", ["FULL", "IGNORE", "FULL"]),
        _row("late-b", ["FULL", "WRITE_ONLY", "FULL"]),
    ]
    boundaries = []
    for row in rows:
        boundary = 1 if row["uid"].startswith("late") else 0
        boundaries.append(
            {
                "uid": row["uid"],
                "dataset": "gqa",
                "boundary_layer": boundary,
                "all_full_prefix_length": boundary,
                "valid_nonfull_actions": [row["valid_routes"][0]["actions"][boundary]],
                "boundary_route_indices": [0],
                "teacher_route_index": 0,
                "singleton": True,
            }
        )

    records, audit = build_matched_state_manifest(rows, boundaries, seed=17)

    assert audit["passed"] is True
    assert audit["positive_states"] == audit["negative_states"] == 4
    assert len(records) == 8
    for pair_id in {row["pair_id"] for row in records}:
        pair = [row for row in records if row["pair_id"] == pair_id]
        assert {row["when_label"] for row in pair} == {0, 1}
        assert len({row["split"] for row in pair}) == 1
        assert len({row["dataset"] for row in pair}) == 1
        assert len({row["target_layer"] for row in pair}) == 1
        negative = next(row for row in pair if row["when_label"] == 0)
        assert negative["valid_actions"] == ["FULL"]
    by_id = {row["state_id"]: row for row in records}
    for row in records:
        partner = by_id[row["shuffle_partner_state_id"]]
        assert partner["state_id"] != row["state_id"]
        assert (partner["split"], partner["dataset"], partner["target_layer"]) == (
            row["split"],
            row["dataset"],
            row["target_layer"],
        )


def test_bit_requirement_labels_exclude_set_valued_bit_ambiguity() -> None:
    assert bit_requirement_labels(["FULL"]) == {"read_off": 0, "write_off": 0}
    assert bit_requirement_labels(["READ_ONLY"]) == {
        "read_off": 0,
        "write_off": 1,
    }
    assert bit_requirement_labels(["WRITE_ONLY", "IGNORE"]) == {
        "read_off": 1,
        "write_off": None,
    }
    assert bit_requirement_labels(["READ_ONLY", "WRITE_ONLY"]) == {
        "read_off": None,
        "write_off": None,
    }


def test_binary_metrics_report_rank_and_threshold_behavior() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])

    assert metrics["auroc"] == pytest.approx(0.75)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["false_positive_rate"] == pytest.approx(0.5)
    assert metrics["false_negative_rate"] == pytest.approx(0.5)
    assert 0.0 <= metrics["auprc"] <= 1.0


def test_first_deviation_bucket_distinguishes_near_late_and_never() -> None:
    assert first_deviation_bucket(7, 7) == "exact"
    assert first_deviation_bucket(6, 7) == "within_1_early"
    assert first_deviation_bucket(9, 7) == "within_2_late"
    assert first_deviation_bucket(3, 7) == "too_early"
    assert first_deviation_bucket(12, 7) == "too_late"
    assert first_deviation_bucket(None, 7) == "never"


def test_label_audit_subset_is_capped_per_architecture_action() -> None:
    states = [
        {
            "state_id": f"s{index}",
            "uid": f"u{index}",
            "split": "validation",
            "dataset": "gqa",
            "state_kind": "mandatory_deviation",
            "target_layer": 1,
            "valid_actions": ["IGNORE"],
            "prefix_actions": ["FULL"],
            "boundary_route_indices": [0],
        }
        for index in range(5)
    ]
    outputs = {
        "polar": {
            row["state_id"]: {
                "predicted_action": "READ_ONLY",
                "action_probabilities": [0.1, 0.7, 0.1, 0.1],
            }
            for row in states
        }
    }

    subset = build_label_incompleteness_subset(
        states, outputs, cap_per_architecture_action=3, seed=19
    )

    assert len(subset) == 3
    assert all(row["architecture"] == "polar" for row in subset)
    assert all(row["predicted_action"] == "READ_ONLY" for row in subset)


def test_candidate_suffix_routes_replace_only_the_audited_boundary_action() -> None:
    manifest_row = {
        "uid": "u",
        "valid_routes": [
            {"actions": ["FULL", "IGNORE", "FULL"]},
            {"actions": ["FULL", "WRITE_ONLY", "READ_ONLY"]},
        ],
    }
    state = {
        "uid": "u",
        "target_layer": 1,
        "prefix_actions": ["FULL"],
        "valid_actions": ["IGNORE", "WRITE_ONLY"],
        "boundary_route_indices": [0, 1],
    }

    routes = candidate_suffix_routes(
        manifest_row,
        state,
        predicted_action="READ_ONLY",
        max_suffixes=8,
        seed=23,
    )

    assert {tuple(row["actions"]) for row in routes} == {
        ("FULL", "READ_ONLY", "FULL"),
        ("FULL", "READ_ONLY", "READ_ONLY"),
    }
    assert all(row["predicted_action_cached_invalid"] for row in routes)


def test_label_incompleteness_audit_summarizes_states_not_route_attempts() -> None:
    subset = [
        {
            "state_id": "s0",
            "uid": "u0",
            "architecture": "polar",
            "predicted_action": "IGNORE",
            "candidate_routes": [{"candidate_index": 0}, {"candidate_index": 1}],
        },
        {
            "state_id": "s1",
            "uid": "u1",
            "architecture": "online",
            "predicted_action": "READ_ONLY",
            "candidate_routes": [{"candidate_index": 0}],
        },
    ]
    executions = [
        {"architecture": "polar", "state_id": "s0", "candidate_index": 0, "correct": False},
        {"architecture": "polar", "state_id": "s0", "candidate_index": 1, "correct": True},
        {"architecture": "online", "state_id": "s1", "candidate_index": 0, "correct": False},
    ]

    summary = summarize_label_incompleteness_audit(subset, executions)

    assert summary["states"] == 2
    assert summary["candidate_executions"] == 3
    assert summary["cached_invalid_but_execution_correct_states"] == 1
    assert summary["cached_invalid_but_execution_correct_fraction"] == pytest.approx(0.5)
    assert summary["no_bounded_rescue_states"] == 1
    assert summary["by_architecture_action"]["polar:IGNORE"]["fraction"] == pytest.approx(1.0)
    assert summary["by_architecture_action"]["online:READ_ONLY"]["fraction"] == pytest.approx(0.0)
    polar = next(
        row for row in summary["state_results"] if row["architecture"] == "polar"
    )
    assert polar["successful_candidate_indices"] == [1]


def test_label_incompleteness_audit_rejects_missing_candidate_execution() -> None:
    subset = [
        {
            "state_id": "s0",
            "uid": "u0",
            "architecture": "polar",
            "predicted_action": "IGNORE",
            "candidate_routes": [{"candidate_index": 0}, {"candidate_index": 1}],
        }
    ]

    with pytest.raises(ValueError, match="exactly cover"):
        summarize_label_incompleteness_audit(
            subset,
            [
                {
                    "architecture": "polar",
                    "state_id": "s0",
                    "candidate_index": 0,
                    "correct": False,
                }
            ],
        )


def test_label_incompleteness_audit_allows_same_state_for_both_architectures() -> None:
    subset = [
        {
            "state_id": "s0",
            "uid": "u0",
            "architecture": architecture,
            "predicted_action": action,
            "candidate_routes": [{"candidate_index": 0}],
        }
        for architecture, action in (("polar", "IGNORE"), ("online", "READ_ONLY"))
    ]
    executions = [
        {
            "architecture": architecture,
            "state_id": "s0",
            "candidate_index": 0,
            "correct": architecture == "polar",
        }
        for architecture in ("polar", "online")
    ]

    summary = summarize_label_incompleteness_audit(subset, executions)

    assert summary["states"] == 2
    assert summary["cached_invalid_but_execution_correct_states"] == 1


def test_layer_only_scores_are_fitted_on_train_layers_only() -> None:
    train = [
        {"target_layer": 0, "label": 1},
        {"target_layer": 0, "label": 1},
        {"target_layer": 1, "label": 0},
    ]
    validation = [{"target_layer": 0}, {"target_layer": 1}, {"target_layer": 2}]

    scores = layer_only_binary_scores(
        train, validation, label_key="label", alpha=0.5, num_layers=3
    )

    assert scores[0] == pytest.approx(2.5 / 3.0)
    assert scores[1] == pytest.approx(0.5 / 2.0)
    assert scores[2] == pytest.approx(0.5)


def test_multiclass_metrics_report_confusion_and_macro_f1() -> None:
    metrics = multiclass_metrics(
        ["FULL", "IGNORE", "READ_ONLY", "WRITE_ONLY"],
        ["FULL", "READ_ONLY", "READ_ONLY", "FULL"],
        classes=["IGNORE", "READ_ONLY", "WRITE_ONLY", "FULL"],
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["confusion"]["FULL"]["FULL"] == 1
    assert metrics["confusion"]["IGNORE"]["READ_ONLY"] == 1
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_knn_uses_prospective_pool_fallback_and_reports_purity() -> None:
    train_features = [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]]
    validation_features = [[0.95, 0.05], [-0.95, -0.05]]
    train_labels = [1, 1, 0, 0]
    validation_labels = [1, 0]
    train_metadata = [
        {"dataset": "gqa", "target_layer": 1, "depth_bin": 0},
        {"dataset": "chartqa", "target_layer": 1, "depth_bin": 0},
        {"dataset": "gqa", "target_layer": 1, "depth_bin": 0},
        {"dataset": "chartqa", "target_layer": 1, "depth_bin": 0},
    ]
    validation_metadata = [
        {"dataset": "textvqa", "target_layer": 1, "depth_bin": 0},
        {"dataset": "textvqa", "target_layer": 1, "depth_bin": 0},
    ]

    result = knn_label_consistency(
        train_features,
        train_labels,
        train_metadata,
        validation_features,
        validation_labels,
        validation_metadata,
        k_values=[1, 3],
    )

    assert result["by_k"]["1"]["majority_accuracy"] == pytest.approx(1.0)
    assert result["by_k"]["1"]["mean_label_purity"] == pytest.approx(1.0)
    assert result["fallback_counts"]["same_layer"] == 2


def test_compact_knn_payload_removes_raw_neighbor_pairs() -> None:
    payload = {
        "contract": {"knn_k": [5]},
        "entropy": {"action_set": {"H_label_given_layer": 1.0}},
        "representations": {
            "online": {
                "when": {
                    "by_k": {"5": {"mean_label_purity": 0.6}},
                    "fallback_counts": {"same_layer": 2},
                    "neighbor_pairs_at_max_k": [
                        {"distance": 0.1, "agreement": 1},
                        {"distance": 0.2, "agreement": 0},
                    ],
                }
            }
        },
    }

    compact = compact_knn_label_consistency(payload)

    task = compact["representations"]["online"]["when"]
    assert task["neighbor_pairs_at_max_k_records"] == 2
    assert "neighbor_pairs_at_max_k" not in task
    assert "neighbor_pairs_at_max_k" in payload["representations"]["online"]["when"]
