from __future__ import annotations

from collections import Counter

import pytest
import torch

from experiments.prepare_persistent_corrective_supervision import select_matched_subset
from experiments.summarize_persistent_corrective_supervision import (
    metric_values,
    runtime_cohort_sensitivity,
)
from four_action_policy.multimodal import make_persistent_boundary_collator
from four_action_policy.persistent import (
    paired_bootstrap_rate_difference,
    matched_epoch_indices,
    persistent_boundary_loss,
    select_behavioral_checkpoint,
)


def _synthetic_rows() -> tuple[list[dict], list[dict]]:
    manifest = []
    boundaries = []
    actions = (["IGNORE"], ["READ_ONLY"], ["WRITE_ONLY"], ["IGNORE", "READ_ONLY"])
    for split in ("train", "validation"):
        for dataset in ("gqa", "chartqa", "textvqa"):
            for route_type in ("W2C", "C2C"):
                for index in range(12):
                    uid = f"{split}:{dataset}:{route_type}:{index}"
                    manifest.append(
                        {
                            "uid": uid,
                            "split": split,
                            "split_group": f"{split}:{dataset}:{route_type}:group:{index}",
                            "dataset": dataset,
                            "route_type": route_type,
                        }
                    )
                    if route_type == "W2C":
                        valid = list(actions[index % len(actions)])
                        boundaries.append(
                            {
                                "uid": uid,
                                "dataset": dataset,
                                "boundary_layer": index % 12,
                                "valid_nonfull_actions": valid,
                                "singleton": len(valid) == 1,
                            }
                        )
    return manifest, boundaries


def test_matched_subset_is_fixed_balanced_and_boundary_diverse() -> None:
    manifest, boundaries = _synthetic_rows()
    first = select_matched_subset(
        manifest,
        boundaries,
        train_per_type=24,
        validation_per_type=24,
        seed=17,
    )
    repeated = select_matched_subset(
        manifest,
        boundaries,
        train_per_type=24,
        validation_per_type=24,
        seed=17,
    )
    assert first == repeated
    selected = first["selected_uids"]
    assert len(selected) == len(set(selected)) == 96

    by_uid = {row["uid"]: row for row in manifest}
    counts = Counter(
        (by_uid[uid]["split"], by_uid[uid]["route_type"]) for uid in selected
    )
    assert counts == {
        ("train", "W2C"): 24,
        ("train", "C2C"): 24,
        ("validation", "W2C"): 24,
        ("validation", "C2C"): 24,
    }
    for split in ("train", "validation"):
        for route_type in ("W2C", "C2C"):
            dataset_counts = Counter(
                by_uid[uid]["dataset"]
                for uid in selected
                if by_uid[uid]["split"] == split
                and by_uid[uid]["route_type"] == route_type
            )
            assert max(dataset_counts.values()) - min(dataset_counts.values()) <= 1

    boundary_by_uid = {row["uid"]: row for row in boundaries}
    chosen_boundaries = [boundary_by_uid[uid] for uid in selected if uid in boundary_by_uid]
    covered_actions = {
        action for row in chosen_boundaries for action in row["valid_nonfull_actions"]
    }
    assert covered_actions == {"IGNORE", "READ_ONLY", "WRITE_ONLY"}
    assert {row["singleton"] for row in chosen_boundaries} == {False, True}
    assert {int(row["boundary_layer"]) // 7 for row in chosen_boundaries} >= {0, 1}


def test_matched_epoch_order_visits_every_sample_once_and_alternates_types() -> None:
    rows = [
        {"uid": f"w{index}", "route_type": "W2C"} for index in range(8)
    ] + [{"uid": f"c{index}", "route_type": "C2C"} for index in range(8)]
    first = matched_epoch_indices(rows, seed=23, epoch=2)
    repeated = matched_epoch_indices(rows, seed=23, epoch=2)
    changed = matched_epoch_indices(rows, seed=23, epoch=3)
    assert first == repeated
    assert first != changed
    assert sorted(first) == list(range(16))
    assert [rows[index]["route_type"] for index in first] == [
        value for _ in range(8) for value in ("W2C", "C2C")
    ]


def test_persistent_boundary_loss_uses_the_valid_set_at_each_selected_layer() -> None:
    logits = torch.zeros(3, 4, 4, requires_grad=True)
    layers = torch.tensor([1, -1, 3])
    valid_actions = torch.tensor(
        [
            [True, True, False, False],
            [False, False, False, False],
            [False, False, True, False],
        ]
    )
    present = torch.tensor([True, False, True])
    loss = persistent_boundary_loss(
        logits,
        boundary_layers=layers,
        valid_actions=valid_actions,
        present=present,
    )
    assert loss.item() == pytest.approx((torch.log(torch.tensor(2.0)) + torch.log(torch.tensor(4.0))).item() / 2)
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[1]).item() == 0


def test_persistent_boundary_loss_rejects_full_as_a_valid_boundary_action() -> None:
    with pytest.raises(ValueError, match="FULL"):
        persistent_boundary_loss(
            torch.zeros(1, 2, 4),
            boundary_layers=torch.tensor([0]),
            valid_actions=torch.tensor([[False, False, False, True]]),
            present=torch.tensor([True]),
        )


def test_persistent_collator_attaches_only_w2c_boundary_targets() -> None:
    rows = [
        {"uid": "w", "route_type": "W2C"},
        {"uid": "c", "route_type": "C2C"},
    ]
    boundaries = {
        "w": {
            "uid": "w",
            "boundary_layer": 2,
            "valid_nonfull_actions": ["READ_ONLY", "WRITE_ONLY"],
        }
    }
    collate = make_persistent_boundary_collator(
        lambda values: {"uids": [row["uid"] for row in values]}, boundaries
    )
    batch = collate(rows)
    assert batch["boundary_present"].tolist() == [True, False]
    assert batch["boundary_layers"].tolist() == [2, -1]
    assert batch["boundary_valid_actions"].tolist() == [
        [False, True, True, False],
        [False, False, False, False],
    ]


def test_behavioral_selection_applies_frozen_c2c_constraint_before_rescue() -> None:
    history = [
        {"epoch": 1, "execution": {"w2c_rescue_rate": 0.10, "c2c_preservation_rate": 0.96, "overall_routed_accuracy": 0.53, "c2c_regressions": 5, "mean_action_layers": {"FULL": 25.0}}},
        {"epoch": 2, "execution": {"w2c_rescue_rate": 0.30, "c2c_preservation_rate": 0.94, "overall_routed_accuracy": 0.62, "c2c_regressions": 8, "mean_action_layers": {"FULL": 20.0}}},
        {"epoch": 3, "execution": {"w2c_rescue_rate": 0.15, "c2c_preservation_rate": 0.97, "overall_routed_accuracy": 0.56, "c2c_regressions": 4, "mean_action_layers": {"FULL": 24.0}}},
    ]
    selection = select_behavioral_checkpoint(history, c2c_threshold=0.95)
    assert selection["selected_epoch"] == 3
    assert selection["eligible_epochs"] == [1, 3]
    assert 2 in selection["pareto_frontier_epochs"]


def test_behavioral_selection_reports_frontier_when_constraint_has_no_checkpoint() -> None:
    history = [
        {"epoch": 1, "execution": {"w2c_rescue_rate": 0.10, "c2c_preservation_rate": 0.90, "overall_routed_accuracy": 0.50, "c2c_regressions": 13, "mean_action_layers": {"FULL": 25.0}}},
        {"epoch": 2, "execution": {"w2c_rescue_rate": 0.20, "c2c_preservation_rate": 0.89, "overall_routed_accuracy": 0.55, "c2c_regressions": 14, "mean_action_layers": {"FULL": 20.0}}},
    ]
    selection = select_behavioral_checkpoint(history, c2c_threshold=0.95)
    assert selection["selected_epoch"] is None
    assert selection["eligible_epochs"] == []
    assert selection["pareto_frontier_epochs"] == [1, 2]


def test_paired_bootstrap_difference_is_deterministic_and_directional() -> None:
    left = [False, False, True, True]
    right = [False, True, True, True]
    first = paired_bootstrap_rate_difference(left, right, draws=1000, seed=9)
    repeated = paired_bootstrap_rate_difference(left, right, draws=1000, seed=9)
    assert first == repeated
    assert first["right_minus_left"] == pytest.approx(0.25)
    assert first["ci_low"] <= first["right_minus_left"] <= first["ci_high"]


def test_matched_metric_values_report_action_and_rollout_fractions() -> None:
    training = [
        {
            "epoch": 1,
            "train": {"boundary_valid_action_at_1": 0.75},
        }
    ]
    execution = [
        {
            "epoch": 1,
            "execution": {
                "records": 2,
                "action_counts": {
                    "IGNORE": 4,
                    "READ_ONLY": 6,
                    "WRITE_ONLY": 2,
                    "FULL": 44,
                },
                "w2c_rescue_rate": 0.25,
                "c2c_preservation_rate": 1.0,
                "w2c_rescues": 1,
                "c2c_regressions": 0,
            },
            "boundary": {
                "valid_action_at_1": 0.5,
                "nonfull_recall": 0.8,
                "free_rollout": {
                    "exact_boundary_fraction": 0.25,
                    "within_1_fraction": 0.5,
                    "within_2_fraction": 0.625,
                    "early_fraction": 0.375,
                    "late_or_no_deviation_fraction": 0.375,
                    "left_all_full_fraction": 0.75,
                },
            },
        }
    ]
    values = metric_values(
        epoch=1, training_history=training, execution_history=execution
    )
    assert values is not None
    assert values["no_deviation_fraction"] == pytest.approx(0.25)
    assert values["within_1_first_deviation"] == pytest.approx(0.5)
    assert values["within_2_first_deviation"] == pytest.approx(0.625)
    assert values["teacher_forced_minus_free_rollout_leave_full_gap"] == pytest.approx(
        0.05
    )
    assert sum(values[f"{action}_fraction"] for action in ("ignore", "read_only", "write_only", "full")) == pytest.approx(1.0)


def test_runtime_cohort_sensitivity_excludes_drift_without_reopening_subset() -> None:
    def output(epoch: int, uid: str, route_type: str, correct: bool) -> dict:
        return {
            "epoch": epoch,
            "uid": uid,
            "route_type": route_type,
            "correct": correct,
            "actions": ["FULL", "FULL"] if epoch == 1 else ["IGNORE", "FULL"],
            "prediction": "yes" if correct else "no",
        }

    polar_outputs = [
        output(1, "w", "W2C", False),
        output(1, "c-good", "C2C", True),
        output(1, "c-drift", "C2C", False),
        output(2, "w", "W2C", True),
        output(2, "c-good", "C2C", True),
        output(2, "c-drift", "C2C", False),
    ]
    online_outputs = [
        output(1, "w", "W2C", False),
        output(1, "c-good", "C2C", True),
        output(1, "c-drift", "C2C", False),
        output(2, "w", "W2C", True),
        output(2, "c-good", "C2C", True),
        output(2, "c-drift", "C2C", False),
    ]

    def history(epoch: int, rescue: float, preservation: float) -> dict:
        return {
            "epoch": epoch,
            "execution": {
                "w2c_rescue_rate": rescue,
                "c2c_preservation_rate": preservation,
                "overall_routed_accuracy": (rescue + 2 * preservation) / 3,
                "c2c_regressions": round(2 * (1 - preservation)),
                "mean_action_layers": {"FULL": 2.0 if epoch == 1 else 1.0},
            },
        }

    histories = [history(1, 0.0, 0.5), history(2, 1.0, 0.5)]
    sensitivity = runtime_cohort_sensitivity(
        polar_outputs=polar_outputs,
        online_outputs=online_outputs,
        polar_history=histories,
        online_history=histories,
        polar_selected_epoch=2,
        online_selected_epoch=2,
        c2c_threshold=0.95,
    )

    assert sensitivity["runtime_drift_uids"] == ["c-drift"]
    assert sensitivity["drift_records"][0]["frozen_expected_correct"] is True
    assert sensitivity["polar"]["selected_epoch"] == 2
    assert sensitivity["polar"]["c2c_correct"] == 1
    assert sensitivity["polar"]["c2c_records"] == 1
    assert sensitivity["decision_invariant"] is True
