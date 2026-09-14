from __future__ import annotations

import numpy as np
import pytest

from dense_failure_stage1.operating_point_compatibility import (
    build_trigger_rows,
    compatibility_category,
    select_retained_operating_points,
    structural_route_status,
    transition_category,
)


@pytest.mark.parametrize(
    ("old", "new", "category", "delta"),
    [
        (3, 3, "SAME_TRIGGER", 0),
        (5, 2, "NEW_EARLIER", -3),
        (2, 5, "NEW_LATER", 3),
        (2, None, "OLD_TRIGGER_ONLY", None),
        (None, 2, "NEW_TRIGGER_ONLY", None),
        (None, None, "NEITHER_TRIGGER", None),
    ],
)
def test_transition_categories(old, new, category, delta) -> None:
    assert transition_category(old, new) == (category, delta)


def test_structural_compatibility_uses_first_required_intervention() -> None:
    assert structural_route_status(new_trigger_layer=4, first_non_full_layer=4) == "STRUCTURALLY_COMPATIBLE"
    assert structural_route_status(new_trigger_layer=2, first_non_full_layer=4) == "STRUCTURALLY_COMPATIBLE"
    assert structural_route_status(new_trigger_layer=5, first_non_full_layer=4) == "STRUCTURALLY_INCOMPATIBLE"
    assert structural_route_status(new_trigger_layer=None, first_non_full_layer=4) == "NOT_NEW_TRIGGERED"


def test_trigger_map_uses_strict_any_layer_rule() -> None:
    rows = [
        {"uid": "a", "dataset": "gqa", "source_regime": "historical", "current_dense_wrong": False, "image_group_id": "ga"},
        {"uid": "b", "dataset": "gqa", "source_regime": "historical", "current_dense_wrong": True, "image_group_id": "gb"},
    ]
    scores = np.zeros((2, 28), dtype=np.float64)
    scores[0, 2] = 0.5
    scores[1, 7] = 0.50001
    mapped = build_trigger_rows(rows, scores, [{"operating_point": "P98", "threshold": 0.5}])
    assert mapped[0]["triggered"] is False
    assert mapped[1]["first_trigger_layer"] == 7


def test_default_retention_requires_stability_and_material_recall_spacing() -> None:
    rows = [
        {"operating_point": "P98", "preservation_target": 0.98, "threshold": 0.97, "pooled_w_recall": 0.10, "fold_stable": True},
        {"operating_point": "P95", "preservation_target": 0.95, "threshold": 0.94, "pooled_w_recall": 0.20, "fold_stable": True},
        {"operating_point": "P90", "preservation_target": 0.90, "threshold": 0.88, "pooled_w_recall": 0.35, "fold_stable": True},
    ]
    retained, reason = select_retained_operating_points(
        rows,
        default_names=("P98", "P95", "P90"),
        minimum_recall_increment=0.05,
        maximum_points=3,
    )
    assert retained == ["P98", "P95", "P90"]
    assert reason.startswith("prospective_default")


def test_retention_falls_back_when_default_middle_is_unstable() -> None:
    rows = [
        {"operating_point": "P98", "preservation_target": 0.98, "threshold": 0.97, "pooled_w_recall": 0.10, "fold_stable": True},
        {"operating_point": "P97", "preservation_target": 0.97, "threshold": 0.96, "pooled_w_recall": 0.18, "fold_stable": True},
        {"operating_point": "P95", "preservation_target": 0.95, "threshold": 0.94, "pooled_w_recall": 0.24, "fold_stable": False},
        {"operating_point": "P90", "preservation_target": 0.90, "threshold": 0.88, "pooled_w_recall": 0.36, "fold_stable": True},
    ]
    retained, reason = select_retained_operating_points(
        rows,
        default_names=("P98", "P95", "P90"),
        minimum_recall_increment=0.05,
        maximum_points=3,
    )
    assert retained == ["P98", "P97", "P90"]
    assert reason.startswith("deterministic")


@pytest.mark.parametrize(
    ("single", "mcts", "prior", "expected"),
    [
        (True, True, "SINGLE_FIXABLE", "EXISTING_SINGLE_REUSABLE"),
        (False, True, "MCTS_ONLY_FIXABLE", "EXISTING_MCTS_REUSABLE"),
        (False, False, "SINGLE_FIXABLE", "EXISTING_LABEL_BUT_INCOMPATIBLE"),
        (False, False, "UNRESOLVED", "EXISTING_UNRESOLVED"),
        (False, False, None, "NEW_TRIGGER_NO_EXISTING_LABEL"),
    ],
)
def test_per_sample_compatibility_precedence(single, mcts, prior, expected) -> None:
    assert compatibility_category(
        has_replay_single=single,
        has_replay_mcts=mcts,
        prior_label=prior,
    ) == expected
