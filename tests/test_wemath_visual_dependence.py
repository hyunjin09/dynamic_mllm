from __future__ import annotations

import pytest

from experiments.analyze_wemath_visual_dependence import (
    classify_visual_regimes,
    decomposition_row,
    mean_decomposition,
)


def row(uid: str, *, full: bool, off: bool, minimum: int, difficulty: str = "base"):
    return {
        "uid": uid,
        "current_all_on_status": "correct" if full else "wrong",
        "all_off_correct": off,
        "raw_min_on": minimum if full or off or minimum > 0 else None,
        "raw_valid_routes": int(off or minimum > 0),
        "difficulty": difficulty,
        "difficulty_degree": 0 if difficulty == "base" else len(difficulty),
        "question_id": uid,
        "image_group_id": uid,
    }


def test_regime_classification_separates_zero_and_positive_vision():
    rows = [
        row("v0", full=True, off=True, minimum=0),
        row("vp", full=True, off=False, minimum=5),
        row("a0", full=False, off=True, minimum=0),
        row("ap", full=False, off=False, minimum=4),
        row("none", full=False, off=False, minimum=0),
    ]
    assert classify_visual_regimes(rows) == {
        "A+": 1,
        "A0": 1,
        "V+": 1,
        "V0": 1,
        "no_correction": 1,
    }


def test_v0_minimum_identity_is_enforced():
    rows = [row("bad", full=True, off=True, minimum=2)]
    with pytest.raises(RuntimeError, match="identity mismatch"):
        classify_visual_regimes(rows)


def test_decomposition_identity():
    result = decomposition_row(
        group_type="difficulty",
        group="base",
        values=[0, 0, 4, 8],
        vplus_values=[4, 8],
    )
    assert result["original_mean_min_on"] == 3.0
    assert result["v0_fraction"] == 0.5
    assert result["vplus_mean_min_positive_on"] == 6.0
    assert result["reconstructed_original_mean"] == 3.0
    assert result["reconstruction_abs_error"] == 0.0


def test_symmetric_components_sum_to_observed_difference():
    rows = []
    for index, minimum in enumerate((0, 4, 8)):
        rows.append(row(f"base-{index}", full=True, off=minimum == 0, minimum=minimum))
    for index, minimum in enumerate((0, 0, 6)):
        rows.append(row(f"x-{index}", full=True, off=minimum == 0, minimum=minimum, difficulty="x"))
    # Supply nonempty cells for the other frozen strata used by grouped_rows.
    for difficulty in ("y", "z", "xy", "xz", "yz", "xyz"):
        rows.append(row(difficulty, full=True, off=False, minimum=5, difficulty=difficulty))
    classify_visual_regimes(rows)
    output = mean_decomposition(rows)
    assert max(item["reconstruction_abs_error"] for item in output) < 1e-12
    assert max(item["component_sum_error"] for item in output) < 1e-12
