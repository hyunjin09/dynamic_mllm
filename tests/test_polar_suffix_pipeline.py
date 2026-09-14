from __future__ import annotations

from collections import Counter

import pytest

from experiments.run_polar_suffix_program import (
    _method_summary,
    _segments,
    _weighted_batches,
)


def test_weight_balanced_batches_include_each_program_once():
    rows = [
        {"program_id": f"p{index}", "program_weight": weight}
        for index, weight in enumerate([0.31, 0.22, 0.17, 0.12, 0.09, 0.05, 0.03, 0.01])
    ]
    batches = _weighted_batches(rows, batch_size=2, seed=9)
    observed = Counter(row["program_id"] for batch in batches for row in batch)
    assert observed == Counter({row["program_id"]: 1 for row in rows})
    assert all(len(batch) <= 2 for batch in batches)
    masses = [sum(row["program_weight"] for row in batch) for batch in batches]
    assert max(masses) - min(masses) < 0.2


def test_method_summary_uses_dense_paired_transitions():
    rows = [
        {"dense_correct": True, "program_correct": True},
        {"dense_correct": True, "program_correct": False},
        {"dense_correct": False, "program_correct": True},
        {"dense_correct": False, "program_correct": False},
        {"dense_correct": False, "program_correct": True},
    ]
    result = _method_summary(rows, "program")
    assert result["w_to_c"] == 2
    assert result["c_to_w"] == 1
    assert result["net"] == 1
    assert result["method_accuracy"] == pytest.approx(3 / 5)


def test_non_full_segments_count_contiguous_interventions():
    assert _segments(["FULL", "IGNORE", "READ_ONLY", "FULL", "WRITE_ONLY"]) == 2
    assert _segments(["FULL", "FULL"]) == 0
