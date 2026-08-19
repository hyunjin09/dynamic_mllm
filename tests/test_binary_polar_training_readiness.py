"""Deterministic no-training tests for the P10 runner and evaluator contracts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from experiments.evaluate_binary_polar_internal import select_rows, summarize
from experiments.train_binary_polar import checkpoint_key, select_smoke_rows, validate_gate


class _Rows:
    def __init__(self, rows):
        self.rows = rows


def _row(uid: str, benchmark: str) -> dict:
    return {"uid": uid, "benchmark": benchmark}


def test_gate_validation_binds_checksum_and_pass_status():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "gate.json"
        path.write_text('{"passed": true}\n', encoding="utf-8")
        import hashlib

        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert validate_gate("test", {"path": str(path), "sha256": expected})["sha256"] == expected
        try:
            validate_gate("test", {"path": str(path), "sha256": "0" * 64})
        except RuntimeError as error:
            assert "checksum mismatch" in str(error)
        else:
            raise AssertionError("a changed gate must block training")


def test_smoke_selection_is_deterministic_and_dataset_balanced():
    rows = [_row(f"{dataset}:{index}", dataset) for dataset in ("gqa", "textvqa", "chartqa") for index in range(8)]
    first = select_smoke_rows(_Rows(rows), per_dataset=3, seed=17)
    second = select_smoke_rows(_Rows(list(reversed(rows))), per_dataset=3, seed=17)
    assert [row["uid"] for row in first] == [row["uid"] for row in second]
    assert {dataset: sum(row["benchmark"] == dataset for row in first) for dataset in ("gqa", "textvqa", "chartqa")} == {
        "gqa": 3,
        "textvqa": 3,
        "chartqa": 3,
    }


def test_execution_smoke_selection_is_deterministic_and_disjoint_by_shard_order():
    rows = [_row(f"{dataset}:{index}", dataset) for dataset in ("gqa", "textvqa", "chartqa") for index in range(8)]
    selected = select_rows(rows, per_dataset=4, seed=23)
    assert len(selected) == 12
    shard0 = {row["uid"] for index, row in enumerate(selected) if index % 2 == 0}
    shard1 = {row["uid"] for index, row in enumerate(selected) if index % 2 == 1}
    assert not shard0 & shard1
    assert len(shard0 | shard1) == len(selected)


def test_checkpoint_selection_rule_is_common_and_deterministic():
    def epoch(number, hit1, hit5, hamming, set_nll):
        return {
            "epoch": number,
            "validation": {
                "top1_valid_route_coverage": hit1,
                "topk_valid_route_coverage": hit5,
                "nearest_valid_hamming": hamming,
                "set_nll": set_nll,
            },
        }

    rows = [epoch(1, 0.2, 0.4, 5, 8), epoch(2, 0.3, 0.3, 6, 7), epoch(3, 0.3, 0.3, 5, 9)]
    assert max(rows, key=checkpoint_key)["epoch"] == 3


def test_execution_summary_counts_uncached_masks_by_observed_execution():
    rows = [
        {
            "predicted_correct": True,
            "baseline_correct": False,
            "num_visual_on_layers": 8,
            "selected_valid_set_size": 2,
            "raw_cached_valid_set_size": 3,
            "predicted_mask_in_selected_valid_set": False,
            "predicted_mask_in_raw_cached_valid_set": False,
            "mcts_has_valid_route": True,
        },
        {
            "predicted_correct": False,
            "baseline_correct": True,
            "num_visual_on_layers": 28,
            "selected_valid_set_size": 1,
            "raw_cached_valid_set_size": 1,
            "predicted_mask_in_selected_valid_set": True,
            "predicted_mask_in_raw_cached_valid_set": True,
            "mcts_has_valid_route": True,
        },
    ]
    result = summarize(rows)
    assert result["full_wrong_to_predicted_correct"] == 1
    assert result["full_correct_to_predicted_wrong"] == 1
    assert result["uncached_top1_records"] == 1
    assert result["uncached_top1_accuracy"] == 1.0


def run_all() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"passed {len(tests)} P10 readiness tests")


if __name__ == "__main__":
    run_all()
