from __future__ import annotations

import unittest

from baseline_relative_visual_router.oracle_pareto import (
    align_policy_rows,
    oracle_select,
)


def row(uid: str, baseline: bool, routed: bool, budget: int, benchmark: str = "gqa"):
    return {
        "uid": uid,
        "benchmark": benchmark,
        "baseline_correct": baseline,
        "router_correct": routed,
        "selected_num_visual_on_layers": budget,
        "selected_mask_key": "1" * budget + "0" * (28 - budget),
    }


class OracleSelectionTest(unittest.TestCase):
    def test_lexicographic_oracle_never_trades_correctness_for_budget(self) -> None:
        baseline = {"correct": True, "budget": 28, "policy": "all_on"}
        sparse_wrong = {"correct": False, "budget": 4, "policy": "sparse"}
        self.assertEqual(oracle_select([baseline, sparse_wrong]), baseline)

    def test_oracle_uses_cheapest_correct_route(self) -> None:
        candidates = [
            {"correct": True, "budget": 28, "policy": "all_on"},
            {"correct": True, "budget": 17, "policy": "policy_a"},
            {"correct": True, "budget": 11, "policy": "policy_b"},
        ]
        self.assertEqual(oracle_select(candidates)["policy"], "policy_b")

    def test_oracle_rescues_with_sparse_route(self) -> None:
        candidates = [
            {"correct": False, "budget": 28, "policy": "all_on"},
            {"correct": True, "budget": 19, "policy": "sparse"},
        ]
        self.assertEqual(oracle_select(candidates)["policy"], "sparse")

    def test_all_wrong_uses_cheapest_route_without_changing_accuracy(self) -> None:
        candidates = [
            {"correct": False, "budget": 28, "policy": "all_on"},
            {"correct": False, "budget": 7, "policy": "sparse"},
        ]
        self.assertEqual(oracle_select(candidates)["budget"], 7)
        self.assertFalse(oracle_select(candidates)["correct"])

    def test_alignment_rejects_baseline_mismatch(self) -> None:
        policies = {
            "a": [row("u1", True, True, 20)],
            "b": [row("u1", False, True, 18)],
        }
        with self.assertRaisesRegex(ValueError, "baseline correctness mismatch"):
            align_policy_rows(policies)

    def test_alignment_rejects_uid_set_mismatch(self) -> None:
        policies = {
            "a": [row("u1", True, True, 20)],
            "b": [row("u2", True, True, 18)],
        }
        with self.assertRaisesRegex(ValueError, "UID set mismatch"):
            align_policy_rows(policies)


if __name__ == "__main__":
    unittest.main()
