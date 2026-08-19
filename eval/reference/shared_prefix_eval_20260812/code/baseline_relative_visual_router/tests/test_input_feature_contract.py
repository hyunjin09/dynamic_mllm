from __future__ import annotations

import unittest

from baseline_relative_visual_router.input_features import align_manifest_policy_rows


def manifest(uid: str, benchmark: str = "gqa") -> dict:
    return {
        "uid": uid,
        "benchmark": benchmark,
        "metric_name": "exact_match",
        "correctness_threshold": 1.0,
    }


def policy(uid: str, benchmark: str = "gqa") -> dict:
    return {
        "uid": uid,
        "benchmark": benchmark,
        "metric_name": "exact_match",
        "correctness_threshold": 1.0,
        "baseline_correct": True,
        "router_correct": False,
        "selected_num_visual_on_layers": 12,
    }


class InputFeatureContractTest(unittest.TestCase):
    def test_alignment_preserves_manifest_order(self) -> None:
        rows = align_manifest_policy_rows(
            [manifest("b"), manifest("a")], [policy("a"), policy("b")]
        )
        self.assertEqual([row[0]["uid"] for row in rows], ["b", "a"])

    def test_alignment_rejects_missing_uid(self) -> None:
        with self.assertRaisesRegex(ValueError, "UID sets differ"):
            align_manifest_policy_rows([manifest("a")], [policy("b")])

    def test_alignment_rejects_benchmark_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "benchmark mismatch"):
            align_manifest_policy_rows([manifest("a")], [policy("a", "textvqa")])

    def test_alignment_rejects_metric_contract_mismatch(self) -> None:
        changed = policy("a")
        changed["correctness_threshold"] = 0.5
        with self.assertRaisesRegex(ValueError, "correctness_threshold mismatch"):
            align_manifest_policy_rows([manifest("a")], [changed])


if __name__ == "__main__":
    unittest.main()
