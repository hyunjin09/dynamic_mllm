import unittest

from tools.audit_binary_router_split_design import singleton_heldout_sufficiency


class SplitDesignAuditTest(unittest.TestCase):
    def test_singleton_groups_witness_disjoint_heldout_feasibility(self) -> None:
        groups = {
            "image-a": [{"historical_all_on_status": "correct"}],
            "image-b": [{"historical_all_on_status": "correct"}],
            "image-c": [{"historical_all_on_status": "wrong"}],
            "image-d": [{"historical_all_on_status": "wrong"}],
            "image-pair": [
                {"historical_all_on_status": "correct"},
                {"historical_all_on_status": "wrong"},
            ],
        }

        counts, feasible = singleton_heldout_sufficiency(
            groups, required_correct=2, required_wrong=2
        )

        self.assertEqual(counts, {"correct": 2, "wrong": 2})
        self.assertTrue(feasible)

    def test_multi_question_groups_do_not_count_as_singleton_witnesses(self) -> None:
        groups = {
            "image-a": [{"historical_all_on_status": "correct"}],
            "image-pair": [
                {"historical_all_on_status": "wrong"},
                {"historical_all_on_status": "wrong"},
            ],
        }

        counts, feasible = singleton_heldout_sufficiency(
            groups, required_correct=1, required_wrong=1
        )

        self.assertEqual(counts, {"correct": 1})
        self.assertFalse(feasible)


if __name__ == "__main__":
    unittest.main()
