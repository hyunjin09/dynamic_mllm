import unittest

from tools.freeze_binary_predictor_split import select_validation_groups


def _row(uid, group, historical, current="correct"):
    return {
        "uid": uid,
        "benchmark": "toy",
        "image_group_id": group,
        "historical_all_on_status": historical,
        "current_all_on_status": current,
    }


class FreezeBinaryPredictorSplitTest(unittest.TestCase):
    def test_exact_balanced_validation_is_group_disjoint_and_deterministic(self):
        rows = [
            _row("a1", "a", "correct"),
            _row("a2", "a", "wrong"),
            _row("b", "b", "correct"),
            _row("c", "c", "wrong"),
            _row("d", "d", "correct"),
            _row("e", "e", "wrong"),
        ]

        first = select_validation_groups(
            rows, dataset="toy", target_records=2, target_correct=1, seed=41
        )
        second = select_validation_groups(
            list(reversed(rows)), dataset="toy", target_records=2, target_correct=1, seed=41
        )

        self.assertEqual(first, second)
        validation = [row for row in rows if row["image_group_id"] in first]
        self.assertEqual(len(validation), 2)
        self.assertEqual(
            sum(row["historical_all_on_status"] == "correct" for row in validation), 1
        )
        self.assertFalse(
            {row["image_group_id"] for row in validation}
            & {row["image_group_id"] for row in rows if row not in validation}
        )

    def test_current_outcomes_cannot_change_assignment(self):
        rows = [
            _row("a", "a", "correct", current="correct"),
            _row("b", "b", "wrong", current="wrong"),
            _row("c", "c", "correct", current="wrong"),
            _row("d", "d", "wrong", current="correct"),
        ]
        changed = [{**row, "current_all_on_status": "wrong"} for row in rows]

        first = select_validation_groups(
            rows, dataset="toy", target_records=2, target_correct=1, seed=17
        )
        second = select_validation_groups(
            changed, dataset="toy", target_records=2, target_correct=1, seed=17
        )

        self.assertEqual(first, second)

    def test_infeasible_exact_target_fails_closed(self):
        rows = [
            _row("a1", "a", "correct"),
            _row("a2", "a", "correct"),
            _row("b1", "b", "wrong"),
            _row("b2", "b", "wrong"),
        ]

        with self.assertRaisesRegex(ValueError, "no exact image-group-disjoint"):
            select_validation_groups(
                rows, dataset="toy", target_records=3, target_correct=1, seed=17
            )


if __name__ == "__main__":
    unittest.main()
