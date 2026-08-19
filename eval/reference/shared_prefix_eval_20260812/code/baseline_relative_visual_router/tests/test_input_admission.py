from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from baseline_relative_visual_router.input_admission import (
    compose_admission_score,
    fixed_uid_train_calibration_split,
    load_prefix_feature_cache,
    prefix_feature_matrix,
    override_scores_with_safe_admission,
    stratified_train_calibration_split,
)


class InputAdmissionTest(unittest.TestCase):
    def test_prefix_cache_loads_sorted_and_builds_five_summary_blocks(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prefix_02_shard_00_of_01_part_00000.pt"
            rows = [
                {"uid": "b", "prefix_layers": 2, "selected_visual_on_mask": [1, 1, 0]},
                {"uid": "a", "prefix_layers": 2, "selected_visual_on_mask": [1, 1, 1]},
            ]
            torch.save(
                {
                    "schema_version": "shared_dense_prefix_actual_policy_v1",
                    "prefix_layers": 2,
                    "instruction_mean": torch.ones(2, 3),
                    "instruction_window_mean": torch.ones(2, 3) * 2,
                    "instruction_last": torch.ones(2, 3) * 3,
                    "visual_summaries": torch.ones(2, 2, 3) * 4,
                    "rows": rows,
                },
                path,
            )
            tensors, metadata = load_prefix_feature_cache(Path(directory), expected_prefix_layers=2)
            self.assertEqual([row["uid"] for row in metadata], ["a", "b"])
            self.assertEqual(tuple(prefix_feature_matrix(tensors).shape), (2, 15))

    def test_fixed_uid_split_is_invariant_to_policy_outcome(self) -> None:
        first = []
        second = []
        for benchmark in ("a", "b"):
            for index in range(40):
                common = {
                    "benchmark": benchmark,
                    "uid": f"{benchmark}-{index}",
                    "baseline_correct": bool(index % 2),
                }
                first.append({**common, "outcome": "harm" if index % 3 == 0 else "preserve"})
                second.append({**common, "outcome": "rescue" if index % 5 == 0 else "unsolved"})
        left = fixed_uid_train_calibration_split(first, train_fraction=0.8, seed=13)
        right = fixed_uid_train_calibration_split(second, train_fraction=0.8, seed=13)
        self.assertEqual(left["train"].tolist(), right["train"].tolist())
        self.assertEqual(left["calibration"].tolist(), right["calibration"].tolist())

    def test_stratified_split_is_disjoint_and_complete(self) -> None:
        metadata = []
        for benchmark in ("a", "b"):
            for outcome in ("preserve", "harm", "rescue", "unsolved"):
                metadata.extend(
                    {"benchmark": benchmark, "outcome": outcome, "uid": f"{benchmark}-{outcome}-{i}"}
                    for i in range(10)
                )
        split = stratified_train_calibration_split(metadata, train_fraction=0.8, seed=7)
        train = set(split["train"].tolist())
        calibration = set(split["calibration"].tolist())
        self.assertFalse(train & calibration)
        self.assertEqual(train | calibration, set(range(len(metadata))))
        self.assertEqual(len(train), 64)
        self.assertEqual(len(calibration), 16)

    def test_override_scores_force_safe_samples_to_be_admitted_first(self) -> None:
        utility = np.asarray([0.4, -0.2, 0.1, 0.7])
        safe = np.asarray([True, False, True, False])
        score = override_scores_with_safe_admission(utility, safe)
        self.assertLess(score[0], utility.min())
        self.assertEqual(score[0], score[2])
        self.assertEqual(score[1], utility[1])
        self.assertEqual(score[3], utility[3])

    def test_admission_score_modes_have_distinct_contracts(self) -> None:
        harm = np.asarray([[0.1, 0.8], [0.2, 0.6]])
        rescue = np.asarray([[0.0, 0.9], [0.2, 0.7]])
        common = dict(
            harm_beta=0.0,
            harm_threshold=0.2,
            utility_beta=0.0,
            rescue_weight=1.0,
        )
        harm_only = compose_admission_score("harm_only", harm, rescue, **common)
        utility = compose_admission_score("utility_only", harm, rescue, **common)
        hierarchical = compose_admission_score("hierarchical", harm, rescue, **common)
        np.testing.assert_allclose(harm_only, [0.15, 0.7])
        np.testing.assert_allclose(utility, [0.05, -0.1])
        self.assertLess(hierarchical[0], hierarchical[1])
        with self.assertRaises(ValueError):
            compose_admission_score("unknown", harm, rescue, **common)


if __name__ == "__main__":
    unittest.main()
