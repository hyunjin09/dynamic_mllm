from __future__ import annotations

import math
import unittest

import numpy as np

from tools.research_analysis.v2.stage_c_analysis import (
    behavior_category,
    classify_stage_c_outcome,
    consensus_bin,
    reference_format_category,
    sign_agreement_fraction,
    spearman_correlation,
    trimmed_mean,
    uniform_accepted_aggregate,
)


class StageCAnalysisTests(unittest.TestCase):
    def test_trimmed_mean_removes_equal_tails(self) -> None:
        values = np.asarray([-100.0, 1.0, 2.0, 3.0, 100.0])
        self.assertEqual(trimmed_mean(values, 0.2), 2.0)

    def test_spearman_uses_average_ranks_for_ties(self) -> None:
        left = np.asarray([1.0, 1.0, 2.0, 3.0])
        right = np.asarray([4.0, 4.0, 2.0, 1.0])
        self.assertAlmostEqual(spearman_correlation(left, right), -1.0)

    def test_uniform_aggregate_ignores_annotation_frequency(self) -> None:
        scores = [
            {"sequence_logprob": -1.0, "mean_logprob": -0.5},
            {"sequence_logprob": -3.0, "mean_logprob": -1.5},
        ]
        result = uniform_accepted_aggregate(scores)
        expected_sequence = math.log((math.exp(-1.0) + math.exp(-3.0)) / 2.0)
        expected_mean = math.log((math.exp(-0.5) + math.exp(-1.5)) / 2.0)
        self.assertAlmostEqual(result["sequence_logprob"], expected_sequence)
        self.assertAlmostEqual(result["mean_logprob"], expected_mean)

    def test_behavior_categories_are_exhaustive_for_strict_correctness(self) -> None:
        self.assertEqual(behavior_category(0.0, 1.0, "a", "b"), "full_wrong_to_write_only_correct")
        self.assertEqual(behavior_category(1.0, 0.0, "a", "b"), "full_correct_to_write_only_wrong")
        self.assertEqual(behavior_category(0.0, 0.0, "a", "b"), "wrong_to_different_wrong")
        self.assertEqual(behavior_category(1.0, 1.0, "a", "a"), "unchanged_correct")
        self.assertEqual(behavior_category(0.0, 0.0, "a", "a"), "unchanged_wrong")

    def test_reference_format_categories_are_predeclared(self) -> None:
        self.assertEqual(reference_format_category(["07 10 2012"]), "numeric")
        self.assertEqual(reference_format_category(["royal air force"]), "alphabetic")
        self.assertEqual(reference_format_category(["36a", "love"]), "mixed_or_symbolic")

    def test_stage_c_outcome_classification_uses_frozen_conjunction(self) -> None:
        self.assertEqual(classify_stage_c_outcome(False, True, True), "Outcome A")
        self.assertEqual(classify_stage_c_outcome(True, True, False), "Outcome B")
        self.assertEqual(classify_stage_c_outcome(True, True, True), "Outcome C")

    def test_sign_agreement_is_paired(self) -> None:
        self.assertAlmostEqual(
            sign_agreement_fraction(
                np.asarray([-1.0, 2.0, 0.0]),
                np.asarray([-3.0, -2.0, 0.0]),
            ),
            2 / 3,
        )

    def test_consensus_bins_keep_partial_credit_separate(self) -> None:
        self.assertEqual(consensus_bin(0.0), "strict_wrong")
        self.assertEqual(consensus_bin(0.6), "partial_consensus")
        self.assertEqual(consensus_bin(1.0), "strict_correct")


if __name__ == "__main__":
    unittest.main()
