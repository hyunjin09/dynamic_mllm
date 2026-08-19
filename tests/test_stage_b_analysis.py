from __future__ import annotations

import unittest

from tools.research_analysis.v2.stage_b_reference_analysis import (
    behavior_category,
    effect_label,
    summarize_values,
)


class StageBAnalysisTests(unittest.TestCase):
    def test_effect_label_respects_frozen_dead_zone(self) -> None:
        self.assertEqual(effect_label(0.01, 0.001), "positive")
        self.assertEqual(effect_label(-0.01, 0.001), "negative")
        self.assertEqual(effect_label(0.001, 0.001), "answer_silent")

    def test_behavior_categories_use_strict_official_correctness(self) -> None:
        self.assertEqual(behavior_category(0.0, 1.0, "same", "new", 1.0), "full_wrong_to_intervention_correct")
        self.assertEqual(behavior_category(1.0, 1.0, "same", "same", 1.0), "full_correct_to_intervention_correct")
        self.assertEqual(behavior_category(1.0, 0.0, "old", "new", 1.0), "full_correct_to_intervention_wrong")
        self.assertEqual(behavior_category(0.0, 0.0, "same", "same", 1.0), "unchanged_wrong")
        self.assertEqual(behavior_category(0.0, 0.0, "old", "new", 1.0), "changed_but_still_wrong")
        self.assertEqual(behavior_category(0.0, 1 / 3, "old", "new", 1.0), "changed_but_still_wrong")

    def test_summary_bootstraps_rows_not_pseudoreplicated_layers(self) -> None:
        summary = summarize_values([1.0, 3.0], bootstrap_replicates=100, seed=7)
        self.assertEqual(summary["n_samples"], 2)
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["median"], 2.0)
        self.assertLessEqual(summary["mean_ci_low"], summary["mean"])
        self.assertGreaterEqual(summary["mean_ci_high"], summary["mean"])


if __name__ == "__main__":
    unittest.main()
