from __future__ import annotations

import unittest

import numpy as np

from baseline_relative_visual_router.admission import (
    calibrate_threshold,
    outcome_label,
    summarize_admission,
)


def paired(baseline: bool, routed: bool, budget: int = 18):
    return {
        "baseline_correct": baseline,
        "router_correct": routed,
        "selected_num_visual_on_layers": budget,
    }


class AdmissionTest(unittest.TestCase):
    def test_four_way_outcome_labels(self) -> None:
        self.assertEqual(outcome_label(paired(True, True)), "preserve")
        self.assertEqual(outcome_label(paired(True, False)), "harm")
        self.assertEqual(outcome_label(paired(False, True)), "rescue")
        self.assertEqual(outcome_label(paired(False, False)), "unsolved")

    def test_gate_routes_low_harm_scores(self) -> None:
        rows = [paired(True, True), paired(True, False), paired(False, True)]
        scores = np.asarray([0.1, 0.9, 0.2])
        summary = summarize_admission(rows, scores, threshold=0.5)
        self.assertEqual(summary["routed_count"], 2)
        self.assertEqual(summary["harm_count"], 0)
        self.assertEqual(summary["rescue_count"], 1)

    def test_calibration_prefers_saving_only_after_noninferiority(self) -> None:
        rows = [
            paired(True, True, 10),
            paired(True, False, 5),
            paired(False, True, 12),
            paired(False, False, 8),
        ] * 100
        scores = np.asarray([0.1, 0.95, 0.2, 0.3] * 100)
        selected, _ = calibrate_threshold(rows, scores, epsilon=0.0)
        self.assertEqual(selected["harm_count"], 0)
        self.assertEqual(selected["rescue_count"], 100)
        self.assertGreater(selected["route_sensitive_layer_saving_fraction"], 0.0)

    def test_fast_calibration_matches_direct_summaries(self) -> None:
        rows = [
            paired(True, True, 9),
            paired(True, False, 7),
            paired(False, True, 13),
            paired(False, False, 4),
            paired(True, True, 17),
        ]
        scores = np.asarray([0.2, 0.8, 0.2, 0.6, 0.4])
        _, sweep = calibrate_threshold(rows, scores, epsilon=1.0)
        for fast in sweep:
            direct = summarize_admission(rows, scores, threshold=fast["threshold"])
            self.assertEqual(fast.keys(), direct.keys())
            for key in fast:
                if isinstance(fast[key], float):
                    self.assertAlmostEqual(fast[key], direct[key], places=12)
                else:
                    self.assertEqual(fast[key], direct[key])


if __name__ == "__main__":
    unittest.main()
