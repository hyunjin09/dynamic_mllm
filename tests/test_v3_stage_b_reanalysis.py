from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tools.research_analysis.v3.reanalyze_stage_b import (
    LAYERS,
    build_agreement,
    conservative_best,
    derive_quantities,
    independent_action,
    trimmed_mean,
)


class V3StageBReanalysisTests(unittest.TestCase):
    def test_four_cell_quantities_follow_v3_definitions(self) -> None:
        q = {"IGNORE": 1.0, "READ_ONLY": 2.0, "WRITE_ONLY": 3.0, "FULL": 4.0}
        result = derive_quantities(q, 1e-6)
        self.assertEqual(result["advantages"], {"IGNORE": -3.0, "READ_ONLY": -2.0, "WRITE_ONLY": -1.0, "FULL": 0.0})
        self.assertEqual(result["g"], -1.0)
        self.assertEqual(result["best_action"], "FULL")
        self.assertEqual(result["effects"]["read_w0"], 1.0)
        self.assertEqual(result["effects"]["read_w1"], 1.0)
        self.assertEqual(result["effects"]["write_r0"], 2.0)
        self.assertEqual(result["effects"]["write_r1"], 2.0)
        self.assertEqual(result["effects"]["interaction"], 0.0)

    def test_ties_prefer_full_without_erasing_tie_set(self) -> None:
        best, ties = conservative_best(
            {"IGNORE": 0.0, "READ_ONLY": 1.0, "WRITE_ONLY": 1.0, "FULL": 1.0 - 5e-7},
            1e-6,
        )
        self.assertEqual(best, "FULL")
        self.assertEqual(set(ties), {"READ_ONLY", "WRITE_ONLY", "FULL"})

    def test_reported_best_action_remains_exact_argmax_inside_epsilon(self) -> None:
        result = derive_quantities(
            {"IGNORE": 0.0, "READ_ONLY": 1.0, "WRITE_ONLY": 0.5, "FULL": 1.0 - 5e-7},
            1e-6,
        )
        self.assertEqual(result["best_action"], "READ_ONLY")
        self.assertEqual(result["epsilon_preferred_action"], "FULL")

    def test_additive_main_effects_fail_on_xor_interaction(self) -> None:
        q = {"IGNORE": 0.0, "READ_ONLY": 1.0, "WRITE_ONLY": 1.0, "FULL": 0.0}
        action = independent_action(q, 1e-6)
        result = derive_quantities(q, 1e-6)
        self.assertEqual(action, "FULL")
        self.assertFalse(result["independent_recovers_best"])

    def test_trimmed_mean_is_heavy_tail_robust(self) -> None:
        values = np.asarray([-100.0, 1.0, 2.0, 3.0, 100.0])
        self.assertEqual(trimmed_mean(values, 0.2), 2.0)

    def test_agreement_uses_stored_advantage_column_order(self) -> None:
        rows = []
        for dataset in ("gqa", "textvqa"):
            for layer in LAYERS:
                row = {"dataset": dataset, "layer": layer, "best_action_mean": "FULL", "best_action_sequence": "FULL"}
                for quantity in ("g", "read_w0", "read_w1", "write_r0", "write_r1", "interaction"):
                    row[f"{quantity}_mean"] = 0.1
                    row[f"{quantity}_sequence"] = 0.2
                for action in ("ignore", "read_only", "write_only"):
                    row[f"a_mean_{action}"] = 0.1
                    row[f"a_sequence_{action}"] = 0.2
                rows.append(row)
        result = build_agreement(pd.DataFrame(rows))
        self.assertTrue(any(row["quantity"] == "A_IGNORE" for row in result))


if __name__ == "__main__":
    unittest.main()
