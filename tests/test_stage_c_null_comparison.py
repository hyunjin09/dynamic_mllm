from __future__ import annotations

import unittest

import numpy as np

from tools.research_analysis.v2.stage_c_null_comparison import (
    evaluate_null_superiority,
    evaluate_real_residual_sensitivity,
)


class StageCNullComparisonTests(unittest.TestCase):
    def test_conjunction_requires_both_paired_upper_bounds_below_zero(self) -> None:
        result = evaluate_null_superiority(
            real=np.array([-2.0, -2.0, -2.0, -2.0]),
            covariance_null=np.array([[-1.0], [-1.0], [-1.0], [-1.0]]),
            real_residual_null=np.array([[-1.5], [-1.5], [-1.5], [-1.5]]),
            image_ids=np.array(["a", "b", "c", "d"]),
            bootstrap_draws=200,
            covariance_seed=1,
            real_residual_seed=2,
        )
        self.assertTrue(result["gate_pass"])
        self.assertLess(result["covariance"]["paired_ci_high"], 0.0)
        self.assertLess(result["real_residual"]["paired_ci_high"], 0.0)

    def test_one_failed_family_fails_intersection_union_gate(self) -> None:
        result = evaluate_null_superiority(
            real=np.array([-1.0, -1.0, -1.0, -1.0]),
            covariance_null=np.array([[-0.5], [-0.5], [-0.5], [-0.5]]),
            real_residual_null=np.array([[-2.0], [-2.0], [-2.0], [-2.0]]),
            image_ids=np.array(["a", "b", "c", "d"]),
            bootstrap_draws=200,
            covariance_seed=1,
            real_residual_seed=2,
        )
        self.assertFalse(result["gate_pass"])

    def test_real_residual_sensitivity_uses_only_prespecified_subset(self) -> None:
        result = evaluate_real_residual_sensitivity(
            real=np.array([-2.0, -2.0, -2.0, 9.0]),
            real_residual_null=np.array([[-1.0], [-1.0], [-1.0], [-9.0]]),
            image_ids=np.array(["a", "b", "c", "d"]),
            included=np.array([True, True, True, False]),
            bootstrap_draws=200,
            seed=2,
        )
        self.assertEqual(result["n_records"], 3)
        self.assertEqual(result["n_image_clusters"], 3)
        self.assertLess(result["paired_ci_high"], 0.0)


if __name__ == "__main__":
    unittest.main()
