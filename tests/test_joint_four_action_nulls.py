import unittest

import torch

from tools.research_analysis.v3.freeze_v3_null_models import bootstrap_ci, joint_score_covariance_error
from nulls.joint_four_action import (
    PairedGeometryMetadata,
    fit_joint_path_covariance,
    fit_paired_donor_calipers,
    generate_paired_isotropic_null,
    generate_paired_real_null,
    generate_joint_path_null,
    paired_geometry_distance,
    search_budget_cells,
    select_paired_donors,
)


class JointFourActionNullTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(19)
        self.read = [torch.randn(3 + index % 2, 6, generator=generator) for index in range(12)]
        self.write = [
            self.read[index].mean(dim=0, keepdim=True).repeat(2 + index % 3, 1)
            + 0.1 * torch.randn(2 + index % 3, 6, generator=generator)
            for index in range(12)
        ]

    def test_joint_draw_is_deterministic_native_shape_and_row_norm_matched(self) -> None:
        fit = fit_joint_path_covariance(
            self.read,
            self.write,
            grid_rows=4,
            variance_target=0.9,
            joint_shrinkage=0.1,
        )
        read_norms = torch.tensor([1.0, 2.0, 3.0])
        write_norms = torch.tensor([0.5, 1.5])
        first = generate_joint_path_null(fit, read_norms, write_norms, seed=23)
        second = generate_joint_path_null(fit, read_norms, write_norms, seed=23)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertEqual((3, 6), tuple(first[0].shape))
        self.assertEqual((2, 6), tuple(first[1].shape))
        self.assertTrue(torch.allclose(first[0].norm(dim=1), read_norms, atol=1e-5))
        self.assertTrue(torch.allclose(first[1].norm(dim=1), write_norms, atol=1e-5))

    def test_paired_donor_distance_uses_both_paths_and_scale(self) -> None:
        target = PairedGeometryMetadata(
            "t", "it", "gqa", 4, 2.0, 4.0, 4, 3, 12, 20, 0.2, 0.4, 0.5, 0.25
        )
        donor = PairedGeometryMetadata(
            "d", "id", "gqa", 4, 3.0, 2.0, 5, 3, 12, 18, 0.25, 0.2, 0.4, 0.5
        )
        self.assertAlmostEqual(2.0, paired_geometry_distance(target, donor))

    def test_real_pair_maps_both_donor_paths_and_matches_target_row_norms(self) -> None:
        read, write = generate_paired_real_null(
            self.read[0],
            self.write[0],
            torch.tensor([1.0, 2.0]),
            torch.tensor([0.5, 1.0, 1.5]),
        )
        self.assertEqual((2, 6), tuple(read.shape))
        self.assertEqual((3, 6), tuple(write.shape))
        self.assertTrue(torch.allclose(read.norm(dim=1), torch.tensor([1.0, 2.0]), atol=1e-5))
        self.assertTrue(
            torch.allclose(write.norm(dim=1), torch.tensor([0.5, 1.0, 1.5]), atol=1e-5)
        )

    def test_isotropic_pair_is_deterministic_and_exactly_row_norm_matched(self) -> None:
        read_norms = torch.tensor([1.0, 2.0])
        write_norms = torch.tensor([0.5, 1.5, 2.5])
        first = generate_paired_isotropic_null(6, read_norms, write_norms, seed=29)
        second = generate_paired_isotropic_null(6, read_norms, write_norms, seed=29)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(torch.allclose(first[0].norm(dim=1), read_norms, atol=1e-5))
        self.assertTrue(torch.allclose(first[1].norm(dim=1), write_norms, atol=1e-5))

    def test_calipers_are_stratified_and_select_exact_nearest_donors(self) -> None:
        rows = []
        for index in range(10):
            rows.append(
                PairedGeometryMetadata(
                    f"s{index}",
                    f"i{index}",
                    "gqa",
                    0,
                    1.0 + index / 100,
                    2.0 + index / 100,
                    4,
                    3,
                    12,
                    20,
                    0.2,
                    0.4,
                    0.5,
                    0.25,
                )
            )
        calipers, coverage = fit_paired_donor_calipers(rows, donor_count=3)
        self.assertEqual({("gqa", 0)}, set(calipers))
        selected = select_paired_donors(rows[0], rows, 3, 7, calipers[("gqa", 0)])
        self.assertEqual(3, len(selected))
        self.assertNotIn("s0", {row.sample_id for row in selected})
        self.assertEqual(10, len(coverage))

    def test_search_budget_has_exactly_twenty_one_cells(self) -> None:
        cells = search_budget_cells(
            [0, 4, 8, 12, 16, 20, 24],
            ["IGNORE", "READ_ONLY", "WRITE_ONLY"],
        )
        self.assertEqual(21, len(cells))
        self.assertEqual(21, len({(row["layer"], row["action"]) for row in cells}))

    def test_joint_score_monte_carlo_fidelity_is_reproducible(self) -> None:
        fit = fit_joint_path_covariance(
            self.read,
            self.write,
            grid_rows=4,
            variance_target=0.9,
            joint_shrinkage=0.1,
        )
        first = joint_score_covariance_error(fit, seed=31, draws=8192)
        second = joint_score_covariance_error(fit, seed=31, draws=8192)
        self.assertEqual(first, second)
        self.assertLess(first, 0.1)

    def test_bootstrap_ci_accepts_float64_values(self) -> None:
        interval = bootstrap_ci(torch.arange(8, dtype=torch.float64), draws=50, seed=41)
        self.assertEqual(2, len(interval))
        self.assertLess(interval[0], interval[1])


if __name__ == "__main__":
    unittest.main()
