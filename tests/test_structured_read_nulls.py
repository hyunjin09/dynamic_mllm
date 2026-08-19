from __future__ import annotations

import unittest

import torch

from experiments.stage_c_entry_gate import image_id
from nulls.structured_read import (
    DonorMetadata,
    compose_null_read_output,
    fit_fixed_grid_covariance,
    fit_real_donor_caliper,
    generate_covariance_null,
    map_rows,
    select_real_donors,
)


class StructuredReadNullTests(unittest.TestCase):
    def test_calibration_image_id_uses_frozen_asset_key_fallback(self) -> None:
        record = {
            "source_asset_id": None,
            "selection_asset_key": "/data/example/image.jpg",
            "local_image_path": "/data/example/image.jpg",
        }
        self.assertEqual(image_id(record), "/data/example/image.jpg")

    def test_row_mapping_and_norm_match_are_exact(self) -> None:
        source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        mapped = map_rows(source, 5)
        self.assertEqual(tuple(mapped.shape), (5, 4))
        self.assertTrue(torch.equal(mapped[0], source[0]))
        self.assertTrue(torch.equal(mapped[-1], source[-1]))

    def test_covariance_fit_and_draw_are_deterministic_and_norm_matched(self) -> None:
        samples = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
            torch.tensor([[0.0, 1.0], [2.0, 1.0]]),
        ]
        fit = fit_fixed_grid_covariance(
            samples,
            grid_rows=2,
            variance_target=0.90,
            eigen_shrinkage=0.05,
        )
        first = generate_covariance_null(fit, target_rows=3, target_norm=7.0, seed=9)
        second = generate_covariance_null(fit, target_rows=3, target_norm=7.0, seed=9)
        self.assertTrue(torch.equal(first, second))
        self.assertAlmostEqual(float(first.norm().item()), 7.0, places=5)
        self.assertGreaterEqual(fit.rank, 1)

    def test_real_donor_selection_excludes_identity_and_obeys_calipers(self) -> None:
        target = DonorMetadata("target", "img-t", 10.0, 20, 100, 120)
        donors = [
            DonorMetadata("target", "img-x", 10.0, 20, 100, 120),
            DonorMetadata("same-image", "img-t", 10.0, 20, 100, 120),
            DonorMetadata("good-1", "img-1", 11.0, 21, 105, 126),
            DonorMetadata("good-2", "img-2", 9.0, 19, 95, 114),
            DonorMetadata("bad-norm", "img-3", 30.0, 20, 100, 120),
        ]
        selected = select_real_donors(
            target,
            donors,
            draws=2,
            seed=5,
            matching_ratio_cap=1.25,
        )
        self.assertEqual({item.sample_id for item in selected}, {"good-1", "good-2"})

    def test_fitted_real_donor_cap_covers_every_calibration_target(self) -> None:
        donors = [
            DonorMetadata(f"s{i}", f"img-{i}", float(i + 1), i + 10, 100 + i, 120 + i)
            for i in range(10)
        ]
        cap = fit_real_donor_caliper(donors, draws=3)
        self.assertGreaterEqual(cap, 1.0)
        for target in donors:
            selected = select_real_donors(
                target,
                donors,
                draws=3,
                seed=5,
                matching_ratio_cap=cap,
            )
            self.assertEqual(len(selected), 3)
            self.assertTrue(all(item.sample_id != target.sample_id for item in selected))

    def test_real_donor_selection_fails_closed(self) -> None:
        target = DonorMetadata("target", "img-t", 10.0, 20, 100, 120)
        with self.assertRaisesRegex(ValueError, "eligible donors"):
            select_real_donors(
                target,
                [DonorMetadata("only", "img-1", 10.0, 20, 100, 120)],
                draws=2,
                seed=5,
                matching_ratio_cap=1.5,
            )

    def test_null_read_composition_preserves_visual_rows(self) -> None:
        full = torch.tensor([[[1.0], [2.0], [3.0]]])
        off = torch.tensor([[[1.0], [1.5], [2.5]]])
        null = torch.tensor([[[0.0], [0.25], [0.75]]])
        visual = torch.tensor([[True, False, False]])
        result = compose_null_read_output(full, off, null, visual)
        self.assertEqual(float(result[0, 0, 0]), 1.0)
        self.assertEqual(float(result[0, 1, 0]), 1.75)
        self.assertEqual(float(result[0, 2, 0]), 3.25)


if __name__ == "__main__":
    unittest.main()
