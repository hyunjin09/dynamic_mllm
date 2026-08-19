import unittest

import torch

from nulls.four_action_structured import (
    ResidualPairMetadata,
    generate_isotropic_null,
    pair_matching_ratio,
    select_pair_donors,
)


class FourActionStructuredNullTests(unittest.TestCase):
    def test_isotropic_draw_is_seeded_shape_and_norm_matched(self):
        first = generate_isotropic_null((3, 4), target_norm=7.0, seed=17)
        second = generate_isotropic_null((3, 4), target_norm=7.0, seed=17)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual((3, 4), tuple(first.shape))
        self.assertAlmostEqual(7.0, float(first.norm().item()), places=5)

    def test_pair_distance_matches_both_path_geometries(self):
        target = ResidualPairMetadata("t", "it", 2.0, 4.0, 10, 20, 30)
        donor = ResidualPairMetadata("d", "id", 3.0, 2.0, 12, 18, 27)
        self.assertAlmostEqual(2.0, pair_matching_ratio(target, donor))

    def test_pair_donors_exclude_same_sample_and_image(self):
        target = ResidualPairMetadata("t", "it", 2.0, 4.0, 10, 20, 30)
        donors = [
            ResidualPairMetadata("t", "other", 2.0, 4.0, 10, 20, 30),
            ResidualPairMetadata("other", "it", 2.0, 4.0, 10, 20, 30),
            ResidualPairMetadata("d1", "i1", 2.1, 4.1, 10, 20, 30),
            ResidualPairMetadata("d2", "i2", 1.9, 3.9, 10, 20, 30),
        ]
        selected = select_pair_donors(target, donors, draws=2, seed=4, caliper=1.2)
        self.assertEqual({"d1", "d2"}, {item.sample_id for item in selected})


if __name__ == "__main__":
    unittest.main()
