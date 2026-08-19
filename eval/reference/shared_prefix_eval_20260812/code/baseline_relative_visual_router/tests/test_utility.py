from __future__ import annotations

import unittest

import numpy as np

from baseline_relative_visual_router.utility import conservative_utility_score


class UtilityScoreTest(unittest.TestCase):
    def test_rescue_probability_makes_routing_more_attractive(self) -> None:
        harm = np.asarray([[0.1, 0.1], [0.1, 0.1]])
        rescue = np.asarray([[0.8, 0.2], [0.8, 0.2]])
        score = conservative_utility_score(
            harm, rescue, uncertainty_beta=0.0, rescue_weight=1.0
        )
        self.assertLess(score[0], score[1])

    def test_uncertainty_penalizes_harm_and_discounts_rescue(self) -> None:
        harm = np.asarray([[0.1], [0.5]])
        rescue = np.asarray([[0.2], [0.8]])
        plain = conservative_utility_score(
            harm, rescue, uncertainty_beta=0.0, rescue_weight=1.0
        )
        conservative = conservative_utility_score(
            harm, rescue, uncertainty_beta=1.0, rescue_weight=1.0
        )
        self.assertGreater(conservative[0], plain[0])


if __name__ == "__main__":
    unittest.main()
