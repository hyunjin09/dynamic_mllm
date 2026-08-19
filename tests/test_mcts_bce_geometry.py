import math
import unittest

from label_regeneration.bce_geometry import (
    connected_components,
    diversity_balanced_indices,
    effective_component_count,
    layer_marginals,
    pareto_efficient_indices,
    polar_route_weights,
    threshold_mask,
)
from experiments.analyze_mcts_bce_labels import fmean_defined


class MctsBceGeometryTest(unittest.TestCase):
    def test_polar_weighted_oracle_matches_all_on_downweight_contract(self):
        masks = [(1, 1, 1, 1), (0, 0, 0, 0), (1, 0, 1, 0)]
        weights = polar_route_weights(masks)
        self.assertEqual(weights, [0.3 / 2.3, 1.0 / 2.3, 1.0 / 2.3])
        marginals = layer_marginals(masks, weights)
        self.assertEqual(threshold_mask(marginals), (1, 0, 1, 0))

    def test_threshold_ties_resolve_on(self):
        self.assertEqual(threshold_mask([0.5, 0.499999, 0.500001]), (1, 0, 1))

    def test_weighted_marginals_clamp_only_roundoff(self):
        masks = [(1, 1)] * 31
        weights = [1.0 / 31] * 31
        self.assertEqual(layer_marginals(masks, weights), [1.0, 1.0])

    def test_undefined_pairwise_values_are_not_imputed(self):
        self.assertEqual(fmean_defined([None, 2.0, 4.0]), 3.0)
        self.assertIsNone(fmean_defined([None]))

    def test_components_and_effective_count(self):
        masks = [(0, 0, 0, 0), (0, 0, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)]
        components = connected_components(masks, radius=1)
        self.assertEqual(sorted(map(len, components)), [2, 2])
        self.assertTrue(math.isclose(effective_component_count(components), 2.0))

    def test_pareto_uses_stored_utility_and_strictly_lower_cost(self):
        masks = [(1, 1, 1), (1, 0, 0), (0, 0, 0), (0, 1, 0)]
        utilities = [1.0, 1.0, 0.5, 0.9]
        self.assertEqual(pareto_efficient_indices(masks, utilities), [1, 2, 3])

    def test_diversity_selection_is_deterministic(self):
        masks = [(1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 0, 0), (0, 0, 1, 1)]
        indices = diversity_balanced_indices(masks, limit=3)
        self.assertEqual([masks[index] for index in indices], [(0, 0, 0, 0), (1, 1, 1, 1), (0, 0, 1, 1)])


if __name__ == "__main__":
    unittest.main()
