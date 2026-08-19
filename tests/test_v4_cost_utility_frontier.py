import unittest

import numpy as np

from tools.research_analysis.v4.cost_utility_frontier import (
    ACTIONS,
    action_cost_geometry,
    pair_frontier_flags,
    select_budget,
    select_penalized,
)


class V4CostUtilityFrontierTests(unittest.TestCase):
    def test_exact_operation_level_flop_formulas(self):
        records = [
            {
                "expected_visual_token_count": 2,
                "expected_visual_first": 1,
                "expected_visual_last": 2,
                "expected_prompt_token_length": 5,
            },
            {
                "expected_visual_token_count": 2,
                "expected_visual_first": 1,
                "expected_visual_last": 2,
                "expected_prompt_token_length": 6,
            },
        ]
        geometry = action_cost_geometry(records, hidden_size=4, intermediate_size=8, kv_dim=2)
        self.assertEqual(48, geometry["read_value_flops"])
        self.assertEqual(64, geometry["write_visual_q_flops"])
        self.assertEqual(80, geometry["write_visual_attention_flops"])
        self.assertEqual(64, geometry["write_visual_o_flops"])
        self.assertEqual(384, geometry["write_visual_ffn_flops"])
        self.assertEqual(592, geometry["write_flops"])
        self.assertEqual(640, geometry["full_visual_flops"])

    def test_query_oracle_can_use_half_cost_action_pair(self):
        pair = {
            "q": np.asarray([[0.0, 1.0, 0.0, 0.9], [0.0, 0.0, 1.0, 0.9]]),
            "costs": np.asarray([0.0, 0.1, 0.9, 1.0]),
            "normalized_costs": np.asarray([0.0, 0.1, 0.9, 1.0]),
            "full_baseline": 0.9,
        }
        image_actions, image_utility, image_cost = select_penalized(
            pair, "image_only", 0.0, 1e-6
        )
        query_actions, query_utility, query_cost = select_penalized(
            pair, "image_query", 0.0, 1e-6
        )
        self.assertEqual((3, 3), image_actions)
        self.assertEqual((1, 2), query_actions)
        self.assertAlmostEqual(0.9, image_utility)
        self.assertAlmostEqual(1.0, query_utility)
        self.assertAlmostEqual(1.0, image_cost)
        self.assertAlmostEqual(0.5, query_cost)

        budget_actions, budget_utility, budget_cost = select_budget(
            pair, "image_query", 0.5, 1e-6
        )
        self.assertEqual((1, 2), budget_actions)
        self.assertAlmostEqual(1.0, budget_utility)
        self.assertAlmostEqual(0.5, budget_cost)
        flags = pair_frontier_flags(pair, 1e-6)
        self.assertTrue(flags["query_frontier_strictly_expands_shared"])
        self.assertTrue(flags["query_strictly_dominates_unconstrained_shared"])

    def test_epsilon_tie_chooses_lower_cost(self):
        pair = {
            "q": np.asarray([[1.0, 1.0 + 5e-7, 0.0, 0.0]] * 2),
            "costs": np.asarray([0.0, 0.1, 0.9, 1.0]),
            "normalized_costs": np.asarray([0.0, 0.1, 0.9, 1.0]),
            "full_baseline": 0.0,
        }
        actions, _, cost = select_penalized(pair, "image_only", 0.0, 1e-6)
        self.assertEqual((0, 0), actions)
        self.assertEqual(ACTIONS[actions[0]], "IGNORE")
        self.assertEqual(cost, 0.0)


if __name__ == "__main__":
    unittest.main()
