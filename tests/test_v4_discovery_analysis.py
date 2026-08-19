import unittest

import pandas as pd

from tools.research_analysis.v4.analyze_discovery import ACTIONS, LAYERS, build_image_layer, effects


class V4DiscoveryAnalysisTests(unittest.TestCase):
    def test_effect_orientation(self):
        q = {"IGNORE": 1.0, "READ_ONLY": 3.0, "WRITE_ONLY": 4.0, "FULL": 10.0}
        self.assertEqual(
            {
                "read_w0": 2.0,
                "read_w1": 6.0,
                "write_r0": 3.0,
                "write_r1": 7.0,
                "interaction": 4.0,
            },
            effects(q),
        )

    def test_image_metrics_use_epsilon_best_transfer_rule(self):
        rows = []
        for image_index in range(120):
            for layer in LAYERS:
                for question_index in (0, 1):
                    q_mean = (
                        {"IGNORE": 1.0, "READ_ONLY": 0.0, "WRITE_ONLY": 0.0, "FULL": 0.0}
                        if question_index == 0
                        else {"IGNORE": 0.0, "READ_ONLY": 0.0, "WRITE_ONLY": 0.0, "FULL": 1.0}
                    )
                    row = {
                        "image_id": str(image_index),
                        "image_index": image_index,
                        "question_index": question_index,
                        "layer": layer,
                        "pair_stratum": "different_evidence" if image_index < 60 else "matched_comparison",
                        "different_evidence": image_index < 60,
                        "official_paraphrase": False,
                        "pair_match_distance": 0.0,
                        "answer_token_length": 1,
                        "answer_format": "boolean",
                        "prompt_token_length": 100,
                        "semantic_program_depth": 1,
                        "question_structural_type": "verify",
                        "question_semantic_type": "attribute",
                        "question_detailed_type": "exist",
                    }
                    for metric in ("mean", "sequence"):
                        for action in ACTIONS:
                            row[f"q_{metric}_{action.lower()}"] = q_mean[action]
                        for name, value in effects(q_mean).items():
                            row[f"{name}_{metric}"] = value
                        row[f"epsilon_best_{metric}"] = "IGNORE" if question_index == 0 else "FULL"
                        row[f"exact_best_{metric}"] = "IGNORE" if question_index == 0 else "FULL"
                    rows.append(row)
        image_layer = build_image_layer(pd.DataFrame(rows), {"mean": 1e-6, "sequence": 1e-5})
        first = image_layer.iloc[0]
        self.assertEqual(1, first["robust_best_action_disagreement_mean"])
        self.assertAlmostEqual(1.0, first["transfer_regret_mean"])
        self.assertAlmostEqual(0.5, first["query_oracle_gap_mean"])


if __name__ == "__main__":
    unittest.main()
