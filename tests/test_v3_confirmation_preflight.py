import unittest

import torch
from transformers.cache_utils import DynamicCache

from tools.research_analysis.v3.confirmation_preflight import (
    choose_one_record_per_image,
    deterministic_rank,
    image_group_summary,
    normal_mde,
    reserve_multi_question_groups,
    right_pad_prompt_inputs,
    search_cells,
)
from interventions.prompt_cache import truncate_dynamic_cache


class ConfirmationPreflightTests(unittest.TestCase):
    def test_deterministic_rank_is_stable_and_seeded(self):
        first = deterministic_rank("gqa:1", 2026080602)
        self.assertEqual(first, deterministic_rank("gqa:1", 2026080602))
        self.assertNotEqual(first, deterministic_rank("gqa:1", 2026080603))

    def test_choose_one_record_per_image_is_unique_and_deterministic(self):
        rows = [
            {"id": "q1", "image_id": "i1"},
            {"id": "q2", "image_id": "i1"},
            {"id": "q3", "image_id": "i2"},
        ]
        selected = choose_one_record_per_image(rows, count=2, seed=9)
        self.assertEqual(2, len(selected))
        self.assertEqual(2, len({row["image_id"] for row in selected}))
        self.assertEqual(selected, choose_one_record_per_image(rows, count=2, seed=9))

    def test_reserve_groups_excludes_confirmation_images(self):
        rows = [
            {"id": "q1", "image_id": "i1"},
            {"id": "q2", "image_id": "i1"},
            {"id": "q3", "image_id": "i2"},
            {"id": "q4", "image_id": "i2"},
            {"id": "q5", "image_id": "i3"},
        ]
        reserved = reserve_multi_question_groups(
            rows, excluded_image_ids={"i1"}, group_count=1, seed=12
        )
        self.assertEqual(["i2"], list(reserved))
        self.assertEqual(["q3", "q4"], [row["id"] for row in reserved["i2"]])

    def test_group_summary_counts_questions_per_image(self):
        rows = [
            {"id": "q1", "image_id": "i1"},
            {"id": "q2", "image_id": "i1"},
            {"id": "q3", "image_id": "i2"},
        ]
        summary = image_group_summary(rows)
        self.assertEqual(2, summary["image_count"])
        self.assertEqual(1, summary["multi_question_image_count"])
        self.assertEqual({"1": 1, "2": 1}, summary["questions_per_image_histogram"])

    def test_search_budget_is_layer_action_cartesian_product(self):
        cells = search_cells([0, 4], ["IGNORE", "READ_ONLY", "WRITE_ONLY"])
        self.assertEqual(6, len(cells))
        self.assertIn({"layer": 4, "action": "WRITE_ONLY"}, cells)

    def test_normal_mde_uses_cluster_count_and_one_sided_alpha(self):
        value = normal_mde(sd=0.5, clusters=800, alpha=0.05 / 3, power=0.8)
        self.assertGreater(value, 0.04)
        self.assertLess(value, 0.06)

    def test_right_pad_prompt_inputs_only_extends_token_rows(self):
        inputs = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "pixel_values": torch.ones((3, 4)),
        }
        padded = right_pad_prompt_inputs(inputs, target_length=4, pad_token_id=9)
        self.assertEqual([[1, 2, 9, 9]], padded["input_ids"].tolist())
        self.assertEqual([[1, 1, 0, 0]], padded["attention_mask"].tolist())
        self.assertTrue(torch.equal(inputs["pixel_values"], padded["pixel_values"]))

    def test_truncate_dynamic_cache_removes_only_right_padding_rows(self):
        cache = DynamicCache()
        key = torch.arange(1 * 2 * 4 * 3).reshape(1, 2, 4, 3)
        value = key + 100
        cache.update(key, value, 0)
        truncated = truncate_dynamic_cache(cache, 2)
        self.assertTrue(torch.equal(truncated.key_cache[0], key[..., :2, :]))
        self.assertTrue(torch.equal(truncated.value_cache[0], value[..., :2, :]))
        self.assertEqual(4, cache.key_cache[0].shape[-2])


if __name__ == "__main__":
    unittest.main()
