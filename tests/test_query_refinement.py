import unittest

import torch
import pandas as pd

from tools.research_analysis.query_refinement.analyze_gqa import safe_spearman
from interventions.query_refinement import (
    conditioned_visual_attention_mask,
    minimal_contextual_question_span,
    replay_compute_macs,
)


class _Encoded(dict):
    pass


class _FastTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        ids = [10, 99, 11, 12, 13]
        offsets = [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6)]
        result = _Encoded(input_ids=ids)
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class _SlowEncoding:
    def __init__(self, ids):
        self.input_ids = ids


class _SlowTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return _SlowEncoding([10, 99, 11, 12, 13])


class QueryRefinementTests(unittest.TestCase):
    def test_contextual_question_span_maps_expanded_image_placeholder(self):
        audit = minimal_contextual_question_span(
            prompt_text="aIhi!\n",
            question="hi!",
            fast_tokenizer=_FastTokenizer(),
            slow_tokenizer=_SlowTokenizer(),
            actual_input_ids=torch.tensor([[10, 99, 99, 99, 11, 12, 13]]),
            image_token_id=99,
        )
        self.assertEqual(audit["token_first"], 4)
        self.assertEqual(audit["token_last"], 5)
        self.assertEqual(audit["covered_text"], "hi!")
        self.assertEqual(torch.where(audit["mask"][0])[0].tolist(), [4, 5])

    def test_conditioned_mask_adds_only_visual_to_question_edges(self):
        minimum = torch.finfo(torch.float32).min
        native = torch.full((1, 1, 6, 6), minimum)
        for query in range(6):
            native[0, 0, query, : query + 1] = 0
        visual = torch.tensor([[False, True, True, False, False, False]])
        question = torch.tensor([[False, False, False, False, True, True]])
        replay, edge_count = conditioned_visual_attention_mask(native, visual, question)
        self.assertEqual(edge_count, 4)
        self.assertTrue(torch.equal(replay[0, 0, 1:3, 4:6], torch.zeros((2, 2))))
        unchanged = native.clone()
        unchanged[0, 0, 1:3, 4:6] = 0
        self.assertTrue(torch.equal(replay, unchanged))

    def test_replay_cost_depends_on_shape_not_conditioning_mask(self):
        first = replay_compute_macs(
            sequence_length=32,
            hidden_size=64,
            intermediate_size=128,
            num_visual_tokens=12,
            num_key_value_heads=2,
            num_attention_heads=8,
        )
        second = replay_compute_macs(
            sequence_length=32,
            hidden_size=64,
            intermediate_size=128,
            num_visual_tokens=12,
            num_key_value_heads=2,
            num_attention_heads=8,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["total_flops_two_per_mac"], 2 * first["total_macs"])

    def test_spearman_does_not_require_scipy(self):
        value = safe_spearman(pd.Series([1.0, 2.0, 3.0]), pd.Series([3.0, 2.0, 1.0]))
        self.assertAlmostEqual(value, -1.0)


if __name__ == "__main__":
    unittest.main()
