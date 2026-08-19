from __future__ import annotations

import unittest

import torch
from transformers.cache_utils import DynamicCache

from interventions.four_state import LayerContext, _write_state
from interventions.read_path import _subtract_read_delta
from interventions.prompt_cache import clone_dynamic_cache


class PromptCacheTests(unittest.TestCase):
    def test_write_replacement_preserves_text_and_replaces_visual_residual(self) -> None:
        pre = torch.tensor([[[1.0], [2.0], [3.0]]])
        full = torch.tensor([[[4.0], [5.0], [6.0]]])
        current = torch.tensor([[[4.0], [50.0], [60.0]]])
        null_delta = torch.tensor([[[0.5], [0.0], [0.0]]])
        context = LayerContext(0, pre, full, {})
        visual = torch.tensor([[True, False, False]])

        replaced, _ = _write_state(current, context, visual, "replace", null_delta)

        self.assertEqual(float(replaced[0, 0, 0]), 1.5)
        self.assertEqual(float(replaced[0, 1, 0]), 50.0)
        self.assertEqual(float(replaced[0, 2, 0]), 60.0)

    def test_write_subtraction_preserves_current_text_and_removes_visual_delta(self) -> None:
        pre = torch.tensor([[[1.0], [2.0], [3.0]]])
        full = torch.tensor([[[4.0], [5.0], [6.0]]])
        current = torch.tensor([[[4.0], [50.0], [60.0]]])
        removal = torch.tensor([[[0.5], [0.0], [0.0]]])
        context = LayerContext(0, pre, full, {})
        visual = torch.tensor([[True, False, False]])

        subtracted, _ = _write_state(current, context, visual, "subtract", removal)

        self.assertEqual(float(subtracted[0, 0, 0]), 3.5)
        self.assertEqual(float(subtracted[0, 1, 0]), 50.0)
        self.assertEqual(float(subtracted[0, 2, 0]), 60.0)

    def test_read_subtraction_preserves_visual_rows_and_removes_text_delta(self) -> None:
        actual = torch.tensor([[[4.0], [5.0], [6.0]]])
        removal = torch.tensor([[[0.0], [0.5], [1.5]]])
        visual = torch.tensor([[True, False, False]])

        subtracted = _subtract_read_delta(actual, removal, visual)

        self.assertEqual(float(subtracted[0, 0, 0]), 4.0)
        self.assertEqual(float(subtracted[0, 1, 0]), 4.5)
        self.assertEqual(float(subtracted[0, 2, 0]), 4.5)

    def test_clone_can_keep_only_prefix_layers_and_is_independent(self) -> None:
        source = DynamicCache()
        for layer in range(3):
            key = torch.full((1, 2, 4, 3), float(layer))
            value = key + 10
            source.update(key, value, layer)

        cloned = clone_dynamic_cache(source, through_layer_exclusive=2)

        self.assertEqual(len(cloned.key_cache), 2)
        self.assertTrue(torch.equal(cloned.key_cache[1], source.key_cache[1]))
        cloned.key_cache[1].add_(5)
        self.assertFalse(torch.equal(cloned.key_cache[1], source.key_cache[1]))
        self.assertEqual(cloned.get_seq_length(0), 4)

    def test_full_clone_preserves_empty_layers(self) -> None:
        source = DynamicCache()
        source.update(torch.ones((1, 1, 2, 2)), torch.ones((1, 1, 2, 2)), 2)

        cloned = clone_dynamic_cache(source)

        self.assertEqual(len(cloned.key_cache), 3)
        self.assertEqual(cloned.key_cache[0].numel(), 0)
        self.assertTrue(torch.equal(cloned.key_cache[2], source.key_cache[2]))


if __name__ == "__main__":
    unittest.main()
