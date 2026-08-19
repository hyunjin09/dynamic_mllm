import random
import unittest

from label_regeneration.derived import (
    canonical_segment_targets,
    select_diverse_valid_routes,
    single_best_valid_route,
)


def _candidate(mask: str, route_id: str | None = None) -> dict:
    values = [int(bit) for bit in mask]
    return {
        "route_id": route_id or f"route_{mask}",
        "mask_key": mask,
        "visual_on_mask": values,
        "num_visual_on_layers": sum(values),
        "num_visual_off_layers": len(values) - sum(values),
        "num_transitions": sum(left != right for left, right in zip(values, values[1:])),
        "score": 1.0,
        "reward": 1.0,
        "result_correct": True,
        "correctness_threshold": 1.0,
    }


class LabelRegenerationP8DerivedTest(unittest.TestCase):
    def test_under_cap_keeps_every_unique_valid_route(self):
        routes = [_candidate("1111"), _candidate("0000"), _candidate("1010")]
        selected = select_diverse_valid_routes(routes, limit=50, seed=7, uid="sample")
        self.assertEqual({route["mask_key"] for route in selected}, {"1111", "0000", "1010"})
        self.assertEqual(len(selected), len(routes))

    def test_over_cap_is_deterministic_and_preserves_required_anchors(self):
        routes = [_candidate(f"{value:08b}") for value in range(256)]
        shuffled = list(routes)
        random.Random(11).shuffle(shuffled)
        first = select_diverse_valid_routes(routes, limit=50, seed=19, uid="sample")
        second = select_diverse_valid_routes(shuffled, limit=50, seed=19, uid="sample")
        first_keys = [route["mask_key"] for route in first]
        second_keys = [route["mask_key"] for route in second]
        self.assertEqual(first_keys, second_keys)
        self.assertEqual(len(first_keys), 50)
        self.assertEqual(len(set(first_keys)), 50)
        self.assertIn("00000000", first_keys)
        self.assertIn("11111111", first_keys)
        self.assertEqual({sum(route["visual_on_mask"]) for route in first}, set(range(9)))

    def test_single_best_uses_minimum_on_then_lexical_mask(self):
        routes = [_candidate("1100"), _candidate("0011"), _candidate("1110")]
        self.assertEqual(single_best_valid_route(routes)["mask_key"], "0011")
        self.assertIsNone(single_best_valid_route([]))

    def test_canonical_segments_reconstruct_the_complete_mask(self):
        mask = [1, 1, 0, 0, 1, 0, 0]
        targets = canonical_segment_targets(mask)
        self.assertEqual(targets["segment_starts"], [0, 2, 4, 5])
        self.assertEqual(targets["segment_actions"], [1, 0, 1, 0])
        self.assertEqual(targets["boundary_targets"], [0, 0, 1, 0, 1, 1, 0])
        self.assertEqual(targets["operation_targets"], [1, -100, 0, -100, 1, 0, -100])


if __name__ == "__main__":
    unittest.main()
