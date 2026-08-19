import unittest

from tools.research_analysis.v4.build_discovery_manifest import (
    farthest_point_preflight,
    is_different_evidence,
    is_official_paraphrase,
    pair_match_distance,
    parse_id_list,
    semantic_object_ids,
)
from tools.research_analysis.v4.freeze_semantic_matching import minimum_cost_assignment
import numpy as np


def row(source_id="1", answer="yes", object_id="10", equivalent="[]"):
    return {
        "source_id": source_id,
        "answer": answer,
        "equivalent": equivalent,
        "types": {"structural": "verify", "semantic": "attr", "detailed": "x"},
        "annotations": {
            "question": [{"objectId": "0", "value": object_id}],
            "answer": [],
            "fullAnswer": [],
        },
        "semantic": [{"operation": "select", "argument": f"thing ({object_id})"}],
        "semanticStr": f"select: thing ({object_id})",
        "question": "Is it visible?",
        "answer_token_length": 1,
    }


class V4DiscoveryUtilsTests(unittest.TestCase):
    def test_parse_id_list_fails_closed(self):
        self.assertEqual({"1", "2"}, parse_id_list("['1', '2']"))
        self.assertEqual(set(), parse_id_list("not a list"))

    def test_semantic_object_ids_uses_annotation_values_and_program(self):
        self.assertEqual({"10"}, semantic_object_ids(row()))

    def test_different_evidence_requires_disjoint_resolved_objects(self):
        self.assertTrue(is_different_evidence(row(object_id="10"), row(object_id="11"), {"10", "11"}))
        self.assertFalse(is_different_evidence(row(object_id="10"), row(object_id="10"), {"10"}))
        self.assertFalse(is_different_evidence(row(object_id="10"), row(object_id="11"), {"10"}))

    def test_official_paraphrase_requires_link_answer_type_and_target(self):
        first = row(source_id="1", equivalent="['2']")
        second = row(source_id="2")
        self.assertTrue(is_official_paraphrase(first, second))
        second["answer"] = "no"
        self.assertFalse(is_official_paraphrase(first, second))

    def test_pair_match_distance_prefers_same_metadata(self):
        first = row()
        second = row(source_id="2")
        matched = pair_match_distance(first, second)
        second["types"] = {"structural": "query", "semantic": "rel", "detailed": "y"}
        self.assertGreater(pair_match_distance(first, second), matched)

    def test_farthest_point_selection_is_deterministic_and_unique(self):
        groups = [
            {"image_id": str(index), "common_prompt_token_length": 100 + index, "visual_token_count": 200 + index * 2}
            for index in range(20)
        ]
        first = farthest_point_preflight(groups, 12)
        self.assertEqual(first, farthest_point_preflight(groups, 12))
        self.assertEqual(12, len(set(first)))

    def test_minimum_cost_assignment_finds_global_optimum(self):
        cost = np.asarray([[9.0, 1.0, 9.0], [9.0, 9.0, 1.0], [1.0, 9.0, 9.0]])
        assignment = minimum_cost_assignment(cost)
        self.assertEqual([1, 2, 0], assignment)
        self.assertAlmostEqual(3.0, sum(cost[row, column] for row, column in enumerate(assignment)))


if __name__ == "__main__":
    unittest.main()
