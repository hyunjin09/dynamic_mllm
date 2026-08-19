import copy
import unittest

from experiments.analyze_wemath2pro_mcts_labels import ALL_OFF, ALL_ON, validate_record


def route(mask, index, correct=False, timeout=False):
    key = "".join(map(str, mask))
    return {
        "route_id": f"route_{index:04d}_{key}", "visual_on_mask": list(mask), "mask_key": key,
        "num_visual_on_layers": sum(mask), "correctness_threshold": 1.0,
        "score": float(correct), "result_correct": correct, "scoring_timed_out": timeout,
        "prediction": f"p{index}",
    }


class WeMathLabelAnalysisTest(unittest.TestCase):
    def fixture(self):
        manifest = {"uid": "wemath2pro:1", "image_content_sha256": "abc", "question": "q", "answer": "a"}
        rollout_masks = [tuple((value >> bit) & 1 for bit in range(28)) for value in range(1, 401)]
        candidates = [route(ALL_ON, 0), route(ALL_OFF, 1, True)] + [
            route(mask, index + 2) for index, mask in enumerate(rollout_masks)
        ]
        record = {
            "sample": {"uid": manifest["uid"], "image_content_sha256": "abc", "question": "q", "answer": "a",
                       "current_all_on_status": "wrong", "current_all_on_score": 0.0, "current_all_on_prediction": "p0"},
            "runtime": {"contract_sha256": "active"},
            "mcts": {
                "requested_simulations": 400,
                "completed_simulations": 400,
                "simulations": [{"evaluated_mask": list(mask)} for mask in rollout_masks],
                "evaluated_masks": [
                    {"visual_on_mask": candidate["visual_on_mask"], "route_id": candidate["route_id"]}
                    for candidate in candidates
                ],
            },
            "candidate_executions": candidates,
            "successful_route_ids": [candidates[1]["route_id"]],
        }
        return record, manifest

    def test_valid_terminal_record(self):
        record, manifest = self.fixture()
        candidates, valid = validate_record(record, manifest, accepted_contracts={"active"})
        self.assertEqual(len(candidates), 402)
        self.assertEqual(len(valid), 1)

    def test_allows_reused_rollout_evaluation(self):
        record, manifest = self.fixture()
        record["mcts"]["simulations"][-1]["evaluated_mask"] = list(ALL_OFF)
        removed = record["candidate_executions"].pop()
        record["mcts"]["evaluated_masks"] = [
            row for row in record["mcts"]["evaluated_masks"] if row["route_id"] != removed["route_id"]
        ]
        candidates, valid = validate_record(record, manifest, accepted_contracts={"active"})
        self.assertEqual(len(candidates), 401)
        self.assertEqual(len(valid), 1)

    def test_rejects_above_cap_or_inconsistent_timeout(self):
        record, manifest = self.fixture()
        record["mcts"] = {"requested_simulations": 600, "completed_simulations": 600}
        with self.assertRaises(ValueError):
            validate_record(record, manifest, accepted_contracts={"active"})
        record, manifest = self.fixture()
        record["candidate_executions"][2].update(scoring_timed_out=True, score=1.0, result_correct=True)
        record["successful_route_ids"].append(record["candidate_executions"][2]["route_id"])
        with self.assertRaises(ValueError):
            validate_record(record, manifest, accepted_contracts={"active"})


if __name__ == "__main__":
    unittest.main()
