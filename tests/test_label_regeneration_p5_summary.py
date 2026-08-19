import unittest

from label_regeneration.summary import aggregate_summaries, summarize_record


def candidate(route_id, mask, correct):
    return {
        "route_id": route_id,
        "visual_on_mask": list(mask),
        "num_visual_on_layers": sum(mask),
        "result_correct": correct,
        "score": float(correct),
    }


def record(*, dataset, historical, current, successes, budget=200, all_off=False):
    all_on_mask = (1,) * 28
    all_off_mask = (0,) * 28
    candidates = [
        candidate("on", all_on_mask, current == "correct"),
        candidate("off", all_off_mask, all_off),
    ]
    for index, on_count in enumerate(successes):
        mask = (1,) * on_count + (0,) * (28 - on_count)
        candidates.append(candidate(f"success_{index}", mask, True))
    successful_ids = [row["route_id"] for row in candidates if row["result_correct"]]
    return {
        "sample": {
            "uid": f"{dataset}:sample_{historical}_{current}_{budget}",
            "benchmark": dataset,
            "sample_id": "sample",
            "image_group_id": "image",
            "historical_all_on_status": historical,
            "current_all_on_status": current,
            "current_all_on_score": float(current == "correct"),
            "current_all_on_prediction": "answer",
            "correctness_threshold": 1.0,
            "actual_text_tokens": 10,
            "actual_visual_tokens": 20,
            "actual_full_prompt_tokens": 30,
        },
        "root_route_id": "on",
        "all_off_route_id": "off",
        "best_sparse_success_route_id": successful_ids[-1] if successful_ids else None,
        "successful_route_ids": successful_ids,
        "candidate_executions": candidates,
        "mcts": {
            "requested_simulations": budget,
            "completed_simulations": budget,
            "extension_reason": "no_correcting_route_after_400" if budget == 600 else None,
        },
    }


class P5SummaryTests(unittest.TestCase):
    def test_current_wrong_counts_only_discovered_correct_routes_as_corrections(self):
        row = summarize_record(
            record(dataset="gqa", historical="correct", current="wrong", successes=[3, 5], budget=400)
        )
        self.assertEqual(row["valid_route_count"], 2)
        self.assertTrue(row["correction_found"])
        self.assertEqual(row["correcting_route_count"], 2)
        self.assertEqual(row["contract_drift"], "historical_correct_to_current_wrong")
        self.assertEqual(row["minimum_visual_on_valid_route"], 3)
        self.assertEqual(row["maximum_visual_off_valid_route"], 25)

    def test_current_correct_preservation_includes_all_on_anchor(self):
        row = summarize_record(
            record(dataset="textvqa", historical="wrong", current="correct", successes=[7])
        )
        self.assertEqual(row["valid_route_count"], 2)
        self.assertIsNone(row["correction_found"])
        self.assertEqual(row["minimum_visual_on_valid_route"], 7)
        self.assertEqual(row["contract_drift"], "historical_wrong_to_current_correct")

    def test_aggregate_separates_dataset_and_current_status(self):
        rows = [
            summarize_record(record(dataset="gqa", historical="correct", current="correct", successes=[4])),
            summarize_record(record(dataset="gqa", historical="wrong", current="wrong", successes=[], budget=600)),
            summarize_record(record(dataset="chartqa", historical="wrong", current="wrong", successes=[8], budget=400)),
        ]
        result = aggregate_summaries(rows)
        self.assertEqual(result["overall"]["samples"], 3)
        self.assertEqual(result["overall"]["current_all_on"]["correct"], 1)
        self.assertEqual(result["overall"]["correction"]["eligible_current_wrong"], 2)
        self.assertEqual(result["overall"]["correction"]["recovered"], 1)
        self.assertEqual(result["by_dataset"]["gqa"]["samples"], 2)
        self.assertEqual(result["by_dataset_and_current_status"]["chartqa"]["wrong"]["samples"], 1)


if __name__ == "__main__":
    unittest.main()
