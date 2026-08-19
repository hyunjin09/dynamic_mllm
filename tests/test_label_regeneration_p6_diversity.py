import unittest

from label_regeneration.diversity import (
    aggregate_diversity,
    mask_run_lengths,
    summarize_record_diversity,
)


def candidate(route_id, mask):
    mask = list(mask)
    return {
        "route_id": route_id,
        "visual_on_mask": mask,
        "num_visual_on_layers": sum(mask),
        "num_visual_off_layers": 28 - sum(mask),
        "num_transitions": sum(mask[i] != mask[i - 1] for i in range(1, 28)),
        "hamming_distance_to_all_on": 28 - sum(mask),
        "result_correct": True,
    }


def record(dataset, status, masks):
    candidates = [candidate(f"r{i}", mask) for i, mask in enumerate(masks)]
    best = (
        min(candidates, key=lambda row: (row["num_visual_on_layers"], tuple(row["visual_on_mask"])))
        if candidates
        else None
    )
    return {
        "sample": {
            "uid": f"{dataset}:{status}:{len(masks)}",
            "benchmark": dataset,
            "sample_id": "sample",
            "image_group_id": "image",
            "current_all_on_status": status,
        },
        "successful_route_ids": [row["route_id"] for row in candidates],
        "best_sparse_success_route_id": best["route_id"] if best is not None else None,
        "candidate_executions": candidates,
    }


class P6DiversityTests(unittest.TestCase):
    def test_run_lengths_are_maximal_and_action_specific(self):
        mask = (1, 1, 1, 0, 0, 1, 0, 0) + (0,) * 20
        self.assertEqual(mask_run_lengths(mask), [(1, 3), (0, 2), (1, 1), (0, 22)])

    def test_exact_pairwise_and_minimum_route_hamming(self):
        all_on = (1,) * 28
        all_off = (0,) * 28
        alternating = tuple(i % 2 for i in range(28))
        row = summarize_record_diversity(record("gqa", "correct", [all_on, all_off, alternating]))
        # Pairwise distances: 28, 14, 14.
        self.assertEqual(row["pairwise_hamming_histogram"][14], 2)
        self.assertEqual(row["pairwise_hamming_histogram"][28], 1)
        self.assertEqual(row["pairwise_hamming"]["mean"], 56 / 3)
        self.assertEqual(row["minimum_visual_on_valid_route"], 0)
        self.assertEqual(row["hamming_to_minimum_histogram"][0], 1)
        self.assertEqual(row["hamming_to_minimum_histogram"][14], 1)
        self.assertEqual(row["hamming_to_minimum_histogram"][28], 1)

    def test_aggregate_retains_sample_balancing_and_zero_valid_samples(self):
        masks = [(1,) * 28, (0,) * 28]
        rows = [
            summarize_record_diversity(record("gqa", "correct", masks)),
            summarize_record_diversity(record("gqa", "wrong", [])),
            summarize_record_diversity(record("chartqa", "wrong", [(1, 0) * 14])),
        ]
        result = aggregate_diversity(rows)
        self.assertEqual(result["overall"]["samples"], 3)
        self.assertEqual(result["overall"]["samples_with_valid_routes"], 2)
        self.assertEqual(result["overall"]["valid_masks"], 3)
        self.assertEqual(result["overall"]["sample_balanced"]["mean_transition_count"]["count"], 2)
        self.assertEqual(result["by_dataset"]["gqa"]["zero_valid_samples"], 1)
        self.assertEqual(result["by_dataset_and_current_status"]["chartqa"]["wrong"]["valid_masks"], 1)


if __name__ == "__main__":
    unittest.main()
