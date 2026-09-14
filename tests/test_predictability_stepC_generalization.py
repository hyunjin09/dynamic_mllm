import unittest

import numpy as np
import torch

from dense_failure_stage2.predictability_generalization import (
    assign_clusters_to_folds,
    cosine_nearest,
    equal_frequency_bins,
    exact_knn_prediction,
    label_blind_inner_roles,
    last_token_pool,
    normalize_question,
    spherical_kmeans,
)


class PredictabilityStepCGeneralizationTests(unittest.TestCase):
    def test_normalization_and_last_token_pool(self):
        self.assertEqual(normalize_question("  full\u3000width １２  "), "full width 12")
        hidden = torch.arange(2 * 4 * 2).reshape(2, 4, 2)
        mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]])
        expected = torch.stack((hidden[0, 3], hidden[1, 1]))
        self.assertTrue(torch.equal(last_token_pool(hidden, mask), expected))

    def test_bins_are_balanced_and_stable_on_ties(self):
        bins = equal_frequency_bins([1.0] * 11, bins=5)
        self.assertEqual(bins.tolist(), [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    def test_nearest_and_knn_are_exact(self):
        reference = np.eye(3, dtype=np.float32)
        query = np.asarray([[1, 0, 0], [0, 0.8, 0.6]], dtype=np.float32)
        query /= np.linalg.norm(query, axis=1, keepdims=True)
        similarity, indices = cosine_nearest(query, reference, block_size=2)
        np.testing.assert_allclose(similarity, [1.0, 0.8])
        np.testing.assert_array_equal(indices, [0, 1])
        prediction = exact_knn_prediction(query, reference, [1, 0, 0.5], k=2)
        np.testing.assert_allclose(prediction, [0.5, 0.25])

    def test_spherical_kmeans_is_deterministic(self):
        values = np.asarray([[1, 0], [.9, .1], [-1, 0], [-.9, -.1]], dtype=np.float32)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        left = spherical_kmeans(values, clusters=2, seed=7)
        right = spherical_kmeans(values, clusters=2, seed=7)
        np.testing.assert_array_equal(left[0], right[0])
        self.assertEqual(len(set(left[0].tolist())), 2)

    def test_split_helpers_are_label_blind_and_group_disjoint(self):
        cluster_rows = [
            {"cluster_id": c, "dataset": d, "groups": c + 1, "p90_groups": c % 2}
            for c in range(6) for d in ("gqa", "chartqa", "textvqa")
        ]
        assignment = assign_clusters_to_folds(cluster_rows, folds=3, seed=1)
        self.assertEqual(set(assignment), set(range(6)))
        self.assertEqual(set(assignment.values()), {0, 1, 2})
        rows = [
            {"uid": "a", "image_group_id": "g1"},
            {"uid": "b", "image_group_id": "g1"},
            {"uid": "c", "image_group_id": "g2"},
            {"uid": "d", "image_group_id": "g3"},
        ]
        roles = label_blind_inner_roles(
            rows,
            eligible_train_uids={"a", "b", "c"},
            test_uids={"d"},
            calibration_fraction=0.5,
            seed=2,
        )
        by_uid = {row["uid"]: row["role"] for row in roles}
        self.assertEqual(by_uid["a"], by_uid["b"])
        self.assertEqual(by_uid["d"], "outer_test")
        self.assertIn(by_uid["c"], {"fit", "calibration"})


if __name__ == "__main__":
    unittest.main()
