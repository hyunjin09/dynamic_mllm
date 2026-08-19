from __future__ import annotations

import unittest
from collections import Counter

from audit.stage_b_manifest import select_balanced_unique_assets


class StageBManifestSelectionTests(unittest.TestCase):
    def test_selects_exact_balanced_cells_deterministically(self) -> None:
        rows_by_cell = {}
        for benchmark in ("gqa", "textvqa"):
            for bucket in ("complete_correct", "complete_wrong"):
                cell = (benchmark, bucket)
                rows_by_cell[cell] = [
                    {
                        "id": f"{benchmark}:{bucket}:{index}",
                        "benchmark": benchmark,
                        "bucket": bucket,
                        "source_asset_id": f"{benchmark}:{bucket}:image:{index}",
                    }
                    for index in range(8)
                ]

        first = select_balanced_unique_assets(rows_by_cell, quota_per_cell=4, seed=17)
        second = select_balanced_unique_assets(rows_by_cell, quota_per_cell=4, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertEqual(
            Counter((row["benchmark"], row["bucket"]) for row in first),
            {
                ("gqa", "complete_correct"): 4,
                ("gqa", "complete_wrong"): 4,
                ("textvqa", "complete_correct"): 4,
                ("textvqa", "complete_wrong"): 4,
            },
        )
        self.assertEqual(len({row["source_asset_id"] for row in first}), 16)
        self.assertTrue(all("selection_hash" in row for row in first))
        self.assertTrue(all("selection_rank_in_cell" in row for row in first))

    def test_excludes_ids_and_shared_assets_across_cells(self) -> None:
        rows_by_cell = {
            ("gqa", "complete_correct"): [
                {"id": "excluded", "benchmark": "gqa", "bucket": "complete_correct", "source_asset_id": "a"},
                {"id": "keep-1", "benchmark": "gqa", "bucket": "complete_correct", "source_asset_id": "shared"},
                {"id": "keep-2", "benchmark": "gqa", "bucket": "complete_correct", "source_asset_id": "b"},
            ],
            ("gqa", "complete_wrong"): [
                {"id": "keep-3", "benchmark": "gqa", "bucket": "complete_wrong", "source_asset_id": "shared"},
                {"id": "keep-4", "benchmark": "gqa", "bucket": "complete_wrong", "source_asset_id": "c"},
                {"id": "keep-5", "benchmark": "gqa", "bucket": "complete_wrong", "source_asset_id": "d"},
            ],
        }

        selected = select_balanced_unique_assets(
            rows_by_cell, quota_per_cell=1, seed=3, excluded_ids={"excluded"}
        )

        self.assertEqual(len(selected), 2)
        self.assertNotIn("excluded", {row["id"] for row in selected})
        self.assertEqual(len({row["source_asset_id"] for row in selected}), 2)

    def test_raises_when_a_quota_cannot_be_filled(self) -> None:
        rows_by_cell = {
            ("gqa", "complete_correct"): [
                {"id": "one", "benchmark": "gqa", "bucket": "complete_correct", "source_asset_id": "same"},
            ],
            ("gqa", "complete_wrong"): [
                {"id": "two", "benchmark": "gqa", "bucket": "complete_wrong", "source_asset_id": "same"},
            ],
        }

        with self.assertRaisesRegex(ValueError, "quota"):
            select_balanced_unique_assets(rows_by_cell, quota_per_cell=1, seed=9)

    def test_uses_local_image_path_when_source_asset_id_is_missing(self) -> None:
        rows_by_cell = {
            ("gqa", "complete_correct"): [
                {"id": "one", "benchmark": "gqa", "bucket": "complete_correct", "source_asset_id": None, "local_image_path": "/images/shared.jpg"},
                {"id": "fallback", "benchmark": "gqa", "bucket": "complete_correct", "source_asset_id": None, "local_image_path": "/images/other.jpg"},
            ],
            ("gqa", "complete_wrong"): [
                {"id": "two", "benchmark": "gqa", "bucket": "complete_wrong", "source_asset_id": None, "local_image_path": "/images/shared.jpg"},
                {"id": "three", "benchmark": "gqa", "bucket": "complete_wrong", "source_asset_id": None, "local_image_path": "/images/third.jpg"},
            ],
        }

        selected = select_balanced_unique_assets(rows_by_cell, quota_per_cell=1, seed=5)

        self.assertEqual(len({row["selection_asset_key"] for row in selected}), 2)

    def test_does_not_select_the_same_sample_id_twice(self) -> None:
        rows_by_cell = {
            ("gqa", "complete_correct"): [
                {"id": "duplicate", "benchmark": "gqa", "bucket": "complete_correct", "source_asset_id": "a"},
            ],
            ("gqa", "complete_wrong"): [
                {"id": "duplicate", "benchmark": "gqa", "bucket": "complete_wrong", "source_asset_id": "b"},
                {"id": "other", "benchmark": "gqa", "bucket": "complete_wrong", "source_asset_id": "c"},
            ],
        }

        selected = select_balanced_unique_assets(rows_by_cell, quota_per_cell=1, seed=1)

        self.assertEqual(len({row["id"] for row in selected}), 2)


if __name__ == "__main__":
    unittest.main()
