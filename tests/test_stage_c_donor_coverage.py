from __future__ import annotations

import unittest

from tools.research_analysis.v2.stage_c_donor_coverage import coverage_summary, target_coverage
from tools.research_analysis.v2.stage_c_real_donor_amendment import (
    build_amended_match_rows,
    validate_frozen_match_row,
)
from nulls.structured_read import DonorMetadata


class StageCDonorCoverageTests(unittest.TestCase):
    def test_target_coverage_uses_frozen_composite_distance_and_eight_donors(self) -> None:
        target = DonorMetadata("target", "target-image", 1.0, 10, 20, 30)
        donors = [
            DonorMetadata(f"d{index}", f"image-{index}", 1.0 + index / 10, 10, 20, 30)
            for index in range(10)
        ]

        row = target_coverage(target, donors, seed=7, original_caliper=1.5)

        self.assertEqual(row["donors_within_original_caliper"], 6)
        self.assertAlmostEqual(row["nearest_distances"]["rank_1"], 1.0)
        self.assertAlmostEqual(row["nearest_distances"]["rank_7"], 1.6)
        self.assertAlmostEqual(row["nearest_distances"]["rank_8"], 1.7)
        self.assertAlmostEqual(row["nearest_distances"]["rank_9"], 1.8)
        self.assertEqual(len(row["nearest_eight_donors"]), 8)
        self.assertFalse(row["original_caliper_supplies_eight"])

    def test_target_coverage_excludes_same_sample_and_same_image(self) -> None:
        target = DonorMetadata("target", "target-image", 1.0, 10, 20, 30)
        donors = [
            DonorMetadata("target", "other-image", 1.0, 10, 20, 30),
            DonorMetadata("other", "target-image", 1.0, 10, 20, 30),
        ] + [
            DonorMetadata(f"d{index}", f"image-{index}", 1.0, 10, 20, 30)
            for index in range(9)
        ]

        row = target_coverage(target, donors, seed=7, original_caliper=1.5)

        selected_ids = {donor["sample_id"] for donor in row["nearest_eight_donors"]}
        selected_images = {donor["image_id"] for donor in row["nearest_eight_donors"]}
        self.assertNotIn("target", selected_ids)
        self.assertNotIn("target-image", selected_images)
        self.assertEqual(row["eligible_donor_pool_count"], 9)

    def test_coverage_summary_returns_exact_maximum_eighth_distance(self) -> None:
        rows = [
            {"id": "a", "nearest_distances": {"rank_8": 1.4}},
            {"id": "b", "nearest_distances": {"rank_8": 1.8}},
            {"id": "c", "nearest_distances": {"rank_8": 1.6}},
        ]

        summary = coverage_summary(rows, [1.5, 1.8, 1.75, 2.0], [0.0, 0.5, 1.0])

        self.assertEqual(summary["c_star"], 1.8)
        self.assertEqual(summary["c_star_target_ids"], ["b"])
        self.assertEqual(summary["targets_supported_by_caliper"]["1.5"], 1)
        self.assertEqual(summary["targets_supported_by_caliper"]["1.75"], 2)
        self.assertEqual(summary["targets_supported_by_caliper"]["1.8"], 3)

    def test_amended_index_preserves_supported_selection_and_marks_new_entries(self) -> None:
        target_rows = []
        for target_id, target_norm in (("supported", 1.3), ("widened", 2.0)):
            target = DonorMetadata(target_id, f"{target_id}-image", target_norm, 10, 20, 30)
            donors = [
                DonorMetadata(
                    f"d{index}", f"image-{index}", 1.0 + index / 10, 10, 20, 30
                )
                for index in range(10)
            ]
            row = target_coverage(target, donors, seed=7, original_caliper=1.5)
            row.update(
                {
                    "manifest_record_sha256": f"sha-{target_id}",
                    "layer": 0,
                    "hook": "decoder.layer.0.self_attn.output.postvisual_nonvisual_rows",
                }
            )
            target_rows.append(row)
        audit = {
            "c_star": 2.0,
            "original_caliper": 1.5,
            "targets": target_rows,
        }
        donors = [
            DonorMetadata(f"d{index}", f"image-{index}", 1.0 + index / 10, 10, 20, 30)
            for index in range(10)
        ]

        rows, summary = build_amended_match_rows(
            audit,
            donors,
            {"supported": 7, "widened": 7},
            draws=8,
            expected_target_count=2,
        )

        by_id = {row["id"]: row for row in rows}
        self.assertTrue(by_id["supported"]["original_caliper_supplies_eight"])
        self.assertFalse(by_id["widened"]["original_caliper_supplies_eight"])
        self.assertTrue(
            all(
                not donor["enters_due_to_amendment"]
                for donor in by_id["supported"]["selected_donors"]
            )
        )
        self.assertTrue(
            any(
                donor["enters_due_to_amendment"]
                for donor in by_id["widened"]["selected_donors"]
            )
        )
        self.assertEqual(summary["original_supported_target_count"], 1)
        self.assertEqual(summary["amended_supported_target_count"], 2)
        self.assertEqual(summary["selection_changed_for_original_supported_count"], 0)

        validated = validate_frozen_match_row(
            DonorMetadata("supported", "supported-image", 1.3, 10, 20, 30),
            donors,
            by_id["supported"],
            draws=8,
            seed=7,
            amended_caliper=2.0,
        )
        self.assertEqual(
            [donor.sample_id for donor in validated],
            [row["sample_id"] for row in by_id["supported"]["selected_donors"]],
        )

        with self.assertRaisesRegex(RuntimeError, "target norm"):
            validate_frozen_match_row(
                DonorMetadata("supported", "supported-image", 1.4, 10, 20, 30),
                donors,
                by_id["supported"],
                draws=8,
                seed=7,
                amended_caliper=2.0,
            )


if __name__ == "__main__":
    unittest.main()
