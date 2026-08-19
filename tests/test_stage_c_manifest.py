from __future__ import annotations

import unittest

from audit.stage_c_manifest import (
    blocking_overlap_reasons,
    overlap_reasons,
    normalize_question,
    record_checksum,
    select_unique_images,
)


class StageCManifestTests(unittest.TestCase):
    def test_normalize_question_is_unicode_case_and_whitespace_stable(self) -> None:
        self.assertEqual(normalize_question("  WHAT\u00a0is  This? "), "what is this?")

    def test_overlap_uses_all_frozen_identifiers(self) -> None:
        discovery = {
            "ids": {"textvqa:1"},
            "question_ids": {1},
            "annotation_ids": {1},
            "image_ids": {"img-a"},
            "image_hashes": {"abc"},
            "normalized_image_paths": {"/data/a.jpg"},
            "normalized_questions": {"what is shown?"},
            "image_question_pairs": {("img-a", "what is shown?")},
        }
        candidate = {
            "id": "textvqa:2",
            "question_id": 2,
            "annotation_id": 2,
            "image_id": "img-b",
            "image_sha256": "abc",
            "local_image_path": "/data/b.jpg",
            "normalized_question": "what else?",
        }
        self.assertEqual(overlap_reasons(candidate, discovery), ["image_sha256"])

    def test_selection_is_deterministic_and_unique_by_image(self) -> None:
        rows = [
            {"id": "a", "image_id": "one"},
            {"id": "b", "image_id": "one"},
            {"id": "c", "image_id": "two"},
            {"id": "d", "image_id": "three"},
        ]
        first = select_unique_images(rows, count=3, seed=17)
        second = select_unique_images(list(reversed(rows)), count=3, seed=17)
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(len({row["image_id"] for row in first}), 3)

    def test_question_text_alone_is_reported_but_not_a_blocking_overlap(self) -> None:
        self.assertEqual(blocking_overlap_reasons(["normalized_question"]), [])
        self.assertEqual(
            blocking_overlap_reasons(["normalized_question", "image_id"]),
            ["image_id"],
        )

    def test_record_checksum_ignores_existing_checksum_field(self) -> None:
        row = {"id": "x", "question": "q", "record_sha256": "old"}
        self.assertEqual(record_checksum(row), record_checksum({"id": "x", "question": "q"}))


if __name__ == "__main__":
    unittest.main()
