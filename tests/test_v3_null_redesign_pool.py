import unittest

from tools.research_analysis.v3.build_null_redesign_calibration_pool import rank, validate_records


class NullRedesignPoolTests(unittest.TestCase):
    def test_rank_is_deterministic_and_dataset_specific(self):
        self.assertEqual(rank(7, "gqa", "x"), rank(7, "gqa", "x"))
        self.assertNotEqual(rank(7, "gqa", "x"), rank(7, "textvqa", "x"))

    def test_validate_records_rejects_answer_fields(self):
        rows = [
            {
                "dataset": "gqa",
                "id": "g",
                "image_id": "gqa:1",
                "local_image_path": __file__,
                "prompt": "q\nAnswer the question using a single word or phrase.",
                "answer": "leak",
            },
            {
                "dataset": "textvqa",
                "id": "t",
                "image_id": "textvqa:1",
                "local_image_path": __file__,
                "prompt": "q\nAnswer the question using a single word or phrase.",
            },
        ]
        with self.assertRaisesRegex(RuntimeError, "Answer fields leaked"):
            validate_records(rows, 1)


if __name__ == "__main__":
    unittest.main()
