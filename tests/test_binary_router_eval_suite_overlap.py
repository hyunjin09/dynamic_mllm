import unittest

from tools.audit_binary_router_eval_suite_overlap import summarize_overlap


class EvalSuiteOverlapAuditTest(unittest.TestCase):
    def test_exact_image_question_and_image_only_overlap_are_separate(self) -> None:
        mcts = [
            {
                "uid": "mcts-a",
                "sample_id": "a",
                "question": "What color?",
                "prompt": "What color? Answer briefly.",
                "computed_image_sha256": "hash-a",
            }
        ]
        evaluation = [
            {
                "uid": "eval-a",
                "sample_id": "eval-a",
                "benchmark": "textvqa",
                "question": "What color?",
                "prompt": "What color? Answer briefly.",
                "image_content_sha256": "hash-a",
            },
            {
                "uid": "eval-b",
                "sample_id": "eval-b",
                "benchmark": "textvqa",
                "question": "What word?",
                "prompt": "What word? Answer briefly.",
                "image_content_sha256": "hash-a",
            },
        ]

        result = summarize_overlap(evaluation, mcts)["textvqa"]

        self.assertEqual(result["records_with_any_mcts_image"], 2)
        self.assertEqual(result["shared_unique_image_hashes"], 1)
        self.assertEqual(result["exact_image_question_pairs"], 1)
        self.assertEqual(result["uid_overlap"], 0)

    def test_multi_image_record_counts_as_image_overlap_but_not_single_pair(self) -> None:
        mcts = [
            {
                "uid": "mcts-a",
                "sample_id": "a",
                "question": "Question",
                "prompt": "Prompt",
                "computed_image_sha256": "hash-a",
            }
        ]
        evaluation = [
            {
                "uid": "eval-a",
                "sample_id": "eval-a",
                "benchmark": "mmmu",
                "instruction_text_chunks": ["Question"],
                "prompt": "Prompt",
                "image_content_sha256s": ["hash-a", "hash-b"],
            }
        ]

        result = summarize_overlap(evaluation, mcts)["mmmu"]

        self.assertEqual(result["records_with_any_mcts_image"], 1)
        self.assertEqual(result["shared_unique_image_hashes"], 1)
        self.assertEqual(result["exact_image_question_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
