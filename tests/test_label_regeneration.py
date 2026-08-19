from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from label_regeneration.data import deterministic_smoke_records
from label_regeneration.mcts import GraphMCTS, MCTSConfig
from label_regeneration.runtime import build_native_processor_inputs
from label_regeneration.wemath import build_wemath_record, technical_invalid_reasons
from experiments.run_label_regeneration import find_completed_record, index_existing_records
from reference.dvr_qwen.eval_metrics import score_prediction


class RecordingProcessor:
    def __init__(self):
        self.messages = None
        self.processor_kwargs = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        return "frozen prompt"

    def __call__(self, **kwargs):
        self.processor_kwargs = kwargs
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }


class LabelRegenerationContractTest(unittest.TestCase):
    def test_wemath_manifest_record_freezes_direct_answer_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "000001.png"
            Image.new("RGB", (20, 30)).save(image_path)
            record = build_wemath_record(
                {
                    "idx": "1",
                    "question_id": "seed-1",
                    "question": "What is one half?",
                    "answer": r"\frac{1}{2}",
                    "difficulty": "base",
                    "knowledge points": ["Fractions"],
                },
                source_index=0,
                image_path=image_path,
                image_sha256="abc123",
            )

        self.assertEqual(record["uid"], "wemath2pro:1")
        self.assertEqual(record["benchmark"], "wemath2pro")
        self.assertEqual(record["metric_name"], "wemath2pro_mathruler_accuracy")
        self.assertEqual(record["correctness_threshold"], 1.0)
        self.assertEqual(record["max_new_tokens"], 96)
        self.assertEqual(record["image_group_id"], "sha256:abc123")
        self.assertEqual(
            record["prompt"],
            "What is one half?\nReturn only the final answer enclosed in "
            "<answer> and </answer>; do not include reasoning.",
        )

    def test_wemath_technical_invalid_rule_marks_only_missing_required_fields(self):
        base = {"question": "What is shown?", "answer": "2"}
        self.assertEqual(technical_invalid_reasons(base), [])
        self.assertEqual(
            technical_invalid_reasons({**base, "question": ""}),
            ["empty_question"],
        )
        self.assertEqual(
            technical_invalid_reasons({**base, "question": "", "answer": ""}),
            ["empty_question", "empty_answer"],
        )

    def test_wemath_metric_uses_official_answer_tag_and_math_equivalence(self):
        self.assertEqual(
            score_prediction(
                "wemath2pro_mathruler_accuracy",
                r"<answer>\frac{1}{2}</answer>",
                "0.5",
            ),
            1.0,
        )
        self.assertEqual(
            score_prediction("wemath2pro_mathruler_accuracy", "16", "16"),
            1.0,
        )
        self.assertEqual(
            score_prediction("wemath2pro_mathruler_accuracy", "15", "16"),
            0.0,
        )

    def test_resume_finds_complete_record_from_a_previous_shard_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            old_sample_root = (
                output_root
                / "raw_route_cache"
                / "shard_001_of_004"
                / "samples"
            )
            old_sample_root.mkdir(parents=True)
            old_record = old_sample_root / "sample.json"
            old_record.write_text(
                """{
                  "sample": {"uid": "gqa:sample"},
                  "runtime": {"contract_sha256": "frozen-contract"},
                  "candidate_executions": [],
                  "mcts": {"completed_simulations": 200, "requested_simulations": 200}
                }""",
                encoding="utf-8",
            )

            existing = index_existing_records(output_root)
            completed = find_completed_record(
                existing,
                filename="sample.json",
                uid="gqa:sample",
                contract_hash="frozen-contract",
            )

        self.assertEqual(completed, old_record)

    def test_native_processor_inputs_do_not_set_a_visual_token_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "fixture.png"
            Image.new("RGB", (32, 24)).save(image_path)
            processor = RecordingProcessor()
            inputs, metadata = build_native_processor_inputs(
                processor,
                {"local_image_path": str(image_path), "prompt": "What is shown?"},
                torch.device("cpu"),
            )

        self.assertEqual(inputs["input_ids"].device.type, "cpu")
        self.assertNotIn("max_pixels", processor.processor_kwargs)
        self.assertEqual(
            processor.messages[0]["content"][0],
            {"type": "image", "image": str(image_path)},
        )
        self.assertIsNone(metadata["custom_max_image_tokens"])
        self.assertEqual(metadata["original_image_dimensions"], [32, 24])

    def test_smoke_selection_is_balanced_and_deterministic(self):
        rows = [
            {"uid": f"{dataset}:{index}", "benchmark": dataset}
            for dataset in ("gqa", "textvqa", "chartqa")
            for index in range(20)
        ]

        first = deterministic_smoke_records(rows, per_dataset=5, seed=17)
        second = deterministic_smoke_records(rows, per_dataset=5, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 15)
        self.assertEqual(
            {dataset: sum(row["benchmark"] == dataset for row in first) for dataset in ("gqa", "textvqa", "chartqa")},
            {"gqa": 5, "textvqa": 5, "chartqa": 5},
        )


class GraphMCTSTest(unittest.TestCase):
    def test_retains_anchors_positive_and_negative_routes(self):
        calls = []

        def evaluate(mask, source):
            calls.append((mask, source))
            correct = sum(mask) <= 2
            return {
                "route_id": f"route_{len(calls)}",
                "reward": float(correct),
                "score": float(correct),
                "result_correct": correct,
            }

        search = GraphMCTS(evaluate, MCTSConfig(num_layers=4, seed=7))
        search.evaluate_anchors()
        search.run(20)
        result = search.result(requested_simulations=20, extension_reason=None)

        self.assertEqual(result["completed_simulations"], 20)
        self.assertGreaterEqual(len(result["evaluated_masks"]), 2)
        self.assertTrue(any(row["result_correct"] for row in result["evaluated_masks"]))
        self.assertTrue(any(not row["result_correct"] for row in result["evaluated_masks"]))
        self.assertEqual(result["best_mask"], min(result["successful_masks"], key=lambda m: (sum(m), m)))

    def test_is_deterministic_for_fixed_seed(self):
        def build():
            def evaluate(mask, source):
                del source
                score = float(mask[0] == 0)
                return {
                    "route_id": "".join(map(str, mask)),
                    "reward": score,
                    "score": score,
                    "result_correct": bool(score),
                }

            search = GraphMCTS(evaluate, MCTSConfig(num_layers=5, seed=13))
            search.evaluate_anchors()
            search.run(30)
            return search.result(requested_simulations=30, extension_reason=None)

        first = build()
        second = build()
        self.assertEqual(first["simulations"], second["simulations"])
        self.assertEqual(first["evaluated_masks"], second["evaluated_masks"])


if __name__ == "__main__":
    unittest.main()
