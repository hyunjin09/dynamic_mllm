from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from audit.sample_manifest import resolve_local_image, select_stage_a_samples
from interventions.read_path import apply_reference_attention_mask, quantized_path_subtraction
from experiments.stage_a_validity import finite_and_abs_max, max_abs_difference, rms_ratio
from scoring.benchmark_metrics import (
    anls,
    normalize_textvqa,
    relaxed_accuracy,
    score_record,
    textvqa_consensus,
)
from scoring.option_scores import answer_margin, score_appended_content


class BenchmarkMetricTests(unittest.TestCase):
    def test_relaxed_accuracy(self) -> None:
        self.assertEqual(relaxed_accuracy("104", "100"), 1.0)
        self.assertEqual(relaxed_accuracy("106", "100"), 0.0)
        self.assertEqual(relaxed_accuracy("46", "0.46"), 0.0)
        self.assertEqual(relaxed_accuracy("Yes.", "yes"), 1.0)

    def test_anls_threshold(self) -> None:
        self.assertEqual(anls("draft", ["draft"]), 1.0)
        self.assertEqual(anls("zzzz", ["draft"]), 0.0)

    def test_textvqa_consensus(self) -> None:
        self.assertEqual(textvqa_consensus("Nokia", ["nokia"] * 3), 1.0)
        self.assertEqual(textvqa_consensus("nokia", ["nokia", "other"]), 1 / 3)

    def test_textvqa_evalai_normalization(self) -> None:
        self.assertEqual(normalize_textvqa("Two, cats!"), "2 cats")
        self.assertEqual(normalize_textvqa("dont"), "don't")
        self.assertEqual(normalize_textvqa("1,000.5"), "1000.5")

    def test_score_record_dispatch(self) -> None:
        record = {"metric_name": "exact_match_ignore_case_punctuation", "answer": "A bed"}
        self.assertEqual(score_record(record, "a bed."), 1.0)


class SampleManifestTests(unittest.TestCase):
    def test_local_path_and_candidate_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images" / "gqa"
            image_dir.mkdir(parents=True)
            for name in ("first.jpg", "second.jpg"):
                (image_dir / name).touch()
            manifest = root / "gqa_complete_correct_2000.jsonl"
            rows = [
                {
                    "id": "one",
                    "benchmark": "gqa",
                    "bucket": "complete_correct",
                    "answer": "bed",
                    "prediction": "bed",
                    "image_path": "/old/first.jpg",
                },
                {
                    "id": "two",
                    "benchmark": "gqa",
                    "bucket": "complete_correct",
                    "answer": "table",
                    "prediction": "chair",
                    "image_path": "/old/second.jpg",
                },
            ]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            selected = select_stage_a_samples(root, ["gqa"], ["complete_correct"], 2)
            self.assertEqual(Path(selected[0]["local_image_path"]), image_dir / "first.jpg")
            self.assertEqual(selected[0]["parity_only_candidates"], ["bed", "table"])
            self.assertEqual(resolve_local_image(root, rows[1]), image_dir / "second.jpg")


class OptionScoreTests(unittest.TestCase):
    def test_score_appended_content(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 0]])
        logits = torch.zeros((1, 4, 4))
        logits[0, 1, 3] = 2.0
        logits[0, 2, 0] = 1.0
        result = score_appended_content(logits, input_ids, prompt_length=2, answer_length=2)
        self.assertEqual(result.token_ids, [3, 0])
        self.assertEqual(len(result.token_logprobs), 2)
        self.assertEqual(answer_margin(-1.0, [-2.0, -3.0]), 1.0)


class ReadPathPrecisionTests(unittest.TestCase):
    def test_reference_mask_recreates_sdpa_causality_when_mask_is_none(self) -> None:
        weights = torch.zeros((1, 1, 2, 4), dtype=torch.float32)
        masked = apply_reference_attention_mask(weights, None, query_start=1, key_length=4)

        self.assertEqual(masked[0, 0, 0, :2].tolist(), [0.0, 0.0])
        self.assertLess(masked[0, 0, 0, 2].item(), -1e20)
        self.assertEqual(masked[0, 0, 1, :3].tolist(), [0.0, 0.0, 0.0])
        self.assertLess(masked[0, 0, 1, 3].item(), -1e20)

    def test_bfloat16_add_back_reconstructs_the_executed_off_state(self) -> None:
        full = torch.tensor([[[8.0, 0.5, -3.0], [16.0, 1.0, -0.25]]], dtype=torch.bfloat16)
        ideal = torch.tensor([[[0.0, 0.0, 0.0], [0.07, 0.003, -0.02]]], dtype=torch.bfloat16)
        affected = torch.tensor([[False, True]])

        off, effective_delta, reconstructed, precision = quantized_path_subtraction(
            full, ideal, affected
        )

        self.assertTrue(torch.equal(reconstructed, full))
        self.assertTrue(torch.equal(off[:, 0], full[:, 0]))
        self.assertTrue(torch.equal((off.float() + effective_delta).to(torch.bfloat16), full))
        self.assertGreater(precision["adjustment_max_abs"], 0.0)
        self.assertLessEqual(precision["half_ulp_ratio_max"], 1.000001)

    def test_chunked_tensor_diagnostics_match_direct_reductions(self) -> None:
        reference = torch.arange(24, dtype=torch.bfloat16).reshape(1, 3, 8)
        candidate = reference.clone()
        candidate[0, 1, 3] += 0.5

        self.assertEqual(
            max_abs_difference(candidate, reference, chunk_size=5),
            float((candidate.float() - reference.float()).abs().max().item()),
        )
        expected_ratio = float(
            ((candidate.float() - reference.float()).square().sum()
            / reference.float().square().sum()).sqrt().item()
        )
        self.assertAlmostEqual(rms_ratio(candidate, reference, chunk_size=5), expected_ratio)
        self.assertEqual(finite_and_abs_max(candidate, chunk_size=5), (True, 23.0))


if __name__ == "__main__":
    unittest.main()
