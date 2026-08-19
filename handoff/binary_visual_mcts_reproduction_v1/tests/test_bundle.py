from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from binary_policy.actions import normalize_visual_on_mask
from label_regeneration.mcts import GraphMCTS, MCTSConfig
from reference.dvr_qwen.eval_metrics import score_prediction


class BundleUnitTests(unittest.TestCase):
    def test_unrestricted_graph_mcts_is_deterministic(self):
        def evaluator(mask, source):
            key = "".join(map(str, mask))
            correct = mask == (0, 1, 0, 1)
            return {
                "route_id": f"route_{key}",
                "score": float(correct),
                "result_correct": correct,
                "reward": float(correct),
            }

        outputs = []
        for _ in range(2):
            search = GraphMCTS(evaluator, MCTSConfig(num_layers=4, seed=17))
            search.evaluate_anchors()
            search.run(50)
            outputs.append(search.result(requested_simulations=50, extension_reason=None))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0]["best_mask"], [0, 1, 0, 1])

    def test_binary_mask_contract(self):
        route = normalize_visual_on_mask([1] * 28)
        self.assertEqual(tuple(route.shape), (1, 28))
        self.assertTrue(bool(route.all().item()))
        with self.assertRaises(ValueError):
            normalize_visual_on_mask([1] * 27)

    def test_bundled_metric_dispatch(self):
        self.assertEqual(score_prediction("exact_match_ignore_case_punctuation", "Blue.", "blue"), 1.0)
        self.assertEqual(score_prediction("relaxed_accuracy", "101", "100"), 1.0)

    def test_manifest_validator_accepts_a_complete_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.png"
            Image.new("RGB", (4, 4), "white").save(image)
            manifest = root / "manifest.jsonl"
            record = {
                "uid": "test:1",
                "sample_id": "1",
                "benchmark": "test",
                "question": "Question?",
                "prompt": "Question? Answer briefly.",
                "answer": "yes",
                "all_answer_norms": None,
                "metric_name": "exact_match_ignore_case_punctuation",
                "correctness_threshold": 1.0,
                "max_new_tokens": 8,
                "image_group_id": "image-1",
                "local_image_path": str(image.resolve()),
                "max_image_tokens": None,
            }
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            report = root / "report.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "validate_manifest.py"),
                    "--manifest",
                    str(manifest),
                    "--expected-count",
                    "1",
                    "--report",
                    str(report),
                ],
                check=True,
            )
            self.assertTrue(json.loads(report.read_text())["passed"])

    def test_contract_freezer_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            for name in (
                "chat_template.json",
                "config.json",
                "generation_config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
            ):
                (model / name).write_text("{}\n", encoding="utf-8")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "benchmark": "test",
                        "metric_name": "exact_match_ignore_case_punctuation",
                        "correctness_threshold": 1.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            hashes = []
            for index in range(2):
                output = root / f"contract_{index}.json"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "freeze_contract.py"),
                        "--manifest",
                        str(manifest),
                        "--model-path",
                        str(model),
                        "--revision",
                        "test-revision",
                        "--dataset-version",
                        "test-v1",
                        "--output",
                        str(output),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                hashes.append(completed.stdout.strip())
            self.assertEqual(hashes[0], hashes[1])

    def test_smoke_selection_spreads_mixed_masks_across_benchmarks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.jsonl"
            rows = [
                {"uid": f"{benchmark}:{index}", "benchmark": benchmark}
                for benchmark in ("a", "b", "c")
                for index in range(5)
            ]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = root / "smoke.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "make_smoke_manifest.py"),
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(output),
                    "--per-benchmark",
                    "5",
                    "--mixed-records",
                    "3",
                ],
                check=True,
                capture_output=True,
            )
            selected = [json.loads(line) for line in output.read_text().splitlines()]
            mixed_benchmarks = {row["benchmark"] for row in selected if row.get("mixed_masks")}
            self.assertEqual(mixed_benchmarks, {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
