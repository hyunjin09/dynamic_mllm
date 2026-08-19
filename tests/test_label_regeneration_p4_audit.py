from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from label_regeneration.audit import audit_cache


def _candidate(mask, route_id, *, correct):
    key = "".join(str(value) for value in mask)
    on_count = sum(mask)
    return {
        "route_id": route_id,
        "visual_on_mask": mask,
        "mask_key": key,
        "mask_one_based": [index + 1 for index, value in enumerate(mask) if value],
        "num_visual_on_layers": on_count,
        "num_visual_off_layers": 28 - on_count,
        "num_transitions": sum(mask[index] != mask[index - 1] for index in range(1, 28)),
        "hamming_distance_to_all_on": 28 - on_count,
        "generated_ids": [10, 11],
        "prediction": "yes" if correct else "no",
        "score": 1.0 if correct else 0.0,
        "correctness_threshold": 1.0,
        "result_correct": correct,
        "reward": 1.0 if correct else 0.0,
        "text_tokens": 10,
        "visual_tokens": 5,
        "full_prompt_tokens": 15,
        "cache_lengths_unique": [10, 15],
    }


def _manifest_row(benchmark="gqa"):
    return {
        "uid": f"{benchmark}:sample",
        "benchmark": benchmark,
        "sample_id": "sample",
        "extraction_index": 0,
        "source_row_sha256": "source-hash",
        "prompt": "Question?",
        "question": "Question?",
        "answer": "yes",
        "metric_name": "exact_match_ignore_case_punctuation",
        "correctness_threshold": 1.0,
        "max_new_tokens": 16,
        "max_image_tokens": None,
        "image_group_id": "image:1",
        "local_image_path": "/data/dataset/example.jpg",
    }


def _record(source, contract="contract"):
    all_on = [1] * 28
    all_off = [0] * 28
    root = _candidate(all_on, "root", correct=True)
    off = _candidate(all_off, "off", correct=False)
    sample = {
        **source,
        "current_all_on_prediction": "yes",
        "current_all_on_score": 1.0,
        "current_all_on_status": "correct",
        "actual_text_tokens": 10,
        "actual_visual_tokens": 5,
        "actual_full_prompt_tokens": 15,
        "input_metadata": {"custom_max_image_tokens": None, "processor_uses_native_defaults": True},
    }
    evaluated = [
        {key: row[key] for key in ("route_id", "visual_on_mask", "mask_key", "score", "result_correct", "reward")}
        for row in (root, off)
    ]
    return {
        "phase": "binary_visual_mask_graph_mcts_regenerated_v1",
        "dataset_version": "8k_native_qwen_unrestricted_mask_regeneration_v1",
        "root_policy": "all_visual_on_recomputed_current_executor",
        "runtime": {
            "contract_sha256": contract,
            "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "attn_implementation": "sdpa",
            "dtype": "bfloat16",
            "custom_max_image_tokens": None,
            "native_image_processing": True,
            "processor_use_fast": False,
            "generation_policy": {"do_sample": False, "max_new_tokens": 16},
        },
        "sample": sample,
        "candidate_executions": [root, off],
        "root_route_id": "root",
        "all_off_route_id": "off",
        "successful_route_ids": ["root"],
        "best_sparse_success_route_id": "root",
        "mcts_config": {
            "base_num_simulations": 200,
            "num_simulations": 200,
            "fixed_layer_permutation": False,
            "stop_on_first_success": False,
            "transposition_table": True,
            "expansion_policy": "choose_layer_and_visual_on_off_from_all_undecided_layers",
        },
        "mcts": {
            "completed_simulations": 200,
            "requested_simulations": 200,
            "extension_reason": None,
            "root_reward": 1.0,
            "all_off_reward": 0.0,
            "evaluated_masks": evaluated,
            "successful_masks": [all_on],
            "best_mask": all_on,
            "simulations": [{"simulation": index + 1} for index in range(200)],
        },
    }


class LabelRegenerationP4AuditTest(unittest.TestCase):
    def _write_fixture(self, root: Path, source: dict, record: dict):
        manifest = root / "manifest.jsonl"
        manifest.write_text(json.dumps(source) + "\n", encoding="utf-8")
        sample_root = root / "cache" / "raw_route_cache" / "shard_000_of_001" / "samples"
        sample_root.mkdir(parents=True)
        (sample_root / "sample.json").write_text(json.dumps(record), encoding="utf-8")
        return manifest, root / "cache"

    def test_complete_contract_bound_cache_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _manifest_row()
            manifest, cache = self._write_fixture(root, source, _record(source))
            report, index_rows = audit_cache(
                manifest,
                cache,
                contract_sha256="contract",
                expected_dataset_counts={"gqa": 1},
            )
        self.assertTrue(report["passed"])
        self.assertEqual(report["valid_terminal_records"], 1)
        self.assertEqual(len(index_rows), 1)
        self.assertEqual(index_rows[0]["uid"], "gqa:sample")

    def test_wemath_is_rejected_from_the_three_benchmark_population(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _manifest_row("wemath2pro")
            manifest, cache = self._write_fixture(root, source, _record(source))
            report, _ = audit_cache(
                manifest,
                cache,
                contract_sha256="contract",
                expected_dataset_counts={"gqa": 1},
            )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["manifest_dataset_population"])

    def test_malformed_route_is_not_counted_as_a_valid_terminal_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _manifest_row()
            record = _record(source)
            record["candidate_executions"][1]["visual_on_mask"] = [0] * 27
            manifest, cache = self._write_fixture(root, source, record)
            report, _ = audit_cache(
                manifest,
                cache,
                contract_sha256="contract",
                expected_dataset_counts={"gqa": 1},
            )
        self.assertFalse(report["passed"])
        self.assertEqual(report["valid_terminal_records"], 0)
        self.assertIn("candidate_1_mask", report["invalid_records"][0]["failed_checks"])


if __name__ == "__main__":
    unittest.main()
