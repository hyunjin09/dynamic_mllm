import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from experiments.run_label_regeneration import record_complete
from label_regeneration.runtime import RouteEvaluator, score_prediction_with_timeout


class WeMathScoringTimeoutTests(unittest.TestCase):
    def test_fast_scorer_is_unchanged(self):
        with patch("label_regeneration.runtime.score_prediction", return_value=1.0):
            score, timed_out = score_prediction_with_timeout(
                "metric", "prediction", "answer", None, timeout_seconds=0.2
            )
        self.assertEqual(score, 1.0)
        self.assertFalse(timed_out)

    def test_nonterminating_scorer_is_bounded(self):
        def spin(*_args, **_kwargs):
            while True:
                pass

        started = time.monotonic()
        with patch("label_regeneration.runtime.score_prediction", side_effect=spin):
            score, timed_out = score_prediction_with_timeout(
                "metric", "prediction", "answer", None, timeout_seconds=0.05
            )
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(score, 0.0)
        self.assertTrue(timed_out)

    def test_timeout_result_is_cached_by_decoded_prediction(self):
        evaluator = RouteEvaluator.__new__(RouteEvaluator)
        evaluator.sample = {
            "metric_name": "metric",
            "answer": "answer",
            "all_answer_norms": None,
            "correctness_threshold": 1.0,
        }
        evaluator.scoring_timeout_seconds = 0.05
        evaluator._score_cache = {}
        evaluator.scoring_timeout_count = 0
        calls = 0

        def spin(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            while True:
                pass

        with patch("label_regeneration.runtime.score_prediction", side_effect=spin):
            first = evaluator._score("same")
            second = evaluator._score("same")
        self.assertEqual(first, (0.0, False, True))
        self.assertEqual(second, first)
        self.assertEqual(calls, 1)
        self.assertEqual(evaluator.scoring_timeout_count, 1)

    def test_prior_contract_is_accepted_only_when_explicit(self):
        record = {
            "sample": {"uid": "u"},
            "runtime": {"contract_sha256": "old"},
            "candidate_executions": [],
            "mcts": {"completed_simulations": 200, "requested_simulations": 200},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertFalse(record_complete(path, uid="u", contract_hash="new"))
            self.assertTrue(
                record_complete(
                    path,
                    uid="u",
                    contract_hash="new",
                    compatible_contract_hashes=("old",),
                )
            )

    def test_resume_rejects_records_above_new_cap(self):
        record = {
            "sample": {"uid": "u"},
            "runtime": {"contract_sha256": "old"},
            "candidate_executions": [],
            "mcts": {"completed_simulations": 600, "requested_simulations": 600},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertFalse(
                record_complete(
                    path,
                    uid="u",
                    contract_hash="new",
                    compatible_contract_hashes=("old",),
                    max_simulations_per_sample=400,
                )
            )
            record["mcts"] = {
                "completed_simulations": 400,
                "requested_simulations": 400,
            }
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertTrue(
                record_complete(
                    path,
                    uid="u",
                    contract_hash="new",
                    compatible_contract_hashes=("old",),
                    max_simulations_per_sample=400,
                )
            )


if __name__ == "__main__":
    unittest.main()
