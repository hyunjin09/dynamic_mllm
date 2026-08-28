from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import experiments.run_sequential_four_action_label_conversion as runner


def test_sample_runner_records_every_exact_branch_and_preserves_c2c(monkeypatch):
    class FakeRuntime:
        def __init__(self, **_kwargs):
            self.calls = Counter()
            self.input_metadata = {"prompt_sha256": "prompt"}
            self.geometry = {"text_tokens": 2, "visual_tokens": 1, "full_prompt_tokens": 3}

        def initialize_full(self):
            return SimpleNamespace(
                evaluation={
                    "generated_ids": [0],
                    "generated_answer": "wrong",
                    "correct": False,
                    "answer_alignment_margin": -1.0,
                }
            )

        def evaluate(self, route):
            route = tuple(route)
            self.calls[route] += 1
            correct = route in {
                ("IGNORE", "FULL"),
                ("READ_ONLY", "FULL"),
                ("WRITE_ONLY", "FULL"),
                ("READ_ONLY", "IGNORE"),
                ("WRITE_ONLY", "IGNORE"),
            }
            return {
                "generated_ids": [1] if correct else [0],
                "generated_answer": "yes" if correct else "wrong",
                "correct": correct,
                "answer_alignment_margin": 1.0 if correct else -1.0,
            }

        def evaluate_old_binary(self, mask):
            return {
                "generated_ids": [1],
                "generated_answer": "yes",
                "correct": True,
                "correctness_score": 1.0,
            }

    runtime = FakeRuntime()
    monkeypatch.setattr(runner, "FourActionSampleRuntime", lambda **_kwargs: runtime)
    monkeypatch.setattr(runner.torch.cuda, "reset_peak_memory_stats", lambda _d: None)
    monkeypatch.setattr(runner.torch.cuda, "max_memory_allocated", lambda _d: 10)

    result = runner.process_sample(
        processor=None,
        model=None,
        record={
            "uid": "gqa:u",
            "dataset": "gqa",
            "sample_id": "u",
            "source_split": "train",
            "source_current_all_on_status": "wrong",
            "source_current_all_on_prediction": "wrong",
            "source_positive_route_count": 1,
            "source_positive_routes": [
                {
                    "source_binary_route_id": "gqa:u::r1",
                    "route_id": "r1",
                    "mask": [0, 1],
                    "source_off_count": 1,
                    "source_all_off": False,
                }
            ],
        },
        device=SimpleNamespace(index=0),
        config={"layer_count": 2, "scoring_timeout_seconds": 1.0},
        rank=0,
        mode="smoke",
        execution_contract={"contract_sha256": "hash"},
    )

    conversion = result["raw_conversions"][0]
    assert result["route_type"] == "W2C"
    assert result["label_semantics"] == "corrective_w2c"
    assert conversion["maximum_branch_count"] == 2
    assert [branch["route"] for branch in conversion["final_branches"]] == [
        ["READ_ONLY", "FULL"],
        ["WRITE_ONLY", "FULL"],
    ]
    assert len(result["unique_valid_four_action_routes"]) == 2
    assert result["later_training_view"] == result["unique_valid_four_action_routes"]
    assert result["pilot_old_binary_semantic_checks"][0]["correctness_match"] is True
    assert result["route_evaluation_cache"]["cache_hits"] >= 2
    assert all(
        branch["evaluation"]["correct"]
        for branch in conversion["final_branches"]
    )


def test_sample_runner_does_not_refine_c2c_routes(monkeypatch):
    class FakeRuntime:
        input_metadata = {}
        geometry = {}

        def __init__(self, **_kwargs):
            self.calls = Counter()

        def initialize_full(self):
            return SimpleNamespace(
                evaluation={
                    "generated_ids": [1],
                    "generated_answer": "yes",
                    "correct": True,
                    "answer_alignment_margin": 1.0,
                }
            )

        def evaluate(self, route):
            route = tuple(route)
            self.calls[route] += 1
            return {
                "generated_ids": [1],
                "generated_answer": "yes",
                "correct": True,
                "answer_alignment_margin": 1.0,
            }

        def evaluate_old_binary(self, _mask):
            return self.evaluate(("IGNORE", "FULL"))

    runtime = FakeRuntime()
    monkeypatch.setattr(runner, "FourActionSampleRuntime", lambda **_kwargs: runtime)
    monkeypatch.setattr(runner.torch.cuda, "reset_peak_memory_stats", lambda _d: None)
    monkeypatch.setattr(runner.torch.cuda, "max_memory_allocated", lambda _d: 10)

    result = runner.process_sample(
        processor=None,
        model=None,
        record={
            "uid": "textvqa:u",
            "dataset": "textvqa",
            "sample_id": "u",
            "source_split": "train",
            "source_current_all_on_status": "correct",
            "source_current_all_on_prediction": "yes",
            "source_positive_route_count": 1,
            "source_positive_routes": [
                {
                    "source_binary_route_id": "textvqa:u::r1",
                    "route_id": "r1",
                    "mask": [0, 1],
                    "source_off_count": 1,
                    "source_all_off": False,
                }
            ],
        },
        device=SimpleNamespace(index=0),
        config={"layer_count": 2, "scoring_timeout_seconds": 1.0},
        rank=0,
        mode="full",
        execution_contract={"contract_sha256": "hash"},
    )

    conversion = result["raw_conversions"][0]
    assert result["route_type"] == "C2C"
    assert conversion["steps"] == []
    assert conversion["final_branches"][0]["route"] == ["IGNORE", "FULL"]
    assert conversion["new_route_evaluations"] == 0
