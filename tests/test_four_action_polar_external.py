from __future__ import annotations

import pytest

from four_action_policy.external import (
    ACTIVE_BENCHMARKS,
    action_statistics,
    active_benchmark,
    predictor_text,
    select_shard,
)


def test_external_population_is_prospectively_restricted() -> None:
    assert ACTIVE_BENCHMARKS == (
        "chartqa",
        "mmmu_pro_standard_test",
        "mmmu_pro_vision_test",
        "pope_adversarial",
        "pope_popular",
        "pope_random",
    )
    assert active_benchmark("chartqa")
    assert not active_benchmark("textvqa")
    assert not active_benchmark("docvqa")


def test_predictor_text_uses_question_or_mc_instruction() -> None:
    assert predictor_text({"benchmark": "chartqa", "question": "  how many? "}) == "how many?"
    assert predictor_text(
        {
            "benchmark": "mmmu_pro_vision_test",
            "instruction_text_chunks": ["Question\n", "(A) one\n", "Answer with the option letter only."],
        }
    ) == "Question\n(A) one"
    with pytest.raises(ValueError, match="inactive"):
        predictor_text({"benchmark": "textvqa", "question": "x"})


def test_action_statistics_and_sharding_are_exact() -> None:
    actions = ["FULL", "READ_ONLY", "WRITE_ONLY", "IGNORE"]
    stats = action_statistics(actions)
    assert stats["route_key"] == "FULL|READ_ONLY|WRITE_ONLY|IGNORE"
    assert stats["action_counts"] == {
        "IGNORE": 1,
        "READ_ONLY": 1,
        "WRITE_ONLY": 1,
        "FULL": 1,
    }
    rows = [{"uid": str(index)} for index in range(7)]
    assert [row["uid"] for row in select_shard(rows, num_shards=3, shard_index=1)] == ["1", "4"]
