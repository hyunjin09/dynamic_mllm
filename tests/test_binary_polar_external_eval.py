"""Contracts for full10 static-predictor external evaluation."""

from experiments.evaluate_binary_polar_external import (
    active_benchmark,
    cluster_key,
    mask_statistics,
    native_generation_inputs,
    predictor_text,
    select_shard,
)

import torch


def test_predictor_text_uses_question_for_core_and_pope():
    row = {"benchmark": "textvqa", "question": "What is written?", "prompt": "unused"}
    assert predictor_text(row) == "What is written?"


def test_predictor_text_joins_ordered_multiple_choice_chunks_without_answer_suffix():
    row = {
        "benchmark": "mmmu_val",
        "instruction_text_chunks": [
            "What follows the image? ",
            "A. one\nB. two\nAnswer with the option letter only.",
        ],
    }
    assert predictor_text(row) == "What follows the image? A. one\nB. two"


def test_docvqa_is_excluded_but_active_benchmarks_are_retained():
    assert active_benchmark("chartqa")
    assert active_benchmark("mmmu_pro_vision_test")
    assert active_benchmark("pope_random")
    assert not active_benchmark("docvqa")


def test_sharding_is_deterministic_and_complete():
    rows = [{"uid": str(index)} for index in range(11)]
    shards = [select_shard(rows, num_shards=3, shard_index=index) for index in range(3)]
    flattened = [row["uid"] for shard in shards for row in shard]
    assert sorted(flattened, key=int) == [str(index) for index in range(11)]
    assert len(flattened) == len(set(flattened))


def test_mask_statistics_preserve_complete_binary_route():
    result = mask_statistics([1, 1, 0, 0, 1])
    assert result == {
        "mask_key": "11001",
        "num_visual_on_layers": 3,
        "transition_count": 2,
    }


def test_cluster_key_uses_complete_multi_image_identity():
    assert cluster_key({"image_content_sha256": "abc"}) == "abc"
    assert cluster_key({"image_content_sha256s": ["abc", "def"]}) == "abc|def"


def test_native_generation_inputs_drop_private_mask_and_move_tensors():
    result = native_generation_inputs(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones(1, 2),
            "instruction_token_mask": torch.ones(1, 2, dtype=torch.bool),
        },
        device=torch.device("cpu"),
    )
    assert set(result) == {"input_ids", "attention_mask"}
    assert all(value.device.type == "cpu" for value in result.values())

