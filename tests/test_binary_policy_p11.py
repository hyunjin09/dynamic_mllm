"""Deterministic contracts introduced by the bounded P11 diagnostic."""

from __future__ import annotations

import math

import torch

from binary_policy.dataset import make_duplicated_path_collator, make_set_collator
from binary_policy.evaluation import batch_offline_metrics
from binary_policy.predictor import BiasOnlyBinaryPredictor
from binary_policy.p11 import deterministic_within_dataset_shuffle, summarize_label_geometry


class _TinyTokenizer:
    def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
        del truncation, max_length
        assert padding and return_tensors == "pt"
        width = max(len(text) for text in texts)
        ids = torch.zeros(len(texts), width, dtype=torch.long)
        attention = torch.zeros_like(ids)
        for index, text in enumerate(texts):
            ids[index, -len(text) :] = torch.arange(1, len(text) + 1)
            attention[index, -len(text) :] = 1
        return {"input_ids": ids, "attention_mask": attention}


def _row(uid: str, masks: list[list[int]]):
    return {
        "uid": uid,
        "question": uid,
        "benchmark": "gqa",
        "valid_routes": [
            {"mask": mask, "key": "".join(map(str, mask)), "weight": 1.0 / len(masks)}
            for mask in masks
        ],
    }


def test_polar_full_downweight_is_normalized_identically_for_both_collators():
    row = _row("weighted", [[1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0]])
    expected = torch.tensor([0.3 / 2.3, 1.0 / 2.3, 1.0 / 2.3])

    set_batch = make_set_collator(
        _TinyTokenizer(), route_weighting="polar_full_downweight_0.3"
    )([row])
    bce_batch = make_duplicated_path_collator(
        _TinyTokenizer(), route_weighting="polar_full_downweight_0.3"
    )([row])

    assert torch.allclose(set_batch["route_weights"][0], expected)
    assert torch.allclose(bce_batch["sample_weights"], expected)


def test_polar_full_downweight_leaves_sets_without_a_cheaper_full_alternative_equal():
    only_full = _row("only-full", [[1, 1, 1, 1]])
    no_full = _row("no-full", [[1, 0, 0, 0], [0, 0, 0, 0]])
    collate = make_set_collator(_TinyTokenizer(), route_weighting="polar_full_downweight_0.3")

    batch = collate([only_full, no_full])

    assert batch["route_weights"][0].tolist() == [1.0, 0.0]
    assert torch.allclose(batch["route_weights"][1], torch.tensor([0.5, 0.5]))


def test_offline_metrics_report_top1_mask_diversity():
    logits = torch.tensor([[2.0, -2.0], [3.0, -3.0], [-2.0, -2.0]])
    valid_masks = torch.tensor([[[1, 0]], [[1, 0]], [[0, 0]]], dtype=torch.float32)
    valid = torch.ones(3, 1, dtype=torch.bool)

    metrics = batch_offline_metrics(logits, valid_masks, valid, top_k=2)

    expected_entropy = -(2 / 3) * math.log(2 / 3) - (1 / 3) * math.log(1 / 3)
    assert metrics["unique_top1_masks"] == 2
    assert metrics["fraction_top1_all_on"] == 0.0
    assert metrics["fraction_top1_all_off"] == 1 / 3
    assert metrics["average_predicted_visual_on"] == 2 / 3
    assert math.isclose(metrics["top1_mask_entropy_nats"], expected_entropy)
    assert metrics["top1_mask_counts"] == {"10": 2, "00": 1}


def test_bias_only_predictor_supports_global_and_dataset_conditioned_logits():
    global_model = BiasOnlyBinaryPredictor(num_layers=4, num_datasets=1)
    dataset_model = BiasOnlyBinaryPredictor(num_layers=4, num_datasets=3)
    with torch.no_grad():
        global_model.logits.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        dataset_model.logits.copy_(
            torch.tensor(
                [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]
            )
        )

    assert global_model(torch.tensor([0, 0])).tolist() == [[1.0, 2.0, 3.0, 4.0]] * 2
    assert dataset_model(torch.tensor([2, 0])).tolist() == [
        [3.0, 3.0, 3.0, 3.0],
        [1.0, 1.0, 1.0, 1.0],
    ]


def test_label_geometry_finds_shortcut_and_best_constant_mask():
    rows = [
        {**_row("a", [[1, 1], [1, 0]]), "benchmark": "gqa"},
        {**_row("b", [[1, 1], [0, 0]]), "benchmark": "gqa"},
        {**_row("c", [[1, 0]]), "benchmark": "textvqa"},
    ]

    summary = summarize_label_geometry(rows)

    assert summary["records"] == 3
    assert summary["all_on_coverage"] == 2 / 3
    assert summary["all_off_coverage"] == 1 / 3
    assert summary["all_on_with_cheaper_valid_coverage"] == 2 / 3
    # Equal-coverage ties are resolved lexicographically for reproducibility.
    assert summary["best_constant"]["mask"] == "10"
    assert summary["best_constant"]["coverage"] == 2 / 3
    assert summary["best_non_all_on_constant"]["mask"] == "10"
    assert summary["best_non_all_on_constant"]["coverage"] == 2 / 3
    assert summary["top_n_union_coverage"]["1"] == 2 / 3
    assert summary["top_n_union_coverage"]["5"] == 1.0


def test_within_dataset_shuffle_is_deterministic_and_has_no_fixed_points():
    rows = [
        {"uid": "g1", "benchmark": "gqa"},
        {"uid": "g2", "benchmark": "gqa"},
        {"uid": "g3", "benchmark": "gqa"},
        {"uid": "t1", "benchmark": "textvqa"},
        {"uid": "t2", "benchmark": "textvqa"},
    ]

    first = deterministic_within_dataset_shuffle(rows, seed=17)
    second = deterministic_within_dataset_shuffle(rows, seed=17)

    assert first == second
    assert all(target != donor for target, donor in first.items())
    uid_to_dataset = {row["uid"]: row["benchmark"] for row in rows}
    assert all(uid_to_dataset[target] == uid_to_dataset[donor] for target, donor in first.items())
