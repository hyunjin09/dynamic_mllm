import numpy as np
import pytest
import torch
from types import SimpleNamespace

from dense_failure_stage1.answer_logit_emergence import (
    LayerPositionCollector,
    GenerationCallCollector,
    answer_prediction_position,
    append_teacher_prefix,
    build_divergence_spec,
    cached_replay_matches,
    classify_wrong_trajectory,
    select_position_sanity,
    summarize_layerwise_trajectories,
    stable_reference_order,
    token_metrics,
    read_layer_logits_at_position,
)


def test_answer_position_is_last_attended_assistant_prefix_token():
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])

    assert answer_prediction_position(mask) == 3


def test_answer_position_rejects_noncontiguous_or_batched_masks():
    with pytest.raises(ValueError, match="batch size one"):
        answer_prediction_position(torch.ones(2, 4, dtype=torch.long))
    with pytest.raises(ValueError, match="contiguous"):
        answer_prediction_position(torch.tensor([[1, 1, 0, 1]]))


def test_append_teacher_prefix_extends_only_sequence_inputs():
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "mm_token_type_ids": torch.tensor([[1, 0, 0]], dtype=torch.long),
        "pixel_values": torch.randn(4, 3),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }

    extended = append_teacher_prefix(inputs, [20, 21])

    assert extended["input_ids"].tolist() == [[10, 11, 12, 20, 21]]
    assert extended["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert extended["mm_token_type_ids"].tolist() == [[1, 0, 0, 0, 0]]
    assert extended["pixel_values"] is inputs["pixel_values"]
    assert extended["image_grid_thw"] is inputs["image_grid_thw"]


def test_divergence_spec_teacher_forces_shared_prefix_and_handles_termination():
    spec = build_divergence_spec(
        canonical_gt_ids=[31, 32],
        generated_ids=[31, 40, 99],
        im_end_token_id=99,
    )

    assert spec.usable
    assert spec.common_prefix_ids == (31,)
    assert spec.divergence_index == 1
    assert spec.gt_token_id == 32
    assert spec.generated_token_id == 40
    assert spec.gt_source == "canonical"

    termination = build_divergence_spec(
        canonical_gt_ids=[31],
        generated_ids=[31, 40, 99],
        im_end_token_id=99,
    )
    assert termination.gt_token_id == 99
    assert termination.generated_token_id == 40


def test_exact_wrong_sequence_uses_first_different_fallback_or_is_unusable():
    fallback = build_divergence_spec(
        canonical_gt_ids=[31],
        generated_ids=[31, 99],
        im_end_token_id=99,
        fallback_gt_sequences=[("textvqa_modal", [41]), ("later", [51])],
    )
    assert fallback.usable
    assert fallback.gt_source == "textvqa_modal"
    assert fallback.gt_token_id == 41
    assert fallback.generated_token_id == 31

    unusable = build_divergence_spec(
        canonical_gt_ids=[31],
        generated_ids=[31, 99],
        im_end_token_id=99,
        fallback_gt_sequences=[("same", [31])],
    )
    assert not unusable.usable
    assert unusable.reason == "no_gt_sequence_diverges_from_generation"


def test_textvqa_reference_order_is_frequency_then_first_occurrence():
    answers = ["dog", "cat", "cat", "dog", "cat", "bird", "bird"]

    assert stable_reference_order(answers) == ["cat", "dog", "bird"]


def test_token_metrics_reports_min_rank_top1_and_non_target_competitor():
    logits = torch.tensor(
        [[5.0, 3.0, 5.0, 1.0], [0.0, 4.0, 2.0, 3.0]], dtype=torch.float32
    )

    metrics = token_metrics(logits, token_id=2)

    assert metrics["token_logits"].tolist() == [5.0, 2.0]
    assert metrics["token_ranks"].tolist() == [1, 3]
    assert metrics["top1_token_ids"].tolist() == [0, 1]
    assert metrics["top1_logits"].tolist() == [5.0, 4.0]
    assert metrics["strongest_non_target_logits"].tolist() == [5.0, 4.0]


def test_layer_collector_captures_only_the_requested_sequence_position():
    layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)

    with LayerPositionCollector(layers, position=2) as collector:
        value = hidden
        for layer in layers:
            value = layer(value)

    expected = hidden[0, 2]
    torch.testing.assert_close(collector.stacked(), torch.stack([expected, expected]))


def test_generation_call_collector_captures_requested_cached_decode_step():
    layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    values = [
        torch.full((1, 5, 3), 1.0),
        torch.full((1, 1, 3), 2.0),
        torch.full((1, 1, 3), 3.0),
    ]

    with GenerationCallCollector(layers, target_call_index=2) as collector:
        for hidden in values:
            value = hidden
            for layer in layers:
                value = layer(value)

    torch.testing.assert_close(collector.stacked(), torch.full((2, 3), 3.0))


def test_cached_replay_requires_exact_prefix_and_next_processed_token():
    assert cached_replay_matches([10, 11], 12, [10, 11, 12])
    assert not cached_replay_matches([10, 11], 12, [10, 11, 13])
    assert not cached_replay_matches([10, 11], 12, [10, 11])


def test_layer_readout_matches_models_own_final_position_logits():
    class AddLayer(torch.nn.Module):
        def __init__(self, amount):
            super().__init__()
            self.amount = amount

        def forward(self, hidden):
            return hidden + self.amount

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            language = SimpleNamespace(
                layers=torch.nn.ModuleList([AddLayer(1.0), AddLayer(2.0)]),
                norm=torch.nn.Identity(),
            )
            self.model = SimpleNamespace(language_model=language)
            self.lm_head = torch.nn.Linear(4, 6, bias=False)

        def forward(self, input_ids, **_kwargs):
            hidden = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
            for layer in self.model.language_model.layers:
                hidden = layer(hidden)
            logits = self.lm_head(self.model.language_model.norm(hidden[:, -1:]))
            # Simulate the small kernel/rounding difference observed between a
            # detached hook reconstruction and the model's native output.
            logits = logits.clone()
            logits[..., 0] += 100.0
            return SimpleNamespace(logits=logits)

    model = TinyModel()
    inputs = {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }

    result = read_layer_logits_at_position(model, inputs, position=2)

    assert result.layer_logits.shape == (2, 6)
    assert torch.equal(result.layer_logits[-1], result.model_final_logits)
    assert result.final_top1_matches_model
    assert not result.reconstructed_final_top1_matches_model
    assert result.final_max_abs_logit_difference == 100.0


def test_wrong_taxonomy_has_no_collision_shortcut():
    early = np.asarray([-0.1, -0.2, -0.3] + [-1.0] * 25)
    progressive = np.asarray([0.0] * 8 + [-0.1, -0.2, -0.3] + [-1.0] * 17)
    erosion = np.asarray([0.1, 0.2, 0.3, 0.2, 0.1, -0.1, -0.2, -0.3] + [-1.0] * 20)
    unstable = np.asarray([0.1 if index % 2 else -0.1 for index in range(28)])

    assert classify_wrong_trajectory(early) == "early_wrong"
    assert classify_wrong_trajectory(progressive) == "progressive_wrong"
    assert classify_wrong_trajectory(erosion) == "answer_erosion"
    assert classify_wrong_trajectory(unstable) == "ambiguous"


def test_sanity_selection_uses_current_outcome_and_distinct_image_groups():
    rows = []
    for dataset in ("gqa", "chartqa", "textvqa"):
        for correct in (False, True):
            for index in range(4):
                rows.append(
                    {
                        "uid": f"{dataset}-{correct}-{index}",
                        "dataset": dataset,
                        "current_dense_correct": correct,
                        "image_group_id": f"{dataset}-{correct}-{index}",
                        "historical_bucket": "wrong" if correct else "correct",
                        "comparison_common_prefix_ids": (
                            [31] if not correct and index < 2 else []
                        ),
                    }
                )

    selected = select_position_sanity(
        rows, seed=7, per_cell=2, teacher_forced_per_wrong_dataset=1
    )

    assert len(selected) == 12
    assert len({row["image_group_id"] for row in selected}) == 12
    assert {
        (row["dataset"], row["current_dense_correct"]) for row in selected
    } == {
        (dataset, correct)
        for dataset in ("gqa", "chartqa", "textvqa")
        for correct in (False, True)
    }
    for dataset in ("gqa", "chartqa", "textvqa"):
        wrong = [
            row
            for row in selected
            if row["dataset"] == dataset and not row["current_dense_correct"]
        ]
        assert sum(bool(row["comparison_common_prefix_ids"]) for row in wrong) >= 1


def test_layerwise_summary_reports_logits_ranks_margins_and_quantiles():
    gt = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    other = np.asarray([[0.0, 1.0], [2.0, 6.0]])
    rows = summarize_layerwise_trajectories(
        gt_logits=gt,
        other_logits=other,
        gt_ranks=np.asarray([[2, 1], [1, 3]]),
        other_ranks=np.asarray([[3, 2], [2, 1]]),
        top1_logits=np.asarray([[4.0, 2.0], [3.0, 6.0]]),
        margins=gt - other,
        gt_is_top1=np.asarray([[False, True], [True, False]]),
        other_is_top1=np.asarray([[False, False], [False, True]]),
    )

    assert [row["layer"] for row in rows] == [0, 1]
    assert rows[0]["mean_gt_logit"] == 2.0
    assert rows[0]["median_margin"] == 1.0
    assert rows[1]["q25_margin"] == -1.25
    assert rows[1]["gt_top1_fraction"] == 0.5
    assert rows[1]["other_top1_fraction"] == 0.5
