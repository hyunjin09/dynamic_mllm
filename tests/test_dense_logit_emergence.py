import numpy as np
import torch

from dense_failure_stage1.logit_emergence import (
    classify_wrong_trajectory,
    first_persistent_layer,
    strongest_non_target_from_top2,
)
from experiments.analyze_dense_logit_emergence import _score_hidden_states


def test_first_persistent_layer_requires_consecutive_strict_crossing():
    values = np.asarray([0.0, 1.1, 0.9, 1.2, 1.3, 1.4, 0.2])

    assert first_persistent_layer(values, threshold=1.0, direction="greater") == 3
    assert first_persistent_layer(values, threshold=0.0, direction="less") is None


def test_first_persistent_layer_supports_zero_crossing_and_custom_run_length():
    values = np.asarray([0.2, -0.1, -0.2, 0.1, -0.3, -0.4, -0.5])

    assert (
        first_persistent_layer(
            values, threshold=0.0, direction="less", consecutive=3
        )
        == 4
    )
    assert (
        first_persistent_layer(
            values, threshold=0.0, direction="less", consecutive=4
        )
        is None
    )


def test_strongest_non_target_uses_second_logit_only_when_top1_is_target():
    top_values = np.asarray([[9.0, 8.0], [7.0, 6.0], [5.0, 4.0]])
    top_ids = np.asarray([[2, 3], [4, 5], [6, 7]])
    target_ids = np.asarray([2, 5, 8])

    actual = strongest_non_target_from_top2(top_values, top_ids, target_ids)

    np.testing.assert_array_equal(actual, np.asarray([8.0, 7.0, 5.0]))


def test_wrong_taxonomy_is_prospective_and_deterministic():
    early = np.asarray([-1.2, -1.3, -1.4] + [-2.0] * 25)
    progressive = np.asarray([0.0] * 8 + [-1.2, -1.3, -1.4] + [-2.0] * 17)
    erosion = np.asarray([0.0, 1.2, 1.3, 1.4, 0.5, 0.0, -1.2, -1.3, -1.4] + [-2.0] * 19)
    unstable = np.asarray([(-1.2 if index % 2 else 1.2) for index in range(28)])

    assert classify_wrong_trajectory(early, gt_token_id=10, pred_token_id=11) == "early_wrong"
    assert classify_wrong_trajectory(progressive, gt_token_id=10, pred_token_id=11) == "progressive_wrong"
    assert classify_wrong_trajectory(erosion, gt_token_id=10, pred_token_id=11) == "answer_erosion"
    assert classify_wrong_trajectory(unstable, gt_token_id=10, pred_token_id=11) == "ambiguous"


def test_wrong_taxonomy_marks_first_token_collision_ambiguous():
    margins = np.asarray([-5.0] * 28)

    assert (
        classify_wrong_trajectory(margins, gt_token_id=10, pred_token_id=10)
        == "ambiguous"
    )


def test_batched_readout_gathers_targets_and_excludes_gt_from_competitor():
    generator = torch.Generator().manual_seed(31)
    hidden = torch.randn(2, 28, 3584, generator=generator, dtype=torch.bfloat16)
    head = torch.nn.Linear(3584, 7, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        head.weight.copy_(
            torch.randn(head.weight.shape, generator=generator, dtype=torch.bfloat16)
        )
    gt_tokens = torch.tensor([2, 4])
    pred_tokens = torch.tensor([3, 5])

    gt, pred, competitor = _score_hidden_states(
        hidden,
        gt_tokens,
        pred_tokens,
        norm=torch.nn.Identity(),
        head=head,
        device=torch.device("cpu"),
        batch_size=17,
    )

    logits = head(hidden.reshape(-1, 3584)).float().reshape(2, 28, 7)
    expected_gt = logits.gather(
        2, gt_tokens[:, None, None].expand(-1, 28, 1)
    ).squeeze(2)
    expected_pred = logits.gather(
        2, pred_tokens[:, None, None].expand(-1, 28, 1)
    ).squeeze(2)
    masked = logits.clone()
    masked.scatter_(2, gt_tokens[:, None, None].expand(-1, 28, 1), -torch.inf)

    torch.testing.assert_close(gt, expected_gt)
    torch.testing.assert_close(pred, expected_pred)
    torch.testing.assert_close(competitor, masked.max(dim=2).values)
