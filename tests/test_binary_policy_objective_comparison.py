"""Deterministic contract tests for duplicated BCE versus exact set NLL."""

from __future__ import annotations

import math
import json
from pathlib import Path
import tempfile

import torch
from torch import nn

from binary_policy.dataset import BinaryPolicyManifestDataset, make_duplicated_path_collator, make_set_collator
from binary_policy.decode import topk_factorized_masks
from binary_policy.losses import (
    bernoulli_mask_log_probability,
    multi_valid_set_nll,
    polar_path_bce,
)
from binary_policy.multimodal import make_multimodal_duplicated_path_collator
from binary_policy.predictor import BinaryPolarBackbone
from binary_policy.training import evaluate_epoch, predictor_state_sha256, train_epoch


class _TinyTokenizer:
    """Small deterministic tokenizer substitute; no model or network access."""

    def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
        assert padding and truncation and return_tensors == "pt"
        rows = [[(ord(char) % 31) + 1 for char in text][:max_length] or [1] for text in texts]
        width = max(len(row) for row in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        attention = torch.zeros(len(rows), width, dtype=torch.long)
        for index, row in enumerate(rows):
            ids[index, -len(row) :] = torch.tensor(row)
            attention[index, -len(row) :] = 1
        return {"input_ids": ids, "attention_mask": attention}


class _FrozenTinyEncoder(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, width)
        self.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return self.embedding(input_ids)


class _FrozenBFloat16TinyEncoder(_FrozenTinyEncoder):
    def forward(self, input_ids, attention_mask):
        return super().forward(input_ids, attention_mask).to(torch.bfloat16)


def _row(uid: str, masks: list[list[int]], weights: list[float] | None = None):
    if weights is None:
        weights = [1.0 / len(masks)] * len(masks)
    return {
        "uid": uid,
        "question": f"question {uid}",
        "valid_routes": [
            {"mask": mask, "weight": weight, "key": "".join(map(str, mask))}
            for mask, weight in zip(masks, weights)
        ],
    }


def test_a_single_valid_route_matches_complete_mask_nll():
    logits = torch.tensor([[0.7, -1.2, 0.2, 1.1]], dtype=torch.float64)
    mask = torch.tensor([[[1, 0, 1, 1]]], dtype=torch.float64)
    expected = -bernoulli_mask_log_probability(logits, mask).squeeze()
    actual = multi_valid_set_nll(logits, mask)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_b_exact_set_nll_can_choose_a_mode_while_duplicated_bce_learns_marginals():
    valid = torch.tensor([[[1, 1, 0, 0], [0, 0, 1, 1]]], dtype=torch.float64)

    set_logits = nn.Parameter(torch.tensor([[0.01, 0.02, -0.01, -0.02]], dtype=torch.float64))
    set_optimizer = torch.optim.Adam([set_logits], lr=0.1)
    for _ in range(300):
        set_optimizer.zero_grad(set_to_none=True)
        set_loss = multi_valid_set_nll(set_logits, valid)
        set_loss.backward()
        assert torch.isfinite(set_logits.grad).all()
        set_optimizer.step()

    bce_logits = nn.Parameter(torch.tensor([[0.01, 0.02, -0.01, -0.02]], dtype=torch.float64))
    bce_optimizer = torch.optim.Adam([bce_logits], lr=0.1)
    duplicated_targets = valid[0]
    for _ in range(300):
        bce_optimizer.zero_grad(set_to_none=True)
        duplicated_logits = bce_logits.expand(2, -1)
        bce_loss = polar_path_bce(duplicated_logits, duplicated_targets)
        bce_loss.backward()
        assert torch.isfinite(bce_logits.grad).all()
        bce_optimizer.step()

    valid_set = {(1, 1, 0, 0), (0, 0, 1, 1)}
    assert topk_factorized_masks(set_logits.detach(), top_k=1)[0][0].mask in valid_set
    assert float(torch.sigmoid(set_logits).sub(0.5).abs().mean()) > 0.45
    assert torch.allclose(torch.sigmoid(bce_logits), torch.full_like(bce_logits, 0.5), atol=1e-5)


def test_c_padded_routes_contribute_no_probability_mass():
    logits = torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float64)
    unpadded = torch.tensor([[[1, 0, 1]]], dtype=torch.float64)
    padded = torch.tensor([[[1, 0, 1], [1, 1, 0], [0, 0, 0]]], dtype=torch.float64)
    valid_mask = torch.tensor([[True, False, False]])
    weights = torch.tensor([[1.0, 9.0, 12.0]], dtype=torch.float64)
    expected = multi_valid_set_nll(logits, unpadded)
    actual = multi_valid_set_nll(logits, padded, valid_mask=valid_mask, route_weights=weights)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=0.0)


def test_c_collator_preserves_variable_sized_valid_sets():
    collate = make_set_collator(_TinyTokenizer(), route_weighting="equal")
    batch = collate([_row("a", [[1, 0, 1, 0]]), _row("b", [[0, 1, 0, 1], [1, 1, 0, 0]])])
    assert batch["valid_masks"].shape == (2, 2, 4)
    assert batch["valid_mask"].tolist() == [[True, False], [True, True]]
    assert torch.allclose(batch["route_weights"][0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(batch["route_weights"][1], torch.tensor([0.5, 0.5]))


def test_d_collators_reject_duplicate_masks_instead_of_treating_multiplicity_as_weight():
    duplicate = _row("duplicate", [[1, 0, 1, 0], [1, 0, 1, 0]])
    for collate in (
        make_set_collator(_TinyTokenizer()),
        make_duplicated_path_collator(_TinyTokenizer()),
    ):
        try:
            collate([duplicate])
        except ValueError as error:
            assert "duplicate valid mask" in str(error)
        else:
            raise AssertionError("duplicate masks must be rejected by the derived-manifest collator")


def test_duplicated_collator_uses_equal_per_input_route_weight_and_preserves_unique_batch_size():
    collate = make_duplicated_path_collator(_TinyTokenizer(), route_weighting="equal")
    batch = collate([_row("a", [[1, 0, 1, 0]]), _row("b", [[0, 1, 0, 1], [1, 1, 0, 0]])])
    assert batch["input_ids"].shape[0] == 2
    assert batch["targets"].shape == (3, 4)
    assert batch["sample_weights"].tolist() == [1.0, 0.5, 0.5]
    assert batch["route_sample_index"].tolist() == [0, 1, 1]
    assert batch["unique_examples"] == 2
    assert batch["uids"] == ["a", "b", "b"]


def test_multimodal_duplicated_collator_keeps_visual_rows_at_unique_input_granularity():
    rows = [_row("a", [[1, 0, 1, 0]]), _row("b", [[0, 1, 0, 1], [1, 1, 0, 0]])]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        second = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        first_path = root / "a.pt"
        second_path = root / "b.pt"
        torch.save(first, first_path)
        torch.save(second, second_path)
        feature_index = {
            "a": {"path": str(first_path), "shape": [3, 4]},
            "b": {"path": str(second_path), "shape": [5, 4]},
        }
        batch = make_multimodal_duplicated_path_collator(
            _TinyTokenizer(), feature_index, route_weighting="equal"
        )(rows)
    assert batch["unique_examples"] == 2
    assert batch["targets"].shape == (3, 4)
    assert batch["route_sample_index"].tolist() == [0, 1, 1]
    assert batch["image_features"].shape == (2, 5, 4)
    assert batch["image_attention_mask"].tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]
    assert torch.equal(batch["image_features"][0, :3], first)
    assert torch.equal(batch["image_features"][1], second)


def test_manifest_dataset_rejects_split_group_leakage_and_route_cap_overflow():
    base = _row("a", [[1, 0, 1, 0], [0, 1, 0, 1]])
    base.update({"split": "train", "split_group": "image:one"})
    leaked = _row("b", [[1, 1, 0, 0]])
    leaked.update({"split": "validation", "split_group": "image:one"})
    with tempfile.TemporaryDirectory() as directory:
        manifest = Path(directory) / "manifest.jsonl"
        manifest.write_text("\n".join(json.dumps(row) for row in (base, leaked)) + "\n", encoding="utf-8")
        try:
            BinaryPolicyManifestDataset(manifest, "train", max_valid_routes=2)
        except ValueError as error:
            assert "split-group leakage" in str(error)
        else:
            raise AssertionError("cross-split image groups must stop training-manifest loading")

        leaked["split_group"] = "image:two"
        manifest.write_text("\n".join(json.dumps(row) for row in (base, leaked)) + "\n", encoding="utf-8")
        try:
            BinaryPolicyManifestDataset(manifest, "train", max_valid_routes=1)
        except ValueError as error:
            assert "will not silently subsample" in str(error)
        else:
            raise AssertionError("route-cap overflow must stop instead of dropping routes")


def test_e_training_keeps_encoder_frozen_and_updates_only_predictor_for_both_objectives():
    tokenizer = _TinyTokenizer()
    rows = [_row("a", [[1, 0, 1, 0], [0, 1, 0, 1]])]
    for objective, collate in (
        ("exact_set_nll", make_set_collator(tokenizer)),
        ("duplicated_bce", make_duplicated_path_collator(tokenizer)),
    ):
        torch.manual_seed(19)
        frozen_encoder = _FrozenTinyEncoder(width=8)
        predictor = BinaryPolarBackbone(
            num_layers=4,
            input_dim=8,
            d_model=16,
            num_heads=4,
            num_layer_blocks=1,
            dropout=0.0,
        )
        before = {name: value.detach().clone() for name, value in predictor.named_parameters()}
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-2)
        result = train_epoch(
            predictor,
            frozen_encoder,
            [collate(rows)],
            optimizer,
            device=torch.device("cpu"),
            objective=objective,
            gradient_clip_norm=1.0,
            duplicated_route_microbatch_size=1,
        )
        assert result["examples"] == 1
        assert math.isfinite(result["loss"])
        assert all(parameter.grad is None for parameter in frozen_encoder.parameters())
        assert all(parameter.requires_grad is False for parameter in frozen_encoder.parameters())
        assert any(not torch.equal(before[name], parameter) for name, parameter in predictor.named_parameters())
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in predictor.parameters())


def test_validation_autocast_accepts_bfloat16_frozen_encoder_features():
    rows = [_row("a", [[1, 0, 1, 0], [0, 1, 0, 1]])]
    batch = make_set_collator(_TinyTokenizer())(rows)
    frozen_encoder = _FrozenBFloat16TinyEncoder(width=8)
    predictor = BinaryPolarBackbone(
        num_layers=4,
        input_dim=8,
        d_model=16,
        num_heads=4,
        num_layer_blocks=1,
        dropout=0.0,
    )
    result = evaluate_epoch(
        predictor,
        frozen_encoder,
        [batch],
        device=torch.device("cpu"),
        top_k=2,
        amp_dtype=torch.bfloat16,
    )
    assert result["examples"] == 1
    assert math.isfinite(result["set_nll"])


def test_matched_initialization_hash_is_deterministic_and_seed_sensitive():
    def build(seed: int):
        torch.manual_seed(seed)
        return BinaryPolarBackbone(
            num_layers=4,
            input_dim=8,
            d_model=16,
            num_heads=4,
            num_layer_blocks=1,
            dropout=0.1,
        )

    assert predictor_state_sha256(build(23)) == predictor_state_sha256(build(23))
    assert predictor_state_sha256(build(23)) != predictor_state_sha256(build(29))


def test_duplicated_route_microbatching_matches_one_shot_gradient_without_dropout():
    rows = [
        _row("a", [[1, 0, 1, 0], [0, 1, 0, 1]]),
        _row("b", [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1]]),
    ]
    batch = make_duplicated_path_collator(_TinyTokenizer())(rows)

    torch.manual_seed(31)
    encoder = _FrozenTinyEncoder(width=8)
    first = BinaryPolarBackbone(
        num_layers=4, input_dim=8, d_model=16, num_heads=4, num_layer_blocks=1, dropout=0.0
    )
    second = BinaryPolarBackbone(
        num_layers=4, input_dim=8, d_model=16, num_heads=4, num_layer_blocks=1, dropout=0.0
    )
    second.load_state_dict(first.state_dict())
    first_optimizer = torch.optim.SGD(first.parameters(), lr=1e-3)
    second_optimizer = torch.optim.SGD(second.parameters(), lr=1e-3)

    first_metrics = train_epoch(
        first,
        encoder,
        [batch],
        first_optimizer,
        device=torch.device("cpu"),
        objective="duplicated_bce",
        duplicated_route_microbatch_size=None,
    )
    second_metrics = train_epoch(
        second,
        encoder,
        [batch],
        second_optimizer,
        device=torch.device("cpu"),
        objective="duplicated_bce",
        duplicated_route_microbatch_size=1,
    )
    assert abs(first_metrics["loss"] - second_metrics["loss"]) <= 1e-7
    for first_value, second_value in zip(first.state_dict().values(), second.state_dict().values()):
        assert torch.allclose(first_value, second_value, atol=1e-7, rtol=1e-6)


def run_all() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"passed {len(tests)} deterministic objective-comparison tests")


if __name__ == "__main__":
    run_all()
