"""P13 native-visual-token and matched modality contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from binary_policy.multimodal import (
    deterministic_modality_permutations,
    make_multimodal_set_collator,
    resolve_modality_inputs,
)
from binary_policy.predictor import BinaryPolarBackbone
from binary_policy.training import evaluate_multimodal_epoch, train_multimodal_epoch


class _Tokenizer:
    def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
        del truncation, max_length
        assert padding and return_tensors == "pt"
        ids = torch.zeros(len(texts), 3, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, text in enumerate(texts):
            width = min(len(text), 3)
            ids[index, -width:] = torch.arange(1, width + 1)
            mask[index, -width:] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _FrozenEncoder(torch.nn.Module):
    output_dim = 5

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, self.output_dim)
        self.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return self.embedding(input_ids).to(torch.bfloat16)


def _feature_index(tmp_path: Path):
    feature_a = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    feature_b = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    torch.save(feature_a, tmp_path / "a.pt")
    torch.save(feature_b, tmp_path / "b.pt")
    return {
        "a": {"path": str(tmp_path / "a.pt"), "shape": [3, 4]},
        "b": {"path": str(tmp_path / "b.pt"), "shape": [2, 4]},
    }


def _rows():
    return [
        {"uid": "a", "benchmark": "gqa", "question": "aa", "valid_routes": [{"mask": [1, 1, 0, 0]}]},
        {"uid": "b", "benchmark": "gqa", "question": "bbb", "valid_routes": [{"mask": [0, 0, 1, 1]}]},
    ]


def test_multimodal_collator_pads_native_visual_rows_without_pooling(tmp_path):
    batch = make_multimodal_set_collator(_Tokenizer(), _feature_index(tmp_path))(_rows())
    assert batch["image_features"].shape == (2, 3, 4)
    assert batch["image_attention_mask"].tolist() == [[True, True, True], [True, True, False]]
    assert batch["image_features"][1, 2].eq(0).all()
    assert batch["valid_masks"].shape == (2, 1, 4)


@pytest.mark.parametrize(
    ("modality", "question_valid", "image_present"),
    (("question", True, False), ("image", False, True), ("image_question", True, True)),
)
def test_modality_resolution_changes_only_visible_input(modality, question_valid, image_present):
    token_features = torch.randn(2, 3, 5)
    text_mask = torch.ones(2, 3, dtype=torch.long)
    image_features = torch.randn(2, 4, 7)
    image_mask = torch.ones(2, 4, dtype=torch.bool)
    resolved = resolve_modality_inputs(
        modality, token_features, text_mask, image_features, image_mask
    )
    assert bool(resolved[1].any()) is question_valid
    assert (resolved[2] is not None) is image_present
    assert (resolved[3] is not None) is image_present


def test_same_direct_head_executes_all_three_modalities_and_image_padding_is_masked():
    torch.manual_seed(4)
    model = BinaryPolarBackbone(
        num_layers=4,
        input_dim=5,
        image_dim=7,
        d_model=8,
        num_heads=2,
        num_layer_blocks=1,
        dropout=0.0,
    ).eval()
    text = torch.randn(2, 3, 5)
    text_mask = torch.ones(2, 3, dtype=torch.long)
    image = torch.randn(2, 4, 7)
    image_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    padded_changed = image.clone()
    padded_changed[1, 2:] = 999
    outputs = {}
    for modality in ("question", "image", "image_question"):
        args = resolve_modality_inputs(modality, text, text_mask, image, image_mask)
        outputs[modality] = model(*args)
    image_changed = model(
        *resolve_modality_inputs("image_question", text, text_mask, padded_changed, image_mask)
    )
    assert all(value.shape == (2, 4) for value in outputs.values())
    assert torch.equal(outputs["image_question"][1], image_changed[1])
    assert not torch.equal(outputs["question"], outputs["image"])


def test_adding_p13_image_projection_preserves_all_p11_shared_initial_tensors():
    arguments = dict(
        num_layers=4, input_dim=5, d_model=8, num_heads=2, num_layer_blocks=1, dropout=0.0
    )
    torch.manual_seed(91)
    p11 = BinaryPolarBackbone(**arguments)
    torch.manual_seed(91)
    p13 = BinaryPolarBackbone(**arguments, image_dim=7)
    p13_state = p13.state_dict()
    for name, tensor in p11.state_dict().items():
        assert torch.equal(tensor, p13_state[name]), name


def test_modality_permutations_are_deterministic_within_dataset_and_have_no_fixed_points():
    rows = [
        {"uid": f"g{index}", "benchmark": "gqa"} for index in range(4)
    ] + [{"uid": f"t{index}", "benchmark": "textvqa"} for index in range(4)]
    first = deterministic_modality_permutations(rows, seed=19)
    second = deterministic_modality_permutations(rows, seed=19)
    assert first == second
    by_uid = {row["uid"]: row for row in rows}
    for target, donors in first.items():
        assert donors["question_uid"] != target
        assert donors["image_uid"] != target
        assert donors["both_question_uid"] != target
        assert donors["both_image_uid"] != target
        assert by_uid[target]["benchmark"] == by_uid[donors["question_uid"]]["benchmark"]
        assert by_uid[target]["benchmark"] == by_uid[donors["image_uid"]]["benchmark"]


@pytest.mark.parametrize("modality", ("question", "image", "image_question"))
def test_matched_multimodal_train_and_validation_are_finite_and_encoder_stays_frozen(
    tmp_path, modality
):
    batch = make_multimodal_set_collator(_Tokenizer(), _feature_index(tmp_path))(_rows())
    encoder = _FrozenEncoder()
    predictor = BinaryPolarBackbone(
        num_layers=4,
        input_dim=5,
        image_dim=4,
        d_model=8,
        num_heads=2,
        num_layer_blocks=1,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-2)
    train = train_multimodal_epoch(
        predictor,
        encoder,
        [batch],
        optimizer,
        modality=modality,
        device=torch.device("cpu"),
        amp_dtype=torch.bfloat16,
    )
    validation = evaluate_multimodal_epoch(
        predictor,
        encoder,
        [batch],
        modality=modality,
        device=torch.device("cpu"),
        amp_dtype=torch.bfloat16,
    )
    assert torch.isfinite(torch.tensor(train["loss"]))
    assert torch.isfinite(torch.tensor(validation["set_nll"]))
    assert all(parameter.grad is None for parameter in encoder.parameters())
