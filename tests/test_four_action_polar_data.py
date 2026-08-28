from __future__ import annotations

import json

import pytest
import torch

from four_action_policy.dataset import (
    FourActionManifestDataset,
    make_duplicated_action_collator,
    make_set_collator,
    route_weights,
)
from four_action_policy.multimodal import attach_visual_features


class FakeTokenizer:
    def __call__(self, texts, **_kwargs):
        maximum = max(len(text.split()) for text in texts)
        input_ids = torch.zeros(len(texts), maximum, dtype=torch.long)
        attention = torch.zeros(len(texts), maximum, dtype=torch.long)
        for index, text in enumerate(texts):
            length = len(text.split())
            input_ids[index, :length] = torch.arange(1, length + 1)
            attention[index, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention}


def route(*actions: str) -> dict:
    return {"actions": list(actions), "route_key": "|".join(actions)}


def row(uid: str, group: str, split: str, routes: list[dict]) -> dict:
    return {
        "uid": uid,
        "dataset": "gqa",
        "question": f"question for {uid}",
        "image_path": f"/images/{uid}.jpg",
        "split_group": group,
        "split": split,
        "valid_routes": routes,
    }


def test_polar_full_route_weighting_generalizes_to_four_actions() -> None:
    routes = [
        route("FULL", "FULL"),
        route("FULL", "IGNORE"),
        route("READ_ONLY", "WRITE_ONLY"),
    ]

    weights = route_weights(routes, "polar_full_downweight_0.3")

    assert weights == pytest.approx([0.3 / 2.3, 1.0 / 2.3, 1.0 / 2.3])


def test_set_collator_preserves_all_routes_and_pads_categorical_indices() -> None:
    rows = [
        row(
            "gqa:a",
            "gqa:image-a",
            "train",
            [route("FULL", "IGNORE"), route("READ_ONLY", "WRITE_ONLY")],
        ),
        row(
            "gqa:b",
            "gqa:image-b",
            "train",
            [route("IGNORE", "IGNORE")],
        ),
    ]
    collate = make_set_collator(
        FakeTokenizer(), max_length=32, route_weighting="equal"
    )

    batch = collate(rows)

    assert batch["valid_routes"].shape == (2, 2, 2)
    assert batch["valid_routes"].tolist() == [
        [[3, 0], [1, 2]],
        [[0, 0], [0, 0]],
    ]
    assert batch["valid_mask"].tolist() == [[True, True], [True, False]]
    assert torch.allclose(
        batch["route_weights"], torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    )


def test_duplicated_collator_encodes_unique_input_once_and_expands_routes() -> None:
    rows = [
        row(
            "gqa:a",
            "gqa:image-a",
            "train",
            [route("FULL", "IGNORE"), route("READ_ONLY", "WRITE_ONLY")],
        ),
        row(
            "gqa:b",
            "gqa:image-b",
            "train",
            [route("IGNORE", "FULL")],
        ),
    ]
    collate = make_duplicated_action_collator(
        FakeTokenizer(), max_length=32, route_weighting="equal"
    )

    batch = collate(rows)

    assert batch["input_ids"].shape[0] == 2
    assert batch["target_actions"].shape == (3, 2)
    assert batch["route_sample_index"].tolist() == [0, 0, 1]
    assert batch["route_weights"].tolist() == pytest.approx([0.5, 0.5, 1.0])
    assert batch["unique_examples"] == 2


def test_manifest_dataset_rejects_duplicate_routes_and_group_leakage(tmp_path) -> None:
    duplicate = route("FULL", "IGNORE")
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(value)
            for value in [
                row("gqa:a", "gqa:image", "train", [duplicate, duplicate]),
                row("gqa:b", "gqa:image", "validation", [route("IGNORE", "FULL")]),
            ]
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="split-group leakage"):
        FourActionManifestDataset(path)

    clean = tmp_path / "clean.jsonl"
    clean.write_text(json.dumps(row("gqa:a", "gqa:image", "train", [duplicate, duplicate])) + "\n")
    with pytest.raises(ValueError, match="duplicate valid route"):
        FourActionManifestDataset(clean)


def test_visual_attachment_loads_and_pads_each_unique_input_once(tmp_path) -> None:
    tensor_a = torch.randn(3, 5, dtype=torch.bfloat16)
    tensor_b = torch.randn(2, 5, dtype=torch.bfloat16)
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    torch.save(tensor_a, path_a)
    torch.save(tensor_b, path_b)
    feature_index = {
        "gqa:a": {"path": str(path_a), "shape": [3, 5]},
        "gqa:b": {"path": str(path_b), "shape": [2, 5]},
    }
    rows = [
        row("gqa:a", "gqa:image-a", "train", [route("FULL", "IGNORE")]),
        row("gqa:b", "gqa:image-b", "train", [route("IGNORE", "FULL")]),
    ]

    batch = attach_visual_features({"uids": ["gqa:a", "gqa:b"]}, rows, feature_index)

    assert batch["image_features"].shape == (2, 3, 5)
    assert batch["image_attention_mask"].tolist() == [
        [True, True, True],
        [True, True, False],
    ]
    assert torch.equal(batch["image_features"][0], tensor_a)
