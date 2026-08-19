from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch

from experiments.binary_executor_preflight import prepare


class RecordingProcessor:
    def __init__(self) -> None:
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        return "frozen prompt"

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return {"input_ids": torch.tensor([[1, 2, 3]])}


def _sample(image_path: Path, max_image_tokens: int) -> dict:
    return {
        "local_image_path": str(image_path),
        "prompt": "What is shown?",
        "max_image_tokens": max_image_tokens,
    }


def test_prepare_forwards_record_image_budget_to_processor(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 24)).save(image_path)
    processor = RecordingProcessor()

    literal, batch = prepare(processor, _sample(image_path, 2048), torch.device("cpu"))

    assert literal == "frozen prompt"
    assert batch["input_ids"].device.type == "cpu"
    assert processor.kwargs["max_pixels"] == 2048 * 28 * 28
    assert processor.messages[0]["content"][0] == {
        "type": "image",
        "image": str(image_path),
        "max_pixels": 2048 * 28 * 28,
    }


def test_prepare_preserves_processor_default_without_image_budget(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 24)).save(image_path)
    processor = RecordingProcessor()

    prepare(processor, _sample(image_path, 0), torch.device("cpu"))

    assert "max_pixels" not in processor.kwargs
    assert processor.messages[0]["content"][0] == {
        "type": "image",
        "image": str(image_path),
    }
