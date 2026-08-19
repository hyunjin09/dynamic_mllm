"""Frozen native-Qwen preprocessing and route evaluation runtime."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from binary_policy.executor import BinaryQwen25VL, binary_greedy_generate
from binary_policy.executor.inputs import build_binary_inputs
from reference.dvr_qwen.eval_metrics import score_prediction


def build_native_processor_inputs(processor, sample: dict, device: torch.device):
    """Build inputs with the pinned processor defaults and no visual-token cap."""
    image_path = Path(sample["local_image_path"])
    image_content = {"type": "image", "image": str(image_path)}
    messages = [
        {
            "role": "user",
            "content": [image_content, {"type": "text", "text": sample["prompt"]}],
        }
    ]
    literal = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    with Image.open(image_path) as raw_image:
        original_dimensions = [int(raw_image.width), int(raw_image.height)]
        image = raw_image.convert("RGB")
        batch = processor(
            text=[literal],
            images=[image],
            videos=None,
            padding=True,
            return_tensors="pt",
            return_mm_token_type_ids=True,
        )
    inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in dict(batch).items()
    }
    metadata = {
        "prompt_sha256": sha256(literal.encode("utf-8")).hexdigest(),
        "literal_prompt": literal,
        "original_image_dimensions": original_dimensions,
        "custom_max_image_tokens": None,
        "processor_uses_native_defaults": True,
    }
    return inputs, metadata


def configure_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_frozen_model(model_path: str, revision: str, device_index: int):
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    processor = AutoProcessor.from_pretrained(
        model_path,
        revision=revision,
        local_files_only=True,
        use_fast=False,
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        revision=revision,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": str(device)},
    ).eval()
    return processor, base, BinaryQwen25VL(base), device


@dataclass
class NativeGeneration:
    generated_ids: list[int]
    prediction: str
    score: float
    result_correct: bool


class RouteEvaluator:
    def __init__(self, *, processor, base_model, wrapped_model, sample: dict, device: torch.device):
        self.processor = processor
        self.base_model = base_model
        self.wrapped_model = wrapped_model
        self.sample = sample
        self.device = device
        self.inputs, self.input_metadata = build_native_processor_inputs(processor, sample, device)
        self.prepared = build_binary_inputs(wrapped_model, self.inputs)
        self.results: list[dict] = []

    @property
    def geometry(self) -> dict:
        return {
            "text_tokens": int(self.prepared.text_valid_mask.sum().item()),
            "visual_tokens": int(self.prepared.visual_valid_mask.sum().item()),
            "full_prompt_tokens": int(self.prepared.full_attention_mask.sum().item()),
        }

    def _decode(self, ids: list[int]) -> str:
        return self.processor.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def _score(self, prediction: str) -> tuple[float, bool]:
        score = float(
            score_prediction(
                self.sample["metric_name"],
                prediction,
                self.sample["answer"],
                self.sample.get("all_answer_norms"),
            )
        )
        return score, bool(score >= float(self.sample["correctness_threshold"]))

    @torch.inference_mode()
    def native_all_on(self) -> NativeGeneration:
        self.base_model.rope_deltas = None
        output = self.base_model.generate(
            **self.inputs,
            max_new_tokens=int(self.sample["max_new_tokens"]),
            do_sample=False,
            use_cache=True,
        )
        ids = output[0, self.inputs["input_ids"].shape[1] :].detach().cpu().tolist()
        prediction = self._decode(ids)
        score, correct = self._score(prediction)
        del output
        return NativeGeneration(ids, prediction, score, correct)

    @torch.inference_mode()
    def evaluate(self, mask: tuple[int, ...], source: str) -> dict:
        self.base_model.rope_deltas = None
        output = binary_greedy_generate(
            self.wrapped_model,
            self.inputs,
            mask,
            max_new_tokens=int(self.sample["max_new_tokens"]),
            prepared_inputs=self.prepared,
        )
        ids = output.generated_ids[0].detach().cpu().tolist()
        prediction = self._decode(ids)
        score, correct = self._score(prediction)
        key = "".join(str(int(value)) for value in mask)
        route_id = f"route_{len(self.results):04d}_{key}"
        transitions = sum(int(mask[index] != mask[index - 1]) for index in range(1, len(mask)))
        row = {
            "route_id": route_id,
            "visual_on_mask": list(mask),
            "mask_key": key,
            "mask_one_based": [index + 1 for index, value in enumerate(mask) if value],
            "num_visual_on_layers": int(sum(mask)),
            "num_visual_off_layers": int(len(mask) - sum(mask)),
            "num_transitions": transitions,
            "hamming_distance_to_all_on": int(len(mask) - sum(mask)),
            "generated_ids": ids,
            "prediction": prediction,
            "score": score,
            "correctness_threshold": float(self.sample["correctness_threshold"]),
            "result_correct": correct,
            "reward": float(correct),
            "mcts_evaluation_count": 1,
            "mcts_first_source": source,
            "cache_lengths_unique": sorted(
                {int(stat.cache_rows) for stat in output.prefill.layer_stats if stat.cache_rows is not None}
            ),
            **self.geometry,
        }
        self.results.append(row)
        del output
        return row
