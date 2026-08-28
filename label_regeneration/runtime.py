"""Frozen native-Qwen preprocessing and route evaluation runtime."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import signal
from typing import Any

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from binary_policy.executor import BinaryQwen25VL, binary_greedy_generate
from binary_policy.executor.inputs import build_binary_inputs
from reference.dvr_qwen.eval_metrics import score_prediction


class ScoringTimeoutError(BaseException):
    """Escape third-party graders that catch ordinary Exception subclasses."""


def _open_frozen_image(
    image_path: Path,
    expected_content_sha256: str | None,
):
    """Open a provenance-verified oversized image under its source contract."""
    try:
        return Image.open(image_path)
    except Image.DecompressionBombError as exc:
        if not expected_content_sha256:
            raise ValueError(
                "oversized image retry requires a frozen content SHA-256"
            ) from exc
        actual_content_sha256 = sha256(image_path.read_bytes()).hexdigest()
        if actual_content_sha256 != expected_content_sha256:
            raise ValueError(
                "oversized image content SHA-256 does not match the frozen source"
            ) from exc
        previous_limit = Image.MAX_IMAGE_PIXELS
        try:
            # Conversion workers are single-threaded. Restore Pillow's global
            # guard immediately after the verified image header is accepted.
            Image.MAX_IMAGE_PIXELS = None
            return Image.open(image_path)
        finally:
            Image.MAX_IMAGE_PIXELS = previous_limit


def score_prediction_with_timeout(
    metric_name: str,
    prediction: str,
    answer: str,
    all_answer_norms: list[str] | None,
    *,
    timeout_seconds: float,
) -> tuple[float, bool]:
    """Run the unchanged scorer, conservatively returning zero on nontermination."""
    if timeout_seconds <= 0:
        raise ValueError("score timeout must be positive")

    def timeout_handler(_signum, _frame):
        raise ScoringTimeoutError()

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        score = float(score_prediction(metric_name, prediction, answer, all_answer_norms))
        return score, False
    except ScoringTimeoutError:
        return 0.0, True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


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
    with _open_frozen_image(
        image_path,
        sample.get("image_content_sha256"),
    ) as raw_image:
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
    scoring_timed_out: bool


class RouteEvaluator:
    def __init__(
        self,
        *,
        processor,
        base_model,
        wrapped_model,
        sample: dict,
        device: torch.device,
        scoring_timeout_seconds: float = 5.0,
    ):
        self.processor = processor
        self.base_model = base_model
        self.wrapped_model = wrapped_model
        self.sample = sample
        self.device = device
        self.scoring_timeout_seconds = float(scoring_timeout_seconds)
        if self.scoring_timeout_seconds <= 0:
            raise ValueError("scoring_timeout_seconds must be positive")
        self._score_cache: dict[str, tuple[float, bool, bool]] = {}
        self.scoring_timeout_count = 0
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

    def _score(self, prediction: str) -> tuple[float, bool, bool]:
        cached = self._score_cache.get(prediction)
        if cached is not None:
            return cached
        score, timed_out = score_prediction_with_timeout(
            self.sample["metric_name"],
            prediction,
            self.sample["answer"],
            self.sample.get("all_answer_norms"),
            timeout_seconds=self.scoring_timeout_seconds,
        )
        if timed_out:
            self.scoring_timeout_count += 1
        result = (
            score,
            bool(score >= float(self.sample["correctness_threshold"])),
            timed_out,
        )
        self._score_cache[prediction] = result
        return result

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
        score, correct, timed_out = self._score(prediction)
        del output
        return NativeGeneration(ids, prediction, score, correct, timed_out)

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
        score, correct, timed_out = self._score(prediction)
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
            "scoring_timed_out": timed_out,
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
