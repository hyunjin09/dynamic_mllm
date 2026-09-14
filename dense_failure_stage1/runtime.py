"""Native all-visual-on Qwen2.5-VL execution with optional passive pooling."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
from typing import Sequence

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from dense_failure_stage1.lmms_scoring import score_lmms_sample
from tools.research_analysis.dense_failure_stage1 import (
    TokenPositions,
    locate_user_and_visual_tokens,
)


def configure_dense_determinism(seed: int, backend_policy: dict) -> None:
    """Apply the frozen deterministic CUDA policy before model execution."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = backend_policy["cublas_workspace_config"]
    torch.backends.cuda.enable_flash_sdp(bool(backend_policy["flash_sdp_enabled"]))
    torch.backends.cuda.enable_mem_efficient_sdp(
        bool(backend_policy["memory_efficient_sdp_enabled"])
    )
    torch.backends.cuda.enable_math_sdp(bool(backend_policy["math_sdp_enabled"]))
    torch.backends.cuda.enable_cudnn_sdp(bool(backend_policy["cudnn_sdp_enabled"]))
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = bool(
        backend_policy["matmul_allow_fp16_reduced_precision_reduction"]
    )
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bool(
        backend_policy["matmul_allow_bf16_reduced_precision_reduction"]
    )
    torch.set_float32_matmul_precision(backend_policy["float32_matmul_precision"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(
        bool(backend_policy["deterministic_algorithms"]),
        warn_only=bool(backend_policy["deterministic_warn_only"]),
    )
    torch.backends.cuda.matmul.allow_tf32 = bool(backend_policy["matmul_allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(backend_policy["cudnn_allow_tf32"])
    torch.backends.cudnn.benchmark = bool(backend_policy["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(backend_policy["cudnn_deterministic"])


def load_dense_runtime(model_path: str, revision: str, device_index: int):
    """Load only the native dense model and processor; no routed wrapper."""

    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    processor = AutoProcessor.from_pretrained(
        model_path,
        revision=revision,
        local_files_only=True,
        use_fast=False,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        revision=revision,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": str(device)},
    ).eval()
    return processor, model, device


@contextmanager
def _open_verified_image(path: Path, expected_sha256: str):
    """Hash and decode the same open file descriptor, preventing path substitution."""

    with path.open("rb") as handle:
        digest = sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"image SHA-256 differs immediately before inference: {path}: "
                f"actual={actual} expected={expected_sha256}"
            )
        handle.seek(0)
        try:
            image = Image.open(handle)
        except Image.DecompressionBombError:
            previous = Image.MAX_IMAGE_PIXELS
            try:
                Image.MAX_IMAGE_PIXELS = None
                handle.seek(0)
                image = Image.open(handle)
            finally:
                Image.MAX_IMAGE_PIXELS = previous
        with image:
            yield image


def build_dense_inputs(processor, sample: dict, device: torch.device):
    """Apply the native Qwen chat and image processor without token caps."""

    image_path = Path(sample["local_image_path"])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": sample["prompt"]},
            ],
        }
    ]
    literal = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    with _open_verified_image(image_path, sample["image_content_sha256"]) as raw:
        dimensions = [int(raw.width), int(raw.height)]
        image = raw.convert("RGB")
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
        "literal_prompt": literal,
        "literal_prompt_sha256": sha256(literal.encode("utf-8")).hexdigest(),
        "original_image_dimensions": dimensions,
        "custom_max_image_tokens": None,
        "image_processing": "native Qwen processor defaults",
        "consumed_image_sha256": sample["image_content_sha256"],
    }
    return inputs, metadata


def token_positions(processor, model, input_ids: torch.Tensor) -> TokenPositions:
    tokenizer = processor.tokenizer
    vision_end = int(tokenizer.convert_tokens_to_ids("<|vision_end|>"))
    im_end = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    image_token = int(model.config.image_token_id)
    return locate_user_and_visual_tokens(
        input_ids[0].detach().cpu().tolist(),
        image_token_id=image_token,
        vision_end_token_id=vision_end,
        im_end_token_id=im_end,
    )


class DenseFeatureCollector:
    """Passively pool prompt states from each native decoder layer."""

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        *,
        visual_positions: Sequence[int],
        user_text_positions: Sequence[int],
        final_user_token_position: int,
    ):
        self.layers = list(layers)
        self.visual_positions = tuple(int(value) for value in visual_positions)
        self.user_text_positions = tuple(int(value) for value in user_text_positions)
        self.final_user_token_position = int(final_user_token_position)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._features: list[dict[str, torch.Tensor] | None] = [None] * len(self.layers)

    def _hook(self, layer_index: int):
        def capture(_module, _inputs, output):
            if self._features[layer_index] is not None:
                return None
            hidden = output[0] if isinstance(output, tuple) else output
            # Only the prefill invocation spans the frozen prompt positions.
            if hidden.ndim != 3 or hidden.shape[1] <= self.final_user_token_position:
                return None
            final = hidden[0, self.final_user_token_position].detach().cpu()
            text = hidden[0, list(self.user_text_positions)].mean(dim=0).detach().cpu()
            visual = hidden[0, list(self.visual_positions)].mean(dim=0).detach().cpu()
            self._features[layer_index] = {
                "text_final": final,
                "text_mean": text,
                "visual_mean": visual,
            }
            return None

        return capture

    def __enter__(self):
        if self._handles:
            raise RuntimeError("feature collector is already active")
        self._handles = [
            layer.register_forward_hook(self._hook(index))
            for index, layer in enumerate(self.layers)
        ]
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return False

    def stacked(self) -> dict[str, torch.Tensor]:
        missing = [index for index, value in enumerate(self._features) if value is None]
        if missing:
            raise RuntimeError(f"feature hooks did not observe decoder layers: {missing}")
        complete = [value for value in self._features if value is not None]
        return {
            key: torch.stack([value[key] for value in complete], dim=0)
            for key in ("text_final", "text_mean", "visual_mean")
        }


@dataclass
class DenseGenerationResult:
    generated_ids: list[int]
    generated_text: str
    normalized_prediction: str | None
    lmms_metric_name: str
    correctness_threshold: float
    score: float
    correct: bool
    input_metadata: dict
    prompt_token_count: int
    visual_token_count: int
    user_text_token_count: int
    features: dict[str, torch.Tensor] | None


@torch.inference_mode()
def generate_dense(
    processor,
    model,
    device: torch.device,
    sample: dict,
    *,
    extract_features: bool,
) -> DenseGenerationResult:
    """Generate one authoritative native dense answer under the frozen policy."""

    inputs, input_metadata = build_dense_inputs(processor, sample, device)
    positions = token_positions(processor, model, inputs["input_ids"])
    if hasattr(model, "model") and hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None
    collector = None
    if extract_features:
        collector = DenseFeatureCollector(
            model.model.language_model.layers,
            visual_positions=positions.visual,
            user_text_positions=positions.user_text,
            final_user_token_position=positions.final_user_token,
        )
        with collector:
            output = model.generate(
                **inputs,
                max_new_tokens=int(sample["max_new_tokens"]),
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
    else:
        output = model.generate(
            **inputs,
            max_new_tokens=int(sample["max_new_tokens"]),
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    generated_ids = output[0, inputs["input_ids"].shape[1] :].detach().cpu().tolist()
    generated_text = processor.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    lmms_score = score_lmms_sample(
        dataset=sample.get("dataset", sample.get("benchmark")),
        prediction=generated_text,
        answer=sample["answer"],
        answers=sample.get("all_answer_norms"),
        uid=sample["uid"],
    )
    features = collector.stacked() if collector is not None else None
    prompt_token_count = int(inputs["attention_mask"].sum().item())
    del output, inputs
    return DenseGenerationResult(
        generated_ids=generated_ids,
        generated_text=generated_text,
        normalized_prediction=lmms_score.normalized_prediction,
        lmms_metric_name=lmms_score.metric_name,
        correctness_threshold=lmms_score.correctness_threshold,
        score=lmms_score.raw_score,
        correct=lmms_score.correct,
        input_metadata=input_metadata,
        prompt_token_count=prompt_token_count,
        visual_token_count=len(positions.visual),
        user_text_token_count=len(positions.user_text),
        features=features,
    )
