#!/usr/bin/env python3
"""Portable Qwen DVR runtime shared by the replay gate and MCTS runner."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

from dvr_qwen.binary_generate import binary_dvrc_greedy_generate, prepare_binary_dvrc_inputs
from dvr_qwen.eval_metrics import score_prediction
from dvr_qwen.generate import generation_policy_record
from dvr_qwen.modeling_dvr_qwen2_5_vl import DVRQwen2_5_VLForConditionalGeneration


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DATA_ROOT = Path(
    "/data/dataset/dynamic_mllm/Qwen2.5VL/easy_hard_5k/complete_correct_wrong_pools_20260713"
)
KNOWN_DATA_ROOTS = [
    ORIGINAL_DATA_ROOT,
    Path(
        "/home/aix7101/hyemin/dynamic_mllm/mnt/dataset/dynamic_mllm/Qwen2.5VL/easy_hard_5k/complete_correct_wrong_pools_20260713"
    ),
    Path("/mnt/hyemin/dvr_qwen/rl/complete_correct_wrong_pools_20260713"),
]


def default_hf_hub_cache() -> Path:
    value = os.environ.get("HF_HUB_CACHE") or os.environ.get("MCTS_HF_HUB_CACHE")
    if value:
        return Path(value).expanduser()
    return Path.home() / ".cache" / "huggingface" / "hub"


def default_model_source() -> Path | None:
    value = os.environ.get("MCTS_MODEL_SOURCE")
    return Path(value).expanduser() if value else None


def default_data_root() -> Path | None:
    value = os.environ.get("MCTS_DATA_ROOT")
    return Path(value).expanduser() if value else None


def is_correct(score: float, threshold: float) -> bool:
    return bool(not math.isnan(score) and score >= threshold)


def mask_one_based(route: list[int]) -> list[int]:
    return [index + 1 for index, value in enumerate(route) if int(value)]


def route_tensor(route: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([[bool(value) for value in route]], dtype=torch.bool, device=device)


def token_id_from_candidates(processor: AutoProcessor, candidates: list[str]) -> int | None:
    tokenizer = getattr(processor, "tokenizer", processor)
    for token in candidates:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            continue
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if unk_id is not None and int(token_id) == int(unk_id):
            continue
        return int(token_id)
    return None


def derive_mm_token_type_ids(processor: AutoProcessor, input_ids: torch.Tensor) -> torch.Tensor:
    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
    image_token_id = token_id_from_candidates(processor, ["<|image_pad|>", "<image>", "<|image|>"])
    video_token_id = token_id_from_candidates(processor, ["<|video_pad|>", "<video>", "<|video|>"])
    if image_token_id is not None:
        mm_token_type_ids = torch.where(
            input_ids == image_token_id,
            torch.ones_like(mm_token_type_ids),
            mm_token_type_ids,
        )
    if video_token_id is not None:
        mm_token_type_ids = torch.where(
            input_ids == video_token_id,
            torch.full_like(mm_token_type_ids, 2),
            mm_token_type_ids,
        )
    if int((mm_token_type_ids > 0).sum().item()) == 0:
        raise RuntimeError("could not derive visual token ids from processor/tokenizer")
    return mm_token_type_ids


def resolve_image_path(sample: dict[str, Any], data_root: Path | None) -> Path:
    original = Path(str(sample["local_image_path"]))
    if original.exists():
        return original
    if data_root is None:
        raise FileNotFoundError(f"missing image and no MCTS_DATA_ROOT remap configured: {original}")

    root = data_root.expanduser().resolve()
    candidates: list[Path] = []
    for known_root in KNOWN_DATA_ROOTS:
        try:
            candidates.append(root / original.relative_to(known_root))
        except ValueError:
            pass
    parts = original.parts
    if "images" in parts:
        candidates.append(root.joinpath(*parts[parts.index("images") :]))
    candidates.extend(
        [
            root / "images" / str(sample["benchmark"]) / original.name,
            root / str(sample["benchmark"]) / original.name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not remap image {original}; tried: {tried}")


def build_processor_inputs(
    processor: AutoProcessor,
    sample: dict[str, Any],
    *,
    data_root: Path | None,
) -> dict[str, Any]:
    image_path = resolve_image_path(sample, data_root)
    max_image_tokens = int(sample.get("max_image_tokens") or 0)
    image_content: dict[str, Any] = {"type": "image", "image": str(image_path)}
    if max_image_tokens > 0:
        image_content["max_pixels"] = max_image_tokens * 28 * 28
    messages = [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": sample["prompt"]},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            return_mm_token_type_ids=True,
        )
    except Exception:
        image = Image.open(image_path).convert("RGB")
        if max_image_tokens > 0:
            max_pixels = max_image_tokens * 28 * 28
            width, height = image.size
            if width * height > max_pixels:
                scale = (max_pixels / float(width * height)) ** 0.5
                image = image.resize(
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    Image.Resampling.BICUBIC,
                )
        inputs = processor(
            text=[text],
            images=[image],
            videos=None,
            padding=True,
            return_tensors="pt",
        )
    if "mm_token_type_ids" not in inputs:
        inputs["mm_token_type_ids"] = derive_mm_token_type_ids(processor, inputs["input_ids"])
    return dict(inputs)


def decode_generated(processor: AutoProcessor, token_ids: torch.Tensor) -> str:
    return processor.batch_decode(
        token_ids.detach().cpu(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


@torch.inference_mode()
def evaluate_binary_route(
    *,
    model: DVRQwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    processor_inputs: dict[str, Any],
    prepared_binary_inputs: Any,
    sample: dict[str, Any],
    route: list[int],
    embed_device: torch.device,
) -> dict[str, Any]:
    output = binary_dvrc_greedy_generate(
        model,
        processor_inputs,
        visual_on_mask=route_tensor(route, embed_device),
        max_new_tokens=int(sample["max_new_tokens"]),
        stop_on_eos=True,
        prepared_binary_inputs=prepared_binary_inputs,
    )
    prediction = decode_generated(processor, output.generated_ids)
    score = score_prediction(
        sample["metric_name"],
        prediction,
        sample["answer"],
        sample.get("all_answer_norms"),
    )
    record = {
        "prediction": prediction,
        "score": float(score),
        "generated_ids": output.generated_ids.detach().cpu().view(-1).tolist(),
        "num_visual_on_layers": int(sum(route)),
        "mask_one_based": mask_one_based(route),
        "visual_on_mask": route,
        "cache_lengths_unique": sorted(set(output.state.cache.lengths() if output.state.cache else [])),
        "text_tokens": int(output.state.binary_inputs.text_valid_mask.sum().item()),
        "visual_tokens": int(output.state.binary_inputs.visual_valid_mask.sum().item()),
        "full_prompt_tokens": int(output.state.binary_inputs.full_attention_mask.sum().item()),
    }
    del output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


@torch.inference_mode()
def evaluate_hf_all_on(
    *,
    model: DVRQwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    processor_inputs: dict[str, Any],
    sample: dict[str, Any],
    embed_device: torch.device,
) -> dict[str, Any]:
    device_inputs = {
        key: value.to(embed_device) if torch.is_tensor(value) else value
        for key, value in processor_inputs.items()
    }
    generate_inputs = dict(device_inputs)
    output = model.base_model.generate(
        **generate_inputs,
        max_new_tokens=int(sample["max_new_tokens"]),
        do_sample=False,
        return_dict_in_generate=True,
    )
    generated = output.sequences[:, generate_inputs["input_ids"].shape[1] :]
    prediction = decode_generated(processor, generated)
    score = score_prediction(
        sample["metric_name"],
        prediction,
        sample["answer"],
        sample.get("all_answer_norms"),
    )
    return {
        "generated_ids": generated.detach().cpu().view(-1).tolist(),
        "prediction": prediction,
        "score": float(score),
    }


def load_model(args: Any):
    model_source = Path(args.model_source).expanduser().resolve()
    hf_hub_cache = Path(args.hf_hub_cache).expanduser().resolve()
    if not model_source.is_dir():
        raise FileNotFoundError(f"model snapshot directory does not exist: {model_source}")
    if not hf_hub_cache.is_dir():
        raise FileNotFoundError(f"HF hub cache directory does not exist: {hf_hub_cache}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen replay/MCTS runtime")

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    processor_kwargs: dict[str, Any] = {
        "cache_dir": str(hf_hub_cache),
        "local_files_only": True,
    }
    if args.processor_use_fast != "auto":
        processor_kwargs["use_fast"] = args.processor_use_fast == "true"
    processor = AutoProcessor.from_pretrained(str(model_source), **processor_kwargs)

    max_memory = None
    if args.first_gpu_max_memory_gb > 0:
        max_memory = {
            device_index: f"{int(args.other_gpu_max_memory_gb)}GiB"
            for device_index in range(torch.cuda.device_count())
        }
        max_memory[0] = f"{int(args.first_gpu_max_memory_gb)}GiB"
        if int(args.cpu_max_memory_gb) > 0:
            max_memory["cpu"] = f"{int(args.cpu_max_memory_gb)}GiB"
    model = DVRQwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_source),
        cache_dir=str(hf_hub_cache),
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        max_memory=max_memory,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    )
    model.eval()
    embed_device = next(model.model.get_input_embeddings().parameters()).device
    return model, processor, embed_device


__all__ = [
    "build_processor_inputs",
    "default_data_root",
    "default_hf_hub_cache",
    "default_model_source",
    "evaluate_binary_route",
    "evaluate_hf_all_on",
    "generation_policy_record",
    "is_correct",
    "load_model",
    "mask_one_based",
    "prepare_binary_dvrc_inputs",
    "resolve_image_path",
]
