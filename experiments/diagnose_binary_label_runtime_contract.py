#!/usr/bin/env python3
"""Replay BP-1 failures under the recorded MCTS label-generation contract.

This bounded diagnostic does not score answers or train a predictor.  It uses
the record's image budget, generation policy, and numerical runtime settings,
then compares generated token IDs against the existing cache.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import __version__ as transformers_version

from binary_policy.executor import BinaryQwen25VL, binary_greedy_generate
from binary_policy.executor.inputs import build_binary_inputs


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def cached_candidate(record: dict, mask: list[int]) -> dict | None:
    return next(
        (row for row in record.get("candidate_executions", []) if row.get("visual_on_mask") == mask),
        None,
    )


def label_runtime_inputs(processor, sample: dict, device: torch.device):
    """Mirror ``reference/dvr_qwen/runtime.py``'s portable fallback path."""
    image_path = Path(sample["local_image_path"])
    max_image_tokens = int(sample.get("max_image_tokens") or 0)
    max_pixels = max_image_tokens * 28 * 28
    image_content = {"type": "image", "image": str(image_path)}
    if max_pixels > 0:
        image_content["max_pixels"] = max_pixels
    messages = [
        {
            "role": "user",
            "content": [image_content, {"type": "text", "text": sample["prompt"]}],
        }
    ]
    literal = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB")
    original_size = list(image.size)
    if max_pixels > 0 and image.width * image.height > max_pixels:
        scale = (max_pixels / float(image.width * image.height)) ** 0.5
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.BICUBIC,
        )
    batch = processor(
        text=[literal],
        images=[image],
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    return (
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in dict(batch).items()},
        {
            "prompt_sha256": sha256(literal.encode("utf-8")).hexdigest(),
            "original_image_size": original_size,
            "processed_image_size": list(image.size),
            "max_image_tokens": max_image_tokens,
            "max_pixels": max_pixels,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attention-implementation", default="sdpa")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("label-runtime diagnostic requires a scheduled GPU")

    # Match reference/dvr_qwen/runtime.py, rather than the stricter BP-1
    # diagnostic settings that disabled TF32 and forced deterministic kernels.
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        revision=args.revision,
        local_files_only=True,
        use_fast=False,
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        revision=args.revision,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attention_implementation,
        device_map="auto",
    ).eval()
    wrapped = BinaryQwen25VL(base)
    device = next(base.parameters()).device

    prior = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
    failing = [row for row in prior["records"] if not row["passed"]]
    rows = []
    for prior_row in failing:
        record_path = Path(prior_row["record_file"])
        record = json.loads(record_path.read_text(encoding="utf-8"))
        sample = record["sample"]
        inputs, input_record = label_runtime_inputs(processor, sample, device)
        prepared = build_binary_inputs(wrapped, inputs)
        geometry = {
            "text_tokens": int(prepared.text_valid_mask.sum().item()),
            "visual_tokens": int(prepared.visual_valid_mask.sum().item()),
            "full_prompt_tokens": int(prepared.full_attention_mask.sum().item()),
        }
        all_on = [1] * len(wrapped.decoder.layers)
        all_off = [0] * len(wrapped.decoder.layers)
        best = record.get("mcts", {}).get("best_mask")
        routes = {"all_on": all_on, "all_off": all_off}
        if isinstance(best, list) and len(best) == len(all_on):
            routes["best_mask"] = [int(value) for value in best]

        policy = record["runtime"]["generation_policy"]
        route_rows = {}
        for name, mask in routes.items():
            output = binary_greedy_generate(
                wrapped,
                inputs,
                mask,
                max_new_tokens=int(sample["max_new_tokens"]),
                eos_token_ids=policy["eos_token_ids"],
                repetition_penalty=float(policy["repetition_penalty"]),
                prepared_inputs=prepared,
            )
            actual = output.generated_ids[0].detach().cpu().tolist()
            candidate = cached_candidate(record, mask)
            cached = None if candidate is None else candidate.get("generated_ids")
            cached_geometry = None
            if candidate is not None:
                cached_geometry = {
                    key: candidate.get(key)
                    for key in ("text_tokens", "visual_tokens", "full_prompt_tokens")
                }
            route_rows[name] = {
                "generated_ids": actual,
                "cached_generated_ids": cached,
                "cached_token_ids_match": actual == cached,
                "cached_geometry": cached_geometry,
            }
        rows.append(
            {
                "uid": sample["uid"],
                "record_file": str(record_path),
                "record_sha256": file_sha256(record_path),
                "input": input_record,
                "replay_geometry": geometry,
                "geometry_matches_cache": all(
                    item["cached_geometry"] == geometry
                    for item in route_rows.values()
                    if item["cached_geometry"] is not None
                ),
                "routes": route_rows,
                "all_cached_routes_match": all(
                    item["cached_token_ids_match"] for item in route_rows.values()
                ),
            }
        )

    report = {
        "schema_version": 1,
        "scientific_outcomes_computed": False,
        "diagnostic_scope": "five previously failing BP-1 fixtures only",
        "reference_contract": "reference/dvr_qwen/runtime.py portable fallback",
        "model_revision": args.revision,
        "transformers_version": transformers_version,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "attention_implementation": args.attention_implementation,
        "dtype": "bfloat16",
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "records": rows,
        "all_geometry_matches_cache": all(row["geometry_matches_cache"] for row in rows),
        "all_cached_routes_match": all(row["all_cached_routes_match"] for row in rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{file_sha256(output_path)}  {output_path.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
