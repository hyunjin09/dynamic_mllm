#!/usr/bin/env python3
"""GPU preflight required before binary-policy labels or training are opened.

The script is intentionally small and bounded. It verifies native all-ON
parity, split/scatter identity, determinism, per-layer cache geometry, and exact
reproduction of cached MCTS route token IDs on explicitly supplied fixtures.
Run it only through ``infra/gpu_scheduler.py --gpus 1``.
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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, __version__ as transformers_version

from binary_policy.executor import BinaryQwen25VL, binary_greedy_generate, binary_route_forward
from binary_policy.executor.inputs import build_binary_inputs, resolve_decoder, scatter_streams
from binary_policy.executor.layers import call_decoder_layer, visual_off_layer
from binary_policy.executor.masks import additive_causal_mask


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    return float((left.float() - right.float()).abs().max().item())


def indexed_max_abs(left: torch.Tensor, right: torch.Tensor, indices: torch.Tensor) -> float:
    """Compare sequence logits only at the supplied batch-one row indices."""
    selected = indices[0][indices[0] >= 0].to(left.device)
    return max_abs(left[:, selected], right[:, selected])


def prepare(processor, sample: dict, device: torch.device):
    image_path = Path(sample["local_image_path"])
    max_image_tokens = int(sample.get("max_image_tokens") or 0)
    image_content = {"type": "image", "image": str(image_path)}
    processor_kwargs = {}
    if max_image_tokens > 0:
        max_pixels = max_image_tokens * 28 * 28
        image_content["max_pixels"] = max_pixels
        # Keep the original pixels until the pinned Qwen processor applies its
        # factor-aligned smart-resize.  A manual area-only resize changes the
        # patch grid and MRoPE positions for portrait documents.
        processor_kwargs["max_pixels"] = max_pixels
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [image_content, {"type": "text", "text": sample["prompt"]}],
        }
    ]
    literal = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(
        text=[literal],
        images=[image],
        videos=None,
        padding=True,
        return_tensors="pt",
        return_mm_token_type_ids=True,
        **processor_kwargs,
    )
    return literal, {key: value.to(device) if torch.is_tensor(value) else value for key, value in dict(batch).items()}


def candidate_for(record: dict, mask: list[int]):
    for candidate in record.get("candidate_executions", []):
        if candidate.get("visual_on_mask") == mask:
            return candidate
    return None


def file_sha256(path: Path) -> str:
    digest = sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--record-file", action="append", default=[])
    parser.add_argument("--record-manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--full-parity-atol", type=float, default=0.005)
    args = parser.parse_args()
    record_files = list(args.record_file)
    if args.record_manifest:
        manifest = json.loads(Path(args.record_manifest).read_text(encoding="utf-8"))
        if manifest.get("passed") is not True:
            raise RuntimeError("executor fixture manifest did not pass")
        record_files.extend(str(row["record_file"]) for row in manifest.get("records", []))
    if not record_files:
        raise ValueError("at least one --record-file or --record-manifest is required")
    if len(record_files) != len(set(record_files)):
        raise ValueError("duplicate executor fixture record")
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("binary executor preflight requires a scheduled GPU")
    torch.manual_seed(20260809)
    torch.cuda.manual_seed_all(20260809)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    processor = AutoProcessor.from_pretrained(
        args.model_path, revision=args.revision, local_files_only=True, use_fast=False
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
    rows = []
    all_pass = True
    for record_name in record_files:
        path = Path(record_name)
        record = json.loads(path.read_text(encoding="utf-8"))
        sample = record["sample"]
        _, inputs = prepare(processor, sample, device)
        prepared = build_binary_inputs(wrapped, inputs)
        reconstructed = scatter_streams(prepared.text_states, prepared.visual_states, prepared)
        split_scatter_error = max_abs(reconstructed, prepared.full_inputs_embeds)
        num_layers = len(wrapped.decoder.layers)
        full_mask = [1] * num_layers
        binary_full = binary_route_forward(wrapped, inputs, full_mask, prepared_inputs=prepared)
        with torch.inference_mode():
            base.rope_deltas = None
            native = base(**inputs, use_cache=False, return_dict=True).logits
        full_logit_error = max_abs(binary_full.logits, native)
        full_text_logit_error = indexed_max_abs(
            binary_full.logits,
            native,
            prepared.text_indices,
        )
        full_visual_logit_error = indexed_max_abs(
            binary_full.logits,
            native,
            prepared.visual_indices,
        )
        last_text_index = prepared.text_indices[0][prepared.text_valid_mask[0]][-1]
        full_last_text_logit_error = max_abs(
            binary_full.logits[:, last_text_index : last_text_index + 1],
            native[:, last_text_index : last_text_index + 1],
        )
        repeat = binary_route_forward(wrapped, inputs, full_mask, prepared_inputs=prepared)
        deterministic_error = max_abs(binary_full.logits, repeat.logits)
        decoder = resolve_decoder(wrapped)
        first_layer = decoder.layers[0]
        off_text, off_visual, _ = visual_off_layer(
            wrapped,
            first_layer,
            prepared.text_states,
            prepared.visual_states,
            prepared,
            layer_index=0,
        )
        layer_device = next(first_layer.parameters()).device
        compact_text = prepared.text_states.to(layer_device)
        compact_mask = additive_causal_mask(
            prepared.text_valid_mask,
            prepared.text_indices,
            dtype=compact_text.dtype,
            device=layer_device,
        )
        compact_positions = decoder.rotary_emb(
            compact_text, prepared.text_position_ids.to(layer_device)
        )
        compact_oracle = call_decoder_layer(
            first_layer,
            compact_text,
            attention_mask=compact_mask,
            position_embeddings=compact_positions,
            cache=None,
            use_cache=False,
        )[0]
        off_oracle_error = max_abs(off_text, compact_oracle)
        off_visual_bypass_error = max_abs(off_visual, prepared.visual_states)
        base.rope_deltas = None
        native_generated = base.generate(
            **inputs,
            max_new_tokens=int(sample["max_new_tokens"]),
            do_sample=False,
            use_cache=True,
        )
        native_new_ids = native_generated[0, inputs["input_ids"].shape[1] :].detach().cpu().tolist()
        routes = {"all_on": full_mask, "all_off": [0] * num_layers}
        best = record.get("mcts", {}).get("best_mask")
        if isinstance(best, list) and len(best) == num_layers:
            routes["best_mask"] = best
        route_rows = {}
        for name, mask in routes.items():
            generation = binary_greedy_generate(
                wrapped,
                inputs,
                mask,
                max_new_tokens=int(sample["max_new_tokens"]),
                prepared_inputs=prepared,
            )
            candidate = candidate_for(record, mask)
            expected_ids = None if candidate is None else candidate.get("generated_ids")
            actual_ids = generation.generated_ids[0].detach().cpu().tolist()
            full_rows = int(prepared.full_attention_mask.long().sum().item())
            text_rows = int(prepared.text_valid_mask.long().sum().item())
            expected_cache_lengths = [full_rows if value else text_rows for value in mask]
            # ``generation.prefill.cache`` is mutated by decode. Layer stats
            # capture the actual prefill geometry before generated tokens are
            # appended and are the appropriate object for this check.
            actual_cache_lengths = [item.cache_rows for item in generation.prefill.layer_stats]
            final_cache_lengths = generation.prefill.cache.lengths() if generation.prefill.cache else None
            route_rows[name] = {
                "mask": mask,
                "generated_ids": actual_ids,
                "cached_generated_ids": expected_ids,
                "cached_token_ids_match": expected_ids is not None and actual_ids == expected_ids,
                "cache_lengths": actual_cache_lengths,
                "final_cache_lengths": final_cache_lengths,
                "expected_cache_lengths": expected_cache_lengths,
                "cache_lengths_match": actual_cache_lengths == expected_cache_lengths,
            }
        arbitrary_repeat_error = 0.0
        if "best_mask" in routes:
            first = binary_route_forward(wrapped, inputs, routes["best_mask"], prepared_inputs=prepared)
            second = binary_route_forward(wrapped, inputs, routes["best_mask"], prepared_inputs=prepared)
            arbitrary_repeat_error = max_abs(first.logits, second.logits)
        row_pass = (
            split_scatter_error == 0.0
            and full_logit_error <= args.full_parity_atol
            and deterministic_error == 0.0
            and off_oracle_error == 0.0
            and off_visual_bypass_error == 0.0
            and arbitrary_repeat_error == 0.0
            and route_rows["all_on"]["generated_ids"] == native_new_ids
            and all(item["cached_token_ids_match"] and item["cache_lengths_match"] for item in route_rows.values())
        )
        all_pass &= row_pass
        rows.append(
            {
                "uid": sample["uid"],
                "record_file": str(path),
                "record_sha256": file_sha256(path),
                "split_scatter_max_abs": split_scatter_error,
                "all_on_native_logit_max_abs": full_logit_error,
                "all_on_native_text_logit_max_abs": full_text_logit_error,
                "all_on_native_visual_logit_max_abs": full_visual_logit_error,
                "all_on_native_last_text_logit_max_abs": full_last_text_logit_error,
                "repeat_logit_max_abs": deterministic_error,
                "arbitrary_route_repeat_logit_max_abs": arbitrary_repeat_error,
                "off_compacted_text_oracle_max_abs": off_oracle_error,
                "off_visual_bypass_max_abs": off_visual_bypass_error,
                "native_all_on_generated_ids": native_new_ids,
                "binary_all_on_matches_native_generation": route_rows["all_on"]["generated_ids"] == native_new_ids,
                "routes": route_rows,
                "passed": row_pass,
            }
        )
    report = {
        "passed": all_pass,
        "scientific_outcomes_computed": False,
        "model_revision": args.revision,
        "attention_implementation": args.attention_implementation,
        "dtype": "bfloat16",
        "transformers_version": transformers_version,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "full_parity_atol": args.full_parity_atol,
        "records": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{file_sha256(output_path)}  {output_path.name}\n", encoding="utf-8"
    )
    if not all_pass:
        raise RuntimeError("binary executor preflight failed; training remains blocked")


if __name__ == "__main__":
    main()
