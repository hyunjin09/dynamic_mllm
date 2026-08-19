#!/usr/bin/env python3
"""Localize the first native/reference/current binary-executor divergence.

This is a bounded BP-1 engineering diagnostic. It never evaluates task
correctness or predictor outcomes; it compares one frozen fixture's inputs and
intermediate tensors under ALL-ON and one cached mixed route.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import types

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, __version__ as transformers_version
from transformers.masking_utils import create_causal_mask

from binary_policy.executor import BinaryQwen25VL
from binary_policy.executor.inputs import build_binary_inputs, resolve_decoder, scatter_streams
from binary_policy.executor.layers import visual_off_layer, visual_on_layer
from binary_policy.executor.masks import full_causal_mask


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    return float((left.float() - right.float()).abs().max().item())


def indexed_max_abs(left: torch.Tensor, right: torch.Tensor, indices: torch.Tensor) -> float:
    selected = indices[0][indices[0] >= 0].to(left.device)
    return max_abs(left[:, selected], right[:, selected])


def tensor_contract(left: torch.Tensor, right: torch.Tensor) -> dict:
    return {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "equal": bool(torch.equal(left.to(right.device), right)),
        "max_abs": max_abs(left.to(right.device), right),
    }


def load_reference_modules(project_root: Path):
    """Expose the vendored reference tree under its original package name."""
    package = types.ModuleType("dvr_qwen")
    package.__path__ = [str(project_root / "reference" / "binary_action_qwen")]
    sys.modules["dvr_qwen"] = package
    from dvr_qwen.core.binary_layer import forward_text_only_layer, forward_visual_on_layer
    from dvr_qwen.core.split_scatter import build_binary_dvrc_inputs, scatter_to_full

    return build_binary_dvrc_inputs, scatter_to_full, forward_visual_on_layer, forward_text_only_layer


def prepare(processor, record: dict, device: torch.device):
    sample = record["sample"]
    image = Image.open(sample["local_image_path"]).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample["prompt"]}]}]
    literal = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[literal], images=[image], padding=True, return_tensors="pt")
    return literal, {key: value.to(device) if torch.is_tensor(value) else value for key, value in dict(batch).items()}


def trace_current(model, meta, route: list[int]):
    decoder = resolve_decoder(model)
    text_states = meta.text_states
    visual_states = meta.visual_states
    rows = []
    full_states = [scatter_streams(text_states, visual_states, meta).detach()]
    for layer_index, layer in enumerate(decoder.layers):
        pre = scatter_streams(text_states.to(next(layer.parameters()).device), visual_states, meta)
        function = visual_on_layer if route[layer_index] else visual_off_layer
        text_states, visual_states, _ = function(
            model,
            layer,
            text_states,
            visual_states,
            meta,
            layer_index=layer_index,
        )
        post = scatter_streams(text_states, visual_states, meta)
        rows.append({"layer": layer_index, "action": int(route[layer_index]), "pre": pre.detach(), "post": post.detach()})
        full_states.append(post.detach())
    return rows, full_states


def trace_reference(base, meta, route, scatter_to_full, ref_on, ref_off):
    decoder = base.model.language_model
    text_states = meta.text_states
    visual_states = meta.visual_states
    rows = []
    full_states = [scatter_to_full(text_states, visual_states, meta).detach()]
    for layer_index, layer in enumerate(decoder.layers):
        pre = scatter_to_full(text_states.to(next(layer.parameters()).device), visual_states, meta)
        function = ref_on if route[layer_index] else ref_off
        text_states, visual_states, _ = function(
            decoder,
            layer,
            text_states,
            visual_states,
            meta,
            layer_idx=layer_index,
            use_cache=False,
        )
        post = scatter_to_full(text_states, visual_states, meta)
        rows.append({"layer": layer_index, "action": int(route[layer_index]), "pre": pre.detach(), "post": post.detach()})
        full_states.append(post.detach())
    return rows, full_states


def component_trace(layer, hidden, attention_mask, position_embeddings, cache_position):
    captured = {}

    def save(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value.detach().clone()
        return hook

    hooks = [
        layer.input_layernorm.register_forward_hook(save("input_layernorm")),
        layer.self_attn.register_forward_hook(save("attention")),
        layer.post_attention_layernorm.register_forward_hook(save("post_attention_layernorm")),
        layer.mlp.register_forward_hook(save("mlp")),
    ]
    try:
        output = layer(
            hidden_states=hidden,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
        )[0]
        captured["layer_output"] = output.detach().clone()
    finally:
        for hook in hooks:
            hook.remove()
    return captured


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--record-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attention-implementation", default="sdpa")
    args = parser.parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("equivalence diagnostic requires a scheduled GPU")

    torch.manual_seed(20260809)
    torch.cuda.manual_seed_all(20260809)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    processor = AutoProcessor.from_pretrained(args.model_path, revision=args.revision, local_files_only=True, use_fast=False)
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
    record_path = Path(args.record_file)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    literal, inputs = prepare(processor, record, device)
    if "mm_token_type_ids" not in inputs:
        raise RuntimeError("reference executor requires processor mm_token_type_ids")

    build_ref, scatter_ref, ref_on, ref_off = load_reference_modules(Path(__file__).resolve().parents[1])
    current = build_binary_inputs(wrapped, inputs)
    reference = build_ref(base.model, **inputs)
    input_checks = {
        "prompt_sha256": sha256(literal.encode("utf-8")).hexdigest(),
        "input_ids": tensor_contract(inputs["input_ids"], inputs["input_ids"]),
        "full_inputs_embeds": tensor_contract(current.full_inputs_embeds, reference.full_inputs_embeds),
        "text_states": tensor_contract(current.text_states, reference.text_states),
        "visual_states": tensor_contract(current.visual_states, reference.visual_states),
        "text_indices": tensor_contract(current.text_indices, reference.text_indices),
        "visual_indices": tensor_contract(current.visual_indices, reference.visual_indices),
        "full_position_ids": tensor_contract(current.full_position_ids, reference.full_position_ids),
        "text_position_ids": tensor_contract(current.text_position_ids, reference.text_position_ids),
        "visual_position_ids": tensor_contract(current.visual_position_ids, reference.visual_position_ids),
        "attention_mask": tensor_contract(current.full_attention_mask, reference.full_attention_mask),
    }

    num_layers = len(base.model.language_model.layers)
    all_on = [1] * num_layers
    mixed = record.get("mcts", {}).get("best_mask")
    if not isinstance(mixed, list) or len(mixed) != num_layers or all(int(value) == 1 for value in mixed):
        raise RuntimeError("fixture must contain a nontrivial cached best mask")

    with torch.inference_mode():
        base.model.rope_deltas = None
        native = base.model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
        native_states = list(native.hidden_states)
        current_all_rows, current_all = trace_current(wrapped, current, all_on)
        reference_all_rows, reference_all = trace_reference(base, reference, all_on, scatter_ref, ref_on, ref_off)
        current_mix_rows, current_mix = trace_current(wrapped, current, mixed)
        reference_mix_rows, reference_mix = trace_reference(base, reference, mixed, scatter_ref, ref_on, ref_off)

    layer_rows = []
    first_current_reference = None
    first_current_native = None
    for layer_index in range(num_layers):
        current_ref_pre = max_abs(current_all_rows[layer_index]["pre"], reference_all_rows[layer_index]["pre"])
        current_ref_post = max_abs(current_all_rows[layer_index]["post"], reference_all_rows[layer_index]["post"])
        current_native_pre = max_abs(current_all_rows[layer_index]["pre"], native_states[layer_index])
        current_native_post = None
        if layer_index + 1 < num_layers:
            current_native_post = max_abs(current_all_rows[layer_index]["post"], native_states[layer_index + 1])
        if first_current_reference is None and (current_ref_pre != 0.0 or current_ref_post != 0.0):
            first_current_reference = layer_index
        if first_current_native is None and (current_native_pre != 0.0 or (current_native_post or 0.0) != 0.0):
            first_current_native = layer_index
        layer_rows.append(
            {
                "layer": layer_index,
                "current_reference_all_on_pre_max_abs": current_ref_pre,
                "current_reference_all_on_post_max_abs": current_ref_post,
                "current_native_all_on_pre_max_abs": current_native_pre,
                "current_native_all_on_post_max_abs": current_native_post,
                "current_reference_mixed_pre_max_abs": max_abs(current_mix_rows[layer_index]["pre"], reference_mix_rows[layer_index]["pre"]),
                "current_reference_mixed_post_max_abs": max_abs(current_mix_rows[layer_index]["post"], reference_mix_rows[layer_index]["post"]),
                "mixed_action": int(mixed[layer_index]),
            }
        )

    # Compare the first native layer under the native mask dispatcher and the
    # explicit reference/current full causal mask. This identifies whether the
    # earliest ALL-ON divergence begins inside attention or later in the block.
    decoder = base.model.language_model
    hidden0 = native_states[0]
    cache_position = torch.arange(hidden0.shape[1], device=hidden0.device)
    mask_kwargs = {
        "config": decoder.config,
        "inputs_embeds": hidden0,
        "attention_mask": inputs["attention_mask"].to(hidden0.device),
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": None,
    }
    native_mask = create_causal_mask(**mask_kwargs)
    explicit_mask = full_causal_mask(inputs["attention_mask"], dtype=hidden0.dtype, device=hidden0.device)
    position_embeddings = decoder.rotary_emb(hidden0, current.full_position_ids.to(hidden0.device))
    with torch.inference_mode():
        native_components = component_trace(decoder.layers[0], hidden0, native_mask, position_embeddings, cache_position)
        explicit_components = component_trace(decoder.layers[0], hidden0, explicit_mask, position_embeddings, cache_position)
    component_diffs = {key: tensor_contract(explicit_components[key], native_components[key]) for key in native_components}

    norm_device = next(decoder.norm.parameters()).device
    current_final = decoder.norm(current_all[-1].to(norm_device))
    reference_final = decoder.norm(reference_all[-1].to(norm_device))
    native_final = native_states[-1]
    lm_device = next(base.lm_head.parameters()).device
    current_logits = base.lm_head(current_final.to(lm_device))
    reference_logits = base.lm_head(reference_final.to(lm_device))
    native_logits = base.lm_head(native_final.to(lm_device))
    last_text = current.text_indices[0][current.text_valid_mask[0]][-1]
    final_checks = {
        "current_reference_hidden": tensor_contract(current_final, reference_final),
        "current_native_hidden": tensor_contract(current_final, native_final),
        "reference_native_hidden": tensor_contract(reference_final, native_final),
        "current_reference_logits": tensor_contract(current_logits, reference_logits),
        "current_native_logits_all": tensor_contract(current_logits, native_logits),
        "current_native_logits_text_max_abs": indexed_max_abs(current_logits, native_logits, current.text_indices),
        "current_native_logits_last_text_max_abs": max_abs(current_logits[:, last_text:last_text + 1], native_logits[:, last_text:last_text + 1]),
    }

    report = {
        "scientific_outcomes_computed": False,
        "fixture_uid": record["sample"]["uid"],
        "record_file": str(record_path),
        "record_sha256": file_sha256(record_path),
        "model_revision": args.revision,
        "transformers_version": transformers_version,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "attention_implementation": args.attention_implementation,
        "input_checks": input_checks,
        "mixed_mask": [int(value) for value in mixed],
        "first_current_reference_all_on_divergence_layer": first_current_reference,
        "first_current_native_all_on_divergence_layer": first_current_native,
        "layer_checks": layer_rows,
        "layer0_native_mask_is_none": native_mask is None,
        "layer0_native_mask_shape": None if native_mask is None else list(native_mask.shape),
        "layer0_explicit_mask_shape": list(explicit_mask.shape),
        "layer0_component_differences_explicit_vs_native_mask": component_diffs,
        "final_checks": final_checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{file_sha256(output_path)}  {output_path.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
