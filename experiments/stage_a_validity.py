from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import traceback
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
import transformers
import yaml
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from tools.research_analysis.v2.activation_plausibility import (
    add_geometry_metrics,
    basic_activation_row,
    deterministic_token_sample,
)
from audit.sample_manifest import select_stage_a_samples, write_jsonl
from audit.token_layout import describe_token_layout
from interventions.four_state import FOUR_STATES, LayerCapture, run_cached_state
from interventions.read_path import ReadInterventionCache
from scoring.benchmark_metrics import normalize_exact, score_record
from scoring.option_scores import score_appended_content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the bounded Stage A validity gate.")
    parser.add_argument("--config", default="configs/stage_a.yaml")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-sample-id")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def prepare_prompt(processor, record: dict[str, Any], device: torch.device) -> tuple[str, dict[str, Any]]:
    image = Image.open(record["local_image_path"]).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": str(record["prompt"])},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    return text, to_device(dict(batch), device)


def capture_forward(causal_lm, inputs: dict[str, Any], layer_indices: list[int]):
    captures = [LayerCapture(causal_lm.model.layers[index], index) for index in layer_indices]
    with ExitStack() as stack:
        for capture in captures:
            stack.enter_context(capture)
        outputs = causal_lm(**inputs, use_cache=False, return_dict=True)
    return outputs, {capture.layer_index: capture.context() for capture in captures}


def fresh_generate(causal_lm, inputs: dict[str, Any], max_new_tokens: int) -> torch.Tensor:
    causal_lm.rope_deltas = None
    return causal_lm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )


def no_op_instrumented_generate(causal_lm, inputs: dict[str, Any], layer_index: int, max_new_tokens: int):
    calls = {"pre": 0, "post": 0}

    def pre_hook(module, args, kwargs):
        calls["pre"] += 1

    def post_hook(module, args, kwargs, output):
        calls["post"] += 1

    layer = causal_lm.model.layers[layer_index]
    pre = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    post = layer.register_forward_hook(post_hook, with_kwargs=True)
    try:
        generated = fresh_generate(causal_lm, inputs, max_new_tokens)
    finally:
        pre.remove()
        post.remove()
    return generated, calls


def decode_new_tokens(processor, generated: torch.Tensor, prompt_length: int) -> str:
    return processor.batch_decode(
        generated[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def max_abs_difference(candidate: torch.Tensor, reference: torch.Tensor, chunk_size: int = 1_000_000) -> float:
    if candidate.shape != reference.shape:
        raise ValueError(f"Comparison shape mismatch: {candidate.shape} != {reference.shape}")
    candidate_flat = candidate.reshape(-1)
    reference_flat = reference.reshape(-1)
    maximum = 0.0
    for start in range(0, candidate_flat.numel(), chunk_size):
        stop = min(start + chunk_size, candidate_flat.numel())
        chunk_max = (
            candidate_flat[start:stop].float() - reference_flat[start:stop].float()
        ).abs().max().item()
        maximum = max(maximum, float(chunk_max))
    return maximum


def rms_ratio(candidate: torch.Tensor, reference: torch.Tensor, chunk_size: int = 1_000_000) -> float:
    if candidate.shape != reference.shape:
        raise ValueError(f"Comparison shape mismatch: {candidate.shape} != {reference.shape}")
    candidate_flat = candidate.reshape(-1)
    reference_flat = reference.reshape(-1)
    difference_square_sum = 0.0
    reference_square_sum = 0.0
    for start in range(0, candidate_flat.numel(), chunk_size):
        stop = min(start + chunk_size, candidate_flat.numel())
        candidate_chunk = candidate_flat[start:stop].float()
        reference_chunk = reference_flat[start:stop].float()
        difference_square_sum += float((candidate_chunk - reference_chunk).square().sum().item())
        reference_square_sum += float(reference_chunk.square().sum().item())
    return float((difference_square_sum / max(reference_square_sum, 1e-24)) ** 0.5)


def finite_and_abs_max(tensor: torch.Tensor, chunk_size: int = 1_000_000) -> tuple[bool, float]:
    flat = tensor.reshape(-1)
    finite = True
    maximum = 0.0
    for start in range(0, flat.numel(), chunk_size):
        chunk = flat[start : start + chunk_size]
        finite = finite and bool(torch.isfinite(chunk).all().item())
        maximum = max(maximum, float(chunk.abs().max().item()))
    return finite, maximum


def combined_candidate_inputs(causal_lm, processor, prompt_inputs, layer_zero_state, candidate: str):
    answer_ids = processor.tokenizer(candidate, add_special_tokens=False, return_tensors="pt").input_ids.to(
        layer_zero_state.device
    )
    if answer_ids.shape[1] < 1:
        raise ValueError(f"Candidate tokenized to an empty sequence: {candidate!r}")
    prompt_ids = prompt_inputs["input_ids"]
    combined_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    combined_attention = torch.cat(
        [prompt_inputs["attention_mask"], torch.ones_like(answer_ids, device=prompt_ids.device)], dim=1
    )
    answer_embeds = causal_lm.model.embed_tokens(answer_ids)
    combined_embeds = torch.cat([layer_zero_state, answer_embeds], dim=1)
    position_ids, _ = causal_lm.get_rope_index(
        input_ids=combined_ids,
        image_grid_thw=prompt_inputs.get("image_grid_thw"),
        attention_mask=combined_attention,
    )
    return combined_ids, combined_attention, combined_embeds, position_ids, answer_ids.shape[1]


def option_parity(
    causal_lm,
    processor,
    record,
    prompt_inputs,
    layer_zero_state,
    primary_layer,
    visual_mask,
):
    rows = []
    baseline_scores: list[float] = []
    reference_scores: list[float] = []
    for candidate in record["parity_only_candidates"]:
        combined_ids, combined_attention, combined_embeds, position_ids, answer_length = combined_candidate_inputs(
            causal_lm, processor, prompt_inputs, layer_zero_state, candidate
        )
        capture = LayerCapture(causal_lm.model.layers[primary_layer], primary_layer)
        with capture:
            base = causal_lm.model(
                inputs_embeds=combined_embeds,
                position_ids=position_ids,
                attention_mask=combined_attention,
                use_cache=False,
                return_dict=True,
            )
        base_logits = causal_lm.lm_head(base.last_hidden_state)
        combined_visual_mask = torch.cat(
            [visual_mask, torch.zeros((1, answer_length), dtype=torch.bool, device=visual_mask.device)], dim=1
        )
        candidate_read_cache = ReadInterventionCache()
        full = run_cached_state(
            causal_lm,
            capture.context(),
            combined_visual_mask,
            "FULL",
            "full",
            "full",
            candidate_read_cache,
        )
        base_score = score_appended_content(
            base_logits, combined_ids, prompt_inputs["input_ids"].shape[1], answer_length
        )
        instrumented_score = score_appended_content(
            full.logits, combined_ids, prompt_inputs["input_ids"].shape[1], answer_length
        )
        full_logit_max_abs = max_abs_difference(full.logits, base_logits)
        del full

        reference = run_cached_state(
            causal_lm,
            capture.context(),
            combined_visual_mask,
            "REFERENCE_FULL",
            "reference",
            "full",
            candidate_read_cache,
        )
        reference_score = score_appended_content(
            reference.logits, combined_ids, prompt_inputs["input_ids"].shape[1], answer_length
        )
        baseline_scores.append(base_score.mean_logprob)
        reference_scores.append(reference_score.mean_logprob)
        rows.append(
            {
                "sample_id": record["id"],
                "candidate": candidate,
                "answer_length": answer_length,
                "logit_max_abs": full_logit_max_abs,
                "score_abs": abs(instrumented_score.mean_logprob - base_score.mean_logprob),
                "baseline_score": base_score.mean_logprob,
                "instrumented_score": instrumented_score.mean_logprob,
                "reference_score": reference_score.mean_logprob,
                "reference_score_abs": abs(reference_score.mean_logprob - base_score.mean_logprob),
                "reference_logit_rms_ratio": rms_ratio(reference.logits, base_logits),
            }
        )
        del base, base_logits, reference
    baseline_order = sorted(range(len(baseline_scores)), key=baseline_scores.__getitem__, reverse=True)
    reference_order = sorted(range(len(reference_scores)), key=reference_scores.__getitem__, reverse=True)
    for row in rows:
        row["reference_candidate_order_match"] = baseline_order == reference_order
    return rows


def architecture_markdown(runtime: dict[str, Any], summary: dict[str, Any]) -> str:
    return f"""# Qwen2.5-VL Stage A Architecture Causal Graph

## Verified Runtime

- Model: `{runtime['model_id']}` revision `{runtime['revision']}`
- Runtime: PyTorch `{runtime['torch_version']}`, Transformers `{runtime['transformers_version']}`, CUDA `{runtime['cuda_version']}`
- Decoder: {runtime['num_hidden_layers']} pre-norm layers, hidden size {runtime['hidden_size']}, {runtime['num_attention_heads']} query heads, {runtime['num_key_value_heads']} KV heads ({runtime['num_key_value_groups']} GQA groups)
- Decoder attention backend: {runtime['attention_backend']}; vision-encoder backend: {runtime['vision_attention_backend']}; dropout disabled in evaluation; decoder output projection bias: {runtime['o_proj_has_bias']}
- Frozen parameters: `{runtime['all_parameters_frozen']}`

## Actual Token/Mask Order

The processor produces interleaved rows, not a literal contiguous `[V; T]`
sequence: system/control prefix -> vision-start control -> contiguous image rows
-> vision-end control -> question/instruction -> assistant-generation prefix.
`V_l` and `T_l` are therefore implemented as row masks over this actual order.
All audited layouts had contiguous image rows: `{summary['all_visual_rows_contiguous']}`.

The decoder mask is causal. Image rows occur before the question, so visual
queries cannot attend to later question/assistant rows (empirical maximum future
attention mass `{summary['visual_future_attention_mass_max']:.8g}`). They can
attend earlier system/control rows. Post-image text queries can attend image
rows. Consequently, earlier text can affect visual WRITE across layers, while
the later question cannot affect already-computed image rows in this single-image
prompt layout.

## Per-Layer Causal Graph

```text
cached H_l (identical for all four states)
  |-- RMSNorm --> Q,K,V + fixed causal mask + multimodal RoPE/GQA
  |                  |
  |                  +-- visual-value context for text queries
  |                         -> o_proj -> delta_read_l
  |-- residual + attention output (READ hook is before this addition)
  |-- RMSNorm -> row-wise MLP -> residual
  `-- H_l+1 (WRITE hook is the complete decoder-layer output)
```

The same-layer Q/K/V projections are all computed from the normalized cached
`H_l`, before the layer writes `H_l+1`; same-layer text READ therefore uses
pre-WRITE visual K/V. The primary READ OFF subtracts only the projected visual
value contribution from text-query attention outputs while retaining the
original softmax weights and every non-visual value path. It does not mask edges
or renormalize attention. The actual decoder uses the unchanged Transformers
stock eager forward. The declared visual-value residual is evaluated in
bounded query chunks with the same Q/K/V, mRoPE, GQA expansion, causal mask,
and fixed softmax; this does not replace or alter the model's FULL path.
Separately split visual/nonvisual BF16 recomposition is retained as a numerical
diagnostic. Because the model runs in BF16, the executed OFF
state is rounded once to BF16. Exact add-back uses the representable
`FULL - OFF` residual; its difference from the ideal visual path is separately
audited against the local round-to-nearest half-ULP and reported relative to
the ideal-path RMS. WRITE OFF replaces only image rows at the decoder
layer output with their cached pre-layer rows; current-layer text rows are left
unchanged.

## KV-Cache Consequences

At prefill, the target layer cache is formed from pre-layer K/V and is unchanged
by its READ output subtraction. READ/WRITE changes propagate into later-layer
prompt states and therefore their caches. Autoregressive decoding is outside
the intervention hook: Stage A generation parity used the normal cache path and
confirmed no-op instrumentation, while reconstruction/four-state identities
were evaluated on the cache-free prompt pass followed by the unchanged decoder
layer suffix.

## Validated Hooks

- READ: `model.layers[l].self_attn` output, after bias-free `o_proj`, before
  residual addition.
- WRITE: `model.layers[l]` output image rows, after attention, MLP, norms, and
  both residual additions.
- Layers validated: {summary['validated_layers']}.
- Maximum READ visual/non-visual decomposition error: `{summary['read_decomposition_max_abs']:.8g}`.
- Maximum split-recomposition/stock-eager attention RMS ratio: `{summary['read_reference_attention_rms_ratio']:.8g}`.
- Maximum split-recomposition/stock-eager suffix-logit RMS ratio: `{summary['read_reference_logit_rms_ratio']:.8g}`.
- Maximum cached-prestate injection error: `{summary['prestate_injection_max_abs']:.8g}`.
"""


def execute(args: argparse.Namespace) -> int:
    config = load_yaml(Path(args.config))
    model_config = load_yaml(Path(config["model_config"]))
    output_dir = Path(args.output_dir or ("outputs/stage_a_probe" if args.probe_only else config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    set_determinism(int(config["seed"]))

    requested_samples = select_stage_a_samples(
        Path(config["dataset_root"]),
        list(config["benchmarks"]),
        list(config["buckets"]),
        int(config["samples_per_benchmark_bucket"]),
    )
    write_jsonl(output_dir / "stage_a_requested_samples.jsonl", requested_samples)
    exclusions = list(config.get("resource_exclusions") or [])
    excluded_ids = {str(row["sample_id"]) for row in exclusions}
    unknown_exclusions = excluded_ids - {str(row["id"]) for row in requested_samples}
    if unknown_exclusions:
        raise ValueError(f"Configured Stage A exclusions were not selected: {sorted(unknown_exclusions)}")
    samples = [row for row in requested_samples if str(row["id"]) not in excluded_ids]
    write_json(output_dir / "stage_a_resource_exclusions.json", exclusions)
    if args.probe_sample_id:
        samples = [row for row in samples if row["id"] == args.probe_sample_id]
        if len(samples) != 1:
            raise ValueError(f"Probe sample was not selected exactly once: {args.probe_sample_id}")
    elif args.probe_only:
        samples = samples[:1]
    write_jsonl(output_dir / "stage_a_samples.jsonl", samples)

    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(model_config["snapshot_path"], local_files_only=True)
    hf_config = AutoConfig.from_pretrained(model_config["snapshot_path"], local_files_only=True)
    if model_config["decoder_attention_backend"] == model_config["vision_attention_backend"]:
        hf_config._attn_implementation = model_config["decoder_attention_backend"]
    else:
        if model_config["decoder_attention_backend"] != "eager":
            raise ValueError("Mixed Qwen backends support only an eager root decoder")
        hf_config._attn_implementation = {
            "vision_config": model_config["vision_attention_backend"],
        }
    causal_lm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_config["snapshot_path"],
        config=hf_config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    causal_lm.eval()
    causal_lm.requires_grad_(False)
    # Keep the model's stock eager forward. Query chunking applies only to the
    # explicit READ residual computation, not to the FULL model path.
    for layer in causal_lm.model.layers:
        layer.self_attn.stage_a_query_chunk_size = int(
            model_config["decoder_attention_query_chunk_size"]
        )

    primary_layer = int(config["primary_layer"])
    sentinel_layers = [] if args.probe_only else [int(value) for value in config["sentinel_layers"]]
    capture_layers = sorted(set([0, primary_layer] + sentinel_layers))
    first_attention = causal_lm.model.layers[0].self_attn
    runtime = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "model_id": model_config["model_id"],
        "revision": model_config["revision"],
        "snapshot_path": model_config["snapshot_path"],
        "dtype": str(next(causal_lm.parameters()).dtype),
        "attention_backend": causal_lm.config._attn_implementation,
        "vision_attention_backend": causal_lm.config.vision_config._attn_implementation,
        "decoder_attention_class": type(first_attention).__name__,
        "decoder_attention_execution": "transformers_stock_eager",
        "decoder_attention_query_chunk_size": int(model_config["decoder_attention_query_chunk_size"]),
        "vision_attention_class": type(causal_lm.visual.blocks[0].attn).__name__,
        "num_hidden_layers": causal_lm.config.num_hidden_layers,
        "hidden_size": causal_lm.config.hidden_size,
        "num_attention_heads": causal_lm.config.num_attention_heads,
        "num_key_value_heads": causal_lm.config.num_key_value_heads,
        "num_key_value_groups": first_attention.num_key_value_groups,
        "o_proj_has_bias": first_attention.o_proj.bias is not None,
        "all_parameters_frozen": all(not parameter.requires_grad for parameter in causal_lm.parameters()),
    }
    write_json(output_dir / "runtime.json", runtime)

    no_op_rows: list[dict[str, Any]] = []
    read_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    write_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    scoring_rows: list[dict[str, Any]] = []
    option_rows: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []
    sampled_activations: list[dict[str, Any]] = []
    token_layouts: list[dict[str, Any]] = []

    with torch.inference_mode():
        for sample_index, record in enumerate(samples):
            prompt_text, prompt_inputs = prepare_prompt(processor, record, device)
            input_ids = prompt_inputs["input_ids"]
            visual_mask = input_ids == causal_lm.config.image_token_id
            position_ids, _ = causal_lm.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=prompt_inputs.get("image_grid_thw"),
                attention_mask=prompt_inputs.get("attention_mask"),
            )
            layout = describe_token_layout(processor.tokenizer, causal_lm.config, input_ids, position_ids)
            layout.update(
                {
                    "sample_id": record["id"],
                    "benchmark": record["benchmark"],
                    "bucket": record["bucket"],
                    "formatted_prompt": prompt_text,
                }
            )
            token_layouts.append(layout)

            baseline, contexts = capture_forward(causal_lm, prompt_inputs, capture_layers)
            baseline_logits = baseline.logits
            primary_context = contexts[primary_layer]
            primary_read_cache = ReadInterventionCache()
            full = run_cached_state(
                causal_lm,
                primary_context,
                visual_mask,
                "FULL",
                "full",
                "full",
                primary_read_cache,
            )
            full_target_output = full.target_output.detach().clone()
            no_op_rows.append(
                {
                    "sample_id": record["id"],
                    "benchmark": record["benchmark"],
                    "bucket": record["bucket"],
                    "layer": primary_layer,
                    "kind": "prompt",
                    "layer_state_max_abs": float(
                        (full.target_output.float() - primary_context.full_layer_output.float()).abs().max().item()
                    ),
                    "final_logit_max_abs": max_abs_difference(full.logits, baseline_logits),
                    "prestate_injection_max_abs": full.injected_prestate_max_abs,
                }
            )

            for state_name in ("FULL", "IGNORE", "READ_ONLY", "WRITE_ONLY"):
                result = full if state_name == "FULL" else run_cached_state(
                    causal_lm,
                    primary_context,
                    visual_mask,
                    state_name,
                    *FOUR_STATES[state_name],
                    read_cache=primary_read_cache,
                )
                repeat = run_cached_state(
                    causal_lm,
                    primary_context,
                    visual_mask,
                    state_name,
                    *FOUR_STATES[state_name],
                    read_cache=primary_read_cache,
                )
                repeat_error = max_abs_difference(repeat.logits, result.logits)
                result_finite, result_abs_max = finite_and_abs_max(result.logits)
                stability_rows.append(
                    {
                        "sample_id": record["id"],
                        "benchmark": record["benchmark"],
                        "bucket": record["bucket"],
                        "layer": primary_layer,
                        "state": state_name,
                        "repeat_logit_max_abs": repeat_error,
                        "finite": result_finite,
                        "final_logit_abs_max": result_abs_max,
                        "prestate_injection_max_abs": result.injected_prestate_max_abs,
                    }
                )
                activation_rows.append(
                    basic_activation_row(
                        record["id"],
                        record["benchmark"],
                        state_name,
                        result.target_output,
                        full_target_output,
                        visual_mask,
                    )
                )
                sampled_activations.append(
                    {
                        "sample_id": record["id"],
                        "benchmark": record["benchmark"],
                        "state": state_name,
                        "tokens": deterministic_token_sample(result.target_output),
                    }
                )
                del repeat
                if state_name != "FULL":
                    del result
            del full

            read_reconstructed = run_cached_state(
                causal_lm,
                primary_context,
                visual_mask,
                "READ_RECONSTRUCTED",
                "reconstruct",
                "full",
                primary_read_cache,
            )
            read_rows.append(
                {
                    "sample_id": record["id"],
                    "benchmark": record["benchmark"],
                    "bucket": record["bucket"],
                    "layer": primary_layer,
                    "hook_identity_max_abs": read_reconstructed.read_hook_identity_max_abs,
                    "attention_decomposition_max_abs": read_reconstructed.read_decomposition.decomposition_max_abs,
                    "reference_actual_rms_ratio": read_reconstructed.read_decomposition.reference_actual_rms_ratio,
                    "quantization_adjustment_max_abs": read_reconstructed.read_decomposition.quantization_adjustment_max_abs,
                    "quantization_half_ulp_ratio_max": read_reconstructed.read_decomposition.quantization_half_ulp_ratio_max,
                    "quantization_adjustment_rms": read_reconstructed.read_decomposition.quantization_adjustment_rms,
                    "ideal_visual_delta_rms": read_reconstructed.read_decomposition.ideal_visual_delta_rms,
                    "quantization_adjustment_to_ideal_rms": read_reconstructed.read_decomposition.quantization_adjustment_to_ideal_rms,
                    "layer_output_max_abs": float(
                        (read_reconstructed.target_output.float() - primary_context.full_layer_output.float())
                        .abs()
                        .max()
                        .item()
                    ),
                    "final_logit_max_abs": max_abs_difference(
                        read_reconstructed.logits, baseline_logits
                    ),
                    "text_visual_attention_mass_mean": read_reconstructed.read_decomposition.text_visual_attention_mass_mean,
                    "visual_future_attention_mass_max": read_reconstructed.read_decomposition.visual_future_attention_mass_max,
                }
            )
            del read_reconstructed

            write_reconstructed = run_cached_state(
                causal_lm,
                primary_context,
                visual_mask,
                "WRITE_RECONSTRUCTED",
                "full",
                "reconstruct",
                primary_read_cache,
            )
            visual_rows = visual_mask.unsqueeze(-1).expand_as(primary_context.full_layer_output)
            write_rows.append(
                {
                    "sample_id": record["id"],
                    "benchmark": record["benchmark"],
                    "bucket": record["bucket"],
                    "layer": primary_layer,
                    "hook_identity_max_abs": write_reconstructed.write_hook_identity_max_abs,
                    "visual_layer_output_max_abs": float(
                        (
                            write_reconstructed.target_output[visual_rows].float()
                            - primary_context.full_layer_output[visual_rows].float()
                        )
                        .abs()
                        .max()
                        .item()
                    ),
                    "final_logit_max_abs": max_abs_difference(
                        write_reconstructed.logits, baseline_logits
                    ),
                }
            )
            del write_reconstructed

            reference_full = run_cached_state(
                causal_lm,
                primary_context,
                visual_mask,
                "REFERENCE_FULL",
                "reference",
                "full",
                primary_read_cache,
            )
            reference_rows.append(
                {
                    "sample_id": record["id"],
                    "benchmark": record["benchmark"],
                    "bucket": record["bucket"],
                    "layer": primary_layer,
                    "kind": "prompt",
                    "attention_output_max_abs": reference_full.read_decomposition.decomposition_max_abs,
                    "attention_output_rms_ratio": reference_full.read_decomposition.reference_actual_rms_ratio,
                    "layer_output_rms_ratio": rms_ratio(
                        reference_full.target_output, primary_context.full_layer_output
                    ),
                    "final_logit_rms_ratio": rms_ratio(reference_full.logits, baseline_logits),
                    "next_token_argmax_match": bool(
                        torch.equal(
                            reference_full.logits[:, -1].argmax(dim=-1),
                            baseline_logits[:, -1].argmax(dim=-1),
                        )
                    ),
                }
            )
            del reference_full

            if sample_index < int(config["sentinel_sample_count"]):
                for layer_index in sentinel_layers:
                    context = contexts[layer_index]
                    sentinel_read_cache = ReadInterventionCache()
                    sentinel_full = run_cached_state(
                        causal_lm,
                        context,
                        visual_mask,
                        "FULL",
                        "full",
                        "full",
                        sentinel_read_cache,
                    )
                    no_op_rows.append(
                        {
                            "sample_id": record["id"],
                            "benchmark": record["benchmark"],
                            "bucket": record["bucket"],
                            "layer": layer_index,
                            "kind": "sentinel_prompt",
                            "layer_state_max_abs": float(
                                (sentinel_full.target_output.float() - context.full_layer_output.float())
                                .abs()
                                .max()
                                .item()
                            ),
                            "final_logit_max_abs": max_abs_difference(
                                sentinel_full.logits, baseline_logits
                            ),
                            "prestate_injection_max_abs": sentinel_full.injected_prestate_max_abs,
                        }
                    )
                    del sentinel_full

                    sentinel_read = run_cached_state(
                        causal_lm,
                        context,
                        visual_mask,
                        "READ_RECONSTRUCTED",
                        "reconstruct",
                        "full",
                        sentinel_read_cache,
                    )
                    read_rows.append(
                        {
                            "sample_id": record["id"],
                            "benchmark": record["benchmark"],
                            "bucket": record["bucket"],
                            "layer": layer_index,
                            "hook_identity_max_abs": sentinel_read.read_hook_identity_max_abs,
                            "attention_decomposition_max_abs": sentinel_read.read_decomposition.decomposition_max_abs,
                            "reference_actual_rms_ratio": sentinel_read.read_decomposition.reference_actual_rms_ratio,
                            "quantization_adjustment_max_abs": sentinel_read.read_decomposition.quantization_adjustment_max_abs,
                            "quantization_half_ulp_ratio_max": sentinel_read.read_decomposition.quantization_half_ulp_ratio_max,
                            "quantization_adjustment_rms": sentinel_read.read_decomposition.quantization_adjustment_rms,
                            "ideal_visual_delta_rms": sentinel_read.read_decomposition.ideal_visual_delta_rms,
                            "quantization_adjustment_to_ideal_rms": sentinel_read.read_decomposition.quantization_adjustment_to_ideal_rms,
                            "layer_output_max_abs": float(
                                (sentinel_read.target_output.float() - context.full_layer_output.float())
                                .abs()
                                .max()
                                .item()
                            ),
                            "final_logit_max_abs": max_abs_difference(
                                sentinel_read.logits, baseline_logits
                            ),
                            "text_visual_attention_mass_mean": sentinel_read.read_decomposition.text_visual_attention_mass_mean,
                            "visual_future_attention_mass_max": sentinel_read.read_decomposition.visual_future_attention_mass_max,
                        }
                    )
                    del sentinel_read

                    sentinel_write = run_cached_state(
                        causal_lm,
                        context,
                        visual_mask,
                        "WRITE_RECONSTRUCTED",
                        "full",
                        "reconstruct",
                        sentinel_read_cache,
                    )
                    sentinel_visual_rows = visual_mask.unsqueeze(-1).expand_as(context.full_layer_output)
                    write_rows.append(
                        {
                            "sample_id": record["id"],
                            "benchmark": record["benchmark"],
                            "bucket": record["bucket"],
                            "layer": layer_index,
                            "hook_identity_max_abs": sentinel_write.write_hook_identity_max_abs,
                            "visual_layer_output_max_abs": float(
                                (
                                    sentinel_write.target_output[sentinel_visual_rows].float()
                                    - context.full_layer_output[sentinel_visual_rows].float()
                                )
                                .abs()
                                .max()
                                .item()
                            ),
                            "final_logit_max_abs": max_abs_difference(
                                sentinel_write.logits, baseline_logits
                            ),
                        }
                    )
                    del sentinel_write

                    sentinel_reference = run_cached_state(
                        causal_lm,
                        context,
                        visual_mask,
                        "REFERENCE_FULL",
                        "reference",
                        "full",
                        sentinel_read_cache,
                    )
                    reference_rows.append(
                        {
                            "sample_id": record["id"],
                            "benchmark": record["benchmark"],
                            "bucket": record["bucket"],
                            "layer": layer_index,
                            "kind": "sentinel_prompt",
                            "attention_output_max_abs": sentinel_reference.read_decomposition.decomposition_max_abs,
                            "attention_output_rms_ratio": sentinel_reference.read_decomposition.reference_actual_rms_ratio,
                            "layer_output_rms_ratio": rms_ratio(
                                sentinel_reference.target_output, context.full_layer_output
                            ),
                            "final_logit_rms_ratio": rms_ratio(
                                sentinel_reference.logits, baseline_logits
                            ),
                            "next_token_argmax_match": bool(
                                torch.equal(
                                    sentinel_reference.logits[:, -1].argmax(dim=-1),
                                    baseline_logits[:, -1].argmax(dim=-1),
                                )
                            ),
                        }
                    )
                    del sentinel_reference

            del baseline, baseline_logits, full_target_output
            torch.cuda.empty_cache()

            option_rows.extend(
                option_parity(
                    causal_lm,
                    processor,
                    record,
                    prompt_inputs,
                    contexts[0].pre_layer_state,
                    primary_layer,
                    visual_mask,
                )
            )

            generated = fresh_generate(causal_lm, prompt_inputs, int(config["max_new_tokens"]))
            instrumented_generated, hook_calls = no_op_instrumented_generate(
                causal_lm, prompt_inputs, primary_layer, int(config["max_new_tokens"])
            )
            generated_text = decode_new_tokens(processor, generated, input_ids.shape[1])
            instrumented_text = decode_new_tokens(processor, instrumented_generated, input_ids.shape[1])
            fresh_score = score_record(record, generated_text)
            scoring_rows.append(
                {
                    "sample_id": record["id"],
                    "benchmark": record["benchmark"],
                    "bucket": record["bucket"],
                    "stored_prediction": record["prediction"],
                    "fresh_prediction": generated_text,
                    "instrumented_full_prediction": instrumented_text,
                    "stored_score": float(record["score"]),
                    "stored_prediction_rescored": score_record(record, record["prediction"]),
                    "fresh_score": fresh_score,
                    "prediction_normalized_match": normalize_exact(generated_text)
                    == normalize_exact(record["prediction"]),
                    "score_match": abs(fresh_score - float(record["score"])) <= 1e-9,
                    "instrumented_token_ids_match": bool(torch.equal(generated, instrumented_generated)),
                    "instrumented_text_match": generated_text == instrumented_text,
                    "instrumented_hook_pre_calls": hook_calls["pre"],
                    "instrumented_hook_post_calls": hook_calls["post"],
                }
            )
            del generated, instrumented_generated, prompt_inputs, contexts
            torch.cuda.empty_cache()
            print(json.dumps({"completed": sample_index + 1, "total": len(samples), "sample_id": record["id"]}), flush=True)

    add_geometry_metrics(activation_rows, sampled_activations)
    write_json(output_dir / "token_layout.json", {"samples": token_layouts})
    write_csv(output_dir / "no_op_parity.csv", no_op_rows)
    write_csv(output_dir / "read_reconstruction.csv", read_rows)
    write_csv(output_dir / "read_split_recomposition_diagnostic.csv", reference_rows)
    write_csv(output_dir / "write_reconstruction.csv", write_rows)
    write_csv(output_dir / "four_state_stability.csv", stability_rows)
    write_csv(output_dir / "benchmark_scoring_reproduction.csv", scoring_rows)
    write_csv(output_dir / "option_score_parity.csv", option_rows)
    write_csv(output_dir / "activation_plausibility.csv", activation_rows)

    tolerances = config["tolerances"]
    maxima = {
        "no_op_layer_max_abs": max(row["layer_state_max_abs"] for row in no_op_rows),
        "no_op_logit_max_abs": max(row["final_logit_max_abs"] for row in no_op_rows),
        "read_hook_max_abs": max(row["hook_identity_max_abs"] for row in read_rows),
        "read_decomposition_max_abs": max(row["attention_decomposition_max_abs"] for row in read_rows),
        "read_reference_attention_rms_ratio": max(
            row["attention_output_rms_ratio"] for row in reference_rows
        ),
        "read_reference_layer_rms_ratio": max(row["layer_output_rms_ratio"] for row in reference_rows),
        "read_reference_logit_rms_ratio": max(row["final_logit_rms_ratio"] for row in reference_rows),
        "read_quantization_adjustment_max_abs": max(
            row["quantization_adjustment_max_abs"] for row in read_rows
        ),
        "read_quantization_half_ulp_ratio_max": max(
            row["quantization_half_ulp_ratio_max"] for row in read_rows
        ),
        "read_quantization_adjustment_to_ideal_rms_max": max(
            row["quantization_adjustment_to_ideal_rms"] for row in read_rows
        ),
        "read_logit_max_abs": max(row["final_logit_max_abs"] for row in read_rows),
        "write_hook_max_abs": max(row["hook_identity_max_abs"] for row in write_rows),
        "write_logit_max_abs": max(row["final_logit_max_abs"] for row in write_rows),
        "repeat_logit_max_abs": max(row["repeat_logit_max_abs"] for row in stability_rows),
        "option_logit_max_abs": max(row["logit_max_abs"] for row in option_rows),
        "option_score_abs": max(row["score_abs"] for row in option_rows),
        "reference_option_score_abs": max(row["reference_score_abs"] for row in option_rows),
        "prestate_injection_max_abs": max(row["prestate_injection_max_abs"] for row in stability_rows),
        "visual_future_attention_mass_max": max(row["visual_future_attention_mass_max"] for row in read_rows),
    }
    split_recomposition_diagnostic = maxima["read_reference_attention_rms_ratio"] \
        <= tolerances["read_reference_activation_rms_ratio"] \
        and maxima["read_reference_layer_rms_ratio"] \
        <= tolerances["read_reference_activation_rms_ratio"] \
        and maxima["read_reference_logit_rms_ratio"] \
        <= tolerances["read_reference_logit_rms_ratio"] \
        and maxima["reference_option_score_abs"] \
        <= tolerances["read_reference_option_score_abs"] \
        and all(row["next_token_argmax_match"] for row in reference_rows) \
        and all(row["reference_candidate_order_match"] for row in option_rows)
    checks = {
        "sample_count_20_to_50": 20 <= len(samples) <= 50,
        "architecture_and_token_layout": all(
            layout["visual_rows"]["contiguous"] for layout in token_layouts
        )
        and maxima["visual_future_attention_mass_max"] == 0.0,
        "no_op_parity": maxima["no_op_layer_max_abs"] <= tolerances["no_op_layer_max_abs"]
        and maxima["no_op_logit_max_abs"] <= tolerances["no_op_logit_max_abs"]
        and all(row["instrumented_token_ids_match"] for row in scoring_rows)
        and maxima["option_logit_max_abs"] <= tolerances["no_op_logit_max_abs"]
        and maxima["option_score_abs"] <= tolerances["scoring_abs"],
        "read_reconstruction": maxima["read_hook_max_abs"] <= tolerances["reconstruction_hook_max_abs"]
        and maxima["read_quantization_half_ulp_ratio_max"]
        <= tolerances["read_quantization_half_ulp_ratio_max"]
        and maxima["read_logit_max_abs"] <= tolerances["reconstruction_logit_max_abs"],
        "write_reconstruction": maxima["write_hook_max_abs"] <= tolerances["reconstruction_hook_max_abs"]
        and maxima["write_logit_max_abs"] <= tolerances["reconstruction_logit_max_abs"],
        "four_state_deterministic_stable": all(row["finite"] for row in stability_rows)
        and maxima["repeat_logit_max_abs"] <= tolerances["repeat_logit_max_abs"]
        and maxima["prestate_injection_max_abs"] == 0.0,
        "benchmark_evaluator_and_full_scoring_reproduced": all(
            abs(row["stored_prediction_rescored"] - row["stored_score"]) <= 1e-9 for row in scoring_rows
        )
        and all(row["instrumented_token_ids_match"] for row in scoring_rows)
        and all(row["instrumented_text_match"] for row in scoring_rows),
    }
    if args.probe_only:
        checks["sample_count_20_to_50"] = False
    gate_pass = all(checks.values())
    summary = {
        "stage": "A",
        "probe_only": args.probe_only,
        "sample_count": len(samples),
        "requested_sample_count": len(requested_samples),
        "resource_exclusions": exclusions,
        "validated_layers": capture_layers,
        "runtime": runtime,
        "tolerances": tolerances,
        "maxima": maxima,
        "checks": checks,
        "diagnostics": {
            "split_recomposition_within_frozen_backend_thresholds": split_recomposition_diagnostic,
            "pinned_checkpoint_bucket_scores_match_inherited_scores": all(
                row["score_match"] for row in scoring_rows
            ),
        },
        "gate_pass": gate_pass,
        "stage_b_entry_gate_satisfied": gate_pass and not args.probe_only,
        "stage_b_executed": False,
        "benchmark_score_match_count": sum(row["score_match"] for row in scoring_rows),
        "benchmark_evaluator_reproduction_count": sum(
            abs(row["stored_prediction_rescored"] - row["stored_score"]) <= 1e-9
            for row in scoring_rows
        ),
        "instrumented_full_prediction_match_count": sum(
            row["instrumented_token_ids_match"] for row in scoring_rows
        ),
        "benchmark_prediction_match_count": sum(row["prediction_normalized_match"] for row in scoring_rows),
    }
    write_json(output_dir / "stage_a_summary.json", summary)
    graph_summary = {
        "all_visual_rows_contiguous": all(layout["visual_rows"]["contiguous"] for layout in token_layouts),
        "visual_future_attention_mass_max": maxima["visual_future_attention_mass_max"],
        "validated_layers": capture_layers,
        "read_decomposition_max_abs": maxima["read_decomposition_max_abs"],
        "read_reference_attention_rms_ratio": maxima["read_reference_attention_rms_ratio"],
        "read_reference_logit_rms_ratio": maxima["read_reference_logit_rms_ratio"],
        "prestate_injection_max_abs": maxima["prestate_injection_max_abs"],
    }
    (output_dir / "architecture_causal_graph.md").write_text(
        architecture_markdown(runtime, graph_summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if gate_pass or args.probe_only else 2


def main() -> int:
    args = parse_args()
    try:
        return execute(args)
    except Exception as exc:
        output_dir = Path(args.output_dir or ("outputs/stage_a_probe" if args.probe_only else "outputs/stage_a"))
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "stage_a_failure.json",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
