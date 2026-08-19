from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention

from audit.sample_manifest import select_stage_a_samples
from experiments.stage_a_validity import (
    capture_forward,
    load_yaml,
    max_abs_difference,
    prepare_prompt,
    rms_ratio,
    set_determinism,
    write_json,
)
from interventions.chunked_eager_attention import install_chunked_eager_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare stock and query-chunked Qwen eager attention on one short fixed sample."
    )
    parser.add_argument("--config", default="configs/stage_a.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-id", required=True)
    return parser.parse_args()


def execute(args: argparse.Namespace) -> int:
    config = load_yaml(Path(args.config))
    model_config = load_yaml(Path(config["model_config"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_determinism(int(config["seed"]))

    # This diagnostic validates the runtime substitution only; it does not
    # change the Stage A sample manifest. The caller must choose a fixed sample
    # whose sequence crosses at least one query-chunk boundary while remaining
    # small enough for stock eager attention.
    samples = select_stage_a_samples(
        Path(config["dataset_root"]),
        list(config["benchmarks"]),
        list(config["buckets"]),
        int(config["samples_per_benchmark_bucket"]),
    )
    matches = [sample for sample in samples if sample["id"] == args.sample_id]
    if len(matches) != 1:
        raise ValueError(f"Diagnostic sample was not selected exactly once: {args.sample_id}")
    sample = matches[0]
    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(
        model_config["snapshot_path"], local_files_only=True
    )
    hf_config = AutoConfig.from_pretrained(
        model_config["snapshot_path"], local_files_only=True
    )
    hf_config._attn_implementation = {
        "vision_config": model_config["vision_attention_backend"],
    }
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_config["snapshot_path"],
        config=hf_config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    _, inputs = prepare_prompt(processor, sample, device)
    sequence_length = int(inputs["input_ids"].shape[1])
    chunk_size = int(model_config["decoder_attention_query_chunk_size"])
    if sequence_length <= chunk_size:
        raise ValueError(
            f"Diagnostic sequence {sequence_length} does not cross chunk size {chunk_size}"
        )
    layer_index = int(config["primary_layer"])

    with torch.inference_mode():
        stock, stock_contexts = capture_forward(model, inputs, [layer_index])
        stock_logits = stock.logits.detach().cpu()
        stock_layer = stock_contexts[layer_index].full_layer_output.detach().cpu()
        del stock, stock_contexts

        install_chunked_eager_attention(
            model.model, chunk_size
        )
        chunked, chunked_contexts = capture_forward(model, inputs, [layer_index])
        chunked_logits = chunked.logits.detach().cpu()
        chunked_layer = chunked_contexts[layer_index].full_layer_output.detach().cpu()

    tolerances = config["tolerances"]
    metrics = {
        "sample_id": sample["id"],
        "sequence_length": sequence_length,
        "stock_attention_forward": f"{Qwen2_5_VLAttention.__module__}.{Qwen2_5_VLAttention.__name__}.forward",
        "chunk_size": chunk_size,
        "layer_output_max_abs": max_abs_difference(chunked_layer, stock_layer),
        "layer_output_rms_ratio": rms_ratio(chunked_layer, stock_layer),
        "final_logit_max_abs": max_abs_difference(chunked_logits, stock_logits),
        "final_logit_rms_ratio": rms_ratio(chunked_logits, stock_logits),
        "next_token_argmax_match": bool(
            torch.equal(
                chunked_logits[:, -1].argmax(dim=-1),
                stock_logits[:, -1].argmax(dim=-1),
            )
        ),
    }
    metrics["passes_frozen_runtime_thresholds"] = bool(
        metrics["layer_output_rms_ratio"]
        <= tolerances["read_reference_activation_rms_ratio"]
        and metrics["final_logit_rms_ratio"]
        <= tolerances["read_reference_logit_rms_ratio"]
        and metrics["next_token_argmax_match"]
    )
    write_json(output_dir / "chunked_eager_equivalence.json", metrics)
    print(metrics, flush=True)
    return 0 if metrics["passes_frozen_runtime_thresholds"] else 2


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
