from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from experiments.stage_a_validity import max_abs_difference, prepare_prompt, set_determinism
from experiments.stage_b_reference_likelihood import capture_prompt_with_cache, read_jsonl
from experiments.v3_confirmation_preflight import load_model
from interventions.read_path import ReadInterventionCache, ReadPathController
from nulls.structured_read import map_rows


LAYERS = [0, 4, 8, 12, 16, 20, 24]
BASE_SEED = 2026080605
RECONSTRUCTION_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract answer-free Stage B READ/WRITE geometry.")
    parser.add_argument(
        "--manifest", default="data_manifests/v3_null_calibration_geometry_400.jsonl"
    )
    parser.add_argument(
        "--output-root", default="artifacts/v3_null_calibration/read_write_geometry_v1"
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_summary(value: torch.Tensor) -> dict[str, Any]:
    norms = value.float().norm(dim=1)
    mean = float(norms.mean().item())
    std = float(norms.std(unbiased=False).item())
    quantiles = torch.quantile(
        norms,
        torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=norms.device),
    )
    return {
        "frobenius_norm": float(value.float().norm().item()),
        "row_norm_mean": mean,
        "row_norm_std": std,
        "row_norm_cv": std / max(mean, 1e-12),
        "row_norm_quantiles": [float(item) for item in quantiles.tolist()],
        "row_rms_mean": float(value.float().square().mean(dim=1).sqrt().mean().item()),
    }


def rms_scale_ratio(residual: torch.Tensor, reference: torch.Tensor) -> float:
    residual_rms = float(residual.float().square().mean().sqrt().item())
    reference_rms = float(reference.float().square().mean().sqrt().item())
    return residual_rms / max(reference_rms, 1e-12)


def energy_correlation(read: torch.Tensor, write: torch.Tensor) -> float:
    read_energy = map_rows(read.float().norm(dim=1, keepdim=True), 32).squeeze(1)
    write_energy = map_rows(write.float().norm(dim=1, keepdim=True), 32).squeeze(1)
    read_centered = read_energy - read_energy.mean()
    write_centered = write_energy - write_energy.mean()
    denominator = float(read_centered.norm().item() * write_centered.norm().item())
    if denominator <= 1e-12:
        return 0.0
    return float(torch.dot(read_centered, write_centered).item() / denominator)


def extract_layer(
    model,
    context,
    visual_mask: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    layer = model.model.layers[context.layer_index]
    kwargs = dict(context.layer_kwargs)
    kwargs["past_key_value"] = None
    kwargs["use_cache"] = False
    kwargs["output_attentions"] = False
    injected = context.pre_layer_state.detach().clone()
    cache = ReadInterventionCache()
    with ReadPathController(layer.self_attn, visual_mask, "off", cache):
        layer(injected, **kwargs)
    with ReadPathController(layer.self_attn, visual_mask, "reconstruct", cache):
        reconstructed_layer = layer(injected, **kwargs)[0]
    if cache.actual_output is None or cache.off_output is None:
        raise RuntimeError("READ cache is incomplete")
    visual_indices = torch.where(visual_mask[0])[0]
    if visual_indices.numel() < 1:
        raise RuntimeError("No visual tokens")
    post_start = int(visual_indices[-1].item()) + 1
    read = (cache.actual_output.float() - cache.off_output.float())[0, post_start:]
    write = (context.full_layer_output.float() - context.pre_layer_state.float())[
        0, visual_indices
    ]
    if read.numel() == 0 or write.numel() == 0:
        raise RuntimeError("READ or WRITE residual is empty")
    reconstructed_write = (
        context.pre_layer_state[0, visual_indices].float() + write.float()
    ).to(context.full_layer_output.dtype)
    read_reconstruction = max_abs_difference(reconstructed_layer, context.full_layer_output)
    write_reconstruction = max_abs_difference(
        reconstructed_write, context.full_layer_output[0, visual_indices]
    )
    normalized = layer.input_layernorm(context.pre_layer_state).detach()
    read_reference = normalized[0, post_start:]
    write_reference = normalized[0, visual_indices]
    read_summary = row_summary(read)
    write_summary = row_summary(write)
    metrics = {
        "layer": int(context.layer_index),
        "read_shape": list(read.shape),
        "write_shape": list(write.shape),
        "read": read_summary,
        "write": write_summary,
        "read_rmsnorm_scale_ratio": rms_scale_ratio(read, read_reference),
        "write_rmsnorm_scale_ratio": rms_scale_ratio(write, write_reference),
        "mapped_row_energy_correlation": energy_correlation(read, write),
        "read_hook_reconstruction_max_abs": float(cache.hook_identity_max_abs or 0.0),
        "read_layer_reconstruction_max_abs": read_reconstruction,
        "write_hook_reconstruction_max_abs": write_reconstruction,
    }
    if max(
        metrics["read_hook_reconstruction_max_abs"],
        metrics["read_layer_reconstruction_max_abs"],
        metrics["write_hook_reconstruction_max_abs"],
    ) > RECONSTRUCTION_TOLERANCE:
        raise RuntimeError(f"Reconstruction failed at layer {context.layer_index}: {metrics}")
    return {
        "read": read.detach().to(device="cpu", dtype=torch.float16),
        "write": write.detach().to(device="cpu", dtype=torch.float16),
        "read_row_norms": read.float().norm(dim=1).detach().cpu(),
        "write_row_norms": write.float().norm(dim=1).detach().cpu(),
    }, metrics


def execute(args: argparse.Namespace) -> None:
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index/count")
    set_determinism(BASE_SEED + args.shard_index)
    rows = read_jsonl(Path(args.manifest))
    selected = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    output_dir = Path(args.output_root) / "shards" / f"shard_{args.shard_index:02d}"
    tensor_dir = output_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = output_dir / "geometry.jsonl"
    if geometry_path.exists():
        raise FileExistsError(f"Refusing to overwrite {geometry_path}")
    model, processor, model_config = load_model()
    device = torch.device("cuda")
    geometry_rows = []
    tensor_paths = []
    with torch.inference_mode():
        for index, record in enumerate(selected):
            _, inputs = prepare_prompt(processor, record, device)
            visual_mask = inputs["input_ids"] == model.config.image_token_id
            baseline, contexts = capture_prompt_with_cache(model, inputs, LAYERS)
            layers = {}
            for layer in LAYERS:
                tensors, metrics = extract_layer(model, contexts[layer], visual_mask)
                layers[layer] = tensors
                geometry_rows.append(
                    {
                        "sample_id": record["id"],
                        "image_id": record["image_id"],
                        "dataset": record["dataset"],
                        "prompt_tokens": int(inputs["input_ids"].shape[1]),
                        "image_tokens": int(visual_mask.sum().item()),
                        **metrics,
                    }
                )
            safe_id = record["id"].replace(":", "__").replace("/", "_")
            tensor_path = tensor_dir / f"{safe_id}.pt"
            torch.save(
                {
                    "schema_version": "v3_read_write_geometry_tensor_v1",
                    "sample_id": record["id"],
                    "image_id": record["image_id"],
                    "dataset": record["dataset"],
                    "layers": layers,
                },
                tensor_path,
            )
            tensor_paths.append(tensor_path)
            del inputs, baseline, contexts, layers
            torch.cuda.empty_cache()
            print(f"shard {args.shard_index}: {index + 1}/{len(selected)}", flush=True)
    with geometry_path.open("w", encoding="utf-8") as handle:
        for row in geometry_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest_path = output_dir / "shard_manifest.json"
    payload = {
        "schema_version": "v3_read_write_geometry_shard_v1",
        "outcome_blind": True,
        "answer_or_action_outcomes_loaded_or_used": False,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "sample_count": len(selected),
        "geometry_row_count": len(geometry_rows),
        "layers": LAYERS,
        "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
        "manifest_sha256": sha256(Path(args.manifest)),
        "geometry_sha256": sha256(geometry_path),
        "tensor_checksums": {str(path): sha256(path) for path in tensor_paths},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "model": model_config,
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "samples": len(selected)}))


if __name__ == "__main__":
    execute(parse_args())
