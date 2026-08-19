from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from experiments.stage_a_validity import max_abs_difference, prepare_prompt, set_determinism
from experiments.stage_b_reference_likelihood import capture_prompt_with_cache, read_jsonl
from interventions.prompt_cache import run_cached_prompt_state
from interventions.read_path import ReadInterventionCache, ReadPathController
from nulls.structured_read import (
    DonorMetadata,
    fit_fixed_grid_covariance,
    fit_real_donor_caliper,
    generate_covariance_null,
    generate_real_residual_null,
    map_rows,
    real_donor_candidates,
    select_real_donors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and smoke-test frozen Stage C nulls.")
    parser.add_argument("--config", default="configs/stage_c_entry.yaml")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derived_seed(base: int, sample_id: str, draw: int) -> int:
    digest = hashlib.sha256(f"{base}:{sample_id}:{draw}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def image_id(record: dict[str, Any]) -> str:
    value = (
        record.get("image_id")
        or record.get("source_asset_id")
        or record.get("selection_asset_key")
        or record.get("local_image_path")
    )
    if not value:
        raise ValueError(f"No effective-image identifier for {record.get('id', '<unknown>')}")
    return str(value)


class _LayerZeroCaptured(RuntimeError):
    pass


def extract_postvisual_residual_early(
    causal_lm, inputs: dict[str, Any], visual_mask: torch.Tensor, layer_index: int
) -> tuple[torch.Tensor, ReadInterventionCache]:
    """Capture the exact layer READ delta and stop before the unchanged suffix."""
    layer = causal_lm.model.layers[layer_index]
    cache = ReadInterventionCache()
    def stop_after_layer(module, args, kwargs, output):
        raise _LayerZeroCaptured()

    stop_handle = layer.register_forward_hook(stop_after_layer, with_kwargs=True)
    causal_lm.rope_deltas = None
    try:
        with ReadPathController(layer.self_attn, visual_mask, "off", cache):
            try:
                causal_lm(**inputs, use_cache=False, return_dict=True)
            except _LayerZeroCaptured:
                pass
            else:
                raise RuntimeError("Calibration early-stop hook did not execute")
    finally:
        stop_handle.remove()
    if cache.actual_output is None or cache.off_output is None:
        raise RuntimeError("READ calibration cache is incomplete")
    indices = torch.where(visual_mask[0])[0]
    start = int(indices[-1].item()) + 1
    residual = (cache.actual_output.float() - cache.off_output.float())[0, start:]
    if residual.shape[0] < 1 or float(residual.norm().item()) <= 0.0:
        raise RuntimeError("Post-visual READ residual is empty or degenerate")
    return residual.detach().cpu(), cache


def load_model(config: dict[str, Any]):
    model_config = yaml.safe_load(Path(config["model_config"]).read_text(encoding="utf-8"))
    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(
        model_config["snapshot_path"], local_files_only=True, use_fast=False
    )
    hf_config = AutoConfig.from_pretrained(model_config["snapshot_path"], local_files_only=True)
    hf_config._attn_implementation = {"vision_config": model_config["vision_attention_backend"]}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_config["snapshot_path"],
        config=hf_config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    for layer in model.model.layers:
        layer.self_attn.stage_a_query_chunk_size = int(
            model_config["decoder_attention_query_chunk_size"]
        )
    return model, processor, device, model_config


def rank_from_raw_gram(raw_gram: torch.Tensor, variance_target: float) -> int:
    row_mean = raw_gram.mean(dim=1, keepdim=True)
    centered_gram = raw_gram - row_mean - row_mean.transpose(0, 1) + raw_gram.mean()
    gram = centered_gram / (raw_gram.shape[0] - 1)
    values = torch.linalg.eigvalsh(gram).flip(0)
    values = values[values > max(float(values[0].item()) * 1e-10, 1e-12)]
    cumulative = torch.cumsum(values, dim=0) / values.sum()
    return int(torch.searchsorted(cumulative, torch.tensor(variance_target)).item()) + 1


def subspace_relative_error(sample: torch.Tensor, fit) -> float:
    grid = map_rows(sample.float(), fit.grid_rows).reshape(-1)
    mean = fit.mean.float().reshape(-1)
    basis = fit.basis.float().reshape(fit.rank, -1)
    augmented = torch.cat([mean.unsqueeze(0), basis], dim=0)
    q, _ = torch.linalg.qr(augmented.transpose(0, 1), mode="reduced")
    projection = q @ (q.transpose(0, 1) @ grid)
    return float((grid - projection).norm().item() / max(float(grid.norm().item()), 1e-12))


def execute(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    null_cfg = config["nulls"]
    output_dir = Path("outputs/stage_c/nulls")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_determinism(int(config["selection_seed"]))
    model, processor, device, model_config = load_model(config)

    calibration_rows = [
        row
        for row in read_jsonl(Path(null_cfg["calibration_manifest"]))
        if row.get("benchmark") == null_cfg["calibration_dataset"]
    ]
    if len(calibration_rows) != 200:
        raise RuntimeError(f"Expected 200 Stage B TextVQA calibration rows, got {len(calibration_rows)}")

    residuals: list[torch.Tensor] = []
    metadata: list[DonorMetadata] = []
    donor_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, record in enumerate(calibration_rows):
            _, inputs = prepare_prompt(processor, record, device)
            visual_mask = inputs["input_ids"] == model.config.image_token_id
            residual, _ = extract_postvisual_residual_early(
                model, inputs, visual_mask, int(config["primary_layer"])
            )
            info = DonorMetadata(
                sample_id=str(record["id"]),
                image_id=image_id(record),
                residual_norm=float(residual.float().norm().item()),
                postvisual_rows=int(residual.shape[0]),
                visual_tokens=int(visual_mask.sum().item()),
                prompt_tokens=int(inputs["input_ids"].shape[1]),
            )
            residuals.append(residual.to(torch.float16))
            metadata.append(info)
            donor_rows.append(
                {
                    **info.__dict__,
                    "layer": int(config["primary_layer"]),
                    "hook": "decoder.layer.0.self_attn.output.postvisual_nonvisual_rows",
                    "task": "textvqa",
                    "residual_tensor_index": index,
                    "downstream_likelihood_used_for_selection": False,
                }
            )
            if (index + 1) % 10 == 0:
                print(f"calibration {index + 1}/{len(calibration_rows)}", flush=True)

    residual_artifact = output_dir / "stage_b_read_residuals_v1.pt"
    torch.save(
        {
            "schema_version": "stage_c_stage_b_read_residuals_v1",
            "layer": int(config["primary_layer"]),
            "hook": "decoder.layer.0.self_attn.output.postvisual_nonvisual_rows",
            "sample_ids": [item.sample_id for item in metadata],
            "residuals": residuals,
        },
        residual_artifact,
    )
    donor_index_path = output_dir / "real_residual_donor_index_v1.jsonl"
    write_jsonl(donor_index_path, donor_rows)

    fit = fit_fixed_grid_covariance(
        residuals,
        grid_rows=int(null_cfg["grid_rows"]),
        variance_target=float(null_cfg["variance_target"]),
        eigen_shrinkage=float(null_cfg["eigen_shrinkage"]),
    )
    covariance_path = output_dir / "covariance_subspace_parameters_v1.pt"
    torch.save(
        {
            "schema_version": "stage_c_covariance_subspace_v1",
            "mean": fit.mean,
            "basis": fit.basis,
            "eigenvalues": fit.eigenvalues,
            "rank": fit.rank,
            "explained_variance": fit.explained_variance,
            "grid_rows": fit.grid_rows,
            "hidden_size": fit.hidden_size,
            "calibration_samples": fit.calibration_samples,
            "variance_target": fit.variance_target,
            "eigen_shrinkage": fit.eigen_shrinkage,
            "covariance_estimator": "unbiased sample covariance over equal-weight fixed-grid sample residuals",
        },
        covariance_path,
    )

    grids = torch.stack([map_rows(item.float(), fit.grid_rows) for item in residuals])
    flat = grids.reshape(len(grids), -1)
    raw_gram = flat @ flat.transpose(0, 1)
    loo_ranks = []
    for index in range(len(residuals)):
        keep = torch.tensor(
            [item.image_id != metadata[index].image_id for item in metadata]
        )
        loo_ranks.append(
            rank_from_raw_gram(raw_gram[keep][:, keep], fit.variance_target)
        )
    matching_ratio_cap = fit_real_donor_caliper(
        metadata, draws=int(null_cfg["draws_per_family"])
    )
    donor_eligible_counts: list[int] = []
    donor_failures: list[str] = []
    for target in metadata:
        eligible_donors = real_donor_candidates(
            target,
            metadata,
            seed=int(null_cfg["real_donor_seed"]),
            matching_ratio_cap=matching_ratio_cap,
        )
        donor_eligible_counts.append(len(eligible_donors))
        try:
            selected = select_real_donors(
                target,
                metadata,
                draws=int(null_cfg["draws_per_family"]),
                seed=int(null_cfg["real_donor_seed"]),
                matching_ratio_cap=matching_ratio_cap,
            )
        except ValueError:
            donor_failures.append(target.sample_id)

    manifest = read_jsonl(Path(config["manifest_path"]))
    seed_rows = []
    for record in manifest:
        seed_rows.append(
            {
                "id": record["id"],
                "covariance_draw_seeds": [
                    derived_seed(int(null_cfg["covariance_seed"]), record["id"], draw)
                    for draw in range(int(null_cfg["draws_per_family"]))
                ],
                "real_donor_tie_break_seed": derived_seed(
                    int(null_cfg["real_donor_seed"]), record["id"], 0
                ),
            }
        )
    seeds_path = output_dir / "deterministic_null_seeds_v1.jsonl"
    write_jsonl(seeds_path, seed_rows)

    smoke_rows: list[dict[str, Any]] = []
    parity_tolerance = 1e-4
    reconstruction_tolerance = 1e-4
    norm_tolerance = 1e-5
    subspace_tolerance = 0.05
    with torch.inference_mode():
        for record in manifest[:2]:
            _, inputs = prepare_prompt(processor, record, device)
            _, repeated_inputs = prepare_prompt(processor, record, device)
            processor_repeat_exact = all(
                torch.equal(value, repeated_inputs[key])
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            )
            baseline, contexts = capture_prompt_with_cache(
                model, inputs, [int(config["primary_layer"])]
            )
            visual_mask = inputs["input_ids"] == model.config.image_token_id
            context = contexts[int(config["primary_layer"])]
            full = run_cached_prompt_state(
                model,
                context,
                baseline.past_key_values,
                visual_mask,
                "FULL",
                "full",
                "full",
            )
            read_cache = ReadInterventionCache()
            write_only = run_cached_prompt_state(
                model,
                context,
                baseline.past_key_values,
                visual_mask,
                "WRITE_ONLY",
                "off",
                "full",
                read_cache,
            )
            reconstructed = run_cached_prompt_state(
                model,
                context,
                baseline.past_key_values,
                visual_mask,
                "READ_RECONSTRUCTED",
                "reconstruct",
                "full",
                read_cache,
            )
            if read_cache.actual_output is None or read_cache.off_output is None:
                raise RuntimeError("Smoke READ cache is incomplete")
            actual_delta = read_cache.actual_output.float() - read_cache.off_output.float()
            visual_indices = torch.where(visual_mask[0])[0]
            post_start = int(visual_indices[-1].item()) + 1
            target_residual = actual_delta[0, post_start:].detach().cpu()
            target_norm = float(target_residual.norm().item())
            covariance_seed = derived_seed(
                int(null_cfg["covariance_seed"]), record["id"], 0
            )
            covariance_rows = generate_covariance_null(
                fit, target_residual.shape[0], target_norm, covariance_seed
            )
            covariance_native = generate_covariance_null(
                fit, fit.grid_rows, target_norm, covariance_seed
            )
            covariance_repeat = generate_covariance_null(
                fit, target_residual.shape[0], target_norm, covariance_seed
            )
            target_meta = DonorMetadata(
                sample_id=record["id"],
                image_id=record["image_id"],
                residual_norm=target_norm,
                postvisual_rows=int(target_residual.shape[0]),
                visual_tokens=int(visual_mask.sum().item()),
                prompt_tokens=int(inputs["input_ids"].shape[1]),
            )
            chosen = select_real_donors(
                target_meta,
                metadata,
                draws=int(null_cfg["draws_per_family"]),
                seed=derived_seed(int(null_cfg["real_donor_seed"]), record["id"], 0),
                matching_ratio_cap=matching_ratio_cap,
            )
            donor_position = {item.sample_id: index for index, item in enumerate(metadata)}
            real_rows = generate_real_residual_null(
                residuals[donor_position[chosen[0].sample_id]],
                target_residual.shape[0],
                target_norm,
            )
            covariance_delta = torch.zeros_like(actual_delta)
            covariance_delta[0, post_start:] = covariance_rows.to(device)
            real_delta = torch.zeros_like(actual_delta)
            real_delta[0, post_start:] = real_rows.to(device)
            covariance_state = run_cached_prompt_state(
                model,
                context,
                baseline.past_key_values,
                visual_mask,
                "COVARIANCE_NULL",
                "replace",
                "full",
                read_cache,
                read_replacement_delta=covariance_delta,
            )
            real_state = run_cached_prompt_state(
                model,
                context,
                baseline.past_key_values,
                visual_mask,
                "REAL_RESIDUAL_NULL",
                "replace",
                "full",
                read_cache,
                read_replacement_delta=real_delta,
            )
            smoke_rows.append(
                {
                    "id": record["id"],
                    "full_parity_logit_max_abs": max_abs_difference(full.prompt_logits, baseline.logits),
                    "read_reconstruction_logit_max_abs": max_abs_difference(
                        reconstructed.prompt_logits, baseline.logits
                    ),
                    "write_only_executed": bool(torch.isfinite(write_only.prompt_logits).all().item()),
                    "covariance_null_shape": list(covariance_rows.shape),
                    "real_null_shape": list(real_rows.shape),
                    "target_residual_shape": list(target_residual.shape),
                    "covariance_norm_relative_error": abs(float(covariance_rows.norm().item()) - target_norm) / target_norm,
                    "real_norm_relative_error": abs(float(real_rows.norm().item()) - target_norm) / target_norm,
                    "covariance_native_subspace_relative_error": subspace_relative_error(covariance_native, fit),
                    "covariance_mapped_subspace_relative_error_descriptor": subspace_relative_error(covariance_rows, fit),
                    "covariance_seed_repeat_exact": bool(torch.equal(covariance_rows, covariance_repeat)),
                    "covariance_null_suffix_finite": bool(torch.isfinite(covariance_state.prompt_logits).all().item()),
                    "real_null_suffix_finite": bool(torch.isfinite(real_state.prompt_logits).all().item()),
                    "same_sample_donor_count": sum(item.sample_id == record["id"] for item in chosen),
                    "same_image_donor_count": sum(item.image_id == record["image_id"] for item in chosen),
                    "donor_ids": [item.sample_id for item in chosen],
                    "accepted_answer_span_count": len(record["accepted_answer_tokenization"]),
                    "prompt_token_length_matches_manifest": int(inputs["input_ids"].shape[1]) == int(record["prompt_token_length"]),
                    "image_token_count_matches_manifest": int(visual_mask.sum().item()) == int(record["image_token_count"]),
                    "pinned_processor_repeat_exact": processor_repeat_exact,
                    "primary_endpoint_computed": False,
                }
            )

    smoke_pass = all(
        row["full_parity_logit_max_abs"] <= parity_tolerance
        and row["read_reconstruction_logit_max_abs"] <= reconstruction_tolerance
        and row["write_only_executed"]
        and row["covariance_norm_relative_error"] <= norm_tolerance
        and row["real_norm_relative_error"] <= norm_tolerance
        and row["covariance_native_subspace_relative_error"] <= subspace_tolerance
        and row["covariance_seed_repeat_exact"]
        and row["covariance_null_suffix_finite"]
        and row["real_null_suffix_finite"]
        and row["same_sample_donor_count"] == 0
        and row["same_image_donor_count"] == 0
        and row["accepted_answer_span_count"] > 0
        and row["prompt_token_length_matches_manifest"]
        and row["image_token_count_matches_manifest"]
        and row["pinned_processor_repeat_exact"]
        and not row["primary_endpoint_computed"]
        for row in smoke_rows
    )
    calibration_audit = {
        "schema_version": "stage_c_null_calibration_audit_v1",
        "outcome_blind": True,
        "stage_c_primary_endpoint_computed": False,
        "calibration_sample_count": len(residuals),
        "calibration_unique_images": len({item.image_id for item in metadata}),
        "fixed_grid_rows": fit.grid_rows,
        "hidden_size": fit.hidden_size,
        "retained_rank": fit.rank,
        "explained_variance": fit.explained_variance,
        "leave_one_image_out_rank_min": min(loo_ranks),
        "leave_one_image_out_rank_max": max(loo_ranks),
        "leave_one_image_out_rank_median": sorted(loo_ranks)[len(loo_ranks) // 2],
        "real_donor_caliper_rule": null_cfg["real_donor_caliper_rule"],
        "real_donor_matching_ratio": "max(norm ratio, postvisual-row ratio, visual-token ratio)",
        "real_donor_matching_ratio_cap": matching_ratio_cap,
        "real_donor_draws": int(null_cfg["draws_per_family"]),
        "leave_one_image_out_donor_failure_count": len(donor_failures),
        "leave_one_image_out_donor_failures": donor_failures,
        "eligible_donor_count_min": min(donor_eligible_counts),
        "eligible_donor_count_median": sorted(donor_eligible_counts)[len(donor_eligible_counts) // 2],
        "eligible_donor_count_max": max(donor_eligible_counts),
        "smoke_tolerances": {
            "full_parity_logit_max_abs": parity_tolerance,
            "read_reconstruction_logit_max_abs": reconstruction_tolerance,
            "norm_relative_error": norm_tolerance,
            "subspace_relative_error": subspace_tolerance,
        },
        "smoke_rows": smoke_rows,
        "smoke_pass": smoke_pass,
        "gate_pass": smoke_pass and not donor_failures,
        "runtime": {
            "model_id": model_config["model_id"],
            "revision": model_config["revision"],
            "snapshot_path": model_config["snapshot_path"],
            "dtype": model_config["dtype"],
            "decoder_attention_backend": model_config["decoder_attention_backend"],
            "vision_attention_backend": model_config["vision_attention_backend"],
            "torch_version": torch.__version__,
        },
    }
    audit_path = output_dir / "null_calibration_and_smoke_v1.json"
    write_json(audit_path, calibration_audit)

    checksum_paths = [
        residual_artifact,
        donor_index_path,
        covariance_path,
        seeds_path,
        audit_path,
    ]
    checksum_file = output_dir / "null_artifacts_v1.sha256"
    checksum_file.write_text(
        "".join(f"{sha256(path)}  {path}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in calibration_audit.items() if key != "smoke_rows"}, indent=2))
    if not calibration_audit["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    execute(Path(parse_args().config))
