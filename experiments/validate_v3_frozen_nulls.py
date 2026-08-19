from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from tools.research_analysis.v3.freeze_v3_null_models import fit_from_payload
from experiments.stage_a_validity import max_abs_difference, prepare_prompt, set_determinism
from experiments.stage_b_reference_likelihood import capture_prompt_with_cache, read_jsonl
from experiments.v3_confirmation_preflight import embed_delta, load_model
from interventions.prompt_cache import run_cached_prompt_state
from interventions.read_path import ReadInterventionCache, ReadPathController
from nulls.joint_four_action import (
    generate_joint_path_null,
    generate_paired_isotropic_null,
    generate_paired_real_null,
    search_budget_cells,
)


LAYERS = [0, 4, 8, 12, 16, 20, 24]
ACTIONS = ["IGNORE", "READ_ONLY", "WRITE_ONLY"]
BASE_SEED = 2026080613


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibration-only frozen v3 null smoke.")
    parser.add_argument(
        "--manifest", default="data_manifests/v3_null_calibration_geometry_400.jsonl"
    )
    parser.add_argument(
        "--geometry-root", default="artifacts/v3_null_calibration/read_write_geometry_v1"
    )
    parser.add_argument(
        "--covariance-root", default="artifacts/v3_null_calibration/joint_covariance_model_v1"
    )
    parser.add_argument(
        "--donor-index",
        default="artifacts/v3_null_calibration/paired_donor_index_v1/paired_donor_index.jsonl",
    )
    parser.add_argument(
        "--output", default="outputs/v3_preflight/frozen_null_technical_validation_v1.json"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed(*parts: Any) -> int:
    value = ":".join([str(BASE_SEED), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % (2**63 - 1)


def populate_read_cache(model, context, visual_mask: torch.Tensor) -> ReadInterventionCache:
    layer = model.model.layers[context.layer_index]
    kwargs = dict(context.layer_kwargs)
    kwargs.update(past_key_value=None, use_cache=False, output_attentions=False)
    cache = ReadInterventionCache()
    with ReadPathController(layer.self_attn, visual_mask, "off", cache):
        layer(context.pre_layer_state.detach().clone(), **kwargs)
    if cache.actual_output is None or cache.off_output is None:
        raise RuntimeError("READ cache population failed")
    return cache


def main() -> None:
    args = parse_args()
    set_determinism(BASE_SEED)
    manifest_path = Path(args.manifest)
    records = read_jsonl(manifest_path)
    selected = [
        min((row for row in records if row["dataset"] == dataset), key=lambda row: row["id"])
        for dataset in ("gqa", "textvqa")
    ]
    geometry_root = Path(args.geometry_root)
    tensor_index = json.loads((geometry_root / "tensor_index.json").read_text())
    donor_rows = {
        (row["sample_id"], int(row["layer"])): row
        for row in (
            json.loads(line) for line in Path(args.donor_index).read_text().splitlines()
        )
    }
    needed_ids = {row["id"] for row in selected}
    for record in selected:
        for layer_index in LAYERS:
            needed_ids.add(
                donor_rows[(record["id"], layer_index)]["donors"][0]["sample_id"]
            )
    index_by_name = {Path(row["path"]).name: row for row in tensor_index}
    tensors = {}
    for sample_id in sorted(needed_ids):
        safe_name = sample_id.replace(":", "__").replace("/", "_") + ".pt"
        row = index_by_name[safe_name]
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"Tensor checksum mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["sample_id"] != sample_id:
            raise RuntimeError(f"Tensor identity mismatch: {path}")
        tensors[sample_id] = payload
    model, processor, model_config = load_model()
    device = torch.device("cuda")
    validation_rows = []
    with torch.inference_mode():
        for record in selected:
            _, inputs = prepare_prompt(processor, record, device)
            visual_mask = inputs["input_ids"] == model.config.image_token_id
            baseline, contexts = capture_prompt_with_cache(model, inputs, LAYERS)
            target_payload = tensors[record["id"]]
            for layer_index in LAYERS:
                context = contexts[layer_index]
                cache = populate_read_cache(model, context, visual_mask)
                full = run_cached_prompt_state(
                    model,
                    context,
                    baseline.past_key_values,
                    visual_mask,
                    "FULL",
                    "full",
                    "full",
                    cache,
                )
                target = target_payload["layers"][layer_index]
                read_norms = target["read_row_norms"].to(device)
                write_norms = target["write_row_norms"].to(device)
                fit_payload = torch.load(
                    Path(args.covariance_root)
                    / f"{record['dataset']}_layer_{layer_index:02d}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                fit = fit_from_payload(fit_payload["fit"], device)
                joint_seed = seed("joint", record["id"], layer_index)
                isotropic_seed = seed("isotropic", record["id"], layer_index)
                covariance_pair = generate_joint_path_null(
                    fit, read_norms, write_norms, joint_seed
                )
                covariance_repeat = generate_joint_path_null(
                    fit, read_norms, write_norms, joint_seed
                )
                isotropic_pair = generate_paired_isotropic_null(
                    int(target["read"].shape[1]),
                    read_norms,
                    write_norms,
                    isotropic_seed,
                )
                donor_row = donor_rows[(record["id"], layer_index)]
                donor_id = donor_row["donors"][0]["sample_id"]
                donor = tensors[donor_id]["layers"][layer_index]
                donor_pair = generate_paired_real_null(
                    donor["read"].float().to(device),
                    donor["write"].float().to(device),
                    read_norms,
                    write_norms,
                )
                families = {
                    "isotropic": isotropic_pair,
                    "joint_covariance": covariance_pair,
                    "paired_real_donor": donor_pair,
                }
                family_rows = []
                for family, pair in families.items():
                    read_delta = embed_delta(
                        pair[0], cache.actual_output, visual_mask, "read"
                    ).to(device)
                    write_delta = embed_delta(
                        pair[1], context.full_layer_output, visual_mask, "write"
                    ).to(device)
                    action_finite = {}
                    for action in ACTIONS:
                        remove_read = action in {"IGNORE", "WRITE_ONLY"}
                        remove_write = action in {"IGNORE", "READ_ONLY"}
                        result = run_cached_prompt_state(
                            model,
                            context,
                            baseline.past_key_values,
                            visual_mask,
                            f"{family}:{action}",
                            "subtract" if remove_read else "full",
                            "subtract" if remove_write else "full",
                            cache,
                            read_replacement_delta=read_delta if remove_read else None,
                            write_replacement_delta=write_delta if remove_write else None,
                        )
                        action_finite[action] = bool(
                            torch.isfinite(result.prompt_logits).all().item()
                        )
                    family_rows.append(
                        {
                            "family": family,
                            "actions": action_finite,
                            "read_shape": list(pair[0].shape),
                            "write_shape": list(pair[1].shape),
                            "read_row_norm_max_abs": float(
                                (pair[0].norm(dim=1) - read_norms).abs().max().item()
                            ),
                            "write_row_norm_max_abs": float(
                                (pair[1].norm(dim=1) - write_norms).abs().max().item()
                            ),
                        }
                    )
                validation_rows.append(
                    {
                        "dataset": record["dataset"],
                        "sample_id": record["id"],
                        "layer": layer_index,
                        "full_parity_logit_max_abs": max_abs_difference(
                            full.prompt_logits, baseline.logits
                        ),
                        "joint_repeat_exact": bool(
                            torch.equal(covariance_pair[0], covariance_repeat[0])
                            and torch.equal(covariance_pair[1], covariance_repeat[1])
                        ),
                        "donor_id": donor_id,
                        "donor_image_id": tensors[donor_id]["image_id"],
                        "donor_is_different_sample": donor_id != record["id"],
                        "donor_is_different_image": (
                            tensors[donor_id]["image_id"] != target_payload["image_id"]
                        ),
                        "families": family_rows,
                    }
                )
                del fit, fit_payload
                torch.cuda.empty_cache()
    gate = all(
        row["full_parity_logit_max_abs"] == 0.0
        and row["joint_repeat_exact"]
        and row["donor_is_different_sample"]
        and row["donor_is_different_image"]
        and all(
            family["read_row_norm_max_abs"] <= 1e-4
            and family["write_row_norm_max_abs"] <= 1e-4
            and all(family["actions"].values())
            for family in row["families"]
        )
        for row in validation_rows
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": "v3_frozen_null_technical_validation_v1",
                "outcome_blind": True,
                "answer_scoring_or_terminal_action_values_computed": False,
                "calibration_records_only": True,
                "model": model_config,
                "manifest_sha256": sha256(manifest_path),
                "geometry_manifest_sha256": sha256(geometry_root / "manifest.json"),
                "donor_index_sha256": sha256(Path(args.donor_index)),
                "search_budget": search_budget_cells(LAYERS, ACTIONS),
                "null_orientation": "subtract the paired null residual from FULL at the validated path; never replace it with an alternative residual",
                "rows": validation_rows,
                "gate_pass": gate,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "sha256": sha256(output), "gate_pass": gate}))
    if not gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
