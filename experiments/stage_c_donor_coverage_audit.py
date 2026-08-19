from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import yaml

from tools.research_analysis.v2.stage_c_donor_coverage import coverage_summary, target_coverage
from experiments.stage_a_validity import prepare_prompt, set_determinism
from experiments.stage_b_reference_likelihood import read_jsonl
from experiments.stage_c_entry_gate import extract_postvisual_residual_early, load_model
from nulls.structured_read import DonorMetadata


HOOK_NAME = "decoder.layer.0.self_attn.output.postvisual_nonvisual_rows"
QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
REPORT_CALIPERS = (1.5, 1.75, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage C donor geometry only.")
    parser.add_argument("command", choices=("run", "merge"))
    parser.add_argument("--config", default="configs/stage_c.yaml")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def frozen_geometry_checks(config: dict[str, Any]) -> dict[str, str]:
    pairs = {
        "source_plan": (config["source_plan"], config["source_plan_sha256"]),
        "structured_null_spec": (
            config["structured_null_spec"],
            config["structured_null_spec_sha256"],
        ),
        "model_config": (config["model_config"], config["model_config_sha256"]),
        "manifest": (config["manifest_path"], config["manifest_sha256"]),
        "donor_index": (
            config["nulls"]["donor_index"],
            config["nulls"]["donor_index_sha256"],
        ),
        "deterministic_seeds": (
            config["nulls"]["deterministic_seeds"],
            config["nulls"]["deterministic_seeds_sha256"],
        ),
    }
    observed = {}
    for name, (raw_path, expected) in pairs.items():
        digest = sha256(Path(raw_path))
        if digest != expected:
            raise RuntimeError(f"Frozen geometry input {name} changed: {digest} != {expected}")
        observed[name] = digest
    return observed


def load_donor_metadata(config: dict[str, Any]) -> list[DonorMetadata]:
    rows = read_jsonl(Path(config["nulls"]["donor_index"]))
    if len(rows) != 200:
        raise RuntimeError("Frozen real-residual donor pool must contain exactly 200 rows")
    for row in rows:
        if (
            row["task"] != "textvqa"
            or int(row["layer"]) != int(config["primary_layer"])
            or row["hook"] != HOOK_NAME
        ):
            raise RuntimeError("Donor task, layer, or hook differs from the frozen specification")
    donors = [
        DonorMetadata(
            sample_id=str(row["sample_id"]),
            image_id=str(row["image_id"]),
            residual_norm=float(row["residual_norm"]),
            postvisual_rows=int(row["postvisual_rows"]),
            visual_tokens=int(row["visual_tokens"]),
            prompt_tokens=int(row["prompt_tokens"]),
        )
        for row in rows
    ]
    if len({donor.sample_id for donor in donors}) != 200 or len(
        {donor.image_id for donor in donors}
    ) != 200:
        raise RuntimeError("Frozen donor sample and image IDs must be unique")
    if any(
        donor.residual_norm <= 0.0
        or not math.isfinite(donor.residual_norm)
        or donor.postvisual_rows < 1
        or donor.visual_tokens < 1
        for donor in donors
    ):
        raise RuntimeError("Frozen donor geometry contains a degenerate value")
    return donors


def run_shard(
    config_path: Path,
    shard_index: int,
    num_shards: int,
    resume: bool,
) -> int:
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("Shard index must lie in [0, num_shards)")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    observed = frozen_geometry_checks(config)
    manifest = read_jsonl(Path(config["manifest_path"]))
    samples = [row for index, row in enumerate(manifest) if index % num_shards == shard_index]
    donors = load_donor_metadata(config)
    seed_rows = {
        row["id"]: row for row in read_jsonl(Path(config["nulls"]["deterministic_seeds"]))
    }
    if set(seed_rows) != {row["id"] for row in manifest}:
        raise RuntimeError("Frozen donor tie seeds do not cover the exact manifest")
    set_determinism(int(config["bootstrap"]["primary_seed"]))
    model, processor, device, model_config = load_model(config)
    output_dir = Path("outputs/stage_c/preflight/donor_coverage_v1/shards") / f"shard_{shard_index:02d}"
    result_path = output_dir / "geometry.jsonl"
    runtime_path = output_dir / "runtime.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = set()
    if result_path.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite geometry audit shard: {result_path}")
        completed = {row["id"] for row in read_jsonl(result_path)}
    write_json(
        runtime_path,
        {
            "schema_version": "stage_c_donor_coverage_runtime_v1",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "model_id": model_config["model_id"],
            "revision": model_config["revision"],
            "dtype": str(next(model.parameters()).dtype),
            "layer": int(config["primary_layer"]),
            "hook": HOOK_NAME,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "frozen_input_sha256": observed,
            "likelihood_or_behavior_loaded": False,
            "partial_stage_c_result_path_accessed": False,
        },
    )
    original_caliper = float(config["nulls"]["real_donor_matching_ratio_cap"])
    with torch.inference_mode():
        for local_index, record in enumerate(samples):
            if record["id"] in completed:
                continue
            prompt_text, inputs = prepare_prompt(processor, record, device)
            if prompt_text != record["prompt_text"]:
                raise RuntimeError(f"Frozen prompt changed for {record['id']}")
            visual_mask = inputs["input_ids"] == model.config.image_token_id
            if int(inputs["input_ids"].shape[1]) != int(record["prompt_token_length"]):
                raise RuntimeError(f"Prompt length changed for {record['id']}")
            if int(visual_mask.sum().item()) != int(record["image_token_count"]):
                raise RuntimeError(f"Image-token count changed for {record['id']}")
            residual, _ = extract_postvisual_residual_early(
                model,
                inputs,
                visual_mask,
                int(config["primary_layer"]),
            )
            target = DonorMetadata(
                sample_id=str(record["id"]),
                image_id=str(record["image_id"]),
                residual_norm=float(residual.float().norm().item()),
                postvisual_rows=int(residual.shape[0]),
                visual_tokens=int(visual_mask.sum().item()),
                prompt_tokens=int(inputs["input_ids"].shape[1]),
            )
            coverage = target_coverage(
                target,
                donors,
                seed=int(seed_rows[record["id"]]["real_donor_tie_break_seed"]),
                original_caliper=original_caliper,
            )
            coverage.update(
                {
                    "schema_version": "stage_c_donor_coverage_target_v1",
                    "manifest_record_sha256": record["record_sha256"],
                    "task": "textvqa",
                    "layer": int(config["primary_layer"]),
                    "hook": HOOK_NAME,
                    "matching_covariates": [
                        "residual_norm",
                        "postvisual_rows",
                        "image_tokens",
                    ],
                    "matching_distance": "maximum multiplicative ratio",
                    "same_sample_excluded": True,
                    "same_image_excluded": True,
                    "donor_count_required": 8,
                }
            )
            append_jsonl(result_path, coverage)
            del inputs, residual
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "completed": local_index + 1,
                        "shard_total": len(samples),
                        "sample_id": record["id"],
                    }
                ),
                flush=True,
            )
    return 0


def merge(config_path: Path, num_shards: int) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    observed = frozen_geometry_checks(config)
    manifest = read_jsonl(Path(config["manifest_path"]))
    shard_root = Path("outputs/stage_c/preflight/donor_coverage_v1/shards")
    rows_by_id: dict[str, dict[str, Any]] = {}
    runtime_checksums = []
    for shard_index in range(num_shards):
        geometry_path = shard_root / f"shard_{shard_index:02d}" / "geometry.jsonl"
        runtime_path = shard_root / f"shard_{shard_index:02d}" / "runtime.json"
        rows = read_jsonl(geometry_path)
        expected_ids = {
            row["id"]
            for index, row in enumerate(manifest)
            if index % num_shards == shard_index
        }
        if {row["id"] for row in rows} != expected_ids or len(rows) != len(expected_ids):
            raise RuntimeError(f"Geometry shard {shard_index} is incomplete or duplicated")
        for row in rows:
            if row["likelihood_or_behavior_loaded"]:
                raise RuntimeError("A geometry row claims likelihood or behavior access")
            if row["id"] in rows_by_id:
                raise RuntimeError(f"Duplicate target geometry row: {row['id']}")
            rows_by_id[row["id"]] = row
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime["likelihood_or_behavior_loaded"] or runtime[
            "partial_stage_c_result_path_accessed"
        ]:
            raise RuntimeError("A geometry runtime claims prohibited outcome access")
        runtime_checksums.append({"path": str(runtime_path), "sha256": sha256(runtime_path)})
    ordered = [rows_by_id[row["id"]] for row in manifest]
    if len(ordered) != 800 or any(
        row["manifest_record_sha256"] != manifest[index]["record_sha256"]
        for index, row in enumerate(ordered)
    ):
        raise RuntimeError("Merged geometry does not align with the frozen 800-record manifest")
    summary = coverage_summary(ordered, REPORT_CALIPERS, QUANTILES)
    c_star_targets = [
        row for row in ordered if row["id"] in set(summary["c_star_target_ids"])
    ]
    output = {
        "schema_version": "stage_c_donor_coverage_audit_v1",
        "audit_scope": "geometry only",
        "outcome_blind": True,
        "likelihood_answer_correctness_intervention_effect_loaded": False,
        "partial_stage_c_result_files_loaded": False,
        "manifest_count": len(ordered),
        "unique_image_count": len({row["image_id"] for row in ordered}),
        "task": "textvqa",
        "layer": int(config["primary_layer"]),
        "hook": HOOK_NAME,
        "donor_pool_count": 200,
        "donor_count_required": int(config["nulls"]["draws_per_family"]),
        "original_caliper": float(config["nulls"]["real_donor_matching_ratio_cap"]),
        "matching_distance": "maximum multiplicative ratio",
        "matching_covariates": ["residual_norm", "postvisual_rows", "image_tokens"],
        "exclusions": ["same sample", "same image"],
        **summary,
        "c_star_determining_targets": c_star_targets,
        "targets": ordered,
        "single_candidate_amendment": {
            "replace_caliper": float(config["nulls"]["real_donor_matching_ratio_cap"]),
            "with_exact_c_star": summary["c_star"],
            "preserve_donor_count": 8,
            "preserve_all_other_matching_rules": True,
            "amendment_applied": False,
        },
        "frozen_input_sha256": observed,
        "runtime_checksums": runtime_checksums,
        "stage_c_sweep_resumed": False,
        "stage_d_authorized": False,
    }
    output_path = Path("outputs/stage_c/preflight/stage_c_donor_coverage_audit_v1.json")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit result: {output_path}")
    write_json(output_path, output)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "output_sha256": sha256(output_path),
                "target_count": len(ordered),
                "c_star": summary["c_star"],
                "c_star_target_ids": summary["c_star_target_ids"],
                "likelihood_or_behavior_loaded": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.command == "merge":
            return merge(Path(args.config), args.num_shards)
        if args.shard_index is None:
            raise ValueError("run requires --shard-index")
        return run_shard(
            Path(args.config),
            args.shard_index,
            args.num_shards,
            args.resume,
        )
    except Exception as exc:
        failure_dir = Path("outputs/stage_c/preflight/donor_coverage_v1/failures")
        failure_dir.mkdir(parents=True, exist_ok=True)
        suffix = args.command if args.shard_index is None else f"shard_{args.shard_index:02d}"
        write_json(
            failure_dir / f"{suffix}_failure.json",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "likelihood_or_behavior_loaded": False,
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
