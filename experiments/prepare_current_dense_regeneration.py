#!/usr/bin/env python3
"""Freeze the candidate population, smoke set, and fail-closed dense contract."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from dense_failure_stage1.contract import (
    collect_contract_integrity,
    feature_schema,
    file_sha256,
)
from dense_failure_stage1.runtime import configure_dense_determinism
from tools.research_analysis.dense_failure_stage1 import (
    build_candidate_manifest,
    canonical_payload_sha256,
    select_dense_smoke,
    validate_candidate_execution_contract,
)


def write_once(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"refusing to overwrite differing frozen artifact: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json_once(path: Path, value) -> None:
    write_once(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def write_jsonl_once(path: Path, rows: list[dict]) -> None:
    write_once(
        path,
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
    )


def verify_images(rows: list[dict]) -> dict:
    format_counts = Counter()
    errors = []
    for row in rows:
        path = Path(row["local_image_path"])
        if not os.access(path, os.R_OK):
            errors.append(
                {"uid": row["uid"], "reason": "not_readable", "path": str(path)}
            )
            continue
        try:
            with Image.open(path) as image:
                format_counts[str(image.format)] += 1
                image.verify()
        except Exception as exc:
            errors.append(
                {"uid": row["uid"], "reason": repr(exc), "path": str(path)}
            )
    if errors:
        raise ValueError(f"candidate image verification failed: {errors[:5]}")
    return {
        "verified_readable_images": len(rows),
        "image_format_counts": dict(format_counts),
    }


def validate_overlap(candidates: list[dict], overlap_path: Path) -> dict:
    known = {
        row["uid"]: row
        for row in (
            json.loads(line)
            for line in overlap_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row.get("benchmark") in {"gqa", "chartqa", "textvqa"}
    }
    candidate_index = {row["uid"]: row for row in candidates}
    missing = sorted(set(known) - set(candidate_index))
    if missing:
        raise ValueError(
            f"transferred overlap UIDs missing from candidates: {missing[:5]}"
        )
    comparisons = {
        "sample_id": "sample_id",
        "benchmark": "benchmark",
        "question": "question",
        "prompt": "prompt",
        "answer": "answer",
        "all_answer_norms": "all_answer_norms",
        "metric_name": "metric_name",
        "correctness_threshold": "correctness_threshold",
        "max_new_tokens": "max_new_tokens",
        "image_identifier": "image_group_id",
    }
    mismatch_counts = {}
    examples = {}
    for candidate_field, known_field in comparisons.items():
        bad = [
            uid
            for uid, row in known.items()
            if candidate_index[uid].get(candidate_field) != row.get(known_field)
        ]
        mismatch_counts[f"{candidate_field}_vs_{known_field}"] = len(bad)
        if bad:
            examples[f"{candidate_field}_vs_{known_field}"] = bad[:5]
    if any(mismatch_counts.values()):
        raise ValueError(
            f"portable/transferred overlap differs: {mismatch_counts}: {examples}"
        )
    return {
        "transferred_overlap_records": len(known),
        "fields_compared": comparisons,
        "mismatch_counts": mismatch_counts,
        "missing_from_candidate": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portable-manifest", required=True)
    parser.add_argument("--overlap-manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).absolute()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    portable_path = Path(args.portable_manifest).resolve()
    portable_rows = [
        json.loads(line)
        for line in portable_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates, audit = build_candidate_manifest(
        portable_rows,
        image_root=Path(args.image_root),
        expected_counts=config["datasets"],
    )
    validate_candidate_execution_contract(
        candidates,
        evaluator_specs=config["evaluators"],
        max_new_tokens=int(config["generation"]["max_new_tokens"]),
    )
    audit.update(verify_images(candidates))
    audit["portable_manifest"] = str(portable_path)
    audit["portable_manifest_sha256"] = file_sha256(portable_path)
    audit["overlap_validation"] = validate_overlap(
        candidates, Path(args.overlap_manifest)
    )
    if len(candidates) != 8000 or audit["unique_uids"] != 8000:
        raise ValueError("the recovered candidate population is not exactly 8,000")

    output_root = Path(args.output_root)
    candidate_path = output_root / "candidate_8k_manifest.jsonl"
    write_jsonl_once(candidate_path, candidates)
    audit["candidate_manifest_sha256"] = file_sha256(candidate_path)
    write_json_once(output_root / "candidate_8k_manifest_audit.json", audit)

    smoke = select_dense_smoke(
        candidates,
        seed=int(config["smoke_seed"]),
        per_cell=int(config["smoke_records_per_dataset_bucket"]),
    )
    smoke_value = {
        "schema_version": "current_dense_smoke_manifest_v1",
        "selection_seed": int(config["smoke_seed"]),
        "selection_policy": (
            "4 records per dataset x historical-bucket cell; distinct content groups"
        ),
        "records": smoke,
    }
    smoke_path = output_root / "smoke" / "smoke_manifest.json"
    write_json_once(smoke_path, smoke_value)

    configure_dense_determinism(
        int(config["execution_seed"]), dict(config["backend_settings"])
    )
    integrity = collect_contract_integrity(
        project=project,
        model_path=Path(args.model_path),
        revision=config["model_revision"],
        processor_revision=config["processor_revision"],
        candidate_manifest=candidate_path,
        smoke_manifest=smoke_path,
        config_path=config_path,
        config=config,
    )
    contract = {
        "schema_version": "current_dense_execution_contract_v2",
        "purpose": "authoritative current-runtime dense all-on Stage-1 failure labels",
        "contract_separation": {
            "executor": "native Qwen2.5-VL dense generation",
            "decoder_layer_semantics": (
                "FULL = READ ON, WRITE ON for every one of 28 decoder layers"
            ),
            "four_action_intervention": False,
            "sparse_route_information_used": False,
            "ground_truth_exposed_to_feature_construction": False,
        },
        "model": {
            "name": config["model_name"],
            "revision": config["model_revision"],
            "weight_path": integrity["model"]["supplied_weight_path"],
            "resolved_weight_path": integrity["model"]["resolved_weight_path"],
            "dtype": config["dtype"],
            "attention_implementation": config["attention_implementation"],
            "file_sha256": integrity["model"]["file_sha256"],
        },
        "tokenizer_processor": {
            **integrity["processor_tokenizer"],
            "prompt_template": (
                "single user message with image then exact manifest prompt; "
                "add_generation_prompt=True"
            ),
            "image_preprocessing": config["image_processing"],
            "custom_max_image_tokens": None,
        },
        "generation": dict(config["generation"]),
        "evaluators": {
            "specifications": dict(config["evaluators"]),
            "implementation": "reference/dvr_qwen/eval_metrics.py",
            "implementation_sha256": integrity["bound_source_sha256"][
                "reference/dvr_qwen/eval_metrics.py"
            ],
            "fallback_allowed": False,
        },
        "features": {
            "enabled_only_after_exact_smoke_token_parity": True,
            "schema": feature_schema(config),
            "schema_sha256": integrity["feature_schema_sha256"],
        },
        "data": {
            "candidate_manifest": integrity["candidate_manifest_path"],
            "candidate_manifest_sha256": integrity[
                "candidate_manifest_sha256"
            ],
            "records": len(candidates),
            "dataset_counts": config["datasets"],
            "historical_bucket_role": (
                "selection metadata only; never the current target"
            ),
            "image_group_definition": "SHA-256 of physical image file content",
        },
        "runtime": {
            "environment": integrity["environment"],
            "backend_settings": integrity["backend_settings"],
            "scheduler": "direct execution; Slurm unavailable on this server",
        },
        "git": integrity["git"],
        "integrity": integrity,
    }
    contract["contract_sha256"] = canonical_payload_sha256(contract)
    contract_path = output_root / "frozen_dense_execution_contract.json"
    write_json_once(contract_path, contract)
    print(
        json.dumps(
            {
                "records": len(candidates),
                "unique_image_groups": audit["unique_image_groups"],
                "smoke": len(smoke),
                "contract_sha256": contract["contract_sha256"],
                "git_commit": integrity["git"]["commit"],
                "worktree_entries": len(integrity["git"]["status_porcelain"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
