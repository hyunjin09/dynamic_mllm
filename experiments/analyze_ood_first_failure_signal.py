#!/usr/bin/env python3
"""Run the frozen leave-one-dataset-out dense-failure diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_EXTERNAL_ROOT = Path("/mnt/hyemin").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dense_failure_stage1.contract import model_file_hashes  # noqa: E402
from dense_failure_stage1.layerwise_probe import (  # noqa: E402
    FEATURE_NAMES,
    binary_metrics,
    compose_layer_features,
    fit_linear_probe,
    preservation_operating_point,
    score_linear_probe,
    validate_checkpoint_provenance,
)
from dense_failure_stage1.ood_signal import (  # noqa: E402
    INPUT_FEATURE_NAMES,
    OOD_RUN_TARGETS,
    PreDecoderInputCollector,
    audit_cross_dataset_groups,
    choose_representative_layer,
    validate_ood_membership,
)
from dense_failure_stage1.runtime import (  # noqa: E402
    build_dense_inputs,
    configure_dense_determinism,
    load_dense_runtime,
    token_positions,
)
from experiments.analyze_layerwise_dense_failure_predictability import (  # noqa: E402
    checkpoint_expected as phase48_checkpoint_expected,
    load_frozen_contract as load_phase48_contract,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/ood_first_failure_signal_diagnostic_v1.json"
DATASETS = ("gqa", "chartqa", "textvqa")
BOUND_CODE_PATHS = (
    "configs/ood_first_failure_signal_diagnostic_v1.json",
    "dense_failure_stage1/ood_signal.py",
    "dense_failure_stage1/runtime.py",
    "dense_failure_stage1/layerwise_probe.py",
    "experiments/analyze_ood_first_failure_signal.py",
)
CONTRACT_FILE = "frozen_protocol.json"
INPUT_SCHEMA_FILE = "input_features/feature_schema.json"
POPULATION_FILE = "population_manifest.jsonl"
SMOKE_FILE = "smoke/smoke_manifest.jsonl"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
    )


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    _atomic_bytes(path, payload.encode())


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode())


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_once_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite incompatible frozen artifact: {path}")
        return
    _atomic_bytes(path, payload)


def command_output(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result.stdout.strip()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def resolve_path(value: str) -> Path:
    unresolved = PROJECT_ROOT / value
    resolved = unresolved.resolve()
    if not (resolved.is_relative_to(PROJECT_ROOT) or resolved.is_relative_to(ALLOWED_EXTERNAL_ROOT)):
        raise ValueError(f"path escapes ACCESS_POLICY roots: {value} -> {resolved}")
    return resolved


def relative_project_path(path: Path) -> str:
    return str(path.absolute().relative_to(PROJECT_ROOT))


def load_static_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "ood_first_failure_signal_diagnostic_config_v1":
        raise ValueError("unsupported OOD diagnostic config")
    if int(config["population"]["records"]) != 7999:
        raise ValueError("OOD diagnostic is frozen to the current 7,999-row population")
    if set(config["runs"]) != set(OOD_RUN_TARGETS):
        raise ValueError("OOD config must declare exactly three leave-one-dataset-out runs")
    for run, target in OOD_RUN_TARGETS.items():
        if config["runs"][run]["target_dataset"] != target:
            raise ValueError(f"OOD target differs for {run}")
    if tuple(config["input_control"]["features"]) != INPUT_FEATURE_NAMES:
        raise ValueError("input-control feature order differs")
    if int(config["input_control"]["input_size"]) != 3 * int(
        config["input_control"]["hidden_size_per_summary"]
    ):
        raise ValueError("input-control feature dimension differs")
    if int(config["world_size"]) != 4:
        raise ValueError("the frozen diagnostic requires four direct workers")
    return config


def _population(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = {str(row["uid"]): row for row in read_jsonl(resolve_path(config["sources"]["candidate_manifest"]))}
    dense = {str(row["uid"]): row for row in read_jsonl(resolve_path(config["sources"]["dense_outputs"]))}
    split = {str(row["uid"]): row for row in read_jsonl(resolve_path(config["sources"]["phase48_split"]))}
    hidden = {str(row["uid"]): row for row in read_jsonl(resolve_path(config["sources"]["hidden_feature_index"]))}
    if len(candidates) != 8000 or len(dense) != 7999:
        raise ValueError("candidate/dense population counts differ from current contract")
    if set(dense) != set(split) or set(dense) != set(hidden) or not set(dense) <= set(candidates):
        raise ValueError("dense, Phase-48 split, and hidden-feature UIDs differ")
    rows = []
    for uid in sorted(dense):
        candidate = candidates[uid]
        outcome = dense[uid]
        split_row = split[uid]
        hidden_row = hidden[uid]
        if not bool(candidate.get("image_present_at_preparation")):
            raise ValueError(f"completed UID lacks a prepared image: {uid}")
        if str(candidate["dataset"]) != str(outcome["dataset"]) or str(candidate["dataset"]) != str(split_row["dataset"]):
            raise ValueError(f"dataset identity differs for {uid}")
        if bool(outcome["current_dense_correct"]) == bool(outcome["current_dense_wrong"]):
            raise ValueError(f"current label is not complementary for {uid}")
        if str(candidate["image_content_sha256"]) != str(outcome["image_content_sha256"]):
            raise ValueError(f"image hash differs between current artifacts for {uid}")
        rows.append(
            {
                "schema_version": "ood_signal_population_v1",
                "uid": uid,
                "dataset": str(outcome["dataset"]),
                "split": str(split_row["split"]),
                "image_group_id": str(outcome["image_group_id"]),
                "current_dense_correct": bool(outcome["current_dense_correct"]),
                "current_dense_wrong": bool(outcome["current_dense_wrong"]),
                "local_image_path": str(candidate["local_image_path"]),
                "image_content_sha256": str(candidate["image_content_sha256"]),
                "prompt": str(candidate["prompt"]),
                "expected_prompt_token_count": int(outcome["prompt_token_count"]),
                "expected_visual_token_count": int(outcome["visual_token_count"]),
                "expected_user_text_token_count": int(outcome["user_text_token_count"]),
                "hidden_feature_shard": str(hidden_row["shard"]),
                "hidden_feature_row": int(hidden_row["row_index"]),
            }
        )
    expected_counts = {key: int(value) for key, value in config["population"]["datasets"].items()}
    if Counter(row["dataset"] for row in rows) != Counter(expected_counts):
        raise ValueError("dataset counts differ from frozen OOD config")
    if sum(row["current_dense_wrong"] for row in rows) != int(config["population"]["wrong"]):
        raise ValueError("wrong-label count differs from frozen OOD config")
    return rows


def _input_schema(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "ood_native_pre_language_decoder_features_v1",
        "representation_name": config["input_control"]["name"],
        "capture_boundary": config["input_control"]["capture_boundary"],
        "features": {
            "text_final": "native merged pre-decoder state at final literal user token",
            "text_mean": "mean native merged pre-decoder state over user text tokens",
            "visual_mean": "mean native merged pre-decoder state over expanded image tokens",
        },
        "composition": "concatenate text_final, text_mean, visual_mean",
        "hidden_size_per_summary": 3584,
        "input_size": 10752,
        "storage_dtype": "torch.bfloat16",
        "contains_learned_visual_encoder_output": True,
        "contains_language_decoder_processing": False,
        "prohibited_inputs": [
            "dataset ID",
            "ground-truth answer",
            "current correctness label",
            "historical correctness",
            "W-to-C membership",
            "route information",
        ],
    }


def _select_smoke(rows: Sequence[Mapping[str, Any]], records: int) -> list[dict[str, Any]]:
    if records != 12:
        raise ValueError("frozen smoke requires 12 rows")
    selected = []
    used_groups: set[str] = set()
    for dataset in DATASETS:
        for wrong in (False, True):
            candidates = [
                row
                for row in rows
                if row["dataset"] == dataset and bool(row["current_dense_wrong"]) == wrong
            ]
            for row in candidates:
                if row["image_group_id"] in used_groups:
                    continue
                selected.append(dict(row))
                used_groups.add(str(row["image_group_id"]))
                if sum(
                    item["dataset"] == dataset
                    and bool(item["current_dense_wrong"]) == wrong
                    for item in selected
                ) == 2:
                    break
    if len(selected) != records:
        raise RuntimeError("could not construct the frozen stratified capture smoke")
    return selected


def _protocol_markdown(contract: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    lines = [
        "# OOD Dense-Failure Signal Protocol",
        "",
        f"- Frozen contract: `{contract['contract_sha256']}`",
        f"- Current population: `{contract['population']['records']:,}` rows; labels are current native-dense LMMS correctness only.",
        "- Source train and validation reuse Phase-48 split membership restricted to the two source datasets.",
        "- OOD target is every row of the held-out dataset and is not used for training, checkpointing, threshold calibration, probability calibration, or layer selection.",
        "- Twenty-eight independent linear probes use the exact Phase-48 form and optimization settings.",
        "- Representative layer is the maximum source-validation AUROC with a lower-layer tie break, frozen before OOD scoring.",
        "- 99%, 98%, and 95% preservation thresholds are calibrated on source validation and transferred unchanged.",
        "- The in-domain/OOD comparison uses identical Phase-48 target-test UIDs. Each model uses its own permitted validation-calibrated 99% threshold.",
        f"- Fixed interpretation heuristic: meaningful transfer requires source-selected OOD AUROC >= `{contract['interpretation']['meaningful_transfer_auroc_floor_all_targets']:.2f}` on all targets; decoder-added evidence requires at least `{contract['interpretation']['hidden_gain_required_targets']}` targets with source-selected hidden-minus-input AUROC >= `{contract['interpretation']['hidden_over_input_minimum_delta']:.2f}` and no target below AUROC `{contract['interpretation']['minimum_target_auroc_for_shared_predictor_support']:.2f}`.",
        "",
        "## Native pre-language-decoder control",
        "",
        "The control captures the input to language-decoder layer 0 after native token embedding, the learned visual encoder, and multimodal insertion. It pools the same user-final, user-mean, and visual-mean token positions as the hidden-state probe and concatenates them. A private sentinel stops execution inside the layer-0 pre-hook; the smoke requires one capture and zero decoder forward-hook firings.",
        "",
        "This is not a raw-pixel/raw-text baseline. It tests information added beyond Qwen's native pre-language-decoder representation.",
        "",
        "## Leakage audit",
        "",
        f"- Cross-dataset image groups: `{audit['cross_dataset_image_groups']}`",
        f"- Cross-dataset records: `{audit['cross_dataset_records']}`",
        f"- Passed: `{audit['passed']}`",
        "",
    ]
    for run, values in contract["membership_audit"].items():
        lines.extend(
            [
                f"### {run}",
                "",
                f"- Source train: `{values['source_train']:,}`",
                f"- Source validation: `{values['source_validation']:,}`",
                f"- Full OOD target: `{values['target_all']:,}`",
                f"- Identical-UID in-domain comparison subset: `{values['target_phase48_test']:,}`",
                "",
            ]
        )
    return "\n".join(lines)


def prepare(config_path: Path) -> None:
    config = load_static_config(config_path)
    output_root = resolve_path(config["output_root"])
    for directory in (
        "smoke/workers",
        "input_features/shards",
        "input_features/workers",
        "checkpoints",
        "training_history",
        "source_workers",
        "ood_workers",
        "figures",
        *config["runs"].keys(),
    ):
        (output_root / directory).mkdir(parents=True, exist_ok=True)

    phase48, _ = load_phase48_contract(resolve_path("configs/layerwise_dense_failure_probe_v1.json"))
    if phase48["contract_sha256"] != read_json(resolve_path(config["sources"]["phase48_contract"]))["contract_sha256"]:
        raise ValueError("Phase-48 frozen contract identity differs")
    rows = _population(config)
    group_audit = audit_cross_dataset_groups(rows)
    if not group_audit["passed"]:
        raise RuntimeError("cross-dataset identical-image leakage invalidates the OOD protocol")
    membership_audit = {}
    for run, definition in config["runs"].items():
        membership = validate_ood_membership(
            rows,
            source_datasets=definition["source_datasets"],
            target_dataset=definition["target_dataset"],
        )
        membership_audit[run] = {key: len(value) for key, value in membership.items()}

    population_path = output_root / POPULATION_FILE
    population_payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    write_once_or_verify(population_path, population_payload)
    smoke_rows = _select_smoke(rows, int(config["input_control"]["smoke_records"]))
    smoke_path = output_root / SMOKE_FILE
    write_once_or_verify(
        smoke_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in smoke_rows).encode(),
    )
    schema = _input_schema(config)
    schema_path = output_root / INPUT_SCHEMA_FILE
    write_once_or_verify(
        schema_path,
        (json.dumps(schema, indent=2, sort_keys=True) + "\n").encode(),
    )
    model_root = resolve_path(config["sources"]["model_path"])
    model_hashes = model_file_hashes(model_root)
    source_hashes = {
        name: file_sha256(resolve_path(path))
        for name, path in config["sources"].items()
        if name != "model_path"
    }
    git_status = command_output(["git", "status", "--porcelain=v1", "--untracked-files=all"])
    contract = json.loads(json.dumps(config))
    contract["schema_version"] = "ood_first_failure_signal_frozen_protocol_v1"
    contract["membership_audit"] = membership_audit
    contract["cross_dataset_group_audit"] = group_audit
    contract["provenance"] = {
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_branch": command_output(["git", "branch", "--show-current"]),
        "git_status_porcelain_at_freeze": git_status.splitlines() if git_status else [],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "numpy": np.__version__,
        "matplotlib": package_version("matplotlib"),
        "cuda_runtime": torch.version.cuda,
        "gpu_inventory": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines(),
        "static_config_sha256": file_sha256(config_path),
        "source_sha256": source_hashes,
        "bound_code_sha256": {
            path: file_sha256(resolve_path(path)) for path in BOUND_CODE_PATHS
        },
        "population_manifest_sha256": file_sha256(population_path),
        "smoke_manifest_sha256": file_sha256(smoke_path),
        "input_feature_schema_sha256": file_sha256(schema_path),
        "model_resolved_path": str(model_root),
        "model_file_sha256": model_hashes,
    }
    contract["contract_sha256"] = canonical_hash(contract)
    contract_path = output_root / CONTRACT_FILE
    write_once_or_verify(
        contract_path,
        (json.dumps(contract, indent=2, sort_keys=True) + "\n").encode(),
    )
    write_once_or_verify(
        output_root / "protocol.md",
        _protocol_markdown(contract, group_audit).encode(),
    )
    atomic_json(
        output_root / "preparation_audit.json",
        {
            "passed": True,
            "contract_sha256": contract["contract_sha256"],
            "population_records": len(rows),
            "membership": membership_audit,
            "cross_dataset_group_audit": group_audit,
        },
    )
    print(json.dumps({"passed": True, "contract_sha256": contract["contract_sha256"], "membership": membership_audit}, sort_keys=True))


def load_contract(config_path: Path, *, verify_model: bool = False) -> tuple[dict[str, Any], Path]:
    static = load_static_config(config_path)
    output_root = resolve_path(static["output_root"])
    contract = read_json(output_root / CONTRACT_FILE)
    if contract.get("schema_version") != "ood_first_failure_signal_frozen_protocol_v1":
        raise ValueError("unsupported OOD frozen protocol")
    if contract.get("contract_sha256") != canonical_hash(contract):
        raise ValueError("OOD frozen protocol hash differs")
    provenance = contract["provenance"]
    if file_sha256(config_path) != provenance["static_config_sha256"]:
        raise ValueError("OOD static config differs from frozen protocol")
    for path, expected in provenance["bound_code_sha256"].items():
        if file_sha256(resolve_path(path)) != expected:
            raise ValueError(f"OOD bound code differs: {path}")
    for name, expected in provenance["source_sha256"].items():
        if file_sha256(resolve_path(contract["sources"][name])) != expected:
            raise ValueError(f"OOD source differs: {name}")
    if file_sha256(output_root / POPULATION_FILE) != provenance["population_manifest_sha256"]:
        raise ValueError("OOD population manifest differs")
    if file_sha256(output_root / SMOKE_FILE) != provenance["smoke_manifest_sha256"]:
        raise ValueError("OOD smoke manifest differs")
    if file_sha256(output_root / INPUT_SCHEMA_FILE) != provenance["input_feature_schema_sha256"]:
        raise ValueError("OOD input feature schema differs")
    model_root = resolve_path(contract["sources"]["model_path"])
    if str(model_root) != provenance["model_resolved_path"]:
        raise ValueError("resolved OOD model path differs")
    if verify_model and model_file_hashes(model_root) != provenance["model_file_sha256"]:
        raise ValueError("OOD model snapshot bytes differ")
    return contract, output_root


def _require_four_cuda(rank: int, world_size: int) -> torch.device:
    if world_size != 4 or not 0 <= rank < world_size:
        raise ValueError("the frozen diagnostic requires ranks 0-3 of four workers")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError("four CUDA devices are not visible")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    return device


def _load_runtime(contract: Mapping[str, Any], rank: int):
    _require_four_cuda(rank, int(contract["world_size"]))
    dense_config = read_json(resolve_path(contract["sources"]["dense_config"]))
    configure_dense_determinism(int(contract["seed"]), dense_config["backend_settings"])
    return load_dense_runtime(
        str(resolve_path(contract["sources"]["model_path"])),
        str(contract["model"]["revision"]),
        rank,
    )


@torch.inference_mode()
def _capture_sample(processor, model, device: torch.device, row: Mapping[str, Any]):
    sample = {
        "uid": row["uid"],
        "local_image_path": row["local_image_path"],
        "image_content_sha256": row["image_content_sha256"],
        "prompt": row["prompt"],
    }
    inputs, metadata = build_dense_inputs(processor, sample, device)
    positions = token_positions(processor, model, inputs["input_ids"])
    if hasattr(model, "model") and hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None
    layers = model.model.language_model.layers
    collector = PreDecoderInputCollector(
        layers,
        visual_positions=positions.visual,
        user_text_positions=positions.user_text,
        final_user_token_position=positions.final_user_token,
    )
    capture = collector.consume(model, **inputs, use_cache=False, return_dict=True)
    prompt_tokens = int(inputs["attention_mask"].sum().item())
    checks = {
        "sequence_equals_input_ids": capture.sequence_length == int(inputs["input_ids"].shape[1]),
        "prompt_token_count_matches_dense": prompt_tokens == int(row["expected_prompt_token_count"]),
        "visual_token_count_matches_dense": len(positions.visual) == int(row["expected_visual_token_count"]),
        "user_text_token_count_matches_dense": len(positions.user_text) == int(row["expected_user_text_token_count"]),
        "feature_dimension": int(capture.feature.numel()) == 10752,
        "feature_finite": bool(torch.isfinite(capture.feature.float()).all()),
        "one_prefill_capture": capture.capture_count == 1,
        "zero_decoder_forwards": capture.decoder_forward_count == 0,
        "hooks_removed": collector.active_handles == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"pre-decoder capture invariants failed for {row['uid']}: {checks}")
    result = {
        "feature": capture.feature.to(torch.bfloat16),
        "prompt_token_count": prompt_tokens,
        "visual_token_count": len(positions.visual),
        "user_text_token_count": len(positions.user_text),
        "sequence_length": capture.sequence_length,
        "literal_prompt_sha256": metadata["literal_prompt_sha256"],
        "consumed_image_sha256": metadata["consumed_image_sha256"],
        "checks": checks,
    }
    del inputs
    return result


def smoke_worker(config_path: Path, *, rank: int, world_size: int) -> None:
    contract, output_root = load_contract(config_path, verify_model=True)
    _require_four_cuda(rank, world_size)
    rows = read_jsonl(output_root / SMOKE_FILE)
    assigned = [row for index, row in enumerate(rows) if index % world_size == rank]
    processor, model, device = _load_runtime(contract, rank)
    results = []
    for row in assigned:
        first = _capture_sample(processor, model, device, row)
        second = _capture_sample(processor, model, device, row)
        exact = torch.equal(first["feature"], second["feature"])
        if not exact or first["checks"] != second["checks"]:
            raise RuntimeError(f"pre-decoder capture is not exactly repeatable for {row['uid']}")
        feature_bytes = first["feature"].view(torch.uint8).numpy().tobytes()
        results.append(
            {
                "schema_version": "ood_input_capture_smoke_row_v1",
                "uid": row["uid"],
                "dataset": row["dataset"],
                "current_dense_wrong": row["current_dense_wrong"],
                "worker_rank": rank,
                "exact_repeat_feature": exact,
                "feature_sha256": sha256(feature_bytes).hexdigest(),
                "prompt_token_count": first["prompt_token_count"],
                "visual_token_count": first["visual_token_count"],
                "user_text_token_count": first["user_text_token_count"],
                "sequence_length": first["sequence_length"],
                "consumed_image_sha256": first["consumed_image_sha256"],
                **first["checks"],
            }
        )
    worker_path = output_root / "smoke/workers" / f"rank{rank:02d}.jsonl"
    atomic_jsonl(worker_path, results)
    atomic_json(
        output_root / "smoke/workers" / f"rank{rank:02d}.complete.json",
        {
            "passed": True,
            "rank": rank,
            "world_size": world_size,
            "contract_sha256": contract["contract_sha256"],
            "records": len(results),
            "result_sha256": file_sha256(worker_path),
        },
    )
    print(json.dumps({"passed": True, "rank": rank, "records": len(results)}))


def aggregate_smoke(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    rows = []
    for rank in range(4):
        worker_path = output_root / "smoke/workers" / f"rank{rank:02d}.jsonl"
        complete = read_json(output_root / "smoke/workers" / f"rank{rank:02d}.complete.json")
        if (
            complete.get("passed") is not True
            or complete.get("contract_sha256") != contract["contract_sha256"]
            or complete.get("result_sha256") != file_sha256(worker_path)
        ):
            raise RuntimeError(f"capture smoke worker {rank} is incomplete or incompatible")
        rows.extend(read_jsonl(worker_path))
    expected = {str(row["uid"]) for row in read_jsonl(output_root / SMOKE_FILE)}
    observed = Counter(str(row["uid"]) for row in rows)
    boolean_checks = (
        "exact_repeat_feature",
        "sequence_equals_input_ids",
        "prompt_token_count_matches_dense",
        "visual_token_count_matches_dense",
        "user_text_token_count_matches_dense",
        "feature_dimension",
        "feature_finite",
        "one_prefill_capture",
        "zero_decoder_forwards",
        "hooks_removed",
    )
    passed = set(observed) == expected and all(observed[uid] == 1 for uid in expected) and all(
        all(bool(row[key]) for key in boolean_checks) for row in rows
    )
    report = {
        "schema_version": "ood_input_capture_smoke_audit_v1",
        "passed": passed,
        "contract_sha256": contract["contract_sha256"],
        "records": len(rows),
        "unique_uids": len(observed),
        "datasets": dict(Counter(row["dataset"] for row in rows)),
        "correct": sum(not bool(row["current_dense_wrong"]) for row in rows),
        "wrong": sum(bool(row["current_dense_wrong"]) for row in rows),
        "all_exact_repeat": all(bool(row["exact_repeat_feature"]) for row in rows),
        "all_zero_decoder_forwards": all(bool(row["zero_decoder_forwards"]) for row in rows),
    }
    atomic_json(output_root / "smoke/smoke_audit.json", report)
    if not passed:
        raise RuntimeError(f"native pre-decoder capture smoke failed: {report}")
    print(json.dumps(report, sort_keys=True))


def _validate_input_shard(
    path: Path,
    *,
    contract_sha256: str,
    schema_sha256: str,
) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("schema_version") != "ood_input_feature_shard_v1":
        raise ValueError(f"unsupported input feature shard: {path}")
    if value.get("contract_sha256") != contract_sha256 or value.get("feature_schema_sha256") != schema_sha256:
        raise ValueError(f"input feature shard provenance differs: {path}")
    features = value.get("features")
    uids = value.get("uids")
    if not isinstance(features, torch.Tensor) or features.dtype != torch.bfloat16 or features.ndim != 2 or features.shape[1] != 10752:
        raise ValueError(f"input feature shard tensor differs: {path}")
    if not isinstance(uids, list) or len(uids) != len(features) or len(set(uids)) != len(uids):
        raise ValueError(f"input feature shard UID coverage differs: {path}")
    if not torch.isfinite(features.float()).all():
        raise ValueError(f"input feature shard is non-finite: {path}")
    return value


def extract_worker(config_path: Path, *, rank: int, world_size: int) -> None:
    contract, output_root = load_contract(config_path, verify_model=True)
    smoke = read_json(output_root / "smoke/smoke_audit.json")
    if smoke.get("passed") is not True or smoke.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("full input extraction requires the compatible passing capture smoke")
    _require_four_cuda(rank, world_size)
    rows = read_jsonl(output_root / POPULATION_FILE)
    assigned = [row for index, row in enumerate(rows) if index % world_size == rank]
    processor, model, device = _load_runtime(contract, rank)
    schema_sha = contract["provenance"]["input_feature_schema_sha256"]
    shard_size = int(contract["input_control"]["shard_size"])
    index_rows = []
    for shard_id, start in enumerate(range(0, len(assigned), shard_size)):
        batch = assigned[start : start + shard_size]
        shard_path = output_root / "input_features/shards" / f"rank{rank:02d}_shard{shard_id:05d}.pt"
        if shard_path.exists():
            shard = _validate_input_shard(
                shard_path,
                contract_sha256=contract["contract_sha256"],
                schema_sha256=schema_sha,
            )
            if shard["uids"] != [row["uid"] for row in batch]:
                raise RuntimeError(f"existing input shard UIDs differ: {shard_path}")
        else:
            captures = [_capture_sample(processor, model, device, row) for row in batch]
            shard = {
                "schema_version": "ood_input_feature_shard_v1",
                "contract_sha256": contract["contract_sha256"],
                "model_revision": contract["model"]["revision"],
                "source_manifest_sha256": contract["provenance"]["population_manifest_sha256"],
                "feature_schema_sha256": schema_sha,
                "rank": rank,
                "world_size": world_size,
                "shard_id": shard_id,
                "uids": [row["uid"] for row in batch],
                "features": torch.stack([item["feature"] for item in captures]),
                "capture_metadata": [
                    {key: value for key, value in item.items() if key != "feature"}
                    for item in captures
                ],
            }
            atomic_torch_save(shard_path, shard)
            _validate_input_shard(
                shard_path,
                contract_sha256=contract["contract_sha256"],
                schema_sha256=schema_sha,
            )
        shard_relative = relative_project_path(shard_path)
        shard_sha = file_sha256(shard_path)
        index_rows.extend(
            {
                "schema_version": "ood_input_feature_index_v1",
                "uid": uid,
                "shard": shard_relative,
                "row_index": row_index,
                "feature_dimension": 10752,
                "contract_sha256": contract["contract_sha256"],
                "feature_schema_sha256": schema_sha,
                "shard_sha256": shard_sha,
            }
            for row_index, uid in enumerate(shard["uids"])
        )
    index_path = output_root / "input_features/workers" / f"rank{rank:02d}.index.jsonl"
    atomic_jsonl(index_path, index_rows)
    atomic_json(
        output_root / "input_features/workers" / f"rank{rank:02d}.complete.json",
        {
            "passed": True,
            "rank": rank,
            "world_size": world_size,
            "contract_sha256": contract["contract_sha256"],
            "records": len(index_rows),
            "shards": len({row["shard"] for row in index_rows}),
            "index_sha256": file_sha256(index_path),
        },
    )
    print(json.dumps({"passed": True, "rank": rank, "records": len(index_rows)}))


def aggregate_extraction(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    expected = {str(row["uid"]) for row in read_jsonl(output_root / POPULATION_FILE)}
    all_rows = []
    for rank in range(4):
        index_path = output_root / "input_features/workers" / f"rank{rank:02d}.index.jsonl"
        complete = read_json(output_root / "input_features/workers" / f"rank{rank:02d}.complete.json")
        if (
            complete.get("passed") is not True
            or complete.get("contract_sha256") != contract["contract_sha256"]
            or complete.get("index_sha256") != file_sha256(index_path)
        ):
            raise RuntimeError(f"input extraction worker {rank} is incomplete or incompatible")
        all_rows.extend(read_jsonl(index_path))
    observed = Counter(str(row["uid"]) for row in all_rows)
    if set(observed) != expected or any(observed[uid] != 1 for uid in expected):
        raise RuntimeError("input extraction does not cover every population UID exactly once")
    schema_sha = contract["provenance"]["input_feature_schema_sha256"]
    shards = {}
    for row in all_rows:
        path = resolve_path(row["shard"])
        if file_sha256(path) != row["shard_sha256"]:
            raise RuntimeError(f"input feature shard hash differs: {path}")
        shards[str(row["shard"])] = str(row["shard_sha256"])
    for relative in sorted(shards):
        _validate_input_shard(
            resolve_path(relative),
            contract_sha256=contract["contract_sha256"],
            schema_sha256=schema_sha,
        )
    all_rows.sort(key=lambda row: str(row["uid"]))
    index_path = output_root / "input_features/feature_index.jsonl"
    atomic_jsonl(index_path, all_rows)
    shard_manifest = {
        "schema_version": "ood_input_feature_shard_manifest_v1",
        "contract_sha256": contract["contract_sha256"],
        "source_manifest_sha256": contract["provenance"]["population_manifest_sha256"],
        "feature_schema_sha256": schema_sha,
        "records": len(all_rows),
        "shards": [{"path": path, "sha256": value} for path, value in sorted(shards.items())],
    }
    manifest_path = output_root / "input_features/shard_manifest.json"
    atomic_json(manifest_path, shard_manifest)
    audit = {
        "schema_version": "ood_input_feature_extraction_audit_v1",
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        "records": len(all_rows),
        "unique_uids": len(observed),
        "shards": len(shards),
        "feature_index_sha256": file_sha256(index_path),
        "shard_manifest_sha256": file_sha256(manifest_path),
        "feature_schema_sha256": schema_sha,
    }
    atomic_json(output_root / "input_features/extraction_audit.json", audit)
    print(json.dumps(audit, sort_keys=True))


def _matrix_from_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    index_by_uid: Mapping[str, Mapping[str, Any]],
    tensor_key: str,
    expected_dimension: int,
    shard_validator=None,
) -> torch.Tensor:
    matrix = torch.empty((len(rows), expected_dimension), dtype=torch.bfloat16)
    by_shard: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for destination, row in enumerate(rows):
        uid = str(row["uid"])
        if uid not in index_by_uid:
            raise ValueError(f"feature index lacks UID: {uid}")
        by_shard[str(index_by_uid[uid]["shard"])].append((destination, index_by_uid[uid]))
    filled = torch.zeros(len(rows), dtype=torch.bool)
    for relative, selections in sorted(by_shard.items()):
        path = resolve_path(relative)
        shard = shard_validator(path) if shard_validator is not None else torch.load(
            path, map_location="cpu", weights_only=True
        )
        tensor = shard.get(tensor_key)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[1] != expected_dimension:
            raise ValueError(f"feature tensor shape differs: {path}")
        destinations = torch.tensor([item[0] for item in selections], dtype=torch.long)
        source_indices = [int(item[1]["row_index"]) for item in selections]
        for (_, index_row), source_index in zip(selections, source_indices):
            if not 0 <= source_index < len(shard["uids"]):
                raise ValueError(f"feature row lies outside shard for {index_row['uid']}")
            if str(shard["uids"][source_index]) != str(index_row["uid"]):
                raise ValueError(f"feature shard UID differs for {index_row['uid']}")
        values = tensor.index_select(0, torch.tensor(source_indices, dtype=torch.long))
        matrix.index_copy_(0, destinations, values.to(torch.bfloat16))
        filled.index_fill_(0, destinations, True)
    if not bool(filled.all()) or not torch.isfinite(matrix.float()).all():
        raise RuntimeError("feature matrix is incomplete or non-finite")
    return matrix


def load_input_matrix(
    contract: Mapping[str, Any], output_root: Path, rows: Sequence[Mapping[str, Any]]
) -> torch.Tensor:
    audit = read_json(output_root / "input_features/extraction_audit.json")
    if audit.get("passed") is not True or audit.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("input feature extraction audit is incompatible")
    index_path = output_root / "input_features/feature_index.jsonl"
    manifest_path = output_root / "input_features/shard_manifest.json"
    if audit.get("feature_index_sha256") != file_sha256(index_path) or audit.get("shard_manifest_sha256") != file_sha256(manifest_path):
        raise RuntimeError("input feature aggregate hashes differ")
    index_rows = read_jsonl(index_path)
    index_by_uid = {str(row["uid"]): row for row in index_rows}
    if len(index_by_uid) != len(index_rows):
        raise ValueError("input feature index contains duplicate UIDs")
    schema_sha = contract["provenance"]["input_feature_schema_sha256"]

    def validator(path: Path):
        return _validate_input_shard(
            path,
            contract_sha256=contract["contract_sha256"],
            schema_sha256=schema_sha,
        )

    return _matrix_from_index(
        rows,
        index_by_uid=index_by_uid,
        tensor_key="features",
        expected_dimension=10752,
        shard_validator=validator,
    )


def load_hidden_matrices(
    contract: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    phase48 = read_json(resolve_path(contract["sources"]["phase48_contract"]))
    phase48_root = resolve_path(phase48["output_root"])
    shard_manifest = read_json(phase48_root / "feature_shard_manifest.json")
    shard_hashes = {str(item["path"]): str(item["sha256"]) for item in shard_manifest["shards"]}
    dimensions = int(phase48["input"]["input_size"])
    matrices = {
        int(layer): torch.empty((len(rows), dimensions), dtype=torch.bfloat16)
        for layer in layers
    }
    filled = torch.zeros(len(rows), dtype=torch.bool)
    by_shard: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for destination, row in enumerate(rows):
        by_shard[str(row["hidden_feature_shard"])].append((destination, row))
    if set(by_shard) - set(shard_hashes):
        raise ValueError("population references hidden shards outside Phase-48 freeze")
    for relative, selections in sorted(by_shard.items()):
        path = resolve_path(relative)
        payload = path.read_bytes()
        if sha256(payload).hexdigest() != shard_hashes[relative]:
            raise ValueError(f"Phase-48 hidden feature shard hash differs: {path}")
        shard = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if shard.get("layer_ids") != list(range(28)):
            raise ValueError(f"hidden shard layer schema differs: {path}")
        destinations = torch.tensor([item[0] for item in selections], dtype=torch.long)
        source_indices = [int(item[1]["hidden_feature_row"]) for item in selections]
        for (_, row), source_index in zip(selections, source_indices):
            if not 0 <= source_index < len(shard["uids"]) or str(shard["uids"][source_index]) != str(row["uid"]):
                raise ValueError(f"hidden shard UID differs for {row['uid']}")
        for layer in layers:
            values = compose_layer_features(shard, row_indices=source_indices, layer=int(layer))
            matrices[int(layer)].index_copy_(0, destinations, values.to(torch.bfloat16))
        filled.index_fill_(0, destinations, True)
    if not bool(filled.all()) or any(not torch.isfinite(value.float()).all() for value in matrices.values()):
        raise RuntimeError("hidden feature matrices are incomplete or non-finite")
    return matrices


def _representation_schedule(rank: int, world_size: int) -> tuple[bool, list[int]]:
    representations: list[str | int] = ["input", *range(28)]
    assigned = [value for index, value in enumerate(representations) if index % world_size == rank]
    return "input" in assigned, [int(value) for value in assigned if value != "input"]


def _metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item.item() if isinstance(item, np.generic) else item for key, item in value.items()}


def _checkpoint_path(output_root: Path, run: str, representation: str, layer: int | None) -> Path:
    name = "input.pt" if representation == "input" else f"layer_{int(layer):02d}.pt"
    return output_root / "checkpoints" / run / name


def _checkpoint_expected(contract: Mapping[str, Any], output_root: Path) -> dict[str, str]:
    extraction = read_json(output_root / "input_features/extraction_audit.json")
    return {
        "contract_sha256": str(contract["contract_sha256"]),
        "population_manifest_sha256": str(contract["provenance"]["population_manifest_sha256"]),
        "phase48_contract_sha256": str(read_json(resolve_path(contract["sources"]["phase48_contract"]))["contract_sha256"]),
        "input_feature_schema_sha256": str(contract["provenance"]["input_feature_schema_sha256"]),
        "input_feature_index_sha256": str(extraction["feature_index_sha256"]),
        "input_feature_shard_manifest_sha256": str(extraction["shard_manifest_sha256"]),
    }


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    run: str,
    representation: str,
    layer: int | None,
    expected: Mapping[str, str],
) -> None:
    if checkpoint.get("schema_version") != "ood_failure_linear_probe_checkpoint_v1":
        raise ValueError("unsupported OOD checkpoint schema")
    if checkpoint.get("run") != run or checkpoint.get("representation") != representation:
        raise ValueError("OOD checkpoint task identity differs")
    if checkpoint.get("layer") != layer:
        raise ValueError("OOD checkpoint layer differs")
    mismatches = [key for key, value in expected.items() if checkpoint.get(key) != value]
    if mismatches:
        raise ValueError(f"OOD checkpoint provenance differs: {mismatches}")


def train_worker(config_path: Path, *, rank: int, world_size: int) -> None:
    contract, output_root = load_contract(config_path)
    device = _require_four_cuda(rank, world_size)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)
    rows = read_jsonl(output_root / POPULATION_FILE)
    include_input, layers = _representation_schedule(rank, world_size)
    input_matrix = load_input_matrix(contract, output_root, rows) if include_input else None
    hidden_matrices = load_hidden_matrices(contract, rows, layers=layers)
    expected = _checkpoint_expected(contract, output_root)
    training_config = dict(contract["training"])
    training_config["normalization_std_floor"] = float(contract["training"]["normalization_std_floor"])
    summaries = []
    run_names = list(contract["runs"])
    representations: list[tuple[str, int | None, torch.Tensor]] = []
    if input_matrix is not None:
        representations.append(("input", None, input_matrix))
    representations.extend(("hidden", layer, hidden_matrices[layer]) for layer in layers)
    for representation, layer, matrix in representations:
        representation_index = 0 if representation == "input" else int(layer) + 1
        for run_index, run in enumerate(run_names):
            definition = contract["runs"][run]
            membership = validate_ood_membership(
                rows,
                source_datasets=definition["source_datasets"],
                target_dataset=definition["target_dataset"],
            )
            train_indices = torch.tensor(membership["source_train"], dtype=torch.long)
            validation_indices = torch.tensor(membership["source_validation"], dtype=torch.long)
            labels = torch.tensor([int(bool(row["current_dense_wrong"])) for row in rows], dtype=torch.int64)
            seed = int(contract["seed"]) + run_index * 1000 + representation_index
            fit = fit_linear_probe(
                matrix.index_select(0, train_indices),
                labels.index_select(0, train_indices),
                matrix.index_select(0, validation_indices),
                labels.index_select(0, validation_indices),
                config=training_config,
                seed=seed,
                device=device,
            )
            validation_scores = score_linear_probe(
                matrix.index_select(0, validation_indices), fit, device=device
            ).numpy()
            validation_labels = labels.index_select(0, validation_indices).numpy()
            validation_metrics = _metrics(binary_metrics(validation_labels, validation_scores))
            selective = {
                f"{float(target):.2f}": _metrics(
                    preservation_operating_point(
                        validation_labels,
                        validation_scores,
                        target_preservation=float(target),
                    )
                )
                for target in contract["evaluation"]["preservation_targets"]
            }
            summary = {
                "schema_version": "ood_failure_source_validation_summary_v1",
                "run": run,
                "source_datasets": list(definition["source_datasets"]),
                "target_dataset": definition["target_dataset"],
                "representation": representation,
                "layer": layer,
                "seed": seed,
                "worker_rank": rank,
                "source_train_records": len(train_indices),
                "source_validation_records": len(validation_indices),
                "best_epoch": int(fit["best_epoch"]),
                "epochs_ran": int(fit["epochs_ran"]),
                "train_loss": float(fit["train_loss"]),
                "validation_loss": float(fit["validation_loss"]),
                "validation": validation_metrics,
                "validation_selective": selective,
            }
            checkpoint = {
                "schema_version": "ood_failure_linear_probe_checkpoint_v1",
                **expected,
                "run": run,
                "representation": representation,
                "layer": layer,
                "input_size": int(matrix.shape[1]),
                "normalization_mean": fit["mean"],
                "normalization_std": fit["std"],
                "weight": fit["weight"],
                "bias": fit["bias"],
                "summary": summary,
            }
            checkpoint_path = _checkpoint_path(output_root, run, representation, layer)
            atomic_torch_save(checkpoint_path, checkpoint)
            _validate_checkpoint(
                torch.load(checkpoint_path, map_location="cpu", weights_only=True),
                run=run,
                representation=representation,
                layer=layer,
                expected=expected,
            )
            summary["checkpoint_sha256"] = file_sha256(checkpoint_path)
            history_name = "input" if representation == "input" else f"layer_{int(layer):02d}"
            atomic_json(
                output_root / "training_history" / run / f"{history_name}.json",
                {
                    "contract_sha256": contract["contract_sha256"],
                    "run": run,
                    "representation": representation,
                    "layer": layer,
                    "history": fit["history"],
                },
            )
            summaries.append(summary)
            del fit, checkpoint
            torch.cuda.empty_cache()
    worker_path = output_root / "source_workers" / f"rank{rank:02d}.jsonl"
    atomic_jsonl(worker_path, summaries)
    atomic_json(
        output_root / "source_workers" / f"rank{rank:02d}.complete.json",
        {
            "passed": True,
            "rank": rank,
            "world_size": world_size,
            "contract_sha256": contract["contract_sha256"],
            "records": len(summaries),
            "result_sha256": file_sha256(worker_path),
        },
    )
    print(json.dumps({"passed": True, "rank": rank, "tasks": len(summaries)}))


def aggregate_source(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    expected_provenance = _checkpoint_expected(contract, output_root)
    records = []
    for rank in range(4):
        worker_path = output_root / "source_workers" / f"rank{rank:02d}.jsonl"
        complete = read_json(output_root / "source_workers" / f"rank{rank:02d}.complete.json")
        if (
            complete.get("passed") is not True
            or complete.get("contract_sha256") != contract["contract_sha256"]
            or complete.get("result_sha256") != file_sha256(worker_path)
        ):
            raise RuntimeError(f"source worker {rank} is incomplete or incompatible")
        records.extend(read_jsonl(worker_path))
    observed = Counter((row["run"], row["representation"], row["layer"]) for row in records)
    expected_tasks = {
        (run, "input", None) for run in contract["runs"]
    } | {
        (run, "hidden", layer) for run in contract["runs"] for layer in range(28)
    }
    if set(observed) != expected_tasks or any(observed[task] != 1 for task in expected_tasks):
        raise RuntimeError("source workers do not cover all 87 probe tasks exactly once")
    for row in records:
        path = _checkpoint_path(output_root, row["run"], row["representation"], row["layer"])
        if row["checkpoint_sha256"] != file_sha256(path):
            raise RuntimeError(f"checkpoint hash differs for {row['run']} {row['layer']}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        _validate_checkpoint(
            checkpoint,
            run=row["run"],
            representation=row["representation"],
            layer=row["layer"],
            expected=expected_provenance,
        )
    records.sort(key=lambda row: (row["run"], row["representation"], -1 if row["layer"] is None else int(row["layer"])))
    results_path = output_root / "source_validation_results.json"
    atomic_json(
        results_path,
        {
            "schema_version": "ood_failure_source_validation_results_v1",
            "contract_sha256": contract["contract_sha256"],
            "ood_target_evaluated": False,
            "tasks": records,
        },
    )
    decisions = {}
    for run in contract["runs"]:
        hidden = [
            {"layer": row["layer"], "validation_auroc": row["validation"]["auroc"]}
            for row in records
            if row["run"] == run and row["representation"] == "hidden"
        ]
        selected = choose_representative_layer(hidden)
        decisions[run] = {
            "target_dataset": contract["runs"][run]["target_dataset"],
            "representative_layer": selected,
            "selection_rule": contract["evaluation"]["representative_layer_selection"],
            "source_validation_auroc": next(
                float(row["validation"]["auroc"])
                for row in records
                if row["run"] == run and row["representation"] == "hidden" and int(row["layer"]) == selected
            ),
        }
    decision = {
        "schema_version": "ood_failure_source_decisions_v1",
        "contract_sha256": contract["contract_sha256"],
        "source_validation_results_sha256": file_sha256(results_path),
        "ood_target_evaluated": False,
        "runs": decisions,
    }
    decision_path = output_root / "source_decisions.json"
    write_once_or_verify(
        decision_path,
        (json.dumps(decision, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps({"passed": True, "tasks": len(records), "decisions": decisions}, sort_keys=True))


def evaluate_worker(config_path: Path, *, rank: int, world_size: int) -> None:
    contract, output_root = load_contract(config_path)
    device = _require_four_cuda(rank, world_size)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)
    decision_path = output_root / "source_decisions.json"
    decisions = read_json(decision_path)
    if decisions.get("contract_sha256") != contract["contract_sha256"] or decisions.get("ood_target_evaluated") is not False:
        raise ValueError("source-only decisions are not compatible or were not pre-target")
    rows = read_jsonl(output_root / POPULATION_FILE)
    include_input, layers = _representation_schedule(rank, world_size)
    input_matrix = load_input_matrix(contract, output_root, rows) if include_input else None
    hidden_matrices = load_hidden_matrices(contract, rows, layers=layers)
    matrices: list[tuple[str, int | None, torch.Tensor]] = []
    if input_matrix is not None:
        matrices.append(("input", None, input_matrix))
    matrices.extend(("hidden", layer, hidden_matrices[layer]) for layer in layers)
    checkpoint_expected = _checkpoint_expected(contract, output_root)
    phase48_contract, phase48_root = load_phase48_contract(
        resolve_path("configs/layerwise_dense_failure_probe_v1.json")
    )
    phase48_expected = phase48_checkpoint_expected(phase48_contract)
    labels = np.asarray([int(bool(row["current_dense_wrong"])) for row in rows], dtype=np.int64)
    results = []
    for representation, layer, matrix in matrices:
        in_domain_checkpoint = None
        if representation == "hidden":
            phase48_path = phase48_root / "checkpoints" / f"layer_{int(layer):02d}.pt"
            in_domain_checkpoint = torch.load(
                phase48_path, map_location="cpu", weights_only=True
            )
            validate_checkpoint_provenance(
                in_domain_checkpoint,
                layer=int(layer),
                expected=phase48_expected,
            )
        for run, definition in contract["runs"].items():
            membership = validate_ood_membership(
                rows,
                source_datasets=definition["source_datasets"],
                target_dataset=definition["target_dataset"],
            )
            target_indices = torch.tensor(membership["target_all"], dtype=torch.long)
            subset_local = [
                membership["target_all"].index(index)
                for index in membership["target_phase48_test"]
            ]
            checkpoint_path = _checkpoint_path(output_root, run, representation, layer)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            _validate_checkpoint(
                checkpoint,
                run=run,
                representation=representation,
                layer=layer,
                expected=checkpoint_expected,
            )
            state = {
                "mean": checkpoint["normalization_mean"],
                "std": checkpoint["normalization_std"],
                "weight": checkpoint["weight"],
                "bias": checkpoint["bias"],
            }
            scores = score_linear_probe(matrix.index_select(0, target_indices), state, device=device).numpy()
            target_labels = labels[np.asarray(membership["target_all"], dtype=np.int64)]
            full_selective = {}
            subset_selective = {}
            subset_scores = scores[np.asarray(subset_local, dtype=np.int64)]
            subset_labels = target_labels[np.asarray(subset_local, dtype=np.int64)]
            for target in contract["evaluation"]["preservation_targets"]:
                key = f"{float(target):.2f}"
                threshold = float(checkpoint["summary"]["validation_selective"][key]["threshold"])
                full_selective[key] = _metrics(binary_metrics(target_labels, scores, threshold=threshold))
                subset_selective[key] = _metrics(binary_metrics(subset_labels, subset_scores, threshold=threshold))
            in_domain_reference = None
            if in_domain_checkpoint is not None:
                reference_state = {
                    "mean": in_domain_checkpoint["normalization_mean"],
                    "std": in_domain_checkpoint["normalization_std"],
                    "weight": in_domain_checkpoint["weight"],
                    "bias": in_domain_checkpoint["bias"],
                }
                reference_scores = score_linear_probe(
                    matrix.index_select(
                        0,
                        torch.tensor(
                            membership["target_phase48_test"], dtype=torch.long
                        ),
                    ),
                    reference_state,
                    device=device,
                ).numpy()
                reference_labels = labels[
                    np.asarray(membership["target_phase48_test"], dtype=np.int64)
                ]
                reference_threshold = float(
                    in_domain_checkpoint["summary"]["validation_selective"]["0.99"][
                        "threshold"
                    ]
                )
                in_domain_reference = {
                    "checkpoint_sha256": file_sha256(phase48_path),
                    "metrics": _metrics(binary_metrics(reference_labels, reference_scores)),
                    "selective_99": _metrics(
                        binary_metrics(
                            reference_labels,
                            reference_scores,
                            threshold=reference_threshold,
                        )
                    ),
                    "validation_threshold_99": reference_threshold,
                }
            results.append(
                {
                    "schema_version": "ood_failure_target_result_v1",
                    "run": run,
                    "target_dataset": definition["target_dataset"],
                    "representation": representation,
                    "layer": layer,
                    "worker_rank": rank,
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "source_validation": checkpoint["summary"]["validation"],
                    "source_validation_selective": checkpoint["summary"]["validation_selective"],
                    "target_full": _metrics(binary_metrics(target_labels, scores)),
                    "target_full_selective": full_selective,
                    "target_phase48_test": _metrics(binary_metrics(subset_labels, subset_scores)),
                    "target_phase48_test_selective": subset_selective,
                    "in_domain_reference_same_uids": in_domain_reference,
                }
            )
            torch.cuda.empty_cache()
    worker_path = output_root / "ood_workers" / f"rank{rank:02d}.jsonl"
    atomic_jsonl(worker_path, results)
    atomic_json(
        output_root / "ood_workers" / f"rank{rank:02d}.complete.json",
        {
            "passed": True,
            "rank": rank,
            "world_size": world_size,
            "contract_sha256": contract["contract_sha256"],
            "source_decisions_sha256": file_sha256(decision_path),
            "records": len(results),
            "result_sha256": file_sha256(worker_path),
        },
    )
    print(json.dumps({"passed": True, "rank": rank, "tasks": len(results)}))


def _run_summary_markdown(
    contract: Mapping[str, Any],
    run: str,
    decision: Mapping[str, Any],
    hidden: Sequence[Mapping[str, Any]],
    input_row: Mapping[str, Any],
) -> str:
    representative = next(row for row in hidden if int(row["layer"]) == int(decision["representative_layer"]))
    target = contract["runs"][run]["target_dataset"]
    lines = [
        f"# Leave {target} Out",
        "",
        f"- Source datasets: `{', '.join(contract['runs'][run]['source_datasets'])}`",
        f"- OOD target: `{target}` (`{int(representative['target_full']['records']):,}` rows)",
        f"- Source-validation-selected representative layer: `{decision['representative_layer']}`",
        f"- Representative OOD AUROC / AUPRC: `{float(representative['target_full']['auroc']):.4f}` / `{float(representative['target_full']['auprc']):.4f}`",
        f"- Native pre-language-decoder OOD AUROC / AUPRC: `{float(input_row['target_full']['auroc']):.4f}` / `{float(input_row['target_full']['auprc']):.4f}`",
        f"- Representative hidden-minus-input AUROC: `{float(representative['target_full']['auroc']) - float(input_row['target_full']['auroc']):+.4f}`",
        "",
        "| Source preservation target | Transferred threshold | OOD actual preservation | OOD wrong recall | OOD failure precision |",
        "|---:|---:|---:|---:|---:|",
    ]
    for target_preservation in contract["evaluation"]["preservation_targets"]:
        key = f"{float(target_preservation):.2f}"
        point = representative["target_full_selective"][key]
        threshold = representative["source_validation_selective"][key]["threshold"]
        lines.append(
            f"| {float(target_preservation):.0%} | {float(threshold):.6f} | "
            f"{float(point['correct_preservation']):.4f} | {float(point['wrong_recall']):.4f} | "
            f"{float(point['failure_precision']):.4f} |"
        )
    lines.extend(
        [
            "",
            "The target dataset was not used for fitting, checkpoint selection, threshold calibration, probability calibration, or representative-layer selection.",
            "",
        ]
    )
    return "\n".join(lines)


def _plot(
    output_root: Path,
    contract: Mapping[str, Any],
    by_run: Mapping[str, Mapping[tuple[str, int | None], Mapping[str, Any]]],
    input_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_root / "figures"
    colors = {"textvqa": "#1b9e77", "chartqa": "#d95f02", "gqa": "#7570b3"}
    layers = np.arange(28)
    for metric, filename, label in (
        ("auroc", "ood_layerwise_auroc.png", "OOD AUROC"),
        ("auprc", "ood_layerwise_auprc.png", "OOD AUPRC"),
    ):
        figure, axis = plt.subplots(figsize=(8.0, 4.8))
        for run in contract["runs"]:
            target = contract["runs"][run]["target_dataset"]
            values = [float(by_run[run][("hidden", layer)]["target_full"][metric]) for layer in layers]
            axis.plot(layers, values, marker="o", markersize=2.5, label=target, color=colors[target])
            selected = int(decisions[run]["representative_layer"])
            axis.scatter([selected], [values[selected]], s=45, color=colors[target], edgecolor="black", zorder=3)
        axis.axhline(0.5, color="black", linestyle=":", linewidth=1)
        axis.set_xlabel("Decoder layer")
        axis.set_ylabel(label)
        axis.set_xticks(range(0, 28, 2))
        axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(figures / filename, dpi=180)
        plt.close(figure)

    fixed = ["input", 0, 14, 21, 27]
    x = np.arange(len(fixed))
    width = 0.24
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for run_index, run in enumerate(contract["runs"]):
        target = contract["runs"][run]["target_dataset"]
        values = [
            float(by_run[run][("input", None)]["target_full"]["auroc"])
            if value == "input"
            else float(by_run[run][("hidden", value)]["target_full"]["auroc"])
            for value in fixed
        ]
        axis.bar(x + (run_index - 1) * width, values, width, label=target, color=colors[target])
    axis.axhline(0.5, color="black", linestyle=":", linewidth=1)
    axis.set_xticks(x, ["pre-decoder", "L0", "L14", "L21", "L27"])
    axis.set_ylabel("OOD AUROC")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figures / "input_vs_hidden_ood.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for target in DATASETS:
        selected = [row for row in comparison_rows if row["dataset"] == target]
        selected.sort(key=lambda row: int(row["layer"]))
        axis.plot(
            [int(row["layer"]) for row in selected],
            [float(row["auroc_drop"]) for row in selected],
            marker="o",
            markersize=2.5,
            label=target,
            color=colors[target],
        )
    axis.axhline(0.0, color="black", linestyle=":", linewidth=1)
    axis.set_xlabel("Decoder layer")
    axis.set_ylabel("In-domain AUROC - OOD AUROC\n(same target-test UIDs)")
    axis.set_xticks(range(0, 28, 2))
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figures / "in_domain_vs_ood_drop.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    targets = [contract["runs"][run]["target_dataset"] for run in contract["runs"]]
    x = np.arange(len(targets))
    for offset_index, preservation in enumerate(contract["evaluation"]["preservation_targets"]):
        key = f"{float(preservation):.2f}"
        values = []
        for run in contract["runs"]:
            layer = int(decisions[run]["representative_layer"])
            values.append(float(by_run[run][("hidden", layer)]["target_full_selective"][key]["wrong_recall"]))
        offset = (offset_index - 1) * width
        axis.bar(x + offset, values, width, label=f"{float(preservation):.0%} preserve")
    axis.set_xticks(x, targets)
    axis.set_ylabel("OOD wrong recall")
    axis.set_ylim(0, 1)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figures / "ood_wrong_recall_at_fixed_preservation.png", dpi=180)
    plt.close(figure)


def _decision_summary(
    contract: Mapping[str, Any],
    by_run: Mapping[str, Mapping[tuple[str, int | None], Mapping[str, Any]]],
    decisions: Mapping[str, Any],
    comparison_rows: Sequence[Mapping[str, Any]],
) -> str:
    representative_rows = []
    for run in contract["runs"]:
        layer = int(decisions[run]["representative_layer"])
        hidden = by_run[run][("hidden", layer)]
        input_row = by_run[run][("input", None)]
        representative_rows.append(
            {
                "run": run,
                "target": contract["runs"][run]["target_dataset"],
                "layer": layer,
                "auroc": float(hidden["target_full"]["auroc"]),
                "auprc": float(hidden["target_full"]["auprc"]),
                "input_auroc": float(input_row["target_full"]["auroc"]),
                "delta": float(hidden["target_full"]["auroc"]) - float(input_row["target_full"]["auroc"]),
                "preserve99": float(hidden["target_full_selective"]["0.99"]["correct_preservation"]),
                "recall99": float(hidden["target_full_selective"]["0.99"]["wrong_recall"]),
                "precision99": float(hidden["target_full_selective"]["0.99"]["failure_precision"]),
            }
        )
    floor = float(contract["interpretation"]["meaningful_transfer_auroc_floor_all_targets"])
    delta_floor = float(contract["interpretation"]["hidden_over_input_minimum_delta"])
    minimum = float(contract["interpretation"]["minimum_target_auroc_for_shared_predictor_support"])
    required = int(contract["interpretation"]["hidden_gain_required_targets"])
    transfer_pass = all(row["auroc"] >= floor for row in representative_rows)
    gain_count = sum(row["delta"] >= delta_floor for row in representative_rows)
    support = transfer_pass and gain_count >= required and all(row["auroc"] >= minimum for row in representative_rows)
    depth_rows = []
    for run in contract["runs"]:
        target = contract["runs"][run]["target_dataset"]
        layer0 = float(by_run[run][("hidden", 0)]["target_full"]["auroc"])
        layer14 = float(by_run[run][("hidden", 14)]["target_full"]["auroc"])
        layer21 = float(by_run[run][("hidden", 21)]["target_full"]["auroc"])
        layer27 = float(by_run[run][("hidden", 27)]["target_full"]["auroc"])
        depth_rows.append((target, layer0, layer14, layer21, layer27))
    representative_comparisons = []
    for row in representative_rows:
        match = next(
            item
            for item in comparison_rows
            if item["dataset"] == row["target"] and int(item["layer"]) == row["layer"]
        )
        representative_comparisons.append(match)
    lines = [
        "# OOD First-Failure Signal Diagnostic Decision Summary",
        "",
        f"Frozen protocol: `{contract['contract_sha256']}`.",
        "",
        "| Source datasets | OOD target | Source-selected layer | OOD AUROC | OOD AUPRC | Input AUROC | Hidden - input | Actual preservation @ source 99% | Wrong recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in representative_rows:
        source = " + ".join(contract["runs"][row["run"]]["source_datasets"])
        lines.append(
            f"| {source} | {row['target']} | {row['layer']} | {row['auroc']:.4f} | {row['auprc']:.4f} | "
            f"{row['input_auroc']:.4f} | {row['delta']:+.4f} | {row['preserve99']:.4f} | {row['recall99']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Q1. Does prediction transfer to an entirely unseen target benchmark?",
            "",
            (
                f"Yes under the frozen descriptive floor: all three source-selected probes reached OOD AUROC at least {floor:.2f}."
                if transfer_pass
                else f"Not uniformly under the frozen descriptive floor of {floor:.2f}; at least one source-selected target fell below it."
            ),
            "",
            "## Q2. How much does performance drop from in-domain to OOD?",
            "",
            "The comparison below uses the exact same Phase-48 target-test UIDs; positive values mean in-domain is better.",
            "",
            "| Target | Layer | In-domain AUROC | OOD AUROC | AUROC drop | In-domain recall @ own val-99% | OOD recall @ source-val-99% | Recall drop |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in representative_comparisons:
        lines.append(
            f"| {row['dataset']} | {int(row['layer'])} | {float(row['in_domain_auroc']):.4f} | "
            f"{float(row['ood_auroc']):.4f} | {float(row['auroc_drop']):+.4f} | "
            f"{float(row['in_domain_wrong_recall_99']):.4f} | {float(row['ood_wrong_recall_99']):.4f} | "
            f"{float(row['wrong_recall_drop_99']):+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Q3. Do decoder hidden states outperform the input control OOD?",
            "",
            f"The source-selected hidden state exceeded the native pre-language-decoder control by at least {delta_floor:.2f} AUROC on `{gain_count}` of 3 targets. "
            "The control already includes learned visual encoding and multimodal insertion, so this comparison isolates additions after language-decoder computation rather than raw-input difficulty.",
            "",
            "## Q4. Does OOD predictability improve with depth?",
            "",
            "| Target | L0 AUROC | L14 AUROC | L21 AUROC | L27 AUROC | L21 - L0 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for target, layer0, layer14, layer21, layer27 in depth_rows:
        lines.append(
            f"| {target} | {layer0:.4f} | {layer14:.4f} | {layer21:.4f} | {layer27:.4f} | {layer21-layer0:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Q5. Is conservative wrong detection useful OOD?",
            "",
            "At each source-selected layer, the table at the top reports the actual OOD preservation and wrong recall after transferring the source-validation 99% threshold unchanged. Full 99/98/95% curves are in each run's `selective_metrics.csv`.",
            "",
            "## Q6. Is the signal benchmark-general rather than purely benchmark-specific?",
            "",
            (
                "The fixed heuristic supports a benchmark-general component: transfer clears the all-target floor and decoder states add the required margin over the native pre-decoder control. This remains evidence across three related VQA datasets, not a universal OOD claim."
                if support
                else "The fixed heuristic does not support a strong benchmark-general computation-dependent claim. The evidence is mixed or benchmark-dependent, even if some individual transfers are useful."
            ),
            "",
            "## Q7. Should the shared Stage-1 predictor be the next experiment?",
            "",
            (
                "Yes as a separately authorized experiment: the frozen criteria support testing a shared predictor with learnable layer embeddings. No such model was trained here."
                if support
                else "Not yet on the benchmark-general rationale. A shared predictor could only be justified as a deployment-mixture predictor or after a separately approved diagnostic/pivot. No further experiment is selected here."
            ),
            "",
            "## Validity limits",
            "",
            "- Target data was evaluated only after source checkpoints, thresholds, and representative layers were frozen.",
            "- The full OOD targets preserve the deliberately selected near-balanced candidate population; AUPRC/precision are conditional on that population.",
            "- Linear accessibility is predictive evidence, not proof of causal self-awareness.",
            "- The native pre-language-decoder control is learned and multimodal, not raw input.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(config_path: Path) -> None:
    contract, output_root = load_contract(config_path)
    decision_path = output_root / "source_decisions.json"
    source_decisions = read_json(decision_path)
    if source_decisions.get("contract_sha256") != contract["contract_sha256"]:
        raise ValueError("source decision contract differs")
    results = []
    for rank in range(4):
        worker_path = output_root / "ood_workers" / f"rank{rank:02d}.jsonl"
        complete = read_json(output_root / "ood_workers" / f"rank{rank:02d}.complete.json")
        if (
            complete.get("passed") is not True
            or complete.get("contract_sha256") != contract["contract_sha256"]
            or complete.get("source_decisions_sha256") != file_sha256(decision_path)
            or complete.get("result_sha256") != file_sha256(worker_path)
        ):
            raise RuntimeError(f"OOD evaluation worker {rank} is incomplete or incompatible")
        results.extend(read_jsonl(worker_path))
    observed = Counter((row["run"], row["representation"], row["layer"]) for row in results)
    expected_tasks = {
        (run, "input", None) for run in contract["runs"]
    } | {
        (run, "hidden", layer) for run in contract["runs"] for layer in range(28)
    }
    if set(observed) != expected_tasks or any(observed[task] != 1 for task in expected_tasks):
        raise RuntimeError("OOD evaluation does not cover all 87 tasks exactly once")
    by_run = {
        run: {
            (row["representation"], row["layer"]): row
            for row in results
            if row["run"] == run
        }
        for run in contract["runs"]
    }
    decisions = source_decisions["runs"]
    input_comparison_rows = []
    in_domain_rows = []
    for run in contract["runs"]:
        run_root = output_root / run
        representative = int(decisions[run]["representative_layer"])
        layer_rows = []
        selective_rows = []
        for layer in range(28):
            row = by_run[run][("hidden", layer)]
            layer_rows.append(
                {
                    "layer": layer,
                    "source_validation_auroc": row["source_validation"]["auroc"],
                    "source_validation_auprc": row["source_validation"]["auprc"],
                    "ood_records": row["target_full"]["records"],
                    "ood_correct": row["target_full"]["correct"],
                    "ood_wrong": row["target_full"]["wrong"],
                    "ood_auroc": row["target_full"]["auroc"],
                    "ood_auprc": row["target_full"]["auprc"],
                    "ood_balanced_accuracy": row["target_full"]["balanced_accuracy"],
                    "source_selected_representative": layer == representative,
                }
            )
            for target in contract["evaluation"]["preservation_targets"]:
                key = f"{float(target):.2f}"
                point = row["target_full_selective"][key]
                selective_rows.append(
                    {
                        "layer": layer,
                        "source_preservation_target": target,
                        "source_validation_threshold": row["source_validation_selective"][key]["threshold"],
                        "source_validation_actual_preservation": row["source_validation_selective"][key]["correct_preservation"],
                        "ood_actual_correct_preservation": point["correct_preservation"],
                        "ood_wrong_recall": point["wrong_recall"],
                        "ood_failure_precision": point["failure_precision"],
                        "ood_false_intervention_rate": point["false_deviation_rate"],
                        "source_selected_representative": layer == representative,
                    }
                )
            reference = row["in_domain_reference_same_uids"]
            ood_subset = row["target_phase48_test"]
            ood_selective = row["target_phase48_test_selective"]["0.99"]
            in_domain_rows.append(
                {
                    "dataset": contract["runs"][run]["target_dataset"],
                    "run": run,
                    "layer": layer,
                    "same_uid_records": ood_subset["records"],
                    "in_domain_auroc": reference["metrics"]["auroc"],
                    "ood_auroc": ood_subset["auroc"],
                    "auroc_drop": float(reference["metrics"]["auroc"]) - float(ood_subset["auroc"]),
                    "in_domain_validation_threshold_99": reference["validation_threshold_99"],
                    "ood_source_validation_threshold_99": row["source_validation_selective"]["0.99"]["threshold"],
                    "in_domain_wrong_recall_99": reference["selective_99"]["wrong_recall"],
                    "ood_wrong_recall_99": ood_selective["wrong_recall"],
                    "wrong_recall_drop_99": float(reference["selective_99"]["wrong_recall"]) - float(ood_selective["wrong_recall"]),
                    "source_selected_representative": layer == representative,
                }
            )
        atomic_csv(run_root / "layerwise_metrics.csv", layer_rows)
        atomic_csv(run_root / "selective_metrics.csv", selective_rows)
        input_row = by_run[run][("input", None)]
        write_once_or_verify(
            run_root / "summary.md",
            _run_summary_markdown(contract, run, decisions[run], [by_run[run][("hidden", layer)] for layer in range(28)], input_row).encode(),
        )
        for representation in ("input", 0, 14, 21, 27):
            row = input_row if representation == "input" else by_run[run][("hidden", representation)]
            input_auroc = float(input_row["target_full"]["auroc"])
            input_comparison_rows.append(
                {
                    "run": run,
                    "target_dataset": contract["runs"][run]["target_dataset"],
                    "representation": "native_pre_language_decoder" if representation == "input" else f"layer_{representation}",
                    "layer": "" if representation == "input" else representation,
                    "ood_auroc": row["target_full"]["auroc"],
                    "ood_auprc": row["target_full"]["auprc"],
                    "ood_balanced_accuracy": row["target_full"]["balanced_accuracy"],
                    "delta_auroc_over_input": 0.0 if representation == "input" else float(row["target_full"]["auroc"]) - input_auroc,
                }
            )
    atomic_csv(output_root / "input_only_vs_hidden.csv", input_comparison_rows)
    atomic_csv(output_root / "in_domain_vs_ood.csv", in_domain_rows)
    _plot(output_root, contract, by_run, input_comparison_rows, in_domain_rows, decisions)
    decision_summary = _decision_summary(contract, by_run, decisions, in_domain_rows)
    write_once_or_verify(output_root / "decision_summary.md", decision_summary.encode())
    required = [
        "protocol.md",
        "leave_textvqa_out/layerwise_metrics.csv",
        "leave_textvqa_out/selective_metrics.csv",
        "leave_textvqa_out/summary.md",
        "leave_chartqa_out/layerwise_metrics.csv",
        "leave_chartqa_out/selective_metrics.csv",
        "leave_chartqa_out/summary.md",
        "leave_gqa_out/layerwise_metrics.csv",
        "leave_gqa_out/selective_metrics.csv",
        "leave_gqa_out/summary.md",
        "input_only_vs_hidden.csv",
        "in_domain_vs_ood.csv",
        "figures/ood_layerwise_auroc.png",
        "figures/ood_layerwise_auprc.png",
        "figures/input_vs_hidden_ood.png",
        "figures/in_domain_vs_ood_drop.png",
        "figures/ood_wrong_recall_at_fixed_preservation.png",
        "decision_summary.md",
    ]
    missing = [path for path in required if not (output_root / path).is_file()]
    if missing:
        raise RuntimeError(f"required OOD artifacts are missing: {missing}")
    manifest = {
        "schema_version": "ood_failure_signal_artifact_manifest_v1",
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        "source_decisions_sha256": file_sha256(decision_path),
        "target_evaluated_after_source_freeze": True,
        "complete_probe_tasks": len(results),
        "required_files": {path: file_sha256(output_root / path) for path in required},
    }
    atomic_json(output_root / "artifact_manifest.json", manifest)
    print(json.dumps({"passed": True, "contract_sha256": contract["contract_sha256"], "tasks": len(results), "decision_summary": str(output_root / "decision_summary.md")}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    for name in ("smoke-worker", "extract-worker", "train-worker", "evaluate-worker"):
        worker = subparsers.add_parser(name)
        worker.add_argument("--rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
        worker.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    subparsers.add_parser("aggregate-smoke")
    subparsers.add_parser("aggregate-extraction")
    subparsers.add_parser("aggregate-source")
    subparsers.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_relative_to(PROJECT_ROOT):
        raise ValueError("config must lie inside the project root")
    if args.command == "prepare":
        prepare(config_path)
    elif args.command == "smoke-worker":
        smoke_worker(config_path, rank=args.rank, world_size=args.world_size)
    elif args.command == "aggregate-smoke":
        aggregate_smoke(config_path)
    elif args.command == "extract-worker":
        extract_worker(config_path, rank=args.rank, world_size=args.world_size)
    elif args.command == "aggregate-extraction":
        aggregate_extraction(config_path)
    elif args.command == "train-worker":
        train_worker(config_path, rank=args.rank, world_size=args.world_size)
    elif args.command == "aggregate-source":
        aggregate_source(config_path)
    elif args.command == "evaluate-worker":
        evaluate_worker(config_path, rank=args.rank, world_size=args.world_size)
    elif args.command == "finalize":
        finalize(config_path)
    else:
        raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
