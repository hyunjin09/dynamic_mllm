"""Fail-closed provenance collection and verification for dense regeneration."""

from __future__ import annotations

from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Mapping, Sequence

import torch


BOUND_SOURCE_PATHS = (
    "configs/current_dense_regeneration_v1.json",
    "dense_failure_stage1/contract.py",
    "dense_failure_stage1/runtime.py",
    "tools/research_analysis/dense_failure_stage1.py",
    "experiments/prepare_current_dense_regeneration.py",
    "experiments/run_current_dense_regeneration.py",
    "experiments/audit_current_dense_smoke.py",
    "experiments/finalize_current_dense_regeneration.py",
    "reference/dvr_qwen/eval_metrics.py",
)

PROCESSOR_TOKENIZER_FILES = (
    "chat_template.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def _command_output(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"contract command failed: {command}: {result.stderr.strip()}")
    return result.stdout.strip()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def model_file_hashes(model_root: Path) -> dict[str, str]:
    names = {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        *PROCESSOR_TOKENIZER_FILES,
    }
    index_path = model_root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError(f"model weight index is missing: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    names.update(index.get("weight_map", {}).values())
    missing = sorted(name for name in names if not (model_root / name).is_file())
    if missing:
        raise ValueError(f"frozen model files are missing: {missing}")
    return {name: file_sha256(model_root / name) for name in sorted(names)}


def backend_observation() -> dict:
    cuda = torch.backends.cuda
    matmul = cuda.matmul
    return {
        "flash_sdp_enabled": bool(cuda.flash_sdp_enabled()),
        "memory_efficient_sdp_enabled": bool(cuda.mem_efficient_sdp_enabled()),
        "math_sdp_enabled": bool(cuda.math_sdp_enabled()),
        "cudnn_sdp_enabled": bool(cuda.cudnn_sdp_enabled()),
        "matmul_allow_tf32": bool(matmul.allow_tf32),
        "matmul_allow_fp16_reduced_precision_reduction": bool(
            matmul.allow_fp16_reduced_precision_reduction
        ),
        "matmul_allow_bf16_reduced_precision_reduction": bool(
            matmul.allow_bf16_reduced_precision_reduction
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": __import__("os").environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def feature_schema(config: Mapping) -> dict:
    return {
        "schema_version": "current_dense_features_v1",
        "layer_ids": list(range(int(config["feature_layers"]))),
        "hidden_size": int(config["feature_hidden_size"]),
        "storage_dtype": "torch.bfloat16",
        "features": {
            "text_final": "post-layer state at the final literal user prompt token",
            "text_mean": "mean post-layer state over literal user prompt text tokens",
            "visual_mean": "mean post-layer state over expanded image-pad tokens",
        },
        "readout_confidence": None,
        "readout_confidence_reason": "no native calibrated intermediate-layer LM readout is defined",
        "prohibited_inputs": [
            "dataset ID",
            "ground-truth answer",
            "W->C membership",
            "route information",
        ],
    }


def collect_contract_integrity(
    *,
    project: Path,
    model_path: Path,
    revision: str,
    processor_revision: str,
    candidate_manifest: Path,
    smoke_manifest: Path,
    config_path: Path,
    config: Mapping,
    bound_source_paths: Sequence[str] = BOUND_SOURCE_PATHS,
) -> dict:
    """Collect every mutable input that must equal the frozen contract."""

    project = project.resolve()
    supplied_model_path = model_path.absolute()
    model_root = model_path.resolve()
    model_hashes = model_file_hashes(model_root)
    git_status = _command_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=project
    )
    gpu_inventory = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "model": {
            "revision": revision,
            "supplied_weight_path": str(supplied_model_path),
            "resolved_weight_path": str(model_root),
            "file_sha256": model_hashes,
        },
        "processor_tokenizer": {
            "revision": processor_revision,
            "use_fast": False,
            "file_sha256": {
                name: model_hashes[name]
                for name in PROCESSOR_TOKENIZER_FILES
            },
        },
        "git": {
            "commit": _command_output(["git", "rev-parse", "HEAD"], cwd=project),
            "branch": _command_output(["git", "branch", "--show-current"], cwd=project),
            "status_porcelain": git_status.splitlines() if git_status else [],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "accelerate": _package_version("accelerate"),
            "pillow": _package_version("Pillow"),
            "cuda_runtime": torch.version.cuda,
            "cudnn": str(torch.backends.cudnn.version()),
            "gpu_inventory": gpu_inventory.splitlines(),
            "cpu_inventory": _command_output(["lscpu"]).splitlines(),
        },
        "backend_settings": backend_observation(),
        "candidate_manifest_path": str(candidate_manifest.absolute()),
        "candidate_manifest_sha256": file_sha256(candidate_manifest),
        "smoke_manifest_path": str(smoke_manifest.absolute()),
        "smoke_manifest_sha256": file_sha256(smoke_manifest),
        "config_path": str(config_path.absolute()),
        "config_sha256": file_sha256(config_path),
        "generation": dict(config["generation"]),
        "evaluators": dict(config["evaluators"]),
        "feature_schema_sha256": canonical_json_sha256(feature_schema(config)),
        "bound_source_sha256": {
            path: file_sha256(project / path)
            for path in bound_source_paths
        },
    }


def _differences(expected, actual, path: str = "integrity") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = []
        for key in sorted(set(expected) | set(actual)):
            next_path = f"{path}.{key}"
            if key not in expected:
                differences.append(f"{next_path}: unexpected actual value")
            elif key not in actual:
                differences.append(f"{next_path}: missing actual value")
            else:
                differences.extend(_differences(expected[key], actual[key], next_path))
        return differences
    if expected != actual:
        return [f"{path}: expected={expected!r} actual={actual!r}"]
    return []


def assert_contract_integrity(contract: Mapping, actual_integrity: Mapping) -> None:
    expected = contract.get("integrity")
    if not isinstance(expected, dict):
        raise ValueError("frozen contract has no integrity payload")
    differences = _differences(expected, dict(actual_integrity))
    if differences:
        raise ValueError("frozen contract integrity mismatch:\n" + "\n".join(differences[:50]))


def validate_smoke_gate(
    gate: Mapping,
    *,
    contract_id: str,
    candidate_manifest_sha256: str,
    require_features: bool,
) -> None:
    if gate.get("schema_version") != "current_dense_smoke_gate_v1":
        raise ValueError("unsupported smoke gate schema")
    if gate.get("contract_id") != contract_id:
        raise ValueError("smoke gate contract differs")
    if gate.get("candidate_manifest_sha256") != candidate_manifest_sha256:
        raise ValueError("smoke gate candidate manifest differs")
    if gate.get("records") != 24 or gate.get("missing_uids") != 0 or gate.get("duplicate_uids") != 0:
        raise ValueError("smoke gate is not globally complete for 24 UIDs")
    required = (
        "repeatability_pass",
        "evaluator_parity_pass",
        "image_sha_pass",
        "contract_integrity_pass",
        "global_completeness_pass",
        "resume_validation_pass",
    )
    failed = [name for name in required if gate.get(name) is not True]
    if failed:
        raise ValueError(f"smoke gate failed required checks: {failed}")
    if require_features and gate.get("hook_token_parity_pass") is not True:
        raise ValueError("smoke gate hook token parity did not pass")
    if require_features and gate.get("feature_provenance_pass") is not True:
        raise ValueError("smoke gate feature provenance did not pass")
