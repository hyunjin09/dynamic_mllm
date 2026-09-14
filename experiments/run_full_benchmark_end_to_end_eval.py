#!/usr/bin/env python3
"""Run the frozen Stage-1 P90 -> Stage-2 A full external paired evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from binary_policy.executor.four_action import (  # noqa: E402
    capture_online_four_action_route,
    greedy_generate_from_cached_prompt,
)
from binary_policy.executor.inputs import build_binary_inputs  # noqa: E402
from dense_failure_stage1.contract import backend_observation, model_file_hashes  # noqa: E402
from dense_failure_stage1.runtime import DenseFeatureCollector, configure_dense_determinism  # noqa: E402
from dense_failure_stage1.shared_global_gate import SharedFailurePredictor  # noqa: E402
from dense_failure_stage2.full_benchmark_eval import (  # noqa: E402
    ACTION_NAMES,
    FAMILY_ORDER,
    TASK_COUNTS,
    benchmark_family,
    feature_positions_from_masks,
    method_decision,
    paired_bootstrap,
    recommendation_direction,
    stage1_summary,
    stage2_summary,
    summarize_pairs,
    transition,
    validate_complete_rows,
    validate_population,
)
from experiments.run_stage2_v1_training_revised import _load_model, _router  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs/full_benchmark_end_to_end_eval_v1.json"
BOUND_CODE = (
    "configs/full_benchmark_end_to_end_eval_v1.json",
    "dense_failure_stage2/full_benchmark_eval.py",
    "experiments/run_full_benchmark_end_to_end_eval.py",
    "dense_failure_stage1/shared_global_gate.py",
    "dense_failure_stage1/runtime.py",
    "dense_failure_stage2/v1_router.py",
    "experiments/run_stage2_v1_training_revised.py",
    "binary_policy/executor/four_action.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/model.py",
    "eval/reference/shared_prefix_eval_20260812/EVAL_PROTOCOL.md",
    "eval/reference/shared_prefix_eval_20260812/README.md",
    "eval/reference/shared_prefix_eval_20260812/REFERENCE_RESULT.md",
)
TASK_METRICS = {
    "chartqa": "relaxed_accuracy",
    "textvqa": "textvqa_evalai_consensus",
    "mmmu_pro_standard_test": "mmmu_acc",
    "mmmu_pro_vision_test": "mmmu_acc",
    "pope_adversarial": "pope_yes_no_accuracy",
    "pope_popular": "pope_yes_no_accuracy",
    "pope_random": "pope_yes_no_accuracy",
}
REFERENCE_DENSE_CORRECT_ACCURACY = {
    "chartqa": 0.8592,
    "textvqa": 0.8574,
    "mmmu_pro_standard_test": 0.3671,
    "mmmu_pro_vision_test": 0.3382,
    "pope_adversarial": 0.8703,
    "pope_popular": 0.8787,
    "pope_random": 0.8903,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    roots = (PROJECT_ROOT.resolve(), Path("/mnt/hyemin").resolve())
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise ValueError(f"path escapes the allowed roots: {value}")
    return resolved


def command_output(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(f"command failed {command}: {result.stderr.strip()}")
    return result.stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{number}")
            rows.append(row)
    return rows


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    _atomic_bytes(path, payload.encode())


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode())


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "full_benchmark_end_to_end_eval_config_v1":
        raise ValueError("full-benchmark config schema differs")
    if int(config["world_size"]) != 4 or int(config["datasets"]["expected_total"]) != 19960:
        raise ValueError("the frozen run requires four workers and 19,960 rows")
    if config["datasets"]["expected_counts"] != TASK_COUNTS:
        raise ValueError("configured task counts differ from the frozen scope")
    if float(config["stage1"]["threshold"]) != 0.9061332901863008:
        raise ValueError("Stage-1 P90 threshold differs")
    if config["stage1"]["comparison"] != "strict_greater_than":
        raise ValueError("Stage-1 comparison must remain strict greater-than")
    if tuple(config["stage2"]["actions"]) != ACTION_NAMES:
        raise ValueError("Stage-2 action order differs")
    if (
        config["stage2"]["parent_contract_sha256"]
        != "b17a81d4749b842fd843a344d3799bffdfc0f41e2c516a835242b656324440f8"
        or config["stage2"]["checkpoint_sha256"]
        != "48536bfbf6ebf62071898aea91952b6fe6a3f13b5723b09f6d814eb0ad9bbc7f"
    ):
        raise ValueError("Stage-2 Experiment A identity differs")
    generation = config["generation"]
    if (
        generation["do_sample"] is not False
        or int(generation["num_beams"]) != 1
        or generation["use_cache"] is not True
        or list(generation["eos_token_ids"]) != [151645]
        or float(generation["repetition_penalty"]) != 1.05
        or generation["max_new_tokens"] != {
            "chartqa": 16,
            "textvqa": 16,
            "mmmu_pro_standard_test": 16,
            "mmmu_pro_vision_test": 16,
            "pope_adversarial": 128,
            "pope_popular": 128,
            "pope_random": 128,
        }
    ):
        raise ValueError("generation policy differs from the external reference contract")


def _manifest_image_paths(row: Mapping[str, Any], source_root: Path) -> list[Path]:
    relatives = row.get("image_relpaths") or [row.get("image_relpath")]
    paths = [(source_root / str(value)).resolve() for value in relatives if value]
    if not paths:
        raise ValueError(f"row has no image path: {row.get('uid')}")
    if any(not path.is_relative_to(source_root.resolve()) for path in paths):
        raise ValueError(f"image path escapes its frozen data root: {row.get('uid')}")
    return paths


def _declared_image_hashes(row: Mapping[str, Any]) -> list[str]:
    values = row.get("image_content_sha256s") or [row.get("image_content_sha256")]
    hashes = [str(value) for value in values if value]
    if not hashes or any(len(value) != 64 for value in hashes):
        raise ValueError(f"row has invalid image hashes: {row.get('uid')}")
    return hashes


def _verify_images(row: Mapping[str, Any]) -> list[str]:
    paths = [resolve_path(value) for value in row["local_image_paths"]]
    expected = [str(value) for value in row["image_content_sha256s"]]
    if len(paths) != len(expected):
        raise RuntimeError(f"image path/hash arity differs for {row['uid']}")
    observed = []
    for path, digest in zip(paths, expected):
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"evaluation image is missing or empty: {row['uid']}: {path}")
        actual = file_sha256(path)
        if actual != digest:
            raise RuntimeError(
                f"image SHA-256 differs immediately before inference: {row['uid']}: "
                f"actual={actual} expected={digest}"
            )
        observed.append(actual)
    return observed


def _load_source_population(config: Mapping[str, Any], *, verify_images: bool) -> list[dict[str, Any]]:
    selected = []
    thresholds = config["evaluation"]["correctness_thresholds"]
    max_tokens = config["generation"]["max_new_tokens"]
    for manifest_value in config["datasets"]["manifests"]:
        manifest_path = resolve_path(manifest_value)
        source_root = manifest_path.parent
        manifest_sha256 = file_sha256(manifest_path)
        for row in read_jsonl(manifest_path):
            benchmark = str(row.get("benchmark", "")).lower()
            if benchmark not in TASK_COUNTS:
                continue
            uid = str(row["uid"])
            if str(row.get("metric_name")) != TASK_METRICS[benchmark]:
                raise ValueError(f"metric differs for {uid}")
            if float(row.get("correctness_threshold")) != float(thresholds[benchmark]):
                raise ValueError(f"correctness threshold differs for {uid}")
            if int(row.get("max_new_tokens")) != int(max_tokens[benchmark]):
                raise ValueError(f"generation limit differs for {uid}")
            paths = _manifest_image_paths(row, source_root)
            hashes = _declared_image_hashes(row)
            if len(paths) != len(hashes) or int(row.get("image_count", len(paths))) != len(paths):
                raise ValueError(f"image arity differs for {uid}")
            current = dict(row)
            current.update(
                {
                    "schema_version": "full_benchmark_eval_manifest_row_v1",
                    "dataset": benchmark,
                    "benchmark": benchmark,
                    "benchmark_family": benchmark_family(benchmark),
                    "data_root": str(source_root),
                    "source_manifest_path": str(manifest_path),
                    "source_manifest_sha256": manifest_sha256,
                    "local_image_paths": [str(path) for path in paths],
                    "image_paths": [str(path) for path in paths],
                    "image_path": str(paths[0]),
                    "image_content_sha256s": hashes,
                    "image_content_sha256": hashes[0],
                    "image_group_id": "sha256:" + "|".join(hashes),
                }
            )
            if verify_images:
                _verify_images(current)
            selected.append(current)
    validate_population(selected)
    return sorted(selected, key=lambda row: (row["benchmark"], row["uid"]))


def _verify_candidate_artifacts(config: Mapping[str, Any]) -> dict[str, str]:
    stage1_manifest_path = resolve_path(config["stage1"]["head_manifest"])
    stage1_manifest = read_json(stage1_manifest_path)
    if len(stage1_manifest.get("checkpoints", [])) != 5:
        raise RuntimeError("Stage-1 head manifest does not contain five checkpoints")
    artifacts = {
        "stage1_head_manifest": stage1_manifest_path,
        "stage1_normalization": resolve_path(config["stage1"]["normalization"]),
        "stage2_parent_protocol": resolve_path(config["stage2"]["parent_protocol"]),
        "stage2_selected_checkpoint": resolve_path(config["stage2"]["selected_checkpoint"]),
        "stage2_checkpoint": resolve_path(config["stage2"]["checkpoint"]),
        "evaluator": resolve_path(config["evaluation"]["scorer"]),
        "input_builder": resolve_path(config["evaluation"]["input_builder"]),
    }
    for index, row in enumerate(stage1_manifest["checkpoints"]):
        path = resolve_path(row["path"])
        if file_sha256(path) != str(row["sha256"]):
            raise RuntimeError(f"Stage-1 checkpoint {index} hash differs")
        artifacts[f"stage1_checkpoint_{index}"] = path
    if file_sha256(artifacts["stage1_normalization"]) != str(stage1_manifest["normalization"]["sha256"]):
        raise RuntimeError("Stage-1 normalization hash differs")
    parent = read_json(artifacts["stage2_parent_protocol"])
    if parent.get("contract_sha256") != config["stage2"]["parent_contract_sha256"]:
        raise RuntimeError("Stage-2 parent contract differs")
    selected = read_json(artifacts["stage2_selected_checkpoint"])
    if (
        selected.get("contract_sha256") != config["stage2"]["parent_contract_sha256"]
        or selected.get("experiment") != "A"
        or selected.get("sha256") != config["stage2"]["checkpoint_sha256"]
        or file_sha256(artifacts["stage2_checkpoint"]) != selected.get("sha256")
    ):
        raise RuntimeError("Stage-2 selected checkpoint identity differs")
    for relative, expected in parent["bound_code_sha256"].items():
        if file_sha256(resolve_path(relative)) != expected:
            raise RuntimeError(f"frozen Stage-2 executor code differs: {relative}")
    return {name: file_sha256(path) for name, path in artifacts.items()}


def prepare(config_path: Path) -> None:
    config = read_json(config_path)
    validate_config(config)
    output_root = resolve_path(config["output_root"])
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation root: {output_root}")
    rows = _load_source_population(config, verify_images=True)
    for family in FAMILY_ORDER:
        family_rows = [row for row in rows if row["benchmark_family"] == family]
        atomic_jsonl(output_root / f"manifests/{family}_full_manifest.jsonl", family_rows)
    atomic_jsonl(output_root / "manifests/all_full_manifest.jsonl", rows)
    artifact_sha = _verify_candidate_artifacts(config)
    model_root = resolve_path(config["model"]["snapshot_path"])
    model_sha = model_file_hashes(model_root)
    source_sha = {
        str(resolve_path(path)): file_sha256(resolve_path(path))
        for path in config["datasets"]["manifests"]
    }
    prepared_sha = {
        str(path.relative_to(output_root)): file_sha256(path)
        for path in sorted((output_root / "manifests").glob("*.jsonl"))
    }
    contract: dict[str, Any] = {
        "schema_version": "full_benchmark_end_to_end_eval_contract_v1",
        "created_at": utc_now(),
        "static_config": config,
        "config_sha256": file_sha256(config_path),
        "git": {
            "commit": command_output(("git", "rev-parse", "HEAD")),
            "branch": command_output(("git", "branch", "--show-current")),
            "worktree_status_at_freeze": command_output(
                ("git", "status", "--porcelain=v1", "--untracked-files=all")
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "lmms_eval": _package_version("lmms-eval"),
            "qwen_vl_utils": _package_version("qwen-vl-utils"),
            "cuda_runtime": torch.version.cuda,
            "nvidia_driver": command_output(
                ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
            ).splitlines()[0],
            "gpu_inventory": command_output(
                (
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader,nounits",
                )
            ).splitlines(),
            "scheduler": "direct_execution_no_slurm",
        },
        "source_manifest_sha256": source_sha,
        "prepared_manifest_sha256": prepared_sha,
        "candidate_artifact_sha256": artifact_sha,
        "model_snapshot_sha256": model_sha,
        "bound_code_sha256": {
            relative: file_sha256(resolve_path(relative)) for relative in BOUND_CODE
        },
        "population": {
            "total": len(rows),
            "task_counts": dict(Counter(row["benchmark"] for row in rows)),
            "family_counts": dict(Counter(row["benchmark_family"] for row in rows)),
            "unique_uids": len({row["uid"] for row in rows}),
            "unique_image_groups": len({row["image_group_id"] for row in rows}),
        },
        "selection_frozen_before_external_results": True,
    }
    contract["contract_sha256"] = canonical_hash(contract)
    atomic_json(output_root / "frozen_protocol.json", contract)
    protocol = f"""# Full-Benchmark End-to-End Evaluation Protocol

- Contract SHA-256: `{contract['contract_sha256']}`
- Git commit: `{contract['git']['commit']}` on `{contract['git']['branch']}`; the complete dirty-worktree snapshot is recorded in `frozen_protocol.json`.
- Model: `{config['model']['name']}` revision `{config['model']['revision']}`, BF16, SDPA, slow processor, local files only.
- Dense control: exact reference-native `model.generate` with all 28 layers `FULL`; passive Stage-1 feature hooks must have exact token/scorer parity with hook-free native generation.
- Semantic no-op rule: a no-trigger route or a triggered route selecting 28 `FULL` actions returns the exact native dense result; the four-action executor is used for generation only when at least one frozen-policy action is non-FULL.
- Routed candidate: five-checkpoint ALL-source Shared Random-4 Stage-1 probability mean, first strict `score > {float(config['stage1']['threshold']):.16g}` crossing, then frozen Phase-66 Experiment A Stage-2 actions `FULL/READ_ONLY/WRITE_ONLY/IGNORE` on the actual routed state.
- Generation: greedy custom argmax, repetition penalty 1.05 in FP32, EOS `[151645]`, row-specific 16 tokens except POPE 128.
- Correctness: frozen LMMS-compatible project scorers—ChartQA relaxed accuracy; TextVQA EvalAI consensus (correct at score >=0.5); MMMU-Pro first standalone A-J accuracy; POPE yes/no accuracy.
- Full paired population: 19,960 unique UIDs: ChartQA 2,500; TextVQA 5,000; MMMU-Pro Standard/Vision 1,730 each; POPE adversarial/popular/random 3,000 each.
- Excluded: GQA, DocVQA, MMStar, and base MMMU.
- Primary criterion: `W→C > C→W`; no threshold, checkpoint, or task selection may use these external results.
- ChartQA, TextVQA, MMMU-Pro, and POPE remain separate primary family metrics because their task scorers differ. Any pooled row is a descriptive 19,960-sample micro-average, not a benchmark macro-average or a common-score claim.
"""
    _atomic_bytes(output_root / "protocol.md", protocol.encode())
    print(json.dumps({"passed": True, "contract_sha256": contract["contract_sha256"], "rows": len(rows)}, sort_keys=True))


def verify_contract(config_path: Path, *, verify_model: bool = False) -> tuple[dict[str, Any], Path]:
    config = read_json(config_path)
    validate_config(config)
    output_root = resolve_path(config["output_root"])
    contract = read_json(output_root / "frozen_protocol.json")
    if canonical_hash(contract) != contract.get("contract_sha256"):
        raise RuntimeError("full-benchmark contract hash differs")
    if contract.get("static_config") != config or contract.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError("active full-benchmark config differs from the frozen contract")
    if command_output(("git", "rev-parse", "HEAD")) != contract["git"]["commit"]:
        raise RuntimeError("Git commit differs from the frozen contract")
    for relative, expected in contract["bound_code_sha256"].items():
        if file_sha256(resolve_path(relative)) != expected:
            raise RuntimeError(f"bound evaluation code differs: {relative}")
    if _verify_candidate_artifacts(config) != contract["candidate_artifact_sha256"]:
        raise RuntimeError("candidate artifact hashes differ")
    for path, expected in contract["source_manifest_sha256"].items():
        if file_sha256(resolve_path(path)) != expected:
            raise RuntimeError(f"source manifest hash differs: {path}")
    for relative, expected in contract["prepared_manifest_sha256"].items():
        if file_sha256(output_root / relative) != expected:
            raise RuntimeError(f"prepared manifest hash differs: {relative}")
    if verify_model and model_file_hashes(resolve_path(config["model"]["snapshot_path"])) != contract["model_snapshot_sha256"]:
        raise RuntimeError("model snapshot differs")
    return contract, output_root


class Runtime:
    def __init__(self, config: Mapping[str, Any], device_index: int):
        self.config = config
        self.device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(self.device)
        self.processor, self.base, self.wrapped = _load_model(config, self.device)
        loader_config = dict(config)
        loader_config["router"] = config["stage2"]
        self.stage2 = _router(loader_config, self.device).eval()
        stage2_checkpoint = torch.load(
            resolve_path(config["stage2"]["checkpoint"]), map_location="cpu", weights_only=False
        )
        if (
            stage2_checkpoint.get("contract_sha256") != config["stage2"]["parent_contract_sha256"]
            or stage2_checkpoint.get("experiment") != "A"
        ):
            raise RuntimeError("loaded Stage-2 checkpoint provenance differs")
        self.stage2.load_state_dict(stage2_checkpoint["state_dict"], strict=True)

        head_manifest = read_json(resolve_path(config["stage1"]["head_manifest"]))
        self.stage1 = []
        for checkpoint_row in head_manifest["checkpoints"]:
            model = SharedFailurePredictor(
                variant="state_layer_random4",
                input_size=10752,
                projection_size=256,
                layer_embedding_size=32,
                hidden_size=256,
            ).to(self.device, dtype=torch.float32).eval()
            checkpoint = torch.load(resolve_path(checkpoint_row["path"]), map_location="cpu", weights_only=True)
            if (
                checkpoint.get("schema_version") != "stage1_all_source_checkpoint_v1"
                or checkpoint.get("contract_sha256") != checkpoint_row["contract_sha256"]
            ):
                raise RuntimeError("loaded Stage-1 checkpoint provenance differs")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.stage1.append(model)
        normalization = torch.load(
            resolve_path(config["stage1"]["normalization"]), map_location="cpu", weights_only=True
        )
        if normalization.get("schema_version") != "shared_stage1_global_normalization_v1":
            raise RuntimeError("Stage-1 normalization schema differs")
        self.mean = normalization["mean"].float().to(self.device)
        self.std = normalization["std"].float().to(self.device)


def _reference_functions(config: Mapping[str, Any]):
    code_root = str(resolve_path(config["evaluation"]["reference_code_root"]))
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    from dvr_qwen.eval_metrics import score_prediction
    from dvr_qwen.scripts.cache_preference_gt_router_features import build_processor_inputs

    return score_prediction, build_processor_inputs


def _score(config: Mapping[str, Any], row: Mapping[str, Any], prediction: str) -> dict[str, Any]:
    score_prediction, _ = _reference_functions(config)
    raw = float(
        score_prediction(
            str(row["metric_name"]), prediction, str(row["answer"]), row.get("all_answer_norms")
        )
    )
    threshold = float(row["correctness_threshold"])
    return {"metric": str(row["metric_name"]), "score": raw, "threshold": threshold, "correct": raw >= threshold}


def _build_inputs(runtime: Runtime, row: Mapping[str, Any]) -> dict[str, Any]:
    _, build_processor_inputs = _reference_functions(runtime.config)
    inputs = build_processor_inputs(runtime.processor, dict(row), data_root=Path(row["data_root"]))
    return dict(inputs)


def _decode(runtime: Runtime, generated: torch.Tensor) -> tuple[list[int], str]:
    ids = generated.detach().cpu().view(-1).tolist()
    text = runtime.processor.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    return ids, text


def _generate(runtime: Runtime, output, input_ids: torch.Tensor, row: Mapping[str, Any]) -> dict[str, Any]:
    if output.cache is None:
        raise RuntimeError("paired generation requires a complete prompt cache")
    generated = greedy_generate_from_cached_prompt(
        runtime.wrapped,
        output.prompt_logits,
        output.inputs,
        output.cache,
        input_ids,
        max_new_tokens=int(row["max_new_tokens"]),
        eos_token_ids=list(runtime.config["generation"]["eos_token_ids"]),
        repetition_penalty=float(runtime.config["generation"]["repetition_penalty"]),
    ).generated_ids
    ids, text = _decode(runtime, generated)
    score = _score(runtime.config, row, text)
    return {"generated_token_ids": ids, "generated_answer": text, **score}


@torch.inference_mode()
def _native_dense_generation(
    runtime: Runtime,
    row: Mapping[str, Any],
    *,
    extract_features: bool,
) -> dict[str, Any]:
    """Run the authoritative reference-native dense path, optionally with passive hooks."""
    consumed_hashes = _verify_images(row)
    inputs = _build_inputs(runtime, row)
    prompt_length = int(inputs["input_ids"].shape[1])
    instruction_positions, visual_positions = feature_positions_from_masks(
        inputs["instruction_token_mask"][0].tolist(),
        inputs["mm_token_type_ids"][0].tolist(),
    )
    native_inputs = {
        key: value.to(runtime.device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
        if key != "instruction_token_mask"
    }
    runtime.base.model.rope_deltas = None
    collector = None
    if extract_features:
        collector = DenseFeatureCollector(
            runtime.base.model.language_model.layers,
            visual_positions=visual_positions,
            user_text_positions=instruction_positions,
            final_user_token_position=instruction_positions[-1],
        )
        context = collector
    else:
        context = nullcontext()
    with context:
        generated = runtime.base.generate(
            **native_inputs,
            max_new_tokens=int(row["max_new_tokens"]),
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=list(runtime.config["generation"]["eos_token_ids"]),
            repetition_penalty=float(runtime.config["generation"]["repetition_penalty"]),
        )
    ids, text = _decode(runtime, generated[:, prompt_length:])
    score = _score(runtime.config, row, text)
    features = None
    if collector is not None:
        packed = collector.stacked()
        features = torch.cat(
            [packed[name] for name in ("text_final", "text_mean", "visual_mean")], dim=-1
        ).to(runtime.device, dtype=torch.float32)
        if tuple(features.shape) != (28, 10752) or not bool(torch.isfinite(features).all()):
            raise RuntimeError("native Stage-1 feature trajectory is invalid")
    return {
        "uid": str(row["uid"]),
        "benchmark": str(row["benchmark"]),
        "generated_token_ids": ids,
        "generated_answer": text,
        "consumed_image_sha256s": consumed_hashes,
        "prompt_tokens": int(inputs["attention_mask"].sum().item()),
        "visual_tokens": len(visual_positions),
        "features": features,
        **score,
    }


@torch.inference_mode()
def _stage1_scores(runtime: Runtime, features: torch.Tensor) -> tuple[list[float], int | None]:
    normalized = (features - runtime.mean) / runtime.std
    layers = torch.arange(28, device=runtime.device, dtype=torch.long)
    probabilities = torch.stack(
        [torch.sigmoid(model(normalized, layers)) for model in runtime.stage1], dim=0
    ).mean(dim=0)
    if not bool(torch.isfinite(probabilities).all()):
        raise RuntimeError("Stage-1 ensemble produced non-finite scores")
    scores = probabilities.detach().cpu().tolist()
    threshold = float(runtime.config["stage1"]["threshold"])
    trigger = next((index for index, value in enumerate(scores) if float(value) > threshold), None)
    return [float(value) for value in scores], trigger


@torch.inference_mode()
def process_row(runtime: Runtime, row: Mapping[str, Any], *, contract_sha256: str) -> dict[str, Any]:
    started = time.monotonic()
    dense = _native_dense_generation(runtime, row, extract_features=True)
    features = dense.pop("features")
    scores, trigger = _stage1_scores(runtime, features)
    del features

    action_rows = []

    def selector(layer_index, text_states, visual_states, meta):
        if trigger is None or int(layer_index) < int(trigger):
            action_rows.append({"layer": int(layer_index), "action": "FULL", "active": False})
            return "FULL"
        logits = runtime.stage2(
            text_states.detach().clone(),
            visual_states.detach().clone(),
            text_mask=meta.text_valid_mask.detach().clone(),
            visual_mask=meta.visual_valid_mask.detach().clone(),
        )
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError(f"Stage-2 logits are non-finite for {row['uid']} at L{layer_index}")
        probabilities = logits.float().softmax(dim=-1)[0]
        index = int(probabilities.argmax().item())
        action = ACTION_NAMES[index]
        action_rows.append(
            {
                "layer": int(layer_index),
                "action": action,
                "active": True,
                "probabilities": {
                    name: float(probabilities[position].item())
                    for position, name in enumerate(ACTION_NAMES)
                },
            }
        )
        return action

    if trigger is None:
        actions = ["FULL"] * 28
        action_rows = [
            {"layer": layer, "action": "FULL", "active": False} for layer in range(28)
        ]
        routed = {
            key: dense[key]
            for key in ("generated_token_ids", "generated_answer", "metric", "score", "threshold", "correct")
        }
    else:
        routed_hashes = _verify_images(row)
        if routed_hashes != dense["consumed_image_sha256s"]:
            raise RuntimeError("image bytes changed between paired native and routed inference")
        inputs = _build_inputs(runtime, row)
        prepared = build_binary_inputs(runtime.wrapped, inputs)
        routed_output = capture_online_four_action_route(
            runtime.wrapped,
            {},
            selector,
            prepared_inputs=prepared,
            use_cache=True,
            native_full_rows=True,
        )
        actions = list(routed_output.layer_actions)
        if len(action_rows) != 28 or actions != [item["action"] for item in action_rows]:
            raise RuntimeError("Stage-2 action trace is incomplete")
        if all(action == "FULL" for action in actions):
            routed = {
                key: dense[key]
                for key in ("generated_token_ids", "generated_answer", "metric", "score", "threshold", "correct")
            }
        else:
            routed = _generate(runtime, routed_output, inputs["input_ids"], row)
        del routed_output, prepared, inputs
    active_actions = [item["action"] for item in action_rows if item["active"]]
    nonfull_layers = [item["layer"] for item in action_rows if item["active"] and item["action"] != "FULL"]
    dense_correct, routed_correct = bool(dense["correct"]), bool(routed["correct"])
    elapsed = time.monotonic() - started
    result = {
        "schema_version": "full_benchmark_eval_paired_row_v1",
        "contract_sha256": contract_sha256,
        "uid": str(row["uid"]),
        "sample_id": str(row["sample_id"]),
        "benchmark": str(row["benchmark"]),
        "benchmark_family": str(row["benchmark_family"]),
        "image_group_id": str(row["image_group_id"]),
        "image_content_sha256s": list(row["image_content_sha256s"]),
        "consumed_image_sha256s": dense["consumed_image_sha256s"],
        "metric_name": str(row["metric_name"]),
        "correctness_threshold": float(row["correctness_threshold"]),
        "answer": row["answer"],
        "all_answer_norms": row.get("all_answer_norms"),
        "dense_generated_answer": dense["generated_answer"],
        "dense_generated_token_ids": dense["generated_token_ids"],
        "dense_score": dense["score"],
        "dense_correct": dense_correct,
        "routed_generated_answer": routed["generated_answer"],
        "routed_generated_token_ids": routed["generated_token_ids"],
        "routed_score": routed["score"],
        "routed_correct": routed_correct,
        "transition": transition(dense_correct, routed_correct),
        "stage1_scores": scores,
        "stage1_max_score": max(scores),
        "stage1_threshold": float(runtime.config["stage1"]["threshold"]),
        "stage1_comparison": "strict_greater_than",
        "triggered": trigger is not None,
        "trigger_layer": trigger,
        "actions": actions,
        "action_rows": action_rows,
        "post_trigger_action_counts": dict(Counter(active_actions)),
        "post_trigger_actions": len(active_actions),
        "non_full_count": len(nonfull_layers),
        "any_non_full": bool(nonfull_layers),
        "first_non_full_layer": nonfull_layers[0] if nonfull_layers else None,
        "trigger_to_first_non_full_delay": nonfull_layers[0] - int(trigger) if nonfull_layers else None,
        "read_enabled_layers": sum(action in {"FULL", "READ_ONLY"} for action in actions),
        "write_enabled_layers": sum(action in {"FULL", "WRITE_ONLY"} for action in actions),
        "decoder_prefill_calls": sum(2 if action == "WRITE_ONLY" else 1 for action in actions),
        "prompt_tokens": int(dense["prompt_tokens"]),
        "visual_tokens": int(dense["visual_tokens"]),
        "elapsed_seconds": elapsed,
    }
    return result


def _load_prepared_rows(output_root: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(output_root / "manifests/all_full_manifest.jsonl")
    validate_population(rows)
    return rows


def smoke(config_path: Path, device_index: int) -> None:
    contract, output_root = verify_contract(config_path, verify_model=True)
    smoke_root = output_root / "smoke"
    if smoke_root.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {smoke_root}")
    config = contract["static_config"]
    configure_dense_determinism(int(config["seed"]), config["backend_settings"])
    runtime = Runtime(config, device_index)
    rows = _load_prepared_rows(output_root)
    selected = []
    for benchmark in TASK_COUNTS:
        candidates = [row for row in rows if row["benchmark"] == benchmark]
        if benchmark == "mmmu_pro_standard_test":
            tail_vision = [
                row
                for row in candidates
                if int(row.get("image_count", 1)) > str(row["prompt"]).count("<image")
            ]
            if not tail_vision:
                raise RuntimeError("smoke cannot find the required trailing-vision MMMU-Pro case")
            candidates = tail_vision
        candidates.sort(key=lambda row: sha256(f"{config['seed']}:smoke:{row['uid']}".encode()).hexdigest())
        selected.append(candidates[0])
    atomic_jsonl(smoke_root / "smoke_manifest.jsonl", selected)
    results = []
    repeats = []
    native_parity = []
    started = time.monotonic()
    for index, row in enumerate(selected):
        first = process_row(runtime, row, contract_sha256=contract["contract_sha256"])
        second = process_row(runtime, row, contract_sha256=contract["contract_sha256"])
        native = _native_dense_generation(runtime, row, extract_features=False)
        results.append(first)
        exact = {
            "uid": row["uid"],
            "dense_tokens_equal": first["dense_generated_token_ids"] == second["dense_generated_token_ids"],
            "routed_tokens_equal": first["routed_generated_token_ids"] == second["routed_generated_token_ids"],
            "dense_score_equal": first["dense_score"] == second["dense_score"],
            "routed_score_equal": first["routed_score"] == second["routed_score"],
            "stage1_scores_equal": first["stage1_scores"] == second["stage1_scores"],
            "trigger_equal": first["trigger_layer"] == second["trigger_layer"],
            "actions_equal": first["actions"] == second["actions"],
        }
        exact["passed"] = all(value for key, value in exact.items() if key not in {"uid", "passed"})
        repeats.append(exact)
        native_row = {
            "uid": row["uid"],
            "benchmark": row["benchmark"],
            "token_sequence_equal": first["dense_generated_token_ids"] == native["generated_token_ids"],
            "prediction_equal": first["dense_generated_answer"] == native["generated_answer"],
            "score_equal": first["dense_score"] == native["score"],
            "correctness_equal": first["dense_correct"] == native["correct"],
            "image_hashes_equal": native["consumed_image_sha256s"] == row["image_content_sha256s"],
            "hooked_generated_token_ids": first["dense_generated_token_ids"],
            "native_generated_token_ids": native["generated_token_ids"],
            "hooked_prediction": first["dense_generated_answer"],
            "native_prediction": native["generated_answer"],
        }
        native_row["passed"] = all(
            native_row[key]
            for key in (
                "token_sequence_equal",
                "prediction_equal",
                "score_equal",
                "correctness_equal",
                "image_hashes_equal",
            )
        )
        native_parity.append(native_row)
        print(json.dumps({"smoke_completed": index + 1, "total": len(selected), "uid": row["uid"], "passed": exact["passed"]}), flush=True)
    atomic_jsonl(smoke_root / "smoke_results.jsonl", results)
    atomic_jsonl(smoke_root / "repeat_parity.jsonl", repeats)
    atomic_jsonl(smoke_root / "native_dense_hook_parity.jsonl", native_parity)
    no_trigger = [row for row in results if not row["triggered"]]
    all_full = [row for row in results if all(action == "FULL" for action in row["actions"])]
    passed = (
        len(results) == 7
        and all(row["passed"] for row in repeats)
        and all(row["passed"] for row in native_parity)
        and bool(no_trigger)
        and all(row["dense_generated_token_ids"] == row["routed_generated_token_ids"] for row in no_trigger)
        and all(row["dense_generated_token_ids"] == row["routed_generated_token_ids"] for row in all_full)
        and all(row["consumed_image_sha256s"] == row["image_content_sha256s"] for row in results)
        and len({row["uid"] for row in results}) == 7
        and all(row["stage1_threshold"] == 0.9061332901863008 for row in results)
    )
    report = {
        "schema_version": "full_benchmark_eval_smoke_v1",
        "passed": passed,
        "contract_sha256": contract["contract_sha256"],
        "samples": len(results),
        "task_variants": [row["benchmark"] for row in results],
        "repeat_exact": sum(row["passed"] for row in repeats),
        "native_hook_free_exact": sum(row["passed"] for row in native_parity),
        "no_trigger_samples": len(no_trigger),
        "all_full_route_samples": len(all_full),
        "image_hash_checks": len(results),
        "backend_observation": backend_observation(),
        "elapsed_seconds": time.monotonic() - started,
        "completed_at": utc_now(),
    }
    atomic_json(smoke_root / "smoke_report.json", report)
    markdown = f"""# Full-Benchmark Parity Smoke

- Passed: **{passed}**
- Contract: `{contract['contract_sha256']}`
- Coverage: {len(results)}/7 task variants, unique UIDs: {len({row['uid'] for row in results})}
- Exact repeated paired execution: {report['repeat_exact']}/7
- Native reference dense with Stage-1 hooks vs hook-free exact token/scorer parity: {report['native_hook_free_exact']}/7
- No-trigger samples with exact dense/routed token parity: {len(no_trigger)}/{len(no_trigger)}
- All-FULL routed samples with exact dense/routed token parity: {len(all_full)}/{len(all_full)}
- Image SHA checks: {len(results)}/7
- Strict Stage-1 threshold: `score > 0.9061332901863008`
- Stage-2 checkpoint SHA-256: `{config['stage2']['checkpoint_sha256']}`
- Wall time: {report['elapsed_seconds']:.2f} seconds
"""
    _atomic_bytes(smoke_root / "smoke_report.md", markdown.encode())
    print(json.dumps(report, sort_keys=True))
    if not passed:
        raise RuntimeError("full-benchmark parity smoke failed")


def worker(config_path: Path, rank: int, resume: bool) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    smoke_report = read_json(output_root / "smoke/smoke_report.json")
    if not smoke_report.get("passed") or smoke_report.get("contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("full execution requires the exact passed smoke")
    world = int(config["world_size"])
    if rank not in range(world):
        raise ValueError("rank is outside the four-worker range")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each direct worker requires exactly one visible GPU")
    configure_dense_determinism(int(config["seed"]) + rank, config["backend_settings"])
    runtime = Runtime(config, 0)
    rows = _load_prepared_rows(output_root)
    assigned = [row for index, row in enumerate(rows) if index % world == rank]
    work_root = output_root / f"work/rank{rank:02d}"
    completion_path = work_root / "complete.json"
    if completion_path.exists():
        raise FileExistsError(f"rank {rank} is already complete")
    parts = sorted(work_root.glob("part_*.jsonl")) if work_root.exists() else []
    if parts and not resume:
        raise FileExistsError(f"rank {rank} has resumable parts; pass --resume")
    existing = [row for path in parts for row in read_jsonl(path)]
    if len(existing) != len({row["uid"] for row in existing}):
        raise RuntimeError(f"rank {rank} resume rows contain duplicates")
    expected = {row["uid"] for row in assigned}
    if any(row.get("contract_sha256") != contract["contract_sha256"] for row in existing):
        raise RuntimeError(f"rank {rank} resume rows belong to another contract")
    if not {row["uid"] for row in existing} <= expected:
        raise RuntimeError(f"rank {rank} resume rows contain unexpected UIDs")
    completed = {row["uid"] for row in existing}
    remaining = [row for row in assigned if row["uid"] not in completed]
    buffer = []
    part_index = len(parts)
    started = time.monotonic()
    for index, row in enumerate(remaining, 1):
        result = process_row(runtime, row, contract_sha256=contract["contract_sha256"])
        result["worker_rank"] = rank
        buffer.append(result)
        if len(buffer) >= 16:
            atomic_jsonl(work_root / f"part_{part_index:05d}.jsonl", buffer)
            part_index += 1
            buffer.clear()
        if index % 25 == 0:
            print(json.dumps({"rank": rank, "completed_this_run": index, "remaining_at_start": len(remaining), "total_assigned": len(assigned), "elapsed_seconds": time.monotonic() - started}), flush=True)
    if buffer:
        atomic_jsonl(work_root / f"part_{part_index:05d}.jsonl", buffer)
    final_rows = [row for path in sorted(work_root.glob("part_*.jsonl")) for row in read_jsonl(path)]
    validate_complete_rows(
        [row["uid"] for row in assigned], final_rows, contract_sha256=contract["contract_sha256"]
    )
    completion = {
        "schema_version": "full_benchmark_eval_worker_complete_v1",
        "passed": True,
        "contract_sha256": contract["contract_sha256"],
        "rank": rank,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "expected": len(assigned),
        "completed": len(final_rows),
        "resumed": len(existing),
        "elapsed_seconds_this_run": time.monotonic() - started,
        "sample_elapsed_seconds_sum": sum(float(row["elapsed_seconds"]) for row in final_rows),
        "completed_at": utc_now(),
    }
    atomic_json(completion_path, completion)
    print(json.dumps(completion, sort_keys=True))


def _scoped(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    output = {family: [row for row in rows if row["benchmark_family"] == family] for family in FAMILY_ORDER}
    output["overall"] = list(rows)
    return output


def _write_figures(output_root: Path, summaries: Mapping[str, Mapping[str, Any]], stage1_rows, stage2_rows) -> None:
    figure_root = output_root / "figures"
    names = list(FAMILY_ORDER) + ["overall"]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, [summaries[name]["dense_accuracy"] for name in names], width, label="Dense")
    ax.bar(x + width / 2, [summaries[name]["routed_accuracy"] for name in names], width, label="Routed")
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    figure_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_root / "dense_vs_routed_accuracy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#2a9d8f" if summaries[name]["net_correction"] >= 0 else "#e76f51" for name in names]
    ax.bar(x, [summaries[name]["net_correction"] for name in names], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylabel("W→C − C→W")
    fig.tight_layout()
    fig.savefig(figure_root / "net_correction_by_benchmark.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter([summaries[name]["c_to_w"] for name in names], [summaries[name]["w_to_c"] for name in names])
    for name in names:
        ax.annotate(name, (summaries[name]["c_to_w"], summaries[name]["w_to_c"]))
    limit = max([summaries[name]["c_to_w"] for name in names] + [summaries[name]["w_to_c"] for name in names] + [1])
    ax.plot([0, limit], [0, limit], linestyle="--", color="gray")
    ax.set_xlabel("C→W")
    ax.set_ylabel("W→C")
    fig.tight_layout()
    fig.savefig(figure_root / "rescue_vs_regression.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, [stage1_rows[name]["p_trigger_given_dense_c"] or 0 for name in names], width, label="P(trigger|Dense-C)")
    ax.bar(x + width / 2, [stage1_rows[name]["p_trigger_given_dense_w"] or 0 for name in names], width, label="P(trigger|Dense-W)")
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylabel("Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_root / "stage1_trigger_behavior.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x, [stage2_rows[name]["fraction_any_non_full"] or 0 for name in names])
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylabel("Triggered samples with any non-FULL")
    fig.tight_layout()
    fig.savefig(figure_root / "stage2_nonfull_behavior.png", dpi=180)
    plt.close(fig)


def aggregate(config_path: Path) -> None:
    contract, output_root = verify_contract(config_path, verify_model=False)
    config = contract["static_config"]
    expected = _load_prepared_rows(output_root)
    rows = []
    completions = []
    for rank in range(int(config["world_size"])):
        work_root = output_root / f"work/rank{rank:02d}"
        complete_path = work_root / "complete.json"
        if not complete_path.is_file():
            raise RuntimeError(f"rank {rank} completion is missing")
        completion = read_json(complete_path)
        rank_rows = [row for path in sorted(work_root.glob("part_*.jsonl")) for row in read_jsonl(path)]
        if (
            not completion.get("passed")
            or completion.get("contract_sha256") != contract["contract_sha256"]
            or len(rank_rows) != int(completion["expected"])
        ):
            raise RuntimeError(f"rank {rank} is partial or incompatible")
        rows.extend(rank_rows)
        completions.append(completion)
    validate_complete_rows(
        [row["uid"] for row in expected], rows, contract_sha256=contract["contract_sha256"]
    )
    rows.sort(key=lambda row: (row["benchmark"], row["uid"]))
    scopes = _scoped(rows)
    for family in FAMILY_ORDER:
        family_rows = scopes[family]
        atomic_jsonl(
            output_root / f"dense/{family}_results.jsonl",
            [
                {
                    "uid": row["uid"], "benchmark": row["benchmark"], "benchmark_family": family,
                    "generated_answer": row["dense_generated_answer"], "generated_token_ids": row["dense_generated_token_ids"],
                    "score": row["dense_score"], "correct": row["dense_correct"], "metric_name": row["metric_name"],
                }
                for row in family_rows
            ],
        )
        atomic_jsonl(
            output_root / f"routed/{family}_results.jsonl",
            [
                {
                    "uid": row["uid"], "benchmark": row["benchmark"], "benchmark_family": family,
                    "generated_answer": row["routed_generated_answer"], "generated_token_ids": row["routed_generated_token_ids"],
                    "score": row["routed_score"], "correct": row["routed_correct"], "triggered": row["triggered"],
                    "trigger_layer": row["trigger_layer"], "actions": row["actions"], "metric_name": row["metric_name"],
                }
                for row in family_rows
            ],
        )
        atomic_jsonl(output_root / f"paired/{family}_paired.jsonl", family_rows)
    atomic_jsonl(output_root / "paired/all_paired.jsonl", rows)

    summaries = {name: summarize_pairs(group) for name, group in scopes.items()}
    summary_rows = [{"benchmark_family": name, **summaries[name]} for name in (*FAMILY_ORDER, "overall")]
    atomic_csv(output_root / "metrics/benchmark_summary.csv", summary_rows)
    transition_rows = []
    for name in (*FAMILY_ORDER, "overall"):
        counts = Counter(row["transition"] for row in scopes[name])
        for outcome in ("C→C", "C→W", "W→C", "W→W"):
            transition_rows.append({"benchmark_family": name, "transition": outcome, "count": counts[outcome]})
    atomic_csv(output_root / "metrics/transition_counts.csv", transition_rows)

    bootstrap_rows = []
    for index, name in enumerate((*FAMILY_ORDER, "overall")):
        for item in paired_bootstrap(
            scopes[name], draws=int(config["bootstrap"]["draws"]), seed=int(config["bootstrap"]["seed"]) + index
        ):
            bootstrap_rows.append({"benchmark_family": name, **item})
    atomic_csv(output_root / "metrics/paired_bootstrap.csv", bootstrap_rows)

    stage1 = {name: stage1_summary(group) for name, group in scopes.items()}
    stage2 = {name: stage2_summary(group) for name, group in scopes.items()}
    atomic_csv(
        output_root / "metrics/stage1_admission.csv",
        [{"benchmark_family": name, **stage1[name]} for name in (*FAMILY_ORDER, "overall")],
    )
    atomic_csv(
        output_root / "metrics/stage2_action_behavior.csv",
        [{"benchmark_family": name, **stage2[name]} for name in (*FAMILY_ORDER, "overall")],
    )
    dataset_rows = []
    for benchmark in TASK_COUNTS:
        group = [row for row in rows if row["benchmark"] == benchmark]
        dataset_rows.append(
            {"benchmark": benchmark, "benchmark_family": benchmark_family(benchmark), **summarize_pairs(group), **{f"stage1_{key}": value for key, value in stage1_summary(group).items() if key not in {"n"}}, **{f"stage2_{key}": value for key, value in stage2_summary(group).items() if key != "triggered"}}
        )
    atomic_csv(output_root / "metrics/dataset_breakdown.csv", dataset_rows)
    reference_parity_rows = []
    for benchmark, reported in REFERENCE_DENSE_CORRECT_ACCURACY.items():
        observed = next(row for row in dataset_rows if row["benchmark"] == benchmark)
        accuracy = float(observed["dense_accuracy"])
        reference_parity_rows.append(
            {
                "benchmark": benchmark,
                "n": observed["n"],
                "current_dense_accuracy": accuracy,
                "reference_reported_dense_accuracy": reported,
                "delta": accuracy - reported,
                "matches_reference_reported_precision": round(accuracy, 4) == reported,
                "note": "reference accuracy is reported to four decimal places",
            }
        )
    atomic_csv(output_root / "metrics/reference_dense_parity.csv", reference_parity_rows)

    compute_rows = [
        {
            "scope": "overall",
            "workers": len(completions),
            "samples": len(rows),
            "sum_worker_wall_seconds": sum(float(row["elapsed_seconds_this_run"]) for row in completions),
            "gpu_hours": sum(float(row["elapsed_seconds_this_run"]) for row in completions) / 3600.0,
            "sum_sample_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
            "failures": 0,
            "retries": sum(int(row["resumed"]) for row in completions),
            "mean_prompt_tokens": float(np.mean([row["prompt_tokens"] for row in rows])),
            "mean_visual_tokens": float(np.mean([row["visual_tokens"] for row in rows])),
            "mean_read_enabled_layers_routed": float(np.mean([row["read_enabled_layers"] for row in rows])),
            "mean_write_enabled_layers_routed": float(np.mean([row["write_enabled_layers"] for row in rows])),
            "mean_prefill_decoder_calls_routed": float(np.mean([row["decoder_prefill_calls"] for row in rows])),
        }
    ]
    atomic_csv(output_root / "metrics/compute_summary.csv", compute_rows)
    _write_figures(output_root, summaries, stage1, stage2)

    decision = method_decision(summaries)
    decision_names = {
        "A": "Full-scale net-positive method signal",
        "B": "Narrow/task-specific positive signal",
        "C": "Neutral/under-active current method",
        "D": "Regression-dominated current method",
    }
    dominant_rescue = max(FAMILY_ORDER, key=lambda name: summaries[name]["w_to_c"])
    dominant_regression = max(FAMILY_ORDER, key=lambda name: summaries[name]["c_to_w"])
    table = [
        "| Benchmark family | N | Dense Acc | Routed Acc | ΔAcc | W→C | C→W | Net | C→C preservation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (*FAMILY_ORDER, "overall"):
        value = summaries[name]
        table.append(
            f"| {name} | {value['n']} | {value['dense_accuracy']:.6f} | {value['routed_accuracy']:.6f} | {value['delta_accuracy']:+.6f} | {value['w_to_c']} | {value['c_to_w']} | {value['net_correction']:+d} | {value['c_to_c_preservation']:.6f} |"
        )
    ci_overall = {row["metric"]: row for row in bootstrap_rows if row["benchmark_family"] == "overall"}
    full_summary = f"""# Full-Benchmark End-to-End Evaluation Summary

All four established benchmark families were evaluated under the frozen contract `{contract['contract_sha256']}`. Every one of the 19,960 UIDs has exactly one successful paired dense/routed record; no partial worker or duplicate UID was accepted.

The population, prompt builder, native dense generation contract, and task scorers are the frozen `shared_prefix_eval_20260812` reference method. Native dense generation with passive Stage-1 hooks versus hook-free native dense passed exact token/scorer parity on all seven preflight task variants. No-trigger and all-FULL policy paths are defined as the exact native dense result; the four-action executor is invoked only when the frozen policy selects at least one non-FULL action. Dataset-level dense agreement with the rounded original-server reference is recorded in `metrics/reference_dense_parity.csv`.

{chr(10).join(table)}

- Pooled ΔAccuracy 95% paired-bootstrap CI: [{ci_overall['delta_accuracy']['ci_low']:.6f}, {ci_overall['delta_accuracy']['ci_high']:.6f}].
- Pooled W→C rate 95% CI: [{ci_overall['w_to_c_rate']['ci_low']:.6f}, {ci_overall['w_to_c_rate']['ci_high']:.6f}].
- Pooled C→C preservation 95% CI: [{ci_overall['c_to_c_preservation']['ci_low']:.6f}, {ci_overall['c_to_c_preservation']['ci_high']:.6f}].
- Stage-1 trigger rate: {stage1['overall']['trigger_rate']:.6f}; P(trigger|Dense-C)={stage1['overall']['p_trigger_given_dense_c']:.6f}, P(trigger|Dense-W)={stage1['overall']['p_trigger_given_dense_w']:.6f}, trigger precision={stage1['overall']['trigger_precision']:.6f}.
- Among triggered samples, any non-FULL: {stage2['overall']['fraction_any_non_full']:.6f}; post-trigger FULL fraction: {stage2['overall']['post_trigger_full_fraction']:.6f}; mean non-FULL actions: {stage2['overall']['mean_non_full_actions']:.4f}.
- Dominant rescue family: **{dominant_rescue}** ({summaries[dominant_rescue]['w_to_c']} W→C). Dominant regression family: **{dominant_regression}** ({summaries[dominant_regression]['c_to_w']} C→W).
- The effect is {'broad across multiple families' if sum(summaries[name]['net_correction'] > 0 for name in FAMILY_ORDER) >= 2 else 'concentrated rather than broad'}.
"""
    _atomic_bytes(output_root / "summaries/full_benchmark_eval_summary.md", full_summary.encode())
    decision_text = f"""# Method-Level Decision

## {decision}. {decision_names[decision]}

The frozen candidate produces pooled W→C/C→W/net = **{summaries['overall']['w_to_c']}/{summaries['overall']['c_to_w']}/{summaries['overall']['net_correction']:+d}** over 19,960 paired samples. This supports only the conclusion encoded by Decision {decision}: {decision_names[decision].lower()} for this exact Stage-1 P90 plus Stage-2 A implementation and evaluation scope.

It does not establish that dynamic visual routing is universally effective or ineffective, that another threshold/checkpoint would behave similarly, or that Stage-1 and Stage-2 bottlenecks are interchangeable. The result matters because it is the first prospectively frozen, full-population external paired test. A small gain is evidence of a real but small effect, not a universal win; one negative benchmark is a task-specific warning, not grounds to reject the method family.
"""
    _atomic_bytes(output_root / "summaries/method_level_decision.md", decision_text.encode())

    direction = recommendation_direction(summaries["overall"], stage2["overall"])
    recommendation = f"""# Next Improvement Recommendation

Recommend exactly one direction: **{direction}**.

This matters because the full evaluation observed W→C/C→W/net = {summaries['overall']['w_to_c']}/{summaries['overall']['c_to_w']}/{summaries['overall']['net_correction']:+d}, with Stage-1 trigger rate {stage1['overall']['trigger_rate']:.4f} and triggered any-non-FULL rate {stage2['overall']['fraction_any_non_full']:.4f}. The motivating failure mode is therefore the measured balance between admission, actual intervention, rescue, and regression—not oracle action recall alone.

A separately authorized discriminating experiment should change only this direction's control while freezing the benchmark population and current candidate as the paired baseline. A positive result would mean that the targeted bottleneck limits end-to-end net correction. A negative result would rule out that specific adjustment under its tested budget; it would not show that READ/WRITE routing or dynamic visual computation is impossible.

This recommendation is not executed in Phase 69.
"""
    _atomic_bytes(output_root / "summaries/next_improvement_recommendation.md", recommendation.encode())

    required = [
        output_root / "protocol.md",
        output_root / "frozen_protocol.json",
        output_root / "paired/all_paired.jsonl",
        output_root / "metrics/benchmark_summary.csv",
        output_root / "metrics/transition_counts.csv",
        output_root / "metrics/paired_bootstrap.csv",
        output_root / "metrics/stage1_admission.csv",
        output_root / "metrics/stage2_action_behavior.csv",
        output_root / "metrics/dataset_breakdown.csv",
        output_root / "metrics/reference_dense_parity.csv",
        output_root / "metrics/compute_summary.csv",
        output_root / "figures/dense_vs_routed_accuracy.png",
        output_root / "figures/net_correction_by_benchmark.png",
        output_root / "figures/rescue_vs_regression.png",
        output_root / "figures/stage1_trigger_behavior.png",
        output_root / "figures/stage2_nonfull_behavior.png",
        output_root / "summaries/full_benchmark_eval_summary.md",
        output_root / "summaries/method_level_decision.md",
        output_root / "summaries/next_improvement_recommendation.md",
    ]
    required.extend(sorted((output_root / "manifests").glob("*.jsonl")))
    required.extend(sorted((output_root / "dense").glob("*.jsonl")))
    required.extend(sorted((output_root / "routed").glob("*.jsonl")))
    required.extend(sorted((output_root / "paired").glob("*.jsonl")))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required outputs are missing: {missing}")
    manifest = {
        "schema_version": "full_benchmark_eval_artifact_manifest_v1",
        "contract_sha256": contract["contract_sha256"],
        "decision": decision,
        "files": [
            {"path": str(path.relative_to(output_root)), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in sorted(set(required))
        ],
        "completed_at": utc_now(),
    }
    atomic_json(output_root / "artifact_manifest.json", manifest)
    print(json.dumps({"passed": True, "rows": len(rows), "decision": decision, "overall": summaries["overall"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "smoke", "worker", "aggregate"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config)
    elif args.command == "smoke":
        smoke(args.config, args.device_index)
    elif args.command == "worker":
        worker(args.config, args.rank, args.resume)
    else:
        aggregate(args.config)


if __name__ == "__main__":
    main()
