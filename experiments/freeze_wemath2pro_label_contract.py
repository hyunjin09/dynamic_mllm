#!/usr/bin/env python3
"""Freeze the We-Math2.0-Pro executor, scorer, and search contract."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform


ACTIVE_SOURCES = (
    "binary_policy/executor/cache.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/masks.py",
    "binary_policy/executor/model.py",
    "label_regeneration/mcts.py",
    "label_regeneration/runtime.py",
    "label_regeneration/wemath.py",
    "experiments/prepare_wemath2pro_label_manifest.py",
    "experiments/run_label_regeneration.py",
    "reference/dvr_qwen/eval_metrics.py",
    "plans/dynamic_mllm_wemath2pro_label_extraction_plan.md",
    "requirements.txt",
)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def write_once(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite differing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-summary", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--supersedes-contract", action="append", default=[])
    parser.add_argument("--scoring-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--max-simulations-per-sample", type=int, default=600)
    parser.add_argument("--planned-gpus", type=int, default=8)
    parser.add_argument("--planned-cpus", type=int, default=96)
    parser.add_argument("--planned-memory", default="240G")
    args = parser.parse_args()
    if args.max_simulations_per_sample not in {400, 600}:
        raise ValueError("max_simulations_per_sample must be 400 or 600")

    project = Path(__file__).resolve().parents[1]
    summary_path = Path(args.manifest_summary).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["total_record_count"] != 4552:
        raise ValueError("We-Math2.0-Pro inventory requires exactly 4,552 records")
    if summary["valid_mcts_record_count"] != 4544 or summary["technical_invalid_record_count"] != 8:
        raise ValueError("approved validity rule requires exactly 4,544 valid and 8 invalid records")

    model_root = Path(args.model_path).resolve()
    model_hashes = {}
    for name in (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "chat_template.json",
    ):
        path = model_root / name
        if path.is_file():
            model_hashes[name] = file_sha256(path)
    contract = {
        "schema_version": "wemath2pro_label_execution_contract_v3",
        "supersedes_compatible_completed_record_contracts": sorted(
            set(args.supersedes_contract)
        ),
        "dataset": "We-Math/We-Math2.0-Pro",
        "dataset_revision": "c1d9f3ccea7361069f0442362e781d1ae7a28e94",
        "source_split": "pro",
        "source_record_count": 4552,
        "valid_mcts_record_count": 4544,
        "technical_invalid_record_count": 8,
        "technical_invalid_rule": "exclude only empty question and/or empty answer",
        "inventory_manifest_sha256": summary["inventory_manifest_sha256"],
        "manifest_sha256": summary["manifest_sha256"],
        "smoke_manifest_sha256": summary["smoke_manifest_sha256"],
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_path": str(model_root),
        "model_revision": args.revision,
        "model_metadata_sha256": model_hashes,
        "processor_use_fast": False,
        "image_processing": "native Qwen processor defaults",
        "custom_max_image_tokens": None,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "generation": {
            "do_sample": False,
            "max_new_tokens": 96,
            "prompt_policy": "direct final answer enclosed in <answer> tags; no reasoning",
        },
        "evaluator": {
            "metric_name": "wemath2pro_mathruler_accuracy",
            "official_contract": "extract <answer> span if present, else stripped response; MathRuler grade_answer",
            "correctness_threshold": 1.0,
            "mathruler_version": package_version("mathruler"),
            "official_reward_source": "We-Math2.0 dynamic_scheduling/examples/reward_function/r1v.py",
            "nontermination_repair": {
                "timeout_seconds": args.scoring_timeout_seconds,
                "normal_completion": "unchanged MathRuler grade_answer result",
                "timeout_result": "score 0.0, incorrect, explicitly marked scoring_timed_out",
                "prediction_cache": "identical decoded predictions reuse the first bounded score",
                "rationale": "prevent unbounded SymPy simplify from stalling a GPU worker",
            },
        },
        "executor": "binary_policy.executor.BinaryQwen25VL",
        "route_semantics": {
            "num_layers": 28,
            "on": "native full text/control plus visual decoder layer",
            "off": "text/control execute while visual hidden states bypass unchanged",
            "search_representation": "unrestricted full mask",
        },
        "mcts": {
            "current_correct_simulations": 200,
            "current_wrong_simulations": 400,
            "max_simulations_per_sample": args.max_simulations_per_sample,
            "current_wrong_extension_max": args.max_simulations_per_sample,
            "extension_rule": (
                "disabled; hard cap at 400"
                if args.max_simulations_per_sample == 400
                else "extend only when no correct route exists after 400"
            ),
            "unrestricted_layer_action_order": True,
            "exploration_constant": 1.8,
            "length_penalty": 3.0,
            "random_probability": 0.1,
            "rollout_off_probability": 0.5,
        },
        "smoke": {
            "records": 5,
            "all_on_native_token_parity_required": "5/5",
            "mixed_masks_per_record": 2,
            "repeated_generated_tokens_and_scores_required": True,
        },
        "planned_hardware": {
            "node": "node06",
            "gpu_type": "A6000",
            "gpus": args.planned_gpus,
            "cpus": args.planned_cpus,
            "memory": args.planned_memory,
            "node06_nccl_workaround": True,
        },
        "python_version": platform.python_version(),
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "pillow": package_version("Pillow"),
            "accelerate": package_version("accelerate"),
            "mathruler": package_version("mathruler"),
            "pylatexenc": package_version("pylatexenc"),
        },
        "source_code_sha256": {
            relative: file_sha256(project / relative) for relative in ACTIVE_SOURCES
        },
    }
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract_hash = sha256(canonical.encode("utf-8")).hexdigest()
    contract["contract_sha256"] = contract_hash

    output = Path(args.output).resolve()
    write_once(output, json.dumps(contract, indent=2, sort_keys=True) + "\n")
    sidecar = output.with_suffix(output.suffix + ".sha256")
    write_once(sidecar, f"{file_sha256(output)}  {output.name}\n")
    print(json.dumps({"contract_sha256": contract_hash, "path": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
