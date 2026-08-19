#!/usr/bin/env python3
"""Freeze the compact P0 contract, 8K extraction manifest, and P1 smoke set."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from label_regeneration.data import deterministic_smoke_records, file_sha256, load_source_records


ACTIVE_SOURCES = (
    "binary_policy/executor/cache.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/masks.py",
    "binary_policy/executor/model.py",
    "label_regeneration/data.py",
    "label_regeneration/mcts.py",
    "label_regeneration/runtime.py",
    "experiments/freeze_label_regeneration.py",
    "experiments/run_label_regeneration.py",
    "reference/dvr_qwen/MODEL_AND_LABEL_GENERATION.md",
    "reference/dvr_qwen/eval_metrics.py",
    "plans/dynamic_mllm_label_regeneration_plan.md",
)

MIXED_MASKS = {
    "gqa": [int(index % 2 == 0) for index in range(28)],
    "textvqa": [0 if index < 14 else 1 for index in range(28)],
    "chartqa": [int(index % 3 == 0) for index in range(28)],
}


def write_text_once(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite differing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json_once(path: Path, value) -> None:
    write_text_once(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl_once(path: Path, rows: list[dict]) -> None:
    write_text_once(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def git_revision(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--smoke-seed", type=int, default=20260810)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    output = Path(args.output_root)
    records = load_source_records(args.data_root)
    counts = Counter((row["benchmark"], row["historical_all_on_status"]) for row in records)
    expected = {
        ("gqa", "correct"): 2000,
        ("gqa", "wrong"): 2000,
        ("textvqa", "correct"): 1000,
        ("textvqa", "wrong"): 1000,
        ("chartqa", "correct"): 1000,
        ("chartqa", "wrong"): 1000,
    }
    if counts != Counter(expected):
        raise RuntimeError(f"source counts differ: {dict(counts)}")

    extraction_rows = []
    for index, row in enumerate(records):
        extraction_rows.append({**row, "extraction_index": index})
    smoke = deterministic_smoke_records(records, per_dataset=5, seed=args.smoke_seed)
    smoke_rows = []
    seen_dataset: set[str] = set()
    for index, row in enumerate(smoke):
        mixed = []
        if row["benchmark"] not in seen_dataset:
            mixed = [MIXED_MASKS[row["benchmark"]]]
            seen_dataset.add(row["benchmark"])
        smoke_rows.append({**row, "smoke_index": index, "mixed_masks": mixed})

    source_hashes = {path: file_sha256(project / path) for path in ACTIVE_SOURCES}
    model_files = {}
    model_root = Path(args.model_path)
    for name in (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "chat_template.json",
    ):
        path = model_root / name
        if path.is_file():
            model_files[name] = file_sha256(path)
    contract = {
        "schema_version": "label_regeneration_contract_v1",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_path": str(model_root),
        "model_revision": args.revision,
        "model_metadata_sha256": model_files,
        "processor_use_fast": False,
        "image_processing": "native Qwen processor defaults",
        "custom_max_image_tokens": None,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "generation": {"do_sample": False, "max_new_tokens": 16},
        "executor": "binary_policy.executor.BinaryQwen25VL",
        "route_semantics": {
            "on": "native full text/control plus visual decoder layer",
            "off": "compacted text/control decoder layer; visual rows bypass unchanged",
            "num_layers": 28,
        },
        "mcts": {
            "unrestricted_layer_action_order": True,
            "exploration_constant": 1.8,
            "length_penalty": 3.0,
            "random_probability": 0.1,
            "rollout_off_probability": 0.5,
            "current_correct_simulations": 200,
            "current_wrong_simulations": 400,
            "current_wrong_extension_max": 600,
            "extension_rule": "extend to 600 only when no correcting route exists after 400",
        },
        "correctness_thresholds": {"gqa": 1.0, "textvqa": 0.5, "chartqa": 1.0},
        "source_pool": str(Path(args.data_root).resolve()),
        "source_record_count": 8000,
        "source_counts": {f"{key[0]}/{key[1]}": value for key, value in sorted(counts.items())},
        "source_code_sha256": source_hashes,
        "git_revision": git_revision(project),
        "python_version": platform.python_version(),
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "pillow": package_version("Pillow"),
            "accelerate": package_version("accelerate"),
        },
        "planned_hardware": {
            "node": "node07",
            "gpu_type": "A6000",
            "gpus": 4,
            "cpus": 32,
            "memory": "240G",
        },
    }
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract_hash = sha256(canonical.encode("utf-8")).hexdigest()
    contract["contract_sha256"] = contract_hash

    output.mkdir(parents=True, exist_ok=True)
    write_json_once(output / "frozen_execution_contract.json", contract)
    md = f"""# Frozen Label-Regeneration Execution Contract

- Contract SHA-256: `{contract_hash}`
- Model revision: `{args.revision}`
- Processor: pinned slow processor, native/default image processing
- Custom `max_image_tokens`: `None`
- Executor: current validated `BinaryQwen25VL`, unrestricted 28-bit ON/OFF masks
- Generation: deterministic greedy, 16 maximum new tokens
- Evaluators: preserved MCTS v2 benchmark scoring implementation
- Data: 8,000 GQA/TextVQA/ChartQA records
- Hardware for full extraction: node07, 4×A6000, 32 CPUs, 240G RAM
- Git revision: `{contract['git_revision'] or 'unavailable; source hashes are authoritative'}`

The complete machine-readable contract and source hashes are in
`frozen_execution_contract.json`.
"""
    write_text_once(output / "frozen_execution_contract.md", md)
    write_jsonl_once(output / "source_manifest_v1.jsonl", extraction_rows)
    write_jsonl_once(output / "smoke_manifest_v1.jsonl", smoke_rows)
    for path in (
        output / "frozen_execution_contract.json",
        output / "frozen_execution_contract.md",
        output / "source_manifest_v1.jsonl",
        output / "smoke_manifest_v1.jsonl",
    ):
        write_text_once(path.with_suffix(path.suffix + ".sha256"), f"{file_sha256(path)}  {path.name}\n")
    print(json.dumps({"contract_sha256": contract_hash, "records": 8000, "smoke": 15}, sort_keys=True))


if __name__ == "__main__":
    main()
