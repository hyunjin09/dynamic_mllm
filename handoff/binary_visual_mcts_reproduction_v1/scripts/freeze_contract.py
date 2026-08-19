#!/usr/bin/env python3
"""Freeze the portable MCTS execution contract before smoke or search."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform


MODEL_METADATA = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
)
SOURCE_FILES = (
    "binary_policy/actions.py",
    "binary_policy/executor/__init__.py",
    "binary_policy/executor/cache.py",
    "binary_policy/executor/generation.py",
    "binary_policy/executor/inputs.py",
    "binary_policy/executor/layers.py",
    "binary_policy/executor/masks.py",
    "binary_policy/executor/model.py",
    "label_regeneration/data.py",
    "label_regeneration/mcts.py",
    "label_regeneration/runtime.py",
    "reference/dvr_qwen/eval_metrics.py",
    "scripts/run_label_regeneration.py",
)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    bundle_root = Path(__file__).resolve().parents[1]
    manifest = Path(args.manifest).resolve()
    model_path = Path(args.model_path).resolve()
    missing_sources = [name for name in SOURCE_FILES if not (bundle_root / name).is_file()]
    if missing_sources:
        raise FileNotFoundError(f"bundle source files are missing: {missing_sources}")
    missing_metadata = [name for name in MODEL_METADATA if not (model_path / name).is_file()]
    if missing_metadata:
        raise FileNotFoundError(f"model metadata files are missing: {missing_metadata}")

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    body = {
        "schema_version": "portable_binary_visual_mcts_contract_v1",
        "dataset_version": args.dataset_version,
        "source_manifest": str(manifest),
        "source_manifest_sha256": digest(manifest),
        "source_record_count": len(rows),
        "benchmarks": sorted({str(row["benchmark"]) for row in rows}),
        "metric_contracts": sorted(
            {
                (str(row["metric_name"]), float(row["correctness_threshold"]))
                for row in rows
            }
        ),
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_path": str(model_path),
        "model_revision": args.revision,
        "model_metadata_sha256": {name: digest(model_path / name) for name in MODEL_METADATA},
        "processor_use_fast": False,
        "image_processing": "native Qwen processor defaults",
        "custom_max_image_tokens": None,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "generation": {"do_sample": False, "max_new_tokens": "per manifest record"},
        "route_semantics": {
            "num_layers": 28,
            "on": "native full text/control plus visual decoder layer",
            "off": "compacted text/control decoder layer; visual rows bypass unchanged",
        },
        "mcts": {
            "root": "all_visual_on",
            "anchor": "all_visual_off",
            "current_correct_simulations": 200,
            "current_wrong_simulations": 400,
            "current_wrong_extension_max": 600,
            "extension_rule": "extend to 600 only when no correct route exists after 400",
            "exploration_constant": 1.8,
            "length_penalty": 3.0,
            "random_probability": 0.1,
            "rollout_off_probability": 0.5,
            "unrestricted_layer_action_order": True,
            "seed": args.seed,
        },
        "source_code_sha256": {name: digest(bundle_root / name) for name in SOURCE_FILES},
        "packages": {
            name: package_version(name)
            for name in ("torch", "torchvision", "transformers", "accelerate", "pillow", "mathruler", "pylatexenc")
        },
        "python_version": platform.python_version(),
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    body["contract_sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest(output)}  {output.name}\n", encoding="utf-8"
    )
    print(body["contract_sha256"])


if __name__ == "__main__":
    main()
