from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from tools.research_analysis.v3.compare_null_covariance_representations import (
    LAYERS,
    fit_c,
    load_dataset_layer,
    sha256,
)


MAXIMUM_RANK = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bounded native-row rank extension.")
    parser.add_argument(
        "--geometry-root", default="artifacts/v3_null_redesign/read_write_geometry_v2"
    )
    parser.add_argument(
        "--model-root", default="artifacts/v3_null_redesign/joint_covariance_models_c_rank1024_v1"
    )
    parser.add_argument(
        "--output", default="outputs/v3_null_redesign/covariance_representation_c_rank_extension.json"
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def execute(args: argparse.Namespace) -> None:
    geometry_root = Path(args.geometry_root)
    manifest_path = geometry_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["sample_count"] != 2000:
        raise RuntimeError("Rank extension must use the frozen 1,000-per-dataset geometry pool")
    index = json.loads((geometry_root / "tensor_index.json").read_text())
    model_root = Path(args.model_root)
    if model_root.exists() and any(model_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty {model_root}")
    model_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = []
    for dataset in ("gqa", "textvqa"):
        for layer in LAYERS:
            sample_ids, read, write = load_dataset_layer(index, dataset, layer, device)
            rows.append(
                fit_c(
                    dataset,
                    layer,
                    sample_ids,
                    read,
                    write,
                    model_root,
                    maximum_rank=MAXIMUM_RANK,
                )
            )
            del read, write
            torch.cuda.empty_cache()
            print(f"completed {dataset} layer {layer}", flush=True)
    gate = all(row["gate_pass"] for row in rows)
    model_paths = sorted(model_root.rglob("*.pt"))
    payload = {
        "schema_version": "v3_null_redesign_native_row_rank_extension_v1",
        "outcome_blind": True,
        "answer_likelihood_correctness_or_action_values_loaded": False,
        "reason": "rank-512 ceiling prevented the 0.85 geometry target in middle/late strata",
        "maximum_rank": MAXIMUM_RANK,
        "all_other_representation_c_rules_unchanged": True,
        "rows": rows,
        "gate_pass": gate,
        "geometry_manifest_sha256": sha256(manifest_path),
        "model_checksums": {str(path): sha256(path) for path in model_paths},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    model_manifest = model_root / "manifest.json"
    model_manifest.write_text(
        json.dumps(
            {
                "schema_version": "v3_null_redesign_native_row_rank_extension_models_v1",
                "maximum_rank": MAXIMUM_RANK,
                "gate_pass": gate,
                "output_sha256": sha256(output),
                "model_checksums": payload["model_checksums"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (model_root / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path}\n" for path in [*model_paths, model_manifest]
        )
    )
    print(json.dumps({"output": str(output), "gate_pass": gate}, sort_keys=True))


if __name__ == "__main__":
    execute(parse_args())
