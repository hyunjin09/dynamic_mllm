from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from tools.research_analysis.v3.freeze_v3_null_models import fit_from_payload, load_geometry
from nulls.joint_four_action import _path_scores


LAYERS = [0, 4, 8, 12, 16, 20, 24]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fitted Stage B geometry coordinates.")
    parser.add_argument(
        "--geometry-root", default="artifacts/v3_null_calibration/read_write_geometry_v1"
    )
    parser.add_argument(
        "--covariance-root", default="artifacts/v3_null_calibration/joint_covariance_model_v1"
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    geometry_root = Path(args.geometry_root)
    covariance_root = Path(args.covariance_root)
    tensors, summaries = load_geometry(geometry_root)
    output = geometry_root / "subspace_coordinates_v1.jsonl"
    rows = []
    for dataset in ("gqa", "textvqa"):
        sample_ids = sorted(
            sample_id
            for sample_id, payload in tensors.items()
            if payload["dataset"] == dataset
        )
        for layer in LAYERS:
            model_path = covariance_root / f"{dataset}_layer_{layer:02d}.pt"
            payload = torch.load(model_path, map_location="cpu", weights_only=False)
            fit = fit_from_payload(payload["fit"], device)
            for sample_id in sample_ids:
                paths = tensors[sample_id]["layers"][layer]
                read = paths["read"].float().to(device)
                write = paths["write"].float().to(device)
                read_score = _path_scores([read], fit.read)[0]
                write_score = _path_scores([write], fit.write)[0]
                rows.append(
                    {
                        "sample_id": sample_id,
                        "image_id": tensors[sample_id]["image_id"],
                        "dataset": dataset,
                        "layer": layer,
                        "read_rank": fit.read.rank,
                        "write_rank": fit.write.rank,
                        "read_standardized_coordinates": read_score.detach().cpu().tolist(),
                        "write_standardized_coordinates": write_score.detach().cpu().tolist(),
                    }
                )
            del fit, payload
            torch.cuda.empty_cache()
    if len(rows) != 2800:
        raise RuntimeError(f"Expected 2,800 coordinate rows; got {len(rows)}")
    rows.sort(key=lambda row: (row["dataset"], row["sample_id"], row["layer"]))
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest_path = geometry_root / "coordinate_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "v3_read_write_subspace_coordinates_v1",
                "outcome_blind": True,
                "terminal_answer_or_action_outcomes_used": False,
                "rows": len(rows),
                "layers": LAYERS,
                "coordinates_sha256": sha256(output),
                "covariance_manifest_sha256": sha256(covariance_root / "manifest.json"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "coordinates": str(output),
                "sha256": sha256(output),
                "rows": len(rows),
            }
        )
    )


if __name__ == "__main__":
    main()
