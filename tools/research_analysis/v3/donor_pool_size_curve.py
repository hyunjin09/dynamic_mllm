from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


GEOMETRY = Path("artifacts/v3_null_redesign/read_write_geometry_v2/geometry.jsonl")
OUTPUT = Path("outputs/v3_null_redesign/donor_pool_size_curve.json")
SEED = 2026080705
EVALUATION_TARGETS = 200
DONOR_SIZES = [200, 400, 600, 800]
DONOR_COUNT = 8


def rank(kind: str, sample_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{kind}:{sample_id}".encode()).hexdigest()


def feature_vector(row: dict) -> list[float]:
    return [
        float(row["read"]["frobenius_norm"]),
        float(row["write"]["frobenius_norm"]),
        float(row["read_shape"][0]),
        float(row["write_shape"][0]),
        float(row["image_tokens"]),
        float(row["prompt_tokens"]),
        float(row["read_rmsnorm_scale_ratio"]),
        float(row["write_rmsnorm_scale_ratio"]),
        float(row["read"]["row_norm_cv"]),
        float(row["write"]["row_norm_cv"]),
    ]


def execute() -> None:
    grouped = defaultdict(list)
    for line in GEOMETRY.read_text().splitlines():
        row = json.loads(line)
        grouped[(row["dataset"], int(row["layer"]))].append(row)
    strata = {}
    for (dataset, layer), rows in sorted(grouped.items()):
        targets = sorted(rows, key=lambda row: (rank("target", row["sample_id"]), row["sample_id"]))[
            :EVALUATION_TARGETS
        ]
        target_ids = {row["sample_id"] for row in targets}
        donors = sorted(
            [row for row in rows if row["sample_id"] not in target_ids],
            key=lambda row: (rank("donor", row["sample_id"]), row["sample_id"]),
        )
        target_features = np.log(np.asarray([feature_vector(row) for row in targets]))
        donor_features = np.log(np.asarray([feature_vector(row) for row in donors]))
        target_images = [row["image_id"] for row in targets]
        donor_images = [row["image_id"] for row in donors]
        curves = {}
        for size in DONOR_SIZES:
            distances = np.exp(
                np.max(
                    np.abs(target_features[:, None, :] - donor_features[None, :size, :]),
                    axis=2,
                )
            )
            for target_index, target_image in enumerate(target_images):
                for donor_index, donor_image in enumerate(donor_images[:size]):
                    if target_image == donor_image:
                        distances[target_index, donor_index] = np.inf
            eighth = np.partition(distances, DONOR_COUNT - 1, axis=1)[:, DONOR_COUNT - 1]
            curves[str(size)] = {
                "median": float(np.quantile(eighth, 0.50)),
                "q90": float(np.quantile(eighth, 0.90)),
                "q95": float(np.quantile(eighth, 0.95)),
                "q99": float(np.quantile(eighth, 0.99)),
                "maximum": float(eighth.max()),
                "fraction_above_1_5": float(np.mean(eighth > 1.5)),
            }
        strata[f"{dataset}:layer_{layer}"] = {
            "evaluation_target_count": len(targets),
            "available_donor_count": len(donors),
            "curves": curves,
        }
    payload = {
        "schema_version": "v3_null_redesign_donor_pool_size_curve_v1",
        "outcome_blind": True,
        "answer_likelihood_correctness_or_action_values_loaded": False,
        "distance_definition_changed": False,
        "evaluation_design": "fixed 200-target geometry-only cross-validation panel per dataset/layer; nested deterministic donor pools",
        "seed": SEED,
        "donor_count": DONOR_COUNT,
        "donor_sizes": DONOR_SIZES,
        "strata": strata,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    execute()
