from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from nulls.joint_four_action import PairedGeometryMetadata, paired_geometry_distance


LAYERS = [0, 4, 8, 12, 16, 20, 24]
DONOR_COUNT = 8
TIGHT_CALIPER = 1.5
LOCAL_REPAIR_MAX_CALIPER = 1.6
LOCAL_REPAIR_MAX_TAIL_FRACTION = 0.01
DONOR_SEED = 2026080702


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit unchanged v3 paired-donor geometry.")
    parser.add_argument(
        "--geometry-root",
        default="artifacts/v3_null_redesign/read_write_geometry_v2",
    )
    parser.add_argument(
        "--output", default="outputs/v3_null_redesign/donor_coverage.json"
    )
    parser.add_argument(
        "--donor-root", default="artifacts/v3_null_redesign/paired_donor_index_v2"
    )
    parser.add_argument("--expected-samples", type=int, default=2000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derived_seed(base: int, *parts: Any) -> int:
    material = ":".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") % (2**63 - 1)


def metadata_from_summary(row: dict[str, Any]) -> PairedGeometryMetadata:
    return PairedGeometryMetadata(
        sample_id=row["sample_id"],
        image_id=row["image_id"],
        dataset=row["dataset"],
        layer=int(row["layer"]),
        read_norm=float(row["read"]["frobenius_norm"]),
        write_norm=float(row["write"]["frobenius_norm"]),
        read_rows=int(row["read_shape"][0]),
        write_rows=int(row["write_shape"][0]),
        image_tokens=int(row["image_tokens"]),
        prompt_tokens=int(row["prompt_tokens"]),
        read_scale_ratio=float(row["read_rmsnorm_scale_ratio"]),
        write_scale_ratio=float(row["write_rmsnorm_scale_ratio"]),
        read_row_cv=float(row["read"]["row_norm_cv"]),
        write_row_cv=float(row["write"]["row_norm_cv"]),
    )


def ratio(first: float, second: float) -> float:
    if first <= 0 or second <= 0 or not math.isfinite(first) or not math.isfinite(second):
        return math.inf
    return max(first / second, second / first)


def distance_components(
    target: PairedGeometryMetadata, donor: PairedGeometryMetadata
) -> dict[str, float]:
    return {
        "read_norm": ratio(target.read_norm, donor.read_norm),
        "write_norm": ratio(target.write_norm, donor.write_norm),
        "read_rows": ratio(target.read_rows, donor.read_rows),
        "write_rows": ratio(target.write_rows, donor.write_rows),
        "image_tokens": ratio(target.image_tokens, donor.image_tokens),
        "prompt_tokens": ratio(target.prompt_tokens, donor.prompt_tokens),
        "read_scale_ratio": ratio(target.read_scale_ratio, donor.read_scale_ratio),
        "write_scale_ratio": ratio(target.write_scale_ratio, donor.write_scale_ratio),
        "read_row_cv": ratio(target.read_row_cv, donor.read_row_cv),
        "write_row_cv": ratio(target.write_row_cv, donor.write_row_cv),
    }


def tie_hash(seed: int, target_id: str, donor_id: str) -> str:
    return hashlib.sha256(f"{seed}:{target_id}:{donor_id}".encode()).hexdigest()


def q(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "maximum": float(array.max()),
    }


def execute(args: argparse.Namespace) -> None:
    geometry_root = Path(args.geometry_root)
    geometry_manifest = json.loads((geometry_root / "manifest.json").read_text())
    if geometry_manifest["sample_count"] != args.expected_samples:
        raise RuntimeError(
            f"Expected a complete {args.expected_samples:,}-record geometry pool"
        )
    summaries = [
        json.loads(line)
        for line in (geometry_root / "geometry.jsonl").read_text().splitlines()
        if line
    ]
    metadata = [metadata_from_summary(row) for row in summaries]
    groups: dict[tuple[str, int], list[PairedGeometryMetadata]] = defaultdict(list)
    for item in metadata:
        groups[(item.dataset, item.layer)].append(item)
    if set(groups) != {(dataset, layer) for dataset in ("gqa", "textvqa") for layer in LAYERS}:
        raise RuntimeError("Dataset/layer geometry grid is incomplete")

    coverage_rows: list[dict[str, Any]] = []
    donor_rows: list[dict[str, Any]] = []
    stratum_summary: dict[str, Any] = {}
    global_minimum_caliper = 1.0
    for (dataset, layer), rows in sorted(groups.items()):
        kth_values = []
        stratum_rows = []
        for target in sorted(rows, key=lambda item: item.sample_id):
            seed = derived_seed(DONOR_SEED, target.sample_id, layer)
            eligible = []
            for donor in rows:
                if donor.sample_id == target.sample_id or donor.image_id == target.image_id:
                    continue
                distance = paired_geometry_distance(target, donor)
                eligible.append((distance, tie_hash(seed, target.sample_id, donor.sample_id), donor))
            eligible.sort(key=lambda item: (item[0], item[1]))
            if len(eligible) < DONOR_COUNT:
                raise RuntimeError(f"Only {len(eligible)} donors for {target.sample_id}")
            selected = eligible[:DONOR_COUNT]
            kth = float(selected[-1][0])
            kth_values.append(kth)
            limiting = distance_components(target, selected[-1][2])
            max_component = max(limiting.values())
            limiting_names = sorted(
                name for name, value in limiting.items() if abs(value - max_component) <= 1e-12
            )
            item = {
                "sample_id": target.sample_id,
                "image_id": target.image_id,
                "dataset": dataset,
                "layer": layer,
                "distance_to_eighth": kth,
                "donors_within_1_5": sum(item[0] <= TIGHT_CALIPER for item in eligible),
                "nearest_distances": [float(item[0]) for item in eligible[:9]],
                "eighth_donor_limiting_components": limiting_names,
                "eighth_donor_component_ratios": limiting,
            }
            coverage_rows.append(item)
            stratum_rows.append(item)
            donor_rows.append(
                {
                    "sample_id": target.sample_id,
                    "image_id": target.image_id,
                    "dataset": dataset,
                    "layer": layer,
                    "tie_seed": seed,
                    "donors": [
                        {
                            "sample_id": donor.sample_id,
                            "image_id": donor.image_id,
                            "distance": float(distance),
                        }
                        for distance, _, donor in selected
                    ],
                }
            )
        summary = q(kth_values)
        above = sum(value > TIGHT_CALIPER for value in kth_values)
        tail_fraction = above / len(kth_values)
        minimum_caliper = max(kth_values)
        local_repair = (
            minimum_caliper <= LOCAL_REPAIR_MAX_CALIPER
            and tail_fraction <= LOCAL_REPAIR_MAX_TAIL_FRACTION
        )
        summary.update(
            {
                "sample_count": len(kth_values),
                "fraction_above_1_5": tail_fraction,
                "count_above_1_5": above,
                "minimum_complete_caliper": minimum_caliper,
                "classification": (
                    "tight"
                    if minimum_caliper <= TIGHT_CALIPER
                    else "minimal_local_repair"
                    if local_repair
                    else "substantively_weak"
                ),
                "weak_tail_limiting_covariates": dict(
                    Counter(
                        name
                        for row in stratum_rows
                        if row["distance_to_eighth"] > TIGHT_CALIPER
                        for name in row["eighth_donor_limiting_components"]
                    )
                ),
            }
        )
        stratum_summary[f"{dataset}:layer_{layer}"] = summary
        global_minimum_caliper = max(global_minimum_caliper, minimum_caliper)

    donor_root = Path(args.donor_root)
    donor_root.mkdir(parents=True, exist_ok=True)
    donor_index = donor_root / "paired_donor_index.jsonl"
    with donor_index.open("w", encoding="utf-8") as handle:
        for row in donor_rows:
            row["frozen_global_caliper"] = global_minimum_caliper
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    donor_manifest = donor_root / "manifest.json"
    donor_manifest.write_text(
        json.dumps(
            {
                "schema_version": "v3_null_redesign_paired_donor_index_v1",
                "outcome_blind": True,
                "distance_definition_changed": False,
                "donor_count": DONOR_COUNT,
                "calibration_only": True,
                "seed": DONOR_SEED,
                "global_caliper": global_minimum_caliper,
                "index_sha256": sha256(donor_index),
                "geometry_manifest_sha256": sha256(geometry_root / "manifest.json"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (donor_root / "SHA256SUMS").write_text(
        f"{sha256(donor_index)}  {donor_index}\n{sha256(donor_manifest)}  {donor_manifest}\n"
    )

    payload = {
        "schema_version": "v3_null_redesign_donor_coverage_v1",
        "outcome_blind": True,
        "answer_likelihood_correctness_or_action_values_loaded": False,
        "distance_definition_changed": False,
        "distance_covariates": [
            "read_norm",
            "write_norm",
            "read_rows",
            "write_rows",
            "image_tokens",
            "prompt_tokens",
            "read_scale_ratio",
            "write_scale_ratio",
            "read_row_cv",
            "write_row_cv",
        ],
        "donor_count": DONOR_COUNT,
        "tight_caliper": TIGHT_CALIPER,
        "local_repair_rule": {
            "maximum_caliper": LOCAL_REPAIR_MAX_CALIPER,
            "maximum_tail_fraction_per_stratum": LOCAL_REPAIR_MAX_TAIL_FRACTION,
        },
        "global_minimum_caliper": global_minimum_caliper,
        "adequate_without_material_weakening": all(
            row["classification"] in {"tight", "minimal_local_repair"}
            for row in stratum_summary.values()
        ),
        "by_dataset_layer": stratum_summary,
        "weak_targets": [
            row for row in coverage_rows if row["distance_to_eighth"] > TIGHT_CALIPER
        ],
        "coverage_rows": coverage_rows,
        "donor_index_path": str(donor_index),
        "donor_index_sha256": sha256(donor_index),
        "geometry_manifest_sha256": sha256(geometry_root / "manifest.json"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "global_minimum_caliper": global_minimum_caliper,
                "adequate": payload["adequate_without_material_weakening"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    execute(parse_args())
